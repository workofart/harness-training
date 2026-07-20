from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import multiprocessing
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from git import GitCommandError, Repo
import pytest
import yaml

import src.measurement as measurement
import src.trainer.trainer as loop
from src.rollout.execution import EagerExecution
import src.worker as worker
from conftest import make_llm_provider_config, TEST_MEASUREMENT_IDENTITY
from src.env.base import benchmark
from src.config import (
    EnvironmentConfig,
    RunConfig,
    PluginsConfig,
    TrainingTargetConfig,
)
from src.trainer.parameter import (
    Candidate,
    CandidateValidationError,
)
from src.trainer.loss import StrictPareto
from src.trainer.optim import GreedyMonotonic
from src.rollout.records import (
    ExperimentResult,
    RolloutResult,
    TaskCertification,
)
from src.env.docker_shell import DockerShellSession
from src.measurement import (
    MeasurementError,
    PreflightError,
    _terminate_experiment_process,
)
from src.trainer.trainer import StdoutLogger, Trainer
from src.rollout.store import RunStore
from src.llm.backend import (
    Completion,
    CompletionBackend,
    ToolCall,
)

_CONFIG_PATH = "config/train_harness.yaml"
_CONFIG_INCLUDE_PATH = "config/llm/test.yaml"


def _write_config_sources(root: Path) -> tuple[str, ...]:
    """_run_config()'s payload on disk as main config + one $include (production layout)."""
    payload = _run_config_payload()
    llm = payload.pop("llm_provider_config")
    (root / _CONFIG_INCLUDE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / _CONFIG_INCLUDE_PATH).write_text(yaml.safe_dump(llm))
    (root / _CONFIG_PATH).write_text(
        yaml.safe_dump(
            {**payload, "llm_provider_config": {"$include": "llm/test.yaml"}}
        )
    )
    return (_CONFIG_PATH, _CONFIG_INCLUDE_PATH)


class FakeEstimator:
    def __init__(
        self,
        *,
        tracker: RunStore,
        base_commits: str | tuple[str, ...],
    ) -> None:
        self.tracker = tracker
        self._base_commits = (
            [base_commits] if isinstance(base_commits, str) else list(base_commits)
        )
        self.proposals: list[str] = []
        self.diagnoses: list[str] = []
        self.proposal_roots: list[Path] = []
        self.diagnosis_roots: list[Path] = []
        self.targets: list[str] = []

    def propose(
        self,
        *,
        repo_root: Path,
        tracker: RunStore,
        target: TrainingTargetConfig,
        emit: object = None,
    ) -> None:
        # The confined agent uses the real store through the harness's staging/publication boundary.
        assert tracker.root == self.tracker.root
        base_commit = self._base_commits.pop(0)
        self.proposals.append(base_commit)
        self.proposal_roots.append(repo_root)
        self.targets.append(target.surface)

    def diagnose(
        self,
        result: ExperimentResult,
        *,
        repo_root: Path,
        tracker: RunStore,
        target: TrainingTargetConfig,
        emit: object = None,
    ) -> None:
        assert tracker.root == self.tracker.root
        self.diagnoses.append(result.experiment_id)
        self.diagnosis_roots.append(repo_root)
        tracker.publish_learning("Current bottleneck\n")


def _trainer(
    *,
    tracker: RunStore,
    run_config: RunConfig,
    estimator: FakeEstimator,
    observer: StdoutLogger | None = None,
) -> Trainer:
    """run_config supplies the path and the criterion's benchmark; the Trainer
    itself loads from config_path (or the _patch_repo seam)."""
    return Trainer(
        config_path=run_config.config_path,
        estimator=estimator,
        criterion=StrictPareto(
            secondary_metrics=benchmark(run_config.environment.kind).secondary_metrics
        ),
        optimizer=GreedyMonotonic(),
        observer=observer,
        default_root_dir=tracker.root,
        worktrees_root=tracker.root.parent / ".worktrees",
    )


def _fit(
    trainer: Trainer, max_epochs: int
) -> list[tuple[ExperimentResult, ExperimentResult]]:
    """The user-authored loop from scripts/train.py, as the tests' driver."""
    pairs: list[tuple[ExperimentResult, ExperimentResult]] = []
    for loss in trainer.epochs(max_epochs):
        pairs.append((loss.candidate, loss.baseline))
        loss.backward()
        trainer.optimizer.step()
    return pairs


def _fit_trainer(
    *, max_epochs: int = 1, **kwargs
) -> list[tuple[ExperimentResult, ExperimentResult]]:
    return _fit(_trainer(**kwargs), max_epochs)


def _decision(tracker: RunStore, experiment_id: str) -> dict:
    payload = json.loads(tracker.experiment_path(experiment_id).read_text())
    return payload["decision"]


def _patch_repo(
    monkeypatch: pytest.MonkeyPatch,
    commit_hashes: str | tuple[str, ...],
    *,
    merges: list[tuple[str, ...]] | None = None,
    lifecycle: list[tuple[str, str, str]] | None = None,
    dirty_files: tuple[str, ...] = (),
    config_payload: dict | None = None,
) -> None:
    hashes = (commit_hashes,) if isinstance(commit_hashes, str) else commit_hashes
    # HEAD moves only when a merge moves it, as in a real repo.
    current = [hashes[0] if hashes else None]

    class _Commit:
        @property
        def hexsha(self) -> str:
            assert current[0] is not None
            return current[0]

    def repo(path: Path) -> SimpleNamespace:
        def merge(*args: str) -> None:
            if merges is not None:
                merges.append(args)
            current[0] = args[-1]

        return SimpleNamespace(
            working_tree_dir=str(path),
            head=SimpleNamespace(commit=_Commit()),
            git=SimpleNamespace(
                ls_files=lambda *args: "",
                merge=merge,
                diff=lambda *args: "\n".join(dirty_files),
                # Config gate reads status; dirty_files feeds only the diff-based warning.
                status=lambda *args: "",
            ),
            is_dirty=lambda: bool(dirty_files),
        )

    @contextlib.contextmanager
    def fake_scratch_worktree(
        repo,
        *,
        commit: str,
        root: Path,
        name: str,
        sparse: tuple[str, ...] = (),
    ):
        del repo
        path = root / f"{name}-{commit[:12]}"
        path.mkdir(parents=True)
        if lifecycle is not None:
            lifecycle.append(("enter", name, commit))
        try:
            yield path
        finally:
            if lifecycle is not None:
                lifecycle.append(("exit", name, commit))
            shutil.rmtree(path)
            if not any(root.iterdir()):
                root.rmdir()

    monkeypatch.setattr(loop, "Repo", repo)
    monkeypatch.setattr(
        loop,
        "load_config_payload",
        # Real-loader contract: absolute resolved sources, main config first.
        lambda path: (
            config_payload if config_payload is not None else _run_config_payload(),
            (Path(path).resolve(), Path(path).resolve().parent / "llm" / "test.yaml"),
        ),
    )
    monkeypatch.setattr(loop, "scratch_worktree", fake_scratch_worktree)
    monkeypatch.setattr(
        loop,
        "capture_candidate",
        lambda repo, *, base_commit: Candidate("candidate", base_commit),
    )
    monkeypatch.setattr(
        loop, "validate_candidate", lambda repo, candidate, **paths: None
    )
    monkeypatch.setattr(
        loop, "run_candidate_suite", lambda repo, candidate, **kwargs: None
    )


class _TrainerCase:
    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        results: tuple[ExperimentResult | BaseException, ...] = (),
        *,
        config: RunConfig | None = None,
        base_commits: str | tuple[str, ...] = "baseline-commit",
        indexed_baseline: ExperimentResult | None = None,
        observer: StdoutLogger | None = None,
    ) -> None:
        self._observer = observer
        self.tracker = RunStore(tmp_path / "experiments")
        if indexed_baseline is not None:
            self.tracker.record_finalized_run(indexed_baseline, kind="baseline")
        self.config = config if config is not None else _run_config()
        self.estimator = FakeEstimator(
            tracker=self.tracker,
            base_commits=base_commits,
        )
        self.results = iter(results)
        self.calls: list[dict] = []
        self.attempts: list[str] = []
        self.merges: list[tuple[str, ...]] = []
        self.lifecycle: list[tuple[str, str, str]] = []
        monkeypatch.setattr(loop, "preflight", lambda config: None)
        _patch_repo(
            monkeypatch,
            base_commits,
            merges=self.merges,
            lifecycle=self.lifecycle,
            config_payload=self.config.model_dump(mode="json"),
        )
        monkeypatch.setattr(loop, "run_isolated_experiment", self.run)
        self._trainer: Trainer | None = None

    @property
    def trainer(self) -> Trainer:
        if self._trainer is None:
            self._trainer = _trainer(
                tracker=self.tracker,
                run_config=self.config,
                estimator=self.estimator,
                observer=self._observer,
            )
        return self._trainer

    def run(self, **kwargs) -> ExperimentResult:
        self.calls.append(kwargs)
        result = next(self.results)
        if isinstance(result, BaseException):
            self.attempts.append(str(result) or type(result).__name__)
            raise result
        self.attempts.append(result.experiment_id)
        return result


def test_trainer_epochs_runs_exactly_max_epochs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        base_commits=("baseline-commit", "baseline-commit", "baseline-commit"),
        indexed_baseline=_experiment("baseline-1", "baseline-commit", {"task-a"}),
    )
    epochs_started: list[int] = []

    def fake_epoch(**kwargs):
        epochs_started.append(1)
        return (None, None)
        yield  # pragma: no cover - makes this a generator for `yield from`

    monkeypatch.setattr(case.trainer, "_epoch", fake_epoch)

    assert _fit(case.trainer, 3) == []

    assert epochs_started == [1, 1, 1]


def test_trainer_rejects_negative_max_epochs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = RunStore(tmp_path / "experiments")
    _patch_repo(monkeypatch, "unused")
    trainer = _trainer(
        tracker=tracker,
        run_config=_run_config(),
        estimator=FakeEstimator(tracker=tracker, base_commits=()),
    )

    with pytest.raises(ValueError, match="max_epochs"):
        trainer.epochs(-1)


def test_trainer_construction_rejects_untracked_config_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Repo.init(tmp_path)
    (tmp_path / "untracked.json").write_text("{}\n")
    tracker = RunStore(tmp_path / "experiments")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(GitCommandError):
        _fit_trainer(
            tracker=tracker,
            run_config=_run_config(config_path="untracked.json"),
            estimator=FakeEstimator(tracker=tracker, base_commits=()),
        )


def test_trainer_runs_baseline_candidate_promotes_and_diagnoses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    rows_before_diagnosis: list[list[str]] = []
    output_before_run: list[str] = []
    finished: list[ExperimentResult] = []

    class RecordingLogger:
        def __init__(self, inner: StdoutLogger) -> None:
            self.inner = inner

        def __getattr__(self, name: str):
            return getattr(self.inner, name)

        def experiment_finished(self, result: ExperimentResult) -> None:
            finished.append(result)
            self.inner.experiment_finished(result)

    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        (
            _experiment("baseline-1", "baseline-commit", {"task-a"}),
            _experiment("candidate-1", "candidate", {"task-a", "task-b"}),
        ),
        observer=RecordingLogger(loop.StdoutLogger()),
    )
    original_diagnose = case.estimator.diagnose

    def diagnose(*args, **kwargs):
        rows_before_diagnosis.append(
            [row.experiment_id for row in case.tracker.read_index()]
        )
        return original_diagnose(*args, **kwargs)

    monkeypatch.setattr(case.estimator, "diagnose", diagnose)

    def run_isolated_experiment(**kwargs) -> ExperimentResult:
        output_before_run.append(capsys.readouterr().out)
        return case.run(**kwargs)

    monkeypatch.setattr(loop, "run_isolated_experiment", run_isolated_experiment)
    pairs = _fit(case.trainer, 1)

    assert len(pairs) == 1
    assert _decision(case.tracker, "candidate-1")["outcome"] == "promoted"
    assert case.merges == [("--ff-only", "candidate")]
    assert [row.kind for row in case.tracker.read_index()] == ["baseline", "baseline"]
    candidate_row = case.tracker.read_index()[1]
    assert candidate_row.verdict == "promoted"
    assert candidate_row.parent_commit_hash == "baseline-commit"
    assert candidate_row.baseline_experiment_id == "baseline-1"
    assert (case.tracker.root / "learning.md").read_text() == "Current bottleneck\n"
    assert case.estimator.proposals == ["baseline-commit"]
    # The trainer hands the estimator the same target validate_candidate enforces.
    assert case.estimator.targets == ["src/policy/core.py"]
    assert case.estimator.diagnoses == ["candidate-1"]
    assert rows_before_diagnosis == [["baseline-1", "candidate-1"]]
    assert output_before_run[0].endswith("epoch 1/1\n")
    assert "baseline  · solved 1/2 · 1m00s" in output_before_run[1]
    assert [result.experiment_id for result in finished] == [
        "baseline-1",
        "candidate-1",
    ]
    assert finished[0].kind == "baseline"
    assert finished[1].baseline_experiment_id == "baseline-1"
    assert finished[1].decision is not None
    assert case.estimator.proposal_roots == [
        tmp_path / ".worktrees" / "propose-baseline-com"
    ]
    assert case.estimator.diagnosis_roots == case.estimator.proposal_roots
    assert case.lifecycle == [
        ("enter", "measure", "baseline-commit"),
        ("exit", "measure", "baseline-commit"),
        ("enter", "propose", "baseline-commit"),
        ("enter", "measure", "candidate"),
        ("exit", "measure", "candidate"),
        ("exit", "propose", "baseline-commit"),
    ]


def test_epoch_criterion_scalar_is_the_backpropagated_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        (
            _experiment("baseline-1", "baseline-commit", {"task-a"}),
            _experiment("candidate-1", "candidate", {"task-a", "task-b"}),
        ),
    )

    loss_values: list[float] = []
    for loss in case.trainer.epochs(1):
        loss_values.append(float(loss))
        loss.backward()
        case.trainer.optimizer.step()

    assert loss_values == [-1.0]


def _stub_exclude(cert_calls: list[str], *, excluded_task_b_verdict: str):
    """Fake exclude_nondeterministic_tasks: record the certified baseline id, stamp
    a two-task certification (task-a deterministic, task-b excluded by the given
    verdict), and drop task-b from the panel."""

    def exclude(**kwargs):
        cert_calls.append(kwargs["baseline"].experiment_id)
        checked = kwargs["baseline"].model_copy(
            update={
                "determinism_certification": {
                    "task-a": TaskCertification(
                        chain_digest="a", verdict="deterministic"
                    ),
                    "task-b": TaskCertification(
                        chain_digest="b", verdict=excluded_task_b_verdict
                    ),
                },
            }
        )
        return checked, ("task-a",)

    return exclude


def _patch_certify(monkeypatch: pytest.MonkeyPatch, exclude) -> None:
    class _StubExecution(EagerExecution):
        async def certify(self, **kwargs):
            return exclude(**kwargs)

    monkeypatch.setattr(loop, "resolve_execution", lambda config: _StubExecution())


def test_fresh_baseline_certification_scopes_candidate_and_criterion_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _experiment(
        "baseline-1", "baseline-commit", {"task-a", "task-b"}
    ).model_copy(
        update={
            "determinism_certification": None,
        }
    )
    candidate = _experiment(
        "candidate-1", "candidate", {"task-a"}, task_ids=("task-a",)
    )
    case = _TrainerCase(tmp_path, monkeypatch, (baseline, candidate))
    cert_calls: list[str] = []

    _patch_certify(
        monkeypatch, _stub_exclude(cert_calls, excluded_task_b_verdict="forked")
    )
    _fit(case.trainer, 1)

    assert cert_calls == ["baseline-1"]
    assert case.calls[0]["task_ids"] == ("task-a", "task-b")
    assert case.calls[1]["task_ids"] == ("task-a",)
    assert _decision(case.tracker, "candidate-1")["regressions"] == []


def test_reused_baseline_without_certification_runs_certification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _experiment("baseline-old", "baseline-commit", {"task-a"}).model_copy(
        update={
            "determinism_certification": None,
        }
    )
    candidate = _experiment(
        "candidate-1", "candidate", {"task-a"}, task_ids=("task-a",)
    )
    case = _TrainerCase(tmp_path, monkeypatch, (candidate,), indexed_baseline=baseline)
    cert_calls: list[str] = []

    _patch_certify(
        monkeypatch, _stub_exclude(cert_calls, excluded_task_b_verdict="no_chain")
    )
    _fit(case.trainer, 1)

    assert cert_calls == ["baseline-old"]


def test_trainer_skips_after_persistent_invalid_infra_without_running_diagnosis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        (
            _experiment("baseline-1", "baseline-commit", {"task-a"}),
            _unmeasurable_candidate("candidate-1"),
            _unmeasurable_candidate("candidate-2"),
        ),
    )
    pairs = _fit(case.trainer, 1)

    assert pairs == []
    assert _decision(case.tracker, "candidate-2")["outcome"] == "invalid_infra"
    assert case.merges == []
    assert case.estimator.diagnoses == []
    assert len(case.tracker.read_index()) == 2
    assert case.tracker.read_index()[1].verdict == "invalid_infra"
    assert case.tracker.read_index()[1].reason == "infra_sensitive_verdict"


def test_diagnosis_failure_clears_grad_after_durable_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        (_experiment("candidate-1", "candidate", {"task-a", "task-b"}),),
        indexed_baseline=_experiment("baseline-1", "baseline-commit", {"task-a"}),
    )
    monkeypatch.setattr(
        case.estimator,
        "diagnose",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("diagnosis failed")),
    )

    with pytest.raises(RuntimeError, match="diagnosis failed"):
        _fit(case.trainer, 1)

    assert case.trainer._harness.grad is None
    assert [row.experiment_id for row in case.tracker.read_index()] == [
        "baseline-1",
        "candidate-1",
    ]
    assert case.tracker.read_index()[1].verdict == "promoted"


def test_candidate_baseline_mismatch_aborts_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        indexed_baseline=_experiment("baseline-1", "baseline-commit", {"task-a"}),
    )
    monkeypatch.setattr(
        loop,
        "capture_candidate",
        lambda repo, *, base_commit: Candidate("candidate", "wrong-baseline"),
    )
    monkeypatch.setattr(
        loop,
        "run_isolated_experiment",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not measure")),
    )

    with pytest.raises(RuntimeError, match="candidate baseline does not match"):
        _fit(case.trainer, 1)


def test_measured_sha_mismatch_aborts_before_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        (_experiment("candidate-1", "other-candidate", {"task-a"}),),
        indexed_baseline=_experiment("baseline-1", "baseline-commit", {"task-a"}),
    )
    comparison_calls: list[str] = []

    class _RecordingCriterion(StrictPareto):
        def __call__(self, **kwargs):
            comparison_calls.append(kwargs["candidate"].git_commit_hash)
            return super().__call__(**kwargs)

    trainer = Trainer(
        config_path=case.config.config_path,
        estimator=case.estimator,
        criterion=_RecordingCriterion(),
        optimizer=GreedyMonotonic(),
        default_root_dir=case.tracker.root,
        worktrees_root=tmp_path / ".worktrees",
    )

    with pytest.raises(RuntimeError, match="measured candidate commit"):
        list(trainer.epochs(1))

    assert comparison_calls == []
    assert [row.experiment_id for row in case.tracker.read_index()] == ["baseline-1"]


def test_break_abandons_pending_epoch_without_concluding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        (_experiment("candidate-1", "candidate", {"task-a", "task-b"}),),
        indexed_baseline=_experiment("baseline-1", "baseline-commit", {"task-a"}),
    )

    for loss in case.trainer.epochs(2):
        loss.backward()
        case.trainer.optimizer.step()
        break

    assert case.attempts == ["candidate-1"]
    assert case.merges == [("--ff-only", "candidate")]
    assert [row.experiment_id for row in case.tracker.read_index()] == ["baseline-1"]
    assert case.estimator.diagnoses == []


def test_reused_trainer_after_abandoned_epoch_does_not_leak_stale_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        (
            _experiment("candidate-1", "candidate", {"task-a", "task-b"}),
            _experiment("baseline-2", "candidate", {"task-a"}),
            _experiment("candidate-2", "candidate-2", {"task-a"}),
        ),
        base_commits=("baseline-commit", "candidate"),
        indexed_baseline=_experiment("baseline-1", "baseline-commit", {"task-a"}),
    )

    for loss in case.trainer.epochs(1):
        loss.backward()
        case.trainer.optimizer.step()
        break

    # A later epoch on the same trainer must not satisfy the backward/step
    # guards with the abandoned epoch's stale grad/applied state.
    monkeypatch.setattr(
        loop,
        "capture_candidate",
        lambda repo, *, base_commit: Candidate("candidate-2", base_commit),
    )
    with pytest.raises(RuntimeError, match="epoch not trained"):
        for _loss in case.trainer.epochs(1):
            pass

    assert case.merges == [("--ff-only", "candidate")]
    assert [row.experiment_id for row in case.tracker.read_index()] == [
        "baseline-1",
        "baseline-2",
    ]
    assert case.estimator.diagnoses == []


def test_backward_on_previous_epochs_loss_fails_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        (
            _experiment("candidate-1", "candidate", {"task-a", "task-b"}),
            _experiment("candidate-2", "candidate-2", {"task-a"}),
        ),
        base_commits=("baseline-commit", "candidate"),
        indexed_baseline=_experiment("baseline-1", "baseline-commit", {"task-a"}),
    )
    candidate_commits = iter(("candidate", "candidate-2"))
    monkeypatch.setattr(
        loop,
        "capture_candidate",
        lambda repo, *, base_commit: Candidate(next(candidate_commits), base_commit),
    )
    held_losses: list = []

    # Epoch 2 records the epoch-1 verdict via a held Loss; the decision must
    # not be attached to epoch 2's candidate.
    with pytest.raises(RuntimeError, match="does not match this epoch's candidate"):
        for loss in case.trainer.epochs(2):
            if not held_losses:
                held_losses.append(loss)
            else:
                loss = held_losses[0]
            loss.backward()
            case.trainer.optimizer.step()

    assert [row.experiment_id for row in case.tracker.read_index()] == [
        "baseline-1",
        "candidate-1",
    ]
    assert case.estimator.diagnoses == ["candidate-1"]


def test_forgotten_step_fails_loudly_instead_of_recording_a_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        (_experiment("candidate-1", "candidate", {"task-a", "task-b"}),),
        indexed_baseline=_experiment("baseline-1", "baseline-commit", {"task-a"}),
    )

    with pytest.raises(RuntimeError, match="epoch not applied"):
        for loss in case.trainer.epochs(1):
            loss.backward()

    # The winning candidate must not be ledgered as an incoherent rejection.
    assert case.merges == []
    assert [row.experiment_id for row in case.tracker.read_index()] == ["baseline-1"]
    assert case.estimator.diagnoses == []


def test_forgotten_backward_fails_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        (_experiment("candidate-1", "candidate", {"task-a", "task-b"}),),
        indexed_baseline=_experiment("baseline-1", "baseline-commit", {"task-a"}),
    )

    with pytest.raises(RuntimeError, match="epoch not trained"):
        for _loss in case.trainer.epochs(1):
            pass

    assert [row.experiment_id for row in case.tracker.read_index()] == ["baseline-1"]
    assert case.estimator.diagnoses == []


def test_trainer_promotes_candidate_with_same_solves_and_fewer_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        (
            _experiment(
                "baseline-1",
                "baseline-commit",
                {"task-a", "task-b"},
                steps_used={"task-a": 10, "task-b": 8},
            ),
            _experiment(
                "candidate-1",
                "candidate",
                {"task-a", "task-b"},
                steps_used={"task-a": 6, "task-b": 7},
            ),
        ),
    )
    _fit(case.trainer, 1)

    decision = _decision(case.tracker, "candidate-1")
    assert decision["outcome"] == "promoted"
    assert decision["reason"] == "secondary_reward_improvement"
    comparison = {c["name"]: c for c in decision["secondary_rewards"]}["steps_used"]
    assert comparison["baseline_value"] == 18
    assert comparison["candidate_value"] == 13
    assert comparison["outcome"] == "candidate_better"
    assert case.merges == [("--ff-only", "candidate")]
    assert case.tracker.read_index()[1].kind == "baseline"
    assert case.tracker.read_index()[1].verdict == "promoted"


def test_trainer_reuses_indexed_baseline_and_leaves_repo_on_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        (_experiment("candidate-1", "candidate", set()),),
        indexed_baseline=_experiment("baseline-1", "baseline-commit", {"task-a"}),
    )
    pairs = _fit(case.trainer, 1)

    assert [baseline.experiment_id for _, baseline in pairs] == ["baseline-1"]
    assert _decision(case.tracker, "candidate-1")["outcome"] == "rejected"
    assert case.merges == [("--ff-only", "baseline-commit")]
    rows = case.tracker.read_index()
    assert [row.kind for row in rows] == ["baseline", "candidate"]
    assert rows[1].baseline_experiment_id == "baseline-1"


@pytest.mark.parametrize(
    ("failure", "expected_attempt"),
    [
        pytest.param("crash", "subprocess died", id="subprocess-crash"),
        pytest.param("unmeasurable", "candidate-1", id="invalid-infra-result"),
    ],
)
def test_trainer_transient_candidate_failure_is_remeasured_and_promoted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected_attempt: str,
) -> None:
    failed_result = {
        "crash": MeasurementError("subprocess died", result=None),
        "unmeasurable": _unmeasurable_candidate("candidate-1"),
    }[failure]
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        (
            _experiment("baseline-1", "baseline-commit", {"task-a"}),
            failed_result,
            _experiment("candidate-2", "candidate", {"task-a", "task-b"}),
        ),
    )
    pairs = _fit(case.trainer, 1)

    assert case.attempts == ["baseline-1", expected_attempt, "candidate-2"]
    assert len(pairs) == 1
    assert _decision(case.tracker, "candidate-2")["outcome"] == "promoted"
    assert case.estimator.diagnoses == ["candidate-2"]
    assert case.merges == [("--ff-only", "candidate")]
    assert [row.experiment_id for row in case.tracker.read_index()] == [
        "baseline-1",
        "candidate-2",
    ]


@pytest.mark.parametrize(
    ("attach_records", "expected_rows"),
    [
        pytest.param(False, (("baseline-1", "baseline", None, None),), id="no-record"),
        pytest.param(
            True,
            (
                ("baseline-1", "baseline", None, None),
                ("candidate-2", "candidate", None, "container exited"),
            ),
            id="concluding-record",
        ),
    ],
)
def test_trainer_persistent_candidate_crash_skips_after_second_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attach_records: bool,
    expected_rows: tuple[tuple[str, str, str | None, str | None], ...],
) -> None:
    first_record, second_record = {
        False: (None, None),
        True: (
            _crashed_experiment("candidate-1"),
            _crashed_experiment("candidate-2"),
        ),
    }[attach_records]
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        (
            _experiment("baseline-1", "baseline-commit", {"task-a"}),
            MeasurementError("first crash", result=first_record),
            MeasurementError("second crash", result=second_record),
        ),
    )
    pairs = _fit(case.trainer, 1)

    assert case.attempts == ["baseline-1", "first crash", "second crash"]
    assert pairs == []
    assert case.estimator.diagnoses == []
    assert case.merges == []
    assert case.lifecycle[-1] == ("exit", "propose", "baseline-commit")
    assert case.lifecycle.count(("exit", "measure", "candidate")) == 2
    assert (
        tuple(
            (row.experiment_id, row.kind, row.verdict, row.crash_reason)
            for row in case.tracker.read_index()
        )
        == expected_rows
    )


def test_trainer_runs_candidate_once_with_all_configured_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _run_config().model_copy(
        update={
            "environment": EnvironmentConfig(
                kind="swe", task_names=["task-a", "task-b", "test-a"]
            )
        }
    )
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        (
            _experiment(
                "baseline-1",
                "baseline-commit",
                {"task-a", "test-a"},
                task_ids=("task-a", "task-b", "test-a"),
            ),
            _experiment(
                "candidate-1",
                "candidate",
                {"task-b", "test-a"},
                task_ids=("task-a", "task-b", "test-a"),
            ),
        ),
        config=config,
    )
    pairs = _fit(case.trainer, 1)

    assert case.attempts == ["baseline-1", "candidate-1"]
    decision = _decision(case.tracker, "candidate-1")
    assert decision["reason"] == "regressed_baseline_tasks"
    assert decision["regressions"] == ["task-a"]
    assert set(pairs[0][0].tasks) == {"task-a", "task-b", "test-a"}
    assert case.merges == [("--ff-only", "baseline-commit")]


def test_trainer_terminal_bench_runs_baseline_candidate_without_prewarm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    config = _run_config(kind="terminal_bench")
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        (
            _experiment("baseline-1", "baseline-commit", {"task-a"}, config=config),
            _experiment(
                "candidate-1", "candidate", {"task-a", "task-b"}, config=config
            ),
        ),
        config=config,
    )
    _fit(case.trainer, 1)

    output = capsys.readouterr().out
    assert case.attempts == ["baseline-1", "candidate-1"]
    assert "baseline  · solved 1/2 · 1m00s" in output
    assert "epoch 1/1 · PROMOTED · 1 → 2/2 (+task-b)" in output
    assert "pre-warming Terminal-Bench caches" not in output
    assert "epoch 1/1" in output
    assert "epoch 1/1 · PROMOTED" in output
    assert [row.experiment_id for row in case.tracker.read_index()] == [
        "baseline-1",
        "candidate-1",
    ]


def test_trainer_terminal_bench_reuses_indexed_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    config = _run_config(kind="terminal_bench")
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        (_experiment("candidate-1", "candidate", {"task-a", "task-b"}, config=config),),
        config=config,
        indexed_baseline=_experiment(
            "baseline-old", "baseline-commit", {"task-a"}, config=config
        ),
    )
    pairs = _fit(case.trainer, 1)

    assert case.attempts == ["candidate-1"]
    assert [baseline.experiment_id for _, baseline in pairs] == ["baseline-old"]
    assert "epoch 1/1 · baseline solved 1/2" in capsys.readouterr().out


def test_trainer_reuses_promoted_candidate_as_next_epoch_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        (
            _experiment("baseline-1", "baseline-commit", {"task-a"}),
            _experiment("candidate-1", "candidate-1", {"task-a", "task-b"}),
            _experiment("candidate-2", "candidate-2", {"task-a"}),
        ),
        base_commits=("baseline-commit", "candidate-1"),
    )
    smoke_checks: list[None] = []
    monkeypatch.setattr(
        loop,
        "preflight",
        lambda config: smoke_checks.append(None),
    )
    captured_commits = iter(("candidate-1", "candidate-2"))
    monkeypatch.setattr(
        loop,
        "capture_candidate",
        lambda repo, *, base_commit: Candidate(next(captured_commits), base_commit),
    )

    pairs = _fit(case.trainer, 2)

    assert [baseline.experiment_id for _, baseline in pairs] == [
        "baseline-1",
        "candidate-1",
    ]
    assert case.attempts == ["baseline-1", "candidate-1", "candidate-2"]
    assert case.merges == [
        ("--ff-only", "candidate-1"),
        ("--ff-only", "candidate-1"),
    ]
    assert len(smoke_checks) == 2
    rows = case.tracker.read_index()
    assert [row.kind for row in rows] == ["baseline", "baseline", "candidate"]
    assert rows[1].verdict == "promoted"
    assert rows[2].baseline_experiment_id == "candidate-1"


def test_trainer_validation_failure_skips_before_candidate_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        indexed_baseline=_experiment("baseline-1", "baseline-commit", {"task-a"}),
    )

    def _propose(**kwargs) -> None:
        del kwargs
        case.estimator.proposals.append("baseline-commit")
        raise CandidateValidationError("bad candidate", cause="invalid_candidate")

    monkeypatch.setattr(case.estimator, "propose", _propose)
    pairs = _fit(case.trainer, 1)

    assert pairs == []
    assert case.estimator.proposals == ["baseline-commit"]
    assert case.merges == []
    assert case.lifecycle == [
        ("enter", "propose", "baseline-commit"),
        ("exit", "propose", "baseline-commit"),
    ]
    assert case.calls == []
    assert [row.experiment_id for row in case.tracker.read_index()] == ["baseline-1"]


def test_trainer_proposer_exception_mid_edit_removes_scratch_and_preserves_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "src" / "policy").mkdir(parents=True)
    (root / "src" / "policy" / "core.py").write_text("VALUE = 1\n")
    (root / "notes.txt").write_text("baseline notes\n")
    config_sources = _write_config_sources(root)
    repo = Repo.init(root)
    with repo.config_writer() as config:
        config.set_value("user", "email", "t@t.t")
        config.set_value("user", "name", "t")
    repo.index.add(["src/policy/core.py", "notes.txt", *config_sources])
    repo.index.commit("baseline")
    baseline_commit = repo.head.commit.hexsha
    (root / "notes.txt").write_text("developer WIP\n")
    main_status = repo.git.status("--porcelain")
    tracker = RunStore(tmp_path / "experiments")
    tracker.record_finalized_run(
        _experiment("baseline-1", baseline_commit, {"task-a"}),
        kind="baseline",
    )

    class _MidEditEstimator:
        def propose(
            self,
            *,
            repo_root: Path,
            tracker: RunStore,
            target: TrainingTargetConfig,
            emit: object = None,
        ) -> Candidate:
            del tracker, target
            (repo_root / "src" / "policy" / "core.py").write_text("VALUE = 2\n")
            raise RuntimeError("agent crashed")

        def diagnose(
            self,
            result: ExperimentResult,
            *,
            repo_root: Path,
            tracker: RunStore,
            target: TrainingTargetConfig,
            emit: object = None,
        ) -> None:
            del target
            del repo_root, tracker
            raise AssertionError(result.experiment_id)

    monkeypatch.chdir(root)
    monkeypatch.setattr(loop, "preflight", lambda config: None)
    monkeypatch.setattr(
        loop,
        "run_isolated_experiment",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not run")),
    )

    with pytest.raises(RuntimeError, match="agent crashed"):
        _fit_trainer(
            tracker=tracker,
            run_config=_run_config(),
            estimator=_MidEditEstimator(),  # type: ignore[arg-type]
            max_epochs=1,
        )

    assert Repo(root).head.commit.hexsha == baseline_commit
    assert Repo(root).git.status("--porcelain") == main_status
    assert (root / "src" / "policy" / "core.py").read_text() == "VALUE = 1\n"
    assert (root / "notes.txt").read_text() == "developer WIP\n"
    assert not any((tmp_path / ".worktrees").iterdir())
    assert ".worktrees" not in Repo(root).git.worktree("list", "--porcelain")


@pytest.mark.parametrize(
    "accepted, reject",
    [
        (True, None),
        (False, None),
        pytest.param(True, "dirty_config", id="rejects-dirty-config"),
        pytest.param(True, "dirty_include", id="rejects-dirty-include"),
    ],
)
def test_real_epoch_fast_forwards_only_accepted_candidate_and_preserves_main_wip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accepted: bool,
    reject: str | None,
) -> None:
    root = tmp_path / "repo"
    (root / "src" / "policy").mkdir(parents=True)
    (root / "src" / "policy" / "core.py").write_text("VALUE = 1\n")
    config_sources = _write_config_sources(root)
    (root / "notes.txt").write_text("baseline notes\n")
    (root / ".gitignore").write_text(".venv\nexperiments\n")
    (root / ".venv").mkdir()
    repo = Repo.init(root)
    with repo.config_writer() as config:
        config.set_value("user", "email", "t@t.t")
        config.set_value("user", "name", "t")
    repo.index.add(["src/policy/core.py", *config_sources, "notes.txt", ".gitignore"])
    repo.index.commit("baseline")
    baseline_commit = repo.head.commit.hexsha
    (root / "notes.txt").write_text("developer WIP\n")

    tracker = RunStore(tmp_path / "experiments")
    tracker.record_finalized_run(
        _experiment("baseline-1", baseline_commit, {"task-a"}),
        kind="baseline",
    )

    class _CapturingEstimator:
        candidate: Candidate | None = None
        agent_head: str | None = None
        diagnosed_root: Path | None = None

        def propose(
            self,
            *,
            repo_root: Path,
            tracker: RunStore,
            target: TrainingTargetConfig,
            emit: object = None,
        ) -> None:
            del target
            assert (repo_root / ".venv").is_symlink()
            assert (repo_root / ".venv").resolve() == root / ".venv"
            assert tracker.root == tmp_path / "experiments"
            propose_repo = Repo(repo_root)
            (repo_root / "src" / "policy" / "core.py").write_text("VALUE = 2\n")
            propose_repo.git.add("--all")
            propose_repo.git.commit("-m", "agent core commit")
            (repo_root / "tests" / "policy").mkdir(parents=True)
            (repo_root / "tests" / "policy" / "test_core_impl.py").write_text(
                "def test_value():\n    assert True\n"
            )
            propose_repo.git.add("--all")
            propose_repo.git.commit("-m", "agent test commit")
            self.agent_head = propose_repo.head.commit.hexsha

        def diagnose(
            self,
            result: ExperimentResult,
            *,
            repo_root: Path,
            tracker: RunStore,
            target: TrainingTargetConfig,
            emit: object = None,
        ) -> None:
            del target
            del result, tracker
            self.diagnosed_root = repo_root
            assert repo_root.exists()

    estimator = _CapturingEstimator()

    def measure(**kwargs) -> ExperimentResult:
        measure_root = kwargs["measure_root"]
        measure_repo = Repo(measure_root)
        candidate_commit = measure_repo.head.commit.hexsha
        estimator.candidate = Candidate(candidate_commit, baseline_commit)
        assert not (measure_root / ".venv").exists()
        measure_repo.create_head(
            "refs/experiments/runs/candidate-1",
            candidate_commit,
            force=True,
        )
        assert (measure_root / "src" / "policy" / "core.py").read_text() == (
            "VALUE = 2\n"
        )
        solved = {"task-a", "task-b"} if accepted else set()
        return _experiment("candidate-1", estimator.candidate.commit, solved)

    monkeypatch.chdir(root)
    monkeypatch.setattr(loop, "preflight", lambda config: None)
    monkeypatch.setattr(loop, "run_isolated_experiment", measure)
    if reject is not None:
        # Config skew vs HEAD must reject before anything runs; notes.txt stays dirty — non-config WIP must not trip the gate.
        dirty_path = _CONFIG_PATH if reject == "dirty_config" else _CONFIG_INCLUDE_PATH
        (root / dirty_path).write_text(
            (root / dirty_path).read_text() + "# tuning wip\n"
        )
        head_before = repo.head.commit.hexsha
        status_before = repo.git.status("--porcelain")
        with pytest.raises(ValueError, match="uncommitted changes"):
            Trainer(
                config_path=_CONFIG_PATH,
                estimator=estimator,  # type: ignore[arg-type]
                criterion=StrictPareto(),
                optimizer=GreedyMonotonic(),
                default_root_dir=tracker.root,
            )
        assert estimator.agent_head is None
        assert estimator.candidate is None
        assert repo.head.commit.hexsha == head_before
        assert repo.git.status("--porcelain") == status_before
        return
    trainer = Trainer(
        config_path=_CONFIG_PATH,
        estimator=estimator,  # type: ignore[arg-type]
        criterion=StrictPareto(),
        optimizer=GreedyMonotonic(),
        default_root_dir=tracker.root,
    )

    pairs = _fit(trainer, 1)

    assert len(pairs) == 1
    assert estimator.candidate is not None
    assert estimator.agent_head is not None
    assert estimator.candidate.commit != estimator.agent_head
    assert repo.commit(estimator.candidate.commit).parents[0].hexsha == baseline_commit
    assert set(
        repo.git.diff(
            "--name-only", baseline_commit, estimator.candidate.commit
        ).splitlines()
    ) == {"src/policy/core.py", "tests/policy/test_core_impl.py"}
    assert _decision(tracker, "candidate-1")["outcome"] == (
        "promoted" if accepted else "rejected"
    )
    assert repo.head.commit.hexsha == (
        estimator.candidate.commit if accepted else baseline_commit
    )
    assert repo.commit("refs/experiments/runs/candidate-1").hexsha == (
        estimator.candidate.commit
    )
    assert (root / "src" / "policy" / "core.py").read_text() == (
        "VALUE = 2\n" if accepted else "VALUE = 1\n"
    )
    assert (root / "notes.txt").read_text() == "developer WIP\n"
    assert estimator.diagnosed_root is not None
    assert not estimator.diagnosed_root.exists()
    assert not any((root.parent / f"{root.name}-worktrees").iterdir())


def test_real_epoch_rejects_candidate_whose_tree_fails_trusted_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only the trusted suite run keeps a broken-frozen-test candidate out of measurement.
    root = tmp_path / "repo"
    (root / "src" / "policy").mkdir(parents=True)
    # Regular pkg like the real repo; namespace src loses imports to the editable install.
    (root / "src" / "__init__.py").write_text("")
    (root / "src" / "policy" / "core.py").write_text("VALUE = 1\n")
    (root / "tests" / "policy").mkdir(parents=True)
    (root / "tests" / "policy" / "test_core_contracts.py").write_text(
        "from src.policy.core import VALUE\n\n\ndef test_contract():\n"
        "    assert VALUE == 1\n"
    )
    (root / "tests" / "policy" / "test_core_impl.py").write_text(
        "def test_impl():\n    assert True\n"
    )
    config_sources = _write_config_sources(root)
    (root / ".gitignore").write_text(".venv\nexperiments\n")
    (root / ".venv").mkdir()
    repo = Repo.init(root)
    with repo.config_writer() as config:
        config.set_value("user", "email", "t@t.t")
        config.set_value("user", "name", "t")
    repo.index.add(
        [
            "src/__init__.py",
            "src/policy/core.py",
            "tests/policy/test_core_contracts.py",
            "tests/policy/test_core_impl.py",
            *config_sources,
            ".gitignore",
        ]
    )
    repo.index.commit("baseline")
    baseline_commit = repo.head.commit.hexsha
    tracker = RunStore(tmp_path / "experiments")
    tracker.record_finalized_run(
        _experiment("baseline-1", baseline_commit, {"task-a"}),
        kind="baseline",
    )

    class _BreakingEstimator:
        def propose(
            self,
            *,
            repo_root: Path,
            tracker: RunStore,
            target: TrainingTargetConfig,
            emit: object = None,
        ) -> None:
            del tracker, target
            # Valid surface, green impl test; frozen contract test pins VALUE == 1.
            (repo_root / "src" / "policy" / "core.py").write_text("VALUE = 2\n")
            (repo_root / "tests" / "policy" / "test_core_impl.py").write_text(
                "def test_impl():\n    assert 2 == 2\n"
            )

        def diagnose(
            self,
            result: ExperimentResult,
            *,
            repo_root: Path,
            tracker: RunStore,
            target: TrainingTargetConfig,
            emit: object = None,
        ) -> None:
            del target
            del repo_root, tracker
            raise AssertionError(result.experiment_id)

    monkeypatch.chdir(root)
    monkeypatch.setattr(loop, "preflight", lambda config: None)
    monkeypatch.setattr(
        loop,
        "run_isolated_experiment",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("measurement must not run for a red-suite candidate")
        ),
    )

    trainer = Trainer(
        config_path=_CONFIG_PATH,
        estimator=_BreakingEstimator(),  # type: ignore[arg-type]
        criterion=StrictPareto(),
        optimizer=GreedyMonotonic(),
        default_root_dir=tracker.root,
    )

    pairs = _fit(trainer, 1)

    assert pairs == []
    assert repo.head.commit.hexsha == baseline_commit
    assert (root / "src" / "policy" / "core.py").read_text() == "VALUE = 1\n"
    assert not any((root.parent / f"{root.name}-worktrees").iterdir())


def _proposing(*outcomes: str):
    """estimator.propose stub replaying a scripted skip/measured sequence: 'skip'
    raises the no_candidate CandidateValidationError (the breaker's trip signal);
    'measured' returns None so the epoch proceeds to measurement."""
    scripted = iter(outcomes)

    def propose(**kwargs) -> None:
        del kwargs
        if next(scripted) == "skip":
            raise CandidateValidationError(
                "agent produced no candidate",
                cause="no_candidate",
            )
        return None

    return propose


def test_trainer_breaker_trips_after_two_consecutive_skips_after_callbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        base_commits=("baseline-commit", "baseline-commit"),
        indexed_baseline=_experiment("baseline-1", "baseline-commit", {"task-a"}),
    )

    monkeypatch.setattr(case.estimator, "propose", _proposing("skip", "skip", "skip"))

    with pytest.raises(loop.CircuitBreakerTripped, match="no_candidate, no_candidate"):
        _fit(case.trainer, 3)


def test_trainer_breaker_resets_after_measured_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        (_experiment("candidate-1", "candidate", {"task-a"}),),
        base_commits=("baseline-commit", "baseline-commit", "baseline-commit"),
        indexed_baseline=_experiment("baseline-1", "baseline-commit", {"task-a"}),
    )

    monkeypatch.setattr(
        case.estimator, "propose", _proposing("skip", "measured", "skip")
    )
    pairs = _fit(case.trainer, 3)

    assert len(pairs) == 1


def test_trainer_keyboard_interrupt_from_candidate_subprocess_propagates_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _TrainerCase(
        tmp_path,
        monkeypatch,
        (
            _experiment("baseline-1", "baseline-commit", {"task-a"}),
            KeyboardInterrupt(),
        ),
    )

    with pytest.raises(KeyboardInterrupt):
        _fit(case.trainer, 1)

    assert case.attempts == ["baseline-1", "KeyboardInterrupt"]
    assert case.merges == []
    assert case.lifecycle[-2:] == [
        ("exit", "measure", "candidate"),
        ("exit", "propose", "baseline-commit"),
    ]


def test_trainer_aborts_before_any_work_when_llm_provider_unreachable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _TrainerCase(tmp_path, monkeypatch)

    def _unreachable(config: RunConfig) -> None:
        del config
        raise PreflightError("endpoint down")

    monkeypatch.setattr(loop, "preflight", _unreachable)

    with pytest.raises(PreflightError, match="endpoint down"):
        _fit(case.trainer, 1)

    assert case.calls == []


class _FakeCompletionBackend(CompletionBackend):
    def __init__(
        self,
        *,
        completion: Completion | None = None,
        error: Exception | None = None,
    ) -> None:
        self._completion = completion
        self._error = error
        self.closed = False
        self.messages = None
        self.tools = None

    async def _complete(self, request):
        self.messages = request.messages
        self.tools = request.tools
        if self._error is not None:
            raise self._error
        assert self._completion is not None
        return self._completion

    async def close(self) -> None:
        self.closed = True


def _patch_completion_factory(
    monkeypatch: pytest.MonkeyPatch, backend: _FakeCompletionBackend
) -> None:
    def _make_backend(config):
        del config
        return backend

    monkeypatch.setattr(measurement, "make_backend", _make_backend)


def test_smoke_check_accepts_submit_without_parsing_args_and_closes_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeCompletionBackend(
        completion=Completion(
            tool_calls=(ToolCall(name="submit", arguments="{"),),
            finish_reason="tool_calls",
        )
    )
    _patch_completion_factory(monkeypatch, backend)

    measurement._assert_llm_provider_reachable(_run_config())

    assert backend.closed is True
    assert backend.tools is not None
    assert {spec["function"]["name"] for spec in backend.tools} == {"submit"}
    assert "submit" in backend.messages[0]["content"]


def test_smoke_check_does_not_import_candidate_editable_policy() -> None:
    script = """
import importlib.abc
import sys

class RejectCandidatePolicy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "src.policy.core":
            raise RuntimeError("candidate-editable policy imported")
        return None

sys.meta_path.insert(0, RejectCandidatePolicy())

import src.measurement as measurement
from src.config import EnvironmentConfig, RunConfig, LlmProviderConfig
from src.llm.backend import Completion, CompletionBackend, ToolCall

class FakeBackend(CompletionBackend):
    def __init__(self):
        self.closed = False

    async def _complete(self, request):
        return Completion(tool_calls=(ToolCall(name="submit", arguments="{}"),))

    async def close(self):
        self.closed = True

backend = FakeBackend()
measurement.make_backend = lambda config: backend
config = RunConfig(
    schema_version=13,
    training_target={"module": "src.policy.core"},
    environment=EnvironmentConfig(kind="swe", task_names=["task-a"]),
    llm_provider_config=LlmProviderConfig(
        model_name="test",
        base_url="http://127.0.0.1:18000/v1",
        api_key_env="TEST_API_KEY",
        max_context_length=1024,
        max_tokens=256,
    ),
)
measurement._assert_llm_provider_reachable(config)
assert backend.closed
assert "src.policy.core" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("completion", "error", "match"),
    [
        pytest.param(
            None,
            ConnectionError("refused"),
            "CompletionInfraError: refused",
            id="connection-error",
        ),
        pytest.param(
            Completion(content="ok", finish_reason="stop"),
            None,
            "did not return a valid submit tool call",
            id="text-only",
        ),
        pytest.param(
            Completion(tool_calls=(ToolCall(name="run", arguments="{}"),)),
            None,
            "did not return a valid submit tool call",
            id="wrong-tool-name",
        ),
    ],
)
def test_smoke_check_rejects_invalid_provider_response(
    monkeypatch: pytest.MonkeyPatch,
    completion: Completion | None,
    error: Exception | None,
    match: str,
) -> None:
    backend = _FakeCompletionBackend(completion=completion, error=error)
    _patch_completion_factory(monkeypatch, backend)

    with pytest.raises(PreflightError, match=match):
        measurement._assert_llm_provider_reachable(_run_config())
    assert backend.closed is True


@pytest.mark.parametrize(
    "patch_path", [_CONFIG_PATH, _CONFIG_INCLUDE_PATH], ids=["main-config", "include"]
)
def test_trainer_rejects_config_inside_patch_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch_path: str,
) -> None:
    tracker = RunStore(tmp_path)
    config = _run_config(
        training_target={
            "module": "src.policy.core",
            "extra_patch_paths": [patch_path],
            "proposer_visible": ["/src/"],
        }
    )
    _patch_repo(monkeypatch, "unused", config_payload=config.model_dump(mode="json"))

    with pytest.raises(ValueError, match="must not include the measurement config"):
        _trainer(
            tracker=tracker,
            run_config=config,
            estimator=FakeEstimator(tracker=tracker, base_commits=()),
        )


def test_trainer_rejects_empty_proposer_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = RunStore(tmp_path)
    config = _run_config(training_target={"module": "src.policy.core"})
    _patch_repo(monkeypatch, "unused", config_payload=config.model_dump(mode="json"))

    with pytest.raises(ValueError, match="proposer_visible must not be empty"):
        _trainer(
            tracker=tracker,
            run_config=config,
            estimator=FakeEstimator(tracker=tracker, base_commits=()),
        )


def test_trainer_warns_dirty_tracked_files_at_fit_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tracker = RunStore(tmp_path)
    _patch_repo(
        monkeypatch,
        "unused",
        dirty_files=("src/example.py", "README.md"),
    )
    trainer = _trainer(
        tracker=tracker,
        run_config=_run_config(),
        estimator=FakeEstimator(tracker=tracker, base_commits=()),
    )

    assert _fit(trainer, 0) == []

    output = capsys.readouterr().out
    assert "warning: dirty tracked files: src/example.py, README.md" in output
    assert (
        "uncommitted changes will NOT be measured (worktrees run committed code only)"
    ) in output


def _training_target() -> dict[str, object]:
    return {
        "module": "src.policy.core",
        "extra_patch_paths": ["tests/policy/test_core_impl.py"],
        "proposer_visible": [
            "/src/",
            "!/src/trainer/",
            "/tests/policy/",
            "/program.md",
        ],
    }


def _run_config_payload(
    *,
    kind: str = "swe",
    training_target: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 13,
        "training_target": (
            _training_target() if training_target is None else training_target
        ),
        "environment": {"kind": kind, "task_names": ["task-a", "task-b"]},
        "llm_provider_config": make_llm_provider_config().model_dump(exclude_none=True),
    }


def _run_config(
    *,
    config_path: str = _CONFIG_PATH,
    kind: str = "swe",
    training_target: dict[str, object] | None = None,
) -> RunConfig:
    """Built from a payload exactly as the Trainer builds it."""
    return RunConfig.model_validate(
        _run_config_payload(kind=kind, training_target=training_target)
    ).model_copy(update={"config_path": config_path})


def _experiment(
    experiment_id: str,
    commit_hash: str,
    solved_tasks: set[str],
    *,
    task_ids: tuple[str, ...] = ("task-a", "task-b"),
    steps_used: dict[str, int | None] | None = None,
    config: RunConfig | None = None,
) -> ExperimentResult:
    steps_used = {} if steps_used is None else steps_used
    tasks = {
        task_id: RolloutResult(
            task_id=task_id,
            failure_mode=("solved" if task_id in solved_tasks else "verified_rejected"),
            error=None,
            metrics=(
                {}
                if steps_used.get(task_id) is None
                else {"steps_used": steps_used[task_id]}
            ),
            rollout_dir=None,
            trace_path=None,
            started_at=None,
            finished_at=None,
        )
        for task_id in task_ids
    }
    measurement_identity = asyncio.run(
        loop.resolve_measurement_identity(
            _run_config() if config is None else config, EagerExecution()
        )
    )
    certification = {
        task_id: TaskCertification(chain_digest=task_id, verdict="deterministic")
        for task_id in task_ids
    }
    return ExperimentResult(
        experiment_id=experiment_id,
        git_commit_hash=commit_hash,
        measurement_identity=measurement_identity,
        git_dirty=False,
        config_path=_CONFIG_PATH,
        started_at="2026-06-21T00:00:00+00:00",
        finished_at="2026-06-21T00:01:00+00:00",
        tasks=tasks,
        determinism_certification=certification,
    )


def _unmeasurable_candidate(experiment_id: str) -> ExperimentResult:
    candidate = _experiment(experiment_id, "candidate", {"task-b"})
    candidate.tasks["task-a"] = RolloutResult(
        task_id="task-a",
        failure_mode="unscorable_infra",
        error=None,
        metrics={},
        rollout_dir=None,
        trace_path=None,
        started_at=None,
        finished_at=None,
    )
    return candidate


def _crashed_experiment(experiment_id: str) -> ExperimentResult:
    return _experiment(experiment_id, "candidate", set()).model_copy(
        update={"crash_reason": "container exited"}
    )


class _Urllib3Resp:
    """Stand-in whose module makes it look like a urllib3 response object."""


_Urllib3Resp.__module__ = "urllib3.response"


def _unraisable(exc: BaseException, obj: object) -> SimpleNamespace:
    return SimpleNamespace(exc_value=exc, object=obj)


def _measurement_experiment(
    experiment_id: str,
    *,
    finished: bool,
    crash_reason: str | None = None,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        git_commit_hash="commit",
        measurement_identity=TEST_MEASUREMENT_IDENTITY,
        git_dirty=False,
        config_path=_CONFIG_PATH,
        started_at=datetime.fromisoformat("2026-06-21T00:00:00+00:00"),
        finished_at=(
            datetime.fromisoformat("2026-06-21T00:01:00+00:00") if finished else None
        ),
        crash_reason=crash_reason,
        tasks={
            "task-a": (
                RolloutResult(
                    task_id="task-a",
                    failure_mode="verified_rejected",
                    error=None,
                    metrics={},
                    rollout_dir=None,
                    trace_path=None,
                    started_at=None,
                    finished_at=None,
                )
                if finished and crash_reason is None
                else None
            )
        },
    )


def _config() -> RunConfig:
    return RunConfig(
        schema_version=13,
        training_target={"module": "src.policy.core"},
        environment=EnvironmentConfig(kind="swe", task_names=["a"]),
        llm_provider_config=make_llm_provider_config(),
        config_path="config/run_config.json",
    )


class _SendConnection:
    def __init__(self) -> None:
        self.events: list[tuple[str, ...] | dict[str, str]] = []
        self.closed = False

    def send_bytes(self, data: bytes) -> None:
        event = json.loads(data)
        self.events.append(tuple(event) if isinstance(event, list) else event)

    def close(self) -> None:
        self.closed = True

    def fileno(self) -> int:
        return 17


class _FakeProcess:
    def __init__(
        self,
        *,
        alive: bool,
        exitcode: int,
        exit_on_join: bool = True,
        waits_before_exit: int = 0,
    ) -> None:
        self.pid = 4242
        self.exitcode = exitcode
        self.returncode = None if alive else exitcode
        self.alive = alive
        self.exit_on_join = exit_on_join
        self.waits_before_exit = waits_before_exit
        self.wait_timeouts: list[float | None] = []
        self.signals: list[int] = []
        self.killed = False

    def poll(self) -> int | None:
        return None if self.alive else self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if self.waits_before_exit:
            self.waits_before_exit -= 1
            if timeout is not None:
                raise subprocess.TimeoutExpired("experiment-worker", timeout)
        elif self.exit_on_join:
            self.alive = False
            self.returncode = self.exitcode
        elif timeout is not None:
            raise subprocess.TimeoutExpired("experiment-worker", timeout)
        assert self.returncode is not None
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.alive = False
        self.exitcode = -signal.SIGKILL
        self.returncode = self.exitcode

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)


class _ReceiveConnection:
    def __init__(
        self,
        events: list[tuple[str, ...] | dict[str, str] | None],
        process: _FakeProcess,
    ) -> None:
        self.events = list(events)
        self.process = process
        self.closed = False

    def poll(self, timeout: float | None = None) -> bool:
        del timeout
        if self.events and self.events[0] is None:
            self.events.pop(0)
            return False
        return bool(self.events)

    def recv_bytes(self) -> bytes:
        event = self.events.pop(0)
        assert event is not None
        if not self.events:
            self.process.alive = False
            self.process.returncode = self.process.exitcode
        return json.dumps(list(event) if isinstance(event, tuple) else event).encode()

    def close(self) -> None:
        self.closed = True


class _FakeContext:
    def __init__(
        self,
        *,
        events: list[tuple[str, ...] | dict[str, str] | None],
        process: _FakeProcess,
    ) -> None:
        self.send = _SendConnection()
        self.receive = _ReceiveConnection(events, process)
        self.process = process
        self.popen_args = None
        self.popen_kwargs = None

    def Pipe(self, *, duplex: bool):
        assert duplex is False
        return self.receive, self.send

    def Popen(self, args, **kwargs):
        self.popen_args = args
        self.popen_kwargs = kwargs
        return self.process


class _MeasurementCase:
    def __init__(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        events: list[tuple[str, ...] | dict[str, str] | None],
        process: _FakeProcess,
        result: ExperimentResult | None = None,
        task_ids: tuple[str, ...] = ("a",),
    ) -> None:
        self.root = tmp_path / "experiments"
        self.tracker = RunStore(self.root)
        if result is not None:
            self.tracker.save_experiment(result)
        self.context = _FakeContext(events=events, process=process)
        self.process = process
        self.task_ids = task_ids
        self.measure_root = tmp_path / "measure"
        self.started: list[str] = []
        self.logs: list[str] = []
        self.tasks: list[tuple[str, str]] = []
        self.heartbeats: list[None] = []
        self.observer = SimpleNamespace(
            log=self.logs.append,
            experiment_started=self.started.append,
            task_finished=lambda task_id, failure_mode: self.tasks.append(
                (task_id, failure_mode)
            ),
            measurement_heartbeat=lambda: self.heartbeats.append(None),
        )
        monkeypatch.setattr(measurement.multiprocessing, "Pipe", self.context.Pipe)
        monkeypatch.setattr(measurement.subprocess, "Popen", self.context.Popen)
        monkeypatch.setattr(
            measurement,
            "_terminate_experiment_process",
            lambda child: child.returncode,
        )
        from src.plugins.caching import store as cache

        monkeypatch.setattr(cache, "DB_PATH", tmp_path / "cache" / "llm_cache.db")

    def run(self) -> ExperimentResult:
        return measurement.run_isolated_experiment(
            config_path=_config().config_path,
            tracker=self.tracker,
            observer=self.observer,
            measure_root=self.measure_root,
            task_ids=self.task_ids,
        )


def test_drops_urllib3_closed_file_finalizer_noise() -> None:
    seen: list[object] = []
    hook = worker._make_quiet_unraisablehook(seen.append)

    hook(_unraisable(ValueError("I/O operation on closed file"), _Urllib3Resp()))

    assert seen == []


def test_delegates_every_other_unraisable_to_previous_hook() -> None:
    seen: list[object] = []
    hook = worker._make_quiet_unraisablehook(seen.append)
    same_msg_not_urllib3 = _unraisable(
        ValueError("I/O operation on closed file"), SimpleNamespace()
    )
    urllib3_other_error = _unraisable(RuntimeError("boom"), _Urllib3Resp())
    urllib3_other_value_error = _unraisable(ValueError("bad value"), _Urllib3Resp())

    for unraisable in (
        same_msg_not_urllib3,
        urllib3_other_error,
        urllib3_other_value_error,
    ):
        hook(unraisable)

    assert seen == [
        same_msg_not_urllib3,
        urllib3_other_error,
        urllib3_other_value_error,
    ]


def test_worker_rejects_src_imported_from_outside_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(AssertionError):
        worker._assert_src_from_cwd()


def test_worker_requires_parent_resolved_cache_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FRAMEWORK_CACHE_DB", raising=False)

    with pytest.raises(AssertionError):
        worker._assert_cache_ready(_config())


def test_worker_all_plugins_off_does_not_open_substrate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_db = tmp_path / "cache" / "llm_cache.db"
    monkeypatch.setenv("FRAMEWORK_CACHE_DB", str(cache_db))
    config = _config().model_copy(
        update={"plugins": PluginsConfig(llm_cache=False, execution="eager")}
    )

    worker._assert_cache_ready(config)

    assert not cache_db.exists()


def test_worker_cache_probe_opens_store_or_dies_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.plugins.caching import store as cache

    monkeypatch.setenv("FRAMEWORK_CACHE_DB", str(tmp_path / "llm_cache.db"))
    monkeypatch.setattr(cache, "_DISABLED", False)
    worker._assert_cache_ready(_config())
    assert cache._STORE is not None

    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    monkeypatch.setattr(cache, "_STORE", None)
    monkeypatch.setattr(cache, "DB_PATH", blocker / "sub" / "llm_cache.db")
    with pytest.raises(OSError):
        worker._assert_cache_ready(_config())


def test_worker_loads_config_runs_experiment_and_sends_structured_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import src.rollout.sampler as sampler

    monkeypatch.setenv("FRAMEWORK_CACHE_DB", str(tmp_path / "cache" / "llm_cache.db"))
    root = tmp_path / "child-evals"
    config = _config()
    result = _measurement_experiment("exp-1", finished=True)
    events = _SendConnection()
    observed: dict[str, object] = {}
    root_logger = logging.getLogger()
    previous_handlers = tuple(root_logger.handlers)
    previous_level = root_logger.level

    async def fake_run_experiment(*, run_config, tracker, observer, experiment_id):
        observed.update(
            requested_experiment_id=experiment_id,
            effective_task_names=run_config.environment.task_names,
            effective_config_task_names=run_config.model_dump(mode="json")[
                "environment"
            ]["task_names"],
            root=tracker.root,
        )
        tracker.run_dir(result.experiment_id).mkdir(parents=True)
        observer.experiment_started(result.experiment_id)
        logging.getLogger("src.plugins.replay.step_cache").warning(
            "step-cache skip:\nRunAction(command='pytest')"
        )
        observer.task_finished("task-a", "solved")
        return result

    monkeypatch.setattr(
        worker.RunConfig,
        "load",
        lambda path: observed.update(config_path=path) or config,
    )
    monkeypatch.setattr(sampler, "run_experiment", fake_run_experiment)

    worker._experiment_worker(
        "config/example.json",
        str(root),
        events,
        task_ids=("task-a",),
        experiment_id="exp-named-by-parent",
    )

    assert observed == {
        "config_path": "config/example.json",
        "requested_experiment_id": "exp-named-by-parent",
        "effective_task_names": ["task-a"],
        "effective_config_task_names": ["task-a"],
        "root": root,
    }
    assert config.environment.task_names == ["a"]
    assert events.events == [
        ("started", "exp-1"),
        ("task", "task-a", "solved"),
    ]
    run_log = (root / "exp-1" / "run.log").read_text()
    assert "src.plugins.replay.step_cache" in run_log
    assert "RunAction(command='pytest')" in run_log
    for handler in root_logger.handlers:
        if handler not in previous_handlers:
            handler.close()
    root_logger.handlers = list(previous_handlers)
    root_logger.setLevel(previous_level)
    assert events.closed is True


def test_worker_heartbeats_while_sampling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import src.rollout.sampler as sampler

    monkeypatch.setenv("FRAMEWORK_CACHE_DB", str(tmp_path / "cache" / "llm_cache.db"))
    events = _SendConnection()
    config = _config()

    async def fake_run_experiment(**kwargs):
        del kwargs
        while {"event": "heartbeat"} not in events.events:
            await asyncio.sleep(0)

    monkeypatch.setattr(worker, "HEARTBEAT_INTERVAL_SEC", 0.0)
    monkeypatch.setattr(worker.RunConfig, "load", lambda path: config)
    monkeypatch.setattr(sampler, "run_experiment", fake_run_experiment)

    worker._experiment_worker(
        "config/example.json",
        str(tmp_path),
        events,
        task_ids=("a",),
        experiment_id=None,
    )

    assert {"event": "heartbeat"} in events.events
    assert events.closed is True


def test_worker_revalidates_folded_task_panel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FRAMEWORK_CACHE_DB", str(tmp_path / "cache" / "llm_cache.db"))
    config = _config()
    events = _SendConnection()
    root_logger = logging.getLogger()
    previous_handlers = tuple(root_logger.handlers)
    previous_level = root_logger.level
    monkeypatch.setattr(worker.RunConfig, "load", lambda path: config)

    try:
        with pytest.raises(ValueError, match="task_names"):
            worker._experiment_worker(
                "config/example.json",
                str(tmp_path / "child-evals"),
                events,
                task_ids=(),
                experiment_id=None,
            )
    finally:
        root_logger.handlers = list(previous_handlers)
        root_logger.setLevel(previous_level)
    assert events.closed is True


def test_isolated_experiment_uses_measure_worktree_subprocess_and_forwards_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _MeasurementCase(
        tmp_path,
        monkeypatch,
        events=[
            ("started", "exp-1"),
            ("task", "task-a", "solved"),
            ("log", "diagnostic"),
        ],
        process=_FakeProcess(alive=True, exitcode=0),
        result=_measurement_experiment("exp-1", finished=True),
        task_ids=("task-a",),
    )
    cache_db_path = tmp_path / "main" / "cache" / "llm_cache.db"
    from src.plugins.caching import store as cache

    monkeypatch.setattr(cache, "DB_PATH", cache_db_path)
    result = case.run()

    assert result.experiment_id == "exp-1"
    assert case.context.popen_args == [
        sys.executable,
        "-m",
        "src.worker",
        "config/run_config.json",
        str(case.root),
        "17",
        '["task-a"]',
        "null",
    ]
    assert case.context.popen_kwargs["cwd"] == case.measure_root
    assert case.context.popen_kwargs["env"]["FRAMEWORK_CACHE_DB"] == str(cache_db_path)
    assert case.context.popen_kwargs["pass_fds"] == (17,)
    assert case.context.send.closed is True
    assert case.context.receive.closed is True
    assert case.started == ["exp-1"]
    assert case.tasks == [("task-a", "solved")]
    assert case.logs == ["diagnostic"]


def test_isolated_experiment_rejects_record_that_ignored_the_requested_panel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _MeasurementCase(
        tmp_path,
        monkeypatch,
        events=[("started", "exp-1")],
        process=_FakeProcess(alive=True, exitcode=0),
        result=_measurement_experiment("exp-1", finished=True),
        task_ids=("task-a", "task-b"),
    )

    with pytest.raises(RuntimeError, match="were requested"):
        case.run()


@pytest.mark.parametrize(
    ("phase", "expected"),
    [
        pytest.param("after-start", ("exp-1", True, True), id="known-experiment"),
        pytest.param("before-start", None, id="unknown-experiment"),
    ],
)
def test_isolated_experiment_failure_attaches_only_known_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    phase: str,
    expected: tuple[str, bool, bool] | None,
) -> None:
    events, saved_result = {
        "after-start": (
            [("started", "exp-1"), ("log", "boom")],
            _measurement_experiment("exp-1", finished=False),
        ),
        "before-start": ([("log", "boom")], None),
    }[phase]
    case = _MeasurementCase(
        tmp_path,
        monkeypatch,
        events=events,
        process=_FakeProcess(alive=True, exitcode=7),
        result=saved_result,
    )

    with pytest.raises(MeasurementError, match="exitcode=7") as raised:
        case.run()

    result = raised.value.result
    actual = (
        None
        if result is None
        else (
            result.experiment_id,
            "exitcode=7" in (result.crash_reason or ""),
            case.tracker.load_experiment(result.experiment_id).crash_reason
            == result.crash_reason,
        )
    )
    assert actual == expected


def test_crash_recovery_delegates_existing_reason_policy_to_store() -> None:
    result = _measurement_experiment(
        "exp-1", finished=True, crash_reason="worker failure"
    )
    calls: list[tuple[str, str]] = []

    def mark_crashed(run_id: str, *, reason: str) -> ExperimentResult:
        calls.append((run_id, reason))
        return result

    tracker = SimpleNamespace(
        load_experiment=lambda run_id: pytest.fail(
            "measurement duplicated RunStore.mark_crashed loading"
        ),
        mark_crashed=mark_crashed,
    )
    observer = SimpleNamespace(log=lambda line: pytest.fail(line))

    recovered = measurement._load_crashed_result(
        tracker=tracker,
        observer=observer,
        experiment_id="exp-1",
        reason="parent fallback",
    )

    assert recovered is result
    assert calls == [("exp-1", "parent fallback")]


def test_isolated_experiment_heartbeats_keep_long_silent_phase_alive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _MeasurementCase(
        tmp_path,
        monkeypatch,
        events=[
            {"event": "heartbeat"},
            None,
            {"event": "heartbeat"},
            None,
            ("started", "exp-1"),
        ],
        process=_FakeProcess(alive=True, exitcode=0),
        result=_measurement_experiment("exp-1", finished=True),
        task_ids=("task-a",),
    )
    monotonic = iter([0.0, 1.0, 1.4, 2.0, 2.4, 3.0])
    monkeypatch.setattr(measurement, "WATCHDOG_INACTIVITY_SEC", 0.5)
    monkeypatch.setattr(measurement.time, "monotonic", lambda: next(monotonic))
    result = case.run()

    assert result.experiment_id == "exp-1"
    assert case.process.killed is False
    assert case.heartbeats == [None, None]


def test_isolated_experiment_watchdog_kills_worker_that_stops_heartbeating(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _MeasurementCase(
        tmp_path,
        monkeypatch,
        events=[],
        process=_FakeProcess(alive=True, exitcode=0, exit_on_join=False),
    )
    monotonic = iter([0.0, 1.0])
    monkeypatch.setattr(measurement, "WATCHDOG_INACTIVITY_SEC", 0.5)
    monkeypatch.setattr(measurement.time, "monotonic", lambda: next(monotonic))

    with pytest.raises(MeasurementError, match="killed by watchdog"):
        case.run()

    assert case.process.killed is True


def test_isolated_experiment_keyboard_interrupt_signals_worker_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _MeasurementCase(
        tmp_path,
        monkeypatch,
        events=[],
        process=_FakeProcess(alive=True, exitcode=-signal.SIGINT),
    )
    monkeypatch.setattr(
        case.context.receive,
        "poll",
        lambda timeout=None: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    def terminate(child: _FakeProcess) -> int:
        assert child.signals == [signal.SIGINT]
        child.alive = False
        child.returncode = child.exitcode
        return child.exitcode

    monkeypatch.setattr(measurement, "_terminate_experiment_process", terminate)

    with pytest.raises(KeyboardInterrupt):
        case.run()

    assert case.context.receive.closed is True


def test_isolated_experiment_popen_failure_closes_pipe_ends(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = _MeasurementCase(
        tmp_path,
        monkeypatch,
        events=[],
        process=_FakeProcess(alive=True, exitcode=0),
    )
    monkeypatch.setattr(
        measurement.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cannot launch")),
    )

    with pytest.raises(OSError, match="cannot launch"):
        case.run()

    assert case.context.receive.closed is True
    assert case.context.send.closed is True


def test_isolated_experiment_invalid_event_hard_fails_and_still_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Invalid protocol events must propagate while cleanup still runs.
    case = _MeasurementCase(
        tmp_path,
        monkeypatch,
        events=[("verdict", "forged")],
        process=_FakeProcess(alive=True, exitcode=0),
    )
    terminated: list[object] = []

    def fake_terminate(child):
        terminated.append(child)
        return 0

    monkeypatch.setattr(measurement, "_terminate_experiment_process", fake_terminate)

    with pytest.raises(ValueError, match="invalid measurement event"):
        case.run()

    assert case.context.receive.closed is True
    assert terminated == [case.process]


@pytest.mark.parametrize(
    ("alive", "exit_on_join", "waits_before_exit", "expected"),
    [
        pytest.param(True, True, 0, (0, (), False), id="voluntary-exit"),
        pytest.param(True, True, 1, (0, (signal.SIGINT,), False), id="sigint-exit"),
        pytest.param(
            True,
            False,
            0,
            (-signal.SIGKILL, (signal.SIGINT,), True),
            id="sigkill-escalation",
        ),
        pytest.param(False, True, 0, (0, (), False), id="already-dead"),
    ],
)
def test_terminate_experiment_process_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    alive: bool,
    exit_on_join: bool,
    waits_before_exit: int,
    expected: tuple[int, tuple[int, ...], bool],
) -> None:
    process = _FakeProcess(
        alive=alive,
        exitcode=0,
        exit_on_join=exit_on_join,
        waits_before_exit=waits_before_exit,
    )
    swept: list[int] = []
    monkeypatch.setattr(
        measurement.os,
        "kill",
        lambda *args: pytest.fail("termination bypassed Popen.send_signal"),
    )
    monkeypatch.setattr(DockerShellSession, "sweep_owner_resources", swept.append)

    exitcode = _terminate_experiment_process(process)  # type: ignore[arg-type]

    expected_exitcode, expected_signals, expected_killed = expected
    assert exitcode == expected_exitcode
    assert process.signals == list(expected_signals)
    assert process.killed is expected_killed
    assert swept == [process.pid]


def _interrupting_process(process: _FakeProcess, *, on_wait_call: int) -> _FakeProcess:
    """Wrap process.wait so the given 1-indexed wait call raises KeyboardInterrupt;
    every other call delegates to the real _FakeProcess.wait."""
    original_wait = process.wait
    calls = 0

    def wait(timeout: float | None = None) -> int:
        nonlocal calls
        calls += 1
        if calls == on_wait_call:
            raise KeyboardInterrupt
        return original_wait(timeout)

    process.wait = wait  # type: ignore[method-assign]
    return process


def test_terminate_keyboard_interrupt_during_grace_kills_and_sweeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _interrupting_process(
        _FakeProcess(alive=True, exitcode=0, exit_on_join=False), on_wait_call=1
    )
    swept: list[int] = []
    monkeypatch.setattr(
        measurement.os,
        "kill",
        lambda *args: pytest.fail("termination bypassed Popen.send_signal"),
    )
    monkeypatch.setattr(DockerShellSession, "sweep_owner_resources", swept.append)

    assert _terminate_experiment_process(process) == -signal.SIGKILL  # type: ignore[arg-type]
    assert process.signals == [signal.SIGINT]
    assert process.killed is True
    assert swept == [process.pid]


def test_terminate_keyboard_interrupt_after_sigint_kills_and_sweeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _interrupting_process(
        _FakeProcess(alive=True, exitcode=0, waits_before_exit=1), on_wait_call=2
    )
    swept: list[int] = []
    monkeypatch.setattr(
        measurement.os,
        "kill",
        lambda *args: pytest.fail("termination bypassed Popen.send_signal"),
    )
    monkeypatch.setattr(DockerShellSession, "sweep_owner_resources", swept.append)

    assert _terminate_experiment_process(process) == -signal.SIGKILL  # type: ignore[arg-type]
    assert process.signals == [signal.SIGINT]
    assert process.killed is True
    assert swept == [process.pid]


def test_connection_fd_round_trip_through_real_subprocess() -> None:
    receive, send = multiprocessing.Pipe(duplex=False)
    script = (
        "from multiprocessing.connection import Connection; import sys, json; "
        "c=Connection(int(sys.argv[1])); "
        "c.send_bytes(json.dumps(['log', 'one']).encode()); "
        "c.send_bytes(json.dumps(['started', 'two']).encode()); c.close()"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(send.fileno())],
        pass_fds=(send.fileno(),),
    )
    send.close()
    try:
        assert json.loads(receive.recv_bytes()) == ["log", "one"]
        assert json.loads(receive.recv_bytes()) == ["started", "two"]
        assert process.wait(timeout=10) == 0
        with pytest.raises(EOFError):
            receive.recv_bytes()
    finally:
        receive.close()
