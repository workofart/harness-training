from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.control import gates, supervisor, supervisor_state
from src.control.agent_backend import (
    MissingThreadRollout,
    TurnResult,
)
from src.experiment.gate import build_gate_verdicts
from src.experiment.record import (
    CandidateChangeEvidence,
    ExperimentEvidence,
    ExperimentRecord,
    ExperimentState,
    PanelRecord,
    TaskOutcomeEvidence,
)
from src.harness.config import HarnessConfig, OpenRouterConfig
from src.harness.contracts import TaskResult

from conftest import _write_task_artifacts


def train_panel(task_ids: list[str]) -> PanelRecord:
    return PanelRecord.initialize(
        panel_id="train",
        purpose="promotion",
        task_ids=task_ids,
        expected_trial_count=1,
        lifecycle="active",
    )


def init_train_record(
    *,
    experiment_id: str,
    git_commit_hash: str,
    parent_baseline_experiment_id: str | None,
    train_task_ids: list[str],
    started_at: str = "2026-04-11T00:00:00+00:00",
) -> ExperimentRecord:
    return ExperimentRecord.initialize(
        experiment_id=experiment_id,
        git_commit_hash=git_commit_hash,
        parent_baseline_experiment_id=parent_baseline_experiment_id,
        panels=[train_panel(train_task_ids)],
        started_at=started_at,
    )


class FakeBackend:
    def __init__(self, results):
        self._results = iter(results)

    def run_turn(self, **_):
        return next(self._results)


class LoopStopped(Exception):
    pass


def freeze_supervisor_time(monkeypatch: pytest.MonkeyPatch, iso_timestamp: str) -> None:
    class FrozenDatetime:
        @classmethod
        def now(cls, tz=None):
            return datetime.fromisoformat(iso_timestamp)

        fromisoformat = staticmethod(datetime.fromisoformat)

    monkeypatch.setattr(supervisor, "datetime", FrozenDatetime)


def make_harness_config(
    *,
    experiment_id: str = "exp-next",
    focus_name: str = "core",
    task_names: list[str] | None = None,
) -> HarnessConfig:
    return HarnessConfig(
        schema_version=2,
        experiment_id=experiment_id,
        focus_name=focus_name,
        panels=[
            {
                "id": "train",
                "purpose": "promotion",
                "task_names": ["task-a"] if task_names is None else task_names,
                "task_timeout_sec": 600.0,
                "run": {"when": "always"},
                "baseline": {"required": False},
            }
        ],
        llm_provider_config=OpenRouterConfig(
            model_name="openrouter/openai/gpt-oss-20b"
        ),
    )


def write_workspace_harness_config(
    workspace_root: Path,
    *,
    experiment_id: str = "exp-next",
    focus_name: str = "core",
) -> None:
    (workspace_root / "config").mkdir(parents=True, exist_ok=True)
    (workspace_root / "config" / "harness_config.json").write_text(
        make_harness_config(
            experiment_id=experiment_id, focus_name=focus_name
        ).model_dump_json(indent=2)
        + "\n"
    )


def make_runtime_snapshot(
    tmp_path: Path,
    *,
    experiment_id: str = "exp-next",
    active_baseline_experiment_id: str | None = "baseline",
    active_baseline_record: ExperimentRecord | None = None,
) -> SimpleNamespace:
    repo_root = tmp_path / "repo"
    (repo_root / "config").mkdir(parents=True, exist_ok=True)
    (repo_root / "program.md").write_text("# program\n")
    harness_config_path = repo_root / "config" / "harness_config.json"
    harness_config = make_harness_config(experiment_id=experiment_id)
    harness_config_path.write_text(harness_config.model_dump_json(indent=2) + "\n")
    experiments_root = repo_root / "experiments"
    experiments_root.mkdir(exist_ok=True)
    ExperimentState(
        active_baseline_experiment_id=active_baseline_experiment_id,
        current_experiment_id=active_baseline_experiment_id,
        updated_at="2026-04-11T00:00:00+00:00",
    ).save(root=experiments_root)
    if active_baseline_record is not None:
        active_baseline_record.write(root=experiments_root)
    return SimpleNamespace(
        repo_root=repo_root,
        harness_config=harness_config,
        experiments_root=experiments_root,
        experiment_state=ExperimentState.load(root=experiments_root),
        active_baseline_record=active_baseline_record,
        current_candidate_record=None,
    )


def make_record(
    *,
    experiment_id: str,
    git_commit_hash: str,
    parent_baseline_experiment_id: str | None,
    status: str | None,
    reward: float,
) -> ExperimentRecord:
    record = init_train_record(
        experiment_id=experiment_id,
        git_commit_hash=git_commit_hash,
        parent_baseline_experiment_id=parent_baseline_experiment_id,
        train_task_ids=["task-a"],
    )
    record.record_task_result(
        TaskResult(
            task_name="task-a",
            reward=reward,
            solved=reward > 0.0,
            steps_used=1,
            error=None,
            trial_dir=f"/tmp/{experiment_id}/task-a",
            trace_path=f"/tmp/{experiment_id}/task-a/agent/steps.jsonl",
            verifier_stdout_path=f"/tmp/{experiment_id}/task-a/verifier.txt",
            started_at="2026-04-11T00:00:00+00:00",
            finished_at="2026-04-11T00:00:01+00:00",
        )
    )
    if status is not None:
        record.finalize(status=status, decision_reason="done")
    return record


def make_unfinished_record(
    *,
    experiment_id: str,
    git_commit_hash: str,
    parent_baseline_experiment_id: str | None,
) -> ExperimentRecord:
    record = init_train_record(
        experiment_id=experiment_id,
        git_commit_hash=git_commit_hash,
        parent_baseline_experiment_id=parent_baseline_experiment_id,
        train_task_ids=["task-a"],
    )
    record.record_task_result(
        TaskResult(
            task_name="task-a",
            reward=None,
            solved=False,
            steps_used=3,
            error=None,
            trial_dir=f"/tmp/{experiment_id}/task-a",
            trace_path=f"/tmp/{experiment_id}/task-a/agent/steps.jsonl",
            verifier_stdout_path=f"/tmp/{experiment_id}/task-a/verifier.txt",
            started_at="2026-04-11T00:00:00+00:00",
            finished_at=None,
        )
    )
    return record


def make_keep_baseline() -> ExperimentRecord:
    """The finalized keep baseline at base123 that the loop tests start from."""
    return make_record(
        experiment_id="baseline",
        git_commit_hash="base123",
        parent_baseline_experiment_id=None,
        status="keep",
        reward=1.0,
    )


def stop_loop_after(
    monkeypatch: pytest.MonkeyPatch, *snapshots: SimpleNamespace
) -> None:
    """Feed ``snapshots`` to run_supervisor_loop, one per iteration, then raise
    LoopStopped when it asks for the next -- the standard way these tests break
    out of the otherwise-infinite loop once the behavior under test has run."""
    pending = iter(snapshots)

    def fake_load_runtime_snapshot(**_):
        try:
            return next(pending)
        except StopIteration:
            raise LoopStopped()

    monkeypatch.setattr(supervisor, "load_runtime_snapshot", fake_load_runtime_snapshot)


def stub_sparse_workspace(
    monkeypatch: pytest.MonkeyPatch, workspace_root: Path
) -> None:
    """Resolve the sparse workspace to ``workspace_root`` for the whole loop."""
    monkeypatch.setattr(
        supervisor, "ensure_sparse_workspace", lambda **_: workspace_root
    )


def prime_supervisor_loop(
    monkeypatch: pytest.MonkeyPatch,
    snapshot: SimpleNamespace,
    workspace_root: Path,
) -> None:
    """The boundary stubs every full-loop test installs together: run one
    iteration on ``snapshot`` then raise LoopStopped, and resolve the sparse
    workspace to ``workspace_root``. (Tests that only need one half call
    ``stop_loop_after`` / ``stub_sparse_workspace`` directly.)"""
    stop_loop_after(monkeypatch, snapshot)
    stub_sparse_workspace(monkeypatch, workspace_root)


def test_validate_candidate_editable_paths_allows_config_and_editable_paths() -> None:
    gates.validate_candidate_editable_paths(
        changed_paths=("config/harness_config.json", "src/harness/core.py"),
    )


def test_validate_candidate_editable_paths_rejects_out_of_scope_paths() -> None:
    with pytest.raises(RuntimeError, match="outside supervisor editable paths"):
        gates.validate_candidate_editable_paths(
            changed_paths=("src/experiment/runner.py",),
        )


def test_append_supervisor_event_writes_jsonl_under_supervisor_root(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    supervisor_state.append_supervisor_event(
        repo_root=repo_root,
        root=tmp_path / "supervisor",
        event="loop_started",
        phase="prelaunch",
    )

    event_path = supervisor_state.supervisor_events_path(
        repo_root=repo_root,
        root=tmp_path / "supervisor",
    )
    payload = json.loads(event_path.read_text().splitlines()[0])
    assert payload["event"] == "loop_started"
    assert payload["fields"] == {"phase": "prelaunch"}


def test_append_supervisor_event_prints_terminal_log(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    supervisor_state.append_supervisor_event(
        repo_root=repo_root,
        root=tmp_path / "supervisor",
        event="loop_iteration_started",
    )

    captured = capsys.readouterr()
    assert "[supervisor]" in captured.out
    assert "loop_iteration_started" in captured.out


def test_launch_tracked_experiment_forwards_progress_bar_without_real_experiment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    experiments_root = tmp_path / "experiments"
    experiment_id = "exp-streamed"
    make_record(
        experiment_id=experiment_id,
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        status="keep",
        reward=1.0,
    ).write(root=experiments_root)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "print('fake exp stdout')",
                "if not sys.stderr.isatty():",
                "    raise SystemExit('stderr is not a tty')",
                "sys.stderr.write('\\r\\033[K[#####-----] 1/2 tasks (50%)\\n')",
                "sys.stderr.flush()",
            ]
        )
    )
    fake_uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")

    record = supervisor.launch_tracked_experiment(
        repo_root=repo_root,
        experiment_id=experiment_id,
        experiments_root=experiments_root,
    )

    captured = capfd.readouterr()
    assert record.experiment_id == experiment_id
    assert "fake exp stdout" in captured.out
    assert "[#####-----] 1/2 tasks (50%)" in captured.err


def test_launch_tracked_experiment_halts_on_credentials_expired_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.adapters.chatgpt_codex import (
        CODEX_CREDENTIALS_EXPIRED_EXIT_CODE,
        ChatGptCodexCredentialsExpiredError,
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    experiments_root = tmp_path / "experiments"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                f"sys.exit({CODEX_CREDENTIALS_EXPIRED_EXIT_CODE})",
            ]
        )
    )
    fake_uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")

    # A dead-credentials exit must halt for operator intervention -- never
    # return a crash record the supervisor would treat like a discard and
    # advance past, looping into the same wall.
    with pytest.raises(ChatGptCodexCredentialsExpiredError):
        supervisor.launch_tracked_experiment(
            repo_root=repo_root,
            experiment_id="exp-dead-token",
            experiments_root=experiments_root,
        )


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def make_sparse_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    (repo_root / "config").mkdir(parents=True)
    (repo_root / "src" / "harness").mkdir(parents=True)
    (repo_root / "src" / "adapters").mkdir(parents=True)
    (repo_root / "tests" / "harness").mkdir(parents=True)
    (repo_root / "src" / "__init__.py").write_text("")
    (repo_root / "src" / "serialization.py").write_text(
        "def json_safe(value):\n    return value\n"
    )
    (repo_root / "program.md").write_text("# program\n")
    (repo_root / "pyproject.toml").write_text("[project]\nname='research'\n")
    (repo_root / "uv.lock").write_text("version = 1\n")
    (repo_root / "config" / "harness_config.json").write_text(
        '{"experiment_id":"exp"}\n'
    )
    (repo_root / "src" / "harness" / "contracts.py").write_text("class RawState: ...\n")
    (repo_root / "src" / "harness" / "core.py").write_text("CORE = 1\n")
    (repo_root / "src" / "adapters" / "llm_base.py").write_text("class BaseLlm: ...\n")
    (repo_root / "src" / "experiment").mkdir(parents=True)
    (repo_root / "src" / "experiment" / "trial.py").write_text("TRIAL = 1\n")
    (repo_root / "src" / "trace.py").write_text("STEP_TRACE_FILENAME='steps.jsonl'\n")
    (repo_root / "tests" / "harness" / "test_core.py").write_text(
        "def test_ok():\n    assert True\n"
    )
    (repo_root / "tests" / "conftest.py").write_text("# shared fixtures\n")
    (repo_root / "src" / "experiment" / "runner.py").write_text("IGNORED = 1\n")
    (repo_root / "experiments").mkdir()
    (repo_root / "experiments" / "state.json").write_text("{}\n")
    _git(repo_root, "init")
    _git(repo_root, "config", "user.email", "test@example.com")
    _git(repo_root, "config", "user.name", "Test User")
    _git(repo_root, "add", ".")
    _git(repo_root, "commit", "-m", "init")
    return repo_root


def make_prepare_candidate_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    (repo_root / "config").mkdir(parents=True)
    (repo_root / "src" / "harness").mkdir(parents=True)
    (repo_root / "tests" / "harness").mkdir(parents=True)
    (repo_root / "config" / "harness_config.json").write_text(
        make_harness_config(experiment_id="exp-base").model_dump_json(indent=2) + "\n"
    )
    (repo_root / "src" / "harness" / "contracts.py").write_text("class RawState: ...\n")
    (repo_root / "src" / "harness" / "core.py").write_text("CORE = 1\n")
    (repo_root / "tests" / "harness" / "test_core.py").write_text(
        "def test_ok():\n    assert True\n"
    )
    _git(repo_root, "init")
    _git(repo_root, "config", "user.email", "test@example.com")
    _git(repo_root, "config", "user.name", "Test User")
    _git(repo_root, "add", ".")
    _git(repo_root, "commit", "-m", "init")
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()
    return repo_root, experiments_root


def test_ensure_sparse_workspace_keeps_only_visible_paths_and_experiments(
    tmp_path: Path,
) -> None:
    repo_root = make_sparse_repo(tmp_path)
    workspace_root = tmp_path / "workspace"

    supervisor.ensure_sparse_workspace(
        repo_root=repo_root,
        workspace_root=workspace_root,
    )

    assert (workspace_root / "program.md").exists()
    assert (workspace_root / "config" / "harness_config.json").exists()
    assert (workspace_root / "src" / "harness" / "contracts.py").exists()
    assert (workspace_root / "src" / "harness" / "core.py").exists()
    assert (workspace_root / "src" / "serialization.py").exists()
    assert (workspace_root / "src" / "experiment" / "trial.py").exists()
    assert (workspace_root / "tests" / "conftest.py").exists()
    assert (workspace_root / "tests" / "harness" / "test_core.py").exists()
    assert (workspace_root / "experiments").is_symlink()
    assert not (workspace_root / "src" / "experiment" / "runner.py").exists()


def test_ensure_sparse_workspace_recovers_missing_registered_worktree(
    tmp_path: Path,
) -> None:
    repo_root = make_sparse_repo(tmp_path)
    workspace_root = tmp_path / "workspace"
    supervisor.ensure_sparse_workspace(
        repo_root=repo_root,
        workspace_root=workspace_root,
    )
    shutil.rmtree(workspace_root)

    supervisor.ensure_sparse_workspace(
        repo_root=repo_root,
        workspace_root=workspace_root,
    )

    assert (workspace_root / "program.md").exists()
    assert (workspace_root / "experiments").is_symlink()


def test_workspace_changed_paths_reports_only_editable_changes(tmp_path: Path) -> None:
    repo_root = make_sparse_repo(tmp_path)
    workspace_root = tmp_path / "workspace"
    supervisor.ensure_sparse_workspace(
        repo_root=repo_root,
        workspace_root=workspace_root,
    )

    (workspace_root / "src" / "harness" / "core.py").write_text("CORE = 2\n")
    changed_paths = supervisor.workspace_changed_paths(
        workspace_root=workspace_root,
    )

    assert changed_paths == ("src/harness/core.py",)


def test_workspace_changed_paths_rejects_out_of_scope_paths(tmp_path: Path) -> None:
    repo_root = make_sparse_repo(tmp_path)
    workspace_root = tmp_path / "workspace"
    supervisor.ensure_sparse_workspace(
        repo_root=repo_root,
        workspace_root=workspace_root,
    )
    visible_extra = workspace_root / "src" / "adapters" / "llm_base.py"
    visible_extra.parent.mkdir(parents=True, exist_ok=True)
    visible_extra.write_text("class BaseLlm: pass\n")

    with pytest.raises(RuntimeError, match="outside supervisor editable paths"):
        supervisor.workspace_changed_paths(workspace_root=workspace_root)


def test_workspace_changed_paths_rejects_contract_changes(tmp_path: Path) -> None:
    repo_root = make_sparse_repo(tmp_path)
    workspace_root = tmp_path / "workspace"
    supervisor.ensure_sparse_workspace(
        repo_root=repo_root,
        workspace_root=workspace_root,
    )

    (workspace_root / "src" / "harness" / "contracts.py").write_text(
        "class RawState: pass\n"
    )

    with pytest.raises(RuntimeError, match="outside supervisor editable paths"):
        supervisor.workspace_changed_paths(workspace_root=workspace_root)


def test_latest_evidence_task_artifact_paths_uses_all_relevant_panel_outcomes(
    tmp_path: Path,
) -> None:
    baseline = init_train_record(
        experiment_id="baseline",
        git_commit_hash="base123",
        parent_baseline_experiment_id=None,
        train_task_ids=["task-a", "task-b", "task-c", "task-d"],
    )
    # task-a is solved 4/4 by the baseline so the candidate's 0/4 below is a
    # genuine Fisher regression (a single-trial 0/1-vs-1/1 dip no longer is).
    for _ in range(3):
        baseline.record_task_result(
            TaskResult(
                task_name="task-a",
                reward=1.0,
                solved=True,
                error=None,
                steps_used=1,
                started_at="2026-04-11T00:00:00+00:00",
                finished_at="2026-04-11T00:00:01+00:00",
            )
        )
    for task_name, solved in (
        ("task-a", True),
        ("task-b", False),
        ("task-c", True),
        ("task-d", False),
    ):
        artifacts = _write_task_artifacts(tmp_path / "baseline", task_name)
        baseline.record_task_result(
            TaskResult(
                task_name=task_name,
                reward=1.0 if solved else 0.0,
                solved=solved,
                error=None,
                steps_used=1,
                trial_dir=artifacts["trial_dir"],
                trace_path=artifacts["trace_path"],
                metrics_path=artifacts["metrics_path"],
                verifier_stdout_path=artifacts["verifier_stdout_path"],
                started_at="2026-04-11T00:00:00+00:00",
                finished_at="2026-04-11T00:00:01+00:00",
            )
        )

    candidate = init_train_record(
        experiment_id="candidate",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        train_task_ids=["task-a", "task-b", "task-c", "task-d"],
    )
    candidate_artifacts = {}
    # Candidate fails task-a 0/4 (vs the baseline's 4/4) -> a real regression.
    # Record the bare failures first so the artifact-bearing trial below stays
    # the last failing trial, i.e. the representative used for evidence paths.
    for _ in range(3):
        candidate.record_task_result(
            TaskResult(
                task_name="task-a",
                reward=0.0,
                solved=False,
                error=None,
                steps_used=1,
                started_at="2026-04-11T00:00:00+00:00",
                finished_at="2026-04-11T00:00:01+00:00",
            )
        )
    for task_name, solved in (
        ("task-a", False),
        ("task-b", True),
        ("task-c", True),
        ("task-d", False),
    ):
        artifacts = _write_task_artifacts(tmp_path / "candidate", task_name)
        candidate_artifacts[task_name] = artifacts
        candidate.record_task_result(
            TaskResult(
                task_name=task_name,
                reward=1.0 if solved else 0.0,
                solved=solved,
                error=None,
                steps_used=1,
                trial_dir=artifacts["trial_dir"],
                trace_path=artifacts["trace_path"],
                metrics_path=artifacts["metrics_path"],
                verifier_stdout_path=artifacts["verifier_stdout_path"],
                started_at="2026-04-11T00:00:00+00:00",
                finished_at="2026-04-11T00:00:01+00:00",
            )
        )
    candidate.finalize(
        status="discard",
        decision_reason="baseline solved task regressed",
    )
    # Mirror production: evidence reads the gate's verdict dict. Treat the
    # active baseline as the entire pool for this unit test.
    pool = {
        tid: (trials.solved_count, trials.trial_count)
        for tid, trials in baseline.panels["train"].task_results.items()
    }
    verdicts = build_gate_verdicts(candidate=candidate, pool=pool)
    candidate.refresh_evidence(baseline=baseline, verdicts=verdicts)

    assert supervisor.latest_evidence_task_artifact_paths(candidate) == (
        candidate_artifacts["task-a"]["trial_dir"],
        candidate_artifacts["task-a"]["trace_path"],
        candidate_artifacts["task-a"]["exec_log_path"],
        candidate_artifacts["task-a"]["metrics_path"],
        candidate_artifacts["task-a"]["verifier_stdout_path"],
        candidate_artifacts["task-b"]["trial_dir"],
        candidate_artifacts["task-b"]["trace_path"],
        candidate_artifacts["task-b"]["exec_log_path"],
        candidate_artifacts["task-b"]["metrics_path"],
        candidate_artifacts["task-b"]["verifier_stdout_path"],
    )


def test_validate_no_task_ids_in_workspace_diff_rejects_literal_task_id(
    tmp_path: Path,
) -> None:
    workspace_root, _experiments_root = make_prepare_candidate_repo(tmp_path)
    core_path = workspace_root / "src" / "harness" / "core.py"
    core_path.write_text(
        "\n".join(
            [
                "CORE = 1",
                "if task_name == 'count-dataset-tokens':",
                "    return special_path",
                "",
            ]
        )
    )

    with pytest.raises(RuntimeError, match="literal task ids"):
        gates.validate_no_task_ids_in_workspace_diff(
            workspace_root=workspace_root,
            task_ids=("count-dataset-tokens", "regex-log"),
        )


def test_validate_no_task_ids_in_workspace_diff_accepts_generic_change(
    tmp_path: Path,
) -> None:
    workspace_root, _experiments_root = make_prepare_candidate_repo(tmp_path)
    core_path = workspace_root / "src" / "harness" / "core.py"
    core_path.write_text(
        "\n".join(
            [
                "CORE = 1",
                "def small_reusable_mechanism(state):",
                "    return state.dirty_artifact_paths",
                "",
            ]
        )
    )

    gates.validate_no_task_ids_in_workspace_diff(
        workspace_root=workspace_root,
        task_ids=("count-dataset-tokens", "regex-log"),
    )


def test_latest_evidence_task_artifact_paths_omits_missing_paths(
    tmp_path: Path,
) -> None:
    artifacts = _write_task_artifacts(tmp_path / "candidate", "task-a")
    record = init_train_record(
        experiment_id="candidate",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        train_task_ids=["task-a"],
    )
    record.evidence = ExperimentEvidence(
        candidate_change=CandidateChangeEvidence(commit="candidate123"),
        panel_outcomes={
            "train": [
                TaskOutcomeEvidence(
                    task_id="task-a",
                    baseline_solved=True,
                    candidate_solved=False,
                    outcome="regression",
                    trial_dir=artifacts["trial_dir"],
                    agent_steps_path=artifacts["trace_path"],
                    agent_exec_log_path=artifacts["exec_log_path"],
                    metrics_path=artifacts["metrics_path"],
                    verifier_stdout_path=str(tmp_path / "candidate" / "missing.txt"),
                )
            ]
        },
    )

    assert supervisor.latest_evidence_task_artifact_paths(record) == (
        artifacts["trial_dir"],
        artifacts["trace_path"],
        artifacts["exec_log_path"],
        artifacts["metrics_path"],
    )


def test_build_prelaunch_prompt_routes_to_sources_without_policy_checklist(
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    candidate = make_record(
        experiment_id="candidate",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        status="discard",
        reward=0.0,
    )
    make_runtime_snapshot(
        tmp_path,
        active_baseline_record=baseline,
    )
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    prompt = supervisor.build_prelaunch_prompt(
        workspace_root=workspace_root,
        active_baseline_record=baseline,
        latest_candidate_record=candidate,
        evidence_artifact_paths=(
            "experiments/candidate/tasks/task-a/agent/steps.jsonl",
        ),
    )

    assert str(workspace_root / "program.md") in prompt
    assert str(workspace_root / "config" / "harness_config.json") in prompt
    assert str(workspace_root / "experiments" / "learning.md") in prompt
    assert "Patch must satisfy before tracked launch:" not in prompt
    assert "do not embed current task ids or benchmark-instance facts" not in prompt
    assert (
        str(workspace_root / "experiments" / "baseline" / "experiment.json") in prompt
    )
    assert (
        str(workspace_root / "experiments" / "candidate" / "experiment.json") in prompt
    )
    assert "experiments/candidate/tasks/task-a/agent/steps.jsonl" in prompt


def test_validate_learning_memo_update_allows_only_learning_change() -> None:
    before = {"experiment_id": "exp-1", "status": "discard"}
    after = dict(before)

    gates.validate_learning_memo_update(
        before_payload=before,
        after_payload=after,
        before_learning="# Learnings\n",
        after_learning="# Learnings\n\n- New durable memo.\n",
    )


def test_validate_learning_memo_update_rejects_experiment_record_changes() -> None:
    before = {"experiment_id": "exp-1", "status": "discard"}
    after = {"experiment_id": "exp-1", "status": "keep"}

    with pytest.raises(RuntimeError, match="modified experiment.json"):
        gates.validate_learning_memo_update(
            before_payload=before,
            after_payload=after,
            before_learning="# Learnings\n",
            after_learning="# Learnings\n\n- New durable memo.\n",
        )


def test_validate_learning_memo_update_rejects_unchanged_learning() -> None:
    payload = {"experiment_id": "exp-1", "status": "discard"}

    with pytest.raises(RuntimeError, match="learning memo was not updated"):
        gates.validate_learning_memo_update(
            before_payload=payload,
            after_payload=dict(payload),
            before_learning="# Learnings\n",
            after_learning="# Learnings\n",
        )


def test_run_prelaunch_phase_reruns_with_feedback_until_workspace_is_valid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    candidate = make_record(
        experiment_id="candidate",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        status="discard",
        reward=0.0,
    )
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    write_workspace_harness_config(
        workspace_root,
        experiment_id="exp-base",
        focus_name="new-focus",
    )
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()
    feedback_notes: list[str | None] = []
    thread_ids: list[str | None] = []
    changed_paths = iter(
        [
            (),
            ("config/harness_config.json", "src/harness/core.py"),
        ]
    )

    class TrackingBackend:
        def __init__(self):
            self._turns = iter(
                [
                    TurnResult(thread_id="thread-1"),
                    TurnResult(thread_id="thread-1"),
                ]
            )

        def run_turn(self, *, thread_id=None, **_):
            thread_ids.append(thread_id)
            return next(self._turns)

    backend = TrackingBackend()

    monkeypatch.setattr(
        supervisor,
        "build_prelaunch_prompt",
        lambda **kwargs: feedback_notes.append(kwargs["feedback_note"]) or "prompt",
    )
    monkeypatch.setattr(
        supervisor,
        "workspace_changed_paths",
        lambda **kwargs: next(changed_paths),
    )
    monkeypatch.setattr(
        supervisor,
        "validate_candidate_config_patch",
        lambda **kwargs: None,
    )
    freeze_supervisor_time(monkeypatch, "2026-05-25T18:26:46+00:00")

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    supervisor_root = tmp_path / "supervisor"

    prepared_candidate = supervisor.run_prelaunch_phase(
        workspace_root=workspace_root,
        repo_root=repo_root,
        experiments_root=experiments_root,
        active_baseline_record=baseline,
        latest_candidate_record=candidate,
        evidence_artifact_paths=(
            "experiments/candidate/tasks/task-a/agent/steps.jsonl",
        ),
        thread_id=None,
        backend=backend,
        supervisor_root=supervisor_root,
    )

    assert prepared_candidate.thread_id == "thread-1"
    assert prepared_candidate.experiment_id == "exp-20260525-182646"
    assert prepared_candidate.changed_paths == (
        "config/harness_config.json",
        "src/harness/core.py",
    )
    assert thread_ids == [None, "thread-1"]
    assert feedback_notes[0] is None
    assert feedback_notes[1].startswith(
        "candidate is not ready to launch: no tracked changes"
    )
    assert "program.md Prelaunch" in feedback_notes[1]

    events_path = supervisor_state.supervisor_events_path(
        repo_root=repo_root,
        root=supervisor_root,
    )
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    feedback_events = [e for e in events if e["event"] == "prelaunch_feedback"]
    assert len(feedback_events) == 1
    assert feedback_events[0]["fields"]["thread_id"] == "thread-1"
    assert feedback_events[0]["fields"]["note"].startswith(
        "candidate is not ready to launch: no tracked changes"
    )


def test_run_supervisor_loop_prelaunch_save_preserves_postrun_completion_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Regression for the exp-v4-0506-007 runaway (69 successive
    postrun_diagnosis_completed events for one discarded candidate).

    After complete_postrun_diagnosis runs on a discard/crash it writes
    SupervisorState(phase="prelaunch", postrun_completed_experiment_id=X),
    so _postrun_completed_for_record returns True on the next loop and
    skips the postrun block. The prelaunch on_turn_complete callback in
    run_supervisor_loop used to construct a fresh SupervisorState that
    dropped postrun_completed_experiment_id; if the prelaunch then
    aborted before producing a candidate (e.g. a codex crash), the next
    iteration loaded a saved_state with marker=None and re-fired the
    same postrun. This test pins the on_turn_complete write path: the
    marker captured at the top of the loop iteration must survive every
    intermediate save.
    """
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    write_workspace_harness_config(
        workspace_root,
        experiment_id="exp-base",
        focus_name="new-focus",
    )
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    supervisor_root = tmp_path / "supervisor"

    class TrackingBackend:
        def __init__(self):
            self._turns = iter([TurnResult(thread_id="thread-1")])

        def run_turn(self, **_):
            return next(self._turns)

    monkeypatch.setattr(
        supervisor,
        "build_prelaunch_prompt",
        lambda **kwargs: "prompt",
    )
    monkeypatch.setattr(
        supervisor,
        "workspace_changed_paths",
        lambda **kwargs: ("config/harness_config.json", "src/harness/core.py"),
    )
    monkeypatch.setattr(
        supervisor,
        "validate_candidate_config_patch",
        lambda **kwargs: None,
    )
    freeze_supervisor_time(monkeypatch, "2026-05-25T18:26:47+00:00")

    # Mirror the run_supervisor_loop construction of on_turn_complete with
    # the fix: the prior postrun_completed_experiment_id is captured at the
    # top of the loop iteration and threaded through every intermediate save.
    prior_postrun_completed = "discarded-exp"

    supervisor.run_prelaunch_phase(
        workspace_root=workspace_root,
        repo_root=repo_root,
        experiments_root=experiments_root,
        active_baseline_record=None,
        latest_candidate_record=None,
        evidence_artifact_paths=(),
        thread_id=None,
        backend=TrackingBackend(),
        on_turn_complete=lambda current_thread_id: supervisor_state.SupervisorState(
            phase="prelaunch",
            thread_id=current_thread_id,
            updated_at="2026-05-17T00:00:01+00:00",
            postrun_completed_experiment_id=prior_postrun_completed,
        ).save(repo_root=repo_root, root=supervisor_root),
        supervisor_root=supervisor_root,
    )

    reloaded = supervisor_state.SupervisorState.maybe_load(
        repo_root=repo_root, root=supervisor_root
    )
    assert reloaded is not None
    assert reloaded.thread_id == "thread-1"
    assert reloaded.postrun_completed_experiment_id == "discarded-exp"


def test_run_prelaunch_phase_rejects_live_repo_edits_and_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".git").mkdir()
    write_workspace_harness_config(workspace_root, experiment_id="exp-next")
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()

    feedback_notes: list[str | None] = []
    thread_ids: list[str | None] = []
    changed_path_responses = iter(
        [
            ("config/harness_config.json", "src/harness/core.py"),
            ("config/harness_config.json", "src/harness/core.py"),
        ]
    )

    class TrackingBackend:
        def __init__(self):
            self._turns = iter(
                [
                    TurnResult(thread_id="thread-1"),
                    TurnResult(thread_id="thread-1"),
                ]
            )

        def run_turn(self, *, thread_id=None, **_):
            thread_ids.append(thread_id)
            return next(self._turns)

    backend = TrackingBackend()
    live_repo_dirty_paths = iter(
        [
            ("src/harness/core.py",),
            (),
        ]
    )
    hard_reset_calls: list[str] = []

    monkeypatch.setattr(
        supervisor,
        "build_prelaunch_prompt",
        lambda **kwargs: feedback_notes.append(kwargs["feedback_note"]) or "prompt",
    )
    monkeypatch.setattr(
        supervisor,
        "workspace_changed_paths",
        lambda **kwargs: next(changed_path_responses),
    )
    monkeypatch.setattr(
        supervisor.control_repo,
        "changed_paths",
        lambda *, cwd: next(live_repo_dirty_paths) if cwd == repo_root else (),
    )
    monkeypatch.setattr(
        supervisor.control_repo,
        "get_head_commit",
        lambda *, cwd: "base123",
    )
    monkeypatch.setattr(
        supervisor.control_repo,
        "hard_reset",
        lambda commit_hash, *, cwd: hard_reset_calls.append(f"{cwd}:{commit_hash}"),
    )
    monkeypatch.setattr(
        supervisor,
        "validate_candidate_config_patch",
        lambda **kwargs: None,
    )

    prepared_candidate = supervisor.run_prelaunch_phase(
        workspace_root=workspace_root,
        repo_root=repo_root,
        experiments_root=experiments_root,
        active_baseline_record=baseline,
        latest_candidate_record=None,
        evidence_artifact_paths=(),
        thread_id=None,
        backend=backend,
    )

    assert prepared_candidate.thread_id == "thread-1"
    assert thread_ids == [None, "thread-1"]
    assert hard_reset_calls == [f"{repo_root}:base123"]
    assert feedback_notes == [
        None,
        (
            "candidate edited the live repo instead of the sparse workspace; "
            f"edit only within {workspace_root} and do not modify {repo_root} directly"
        ),
    ]


def test_run_prelaunch_phase_sends_task_id_lint_before_accepting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    write_workspace_harness_config(workspace_root, experiment_id="exp-next")
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()
    feedback_notes: list[str | None] = []
    thread_ids: list[str | None] = []
    lint_results = iter(
        [
            RuntimeError("candidate diff embeds literal task ids in harness paths"),
            None,
        ]
    )

    class TrackingBackend:
        def __init__(self):
            self._turns = iter(
                [
                    TurnResult(thread_id="thread-1"),
                    TurnResult(thread_id="thread-1"),
                ]
            )

        def run_turn(self, *, thread_id=None, **_):
            thread_ids.append(thread_id)
            return next(self._turns)

    def fake_task_id_lint(**_kwargs):
        result = next(lint_results)
        if result is not None:
            raise result

    monkeypatch.setattr(
        supervisor,
        "build_prelaunch_prompt",
        lambda **kwargs: feedback_notes.append(kwargs["feedback_note"]) or "prompt",
    )
    monkeypatch.setattr(
        supervisor,
        "prepare_candidate",
        lambda **kwargs: supervisor.PreparedCandidate(
            thread_id=kwargs["thread_id"],
            experiment_id="exp-next",
            changed_paths=("config/harness_config.json", "src/harness/core.py"),
            harness_config=make_harness_config(experiment_id="exp-next"),
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "validate_no_task_ids_in_workspace_diff",
        fake_task_id_lint,
    )

    prepared_candidate = supervisor.run_prelaunch_phase(
        workspace_root=workspace_root,
        repo_root=workspace_root,
        experiments_root=experiments_root,
        active_baseline_record=None,
        latest_candidate_record=None,
        evidence_artifact_paths=(),
        thread_id=None,
        backend=TrackingBackend(),
    )

    assert prepared_candidate.thread_id == "thread-1"
    assert thread_ids == [None, "thread-1"]
    assert feedback_notes == [
        None,
        "candidate diff embeds literal task ids in harness paths",
    ]


def test_run_prelaunch_phase_batches_multiple_audit_failures_into_one_feedback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # When several audit gates fail on the same agent turn, the supervisor
    # must surface every fixable issue in one feedback string rather than
    # burning one agent turn per gate.
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    write_workspace_harness_config(workspace_root, experiment_id="exp-next")
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()
    feedback_notes: list[str | None] = []
    thread_ids: list[str | None] = []

    class TrackingBackend:
        def __init__(self):
            self._turns = iter(
                [
                    TurnResult(thread_id="thread-1"),
                    TurnResult(thread_id="thread-1"),
                ]
            )

        def run_turn(self, *, thread_id=None, **_):
            thread_ids.append(thread_id)
            return next(self._turns)

    backend = TrackingBackend()

    monkeypatch.setattr(
        supervisor,
        "build_prelaunch_prompt",
        lambda **kwargs: feedback_notes.append(kwargs["feedback_note"]) or "prompt",
    )
    monkeypatch.setattr(
        supervisor,
        "prepare_candidate",
        lambda **kwargs: supervisor.PreparedCandidate(
            thread_id=kwargs["thread_id"],
            experiment_id="exp-next",
            changed_paths=("config/harness_config.json", "src/harness/core.py"),
            harness_config=make_harness_config(experiment_id="exp-next"),
        ),
    )

    novelty_returns = iter(["NOVELTY_FEEDBACK", None])
    task_id_raises = iter([True, False])

    def task_id_lint(**_kwargs):
        if next(task_id_raises):
            raise RuntimeError("TASK_ID_FEEDBACK")

    monkeypatch.setattr(
        supervisor, "validate_no_task_ids_in_workspace_diff", task_id_lint
    )
    monkeypatch.setattr(
        supervisor,
        "build_mechanism_novelty_rejection",
        lambda **_kwargs: next(novelty_returns),
    )

    prepared_candidate = supervisor.run_prelaunch_phase(
        workspace_root=workspace_root,
        repo_root=workspace_root,
        experiments_root=experiments_root,
        active_baseline_record=None,
        latest_candidate_record=None,
        evidence_artifact_paths=(),
        thread_id=None,
        backend=backend,
    )

    assert prepared_candidate.thread_id == "thread-1"
    assert thread_ids == [None, "thread-1"]
    assert feedback_notes[0] is None
    # Both failing gates must appear in a single feedback string.
    bundled = feedback_notes[1]
    assert bundled is not None
    assert "TASK_ID_FEEDBACK" in bundled
    assert "NOVELTY_FEEDBACK" in bundled


def test_prepare_candidate_rejects_config_only_focus_name_churn(
    tmp_path: Path,
) -> None:
    workspace_root, experiments_root = make_prepare_candidate_repo(tmp_path)
    config_path = workspace_root / "config" / "harness_config.json"
    updated = make_harness_config(experiment_id="exp-base").model_dump()
    updated["focus_name"] = "new-focus"
    config_path.write_text(json.dumps(updated, indent=2) + "\n")

    with pytest.raises(RuntimeError, match="no behavioral harness change"):
        supervisor.prepare_candidate(
            workspace_root=workspace_root,
            experiments_root=experiments_root,
            thread_id="thread-1",
        )


@pytest.mark.parametrize(
    ("mutate_config", "write_core_diff"),
    [
        pytest.param(
            lambda updated: updated["panels"][0].update(
                task_names=["task-a", "task-b"]
            ),
            False,
            id="task-panel",
        ),
        pytest.param(
            lambda updated: updated["llm_provider_config"].update(
                model_name="openrouter/openai/gpt-oss-120b"
            ),
            True,
            id="provider",
        ),
        pytest.param(
            lambda updated: updated.update(max_steps=updated["max_steps"] + 1),
            True,
            id="budget",
        ),
    ],
)
def test_prepare_candidate_rejects_supervisor_owned_config_field_churn(
    mutate_config,
    write_core_diff: bool,
    tmp_path: Path,
) -> None:
    workspace_root, experiments_root = make_prepare_candidate_repo(tmp_path)
    config_path = workspace_root / "config" / "harness_config.json"
    updated = make_harness_config(experiment_id="exp-base").model_dump()
    mutate_config(updated)
    config_path.write_text(json.dumps(updated, indent=2) + "\n")
    if write_core_diff:
        (workspace_root / "src" / "harness" / "core.py").write_text("CORE = 2\n")

    with pytest.raises(RuntimeError, match="supervisor-owned harness config fields"):
        supervisor.prepare_candidate(
            workspace_root=workspace_root,
            experiments_root=experiments_root,
            thread_id="thread-1",
        )


def test_prepare_candidate_allows_core_behavior_diff_and_assigns_experiment_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace_root, experiments_root = make_prepare_candidate_repo(tmp_path)
    config_path = workspace_root / "config" / "harness_config.json"
    updated = make_harness_config(experiment_id="exp-base").model_dump()
    updated["focus_name"] = "new-focus"
    config_path.write_text(json.dumps(updated, indent=2) + "\n")
    (workspace_root / "src" / "harness" / "core.py").write_text("CORE = 2\n")
    freeze_supervisor_time(monkeypatch, "2026-05-25T18:26:46+00:00")

    prepared = supervisor.prepare_candidate(
        workspace_root=workspace_root,
        experiments_root=experiments_root,
        thread_id="thread-1",
    )

    assert prepared.experiment_id == "exp-20260525-182646"
    updated_config = HarnessConfig.model_validate_json(config_path.read_text())
    assert updated_config.experiment_id == "exp-20260525-182646"
    assert updated_config.focus_name == "new-focus"
    assert prepared.changed_paths == (
        "config/harness_config.json",
        "src/harness/core.py",
    )


def test_run_supervisor_loop_reuses_same_thread_until_candidate_changes_then_launches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    snapshot = make_runtime_snapshot(tmp_path, active_baseline_record=baseline)
    thread_ids: list[str | None] = []
    commit_messages: list[str] = []
    workspace_roots: list[Path] = []
    synced_commits: list[str] = []
    hard_reset_calls: list[str] = []
    feedback_notes: list[str | None] = []
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    write_workspace_harness_config(
        workspace_root,
        experiment_id="exp-base",
        focus_name="new-focus",
    )

    class TrackingBackend:
        def __init__(self):
            self._turns = iter(
                [
                    TurnResult(thread_id="thread-1"),
                    TurnResult(thread_id="thread-1"),
                ]
            )

        def run_turn(self, *, repo_root=None, thread_id=None, **_):
            workspace_roots.append(repo_root)
            thread_ids.append(thread_id)
            return next(self._turns)

    backend = TrackingBackend()
    changed_paths = iter(
        [
            (),
            ("config/harness_config.json", "src/harness/core.py"),
        ]
    )

    prime_supervisor_loop(monkeypatch, snapshot, workspace_root)

    def fake_build_prelaunch_prompt(**kwargs):
        feedback_notes.append(kwargs["feedback_note"])
        return "prompt"

    monkeypatch.setattr(
        supervisor, "build_prelaunch_prompt", fake_build_prelaunch_prompt
    )
    monkeypatch.setattr(
        supervisor.control_repo,
        "get_head_commit",
        lambda **_: "base123",
    )
    monkeypatch.setattr(
        supervisor,
        "workspace_changed_paths",
        lambda *, workspace_root: next(changed_paths),
    )
    monkeypatch.setattr(
        supervisor,
        "validate_candidate_config_patch",
        lambda **kwargs: None,
    )
    freeze_supervisor_time(monkeypatch, "2026-05-25T18:26:47+00:00")
    monkeypatch.setattr(
        supervisor,
        "sync_sparse_workspace_to_commit",
        lambda *, workspace_root, commit_hash: synced_commits.append(commit_hash),
    )
    monkeypatch.setattr(
        supervisor,
        "commit_candidate",
        lambda **kwargs: commit_messages.append(kwargs["experiment_id"])
        or "candidate123",
    )
    monkeypatch.setattr(
        supervisor,
        "promote_workspace_commit_to_repo",
        lambda *, repo_root, commit_hash: hard_reset_calls.append(commit_hash),
    )
    monkeypatch.setattr(
        supervisor,
        "launch_tracked_experiment",
        lambda **kwargs: make_record(
            experiment_id=kwargs["experiment_id"],
            git_commit_hash="candidate123",
            parent_baseline_experiment_id="baseline",
            status="keep",
            reward=1.0,
        ),
    )
    monkeypatch.setattr(
        supervisor.control_repo,
        "hard_reset",
        lambda commit_hash, **_: hard_reset_calls.append(commit_hash),
    )
    monkeypatch.setattr(
        supervisor,
        "run_postrun_diagnosis_phase",
        lambda **_: "thread-1",
    )

    with pytest.raises(LoopStopped):
        supervisor.run_supervisor_loop(
            repo_root=snapshot.repo_root,
            supervisor_root=tmp_path / "supervisor",
            backend=backend,
        )

    assert thread_ids == [None, "thread-1"]
    assert workspace_roots == [workspace_root, workspace_root]
    assert feedback_notes[0] is None
    assert feedback_notes[1].startswith(
        "candidate is not ready to launch: no tracked changes"
    )
    assert "program.md Prelaunch" in feedback_notes[1]
    assert synced_commits == ["base123", "candidate123", "base123"]
    assert hard_reset_calls == ["candidate123"]
    assert commit_messages == ["exp-20260525-182647"]
    assert not supervisor_state.SupervisorState.path(
        repo_root=snapshot.repo_root,
        root=tmp_path / "supervisor",
    ).exists()


def test_run_supervisor_loop_logs_failure_before_raising(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    snapshot = make_runtime_snapshot(
        tmp_path,
        active_baseline_experiment_id=None,
        active_baseline_record=None,
    )
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    supervisor_root = tmp_path / "supervisor"

    monkeypatch.setattr(
        supervisor,
        "load_runtime_snapshot",
        lambda **_: snapshot,
    )
    stub_sparse_workspace(monkeypatch, workspace_root)
    monkeypatch.setattr(
        supervisor,
        "_ensure_baseline_at_head",
        lambda **_: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        supervisor,
        "current_experiment_record",
        lambda snapshot: None,
    )
    with pytest.raises(RuntimeError, match="boom"):
        supervisor.run_supervisor_loop(
            repo_root=snapshot.repo_root,
            supervisor_root=supervisor_root,
        )

    captured = capsys.readouterr()
    assert "loop_iteration_started" in captured.out
    assert "loop_iteration_failed" in captured.out
    assert "boom" in captured.out

    event_payloads = [
        json.loads(line)
        for line in supervisor_state.supervisor_events_path(
            repo_root=snapshot.repo_root,
            root=supervisor_root,
        )
        .read_text()
        .splitlines()
    ]
    assert [payload["event"] for payload in event_payloads] == [
        "loop_iteration_started",
        "loop_iteration_failed",
    ]
    assert event_payloads[-1]["fields"]["error"] == "boom"


def test_run_supervisor_loop_uses_prepared_candidate_without_reloading_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    snapshot = make_runtime_snapshot(
        tmp_path,
        experiment_id="exp-old",
        active_baseline_record=baseline,
    )
    workspace_root = tmp_path / "workspace"
    write_workspace_harness_config(workspace_root, experiment_id="exp-new")
    committed_experiment_ids: list[str] = []
    launched_experiment_ids: list[str] = []

    prime_supervisor_loop(monkeypatch, snapshot, workspace_root)
    monkeypatch.setattr(
        supervisor,
        "run_prelaunch_phase",
        lambda **_: supervisor.PreparedCandidate(
            thread_id="thread-1",
            experiment_id="exp-new",
            changed_paths=("config/harness_config.json", "src/harness/core.py"),
            harness_config=make_harness_config(experiment_id="exp-new"),
        ),
    )
    monkeypatch.setattr(
        supervisor.control_repo,
        "get_head_commit",
        lambda **_: "base123",
    )
    monkeypatch.setattr(
        supervisor,
        "commit_candidate",
        lambda **kwargs: committed_experiment_ids.append(kwargs["experiment_id"])
        or "candidate123",
    )

    monkeypatch.setattr(
        supervisor,
        "promote_workspace_commit_to_repo",
        lambda **_: None,
    )
    monkeypatch.setattr(
        supervisor,
        "load_harness_config_for_repo",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError(
                "launch path should not reload harness config after prelaunch"
            )
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "launch_tracked_experiment",
        lambda **kwargs: (
            launched_experiment_ids.append(kwargs["experiment_id"])
            or make_record(
                experiment_id=kwargs["experiment_id"],
                git_commit_hash="candidate123",
                parent_baseline_experiment_id="baseline",
                status="keep",
                reward=1.0,
            )
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "sync_sparse_workspace_to_commit",
        lambda **_: None,
    )
    monkeypatch.setattr(
        supervisor,
        "run_postrun_diagnosis_phase",
        lambda **_: "thread-1",
    )

    with pytest.raises(LoopStopped):
        supervisor.run_supervisor_loop(
            repo_root=snapshot.repo_root,
            supervisor_root=tmp_path / "supervisor",
        )

    assert committed_experiment_ids == ["exp-new"]
    assert launched_experiment_ids == ["exp-new"]


def test_run_supervisor_loop_proceeds_to_candidate_when_head_matches_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    snapshot = make_runtime_snapshot(tmp_path, active_baseline_record=baseline)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    hard_reset_calls: list[str] = []
    synced_commits: list[str] = []
    thread_ids: list[str | None] = []
    changed_paths = iter(
        [
            (),
            ("config/harness_config.json", "src/harness/core.py"),
        ]
    )
    write_workspace_harness_config(workspace_root, experiment_id="exp-next")

    stop_loop_after(monkeypatch, snapshot)
    monkeypatch.setattr(
        supervisor,
        "_ensure_baseline_at_head",
        lambda **_: False,
    )
    stub_sparse_workspace(monkeypatch, workspace_root)
    monkeypatch.setattr(
        supervisor.control_repo,
        "get_head_commit",
        lambda **_: "base123",
    )
    monkeypatch.setattr(
        supervisor.control_repo,
        "hard_reset",
        lambda commit_hash, **_: hard_reset_calls.append(commit_hash),
    )
    monkeypatch.setattr(
        supervisor,
        "sync_sparse_workspace_to_commit",
        lambda *, workspace_root, commit_hash: synced_commits.append(commit_hash),
    )

    class TrackingBackend:
        def run_turn(self, *, thread_id=None, **_):
            thread_ids.append(thread_id)
            return TurnResult(thread_id="thread-1")

    backend = TrackingBackend()
    monkeypatch.setattr(
        supervisor,
        "workspace_changed_paths",
        lambda **_: next(changed_paths),
    )
    monkeypatch.setattr(
        supervisor,
        "validate_candidate_config_patch",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        supervisor,
        "commit_candidate",
        lambda **_: "candidate123",
    )
    monkeypatch.setattr(
        supervisor,
        "promote_workspace_commit_to_repo",
        lambda *, repo_root, commit_hash: hard_reset_calls.append(commit_hash),
    )
    monkeypatch.setattr(
        supervisor,
        "launch_tracked_experiment",
        lambda **_: make_record(
            experiment_id="exp-next",
            git_commit_hash="candidate123",
            parent_baseline_experiment_id="baseline",
            status="keep",
            reward=1.0,
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "run_postrun_diagnosis_phase",
        lambda **_: "thread-1",
    )

    with pytest.raises(LoopStopped):
        supervisor.run_supervisor_loop(
            repo_root=snapshot.repo_root,
            supervisor_root=tmp_path / "supervisor",
            backend=backend,
        )

    assert thread_ids == [None, "thread-1"]
    assert hard_reset_calls == ["candidate123"]
    assert "candidate123" in synced_commits


@pytest.mark.parametrize("status", ["discard", "crash"])
def test_run_supervisor_loop_resets_to_baseline_after_failed_candidate(
    status: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    snapshot = make_runtime_snapshot(tmp_path, active_baseline_record=baseline)
    hard_reset_calls: list[str] = []
    synced_commits: list[str] = []
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    write_workspace_harness_config(workspace_root, experiment_id="exp-next")

    snapshots = iter([snapshot])

    def fake_load_runtime_snapshot(**_):
        try:
            return next(snapshots)
        except StopIteration:
            raise LoopStopped()

    backend = FakeBackend([TurnResult(thread_id="thread-1")])
    monkeypatch.setattr(supervisor, "load_runtime_snapshot", fake_load_runtime_snapshot)
    stub_sparse_workspace(monkeypatch, workspace_root)
    monkeypatch.setattr(
        supervisor.control_repo,
        "get_head_commit",
        lambda **_: "base123",
    )
    monkeypatch.setattr(
        supervisor,
        "workspace_changed_paths",
        lambda **_: ("config/harness_config.json", "src/harness/core.py"),
    )
    monkeypatch.setattr(
        supervisor,
        "validate_candidate_config_patch",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        supervisor,
        "commit_candidate",
        lambda **_: "candidate123",
    )
    monkeypatch.setattr(
        supervisor,
        "promote_workspace_commit_to_repo",
        lambda *, repo_root, commit_hash: hard_reset_calls.append(commit_hash),
    )
    monkeypatch.setattr(
        supervisor,
        "launch_tracked_experiment",
        lambda **_: make_record(
            experiment_id="exp-next",
            git_commit_hash="candidate123",
            parent_baseline_experiment_id="baseline",
            status=status,
            reward=0.0,
        ),
    )
    monkeypatch.setattr(
        supervisor.control_repo,
        "hard_reset",
        lambda commit_hash, **_: hard_reset_calls.append(commit_hash),
    )
    monkeypatch.setattr(
        supervisor,
        "sync_sparse_workspace_to_commit",
        lambda *, workspace_root, commit_hash: synced_commits.append(commit_hash),
    )
    monkeypatch.setattr(
        supervisor,
        "run_postrun_diagnosis_phase",
        lambda **_: "thread-1",
    )

    with pytest.raises(LoopStopped):
        supervisor.run_supervisor_loop(
            repo_root=snapshot.repo_root,
            supervisor_root=tmp_path / "supervisor",
            backend=backend,
        )

    assert hard_reset_calls == ["candidate123", "base123"]
    assert synced_commits == ["base123", "base123", "base123"]
    state = supervisor_state.SupervisorState.maybe_load(
        repo_root=snapshot.repo_root,
        root=tmp_path / "supervisor",
    )
    assert state is not None
    assert state.thread_id == "thread-1"
    assert state.phase == "prelaunch"


def test_run_supervisor_loop_resumes_postrun_phase_before_prelaunch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    candidate = make_record(
        experiment_id="exp-next",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        status="discard",
        reward=0.0,
    )
    snapshot = make_runtime_snapshot(tmp_path, active_baseline_record=baseline)
    candidate.write(root=snapshot.experiments_root)
    snapshot.experiment_state.current_experiment_id = "exp-next"
    snapshot.current_candidate_record = candidate
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    supervisor_state.SupervisorState(
        phase="postrun",
        thread_id="thread-1",
        updated_at="2026-04-11T00:00:00+00:00",
    ).save(
        repo_root=snapshot.repo_root,
        root=tmp_path / "supervisor",
    )

    backend_calls: list[object] = []
    diagnosis_calls: list[tuple[str, str]] = []

    class TrackingBackend:
        def run_turn(self, **kwargs):
            backend_calls.append(kwargs)
            return TurnResult(thread_id="thread-2")

    backend = TrackingBackend()

    prime_supervisor_loop(monkeypatch, snapshot, workspace_root)
    monkeypatch.setattr(
        supervisor,
        "run_postrun_diagnosis_phase",
        lambda **kwargs: diagnosis_calls.append(
            (kwargs["thread_id"], kwargs["experiment_record"].experiment_id)
        )
        or "thread-1",
    )
    monkeypatch.setattr(
        supervisor.control_repo,
        "get_head_commit",
        lambda **_: "base123",
    )
    monkeypatch.setattr(
        supervisor,
        "sync_sparse_workspace_to_commit",
        lambda **_: None,
    )

    with pytest.raises(LoopStopped):
        supervisor.run_supervisor_loop(
            repo_root=snapshot.repo_root,
            supervisor_root=tmp_path / "supervisor",
            backend=backend,
        )

    assert backend_calls == []
    assert diagnosis_calls == [("thread-1", "exp-next")]


def test_run_supervisor_loop_resumes_postrun_after_keep_advances_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = make_record(
        experiment_id="exp-next",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        status="keep",
        reward=1.0,
    )
    snapshot = make_runtime_snapshot(
        tmp_path,
        experiment_id="exp-next",
        active_baseline_experiment_id="exp-next",
        active_baseline_record=candidate,
    )
    candidate.write(root=snapshot.experiments_root)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    state_path = supervisor_state.SupervisorState.path(
        repo_root=snapshot.repo_root,
        root=tmp_path / "supervisor",
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "phase": "postrun",
                "thread_id": "thread-1",
                "updated_at": "2026-04-11T00:00:00+00:00",
            }
        )
        + "\n"
    )

    diagnosis_calls: list[tuple[str | None, str]] = []
    prime_supervisor_loop(monkeypatch, snapshot, workspace_root)
    monkeypatch.setattr(
        supervisor,
        "run_postrun_diagnosis_phase",
        lambda **kwargs: diagnosis_calls.append(
            (kwargs["thread_id"], kwargs["experiment_record"].experiment_id)
        )
        or "thread-2",
    )
    monkeypatch.setattr(
        supervisor,
        "run_prelaunch_phase",
        lambda **_: (_ for _ in ()).throw(
            AssertionError(
                "startup should finish pending keep postrun before prelaunch"
            )
        ),
    )
    monkeypatch.setattr(
        supervisor.control_repo,
        "get_head_commit",
        lambda **_: "candidate123",
    )
    monkeypatch.setattr(
        supervisor,
        "sync_sparse_workspace_to_commit",
        lambda **_: None,
    )

    with pytest.raises(LoopStopped):
        supervisor.run_supervisor_loop(
            repo_root=snapshot.repo_root,
            supervisor_root=tmp_path / "supervisor",
        )

    assert diagnosis_calls == [("thread-1", "exp-next")]


def test_run_supervisor_loop_recovers_postrun_from_concluded_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    candidate = make_record(
        experiment_id="exp-next",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        status="discard",
        reward=0.0,
    )
    snapshot = make_runtime_snapshot(tmp_path, active_baseline_record=baseline)
    candidate.write(root=snapshot.experiments_root)
    snapshot.experiment_state.current_experiment_id = "exp-next"
    snapshot.current_candidate_record = candidate
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    supervisor_state.SupervisorState(
        phase="prelaunch",
        thread_id="thread-1",
        updated_at="2026-04-11T00:00:00+00:00",
    ).save(
        repo_root=snapshot.repo_root,
        root=tmp_path / "supervisor",
    )

    diagnosis_calls: list[tuple[str, str]] = []
    prime_supervisor_loop(monkeypatch, snapshot, workspace_root)
    monkeypatch.setattr(
        supervisor.control_repo,
        "get_head_commit",
        lambda **_: "base123",
    )
    monkeypatch.setattr(
        supervisor,
        "sync_sparse_workspace_to_commit",
        lambda **_: None,
    )
    monkeypatch.setattr(
        supervisor,
        "run_postrun_diagnosis_phase",
        lambda **kwargs: diagnosis_calls.append(
            (kwargs["thread_id"], kwargs["experiment_record"].experiment_id)
        )
        or "thread-2",
    )
    monkeypatch.setattr(
        supervisor,
        "run_prelaunch_phase",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("startup should finish pending postrun before prelaunch")
        ),
    )

    with pytest.raises(LoopStopped):
        supervisor.run_supervisor_loop(
            repo_root=snapshot.repo_root,
            supervisor_root=tmp_path / "supervisor",
        )

    assert diagnosis_calls == [(None, "exp-next")]


def test_run_supervisor_loop_requires_postrun_without_saved_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    candidate = make_record(
        experiment_id="exp-next",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        status="discard",
        reward=0.0,
    )
    snapshot = make_runtime_snapshot(tmp_path, active_baseline_record=baseline)
    candidate.write(root=snapshot.experiments_root)
    snapshot.experiment_state.current_experiment_id = "exp-next"
    snapshot.current_candidate_record = candidate
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    diagnosis_calls: list[tuple[str | None, str]] = []
    prime_supervisor_loop(monkeypatch, snapshot, workspace_root)
    monkeypatch.setattr(
        supervisor.control_repo,
        "get_head_commit",
        lambda **_: "base123",
    )
    monkeypatch.setattr(
        supervisor,
        "sync_sparse_workspace_to_commit",
        lambda **_: None,
    )
    monkeypatch.setattr(
        supervisor,
        "run_postrun_diagnosis_phase",
        lambda **kwargs: diagnosis_calls.append(
            (kwargs["thread_id"], kwargs["experiment_record"].experiment_id)
        )
        or (_ for _ in ()).throw(LoopStopped()),
    )
    monkeypatch.setattr(
        supervisor,
        "run_prelaunch_phase",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("missing supervisor state should not block required postrun")
        ),
    )

    with pytest.raises(LoopStopped):
        supervisor.run_supervisor_loop(
            repo_root=snapshot.repo_root,
            supervisor_root=tmp_path / "supervisor",
        )

    assert diagnosis_calls == [(None, "exp-next")]


def test_run_supervisor_loop_restarts_prelaunch_when_learning_memo_already_updated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    candidate = make_record(
        experiment_id="exp-next",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        status="discard",
        reward=0.0,
    )
    snapshot = make_runtime_snapshot(tmp_path, active_baseline_record=baseline)
    candidate.write(root=snapshot.experiments_root)
    snapshot.experiment_state.current_experiment_id = "exp-next"
    snapshot.current_candidate_record = candidate
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    supervisor_state.SupervisorState(
        phase="prelaunch",
        thread_id="thread-1",
        updated_at="2026-04-11T00:00:00+00:00",
        postrun_completed_experiment_id="exp-next",
    ).save(
        repo_root=snapshot.repo_root,
        root=tmp_path / "supervisor",
    )

    prelaunch_thread_ids: list[str | None] = []
    prime_supervisor_loop(monkeypatch, snapshot, workspace_root)
    monkeypatch.setattr(
        supervisor,
        "run_postrun_diagnosis_phase",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("startup should derive prelaunch from experiment artifacts")
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "run_prelaunch_phase",
        lambda **kwargs: prelaunch_thread_ids.append(kwargs["thread_id"])
        or (_ for _ in ()).throw(LoopStopped()),
    )
    monkeypatch.setattr(
        supervisor.control_repo,
        "get_head_commit",
        lambda **_: "base123",
    )
    monkeypatch.setattr(
        supervisor,
        "sync_sparse_workspace_to_commit",
        lambda **_: None,
    )

    with pytest.raises(LoopStopped):
        supervisor.run_supervisor_loop(
            repo_root=snapshot.repo_root,
            supervisor_root=tmp_path / "supervisor",
        )

    assert prelaunch_thread_ids == ["thread-1"]


def test_run_supervisor_loop_preserves_completed_postrun_marker_across_iterations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    candidate = make_record(
        experiment_id="exp-next",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        status="discard",
        reward=0.0,
    )
    snapshot = make_runtime_snapshot(tmp_path, active_baseline_record=baseline)
    (snapshot.repo_root / ".git").write_text("gitdir: /tmp/fake\n")
    candidate.write(root=snapshot.experiments_root)
    snapshot.experiment_state.current_experiment_id = "exp-next"
    snapshot.current_candidate_record = candidate
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    supervisor_state.SupervisorState(
        phase="prelaunch",
        thread_id="thread-1",
        updated_at="2026-04-11T00:00:00+00:00",
        postrun_completed_experiment_id="exp-next",
    ).save(
        repo_root=snapshot.repo_root,
        root=tmp_path / "supervisor",
    )

    prelaunch_thread_ids: list[str | None] = []
    prime_supervisor_loop(monkeypatch, snapshot, workspace_root)
    monkeypatch.setattr(
        supervisor,
        "run_postrun_diagnosis_phase",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("completed postrun should not be diagnosed again")
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "run_prelaunch_phase",
        lambda **kwargs: prelaunch_thread_ids.append(kwargs["thread_id"])
        or (_ for _ in ()).throw(LoopStopped()),
    )
    monkeypatch.setattr(
        supervisor.control_repo,
        "get_head_commit",
        lambda **_: "base123",
    )
    monkeypatch.setattr(
        supervisor.control_repo,
        "changed_paths",
        lambda **_: (),
    )
    monkeypatch.setattr(
        supervisor,
        "sync_sparse_workspace_to_commit",
        lambda **_: None,
    )

    with pytest.raises(LoopStopped):
        supervisor.run_supervisor_loop(
            repo_root=snapshot.repo_root,
            supervisor_root=tmp_path / "supervisor",
        )

    assert prelaunch_thread_ids == ["thread-1"]


def test_run_supervisor_loop_restores_stale_postrun_cache_before_resuming_diagnosis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    candidate = make_record(
        experiment_id="exp-next",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        status="discard",
        reward=0.0,
    )
    snapshot = make_runtime_snapshot(tmp_path, active_baseline_record=baseline)
    candidate.write(root=snapshot.experiments_root)
    snapshot.experiment_state.current_experiment_id = "exp-next"
    record_path = ExperimentRecord.path("exp-next", root=snapshot.experiments_root)
    original_payload = json.loads(record_path.read_text())
    invalid_payload = dict(original_payload)
    invalid_payload["decision_reason"] = "tampered"
    record_path.write_text(json.dumps(invalid_payload) + "\n")
    snapshot.current_candidate_record = ExperimentRecord.load(
        "exp-next",
        root=snapshot.experiments_root,
    )
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    supervisor_state.SupervisorState(
        phase="postrun",
        thread_id="thread-1",
        updated_at="2026-04-11T00:00:00+00:00",
        postrun_original_payload=original_payload,
    ).save(
        repo_root=snapshot.repo_root,
        root=tmp_path / "supervisor",
    )

    prime_supervisor_loop(monkeypatch, snapshot, workspace_root)
    monkeypatch.setattr(
        supervisor.control_repo,
        "get_head_commit",
        lambda **_: "base123",
    )
    monkeypatch.setattr(
        supervisor,
        "sync_sparse_workspace_to_commit",
        lambda **_: None,
    )
    diagnosis_calls: list[tuple[str | None, str, str]] = []
    monkeypatch.setattr(
        supervisor,
        "run_postrun_diagnosis_phase",
        lambda **kwargs: diagnosis_calls.append(
            (
                kwargs["thread_id"],
                kwargs["original_payload"]["decision_reason"],
                ExperimentRecord.load(
                    "exp-next",
                    root=snapshot.experiments_root,
                ).decision_reason,
            )
        )
        or (_ for _ in ()).throw(LoopStopped()),
    )
    monkeypatch.setattr(
        supervisor,
        "run_prelaunch_phase",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("stale postrun cache should be restored before prelaunch")
        ),
    )

    with pytest.raises(LoopStopped):
        supervisor.run_supervisor_loop(
            repo_root=snapshot.repo_root,
            supervisor_root=tmp_path / "supervisor",
        )

    restored = ExperimentRecord.load("exp-next", root=snapshot.experiments_root)
    assert diagnosis_calls == [("thread-1", "done", "done")]
    assert restored.decision_reason == "done"


def test_run_supervisor_loop_abandons_unfinished_candidate_and_restarts_prelaunch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    candidate = make_unfinished_record(
        experiment_id="exp-next",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
    )
    snapshot = make_runtime_snapshot(tmp_path, active_baseline_record=baseline)
    candidate.write(root=snapshot.experiments_root)
    snapshot.experiment_state.current_experiment_id = "exp-next"
    snapshot.current_candidate_record = candidate
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    supervisor_state.SupervisorState(
        phase="postrun",
        thread_id="thread-1",
        updated_at="2026-04-11T00:00:00+00:00",
        postrun_original_payload={"stale": True},
    ).save(
        repo_root=snapshot.repo_root,
        root=tmp_path / "supervisor",
    )

    synced_commits: list[str] = []
    update_ref_calls: list[tuple[str, str]] = []
    snapshots = iter([snapshot])

    def fake_load_runtime_snapshot(**_):
        try:
            return next(snapshots)
        except StopIteration:
            raise LoopStopped()

    def fake_run_prelaunch_phase(**kwargs):
        latest = kwargs["latest_candidate_record"]
        assert kwargs["thread_id"] is None
        assert latest is not None
        assert latest.experiment_id == "exp-next"
        assert latest.status == "crash"
        assert latest.error == "abandoned after supervisor restart"
        raise LoopStopped()

    monkeypatch.setattr(supervisor, "load_runtime_snapshot", fake_load_runtime_snapshot)
    stub_sparse_workspace(monkeypatch, workspace_root)
    monkeypatch.setattr(
        supervisor.control_repo,
        "get_head_commit",
        lambda **_: "base123",
    )
    monkeypatch.setattr(
        supervisor.control_repo,
        "changed_paths",
        lambda **_: (),
    )
    monkeypatch.setattr(
        supervisor.control_repo,
        "update_ref",
        lambda ref_name, commit_hash, **_: update_ref_calls.append(
            (ref_name, commit_hash)
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "sync_sparse_workspace_to_commit",
        lambda *, workspace_root, commit_hash: synced_commits.append(commit_hash),
    )
    monkeypatch.setattr(
        supervisor,
        "run_postrun_diagnosis_phase",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("unfinished candidates should be abandoned, not resumed")
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "run_prelaunch_phase",
        fake_run_prelaunch_phase,
    )

    with pytest.raises(LoopStopped):
        supervisor.run_supervisor_loop(
            repo_root=snapshot.repo_root,
            supervisor_root=tmp_path / "supervisor",
        )

    updated = ExperimentRecord.load("exp-next", root=snapshot.experiments_root)
    assert updated.status == "crash"
    assert updated.error == "abandoned after supervisor restart"
    assert updated.finished_at is not None
    updated_task_a = updated.panels["train"].task_results["task-a"]
    assert updated_task_a.is_finished
    assert updated_task_a.trials[-1].error == "abandoned after supervisor restart"
    assert updated_task_a.trials[-1].metrics.failure_mode == "interrupted"
    assert updated.evidence is not None
    assert updated.evidence.candidate_change.parent_baseline_commit == "base123"
    assert len(updated.evidence.panel_outcomes["train"]) == 1
    assert updated.evidence.panel_outcomes["train"][0].baseline_solved is True
    # The abandoned trial is `interrupted` (error set), excluded from evidence,
    # so the candidate has no valid trials and no solve verdict.
    assert updated.evidence.panel_outcomes["train"][0].candidate_solved is None
    assert update_ref_calls == [
        (supervisor.failed_experiment_git_ref("exp-next"), "candidate123")
    ]
    assert synced_commits == []
    state = ExperimentState.load(root=snapshot.experiments_root)
    assert state.active_baseline_experiment_id == "baseline"
    assert state.current_experiment_id == "exp-next"
    assert (
        supervisor_state.SupervisorState.maybe_load(
            repo_root=snapshot.repo_root,
            root=tmp_path / "supervisor",
        )
        is None
    )


def test_run_supervisor_loop_prelaunch_state_keeps_only_failed_candidate_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    snapshot = make_runtime_snapshot(tmp_path, active_baseline_record=baseline)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    snapshots = iter([snapshot])

    def fake_load_runtime_snapshot(**_):
        try:
            return next(snapshots)
        except StopIteration:
            raise LoopStopped()

    def fake_run_prelaunch_phase(**kwargs):
        kwargs["on_turn_complete"]("thread-1")
        raise LoopStopped()

    monkeypatch.setattr(supervisor, "load_runtime_snapshot", fake_load_runtime_snapshot)
    stub_sparse_workspace(monkeypatch, workspace_root)
    monkeypatch.setattr(
        supervisor,
        "sync_sparse_workspace_to_commit",
        lambda **_: None,
    )
    monkeypatch.setattr(
        supervisor.control_repo,
        "get_head_commit",
        lambda **_: "base123",
    )
    monkeypatch.setattr(
        supervisor,
        "run_prelaunch_phase",
        fake_run_prelaunch_phase,
    )

    with pytest.raises(LoopStopped):
        supervisor.run_supervisor_loop(
            repo_root=snapshot.repo_root,
            supervisor_root=tmp_path / "supervisor",
        )

    state = supervisor_state.SupervisorState.maybe_load(
        repo_root=snapshot.repo_root,
        root=tmp_path / "supervisor",
    )
    assert state is not None
    assert state.phase == "prelaunch"
    assert state.thread_id == "thread-1"


def test_run_supervisor_loop_discards_sparse_workspace_and_reuses_only_prelaunch_thread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    snapshot = make_runtime_snapshot(tmp_path, active_baseline_record=baseline)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    supervisor_state.SupervisorState(
        phase="prelaunch",
        thread_id="thread-1",
        updated_at="2026-04-11T00:00:00+00:00",
    ).save(
        repo_root=snapshot.repo_root,
        root=tmp_path / "supervisor",
    )

    synced_commits: list[str] = []
    prelaunch_thread_ids: list[str | None] = []
    prime_supervisor_loop(monkeypatch, snapshot, workspace_root)
    monkeypatch.setattr(
        supervisor.control_repo,
        "get_head_commit",
        lambda **_: "base123",
    )
    monkeypatch.setattr(
        supervisor,
        "sync_sparse_workspace_to_commit",
        lambda *, workspace_root, commit_hash: synced_commits.append(commit_hash),
    )
    monkeypatch.setattr(
        supervisor,
        "run_prelaunch_phase",
        lambda **kwargs: prelaunch_thread_ids.append(kwargs["thread_id"])
        or (_ for _ in ()).throw(LoopStopped()),
    )

    with pytest.raises(LoopStopped):
        supervisor.run_supervisor_loop(
            repo_root=snapshot.repo_root,
            supervisor_root=tmp_path / "supervisor",
        )

    assert synced_commits == ["base123"]
    assert prelaunch_thread_ids == ["thread-1"]


def test_run_postrun_diagnosis_phase_reruns_until_learning_memo_is_valid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    experiments_root = repo_root / "experiments"
    experiments_root.mkdir(parents=True)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    record = make_record(
        experiment_id="candidate",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        status="discard",
        reward=0.0,
    )
    record.write(root=experiments_root)

    feedback_notes: list[str | None] = []
    thread_ids: list[str | None] = []

    def fake_build_experiment_diagnosis_prompt(**kwargs):
        feedback_notes.append(kwargs["feedback_note"])
        return "prompt"

    class SideEffectBackend:
        def run_turn(self, *, thread_id=None, **_):
            thread_ids.append(thread_id)
            loaded = json.loads(
                ExperimentRecord.path("candidate", root=experiments_root).read_text()
            )
            if len(thread_ids) == 1:
                loaded["decision_reason"] = "wrong field"
            else:
                (experiments_root / "learning.md").write_text(
                    "# Learnings\n\n- Hypothesis and evidence artifacts.\n"
                )
            ExperimentRecord.path("candidate", root=experiments_root).write_text(
                json.dumps(loaded, indent=2) + "\n"
            )
            return TurnResult(thread_id=thread_id or "thread-1")

    backend = SideEffectBackend()

    monkeypatch.setattr(
        supervisor,
        "build_experiment_diagnosis_prompt",
        fake_build_experiment_diagnosis_prompt,
    )

    thread_id = supervisor.run_postrun_diagnosis_phase(
        workspace_root=workspace_root,
        repo_root=repo_root,
        experiments_root=experiments_root,
        experiment_record=record,
        thread_id="thread-1",
        backend=backend,
    )

    assert thread_id == "thread-1"
    assert thread_ids == ["thread-1", "thread-1"]
    assert feedback_notes == [
        None,
        "diagnosis turn modified experiment.json; update only "
        "`experiments/learning.md`: decision_reason",
    ]
    updated = ExperimentRecord.load("candidate", root=experiments_root)
    assert updated.decision_reason == "done"
    assert (
        "Hypothesis and evidence artifacts"
        in (experiments_root / "learning.md").read_text()
    )


def test_run_postrun_diagnosis_phase_restores_original_payload_between_invalid_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    experiments_root = repo_root / "experiments"
    experiments_root.mkdir(parents=True)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    record = make_record(
        experiment_id="candidate",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        status="discard",
        reward=0.0,
    )
    record.write(root=experiments_root)

    attempts = {"count": 0}

    class SideEffectBackend:
        def run_turn(self, *, thread_id=None, **_):
            attempts["count"] += 1
            loaded = json.loads(
                ExperimentRecord.path("candidate", root=experiments_root).read_text()
            )
            if attempts["count"] == 1:
                loaded["decision_reason"] = "wrong field"
            else:
                assert loaded["decision_reason"] == "done"
                (experiments_root / "learning.md").write_text(
                    "# Learnings\n\n- Diagnosis.\n"
                )
            ExperimentRecord.path("candidate", root=experiments_root).write_text(
                json.dumps(loaded, indent=2) + "\n"
            )
            return TurnResult(thread_id=thread_id or "thread-1")

    backend = SideEffectBackend()

    thread_id = supervisor.run_postrun_diagnosis_phase(
        workspace_root=workspace_root,
        repo_root=repo_root,
        experiments_root=experiments_root,
        experiment_record=record,
        thread_id="thread-1",
        backend=backend,
    )

    assert thread_id == "thread-1"
    updated = ExperimentRecord.load("candidate", root=experiments_root)
    assert updated.decision_reason == "done"
    assert "Diagnosis" in (experiments_root / "learning.md").read_text()


def test_run_postrun_diagnosis_phase_recovers_from_missing_resume_thread(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    experiments_root = repo_root / "experiments"
    experiments_root.mkdir(parents=True)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    record = make_record(
        experiment_id="candidate",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        status="discard",
        reward=0.0,
    )
    record.write(root=experiments_root)

    thread_ids: list[str | None] = []

    class SideEffectBackend:
        def run_turn(self, *, thread_id=None, **_):
            thread_ids.append(thread_id)
            if thread_id == "missing-thread":
                raise MissingThreadRollout("missing-thread")
            (experiments_root / "learning.md").write_text(
                "# Learnings\n\n- Diagnosis from fresh thread.\n"
            )
            return TurnResult(thread_id="fresh-thread")

    thread_id = supervisor.run_postrun_diagnosis_phase(
        workspace_root=workspace_root,
        repo_root=repo_root,
        experiments_root=experiments_root,
        experiment_record=record,
        thread_id="missing-thread",
        backend=SideEffectBackend(),
    )

    assert thread_id == "fresh-thread"
    assert thread_ids == ["missing-thread", None]
    assert (
        "Diagnosis from fresh thread" in (experiments_root / "learning.md").read_text()
    )


def test_run_supervisor_loop_syncs_workspace_after_postrun_diagnosis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    candidate = make_record(
        experiment_id="exp-next",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        status="discard",
        reward=0.0,
    )
    snapshot = make_runtime_snapshot(tmp_path, active_baseline_record=baseline)
    candidate.write(root=snapshot.experiments_root)
    snapshot.experiment_state.current_experiment_id = "exp-next"
    snapshot.current_candidate_record = candidate
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    supervisor_state.SupervisorState(
        phase="postrun",
        thread_id="thread-1",
        updated_at="2026-04-11T00:00:00+00:00",
    ).save(
        repo_root=snapshot.repo_root,
        root=tmp_path / "supervisor",
    )

    synced_commits: list[str] = []
    prime_supervisor_loop(monkeypatch, snapshot, workspace_root)
    monkeypatch.setattr(
        supervisor,
        "run_postrun_diagnosis_phase",
        lambda **_: "thread-2",
    )
    monkeypatch.setattr(
        supervisor.control_repo,
        "get_head_commit",
        lambda **_: "base123",
    )
    monkeypatch.setattr(
        supervisor,
        "sync_sparse_workspace_to_commit",
        lambda *, workspace_root, commit_hash: synced_commits.append(commit_hash),
    )

    with pytest.raises(LoopStopped):
        supervisor.run_supervisor_loop(
            repo_root=snapshot.repo_root,
            supervisor_root=tmp_path / "supervisor",
        )

    assert synced_commits == ["base123", "base123"]


def test_ensure_baseline_at_head_no_op_when_head_matches_and_clean(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    snapshot = make_runtime_snapshot(tmp_path, active_baseline_record=baseline)
    (snapshot.repo_root / ".git").write_text("gitdir: /tmp/fake\n")

    monkeypatch.setattr(
        supervisor.control_repo,
        "changed_paths",
        lambda **_: (),
    )
    monkeypatch.setattr(
        supervisor.control_repo,
        "get_head_commit",
        lambda **_: "base123",
    )

    refreshed = supervisor._ensure_baseline_at_head(
        snapshot=snapshot,
        repo_root=snapshot.repo_root,
        workspace_root=tmp_path / "workspace",
        supervisor_root=tmp_path / "supervisor",
    )

    assert refreshed is False


def test_ensure_baseline_at_head_reruns_when_active_baseline_interrupted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # A baseline run killed mid-flight leaves state.json pointing the active
    # baseline at a record that never finalized as a keep. Even when that record
    # sits at HEAD with matching panels, the loop must discard it and measure a
    # fresh baseline -- short-circuiting here would compare every later
    # candidate against partial, never-graded evidence.
    interrupted = make_unfinished_record(
        experiment_id="baseline",
        git_commit_hash="base123",
        parent_baseline_experiment_id=None,
    )
    snapshot = make_runtime_snapshot(tmp_path, active_baseline_record=interrupted)
    (snapshot.repo_root / ".git").write_text("gitdir: /tmp/fake\n")

    new_baseline = make_record(
        experiment_id="baseline-20260411-000000",
        git_commit_hash="base123",
        parent_baseline_experiment_id=None,
        status="keep",
        reward=1.0,
    )
    baseline_calls: dict[str, object] = {}

    monkeypatch.setattr(supervisor.control_repo, "changed_paths", lambda **_: ())
    monkeypatch.setattr(
        supervisor.control_repo, "get_head_commit", lambda **_: "base123"
    )
    freeze_supervisor_time(monkeypatch, "2026-04-11T00:00:00+00:00")
    monkeypatch.setattr(
        supervisor,
        "_load_runtime",
        lambda *, repo_root: (
            SimpleNamespace(experiments_dir=snapshot.experiments_root),
            "api-key",
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "sync_sparse_workspace_to_commit",
        lambda *, workspace_root, commit_hash: None,
    )

    def fake_run_baseline(*, harness_config, harbor_config, api_key, **_kwargs):
        baseline_calls["experiment_id"] = _kwargs["experiment_id"]
        return new_baseline

    monkeypatch.setattr(
        supervisor.ExperimentRunner,
        "run_baseline_at_head",
        fake_run_baseline,
    )

    refreshed = supervisor._ensure_baseline_at_head(
        snapshot=snapshot,
        repo_root=snapshot.repo_root,
        workspace_root=tmp_path / "workspace",
        supervisor_root=tmp_path / "supervisor",
    )

    assert refreshed is True
    assert baseline_calls["experiment_id"] == "baseline-20260411-000000"
    # The interrupted record must be finalized (not left dangling) and dropped
    # as the active baseline so run_baseline_at_head seeds from a clean slate.
    discarded = ExperimentRecord.load("baseline", root=snapshot.experiments_root)
    assert discarded.is_concluded()
    reloaded_state = ExperimentState.load(root=snapshot.experiments_root)
    assert reloaded_state.active_baseline_experiment_id is None


def test_ensure_baseline_at_head_aborts_on_dirty_worktree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    snapshot = make_runtime_snapshot(tmp_path, active_baseline_record=baseline)
    (snapshot.repo_root / ".git").write_text("gitdir: /tmp/fake\n")

    monkeypatch.setattr(
        supervisor.control_repo,
        "changed_paths",
        lambda **_: ("config/harness_config.json",),
    )

    with pytest.raises(RuntimeError, match="clean worktree"):
        supervisor._ensure_baseline_at_head(
            snapshot=snapshot,
            repo_root=snapshot.repo_root,
            workspace_root=tmp_path / "workspace",
            supervisor_root=tmp_path / "supervisor",
        )


def test_ensure_baseline_at_head_runs_baseline_when_head_advanced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = make_keep_baseline()
    snapshot = make_runtime_snapshot(tmp_path, active_baseline_record=baseline)
    (snapshot.repo_root / ".git").write_text("gitdir: /tmp/fake\n")

    supervisor_state.SupervisorState(
        phase="prelaunch",
        thread_id="thread-1",
        updated_at="2026-04-11T00:00:00+00:00",
    ).save(
        repo_root=snapshot.repo_root,
        root=tmp_path / "supervisor",
    )

    expected_started_at = "2026-04-11T00:00:00+00:00"
    expected_experiment_id = "baseline-20260411-000000"
    expected_experiment_json_path = str(
        ExperimentRecord.path(expected_experiment_id, root=snapshot.experiments_root)
    )
    baseline_calls: dict[str, object] = {}
    synced_commits: list[str] = []
    new_baseline = make_record(
        experiment_id=expected_experiment_id,
        git_commit_hash="head456",
        parent_baseline_experiment_id="baseline",
        status="keep",
        reward=1.0,
    )

    monkeypatch.setattr(
        supervisor.control_repo,
        "changed_paths",
        lambda **_: (),
    )
    monkeypatch.setattr(
        supervisor.control_repo,
        "get_head_commit",
        lambda **_: "head456",
    )
    freeze_supervisor_time(monkeypatch, expected_started_at)
    monkeypatch.setattr(
        supervisor,
        "_load_runtime",
        lambda *, repo_root: (
            SimpleNamespace(experiments_dir=snapshot.experiments_root),
            "api-key",
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "load_harness_config_for_repo",
        lambda repo_root: snapshot.harness_config,
    )
    monkeypatch.setattr(
        supervisor,
        "sync_sparse_workspace_to_commit",
        lambda *, workspace_root, commit_hash: synced_commits.append(commit_hash),
    )

    def fake_run_baseline(*, harness_config, harbor_config, api_key, **_kwargs):
        baseline_calls["harness_config"] = harness_config
        baseline_calls["api_key"] = api_key
        baseline_calls["experiment_id"] = _kwargs["experiment_id"]
        baseline_calls["started_at"] = _kwargs["started_at"]
        return new_baseline

    monkeypatch.setattr(
        supervisor.ExperimentRunner,
        "run_baseline_at_head",
        fake_run_baseline,
    )

    refreshed = supervisor._ensure_baseline_at_head(
        snapshot=snapshot,
        repo_root=snapshot.repo_root,
        workspace_root=tmp_path / "workspace",
        supervisor_root=tmp_path / "supervisor",
    )

    assert refreshed is True
    assert baseline_calls["harness_config"] is snapshot.harness_config
    assert baseline_calls["api_key"] == "api-key"
    assert baseline_calls["experiment_id"] == expected_experiment_id
    assert baseline_calls["started_at"] == expected_started_at
    assert synced_commits == ["head456"]
    captured = capsys.readouterr()
    assert f"experiment_json_path={expected_experiment_json_path}" in captured.out
    event_payloads = [
        json.loads(line)
        for line in supervisor_state.supervisor_events_path(
            repo_root=snapshot.repo_root,
            root=tmp_path / "supervisor",
        )
        .read_text()
        .splitlines()
    ]
    started_event = next(
        payload
        for payload in event_payloads
        if payload["event"] == "baseline_run_started"
    )
    assert started_event["fields"]["experiment_id"] == expected_experiment_id
    assert (
        started_event["fields"]["experiment_json_path"] == expected_experiment_json_path
    )
    assert (
        supervisor_state.SupervisorState.maybe_load(
            repo_root=snapshot.repo_root,
            root=tmp_path / "supervisor",
        )
        is None
    )


def test_ensure_baseline_at_head_runs_baseline_when_panel_changed_at_same_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    snapshot = make_runtime_snapshot(tmp_path, active_baseline_record=baseline)
    snapshot.harness_config = make_harness_config(task_names=["task-a", "task-b"])
    (snapshot.repo_root / ".git").write_text("gitdir: /tmp/fake\n")

    baseline_calls: dict[str, object] = {}
    synced_commits: list[str] = []
    new_baseline = make_record(
        experiment_id="baseline-rerun",
        git_commit_hash="base123",
        parent_baseline_experiment_id="baseline",
        status="keep",
        reward=1.0,
    )

    monkeypatch.setattr(
        supervisor.control_repo,
        "changed_paths",
        lambda **_: (),
    )
    monkeypatch.setattr(
        supervisor.control_repo,
        "get_head_commit",
        lambda **_: "base123",
    )
    monkeypatch.setattr(
        supervisor,
        "_load_runtime",
        lambda *, repo_root: (
            SimpleNamespace(experiments_dir=snapshot.experiments_root),
            "api-key",
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "load_harness_config_for_repo",
        lambda repo_root: snapshot.harness_config,
    )
    monkeypatch.setattr(
        supervisor,
        "sync_sparse_workspace_to_commit",
        lambda *, workspace_root, commit_hash: synced_commits.append(commit_hash),
    )

    def fake_run_baseline(*, harness_config, harbor_config, api_key, **_kwargs):
        baseline_calls["harness_config"] = harness_config
        baseline_calls["api_key"] = api_key
        return new_baseline

    monkeypatch.setattr(
        supervisor.ExperimentRunner,
        "run_baseline_at_head",
        fake_run_baseline,
    )

    refreshed = supervisor._ensure_baseline_at_head(
        snapshot=snapshot,
        repo_root=snapshot.repo_root,
        workspace_root=tmp_path / "workspace",
        supervisor_root=tmp_path / "supervisor",
    )

    assert refreshed is True
    assert baseline_calls["harness_config"] is snapshot.harness_config
    assert baseline_calls["api_key"] == "api-key"
    assert synced_commits == ["base123"]


def test_ensure_baseline_at_head_rejects_crashed_baseline_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    snapshot = make_runtime_snapshot(tmp_path, active_baseline_record=baseline)
    (snapshot.repo_root / ".git").write_text("gitdir: /tmp/fake\n")

    synced_commits: list[str] = []
    crashed_baseline = make_record(
        experiment_id="baseline-rerun",
        git_commit_hash="head456",
        parent_baseline_experiment_id="baseline",
        status="crash",
        reward=0.0,
    )

    monkeypatch.setattr(
        supervisor.control_repo,
        "changed_paths",
        lambda **_: (),
    )
    monkeypatch.setattr(
        supervisor.control_repo,
        "get_head_commit",
        lambda **_: "head456",
    )
    monkeypatch.setattr(
        supervisor,
        "_load_runtime",
        lambda *, repo_root: (
            SimpleNamespace(experiments_dir=snapshot.experiments_root),
            "api-key",
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "load_harness_config_for_repo",
        lambda repo_root: snapshot.harness_config,
    )
    monkeypatch.setattr(
        supervisor,
        "sync_sparse_workspace_to_commit",
        lambda *, workspace_root, commit_hash: synced_commits.append(commit_hash),
    )
    monkeypatch.setattr(
        supervisor.ExperimentRunner,
        "run_baseline_at_head",
        lambda **_: crashed_baseline,
    )

    with pytest.raises(RuntimeError, match="baseline run failed"):
        supervisor._ensure_baseline_at_head(
            snapshot=snapshot,
            repo_root=snapshot.repo_root,
            workspace_root=tmp_path / "workspace",
            supervisor_root=tmp_path / "supervisor",
        )

    assert synced_commits == []
    assert (
        supervisor_state.SupervisorState.maybe_load(
            repo_root=snapshot.repo_root,
            root=tmp_path / "supervisor",
        )
        is None
    )


def test_cleanup_orphaned_experiment_artifacts_removes_partial_launch_state(
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    snapshot = make_runtime_snapshot(tmp_path, active_baseline_record=baseline)
    partial_dir = snapshot.experiments_root / "partial-dir"
    partial_dir.mkdir()
    (partial_dir / "scratch.txt").write_text("partial\n")

    orphan = init_train_record(
        experiment_id="orphan-exp",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        train_task_ids=["task-a"],
    )
    orphan.write(root=snapshot.experiments_root)

    cleaned = supervisor._cleanup_orphaned_experiment_artifacts(
        experiments_root=snapshot.experiments_root,
        current_experiment_id=snapshot.experiment_state.current_experiment_id,
    )

    assert cleaned is True
    assert not partial_dir.exists()
    assert not (snapshot.experiments_root / "orphan-exp").exists()


def test_run_supervisor_loop_syncs_workspace_before_resuming_postrun_diagnosis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = make_keep_baseline()
    candidate = make_record(
        experiment_id="exp-next",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        status="discard",
        reward=0.0,
    )
    snapshot = make_runtime_snapshot(tmp_path, active_baseline_record=baseline)
    candidate.write(root=snapshot.experiments_root)
    snapshot.experiment_state.current_experiment_id = "exp-next"
    snapshot.current_candidate_record = candidate
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    supervisor_state.SupervisorState(
        phase="postrun",
        thread_id="thread-1",
        updated_at="2026-04-11T00:00:00+00:00",
    ).save(
        repo_root=snapshot.repo_root,
        root=tmp_path / "supervisor",
    )

    calls: list[tuple[str, str]] = []
    prime_supervisor_loop(monkeypatch, snapshot, workspace_root)
    monkeypatch.setattr(
        supervisor.control_repo,
        "get_head_commit",
        lambda **_: "base123",
    )
    monkeypatch.setattr(
        supervisor,
        "sync_sparse_workspace_to_commit",
        lambda *, workspace_root, commit_hash: calls.append(("sync", commit_hash)),
    )
    monkeypatch.setattr(
        supervisor,
        "run_postrun_diagnosis_phase",
        lambda **kwargs: calls.append(("diagnosis", kwargs["thread_id"])) or "thread-2",
    )

    with pytest.raises(LoopStopped):
        supervisor.run_supervisor_loop(
            repo_root=snapshot.repo_root,
            supervisor_root=tmp_path / "supervisor",
        )

    assert calls[:2] == [("sync", "base123"), ("diagnosis", "thread-1")]
