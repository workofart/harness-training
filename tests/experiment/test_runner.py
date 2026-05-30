from __future__ import annotations

import importlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.harness.contracts import TaskResult
from src.metrics import TaskMetrics


def _install_harness_stubs() -> type:
    sys.modules.pop("src.harness.config", None)

    config_module = types.ModuleType("src.harness.config")

    @dataclass
    class HarnessConfig:
        experiment_id: str
        focus_name: str
        train_task_names: list[str]
        max_steps: int = 30
        max_concurrency: int = 1
        task_timeout_sec: float = 600.0
        max_output_retries: int = 2
        max_disallowed_retries: int = 2
        task_trials: int = 1
        llm_provider_config: object | None = None

    config_module.DEFAULT_HARNESS_CONFIG_PATH = Path("config/harness_config.json")
    config_module.HarnessConfig = HarnessConfig
    sys.modules["src.harness.config"] = config_module

    return TaskResult


def _load_experiment_runner():
    _install_harness_stubs()
    sys.modules.pop("src.experiment.runner", None)
    return importlib.import_module("src.experiment.runner")


def _write_task_artifacts(root: Path, task_name: str) -> dict[str, str]:
    task_dir = root / task_name
    agent_dir = task_dir / "agent"
    agent_dir.mkdir(parents=True)
    steps_path = agent_dir / "steps.jsonl"
    metrics_path = agent_dir / "metrics.json"
    exec_log_path = agent_dir / "exec.log"
    verifier_path = task_dir / "verifier.txt"
    for path in (steps_path, metrics_path, exec_log_path, verifier_path):
        path.write_text("{}\n")
    return {
        "trial_dir": str(task_dir),
        "trace_path": str(steps_path),
        "metrics_path": str(metrics_path),
        "verifier_stdout_path": str(verifier_path),
        "exec_log_path": str(exec_log_path),
    }


def _task_result(
    task_result_cls: type,
    *,
    task_name: str,
    reward: float | None,
    solved: bool | None = None,
    error: str | None = None,
) -> object:
    if solved is None:
        solved = error is None and reward is not None and reward > 0.0
    return task_result_cls(
        task_name=task_name,
        reward=reward,
        steps_used=1,
        error=error,
        trial_dir=None,
        verifier_stdout_path=None,
        started_at="2026-04-10T00:00:00+00:00",
        finished_at="2026-04-10T00:00:01+00:00",
        solved=solved,
    )


def _set_task_dirs(experiment_runner, root: Path, *task_names: str) -> None:
    experiment_runner._task_dirs = {
        task_name: root / "tasks" / task_name for task_name in task_names
    }


def _run_panel_for_experiment_runner(
    runner, experiment_runner, task_names: list[str]
) -> None:
    import asyncio

    asyncio.run(
        runner._run_panel(
            record=experiment_runner.record,
            experiments_root=experiment_runner.experiments_root,
            task_names=task_names,
            task_dirs=experiment_runner._task_dirs,
            harness_config=experiment_runner.harness_config,
            make_llm=experiment_runner._make_llm,
            make_env=experiment_runner._make_env,
        )
    )


def test_make_llm_for_config_builds_chatgpt_adapter(monkeypatch):
    runner = _load_experiment_runner()
    fake_chatgpt_module = types.ModuleType("src.adapters.chatgpt_codex")
    calls: dict[str, object] = {}

    class FakeChatGptCodex:
        def __init__(self, *, config):
            calls["config"] = config

    fake_chatgpt_module.ChatGptCodex = FakeChatGptCodex
    monkeypatch.setitem(sys.modules, "src.adapters.chatgpt_codex", fake_chatgpt_module)
    config = types.SimpleNamespace(provider="chatgpt_codex")

    llm = runner._make_llm_for_config(config=config, api_key=None)

    assert isinstance(llm, FakeChatGptCodex)
    assert calls["config"] is config


class _FakeHarborConfig(SimpleNamespace):
    def model_copy(self, *, update):
        return _FakeHarborConfig(**{**self.__dict__, **update})


def _stub_baseline_task_environment(monkeypatch, tmp_path: Path) -> None:
    from src.adapters import env as adapter_env

    class FakeTaskDirectoryResolver:
        def __init__(self, config):
            self.config = config

        def resolve(self, task_names):
            return {
                task_name: tmp_path / "tasks" / task_name for task_name in task_names
            }

    monkeypatch.setattr(adapter_env, "TaskDirectoryResolver", FakeTaskDirectoryResolver)
    monkeypatch.setattr(adapter_env, "Harbor", lambda *args, **kwargs: object())


def _write_baseline_record(
    runner,
    experiments_root: Path,
    *,
    experiment_id: str = "baseline",
    git_commit_hash: str = "base123",
    task_ids: list[str] | None = None,
) -> object:
    resolved_task_ids = ["train-a"] if task_ids is None else task_ids
    baseline = runner.ExperimentRecord.initialize(
        experiment_id=experiment_id,
        git_commit_hash=git_commit_hash,
        parent_baseline_experiment_id=None,
        train_task_ids=resolved_task_ids,
        focus_name="focus",
        started_at="2026-04-10T00:00:00+00:00",
    )
    for task_id in resolved_task_ids:
        baseline.record_task_result(
            TaskResult(
                task_name=task_id,
                reward=1.0,
                solved=True,
                steps_used=1,
                error=None,
                trial_dir=f"/tmp/baseline/{task_id}",
                started_at="2026-04-10T00:00:00+00:00",
                finished_at="2026-04-10T00:00:01+00:00",
            )
        )
    baseline.finalize(status="keep", decision_reason="seed baseline")
    baseline.write(root=experiments_root)
    return baseline


def test_experiment_record_load_reads_panel_schema(tmp_path):
    runner = _load_experiment_runner()

    root = tmp_path / "experiments"
    record_dir = root / "exp-panels"
    record_dir.mkdir(parents=True)
    payload = {
        "experiment_id": "exp-panels",
        "parent_baseline_experiment_id": None,
        "git_commit_hash": "abc123",
        "focus_name": "action-set",
        "train_task_ids": ["train-b", "train-a"],
        "status": "keep",
        "train_solved_count": 1,
        "decision_reason": "train improvement",
        "error": "",
        "started_at": "2026-04-10T00:00:00+00:00",
        "finished_at": "2026-04-10T00:00:01+00:00",
        "train_task_results": {
            "train-a": {
                "task_name": "train-a",
                "expected_trial_count": 1,
                "trials": [
                    {
                        "task_name": "train-a",
                        "reward": 1.0,
                        "steps_used": 1,
                        "error": None,
                        "trial_dir": None,
                        "trace_path": None,
                        "metrics_path": None,
                        "verifier_stdout_path": None,
                        "metrics": {},
                        "started_at": "2026-04-10T00:00:00+00:00",
                        "finished_at": "2026-04-10T00:00:01+00:00",
                        "solved": True,
                    }
                ],
            },
            "train-b": {
                "task_name": "train-b",
                "expected_trial_count": 1,
                "trials": [
                    {
                        "task_name": "train-b",
                        "reward": 0.0,
                        "steps_used": 1,
                        "error": None,
                        "trial_dir": None,
                        "trace_path": None,
                        "metrics_path": None,
                        "verifier_stdout_path": None,
                        "metrics": {},
                        "started_at": "2026-04-10T00:00:00+00:00",
                        "finished_at": "2026-04-10T00:00:01+00:00",
                        "solved": False,
                    }
                ],
            },
        },
        "evidence": {
            "candidate_change": {
                "commit": "abc123",
                "parent_baseline_experiment_id": None,
                "parent_baseline_commit": None,
            },
            "task_outcomes": [],
        },
    }
    (record_dir / "experiment.json").write_text(json.dumps(payload))

    record = runner.ExperimentRecord.load("exp-panels", root=root)

    assert record.train_task_ids == ["train-a", "train-b"]
    assert record.train_solved_count == 1
    assert sorted(record.train_task_results) == ["train-a", "train-b"]


def test_experiment_record_updates_train_counts():
    task_result_cls = _install_harness_stubs()
    runner = _load_experiment_runner()
    record = runner.ExperimentRecord.initialize(
        experiment_id="exp-panels",
        git_commit_hash="abc123",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-a"],
        started_at="2026-04-10T00:00:00+00:00",
    )

    record.record_task_result(
        _task_result(task_result_cls, task_name="train-a", reward=1.0)
    )

    assert record.train_solved_count == 1


@pytest.mark.parametrize(
    ("reward", "solved"),
    [
        pytest.param(1.0, False, id="positive-reward-unsolved-stays-unsolved"),
        pytest.param(0.0, True, id="zero-reward-solved-stays-solved"),
    ],
)
def test_experiment_record_trusts_task_result_solved(
    reward: float,
    solved: bool,
):
    task_result_cls = _install_harness_stubs()
    runner = _load_experiment_runner()
    record = runner.ExperimentRecord.initialize(
        experiment_id="exp-panels",
        git_commit_hash="abc123",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-a"],
        started_at="2026-04-10T00:00:00+00:00",
    )

    record.record_task_result(
        _task_result(
            task_result_cls,
            task_name="train-a",
            reward=reward,
            solved=solved,
        )
    )

    assert record.train_task_results["train-a"].trials[0].solved is solved
    assert record.train_task_results["train-a"].majority_solved is solved
    assert record.train_solved_count == (1 if solved else 0)


def test_experiment_record_evidence_marks_outcomes_without_rule_internals(
    tmp_path,
):
    runner = _load_experiment_runner()
    baseline = runner.ExperimentRecord.initialize(
        experiment_id="baseline",
        git_commit_hash="base123",
        parent_baseline_experiment_id=None,
        train_task_ids=["already-solved", "hard-task"],
        started_at="2026-04-10T00:00:00+00:00",
    )
    baseline.record_task_result(
        TaskResult(
            task_name="already-solved",
            reward=1.0,
            solved=True,
            steps_used=1,
            error=None,
            trial_dir="/tmp/baseline/already-solved",
            trace_path="/tmp/baseline/already-solved/agent/steps.jsonl",
            verifier_stdout_path="/tmp/baseline/already-solved/verifier.txt",
            started_at="2026-04-10T00:00:00+00:00",
            finished_at="2026-04-10T00:00:01+00:00",
        )
    )
    baseline.record_task_result(
        TaskResult(
            task_name="hard-task",
            reward=0.0,
            solved=False,
            steps_used=1,
            error=None,
            trial_dir="/tmp/baseline/hard-task",
            trace_path="/tmp/baseline/hard-task/agent/steps.jsonl",
            verifier_stdout_path="/tmp/baseline/hard-task/verifier.txt",
            started_at="2026-04-10T00:00:00+00:00",
            finished_at="2026-04-10T00:00:01+00:00",
        )
    )

    candidate = runner.ExperimentRecord.initialize(
        experiment_id="candidate",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        train_task_ids=["already-solved", "hard-task"],
        started_at="2026-04-10T00:00:00+00:00",
    )
    already_solved_artifacts = _write_task_artifacts(
        tmp_path / "candidate-artifacts", "already-solved"
    )
    candidate.record_task_result(
        TaskResult(
            task_name="already-solved",
            reward=0.0,
            solved=False,
            steps_used=4,
            error=None,
            trial_dir=already_solved_artifacts["trial_dir"],
            trace_path=already_solved_artifacts["trace_path"],
            metrics_path=already_solved_artifacts["metrics_path"],
            verifier_stdout_path=already_solved_artifacts["verifier_stdout_path"],
            started_at="2026-04-10T00:00:00+00:00",
            finished_at="2026-04-10T00:00:01+00:00",
        )
    )
    hard_task_artifacts = _write_task_artifacts(
        tmp_path / "candidate-artifacts", "hard-task"
    )
    candidate.record_task_result(
        TaskResult(
            task_name="hard-task",
            reward=1.0,
            solved=True,
            steps_used=8,
            error=None,
            trial_dir=hard_task_artifacts["trial_dir"],
            trace_path=hard_task_artifacts["trace_path"],
            metrics_path=hard_task_artifacts["metrics_path"],
            verifier_stdout_path=hard_task_artifacts["verifier_stdout_path"],
            started_at="2026-04-10T00:00:00+00:00",
            finished_at="2026-04-10T00:00:01+00:00",
        )
    )
    candidate.finalize(
        status="discard",
        decision_reason="baseline solved task regressed",
    )

    # Mirror production: evidence labels are derived from the gate's
    # verdict dict, not from a separate majority-bool comparison. Treat
    # the active baseline as the entire pool for this unit test.
    pool = {
        tid: (trials.solved_count, trials.trial_count)
        for tid, trials in baseline.train_task_results.items()
    }
    verdicts = runner.build_gate_verdicts(candidate=candidate, pool=pool)
    candidate.refresh_evidence(baseline=baseline, verdicts=verdicts)

    assert candidate.evidence.candidate_change.commit == "candidate123"
    assert candidate.evidence.candidate_change.parent_baseline_commit == "base123"
    outcomes = {
        outcome.task_id: outcome for outcome in candidate.evidence.task_outcomes
    }
    assert outcomes["hard-task"].outcome == "new_solve"
    assert (
        outcomes["hard-task"].agent_exec_log_path
        == hard_task_artifacts["exec_log_path"]
    )
    assert outcomes["already-solved"].outcome == "regression"

    experiments_root = tmp_path / "experiments"
    candidate.write(root=experiments_root)
    payload = json.loads(
        (experiments_root / "candidate" / "experiment.json").read_text()
    )
    assert payload["evidence"]["task_outcomes"][1]["outcome"] == "new_solve"
    assert set(payload["evidence"]["task_outcomes"][1]) == {
        "task_id",
        "baseline_solved",
        "candidate_solved",
        "outcome",
        "trial_dir",
        "agent_steps_path",
        "agent_exec_log_path",
        "metrics_path",
        "verifier_stdout_path",
        "error",
    }
    loaded = runner.ExperimentRecord.load("candidate", root=experiments_root)
    loaded_outcomes = {
        outcome.task_id: outcome for outcome in loaded.evidence.task_outcomes
    }
    assert loaded_outcomes["hard-task"].outcome == "new_solve"


def test_evidence_outcome_follows_gate_verdict_not_majority_flip(tmp_path):
    # Regression: before Phase 4 the persisted outcome label was derived
    # from majority_solved booleans against the single baseline record.
    # That mechanism disagreed with the gate whenever the candidate's
    # rate flipped majority without statistical significance. Now both
    # the gate decision and the persisted label come from one verdict
    # dict, so the disagreement can no longer happen.
    runner = _load_experiment_runner()
    baseline = runner.ExperimentRecord.initialize(
        experiment_id="baseline",
        git_commit_hash="base123",
        parent_baseline_experiment_id=None,
        train_task_ids=["wobbly"],
        started_at="2026-04-10T00:00:00+00:00",
        expected_trial_count=10,
    )
    # Baseline majority-solves 5/10 -> majority_solved is True (>= ceil(10/2)).
    for solved in [True] * 5 + [False] * 5:
        baseline.record_task_result(
            TaskResult(
                task_name="wobbly",
                reward=1.0 if solved else 0.0,
                solved=solved,
                steps_used=1,
                error=None,
                started_at="2026-04-10T00:00:00+00:00",
                finished_at="2026-04-10T00:00:01+00:00",
            )
        )

    candidate = runner.ExperimentRecord.initialize(
        experiment_id="candidate",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        train_task_ids=["wobbly"],
        started_at="2026-04-10T00:00:00+00:00",
        expected_trial_count=10,
    )
    # Candidate slips to 4/10 -> majority_solved is False. Old majority-flip
    # logic would label this "regression". The gate's binomial at p_hat=0.5
    # does not reject H0 at alpha=0.05 (p_value > 0.5), so the verdict is
    # "unchanged" and the new label is "unchanged_unsolved".
    for solved in [True] * 4 + [False] * 6:
        candidate.record_task_result(
            TaskResult(
                task_name="wobbly",
                reward=1.0 if solved else 0.0,
                solved=solved,
                steps_used=1,
                error=None,
                started_at="2026-04-10T00:00:00+00:00",
                finished_at="2026-04-10T00:00:01+00:00",
            )
        )
    candidate.finalize(status="discard", decision_reason="no train improvement")
    pool = {
        tid: (trials.solved_count, trials.trial_count)
        for tid, trials in baseline.train_task_results.items()
    }
    verdicts = runner.build_gate_verdicts(candidate=candidate, pool=pool)
    candidate.refresh_evidence(baseline=baseline, verdicts=verdicts)

    outcome = candidate.evidence.task_outcomes[0]
    assert outcome.outcome == "unchanged_unsolved"
    assert verdicts["wobbly"].kind == "unchanged"


def test_single_source_of_truth_gate_and_evidence_agree():
    # End-to-end consolidation guarantee: one verdict dict per task drives
    # the promotion decision and the persisted evidence label, so they cannot
    # disagree about what happened on a task.
    runner = _load_experiment_runner()

    def _pool(baseline):
        return {
            tid: (trials.solved_count, trials.trial_count)
            for tid, trials in baseline.train_task_results.items()
        }

    # Three tasks crafted to exercise three different verdict kinds:
    #   frontier: pool 0/0  -> no-baseline frontier; candidate majority-solves
    #             -> verdict.kind="improvement", outcome="new_solve"
    #   solid:    pool 10/10 -> pool-100%; candidate 3/10
    #             -> verdict.kind="regression", outcome="regression"
    #   wobbly:   pool 5/10  -> partial; candidate 4/10
    #             -> verdict.kind="unchanged", outcome="unchanged_unsolved"
    baseline = _make_record(
        runner,
        experiment_id="baseline",
        parent=None,
        train_ids=["frontier", "solid", "wobbly"],
        k=10,
    )
    _record_solves(runner, baseline, "solid", [True] * 10)
    _record_solves(runner, baseline, "wobbly", [True] * 5 + [False] * 5)
    # frontier: no baseline trials.
    candidate = _make_record(
        runner,
        experiment_id="cand",
        parent="baseline",
        train_ids=["frontier", "solid", "wobbly"],
        k=10,
    )
    _record_solves(runner, candidate, "frontier", [True] * 6 + [False] * 4)
    _record_solves(runner, candidate, "solid", [True] * 3 + [False] * 7)
    _record_solves(runner, candidate, "wobbly", [True] * 4 + [False] * 6)

    pool = _pool(baseline)
    verdicts = runner.build_gate_verdicts(candidate=candidate, pool=pool)

    # Verdict layer
    assert verdicts["frontier"].kind == "improvement"
    assert verdicts["solid"].kind == "regression"
    assert verdicts["wobbly"].kind == "unchanged"

    # Decision layer reads the same verdicts. Regression wins over
    # improvement within a candidate.
    status, reason = runner.decide_from_verdicts(candidate=candidate, verdicts=verdicts)
    assert (status, reason) == ("discard", "train task solid regressed")

    # Evidence layer reads the same verdicts. Labels are the 5-state
    # vocabulary derived from verdict.kind + majority bools.
    candidate.finalize(status=status, decision_reason=reason)
    candidate.refresh_evidence(baseline=baseline, verdicts=verdicts)
    outcomes = {
        outcome.task_id: outcome.outcome for outcome in candidate.evidence.task_outcomes
    }
    assert outcomes["frontier"] == "new_solve"
    assert outcomes["solid"] == "regression"
    assert outcomes["wobbly"] == "unchanged_unsolved"


def test_evidence_uses_pool_verdict_not_baseline_only_recomputation():
    # Companion to the multi-kind end-to-end test above: that test happens
    # to construct `pool` as an exact mirror of the baseline record, so a
    # future refactor that reverted evidence to compare against the
    # baseline record directly (same function, different inputs) would
    # produce identical labels and slip past the assertions. This test
    # forces the pool to disagree with baseline-only so the false-negative
    # has nowhere to hide.
    #
    # Setup: baseline solved task "X" 1/1. The pool includes 7 additional
    # rule-untouched failing trials from prior candidates -> pool = 1/8
    # (12.5% rate). Current candidate fails 0/1.
    #   pool-based verdict:    candidate 0/1 vs pool 1/8, binomial not
    #                          significant -> kind="unchanged",
    #                          outcome="unchanged_unsolved"
    #   baseline-only verdict: candidate 0/1 vs baseline 1/1, p_hat=1.0,
    #                          p_value=0 -> kind="regression",
    #                          outcome="regression"
    # If evidence silently recomputes from the baseline record, this assertion
    # flips to "regression" and the test fails.
    runner = _load_experiment_runner()
    baseline = _make_record(
        runner,
        experiment_id="baseline",
        parent=None,
        train_ids=["X"],
        k=1,
    )
    _record_solves(runner, baseline, "X", [True])

    candidate = _make_record(
        runner,
        experiment_id="cand",
        parent="baseline",
        train_ids=["X"],
        k=1,
    )
    _record_solves(runner, candidate, "X", [False])

    # Pool deliberately differs from baseline alone: baseline contributes
    # 1/1 and seven prior candidates contributed 0/1 each.
    pool = {"X": (1, 8)}

    verdicts = runner.build_gate_verdicts(candidate=candidate, pool=pool)
    assert verdicts["X"].kind == "unchanged"
    # Sanity-check: had the gate been pointed at baseline alone, the verdict
    # would have flipped. Keeping this assertion documents the contrast that
    # the test relies on.
    baseline_only = runner.build_gate_verdicts(
        candidate=candidate,
        pool={"X": (1, 1)},
    )
    assert baseline_only["X"].kind == "regression"

    status, reason = runner.decide_from_verdicts(candidate=candidate, verdicts=verdicts)
    assert (status, reason) == (
        "discard",
        "no train task improvement reached significance",
    )

    candidate.finalize(status=status, decision_reason=reason)
    candidate.refresh_evidence(baseline=baseline, verdicts=verdicts)

    outcomes = {
        outcome.task_id: outcome.outcome for outcome in candidate.evidence.task_outcomes
    }
    assert outcomes["X"] == "unchanged_unsolved"


def test_experiment_record_evidence_omits_missing_artifact_paths(tmp_path):
    runner = _load_experiment_runner()
    existing_trial_dir = tmp_path / "trial"
    existing_agent_dir = existing_trial_dir / "agent"
    existing_agent_dir.mkdir(parents=True)
    steps_path = existing_agent_dir / "steps.jsonl"
    metrics_path = existing_agent_dir / "metrics.json"
    exec_log_path = existing_agent_dir / "exec.log"
    for path in (steps_path, metrics_path, exec_log_path):
        path.write_text("{}\n")

    candidate = runner.ExperimentRecord.initialize(
        experiment_id="candidate",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id=None,
        train_task_ids=["task-a"],
        started_at="2026-04-10T00:00:00+00:00",
    )
    candidate.record_task_result(
        TaskResult(
            task_name="task-a",
            reward=0.0,
            solved=False,
            steps_used=4,
            error=None,
            trial_dir=str(existing_trial_dir),
            trace_path=str(steps_path),
            metrics_path=str(metrics_path),
            verifier_stdout_path=str(tmp_path / "missing-verifier.txt"),
            started_at="2026-04-10T00:00:00+00:00",
            finished_at="2026-04-10T00:00:01+00:00",
        )
    )
    candidate.finalize(status="discard", decision_reason="no train improvement")
    candidate.refresh_evidence(baseline=None)

    outcome = candidate.evidence.task_outcomes[0]
    assert outcome.trial_dir == str(existing_trial_dir)
    assert outcome.agent_steps_path == str(steps_path)
    assert outcome.agent_exec_log_path == str(exec_log_path)
    assert outcome.metrics_path == str(metrics_path)
    assert outcome.verifier_stdout_path is None


def _make_record(runner, *, experiment_id, parent, train_ids, k):
    return runner.ExperimentRecord.initialize(
        experiment_id=experiment_id,
        git_commit_hash=experiment_id,
        parent_baseline_experiment_id=parent,
        train_task_ids=train_ids,
        started_at="2026-05-05T00:00:00+00:00",
        expected_trial_count=k,
    )


def _record_solves(runner, record, task_id, solves):
    for solved in solves:
        record.record_task_result(
            TaskResult(
                task_name=task_id,
                reward=1.0 if solved else 0.0,
                solved=solved,
                error=None,
                steps_used=1,
                started_at="2026-05-05T00:00:00+00:00",
                finished_at="2026-05-05T00:00:01+00:00",
            )
        )


def _pool_from_baseline(baseline) -> dict[str, tuple[int, int]]:
    """Convert a baseline record's per-task trial counts into the (solved,
    total) pool dict the gate primitives expect. Tests that used to thread
    the baseline record directly into evaluate() now thread it through this
    helper instead."""
    if baseline is None:
        return {}
    return {
        tid: (trials.solved_count, trials.trial_count)
        for tid, trials in baseline.train_task_results.items()
    }


def _gate(runner, *, candidate, pool) -> tuple[str, str]:
    """Two-step gate composition for tests that only care about the
    decision. Production code does the same two calls in
    ``_run_experiment`` but holds onto the verdict dict so it can thread
    it into evidence; tests below only assert on (status, reason)."""
    verdicts = runner.build_gate_verdicts(candidate=candidate, pool=pool)
    return runner.decide_from_verdicts(candidate=candidate, verdicts=verdicts)


def test_evaluate_keeps_candidate_with_significant_train_gain_at_k10():
    runner = _load_experiment_runner()
    baseline = _make_record(
        runner,
        experiment_id="baseline",
        parent=None,
        train_ids=["frontier"],
        k=10,
    )
    # 1/10 baseline (not 0/10) so the binomial path applies — at p_hat=0
    # compare_candidate_against_baseline switches to a majority-solve rule,
    # which 4/10 would not satisfy.
    _record_solves(runner, baseline, "frontier", [True] + [False] * 9)
    candidate = _make_record(
        runner,
        experiment_id="cand",
        parent="baseline",
        train_ids=["frontier"],
        k=10,
    )
    _record_solves(runner, candidate, "frontier", [True] * 4 + [False] * 6)

    status, reason = _gate(
        runner, candidate=candidate, pool=_pool_from_baseline(baseline)
    )
    assert status == "keep"
    assert reason.startswith("train task frontier improved")


@pytest.mark.parametrize(
    ("train_ids", "baseline_solves", "candidate_solves"),
    [
        pytest.param(
            ["frontier"],
            {"frontier": [True] * 5 + [False] * 5},
            {"frontier": [True] * 6 + [False] * 4},
            id="small-train-gain",
        ),
        pytest.param(
            ["a", "h"],
            {"a": [True] * 6 + [False] * 4, "h": [True] * 3 + [False] * 7},
            {"a": [True] * 6 + [False] * 4, "h": [True] * 3 + [False] * 7},
            id="identical-candidate",
        ),
        pytest.param(
            ["a"],
            {"a": [True] * 10},
            {},
            id="empty-candidate-panel",
        ),
    ],
)
def test_evaluate_discards_without_significant_train_improvement(
    train_ids, baseline_solves, candidate_solves
):
    runner = _load_experiment_runner()
    baseline = _make_record(
        runner,
        experiment_id="baseline",
        parent=None,
        train_ids=train_ids,
        k=10,
    )
    for task_id, solves in baseline_solves.items():
        _record_solves(runner, baseline, task_id, solves)
    candidate = _make_record(
        runner,
        experiment_id="cand",
        parent="baseline",
        train_ids=train_ids,
        k=10,
    )
    for task_id, solves in candidate_solves.items():
        _record_solves(runner, candidate, task_id, solves)

    assert _gate(runner, candidate=candidate, pool=_pool_from_baseline(baseline)) == (
        "discard",
        "no train task improvement reached significance",
    )


def test_evaluate_discards_when_baseline_solved_task_loses_majority():
    runner = _load_experiment_runner()
    baseline = _make_record(
        runner,
        experiment_id="baseline",
        parent=None,
        train_ids=["guard"],
        k=3,
    )
    _record_solves(runner, baseline, "guard", [True, True, True])
    candidate = _make_record(
        runner,
        experiment_id="cand",
        parent="baseline",
        train_ids=["guard"],
        k=3,
    )
    _record_solves(runner, candidate, "guard", [True, False, False])

    assert _gate(runner, candidate=candidate, pool=_pool_from_baseline(baseline)) == (
        "discard",
        "train task guard regressed",
    )


def test_evaluate_uses_rates_not_counts_when_trial_counts_differ():
    runner = _load_experiment_runner()
    baseline = _make_record(
        runner,
        experiment_id="baseline",
        parent=None,
        train_ids=["frontier"],
        k=47,
    )
    _record_solves(runner, baseline, "frontier", [True] * 25 + [False] * 22)
    candidate = _make_record(
        runner,
        experiment_id="cand",
        parent="baseline",
        train_ids=["frontier"],
        k=10,
    )
    _record_solves(runner, candidate, "frontier", [True] * 10)

    status, reason = _gate(
        runner, candidate=candidate, pool=_pool_from_baseline(baseline)
    )
    assert status == "keep"
    assert reason == "train task frontier improved"


def test_evaluate_no_baseline_discards_unsolved_candidate():
    # With an empty pool, every task is treated as no-baseline frontier
    # (baseline_solved == 0). A candidate that fails to majority-solve
    # cannot promote itself, so the gate must discard.
    runner = _load_experiment_runner()
    candidate = _make_record(
        runner,
        experiment_id="cand",
        parent=None,
        train_ids=["a"],
        k=3,
    )
    _record_solves(runner, candidate, "a", [False, False, False])
    assert _gate(runner, candidate=candidate, pool={}) == (
        "discard",
        "no train task improvement reached significance",
    )


def test_evaluate_train_regression_wins_over_train_improvement():
    # Mixed-panel candidate: one train task improves significantly while
    # another regresses. The regression pass runs first across the whole
    # panel, so regressions must take priority over later improvements.
    runner = _load_experiment_runner()
    baseline = _make_record(
        runner,
        experiment_id="baseline",
        parent=None,
        train_ids=["winner", "loser"],
        k=10,
    )
    _record_solves(runner, baseline, "winner", [False] * 10)
    _record_solves(runner, baseline, "loser", [True] * 10)
    candidate = _make_record(
        runner,
        experiment_id="cand",
        parent="baseline",
        train_ids=["winner", "loser"],
        k=10,
    )
    _record_solves(runner, candidate, "winner", [True] * 6 + [False] * 4)
    _record_solves(runner, candidate, "loser", [False] * 10)

    assert _gate(runner, candidate=candidate, pool=_pool_from_baseline(baseline)) == (
        "discard",
        "train task loser regressed",
    )


def test_evaluate_keeps_when_no_baseline_task_solved_with_no_baseline_entry():
    # A task with no baseline train_task_results entry can still appear in a
    # candidate panel after a panel edit. If the candidate majority-solves it,
    # evaluate() must treat that as "improvement" on first measurement.
    runner = _load_experiment_runner()
    baseline = _make_record(
        runner,
        experiment_id="baseline",
        parent=None,
        train_ids=["b"],
        k=6,
    )
    _record_solves(runner, baseline, "b", [True] * 4 + [False] * 2)
    candidate = _make_record(
        runner,
        experiment_id="cand",
        parent="baseline",
        train_ids=["b", "c"],  # c has no baseline entry
        k=6,
    )
    _record_solves(runner, candidate, "b", [True] * 4 + [False] * 2)
    _record_solves(runner, candidate, "c", [True] * 5 + [False])

    # Under compare_candidate_against_baseline, a missing pool entry is the
    # baseline_solved == 0 path: a candidate that majority-solves becomes an
    # "improvement". The reason reads like any other train-task improvement.
    assert _gate(runner, candidate=candidate, pool=_pool_from_baseline(baseline)) == (
        "keep",
        "train task c improved",
    )


def test_evaluate_discards_when_no_baseline_task_unsolved():
    # Task with no baseline entry: if the candidate fails
    # to majority-solve it, that does not establish a de-facto baseline and
    # cannot drive promotion.
    runner = _load_experiment_runner()
    baseline = _make_record(
        runner,
        experiment_id="baseline",
        parent=None,
        train_ids=["b"],
        k=6,
    )
    _record_solves(runner, baseline, "b", [True] * 4 + [False] * 2)
    candidate = _make_record(
        runner,
        experiment_id="cand",
        parent="baseline",
        train_ids=["b", "c"],
        k=6,
    )
    _record_solves(runner, candidate, "b", [True] * 4 + [False] * 2)
    _record_solves(runner, candidate, "c", [False] * 6)

    assert _gate(runner, candidate=candidate, pool=_pool_from_baseline(baseline)) == (
        "discard",
        "no train task improvement reached significance",
    )


def test_task_trials_majority_solved_with_k_trials():
    runner = _load_experiment_runner()
    trials = runner.TaskTrials(task_name="t", expected_trial_count=3)
    assert trials.majority_solved is None
    assert trials.is_finished is False

    def make(solved: bool):
        return TaskResult(
            task_name="t",
            reward=1.0 if solved else 0.0,
            solved=solved,
            error=None,
            steps_used=1,
            trial_dir=None,
            trace_path=None,
            metrics_path=None,
            verifier_stdout_path=None,
            started_at="2026-05-05T00:00:00+00:00",
            finished_at="2026-05-05T00:00:01+00:00",
        )

    trials.append(make(True))
    trials.append(make(False))
    trials.append(make(True))
    assert trials.solved_count == 2
    assert trials.majority_solved is True
    assert trials.is_finished is True


def test_task_trials_unfinished_trials_do_not_block_completion_after_padding():
    runner = _load_experiment_runner()
    trials = runner.TaskTrials(task_name="t", expected_trial_count=1)
    in_progress = TaskResult(
        task_name="t",
        reward=None,
        solved=None,
        error=None,
        steps_used=0,
        started_at="2026-05-05T00:00:00+00:00",
        finished_at=None,
    )
    trials.append(in_progress)
    assert trials.is_finished is False

    finalized = TaskResult(
        task_name="t",
        reward=0.0,
        solved=False,
        error="abandoned",
        steps_used=0,
        started_at="2026-05-05T00:00:00+00:00",
        finished_at="2026-05-05T00:00:02+00:00",
    )
    trials.append(finalized)
    # A finalized-but-errored trial is a `crash`: it counts toward the budget
    # (so the panel terminates) but is excluded from solve scoring, so there is
    # no valid evidence and therefore no majority verdict.
    assert trials.is_finished is True
    assert trials.valid_trials == []
    assert trials.majority_solved is None


def test_run_panel_early_stops_after_majority_decided_when_trials_agree(
    monkeypatch, tmp_path
):
    runner = _load_experiment_runner()
    harness_config_cls = sys.modules["src.harness.config"].HarnessConfig

    monkeypatch.setattr(runner.control_repo, "require_clean_worktree", lambda: None)
    monkeypatch.setattr(runner.control_repo, "get_head_commit", lambda: "abc")

    experiment_runner = runner.ExperimentRunner(
        harness_config=harness_config_cls(
            experiment_id="exp-trials",
            focus_name="focus",
            train_task_names=["task-a", "task-b"],
            task_trials=3,
            max_concurrency=1,
        ),
        harbor_config=type(
            "HarborConfig",
            (),
            {"experiments_dir": tmp_path / "experiments"},
        )(),
        api_key="key",
    )

    call_log: list[str] = []

    async def fake_run_task(*, task_name, **_kwargs):
        # Yield to the event loop so the majority watcher in run_task_trials
        # gets a chance to observe each completion and cancel siblings.
        await asyncio.sleep(0)
        call_log.append(task_name)
        return TaskResult(
            task_name=task_name,
            reward=1.0,
            solved=True,
            error=None,
            steps_used=1,
            trial_dir=f"/tmp/{task_name}/{len(call_log)}",
            trace_path=None,
            metrics_path=None,
            verifier_stdout_path=None,
            metrics=TaskMetrics(),
            started_at="2026-05-05T00:00:00+00:00",
            finished_at="2026-05-05T00:00:01+00:00",
        )

    monkeypatch.setattr(runner, "run_task", fake_run_task)
    monkeypatch.setattr(experiment_runner, "_make_llm", lambda: object(), raising=False)
    monkeypatch.setattr(
        experiment_runner,
        "_make_env",
        lambda task_name, *, task_dir: object(),
        raising=False,
    )
    _set_task_dirs(experiment_runner, tmp_path, "task-a", "task-b")

    import asyncio

    _run_panel_for_experiment_runner(runner, experiment_runner, ["task-a", "task-b"])

    # With k=3 all-pass: trial 1 + trial 2 = majority decided True; trial 3
    # is cancelled by the majority watcher before it can record a result, and
    # expected_trial_count shrinks to len(finished_trials). Cancelled trials
    # leave no synthetic record entry.
    for task_id in ("task-a", "task-b"):
        trials = experiment_runner.record._task_trials(task_id)
        assert trials.trial_count == 2
        assert trials.expected_trial_count == 2
        assert trials.is_finished is True
        assert trials.majority_solved is True
    assert sorted(call_log) == ["task-a"] * 2 + ["task-b"] * 2


def test_run_panel_calls_run_task_with_current_contract(monkeypatch, tmp_path):
    runner = _load_experiment_runner()
    harness_config_cls = sys.modules["src.harness.config"].HarnessConfig

    monkeypatch.setattr(runner.control_repo, "require_clean_worktree", lambda: None)
    monkeypatch.setattr(runner.control_repo, "get_head_commit", lambda: "abc")

    experiment_runner = runner.ExperimentRunner(
        harness_config=harness_config_cls(
            experiment_id="exp-run-task-contract",
            focus_name="focus",
            train_task_names=["task-a"],
            task_trials=1,
            max_concurrency=1,
        ),
        harbor_config=type(
            "HarborConfig",
            (),
            {"experiments_dir": tmp_path / "experiments"},
        )(),
        api_key="key",
    )

    call_log: list[str] = []

    async def fake_run_task(
        *,
        task_name,
        llm,
        env,
        max_steps,
        max_output_retries=2,
        task_timeout_sec=None,
        trace_path=None,
    ):
        del llm, env, max_steps, max_output_retries, task_timeout_sec, trace_path
        call_log.append(task_name)
        return TaskResult(
            task_name=task_name,
            reward=1.0,
            solved=True,
            error=None,
            steps_used=1,
            trial_dir=None,
            trace_path=None,
            metrics_path=None,
            verifier_stdout_path=None,
            metrics=TaskMetrics(),
            started_at="2026-05-05T00:00:00+00:00",
            finished_at="2026-05-05T00:00:01+00:00",
        )

    monkeypatch.setattr(runner, "run_task", fake_run_task)
    monkeypatch.setattr(experiment_runner, "_make_llm", lambda: object(), raising=False)
    monkeypatch.setattr(
        experiment_runner,
        "_make_env",
        lambda task_name, *, task_dir: object(),
        raising=False,
    )
    _set_task_dirs(experiment_runner, tmp_path, "task-a")

    _run_panel_for_experiment_runner(runner, experiment_runner, ["task-a"])

    assert call_log == ["task-a"]
    trials = experiment_runner.record._task_trials("task-a")
    assert trials.trials[0].solved is True
    assert trials.trials[0].error is None


def test_run_panel_passes_task_timeout_sec_to_trial_boundary(monkeypatch, tmp_path):
    runner = _load_experiment_runner()
    harness_config_cls = sys.modules["src.harness.config"].HarnessConfig

    monkeypatch.setattr(runner.control_repo, "require_clean_worktree", lambda: None)
    monkeypatch.setattr(runner.control_repo, "get_head_commit", lambda: "abc")

    experiment_runner = runner.ExperimentRunner(
        harness_config=harness_config_cls(
            experiment_id="exp-run-task-timeout",
            focus_name="focus",
            train_task_names=["task-a"],
            task_trials=1,
            max_concurrency=1,
            task_timeout_sec=0.01,
        ),
        harbor_config=type(
            "HarborConfig",
            (),
            {"experiments_dir": tmp_path / "experiments"},
        )(),
        api_key="key",
    )

    seen_timeouts: list[float | None] = []

    async def fake_run_task(
        *,
        task_name,
        llm,
        env,
        max_steps,
        max_output_retries=2,
        task_timeout_sec=None,
        trace_path=None,
    ):
        del llm, env, max_steps, max_output_retries, trace_path
        seen_timeouts.append(task_timeout_sec)
        return TaskResult(
            task_name=task_name,
            reward=0.0,
            solved=False,
            error=None,
            steps_used=1,
            metrics=TaskMetrics(failure_mode="hit_timeout"),
            started_at="2026-05-05T00:00:00+00:00",
            finished_at="2026-05-05T00:00:01+00:00",
        )

    monkeypatch.setattr(runner, "run_task", fake_run_task)
    monkeypatch.setattr(experiment_runner, "_make_llm", lambda: object(), raising=False)
    monkeypatch.setattr(
        experiment_runner,
        "_make_env",
        lambda task_name, *, task_dir: object(),
        raising=False,
    )
    _set_task_dirs(experiment_runner, tmp_path, "task-a")

    _run_panel_for_experiment_runner(runner, experiment_runner, ["task-a"])

    trials = experiment_runner.record._task_trials("task-a")
    assert seen_timeouts == [0.01]
    assert trials.trials[0].solved is False
    assert trials.trials[0].error is None
    assert trials.trials[0].metrics.failure_mode == "hit_timeout"


def test_run_panel_runs_all_trials_when_first_two_split(monkeypatch, tmp_path):
    runner = _load_experiment_runner()
    harness_config_cls = sys.modules["src.harness.config"].HarnessConfig

    monkeypatch.setattr(runner.control_repo, "require_clean_worktree", lambda: None)
    monkeypatch.setattr(runner.control_repo, "get_head_commit", lambda: "abc")

    experiment_runner = runner.ExperimentRunner(
        harness_config=harness_config_cls(
            experiment_id="exp-split",
            focus_name="focus",
            train_task_names=["task-a"],
            task_trials=3,
            max_concurrency=1,
        ),
        harbor_config=type(
            "HarborConfig",
            (),
            {"experiments_dir": tmp_path / "experiments"},
        )(),
        api_key="key",
    )

    call_log: list[str] = []

    async def fake_run_task(*, task_name, **_kwargs):
        call_log.append(task_name)
        solved = len(call_log) != 2  # trial 1 pass, trial 2 fail, trial 3 pass
        return TaskResult(
            task_name=task_name,
            reward=1.0 if solved else 0.0,
            solved=solved,
            error=None,
            steps_used=1,
            trial_dir=f"/tmp/{task_name}/{len(call_log)}",
            trace_path=None,
            metrics_path=None,
            verifier_stdout_path=None,
            metrics=TaskMetrics(),
            started_at="2026-05-05T00:00:00+00:00",
            finished_at="2026-05-05T00:00:01+00:00",
        )

    monkeypatch.setattr(runner, "run_task", fake_run_task)
    monkeypatch.setattr(experiment_runner, "_make_llm", lambda: object(), raising=False)
    monkeypatch.setattr(
        experiment_runner,
        "_make_env",
        lambda task_name, *, task_dir: object(),
        raising=False,
    )
    _set_task_dirs(experiment_runner, tmp_path, "task-a")

    _run_panel_for_experiment_runner(runner, experiment_runner, ["task-a"])

    # First two trials split (1 pass, 1 fail) → majority undecided → trial 3
    # must run. expected_trial_count stays at 3.
    assert call_log == ["task-a", "task-a", "task-a"]
    trials = experiment_runner.record._task_trials("task-a")
    assert trials.trial_count == 3
    assert trials.expected_trial_count == 3
    assert trials.is_finished is True
    assert trials.majority_solved is True


def test_run_panel_early_stops_when_two_trials_fail(monkeypatch, tmp_path):
    runner = _load_experiment_runner()
    harness_config_cls = sys.modules["src.harness.config"].HarnessConfig

    monkeypatch.setattr(runner.control_repo, "require_clean_worktree", lambda: None)
    monkeypatch.setattr(runner.control_repo, "get_head_commit", lambda: "abc")

    experiment_runner = runner.ExperimentRunner(
        harness_config=harness_config_cls(
            experiment_id="exp-fail",
            focus_name="focus",
            train_task_names=["task-a"],
            task_trials=3,
            max_concurrency=1,
        ),
        harbor_config=type(
            "HarborConfig",
            (),
            {"experiments_dir": tmp_path / "experiments"},
        )(),
        api_key="key",
    )

    call_log: list[str] = []

    async def fake_run_task(*, task_name, **_kwargs):
        # Yield to the event loop so the majority watcher in run_task_trials
        # gets a chance to observe each completion and cancel siblings.
        await asyncio.sleep(0)
        call_log.append(task_name)
        return TaskResult(
            task_name=task_name,
            reward=0.0,
            solved=False,
            error=None,
            steps_used=1,
            trial_dir=f"/tmp/{task_name}/{len(call_log)}",
            trace_path=None,
            metrics_path=None,
            verifier_stdout_path=None,
            metrics=TaskMetrics(),
            started_at="2026-05-05T00:00:00+00:00",
            finished_at="2026-05-05T00:00:01+00:00",
        )

    monkeypatch.setattr(runner, "run_task", fake_run_task)
    monkeypatch.setattr(experiment_runner, "_make_llm", lambda: object(), raising=False)
    monkeypatch.setattr(
        experiment_runner,
        "_make_env",
        lambda task_name, *, task_dir: object(),
        raising=False,
    )
    _set_task_dirs(experiment_runner, tmp_path, "task-a")

    import asyncio

    _run_panel_for_experiment_runner(runner, experiment_runner, ["task-a"])

    # Trials 1+2 fail → majority decided False; trial 3 must be cancelled
    # before it records a result.
    assert call_log == ["task-a", "task-a"]
    trials = experiment_runner.record._task_trials("task-a")
    assert trials.trial_count == 2
    assert trials.expected_trial_count == 2
    assert trials.is_finished is True
    assert trials.majority_solved is False


def test_run_panel_requires_resolved_task_dir(monkeypatch, tmp_path):
    runner = _load_experiment_runner()
    harness_config_cls = sys.modules["src.harness.config"].HarnessConfig

    monkeypatch.setattr(runner.control_repo, "require_clean_worktree", lambda: None)
    monkeypatch.setattr(runner.control_repo, "get_head_commit", lambda: "abc")

    experiment_runner = runner.ExperimentRunner(
        harness_config=harness_config_cls(
            experiment_id="exp-missing-task-dir",
            focus_name="focus",
            train_task_names=["task-a"],
            task_trials=1,
            max_concurrency=1,
        ),
        harbor_config=type(
            "HarborConfig",
            (),
            {"experiments_dir": tmp_path / "experiments"},
        )(),
        api_key="key",
    )
    monkeypatch.setattr(experiment_runner, "_make_llm", lambda: object(), raising=False)

    with pytest.raises(KeyError, match="task-a"):
        _run_panel_for_experiment_runner(runner, experiment_runner, ["task-a"])


def test_task_trials_is_deterministic_solved_predicate():
    runner = _load_experiment_runner()
    finished_at = "2026-05-05T00:00:00+00:00"

    def trial(*, solved):
        return TaskResult(
            task_name="t",
            reward=1.0 if solved else 0.0,
            solved=solved,
            error=None,
            steps_used=1,
            started_at=finished_at,
            finished_at=finished_at,
        )

    empty = runner.TaskTrials(task_name="t", expected_trial_count=3)
    assert empty.is_deterministic_solved is False

    one_pass = runner.TaskTrials(
        task_name="t", expected_trial_count=3, trials=[trial(solved=True)]
    )
    assert one_pass.is_deterministic_solved is True

    two_pass = runner.TaskTrials(
        task_name="t",
        expected_trial_count=3,
        trials=[trial(solved=True), trial(solved=True)],
    )
    assert two_pass.is_deterministic_solved is True

    one_fail = runner.TaskTrials(
        task_name="t", expected_trial_count=3, trials=[trial(solved=False)]
    )
    assert one_fail.is_deterministic_solved is False

    mixed = runner.TaskTrials(
        task_name="t",
        expected_trial_count=3,
        trials=[trial(solved=True), trial(solved=False)],
    )
    assert mixed.is_deterministic_solved is False


def test_run_experiment_runs_single_trial_for_deterministic_baseline_task(
    monkeypatch, tmp_path
):
    runner = _load_experiment_runner()
    harness_config_cls = sys.modules["src.harness.config"].HarnessConfig

    monkeypatch.setattr(runner.control_repo, "require_clean_worktree", lambda: None)
    monkeypatch.setattr(runner.control_repo, "get_head_commit", lambda: "abc")

    experiment_runner = runner.ExperimentRunner(
        harness_config=harness_config_cls(
            experiment_id="exp-det",
            focus_name="focus",
            train_task_names=["train-det", "train-noise"],
            task_trials=3,
            max_concurrency=2,
        ),
        harbor_config=type(
            "HarborConfig",
            (),
            {"experiments_dir": tmp_path / "experiments"},
        )(),
        api_key="key",
    )

    task_result_cls = TaskResult
    baseline = runner.ExperimentRecord.initialize(
        experiment_id="baseline",
        git_commit_hash="bca",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-det", "train-noise"],
        started_at="2026-04-10T00:00:00+00:00",
        expected_trial_count=3,
    )
    # train-det: 2/2 all-pass (deterministic-solved).
    for _ in range(2):
        baseline.record_task_result(
            _task_result(task_result_cls, task_name="train-det", reward=1.0)
        )
    baseline.train_task_results["train-det"].expected_trial_count = 2
    # train-noise: 2/3 pass (not deterministic).
    for solved in (True, False, True):
        baseline.record_task_result(
            _task_result(
                task_result_cls,
                task_name="train-noise",
                reward=1.0 if solved else 0.0,
            )
        )

    call_log: list[str] = []
    counter = 0

    async def fake_run_task(*, task_name, **_kwargs):
        # Stagger completions so the majority watcher gets a turn between
        # trial completions and can cancel pending siblings under the
        # parallel trial scheduler.
        nonlocal counter
        counter += 1
        await asyncio.sleep(counter * 0.001)
        call_log.append(task_name)

        # All trials pass for both tasks; we want to see deterministic stop
        # at 1 and non-deterministic stop at 2 (via #3 early-stop).
        return TaskResult(
            task_name=task_name,
            reward=1.0,
            solved=True,
            error=None,
            steps_used=1,
            trial_dir=f"/tmp/{task_name}/{len(call_log)}",
            trace_path=None,
            metrics_path=None,
            verifier_stdout_path=None,
            metrics=TaskMetrics(),
            started_at="2026-05-15T00:00:00+00:00",
            finished_at="2026-05-15T00:00:01+00:00",
        )

    monkeypatch.setattr(runner, "run_task", fake_run_task)
    monkeypatch.setattr(experiment_runner, "_make_llm", lambda: object(), raising=False)
    monkeypatch.setattr(
        experiment_runner,
        "_make_env",
        lambda task_name, *, task_dir: object(),
        raising=False,
    )
    _set_task_dirs(experiment_runner, tmp_path, "train-det", "train-noise")
    monkeypatch.setattr(experiment_runner, "_conclude_experiment", lambda: None)

    import asyncio

    asyncio.run(
        experiment_runner._run_experiment(
            baseline=baseline,
        )
    )

    # train-det runs 1 trial (k=1 budget from deterministic baseline, passes).
    # train-noise runs 2 trials (k=3 budget, early-stops on 2/2 pass via #3).
    assert sorted(call_log) == ["train-det"] + ["train-noise"] * 2
    det_trials = experiment_runner.record._task_trials("train-det")
    assert det_trials.expected_trial_count == 1
    assert det_trials.trial_count == 1
    assert det_trials.majority_solved is True
    noise_trials = experiment_runner.record._task_trials("train-noise")
    assert noise_trials.expected_trial_count == 2
    assert noise_trials.trial_count == 2
    assert noise_trials.majority_solved is True


def test_run_experiment_confirms_on_fail_when_deterministic_trial_fails(
    monkeypatch, tmp_path
):
    runner = _load_experiment_runner()
    harness_config_cls = sys.modules["src.harness.config"].HarnessConfig

    monkeypatch.setattr(runner.control_repo, "require_clean_worktree", lambda: None)
    monkeypatch.setattr(runner.control_repo, "get_head_commit", lambda: "abc")

    experiment_runner = runner.ExperimentRunner(
        harness_config=harness_config_cls(
            experiment_id="exp-confirm",
            focus_name="focus",
            train_task_names=["train-det"],
            task_trials=3,
            max_concurrency=1,
        ),
        harbor_config=type(
            "HarborConfig",
            (),
            {"experiments_dir": tmp_path / "experiments"},
        )(),
        api_key="key",
    )

    task_result_cls = TaskResult
    baseline = runner.ExperimentRecord.initialize(
        experiment_id="baseline",
        git_commit_hash="bca",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-det"],
        started_at="2026-04-10T00:00:00+00:00",
        expected_trial_count=3,
    )
    for _ in range(2):
        baseline.record_task_result(
            _task_result(task_result_cls, task_name="train-det", reward=1.0)
        )
    baseline.train_task_results["train-det"].expected_trial_count = 2

    call_log: list[str] = []
    counter = 0

    async def fake_run_task(*, task_name, **_kwargs):
        # Stagger completions so the majority watcher gets a turn between
        # trial completions and can cancel pending siblings under the
        # parallel trial scheduler.
        nonlocal counter
        counter += 1
        await asyncio.sleep(counter * 0.001)
        call_log.append(task_name)

        # All trials fail to simulate a real regression. Trial 1 fails →
        # confirm-on-fail expands to k=3; trial 2 fails → majority decided
        # False after 2 trials (via #3 early-stop).
        return TaskResult(
            task_name=task_name,
            reward=0.0,
            solved=False,
            error=None,
            steps_used=1,
            trial_dir=f"/tmp/{task_name}/{len(call_log)}",
            trace_path=None,
            metrics_path=None,
            verifier_stdout_path=None,
            metrics=TaskMetrics(),
            started_at="2026-05-15T00:00:00+00:00",
            finished_at="2026-05-15T00:00:01+00:00",
        )

    monkeypatch.setattr(runner, "run_task", fake_run_task)
    monkeypatch.setattr(experiment_runner, "_make_llm", lambda: object(), raising=False)
    monkeypatch.setattr(
        experiment_runner,
        "_make_env",
        lambda task_name, *, task_dir: object(),
        raising=False,
    )
    _set_task_dirs(experiment_runner, tmp_path, "train-det")
    monkeypatch.setattr(experiment_runner, "_conclude_experiment", lambda: None)

    import asyncio

    asyncio.run(
        experiment_runner._run_experiment(
            baseline=baseline,
        )
    )

    assert call_log == ["train-det", "train-det"]
    trials = experiment_runner.record._task_trials("train-det")
    assert trials.expected_trial_count == 2
    assert trials.trial_count == 2
    assert trials.majority_solved is False


def test_apply_baseline_derived_trial_counts_skips_non_deterministic(
    monkeypatch, tmp_path
):
    runner = _load_experiment_runner()
    harness_config_cls = sys.modules["src.harness.config"].HarnessConfig

    monkeypatch.setattr(runner.control_repo, "require_clean_worktree", lambda: None)
    monkeypatch.setattr(runner.control_repo, "get_head_commit", lambda: "abc")

    experiment_runner = runner.ExperimentRunner(
        harness_config=harness_config_cls(
            experiment_id="exp-mixed",
            focus_name="focus",
            train_task_names=["task-a", "task-b", "task-c"],
            task_trials=3,
            max_concurrency=2,
        ),
        harbor_config=type(
            "HarborConfig",
            (),
            {"experiments_dir": tmp_path / "experiments"},
        )(),
        api_key="key",
    )

    task_result_cls = TaskResult
    baseline = runner.ExperimentRecord.initialize(
        experiment_id="baseline",
        git_commit_hash="bca",
        parent_baseline_experiment_id=None,
        train_task_ids=["task-a", "task-b", "task-c"],
        started_at="2026-04-10T00:00:00+00:00",
        expected_trial_count=3,
    )
    # task-a: all-pass deterministic at baseline.
    for _ in range(3):
        baseline.record_task_result(
            _task_result(task_result_cls, task_name="task-a", reward=1.0)
        )
    # task-b: mixed (1 pass, 2 fail) — not deterministic.
    baseline.record_task_result(
        _task_result(task_result_cls, task_name="task-b", reward=1.0)
    )
    for _ in range(2):
        baseline.record_task_result(
            _task_result(task_result_cls, task_name="task-b", reward=0.0)
        )
    # task-c: all-pass deterministic.
    for _ in range(2):
        baseline.record_task_result(
            _task_result(task_result_cls, task_name="task-c", reward=1.0)
        )
    baseline.train_task_results["task-c"].expected_trial_count = 2

    experiment_runner._apply_baseline_derived_trial_counts(baseline)

    assert experiment_runner.record._task_trials("task-a").expected_trial_count == 1
    assert experiment_runner.record._task_trials("task-b").expected_trial_count == 3
    assert experiment_runner.record._task_trials("task-c").expected_trial_count == 1


def test_run_experiment_runs_train_when_baseline_absent(monkeypatch, tmp_path):
    runner = _load_experiment_runner()
    harness_config_cls = sys.modules["src.harness.config"].HarnessConfig

    monkeypatch.setattr(runner.control_repo, "require_clean_worktree", lambda: None)
    monkeypatch.setattr(runner.control_repo, "get_head_commit", lambda: "abc")

    experiment_runner = runner.ExperimentRunner(
        harness_config=harness_config_cls(
            experiment_id="exp-no-baseline",
            focus_name="focus",
            train_task_names=["train-a"],
            task_trials=3,
            max_concurrency=2,
        ),
        harbor_config=type(
            "HarborConfig",
            (),
            {"experiments_dir": tmp_path / "experiments"},
        )(),
        api_key="key",
    )

    call_log: list[str] = []
    counter = 0

    async def fake_run_task(*, task_name, **_kwargs):
        # Stagger completions so the majority watcher gets a turn between
        # trial completions and can cancel pending siblings under the
        # parallel trial scheduler.
        nonlocal counter
        counter += 1
        await asyncio.sleep(counter * 0.001)
        call_log.append(task_name)
        return TaskResult(
            task_name=task_name,
            reward=1.0,
            solved=True,
            error=None,
            steps_used=1,
            trial_dir=f"/tmp/{task_name}/{len(call_log)}",
            trace_path=None,
            metrics_path=None,
            verifier_stdout_path=None,
            metrics=TaskMetrics(),
            started_at="2026-05-15T00:00:00+00:00",
            finished_at="2026-05-15T00:00:01+00:00",
        )

    monkeypatch.setattr(runner, "run_task", fake_run_task)
    monkeypatch.setattr(experiment_runner, "_make_llm", lambda: object(), raising=False)
    monkeypatch.setattr(
        experiment_runner,
        "_make_env",
        lambda task_name, *, task_dir: object(),
        raising=False,
    )
    _set_task_dirs(experiment_runner, tmp_path, "train-a")
    monkeypatch.setattr(experiment_runner, "_conclude_experiment", lambda: None)

    import asyncio

    asyncio.run(
        experiment_runner._run_experiment(
            baseline=None,
        )
    )

    # With k=3 all-pass: train-a runs 2 trials then early-stops.
    assert sorted(call_log) == ["train-a"] * 2
    trials = experiment_runner.record._task_trials("train-a")
    assert trials.majority_solved is True
    assert trials.expected_trial_count == 2


def test_validate_setup_contract_rejects_train_panel_drift():
    runner = _load_experiment_runner()

    experiment_runner = runner.ExperimentRunner.__new__(runner.ExperimentRunner)
    state = runner.ExperimentState(active_baseline_experiment_id="baseline")
    baseline = runner.ExperimentRecord.initialize(
        experiment_id="baseline",
        git_commit_hash="abc123",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-a"],
        started_at="2026-04-10T00:00:00+00:00",
    )
    candidate = runner.ExperimentRecord.initialize(
        experiment_id="candidate",
        git_commit_hash="def456",
        parent_baseline_experiment_id="baseline",
        train_task_ids=["train-a", "train-b"],
        started_at="2026-04-10T00:00:00+00:00",
    )

    with pytest.raises(ValueError, match="candidate train panel must match"):
        experiment_runner._validate_setup_contract(
            state=state,
            candidate=candidate,
            baseline=baseline,
        )


def test_experiment_runner_requires_clean_worktree_by_default(monkeypatch):
    runner = _load_experiment_runner()
    harness_config_cls = sys.modules["src.harness.config"].HarnessConfig
    calls: list[str] = []

    monkeypatch.setattr(
        runner.control_repo,
        "require_clean_worktree",
        lambda: calls.append("clean"),
    )
    monkeypatch.setattr(runner.control_repo, "get_head_commit", lambda: "abc123")

    experiment_runner = runner.ExperimentRunner(
        harness_config=harness_config_cls(
            experiment_id="exp-1",
            focus_name="focus",
            train_task_names=["task-a"],
        ),
        harbor_config=type(
            "HarborConfig",
            (),
            {"experiments_dir": Path("/tmp/experiments")},
        )(),
        api_key="test-key",
    )

    assert experiment_runner.record.git_commit_hash == "abc123"
    assert calls == ["clean"]


def test_experiment_runner_allows_dirty_worktree_when_explicitly_disabled(monkeypatch):
    runner = _load_experiment_runner()
    harness_config_cls = sys.modules["src.harness.config"].HarnessConfig

    def _boom() -> None:
        raise AssertionError("dirty-worktree gate should not run")

    monkeypatch.setattr(runner.control_repo, "require_clean_worktree", _boom)
    monkeypatch.setattr(runner.control_repo, "get_head_commit", lambda: "abc123")

    experiment_runner = runner.ExperimentRunner(
        harness_config=harness_config_cls(
            experiment_id="exp-1",
            focus_name="focus",
            train_task_names=["task-a"],
        ),
        harbor_config=type(
            "HarborConfig",
            (),
            {"experiments_dir": Path("/tmp/experiments")},
        )(),
        api_key="test-key",
        require_clean_worktree=False,
    )

    assert experiment_runner.record.git_commit_hash == "abc123"


def test_conclude_experiment_does_not_hard_reset_on_discard(monkeypatch, tmp_path):
    runner = _load_experiment_runner()

    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()
    baseline = runner.ExperimentRecord.initialize(
        experiment_id="baseline",
        git_commit_hash="base123",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-a"],
        started_at="2026-04-10T00:00:00+00:00",
    )
    baseline.record_task_result(
        TaskResult(
            task_name="train-a",
            reward=1.0,
            solved=True,
            steps_used=1,
            error=None,
            trial_dir=None,
            trace_path=None,
            metrics_path=None,
            verifier_stdout_path=None,
            started_at="2026-04-10T00:00:00+00:00",
            finished_at="2026-04-10T00:00:01+00:00",
        )
    )
    baseline.finalize(status="keep")
    baseline.write(root=experiments_root)

    experiment_runner = runner.ExperimentRunner.__new__(runner.ExperimentRunner)
    experiment_runner.experiments_root = experiments_root
    experiment_runner.frozen_baseline_experiment_id = "baseline"
    experiment_runner.state = runner.ExperimentState(
        active_baseline_experiment_id="baseline",
        current_experiment_id="candidate",
        updated_at=None,
    )
    experiment_runner.record = runner.ExperimentRecord.initialize(
        experiment_id="candidate",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        train_task_ids=["train-a"],
        started_at="2026-04-10T00:00:00+00:00",
    )
    experiment_runner.record.record_task_result(
        TaskResult(
            task_name="train-a",
            reward=0.0,
            solved=False,
            steps_used=1,
            error=None,
            trial_dir=None,
            trace_path=None,
            metrics_path=None,
            verifier_stdout_path=None,
            started_at="2026-04-10T00:00:00+00:00",
            finished_at="2026-04-10T00:00:01+00:00",
        )
    )
    experiment_runner.record.finalize(
        status="discard", decision_reason="no train improvement"
    )

    hard_reset_calls: list[str] = []
    update_ref_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        runner.control_repo,
        "hard_reset",
        lambda commit_hash: hard_reset_calls.append(commit_hash),
    )
    monkeypatch.setattr(
        runner.control_repo,
        "update_ref",
        lambda ref_name, commit_hash: update_ref_calls.append((ref_name, commit_hash)),
    )

    experiment_runner._conclude_experiment()

    assert hard_reset_calls == []
    assert update_ref_calls == [
        (runner.failed_experiment_git_ref("candidate"), "candidate123")
    ]


def test_run_baseline_at_head_returns_existing_baseline_when_unchanged(
    monkeypatch,
    tmp_path,
):
    runner = _load_experiment_runner()
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()
    _write_baseline_record(runner, experiments_root)
    runner.ExperimentState(
        active_baseline_experiment_id="baseline",
        current_experiment_id=None,
    ).save(root=experiments_root)

    monkeypatch.setattr(
        runner.control_repo,
        "require_clean_worktree",
        lambda cwd=None: None,
    )
    monkeypatch.setattr(
        runner.control_repo,
        "get_head_commit",
        lambda cwd=None: "base123",
    )

    baseline = runner.ExperimentRunner.run_baseline_at_head(
        harness_config=SimpleNamespace(train_task_names=["train-a"]),
        harbor_config=SimpleNamespace(experiments_dir=experiments_root),
        api_key="key",
    )

    state = runner.ExperimentState.load(root=experiments_root)
    assert baseline.experiment_id == "baseline"
    assert state.current_experiment_id == "baseline"
    assert state.active_baseline_experiment_id == "baseline"


def test_run_baseline_at_head_runs_full_current_panel(monkeypatch, tmp_path):
    runner = _load_experiment_runner()
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()
    _write_baseline_record(runner, experiments_root)
    runner.ExperimentState(
        active_baseline_experiment_id="baseline",
        current_experiment_id="baseline",
    ).save(root=experiments_root)

    monkeypatch.setattr(
        runner.control_repo,
        "require_clean_worktree",
        lambda cwd=None: None,
    )
    monkeypatch.setattr(
        runner.control_repo,
        "get_head_commit",
        lambda cwd=None: "head456",
    )
    _stub_baseline_task_environment(monkeypatch, tmp_path)
    panel_calls: list[list[str]] = []

    async def fake_run_panel(*, record, experiments_root, task_names, **_kwargs):
        panel_calls.append(list(task_names))
        for task_id in task_names:
            record.record_task_result(
                TaskResult(
                    task_name=task_id,
                    reward=0.0,
                    solved=False,
                    error=None,
                    steps_used=1,
                    started_at="2026-04-11T00:00:00+00:00",
                    finished_at="2026-04-11T00:00:01+00:00",
                )
            )
        record.write(root=experiments_root)

    monkeypatch.setattr(runner, "_run_panel", fake_run_panel)

    record = runner.ExperimentRunner.run_baseline_at_head(
        harness_config=SimpleNamespace(
            focus_name="new-focus",
            train_task_names=["train-a", "train-b"],
            task_trials=1,
            llm_provider_config=None,
            max_concurrency=1,
            max_steps=30,
            task_timeout_sec=600.0,
            max_output_retries=2,
        ),
        harbor_config=_FakeHarborConfig(experiments_dir=experiments_root),
        api_key="key",
        experiment_id="baseline_rerun",
        started_at="2026-04-11T00:00:00+00:00",
    )

    assert panel_calls == [["train-a", "train-b"]]
    assert record.status == "keep"
    assert record.decision_reason == "baseline rerun"
    assert record.parent_baseline_experiment_id == "baseline"
    assert record.git_commit_hash == "head456"
    assert record.train_task_ids == ["train-a", "train-b"]
    state = runner.ExperimentState.load(root=experiments_root)
    assert state.active_baseline_experiment_id == "baseline_rerun"


def test_run_baseline_at_head_seeds_when_no_active_baseline(monkeypatch, tmp_path):
    runner = _load_experiment_runner()
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()
    runner.ExperimentState(active_baseline_experiment_id=None).save(
        root=experiments_root
    )

    monkeypatch.setattr(
        runner.control_repo,
        "require_clean_worktree",
        lambda cwd=None: None,
    )
    monkeypatch.setattr(
        runner.control_repo,
        "get_head_commit",
        lambda cwd=None: "head000",
    )
    _stub_baseline_task_environment(monkeypatch, tmp_path)

    async def fake_run_panel(*, record, experiments_root, task_names, **_kwargs):
        for task_id in task_names:
            record.record_task_result(
                TaskResult(
                    task_name=task_id,
                    reward=1.0,
                    solved=True,
                    error=None,
                    steps_used=1,
                    started_at="2026-04-11T00:00:00+00:00",
                    finished_at="2026-04-11T00:00:01+00:00",
                )
            )
        record.write(root=experiments_root)

    monkeypatch.setattr(runner, "_run_panel", fake_run_panel)

    record = runner.ExperimentRunner.run_baseline_at_head(
        harness_config=SimpleNamespace(
            focus_name="seed",
            train_task_names=["train-a"],
            task_trials=1,
            llm_provider_config=None,
            max_concurrency=1,
            max_steps=30,
            task_timeout_sec=600.0,
            max_output_retries=2,
        ),
        harbor_config=_FakeHarborConfig(experiments_dir=experiments_root),
        api_key="key",
        decision_reason="baseline seed",
        experiment_id="baseline_seed",
        started_at="2026-04-11T00:00:00+00:00",
    )

    assert record.parent_baseline_experiment_id is None
    assert record.status == "keep"
    assert record.decision_reason == "baseline seed"
    assert (
        runner.ExperimentState.load(root=experiments_root).active_baseline_experiment_id
        == "baseline_seed"
    )


@pytest.mark.parametrize(
    "trial_error",
    [
        pytest.param("environment reset/bootstrap timed out", id="with-message"),
        # An empty-string error still marks a `crash`. The `error is None`
        # contract is load-bearing: a truthiness check on `trial.error` would
        # misread "" as valid evidence.
        pytest.param("", id="empty-string-crash"),
    ],
)
def test_run_baseline_at_head_crashes_when_no_valid_evidence(
    trial_error, monkeypatch, tmp_path
):
    """When every baseline trial is a `crash`, the run produced no valid
    evidence: it must finalize as `crash` and must NOT update
    `active_baseline_experiment_id`. Promoting an empty baseline would make
    every later candidate compare against an all-zero pool."""
    runner = _load_experiment_runner()
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()
    runner.ExperimentState(active_baseline_experiment_id=None).save(
        root=experiments_root
    )

    monkeypatch.setattr(
        runner.control_repo,
        "require_clean_worktree",
        lambda cwd=None: None,
    )
    monkeypatch.setattr(
        runner.control_repo,
        "get_head_commit",
        lambda cwd=None: "head000",
    )
    _stub_baseline_task_environment(monkeypatch, tmp_path)

    async def fake_run_panel(*, record, experiments_root, task_names, **_kwargs):
        for task_id in task_names:
            record.record_task_result(
                TaskResult(
                    task_name=task_id,
                    reward=0.0,
                    solved=False,
                    error=trial_error,
                    steps_used=42,
                    started_at="2026-04-11T00:00:00+00:00",
                    finished_at="2026-04-11T00:00:01+00:00",
                )
            )
        record.write(root=experiments_root)

    monkeypatch.setattr(runner, "_run_panel", fake_run_panel)

    record = runner.ExperimentRunner.run_baseline_at_head(
        harness_config=SimpleNamespace(
            focus_name="seed",
            train_task_names=["train-a"],
            task_trials=1,
            llm_provider_config=None,
            max_concurrency=1,
            max_steps=30,
            task_timeout_sec=600.0,
            max_output_retries=2,
        ),
        harbor_config=_FakeHarborConfig(experiments_dir=experiments_root),
        api_key="key",
        decision_reason="baseline seed",
        experiment_id="baseline_errored",
        started_at="2026-04-11T00:00:00+00:00",
    )

    assert record.status == "crash"
    assert "no valid trials" in record.error
    assert (
        runner.ExperimentState.load(root=experiments_root).active_baseline_experiment_id
        is None
    )


@pytest.mark.parametrize(
    "trial_error",
    [
        pytest.param("environment reset/bootstrap timed out", id="with-message"),
        pytest.param("", id="empty-string-crash"),
    ],
)
def test_run_experiment_finalizes_crash_when_no_valid_evidence(
    trial_error, monkeypatch, tmp_path
):
    """When the only task's only trial is a `crash`, the run produced no valid
    evidence and must finalize as `crash`, never `keep`/`discard` — there is
    nothing for the gate to compare. (An isolated crash alongside valid trials
    is instead excluded and tolerated; see the mixed-evidence test.)"""
    import asyncio

    runner = _load_experiment_runner()
    harness_config_cls = sys.modules["src.harness.config"].HarnessConfig
    experiment_id = "exp-crash-on-trial-error"

    monkeypatch.setattr(runner.control_repo, "require_clean_worktree", lambda: None)
    monkeypatch.setattr(runner.control_repo, "get_head_commit", lambda: "abc")
    monkeypatch.setattr(
        runner.control_repo,
        "update_ref",
        lambda ref_name, commit_hash, *, cwd=None: None,
    )

    experiment_runner = runner.ExperimentRunner(
        harness_config=harness_config_cls(
            experiment_id=experiment_id,
            focus_name="focus",
            train_task_names=["task-a"],
            task_trials=1,
            max_concurrency=1,
        ),
        harbor_config=type(
            "HarborConfig",
            (),
            {"experiments_dir": tmp_path / "experiments"},
        )(),
        api_key="key",
    )
    experiment_runner.experiment_dir.mkdir(parents=True, exist_ok=True)

    async def fake_run_task(*, task_name, **_kwargs):
        return TaskResult(
            task_name=task_name,
            reward=0.0,
            solved=False,
            error=trial_error,
            steps_used=12,
            trial_dir=None,
            trace_path=None,
            metrics_path=None,
            verifier_stdout_path=None,
            metrics=TaskMetrics(),
            started_at="2026-05-05T00:00:00+00:00",
            finished_at="2026-05-05T00:00:01+00:00",
        )

    monkeypatch.setattr(runner, "run_task", fake_run_task)
    monkeypatch.setattr(experiment_runner, "_make_llm", lambda: object(), raising=False)
    monkeypatch.setattr(
        experiment_runner,
        "_make_env",
        lambda task_name, *, task_dir: object(),
        raising=False,
    )
    _set_task_dirs(experiment_runner, tmp_path, "task-a")

    asyncio.run(experiment_runner._run_experiment(baseline=None))

    record = experiment_runner.record
    assert record.status == "crash"
    assert "no valid trials" in record.error


def test_run_experiment_excludes_crash_trial_but_scores_valid_ones(
    monkeypatch, tmp_path
):
    """A lone `crash` trial alongside valid trials must not sink the run: it is
    dropped from the gate's evidence and the surviving valid trials are scored.
    Here trial 1 crashes and trials 2-3 solve, so the frontier task
    majority-solves on 2 valid trials and the run is kept."""
    import asyncio

    runner = _load_experiment_runner()
    harness_config_cls = sys.modules["src.harness.config"].HarnessConfig
    experiment_id = "exp-mixed-evidence"

    monkeypatch.setattr(runner.control_repo, "require_clean_worktree", lambda: None)
    monkeypatch.setattr(runner.control_repo, "get_head_commit", lambda: "abc")
    monkeypatch.setattr(
        runner.control_repo,
        "update_ref",
        lambda ref_name, commit_hash, *, cwd=None: None,
    )

    experiment_runner = runner.ExperimentRunner(
        harness_config=harness_config_cls(
            experiment_id=experiment_id,
            focus_name="focus",
            train_task_names=["task-a"],
            task_trials=3,
            max_concurrency=1,
        ),
        harbor_config=type(
            "HarborConfig",
            (),
            {"experiments_dir": tmp_path / "experiments"},
        )(),
        api_key="key",
    )
    experiment_runner.experiment_dir.mkdir(parents=True, exist_ok=True)

    call_count = 0

    async def fake_run_task(*, task_name, **_kwargs):
        # max_concurrency=1 serializes trials, so the counter is deterministic.
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return TaskResult(
                task_name=task_name,
                reward=0.0,
                solved=False,
                error="environment reset/bootstrap timed out",
                steps_used=12,
                metrics=TaskMetrics(failure_mode="crash"),
                started_at="2026-05-05T00:00:00+00:00",
                finished_at="2026-05-05T00:00:01+00:00",
            )
        return TaskResult(
            task_name=task_name,
            reward=1.0,
            solved=True,
            error=None,
            steps_used=5,
            metrics=TaskMetrics(failure_mode="solved"),
            started_at="2026-05-05T00:00:00+00:00",
            finished_at="2026-05-05T00:00:01+00:00",
        )

    monkeypatch.setattr(runner, "run_task", fake_run_task)
    monkeypatch.setattr(experiment_runner, "_make_llm", lambda: object(), raising=False)
    monkeypatch.setattr(
        experiment_runner,
        "_make_env",
        lambda task_name, *, task_dir: object(),
        raising=False,
    )
    _set_task_dirs(experiment_runner, tmp_path, "task-a")

    asyncio.run(experiment_runner._run_experiment(baseline=None))

    record = experiment_runner.record
    trials = record.train_task_results["task-a"]
    # All three slots ran; only the two non-crash trials count as evidence.
    assert len(trials.finished_trials) == 3
    assert len(trials.valid_trials) == 2
    assert trials.solved_count == 2
    # 2/2 valid trials solved on the frontier ⇒ improvement ⇒ keep.
    assert record.status == "keep"
    assert record.decision_reason == "train task task-a improved"


def test_run_experiment_discards_task_timeout_without_crashing(monkeypatch, tmp_path):
    """A task episode timeout is a measured unsolved trial, not an
    experiment-level infrastructure crash."""
    import asyncio

    runner = _load_experiment_runner()
    harness_config_cls = sys.modules["src.harness.config"].HarnessConfig
    experiment_id = "exp-timeout-is-unsolved"

    monkeypatch.setattr(runner.control_repo, "require_clean_worktree", lambda: None)
    monkeypatch.setattr(runner.control_repo, "get_head_commit", lambda: "abc")
    monkeypatch.setattr(
        runner.control_repo,
        "update_ref",
        lambda ref_name, commit_hash, *, cwd=None: None,
    )

    experiment_runner = runner.ExperimentRunner(
        harness_config=harness_config_cls(
            experiment_id=experiment_id,
            focus_name="focus",
            train_task_names=["task-a"],
            task_trials=1,
            max_concurrency=1,
        ),
        harbor_config=type(
            "HarborConfig",
            (),
            {"experiments_dir": tmp_path / "experiments"},
        )(),
        api_key="key",
    )
    experiment_runner.experiment_dir.mkdir(parents=True, exist_ok=True)

    async def fake_run_task(*, task_name, **_kwargs):
        return TaskResult(
            task_name=task_name,
            reward=0.0,
            solved=False,
            error=None,
            steps_used=12,
            metrics=TaskMetrics(failure_mode="hit_timeout"),
            started_at="2026-05-05T00:00:00+00:00",
            finished_at="2026-05-05T00:00:01+00:00",
        )

    monkeypatch.setattr(runner, "run_task", fake_run_task)
    monkeypatch.setattr(experiment_runner, "_make_llm", lambda: object(), raising=False)
    monkeypatch.setattr(
        experiment_runner,
        "_make_env",
        lambda task_name, *, task_dir: object(),
        raising=False,
    )
    _set_task_dirs(experiment_runner, tmp_path, "task-a")

    asyncio.run(experiment_runner._run_experiment(baseline=None))

    record = experiment_runner.record
    assert record.status == "discard"
    assert record.error == ""
    assert record.decision_reason == "no train task improvement reached significance"


@pytest.mark.parametrize("trial_mode", ["result", "exception"])
def test_run_panel_persists_completed_trial_without_refreshing_evidence(
    trial_mode, monkeypatch, tmp_path
):
    runner = _load_experiment_runner()
    harness_config_cls = sys.modules["src.harness.config"].HarnessConfig
    experiment_id = f"exp-g4-{trial_mode}"

    monkeypatch.setattr(runner.control_repo, "require_clean_worktree", lambda: None)
    monkeypatch.setattr(runner.control_repo, "get_head_commit", lambda: "abc")

    experiment_runner = runner.ExperimentRunner(
        harness_config=harness_config_cls(
            experiment_id=experiment_id,
            focus_name="focus",
            train_task_names=["task-a"],
            task_trials=1,
            max_concurrency=1,
        ),
        harbor_config=type(
            "HarborConfig",
            (),
            {"experiments_dir": tmp_path / "experiments"},
        )(),
        api_key="key",
    )

    async def fake_run_task(*, task_name, **_kwargs):
        if trial_mode == "exception":
            raise RuntimeError("simulated env reset crash")
        return TaskResult(
            task_name=task_name,
            reward=1.0,
            solved=True,
            error=None,
            steps_used=1,
            trial_dir=None,
            trace_path=None,
            metrics_path=None,
            verifier_stdout_path=None,
            metrics=TaskMetrics(),
            started_at="2026-05-05T00:00:00+00:00",
            finished_at="2026-05-05T00:00:01+00:00",
        )

    monkeypatch.setattr(runner, "run_task", fake_run_task)
    monkeypatch.setattr(experiment_runner, "_make_llm", lambda: object(), raising=False)
    monkeypatch.setattr(
        experiment_runner,
        "_make_env",
        lambda task_name, *, task_dir: object(),
        raising=False,
    )
    _set_task_dirs(experiment_runner, tmp_path, "task-a")

    _run_panel_for_experiment_runner(runner, experiment_runner, ["task-a"])

    record_path = tmp_path / "experiments" / experiment_id / "experiment.json"
    payload = json.loads(record_path.read_text())
    trials = payload["train_task_results"]["task-a"]["trials"]
    assert len(trials) == 1
    if trial_mode == "exception":
        assert trials[0]["solved"] is False
        assert trials[0]["error"] == "simulated env reset crash"
        assert trials[0]["metrics"]["failure_mode"] == "crash"
    else:
        assert trials[0]["solved"] is True
        assert trials[0]["error"] is None
    outcomes = (payload.get("evidence") or {}).get("task_outcomes") or []
    assert outcomes == []


def _pool_record(
    runner,
    *,
    experiment_id: str,
    parent_baseline_id: str | None,
    train_task_ids: list[str],
    finished_at: str = "2026-05-10T00:00:00+00:00",
    status: str = "discard",
    decision_reason: str = "no train improvement",
):
    record = runner.ExperimentRecord.initialize(
        experiment_id=experiment_id,
        git_commit_hash="cand-" + experiment_id,
        parent_baseline_experiment_id=parent_baseline_id,
        train_task_ids=train_task_ids,
        started_at="2026-05-10T00:00:00+00:00",
        expected_trial_count=1,
    )
    record.status = status
    record.decision_reason = decision_reason
    record.finished_at = finished_at
    return record


def _pool_trial(
    task_name: str, *, solved: bool, rule_fires: dict[str, int] | None = None
) -> TaskResult:
    return TaskResult(
        task_name=task_name,
        reward=1.0 if solved else 0.0,
        solved=solved,
        error=None,
        steps_used=1,
        started_at="2026-05-10T00:00:00+00:00",
        finished_at="2026-05-10T00:00:01+00:00",
        metrics=TaskMetrics(rule_fires=dict(rule_fires or {})),
    )


def _populate_pool_trials(record, trials_by_task: dict[str, list[TaskResult]]) -> None:
    for trials in trials_by_task.values():
        for trial in trials:
            record.record_task_result(trial)


def test_rule_names_from_added_lines_matches_three_declaration_styles() -> None:
    runner = _load_experiment_runner()
    added = [
        'ARGUMENT_NORMALIZERS = (ArgumentRule(name="unwrap_input", order=10, apply=fn),)',
        'VERIFY_NUDGE_RULE_NAME = "verify_absent_nudge"',
        '    recorder.metrics.record_rule_fire("direct_literal_rule")',
    ]
    assert runner.rule_names_from_added_lines(added) == {
        "unwrap_input",
        "verify_absent_nudge",
        "direct_literal_rule",
    }


def test_rule_names_from_added_lines_handles_pep8_wrapped_constant() -> None:
    runner = _load_experiment_runner()
    added = [
        "SYSTEM_PROMPT_PRE_VERIFY_VALIDATION_RULE_NAME = (",
        '    "system_prompt_pre_verify_validation"',
        ")",
    ]
    assert runner.rule_names_from_added_lines(added) == {
        "system_prompt_pre_verify_validation",
    }


def test_rule_names_from_added_lines_ignores_plain_string_literals() -> None:
    runner = _load_experiment_runner()
    added = [
        '    """docstring describing a feature without a rule name"""',
        '    return f"trial {task_id}: solved"',
        "    if final_passed is True:",
    ]
    assert runner.rule_names_from_added_lines(added) == set()


def _pool_trial_from_spec(task_name: str, spec) -> TaskResult:
    if isinstance(spec, bool):
        return _pool_trial(task_name, solved=spec)
    return TaskResult(
        task_name=task_name,
        reward=1.0 if spec["solved"] else 0.0,
        solved=spec["solved"],
        error=spec.get("error"),
        steps_used=spec.get("steps_used", 1),
        started_at="2026-05-10T00:00:00+00:00",
        finished_at="2026-05-10T00:00:01+00:00",
        metrics=TaskMetrics(
            rule_fires=dict(spec.get("rule_fires") or {}),
            failure_mode=spec.get("failure_mode"),
        ),
    )


def _pool_record_from_spec(runner, spec):
    record = _pool_record(
        runner,
        experiment_id=spec["experiment_id"],
        parent_baseline_id=spec.get("parent_baseline_id"),
        train_task_ids=spec.get("train_task_ids", ["t"]),
        status=spec.get("status", "discard"),
        decision_reason=spec.get("decision_reason", "no train improvement"),
    )
    _populate_pool_trials(
        record,
        {
            task_name: [
                _pool_trial_from_spec(task_name, trial_spec)
                for trial_spec in trial_specs
            ]
            for task_name, trial_specs in spec.get("trials", {}).items()
        },
    )
    return record


@pytest.mark.parametrize(
    "pool_case",
    [
        pytest.param(
            {
                "baseline": {
                    "experiment_id": "baseline",
                    "status": "keep",
                    "decision_reason": "baseline seed",
                    "trials": {"t": [True, True, False]},
                },
                "expected": {"t": (2, 3)},
            },
            id="includes-baseline-trials",
        ),
        pytest.param(
            {
                "baseline": {
                    "experiment_id": "baseline",
                    "status": "keep",
                    "decision_reason": "baseline seed",
                    "trials": {"t": [True]},
                },
                "recent": [
                    {
                        "experiment_id": "baseline-rerun",
                        "parent_baseline_id": "baseline",
                        "status": "keep",
                        "decision_reason": "baseline rerun",
                        "trials": {"t": [True, True]},
                    }
                ],
                "rules": {"baseline-rerun": ()},
                "expected": {"t": (1, 1)},
            },
            id="excludes-baseline-bookkeeping",
        ),
        pytest.param(
            {
                "baseline": {
                    "experiment_id": "baseline",
                    "status": "keep",
                    "decision_reason": "baseline seed",
                    "trials": {"t": [True]},
                },
                "recent": [
                    {
                        "experiment_id": "cand-1",
                        "parent_baseline_id": "baseline",
                        "trials": {
                            "t": [
                                {"solved": True, "rule_fires": {"new_rule": 5}},
                                True,
                                False,
                            ]
                        },
                    }
                ],
                "rules": {"cand-1": ("new_rule",)},
                "expected": {"t": (2, 3)},
            },
            id="excludes-mechanism-touched-trials",
        ),
        pytest.param(
            {
                "baseline": {
                    "experiment_id": "baseline",
                    "parent_baseline_id": "prior-baseline",
                    "status": "keep",
                    "trials": {"t": [True, False]},
                },
                "include_active_baseline_as_recent": True,
                "rules": {"baseline": ()},
                "expected": {"t": (1, 2)},
            },
            id="does-not-double-count-active-baseline",
        ),
        pytest.param(
            {
                "baseline": {
                    "experiment_id": "baseline",
                    "status": "keep",
                    "decision_reason": "baseline seed",
                    "trials": {"t": [True]},
                },
                "recent": [
                    {
                        "experiment_id": "crashed",
                        "parent_baseline_id": "baseline",
                        "status": "crash",
                        "decision_reason": "",
                        "trials": {"t": [True, True]},
                    }
                ],
                "rules": {"crashed": ()},
                "expected": {"t": (1, 1)},
            },
            id="excludes-crashed-records",
        ),
        pytest.param(
            {
                "baseline": {
                    "experiment_id": "baseline",
                    "status": "keep",
                    "decision_reason": "baseline seed",
                    "trials": {
                        "t": [
                            {"solved": True, "steps_used": 19},
                            {
                                "solved": False,
                                "failure_mode": "hit_timeout",
                                "steps_used": 13,
                            },
                            {
                                "solved": False,
                                "error": "environment reset/bootstrap timed out",
                                "steps_used": 0,
                            },
                        ]
                    },
                },
                "expected": {"t": (1, 2)},
            },
            id="includes-task-timeouts-excludes-infrastructure-errors",
        ),
        pytest.param(
            {
                # `run_task` produces `error=""` from bare `RuntimeError()`
                # (`str(exc)` is the empty string). A truthiness check on
                # `trial.error` would silently let this slip into the pool.
                "baseline": {
                    "experiment_id": "baseline",
                    "status": "keep",
                    "decision_reason": "baseline seed",
                    "trials": {
                        "t": [
                            True,
                            {"solved": False, "error": "", "steps_used": 30},
                        ]
                    },
                },
                "expected": {"t": (1, 1)},
            },
            id="excludes-trial-with-empty-string-error",
        ),
        pytest.param(
            {
                "baseline": {
                    "experiment_id": "baseline",
                    "status": "keep",
                    "decision_reason": "baseline seed",
                    "trials": {"t": [True]},
                },
                "recent": [
                    {
                        "experiment_id": "cand",
                        "parent_baseline_id": "baseline",
                        "decision_reason": "train regressed",
                        "trials": {
                            "t": [
                                False,
                                {
                                    "solved": False,
                                    "error": "abandoned after supervisor restart",
                                    "steps_used": 0,
                                },
                            ]
                        },
                    }
                ],
                "rules": {"cand": ()},
                "expected": {"t": (1, 2)},
            },
            id="excludes-trials-with-error-marker",
        ),
        pytest.param(
            {
                "baseline": {
                    "experiment_id": "baseline",
                    "status": "keep",
                    "decision_reason": "baseline seed",
                    "trials": {"t": [True]},
                },
                "recent": [
                    {
                        "experiment_id": "cand-1",
                        "parent_baseline_id": "baseline",
                        "trials": {"t": [False]},
                    }
                ],
                "rules": {"cand-1": ()},
                "task_ids": ["t", "unseen-task"],
                "expected": {"t": (1, 2), "unseen-task": (0, 0)},
            },
            id="handles-missing-task",
        ),
    ],
)
def test_build_pooled_control_samples_filters_records_and_trials(pool_case):
    runner = _load_experiment_runner()
    baseline = _pool_record_from_spec(runner, pool_case["baseline"])
    recent_records = [
        _pool_record_from_spec(runner, spec) for spec in pool_case.get("recent", [])
    ]
    if pool_case.get("include_active_baseline_as_recent"):
        recent_records.append(baseline)

    pool = runner.build_pooled_control_samples(
        active_baseline=baseline,
        recent_candidates=recent_records,
        candidate_new_rule_names=pool_case.get("rules", {}),
        task_ids=pool_case.get("task_ids", ["t"]),
    )

    assert pool == pool_case["expected"]


class _FakeStream:
    def __init__(self, *, is_tty: bool) -> None:
        self._is_tty = is_tty
        self.buffer: list[str] = []

    def isatty(self) -> bool:
        return self._is_tty

    def write(self, text: str) -> int:
        self.buffer.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    @property
    def text(self) -> str:
        return "".join(self.buffer)


def test_format_panel_progress_anchors_on_tasks_and_computes_eta():
    runner = _load_experiment_runner()
    line = runner.format_panel_progress(
        tasks_done=25,
        total_tasks=100,
        trials_done=80,
        trials_planned=320,
        solved=17,
        decided=24,
        crash_trials=6,
        in_flight=10,
        elapsed_sec=3600,
    )
    assert "25/100 tasks (25%)" in line
    assert "trials 80/320" in line
    assert "solved 17/24" in line
    assert "crash 6" in line
    assert "run 10" in line
    # 25 tasks in 1h -> 75 remaining at 25/h -> 3h elapsed-equivalent left.
    assert "1h00m elapsed" in line
    assert "~3h00m left" in line
    # Bar fill is proportional to task completion.
    assert line.count("#") == int(0.25 * runner.PROGRESS_BAR_WIDTH)


def test_format_panel_progress_unknown_eta_before_first_task():
    runner = _load_experiment_runner()
    line = runner.format_panel_progress(
        tasks_done=0,
        total_tasks=10,
        trials_done=3,
        trials_planned=50,
        solved=0,
        decided=0,
        crash_trials=0,
        in_flight=10,
        elapsed_sec=42,
    )
    assert "0/10 tasks (0%)" in line
    assert "~--m left" in line
    assert "#" not in line


def test_panel_progress_reporter_silent_on_non_tty():
    task_result_cls = _install_harness_stubs()
    runner = _load_experiment_runner()
    record = runner.ExperimentRecord.initialize(
        experiment_id="exp-progress",
        git_commit_hash="abc123",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-a"],
        started_at="2026-04-10T00:00:00+00:00",
    )
    record.record_task_result(
        _task_result(task_result_cls, task_name="train-a", reward=1.0)
    )
    stream = _FakeStream(is_tty=False)
    reporter = runner.PanelProgressReporter(
        total_tasks=1, max_concurrency=10, stream=stream
    )
    reporter.render(record)
    reporter.close()
    assert stream.text == ""


def test_panel_progress_reporter_draws_and_finalizes_on_tty():
    task_result_cls = _install_harness_stubs()
    runner = _load_experiment_runner()
    record = runner.ExperimentRecord.initialize(
        experiment_id="exp-progress",
        git_commit_hash="abc123",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-a", "train-b"],
        started_at="2026-04-10T00:00:00+00:00",
    )
    stream = _FakeStream(is_tty=True)
    reporter = runner.PanelProgressReporter(
        total_tasks=2, max_concurrency=10, stream=stream
    )

    record.record_task_result(
        _task_result(task_result_cls, task_name="train-a", reward=1.0)
    )
    reporter.render(record)
    assert "1/2 tasks" in stream.text
    assert "\r\033[K" in stream.text
    assert not stream.text.endswith("\n")  # bar is still live

    # Completing the panel finalizes the line with a trailing newline.
    record.record_task_result(
        _task_result(task_result_cls, task_name="train-b", reward=0.0, solved=False)
    )
    reporter.render(record)
    assert "2/2 tasks (100%)" in stream.text
    assert stream.text.endswith("\n")

    # Further renders are no-ops once finalized.
    before = stream.text
    reporter.render(record)
    assert stream.text == before


def test_panel_progress_reporter_close_terminates_dangling_line():
    task_result_cls = _install_harness_stubs()
    runner = _load_experiment_runner()
    record = runner.ExperimentRecord.initialize(
        experiment_id="exp-progress",
        git_commit_hash="abc123",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-a", "train-b"],
        started_at="2026-04-10T00:00:00+00:00",
    )
    stream = _FakeStream(is_tty=True)
    reporter = runner.PanelProgressReporter(
        total_tasks=2, max_concurrency=10, stream=stream
    )
    record.record_task_result(
        _task_result(task_result_cls, task_name="train-a", reward=1.0)
    )
    reporter.render(record)  # one task done, panel incomplete -> no newline yet
    assert not stream.text.endswith("\n")
    reporter.close()  # crash/cancel path terminates the line
    assert stream.text.endswith("\n")
