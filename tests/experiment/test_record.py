from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.experiment.gate import build_gate_verdicts
from src.experiment.record import ExperimentRecord, TaskTrials
from src.harness.contracts import TaskResult


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
    *,
    task_name: str,
    reward: float | None,
    solved: bool | None = None,
    error: str | None = None,
) -> TaskResult:
    if solved is None:
        solved = error is None and reward is not None and reward > 0.0
    return TaskResult(
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


def test_experiment_record_load_reads_panel_schema(tmp_path):
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

    record = ExperimentRecord.load("exp-panels", root=root)

    assert record.train_task_ids == ["train-a", "train-b"]
    assert record.train_solved_count == 1
    assert sorted(record.train_task_results) == ["train-a", "train-b"]


def test_experiment_record_updates_train_counts():
    record = ExperimentRecord.initialize(
        experiment_id="exp-panels",
        git_commit_hash="abc123",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-a"],
        started_at="2026-04-10T00:00:00+00:00",
    )

    record.record_task_result(_task_result(task_name="train-a", reward=1.0))

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
    record = ExperimentRecord.initialize(
        experiment_id="exp-panels",
        git_commit_hash="abc123",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-a"],
        started_at="2026-04-10T00:00:00+00:00",
    )

    record.record_task_result(
        _task_result(
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
    baseline = ExperimentRecord.initialize(
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

    candidate = ExperimentRecord.initialize(
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
    verdicts = build_gate_verdicts(candidate=candidate, pool=pool)
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
    loaded = ExperimentRecord.load("candidate", root=experiments_root)
    loaded_outcomes = {
        outcome.task_id: outcome for outcome in loaded.evidence.task_outcomes
    }
    assert loaded_outcomes["hard-task"].outcome == "new_solve"


def test_experiment_record_evidence_omits_missing_artifact_paths(tmp_path):
    existing_trial_dir = tmp_path / "trial"
    existing_agent_dir = existing_trial_dir / "agent"
    existing_agent_dir.mkdir(parents=True)
    steps_path = existing_agent_dir / "steps.jsonl"
    metrics_path = existing_agent_dir / "metrics.json"
    exec_log_path = existing_agent_dir / "exec.log"
    for path in (steps_path, metrics_path, exec_log_path):
        path.write_text("{}\n")

    candidate = ExperimentRecord.initialize(
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


def test_task_trials_majority_solved_with_k_trials():
    trials = TaskTrials(task_name="t", expected_trial_count=3)
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
    trials = TaskTrials(task_name="t", expected_trial_count=1)
    in_progress = TaskResult(
        task_name="t",
        reward=None,
        solved=False,
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


def test_task_trials_is_deterministic_solved_predicate():
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

    empty = TaskTrials(task_name="t", expected_trial_count=3)
    assert empty.is_deterministic_solved is False

    one_pass = TaskTrials(
        task_name="t", expected_trial_count=3, trials=[trial(solved=True)]
    )
    assert one_pass.is_deterministic_solved is True

    two_pass = TaskTrials(
        task_name="t",
        expected_trial_count=3,
        trials=[trial(solved=True), trial(solved=True)],
    )
    assert two_pass.is_deterministic_solved is True

    one_fail = TaskTrials(
        task_name="t", expected_trial_count=3, trials=[trial(solved=False)]
    )
    assert one_fail.is_deterministic_solved is False

    mixed = TaskTrials(
        task_name="t",
        expected_trial_count=3,
        trials=[trial(solved=True), trial(solved=False)],
    )
    assert mixed.is_deterministic_solved is False
