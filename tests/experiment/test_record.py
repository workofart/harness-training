from __future__ import annotations

import asyncio
import copy
import json

import pytest
from pydantic import ValidationError

from src.experiment.gate import build_gate_verdicts
from src.experiment.record import (
    ExperimentAbandoned,
    ExperimentRecord,
    PanelRecord,
    TaskTrials,
    terminal_task_result,
)
from src.harness.contracts import TaskResult

from conftest import _task_result, _write_task_artifacts


def train_panel(
    task_ids: list[str],
    *,
    expected_trial_count: int = 1,
    lifecycle: str = "active",
) -> PanelRecord:
    return PanelRecord.initialize(
        panel_id="train",
        purpose="promotion",
        task_ids=task_ids,
        expected_trial_count=expected_trial_count,
        lifecycle=lifecycle,
    )


def init_record(
    *,
    experiment_id: str,
    git_commit_hash: str = "abc123",
    parent_baseline_experiment_id: str | None = None,
    train_task_ids: list[str],
    started_at: str = "2026-04-10T00:00:00+00:00",
) -> ExperimentRecord:
    return ExperimentRecord.initialize(
        experiment_id=experiment_id,
        git_commit_hash=git_commit_hash,
        parent_baseline_experiment_id=parent_baseline_experiment_id,
        panels=[train_panel(train_task_ids)],
        started_at=started_at,
    )


def train_results(record: ExperimentRecord) -> dict[str, TaskTrials]:
    return record.panels["train"].task_results


def train_outcomes(record: ExperimentRecord):
    assert record.evidence is not None
    return record.evidence.panel_outcomes["train"]


def test_experiment_record_load_reads_v2_panel_schema(tmp_path):
    root = tmp_path / "experiments"
    record_dir = root / "exp-panels"
    record_dir.mkdir(parents=True)
    payload = {
        "schema_version": 2,
        "experiment_id": "exp-panels",
        "parent_baseline_experiment_id": None,
        "git_commit_hash": "abc123",
        "focus_name": "action-set",
        "status": "keep",
        "decision_reason": "train improvement",
        "error": "",
        "started_at": "2026-04-10T00:00:00+00:00",
        "finished_at": "2026-04-10T00:00:01+00:00",
        "panel_order": ["train"],
        "panels": {
            "train": {
                "panel_id": "train",
                "purpose": "promotion",
                "lifecycle": "finished",
                "task_ids": ["train-a", "train-b"],
                "task_results": {
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
                "started_at": None,
                "finished_at": None,
                "skip_reason": "",
                "evaluation": None,
            }
        },
        "evidence": {
            "candidate_change": {
                "commit": "abc123",
                "parent_baseline_experiment_id": None,
                "parent_baseline_commit": None,
            },
            "panel_outcomes": {"train": []},
        },
    }
    (record_dir / "experiment.json").write_text(json.dumps(payload))

    record = ExperimentRecord.load("exp-panels", root=root)

    assert record.panels["train"].task_ids == ["train-a", "train-b"]
    assert record.panels["train"].solved_count == 1
    assert sorted(record.panels["train"].task_results) == ["train-a", "train-b"]


def test_experiment_record_rejects_v1_payload():
    with pytest.raises(ValidationError):
        ExperimentRecord.model_validate(
            {
                "experiment_id": "exp",
                "parent_baseline_experiment_id": None,
                "git_commit_hash": "abc123",
                "focus_name": "",
                "train_task_ids": ["train-a"],
                "status": None,
                "train_solved_count": 0,
                "decision_reason": "",
                "error": "",
                "started_at": "2026-04-10T00:00:00+00:00",
                "finished_at": None,
                "train_task_results": {},
                "evidence": None,
            }
        )


def test_experiment_record_rejects_duplicate_panel_order():
    record = init_record(experiment_id="exp", train_task_ids=["train-a"])
    payload = record.model_dump(mode="json")
    payload["panel_order"].append("train")

    with pytest.raises(ValidationError):
        ExperimentRecord.model_validate(payload)


def test_panel_record_rejects_duplicate_task_ids():
    panel = train_panel(["train-a"])
    payload = panel.model_dump(mode="json")
    payload["task_ids"].append("train-a")

    with pytest.raises(ValidationError):
        PanelRecord.model_validate(payload)


def test_load_fails_fast_on_missing_decision_bearing_fields():
    # Fields that drive a decision -- a trial's `solved` (scoring), the trial
    # budget and `trials` list (completion/majority), and the `evidence` block
    # (gate/diagnosis) -- must be present on load. A corrupt record fails fast
    # rather than loading as a plausible-but-wrong trial filled from a
    # constructor default. `metrics` is diagnostic-only and stays optional.
    record = init_record(
        experiment_id="exp",
        git_commit_hash="abc123",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-a"],
        started_at="2026-04-10T00:00:00+00:00",
    )
    record.record_task_result(_task_result(task_name="train-a", reward=1.0))
    record.finalize(status="keep")
    record.refresh_evidence(baseline=None)
    payload = record.model_dump(mode="json")
    ExperimentRecord.model_validate(payload)  # sanity: a complete record loads

    def without(*path: str | int) -> dict:
        corrupt = copy.deepcopy(payload)
        node = corrupt
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]
        return corrupt

    trial_path = ("panels", "train", "task_results", "train-a", "trials", 0)
    for path in (
        ("evidence",),
        ("evidence", "panel_outcomes"),
        ("panels", "train", "task_results", "train-a", "expected_trial_count"),
        ("panels", "train", "task_results", "train-a", "trials"),
        (*trial_path, "solved"),
    ):
        with pytest.raises(ValidationError):
            ExperimentRecord.model_validate(without(*path))

    # `metrics` is the lone diagnostic-only field: its absence still loads.
    ExperimentRecord.model_validate(without(*trial_path, "metrics"))


def test_experiment_record_updates_train_counts():
    record = init_record(
        experiment_id="exp-panels",
        git_commit_hash="abc123",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-a"],
        started_at="2026-04-10T00:00:00+00:00",
    )

    record.record_task_result(_task_result(task_name="train-a", reward=1.0))

    assert record.panels["train"].solved_count == 1


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
    record = init_record(
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

    assert train_results(record)["train-a"].trials[0].solved is solved
    assert train_results(record)["train-a"].majority_solved is solved
    assert record.panels["train"].solved_count == (1 if solved else 0)


def test_experiment_record_evidence_marks_outcomes_without_rule_internals(
    tmp_path,
):
    baseline = init_record(
        experiment_id="baseline",
        git_commit_hash="base123",
        parent_baseline_experiment_id=None,
        train_task_ids=["already-solved", "hard-task"],
        started_at="2026-04-10T00:00:00+00:00",
    )
    # already-solved is solidly solved by the baseline (4/4); a clear,
    # well-separated regression (candidate 0/4) is needed for the two-sample
    # Fisher gate to flag it -- a single-trial dip no longer counts.
    for _ in range(4):
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

    candidate = init_record(
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
    for _ in range(3):
        candidate.record_task_result(
            TaskResult(
                task_name="already-solved",
                reward=0.0,
                solved=False,
                steps_used=4,
                error=None,
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
        for tid, trials in train_results(baseline).items()
    }
    verdicts = build_gate_verdicts(candidate=candidate, pool=pool)
    candidate.refresh_evidence(baseline=baseline, verdicts=verdicts)

    assert candidate.evidence.candidate_change.commit == "candidate123"
    assert candidate.evidence.candidate_change.parent_baseline_commit == "base123"
    outcomes = {outcome.task_id: outcome for outcome in train_outcomes(candidate)}
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
    assert payload["evidence"]["panel_outcomes"]["train"][1]["outcome"] == "new_solve"
    assert set(payload["evidence"]["panel_outcomes"]["train"][1]) == {
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
    loaded_outcomes = {outcome.task_id: outcome for outcome in train_outcomes(loaded)}
    assert loaded_outcomes["hard-task"].outcome == "new_solve"


def test_experiment_record_evidence_writes_train_and_test_panel_outcomes():
    baseline = ExperimentRecord.initialize(
        experiment_id="baseline",
        git_commit_hash="base123",
        parent_baseline_experiment_id=None,
        panels=[
            PanelRecord.initialize(
                panel_id="train",
                purpose="promotion",
                task_ids=["train-a"],
                expected_trial_count=1,
                lifecycle="active",
            ),
            PanelRecord.initialize(
                panel_id="test",
                purpose="regression_veto",
                task_ids=["test-a"],
                expected_trial_count=1,
                lifecycle="active",
            ),
        ],
        started_at="2026-04-10T00:00:00+00:00",
    )
    baseline.record_task_result(
        _task_result(task_name="train-a", reward=0.0, solved=False)
    )
    baseline.record_task_result(
        _task_result(task_name="test-a", reward=1.0, solved=True)
    )

    candidate = ExperimentRecord.initialize(
        experiment_id="candidate",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        panels=[
            PanelRecord.initialize(
                panel_id="train",
                purpose="promotion",
                task_ids=["train-a"],
                expected_trial_count=1,
                lifecycle="active",
            ),
            PanelRecord.initialize(
                panel_id="test",
                purpose="regression_veto",
                task_ids=["test-a"],
                expected_trial_count=1,
                lifecycle="active",
            ),
        ],
        started_at="2026-04-10T00:00:00+00:00",
    )
    candidate.record_task_result(
        _task_result(task_name="train-a", reward=1.0, solved=True)
    )
    # test-a is a clear regression (candidate 0/4 vs a 4/4 control) so the
    # two-sample Fisher gate flags it -- a single failed trial no longer does.
    for _ in range(4):
        candidate.record_task_result(
            _task_result(task_name="test-a", reward=0.0, solved=False)
        )

    train_verdicts = build_gate_verdicts(
        candidate=candidate,
        pool={"train-a": (0, 1)},
    )
    test_verdicts = build_gate_verdicts(
        candidate=candidate,
        pool={"test-a": (4, 4)},
        panel="test",
    )
    candidate.refresh_evidence(
        baseline=baseline,
        verdicts={**train_verdicts, **test_verdicts},
    )

    assert candidate.evidence is not None
    assert set(candidate.evidence.panel_outcomes) == {"train", "test"}
    assert candidate.evidence.panel_outcomes["train"][0].outcome == "new_solve"
    assert candidate.evidence.panel_outcomes["test"][0].outcome == "regression"


def test_experiment_record_evidence_omits_missing_artifact_paths(tmp_path):
    existing_trial_dir = tmp_path / "trial"
    existing_agent_dir = existing_trial_dir / "agent"
    existing_agent_dir.mkdir(parents=True)
    steps_path = existing_agent_dir / "steps.jsonl"
    metrics_path = existing_agent_dir / "metrics.json"
    exec_log_path = existing_agent_dir / "exec.log"
    for path in (steps_path, metrics_path, exec_log_path):
        path.write_text("{}\n")

    candidate = init_record(
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

    outcome = train_outcomes(candidate)[0]
    assert outcome.trial_dir == str(existing_trial_dir)
    assert outcome.agent_steps_path == str(steps_path)
    assert outcome.agent_exec_log_path == str(exec_log_path)
    assert outcome.metrics_path == str(metrics_path)
    assert outcome.verifier_stdout_path is None


def test_task_trials_majority_solved_with_k_trials():
    trials = TaskTrials(task_name="t", expected_trial_count=3, trials=[])
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
    trials = TaskTrials(task_name="t", expected_trial_count=1, trials=[])
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

    empty = TaskTrials(task_name="t", expected_trial_count=3, trials=[])
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


def test_terminal_task_result_classifies_interrupts_distinct_from_crash():
    # A trial stopped from the outside -- a cancellation/Ctrl-C or an outer-loop
    # `ExperimentAbandoned` -- is bucketed `interrupted`, not `crash`; a genuine
    # exception stays `crash`. All three keep `error` set, so all three remain
    # excluded from the gate's valid trials.
    canceled = terminal_task_result(task_id="t", exc=asyncio.CancelledError())
    assert canceled.metrics.failure_mode == "interrupted"
    assert canceled.error == "canceled"

    abandoned = terminal_task_result(
        task_id="t", exc=ExperimentAbandoned("abandoned after supervisor restart")
    )
    assert abandoned.metrics.failure_mode == "interrupted"
    assert abandoned.error == "abandoned after supervisor restart"

    crashed = terminal_task_result(task_id="t", exc=RuntimeError("boom"))
    assert crashed.metrics.failure_mode == "crash"
    assert crashed.error == "boom"

    for result in (canceled, abandoned, crashed):
        assert result.solved is False
        assert result.error is not None  # error set => excluded from valid_trials


def test_finalize_crash_concludes_loaded_partial_panel(tmp_path):
    # Mirrors the supervisor's cross-process crash cleanup
    # (abandon_unfinished_candidate / recover_interrupted_launch), which both
    # `ExperimentRecord.load(...)` a mid-run candidate and then call
    # `finalize_crash` on that freshly loaded copy. The conclusion must rest
    # only on the trials that actually ran. Every assertion here is filler-blind
    # (valid_trials / status / conclusion / evidence presence), so the contract
    # holds whether finalize fabricates crash placeholders for the unrun task or
    # leaves it empty -- the regression net for the A-relax refactor.
    root = tmp_path / "experiments"
    record = init_record(
        experiment_id="cand",
        git_commit_hash="abc123",
        parent_baseline_experiment_id=None,
        train_task_ids=["ran", "never-ran"],
        started_at="2026-04-10T00:00:00+00:00",
    )
    # "ran" produced one real solved trial before the crash; "never-ran" was
    # never scheduled and has no trials at all.
    record.record_task_result(_task_result(task_name="ran", reward=1.0))
    record.write(root=root)

    loaded = ExperimentRecord.load("cand", root=root)
    loaded.finalize_crash(
        exc=ExperimentAbandoned("abandoned after supervisor restart"),
        baseline=None,
        root=root,
    )

    # Concludes as a crash with a finish timestamp.
    assert loaded.status == "crash"
    assert loaded.finished_at is not None
    assert loaded.is_concluded() is True

    # The one real trial is preserved; the unrun task yields no valid evidence
    # (true whether it stays empty or is crash-filled).
    assert [t.solved for t in train_results(loaded)["ran"].valid_trials] == [True]
    assert train_results(loaded)["never-ran"].valid_trials == []

    # Evidence is populated and the gate still scores the single real trial.
    assert loaded.evidence is not None
    verdicts = build_gate_verdicts(
        candidate=loaded, pool={"ran": (0, 0), "never-ran": (0, 0)}
    )
    assert verdicts["ran"].candidate_solved == 1
    assert verdicts["ran"].candidate_total == 1

    # The conclusion is durable on disk.
    reloaded = ExperimentRecord.load("cand", root=root)
    assert reloaded.status == "crash"
    assert reloaded.is_concluded() is True
