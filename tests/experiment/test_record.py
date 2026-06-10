"""Tests for the dumb experiment-record models (record.py + writer.py).

Covers the outcome invariants on ``TrialResult``, the per-task derivations on
``TaskResult`` (which the scheduler and gate read), and the ``ExperimentResult``
write -> load round-trip through the writer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiment.record import ExperimentResult, TaskResult, TrialResult
from src.experiment.writer import write_experiment_result


def _trial(
    run_id: str,
    *,
    solved: bool,
    error: str | None = None,
    failure_mode: str | None = None,
) -> TrialResult:
    # Default the failure_mode to the invariant-consistent value so call sites
    # only specify it when exercising a specific bucket.
    if failure_mode is None:
        failure_mode = "solved" if solved else "verified_rejected"
    return TrialResult(
        run_id=run_id, solved=solved, error=error, failure_mode=failure_mode
    )


# --- TrialResult invariants -------------------------------------------------


def test_solved_requires_solved_failure_mode() -> None:
    assert _trial("r1", solved=True).solved is True
    with pytest.raises(ValueError, match="contradicts failure_mode"):
        TrialResult(run_id="r1", solved=True, failure_mode="hit_timeout")


def test_unsolved_cannot_claim_solved_failure_mode() -> None:
    with pytest.raises(ValueError, match="contradicts failure_mode"):
        TrialResult(run_id="r1", solved=False, failure_mode="solved")


def test_errored_trial_cannot_be_solved() -> None:
    # A crash/interrupt trial carries `error`; it is excluded from scoring and
    # must never read as solved.
    with pytest.raises(ValueError, match="cannot be solved"):
        TrialResult(run_id="r1", solved=True, failure_mode="solved", error="boom")


def test_failure_mode_is_required() -> None:
    # A recorded trial is always classified: `failure_mode=None` would let an
    # unclassified ("executor forgot to classify") trial pass as a scorable
    # unsolved sample, so it is rejected at construction.
    with pytest.raises(ValueError):
        TrialResult(run_id="r1", solved=False, failure_mode=None)  # type: ignore[arg-type]


def test_trial_is_frozen() -> None:
    trial = _trial("r1", solved=True)
    with pytest.raises(Exception):
        trial.solved = False  # type: ignore[misc]


# --- TaskResult derivations -------------------------------------------------


def test_valid_trials_exclude_errored() -> None:
    task = TaskResult(
        expected_trial_count=3,
        trials=[
            _trial("r1", solved=True),
            _trial("r2", solved=False, error="crash", failure_mode="crash"),
            _trial("r3", solved=False),
        ],
    )
    assert [t.run_id for t in task.valid_trials] == ["r1", "r3"]
    assert task.solved_count == 1


def test_majority_solved_is_none_without_valid_trials() -> None:
    task = TaskResult(
        expected_trial_count=1,
        trials=[_trial("r1", solved=False, error="crash", failure_mode="crash")],
    )
    assert task.majority_solved is None


@pytest.mark.parametrize(
    "outcomes, expected",
    [
        ([True, True, False], True),  # 2/3 -> majority
        ([True, False, False], False),  # 1/3 -> not majority
        ([True, False], True),  # 1/2 -> ceil(2/2)=1 threshold met
    ],
)
def test_majority_solved_threshold(outcomes: list[bool], expected: bool) -> None:
    task = TaskResult(
        expected_trial_count=len(outcomes),
        trials=[_trial(f"r{i}", solved=s) for i, s in enumerate(outcomes)],
    )
    assert task.majority_solved is expected


def test_is_deterministic_solved_requires_all_valid_passing() -> None:
    all_pass = TaskResult(
        expected_trial_count=2,
        trials=[_trial("r1", solved=True), _trial("r2", solved=True)],
    )
    assert all_pass.is_deterministic_solved is True

    one_fail = TaskResult(
        expected_trial_count=2,
        trials=[_trial("r1", solved=True), _trial("r2", solved=False)],
    )
    assert one_fail.is_deterministic_solved is False

    assert TaskResult(expected_trial_count=1).is_deterministic_solved is False


def test_is_finished_counts_every_slot_including_crashes() -> None:
    task = TaskResult(expected_trial_count=2)
    assert task.is_finished is False
    task.append(_trial("r1", solved=True))
    assert task.is_finished is False
    task.append(_trial("r2", solved=False, error="crash", failure_mode="crash"))
    # Crash slots count toward termination so an all-crash task cannot loop.
    assert task.is_finished is True


# --- ExperimentResult write -> load round-trip ------------------------------


def test_experiment_result_round_trip(tmp_path: Path) -> None:
    result = ExperimentResult(
        experiment_id="exp-1",
        git_commit_hash="abc123",
        run_status="completed",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:05:00+00:00",
        tasks={
            "task-a": TaskResult(
                expected_trial_count=2,
                trials=[_trial("r1", solved=True), _trial("r2", solved=False)],
            )
        },
    )
    write_experiment_result(result, root=tmp_path)

    assert ExperimentResult.path("exp-1", root=tmp_path).exists()
    loaded = ExperimentResult.load("exp-1", root=tmp_path)
    assert loaded == result
    assert loaded.tasks["task-a"].solved_count == 1
    assert loaded.run_status == "completed"


def test_completed_record_requires_finished_at() -> None:
    # `finished_at` is the ordering/recovery key; a terminal run must carry it.
    with pytest.raises(ValueError, match="inconsistent with"):
        ExperimentResult(
            experiment_id="exp-1",
            git_commit_hash="abc",
            run_status="completed",
            started_at="2026-01-01T00:00:00+00:00",
            finished_at=None,
        )


def test_running_record_forbids_finished_at() -> None:
    with pytest.raises(ValueError, match="inconsistent with"):
        ExperimentResult(
            experiment_id="exp-1",
            git_commit_hash="abc",
            run_status="running",
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:05:00+00:00",
        )


def test_running_record_has_no_finish_timestamp() -> None:
    result = ExperimentResult(
        experiment_id="exp-1",
        git_commit_hash="abc",
        run_status="running",
        started_at="2026-01-01T00:00:00+00:00",
    )
    assert result.finished_at is None
