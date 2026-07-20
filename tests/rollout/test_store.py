from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import cast

import pytest
from conftest import TEST_MEASUREMENT_IDENTITY

from src.rollout.records import (
    ExperimentResult,
    ResultDecision,
    RunKind,
    RolloutResult,
    solved_task_ids,
)
from src.rollout.store import (
    RunStore,
)
from src.rollout.telemetry import RolloutTelemetry


def _experiment(
    experiment_id: str,
    *,
    commit_hash: str = "baseline-commit",
    started_at: str = "2026-06-21T00:00:00+00:00",
    finished_at: str | None = "2026-06-21T00:01:00+00:00",
    crash_reason: str | None = None,
    tasks: dict[str, RolloutResult | None] | None = None,
) -> ExperimentResult:
    # Fresh RolloutResult per call so callers never share a mutable task record.
    if tasks is None:
        tasks = {
            "task-a": RolloutResult(
                task_id="task-a",
                failure_mode="verified_rejected",
                error=None,
                metrics={},
                rollout_dir=None,
                trace_path=None,
                started_at=None,
                finished_at=None,
            )
        }
    return ExperimentResult(
        experiment_id=experiment_id,
        git_commit_hash=commit_hash,
        measurement_identity=TEST_MEASUREMENT_IDENTITY,
        git_dirty=False,
        config_path="config/run.json",
        started_at=started_at,
        finished_at=finished_at,
        crash_reason=crash_reason,
        tasks=tasks,
    )


def _record(
    tracker: RunStore,
    result: ExperimentResult,
    *,
    kind: RunKind = "baseline",
) -> None:
    tracker.record_finalized_run(result, kind=kind)


def _latest(tracker: RunStore) -> ExperimentResult | None:
    return tracker.latest_completed_baseline(
        "baseline-commit", TEST_MEASUREMENT_IDENTITY.digest
    )


def test_mark_crashed_finalizes_a_running_experiment(tmp_path: Path) -> None:
    tracker = RunStore(tmp_path)
    tracker.save_experiment(
        _experiment("exp-1", commit_hash="abc123", finished_at=None, tasks={})
    )

    crashed = tracker.mark_crashed("exp-1", reason="panel failed")

    assert crashed.crash_reason == "panel failed"
    assert crashed.finished_at is not None
    reloaded = tracker.load_experiment("exp-1")
    assert reloaded.crash_reason == "panel failed"
    assert reloaded.finished_at is not None


def test_mark_crashed_preserves_an_existing_crash(tmp_path: Path) -> None:
    tracker = RunStore(tmp_path)
    original = _experiment(
        "exp-1",
        finished_at="2026-06-21T00:02:00+00:00",
        crash_reason="worker failure",
    )
    tracker.save_experiment(original)

    crashed = tracker.mark_crashed("exp-1", reason="parent fallback")

    assert crashed.crash_reason == "worker failure"
    assert crashed.finished_at == original.finished_at
    reloaded = tracker.load_experiment("exp-1")
    assert reloaded.crash_reason == "worker failure"
    assert reloaded.finished_at == original.finished_at


def test_buffer_persists_rollout_and_finalized_index(tmp_path: Path) -> None:
    tracker = RunStore(tmp_path)
    result = _experiment(
        "exp-1", commit_hash="abc123", finished_at=None, tasks={"task-a": None}
    )

    assert solved_task_ids(result) == frozenset()

    tracker.save_experiment(result)
    tracker.log_rollout(
        "exp-1",
        RolloutResult(
            task_id="task-a",
            failure_mode="solved",
            error=None,
            metrics={"reward": 1.0},
            rollout_dir=None,
            trace_path=None,
            started_at=None,
            finished_at=None,
        ),
    )

    loaded = tracker.load_experiment("exp-1")
    assert tracker.experiment_path("exp-1") == tmp_path / "exp-1" / "experiment.json"
    assert loaded.tasks["task-a"] is not None
    assert loaded.tasks["task-a"].failure_mode == "solved"

    completed = loaded.model_copy(
        update={
            "finished_at": datetime.fromisoformat("2026-06-21T00:02:00+00:00"),
        }
    )
    tracker.save_experiment(completed)
    finalized = tracker.record_finalized_run(
        completed,
        kind="candidate",
        parent_commit_hash="parent",
        baseline_experiment_id="baseline-1",
        decision=ResultDecision(
            outcome="promoted",
            reason="strict_improvement_without_regression",
        ),
    )

    assert finalized.kind == "candidate"
    assert finalized.parent_commit_hash == "parent"
    assert finalized.baseline_experiment_id == "baseline-1"
    assert finalized.decision is not None
    [row] = tracker.read_index()
    assert row.experiment_id == "exp-1"
    assert row.kind == "candidate"
    assert row.parent_commit_hash == "parent"
    assert row.baseline_experiment_id == "baseline-1"
    assert row.crash_reason is None
    assert row.solved == 1
    assert row.verdict == "promoted"
    assert row.reason == "strict_improvement_without_regression"


def test_rollout_telemetry_writes_generic_events_without_metrics_file(
    tmp_path: Path,
) -> None:
    telemetry = RolloutTelemetry(
        rollout_dir=tmp_path,
        trace_path=tmp_path / "agent" / "steps.jsonl",
    )

    telemetry.event("action_parse_failed", step_index=1, error="bad", detail="bad")
    telemetry.event(
        "step_completed",
        step_index=2,
        call_index=0,
        step_result={"reward": 1.0, "terminated": True, "truncated": False},
        terminal=True,
    )
    telemetry.event("mechanism_event", family="probe", mechanism="opportunity_seen")

    rows = [
        json.loads(line)
        for line in (tmp_path / "agent" / "steps.jsonl").read_text().splitlines()
    ]
    assert [row["event"] for row in rows] == [
        "action_parse_failed",
        "step_completed",
        "mechanism_event",
    ]
    assert rows[1]["call_index"] == 0
    assert rows[1]["step_result"]["reward"] == 1.0
    assert rows[2]["family"] == "probe"
    assert rows[2]["mechanism"] == "opportunity_seen"
    assert not (tmp_path / "agent" / "metrics.json").exists()


def test_invalid_index_row_does_not_persist_finalized_experiment(
    tmp_path: Path,
) -> None:
    tracker = RunStore(tmp_path)
    result = _experiment(
        "exp-1",
        commit_hash="abc123",
        finished_at="2026-06-21T00:02:00+00:00",
        tasks={
            "task-a": RolloutResult(
                task_id="task-a",
                failure_mode="solved",
                error=None,
                metrics={},
                rollout_dir="runs/exp-1/tasks/task-a",
                trace_path="runs/exp-1/tasks/task-a/agent/steps.jsonl",
                started_at="2026-06-21T00:00:00+00:00",
                finished_at="2026-06-21T00:01:00+00:00",
            )
        },
    )

    with pytest.raises(ValueError, match="kind"):
        tracker.record_finalized_run(result, kind=cast(RunKind, "unknown"))

    assert not tracker.experiment_path("exp-1").exists()


def test_latest_completed_baseline_returns_none_when_no_source_exists(
    tmp_path: Path,
) -> None:
    tracker = RunStore(tmp_path)
    assert _latest(tracker) is None

    _record(
        tracker,
        _experiment("other-commit", commit_hash="other"),
    )
    _record(
        tracker,
        _experiment("candidate", finished_at="2026-06-21T00:02:00+00:00"),
        kind="candidate",
    )
    _record(
        tracker,
        _experiment(
            "crashed-baseline",
            finished_at="2026-06-21T00:03:00+00:00",
            crash_reason="crashed",
        ),
    )

    assert _latest(tracker) is None


def test_latest_completed_baseline_loads_newest_completed_baseline(
    tmp_path: Path,
) -> None:
    tracker = RunStore(tmp_path)
    _record(
        tracker,
        _experiment("old-baseline", finished_at="2026-06-21T00:01:00+00:00"),
    )
    _record(
        tracker,
        _experiment("candidate", finished_at="2026-06-21T00:04:00+00:00"),
        kind="candidate",
    )
    _record(
        tracker,
        _experiment(
            "other-commit", commit_hash="other", finished_at="2026-06-21T00:05:00+00:00"
        ),
    )
    _record(
        tracker,
        _experiment("new-baseline", finished_at="2026-06-21T00:03:00+00:00"),
    )

    baseline = _latest(tracker)

    assert baseline is not None
    assert baseline.experiment_id == "new-baseline"
    assert baseline.tasks["task-a"] is not None


def test_latest_completed_baseline_breaks_timestamp_ties_by_index_order(
    tmp_path: Path,
) -> None:
    tracker = RunStore(tmp_path)
    _record(tracker, _experiment("first-baseline"))
    _record(tracker, _experiment("second-baseline"))

    baseline = _latest(tracker)

    assert baseline is not None
    assert baseline.experiment_id == "second-baseline"


def test_latest_completed_baseline_requires_measurement_identity(
    tmp_path: Path,
) -> None:
    tracker = RunStore(tmp_path)
    _record(tracker, _experiment("matching"))
    other_identity = TEST_MEASUREMENT_IDENTITY.model_copy(
        update={"provider_revision": "other-provider"}
    )
    _record(
        tracker,
        _experiment("newer-mismatch").model_copy(
            update={"measurement_identity": other_identity}
        ),
    )

    baseline = _latest(tracker)

    assert baseline is not None
    assert baseline.experiment_id == "matching"
