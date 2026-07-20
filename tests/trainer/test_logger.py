from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from conftest import TEST_MEASUREMENT_IDENTITY

import src.trainer.logger as logger
from src.rollout.records import ExperimentResult, ResultDecision, RolloutResult
from src.trainer.logger import StdoutLogger


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [0.0]
    monkeypatch.setattr(logger.time, "monotonic", lambda: now[0])
    return now


def test_epoch_block_and_training_summary(
    capsys: pytest.CaptureFixture[str], clock: list[float]
) -> None:
    baseline = _experiment_payload(
        "exp-baseline",
        tasks={
            "solved-a": _rollout_payload("solved-a", "solved"),
            "solved-b": _rollout_payload("solved-b", "solved"),
            "failed": _rollout_payload("failed", "verified_rejected"),
        },
    )
    rejected = ResultDecision(
        outcome="rejected",
        reason="no_net_improvement",
        candidate_solved=["solved-a", "solved-b"],
        baseline_solved=["solved-a", "solved-b"],
    )
    callback = StdoutLogger()

    callback.epoch_started(1, 1, baseline)
    callback.decision_finished(rejected)
    clock[0] = 65.0
    callback.epoch_finished("rejected")
    callback.loop_finished(1, [rejected], baseline)

    assert capsys.readouterr().out == (
        "epoch 1/1 · baseline solved 2/3\n"
        "epoch 1/1 · rejected · no net improvement · 1m05s\n"
        "train     · done · 2 → 2/3 · 0 promoted · 1 rejected\n"
    )


def test_run_started_prints_orientation_banner(
    capsys: pytest.CaptureFixture[str],
) -> None:
    callback = StdoutLogger()

    callback.run_started(
        "train · config/train.yaml",
        (
            ("policy", "src/policy/core.py @ abc12345"),
            ("llm", "gpt-oss-20b · http://localhost:8080/v1"),
            ("criterion", "strict_pareto · tiebreak: steps_used"),
        ),
    )

    assert capsys.readouterr().out == (
        "train · config/train.yaml\n"
        "  policy    src/policy/core.py @ abc12345\n"
        "  llm       gpt-oss-20b · http://localhost:8080/v1\n"
        "  criterion strict_pareto · tiebreak: steps_used\n"
    )


def test_candidate_measurement_prints_only_decision_relevant_events(
    capsys: pytest.CaptureFixture[str], clock: list[float]
) -> None:
    baseline = _experiment_payload(
        "baseline-1",
        tasks={
            "unchanged": _rollout_payload("unchanged", "solved"),
            "new-solve": _rollout_payload("new-solve", "verified_rejected"),
            "regressed": _rollout_payload("regressed", "solved"),
            "slow-failure": _rollout_payload("slow-failure", "hit_step_cap"),
            "excluded": _rollout_payload("excluded", "solved"),
        },
    )
    candidate = _experiment_payload(
        "candidate-1",
        tasks={
            "unchanged": _rollout_payload("unchanged", "solved"),
            "new-solve": _rollout_payload("new-solve", "solved"),
            "regressed": _rollout_payload("regressed", "verified_rejected"),
            "slow-failure": _rollout_payload("slow-failure", "hit_timeout"),
        },
    )
    callback = StdoutLogger()
    callback.epoch_started(2, 3, baseline)
    callback.measurement_started(tuple(candidate.tasks), baseline, subject="candidate")

    callback.experiment_started(candidate.experiment_id)
    callback.task_finished("unchanged", "solved")
    callback.task_finished("new-solve", "solved")
    callback.task_finished("regressed", "verified_rejected")
    callback.task_finished("slow-failure", "hit_timeout")
    callback.experiment_finished(candidate)
    callback.decision_finished(
        ResultDecision(
            outcome="rejected",
            reason="regressed_baseline_tasks",
            candidate_solved=["new-solve", "unchanged"],
            baseline_solved=["regressed", "unchanged"],
            new_solves=["new-solve"],
            regressions=["regressed"],
        )
    )
    clock[0] = 300.0
    callback.epoch_finished("rejected")

    assert capsys.readouterr().out == (
        "epoch 2/3 · baseline solved 3/5\n"
        "measure   · candidate · candidate-1 · 4 tasks\n"
        "measure   · NEW SOLVE · new-solve\n"
        "measure   · REGRESSION · regressed (verified_rejected)\n"
        "measure   · slow-failure · hit_timeout\n"
        # 3/5 and 2/4 are different task panels; certification pruned one.
        "candidate · solved 2/4 · +1 new · 1 regr · 1 excluded (nondet) · 3m00s\n"
        "epoch 2/3 · rejected · regressed regressed · candidate-1 · 5m00s\n"
    )


def test_measurement_heartbeat_reports_only_after_sustained_silence(
    capsys: pytest.CaptureFixture[str], clock: list[float]
) -> None:
    callback = StdoutLogger()
    callback.measurement_started(
        ("task-a", "task-b", "task-c", "task-d"), None, subject="baseline"
    )
    callback.experiment_started("baseline-1")
    callback.task_finished("task-a", "solved")
    callback.task_finished("task-b", "verified_rejected")
    capsys.readouterr()

    clock[0] = 119.0
    callback.measurement_heartbeat()
    assert capsys.readouterr().out == ""

    clock[0] = 120.0
    callback.measurement_heartbeat()
    assert capsys.readouterr().out == (
        "measure   · 2/4 complete · 2m00s · waiting: task-c, task-d\n"
    )

    clock[0] = 419.0
    callback.measurement_heartbeat()
    assert capsys.readouterr().out == ""

    clock[0] = 420.0
    callback.measurement_heartbeat()
    assert capsys.readouterr().out == (
        "measure   · 2/4 complete · 7m00s · waiting: task-c, task-d\n"
    )


def test_baseline_measurement_suppresses_ordinary_task_rows(
    capsys: pytest.CaptureFixture[str], clock: list[float]
) -> None:
    result = _experiment_payload(
        "baseline-1",
        tasks={
            "task-a": _rollout_payload("task-a", "solved"),
            "task-b": _rollout_payload("task-b", "verified_rejected"),
        },
    )
    callback = StdoutLogger()
    callback.measurement_started(tuple(result.tasks), None, subject="baseline")

    callback.experiment_started(result.experiment_id)
    callback.task_finished("task-a", "solved")
    callback.task_finished("task-b", "verified_rejected")
    callback.experiment_finished(result)

    assert capsys.readouterr().out == (
        "measure   · baseline · baseline-1 · 2 tasks\nbaseline  · solved 1/2 · 3m00s\n"
    )


def test_freshly_measured_baseline_seeds_the_run_movement(
    capsys: pytest.CaptureFixture[str], clock: list[float]
) -> None:
    # No cached baseline, so epoch_started sees None; experiment_finished must
    # capture the first measured baseline as the run's starting score.
    initial_baseline = _experiment_payload(
        "baseline-1",
        tasks={
            "task-a": _rollout_payload("task-a", "solved"),
            "task-b": _rollout_payload("task-b", "verified_rejected"),
        },
    )
    promoted_baseline = _experiment_payload(
        "candidate-1",
        tasks={
            "task-a": _rollout_payload("task-a", "solved"),
            "task-b": _rollout_payload("task-b", "solved"),
        },
    )
    promoted = ResultDecision(
        outcome="promoted",
        reason="strict_improvement_without_regression",
        candidate_solved=["task-a", "task-b"],
        baseline_solved=["task-a"],
    )
    callback = StdoutLogger()

    callback.epoch_started(1, 1, None)
    callback.measurement_started(
        tuple(initial_baseline.tasks), None, subject="baseline"
    )
    callback.experiment_finished(initial_baseline)
    callback.loop_finished(1, [promoted], promoted_baseline)

    assert capsys.readouterr().out == (
        "epoch 1/1\n"
        "baseline  · solved 1/2 · 3m00s\n"
        "train     · done · 1 → 2/2 · 1 promoted · 0 rejected\n"
    )


def test_live_epoch_collapses_to_single_promoted_line(
    capsys: pytest.CaptureFixture[str], clock: list[float]
) -> None:
    baseline = _experiment_payload(
        "baseline-1",
        tasks={
            "task-a": _rollout_payload("task-a", "solved"),
            "task-b": _rollout_payload("task-b", "verified_rejected"),
        },
    )
    candidate = _experiment_payload(
        "candidate-1",
        tasks={
            "task-a": _rollout_payload("task-a", "solved"),
            "task-b": _rollout_payload("task-b", "solved"),
        },
    )
    callback = StdoutLogger(live=True)

    callback.epoch_started(1, 2, baseline)
    callback.agent_progress("propose", "running · 10s · agent: cmd 3")
    callback.agent_progress("propose", "done · 1m00s · agent: cmd 5")
    callback.measurement_started(tuple(candidate.tasks), baseline, subject="candidate")
    callback.experiment_started(candidate.experiment_id)
    callback.task_finished("task-a", "solved")
    callback.task_finished("task-b", "solved")
    callback.experiment_finished(candidate)
    callback.decision_finished(
        ResultDecision(
            outcome="promoted",
            reason="strict_improvement_without_regression",
            candidate_solved=["task-a", "task-b"],
            baseline_solved=["task-a"],
            new_solves=["task-b"],
        )
    )
    clock[0] = 60.0
    callback.epoch_finished("promoted")

    out = capsys.readouterr().out
    # The whole epoch block (header, propose line, candidate line) collapses.
    assert out.endswith(
        "\x1b[3A\x1b[0J"
        "epoch 1/2 · \x1b[32m\x1b[1mPROMOTED\x1b[0m · 1 → 2/2 (+task-b)"
        " · candidate-1 · 1m00s\n"
    )
    assert "  propose   · running · 10s · agent: cmd 3\n" in out
    assert "  candidate · solved 2/2 · +1 new · 0 regr · 3m00s\n" in out
    assert "  1/2 done · 0s\n" in out
    assert f"{' ' * 14}\x1b[2mcandidate-1\x1b[0m\n" in out


def test_live_log_lines_land_above_the_block(
    capsys: pytest.CaptureFixture[str], clock: list[float]
) -> None:
    callback = StdoutLogger(live=True)
    callback.epoch_started(1, 1, None)
    capsys.readouterr()

    callback.log("warning: something odd")

    # The block (one header line) is rewound, the warning printed, block redrawn.
    assert capsys.readouterr().out == (
        "\x1b[1A\x1b[0Jwarning: something odd\nepoch 1/1\n"
    )


def test_experiment_failure_uses_compact_first_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    callback = StdoutLogger()
    callback.measurement_started(("task-a",), None, subject="eval config.yaml")

    callback.experiment_failed(RuntimeError("worker exited\ndiagnostic detail"))

    assert capsys.readouterr().out == ("eval config.yaml · FAILED — worker exited\n")


def test_epoch_log_scopes_relevant_determinism_and_measurement_messages(
    capsys: pytest.CaptureFixture[str], clock: list[float]
) -> None:
    callback = StdoutLogger()
    callback.epoch_started(1, 1, None)

    callback.log("determinism: cached (exp-1, excluded 0/36 nondeterministic tasks)")
    callback.log("determinism: excluded 1/36 nondeterministic tasks")
    callback.measurement_retrying("unmeasurable run")

    assert capsys.readouterr().out == (
        "epoch 1/1\n"
        "measure   · determinism: excluded 1/36 nondeterministic tasks\n"
        "measure   · retry · unmeasurable run\n"
    )


def test_skipped_epoch_line_carries_the_cause(
    capsys: pytest.CaptureFixture[str], clock: list[float]
) -> None:
    callback = StdoutLogger()
    callback.epoch_started(1, 1, None)
    callback.epoch_skipped("no_candidate_patch")
    clock[0] = 30.0
    callback.epoch_finished("skipped")

    assert capsys.readouterr().out == (
        "epoch 1/1\nepoch 1/1 · skipped · no candidate patch · 30s\n"
    )


def test_infra_failures_surface_on_the_epoch_line(
    capsys: pytest.CaptureFixture[str], clock: list[float]
) -> None:
    baseline = _experiment_payload(
        "baseline-1",
        tasks={"task-a": _rollout_payload("task-a", "solved")},
    )
    candidate = _experiment_payload(
        "candidate-1",
        tasks={"task-a": _rollout_payload("task-a", "crash")},
    )
    callback = StdoutLogger()
    callback.epoch_started(1, 1, baseline)
    callback.measurement_started(("task-a",), baseline, subject="candidate")
    callback.task_finished("task-a", "crash")
    callback.experiment_finished(candidate)
    clock[0] = 60.0
    callback.epoch_finished("invalid_infra")

    out = capsys.readouterr().out
    assert "candidate · solved 0/1 · +0 new · 1 regr · 3m00s · crash 1\n" in out
    assert "epoch 1/1 · invalid infra · 1m00s · crash 1\n" in out


def test_suite_summary_renders_one_aligned_row_per_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    clean = _experiment_payload(
        "exp-model__panel-a",
        tasks={
            "solved-a": _rollout_payload("solved-a", "solved"),
            "solved-b": _rollout_payload("solved-b", "solved"),
        },
    )
    crashed = _experiment_payload(
        "exp-model__panel-b__iteration-2",
        tasks={
            "solved-a": _rollout_payload("solved-a", "solved"),
            "timed-out": _rollout_payload("timed-out", "hit_timeout"),
            "unran": None,
        },
        crash_reason="worker exited",
    )

    logger.suite_summary([clean, crashed])

    assert capsys.readouterr().out == (
        "\n"
        "summary   · 2 runs · 3m00s\n"
        "  exp-model__panel-a                 2/2      3m00s  -\n"
        "  exp-model__panel-b__iteration-2    1/3      3m00s  "
        "CRASHED · hit_timeout 1 · unfinished 1\n"
    )


def test_suite_summary_prints_nothing_without_results(
    capsys: pytest.CaptureFixture[str],
) -> None:
    logger.suite_summary([])

    assert capsys.readouterr().out == ""


def _experiment_payload(
    experiment_id: str,
    *,
    tasks: dict[str, dict[str, Any] | None],
    crash_reason: str | None = None,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        git_commit_hash="abc123",
        measurement_identity=TEST_MEASUREMENT_IDENTITY,
        git_dirty=False,
        config_path="config/run.yaml",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at=datetime.fromisoformat("2026-01-01T00:03:00+00:00"),
        crash_reason=crash_reason,
        tasks={
            task_id: None if row is None else RolloutResult.model_validate(row)
            for task_id, row in tasks.items()
        },
    )


def _rollout_payload(task_id: str, failure_mode: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "failure_mode": failure_mode,
        "failure_origin": "policy" if failure_mode == "crash" else None,
        "error": None,
        "metrics": {},
        "rollout_dir": None,
        "trace_path": None,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:02:00+00:00",
    }
