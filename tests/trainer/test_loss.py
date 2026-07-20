from __future__ import annotations

from collections.abc import Collection, Mapping
from math import inf
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from conftest import TEST_MEASUREMENT_IDENTITY

from src.env.base import benchmark
from src.rollout.metrics import GENERIC_SECONDARY_METRICS, SecondaryRewardMetric
from src.rollout.records import (
    ExperimentResult,
    FailureMode,
    ResultDecision,
    RolloutResult,
    SecondaryRewardComparison,
    SecondaryRewardOutcome,
)
from src.trainer.loss import Comparison, Loss, StrictPareto, UnmeasurableRun

_SWE_METRICS = benchmark("swe").secondary_metrics


class _StubHarness:
    def __init__(self, data: str = "abc123") -> None:
        self.data = data
        self.grad: Loss | None = None
        self.applied: str | None = None


def _criterion(
    data: str = "abc123",
    metrics: tuple[SecondaryRewardMetric, ...] = GENERIC_SECONDARY_METRICS,
) -> StrictPareto:
    criterion = StrictPareto(secondary_metrics=metrics)
    criterion.harness = _StubHarness(data)  # type: ignore[assignment]
    return criterion


class TotalTaskSolves(StrictPareto):
    """Net-count criterion: overrides Pareto scalarization, keeps set bracketing."""

    name = "total_task_solves"

    def _compare(
        self, comparison: Comparison
    ) -> tuple[str, tuple[SecondaryRewardComparison, ...]]:
        delta = len(comparison.candidate_solved) - len(comparison.baseline_solved)
        if delta > 0:
            reason = "higher_total_task_solves"
        elif delta < 0:
            reason = "lower_total_task_solves"
        else:
            reason = "no_total_task_solve_improvement"
        return reason, ()

    def _loss_value(self, report: Comparison) -> float:
        return float(len(report.baseline_solved) - len(report.candidate_solved))


def _total_task_solves_criterion(
    data: str = "abc123",
) -> TotalTaskSolves:
    criterion = TotalTaskSolves()
    criterion.harness = _StubHarness(data)  # type: ignore[assignment]
    return criterion


def _rollout(
    task_id: str,
    *,
    solved: bool,
    failure_mode: FailureMode | None = None,
    steps: int | None = None,
    f2p: int | None = None,
    p2p: int | None = None,
    validity: tuple[int, int] | None = None,
) -> RolloutResult:
    metrics = {
        key: value
        for key, value in (
            ("steps_used", steps),
            ("fail_to_pass_passed", f2p),
            ("pass_to_pass_failed", p2p),
        )
        if value is not None
    }
    if validity is not None:
        metrics.update(first_attempt_valid=validity[0], first_attempt_total=validity[1])
    mode = failure_mode or ("solved" if solved else "verified_rejected")
    return RolloutResult(
        task_id=task_id,
        failure_mode=mode,
        failure_origin="policy" if mode == "crash" else None,
        error=None,
        metrics=metrics,
        rollout_dir=None,
        trace_path=None,
        started_at=None,
        finished_at=None,
    )


def _experiment(
    experiment_id: str,
    *,
    solved: Collection[str] = (),
    steps: Mapping[str, int | None] | None = None,
    f2p: Mapping[str, int] | None = None,
    p2p: Mapping[str, int] | None = None,
    validity: Mapping[str, tuple[int, int]] | None = None,
    failure_modes: Mapping[str, FailureMode] | None = None,
    missing: Collection[str] = (),
    crash_reason: str | None = None,
) -> ExperimentResult:
    steps, f2p, p2p = steps or {}, f2p or {}, p2p or {}
    validity, failure_modes = validity or {}, failure_modes or {}
    task_ids = set(solved) | steps.keys() | f2p.keys() | p2p.keys()
    task_ids |= validity.keys() | failure_modes.keys()
    tasks: dict[str, RolloutResult | None] = {
        task_id: _rollout(
            task_id,
            solved=task_id in solved,
            failure_mode=failure_modes.get(task_id),
            steps=steps.get(task_id),
            f2p=f2p.get(task_id),
            p2p=p2p.get(task_id),
            validity=validity.get(task_id),
        )
        for task_id in task_ids
    }
    tasks.update(dict.fromkeys(missing))
    return ExperimentResult(
        experiment_id=experiment_id,
        git_commit_hash="cand456"
        if experiment_id.startswith("candidate")
        else "abc123",
        git_dirty=False,
        measurement_identity=TEST_MEASUREMENT_IDENTITY,
        config_path="config/run.json",
        started_at="2026-06-21T00:00:00+00:00",
        finished_at="2026-06-21T00:01:00+00:00",
        crash_reason=crash_reason,
        tasks=tasks,
    )


def _greedy_decision(loss: Loss) -> ResultDecision:
    loss.harness.applied = (
        loss.candidate.git_commit_hash if loss.value < 0 else "abc123"
    )
    return StrictPareto().decision(loss)


def _decision(
    baseline: ExperimentResult,
    candidate: ExperimentResult,
    metrics: tuple[SecondaryRewardMetric, ...] = GENERIC_SECONDARY_METRICS,
) -> ResultDecision:
    panel = sorted(baseline.tasks.keys() | candidate.tasks.keys())
    baseline = baseline.model_copy(
        update={"tasks": {task: baseline.tasks.get(task) for task in panel}}
    )
    candidate = candidate.model_copy(
        update={"tasks": {task: candidate.tasks.get(task) for task in panel}}
    )
    criterion = _criterion(baseline.git_commit_hash, metrics)
    try:
        return _greedy_decision(criterion(candidate=candidate, baseline=baseline))
    except UnmeasurableRun as exc:
        return exc.decision


def _assert_fields(actual: object, **expected: object) -> None:
    for name, value in expected.items():
        assert getattr(actual, name) == value, name


@pytest.mark.parametrize(
    ("baseline", "candidate", "expected"),
    [
        pytest.param(
            dict(solved=("task-a",)),
            dict(solved=("task-a", "task-b")),
            (
                "promoted",
                "strict_improvement_without_regression",
                ["task-b"],
                [],
                [],
                True,
            ),
            id="strict-solve-improvement",
        ),
        pytest.param(
            dict(solved=("task-a",)),
            dict(solved=("task-b", "task-c")),
            (
                "rejected",
                "regressed_baseline_tasks",
                ["task-b", "task-c"],
                ["task-a"],
                [],
                True,
            ),
            id="regression-overrides-new-solves",
        ),
        pytest.param(
            dict(solved=("task-a",)),
            dict(solved=("task-a",)),
            ("rejected", "no_net_improvement", [], [], [], False),
            id="no-strict-improvement",
        ),
        pytest.param(
            dict(solved=("a", "b"), steps={"a": 10, "b": 10}),
            dict(solved=("a",), steps={"a": 1}),
            ("rejected", "regressed_baseline_tasks", [], ["b"], [], True),
            id="regression-precedes-secondary",
        ),
        pytest.param(
            dict(solved=("a",), steps={"a": 1}),
            dict(solved=("a", "b"), steps={"a": 100, "b": 100}),
            (
                "promoted",
                "strict_improvement_without_regression",
                ["b"],
                [],
                [],
                True,
            ),
            id="primary-precedes-secondary",
        ),
        pytest.param(
            dict(solved=("a",)),
            dict(solved=("b",), failure_modes={"a": "crash"}),
            (
                "invalid_infra",
                "infra_sensitive_verdict",
                ["b"],
                [],
                ["a"],
                False,
            ),
            id="candidate-crash-sensitive",
        ),
        pytest.param(
            dict(solved=("a",)),
            dict(solved=("a", "b"), failure_modes={"c": "crash"}),
            (
                "promoted",
                "strict_improvement_without_regression",
                ["b"],
                [],
                ["c"],
                True,
            ),
            id="candidate-crash-on-unsolved-task",
        ),
        pytest.param(
            dict(solved=("a",)),
            dict(solved=("b",), failure_modes={"a": "unscorable_infra"}),
            (
                "invalid_infra",
                "infra_sensitive_verdict",
                ["b"],
                [],
                ["a"],
                False,
            ),
            id="unscorable-infra-sensitive",
        ),
        pytest.param(
            dict(solved=("a",)),
            dict(solved=("a", "b"), failure_modes={"c": "unscorable_infra"}),
            (
                "promoted",
                "strict_improvement_without_regression",
                ["b"],
                [],
                ["c"],
                True,
            ),
            id="unscorable-infra-stable-promotion",
        ),
        pytest.param(
            dict(solved=("a", "b")),
            dict(solved=("b",), failure_modes={"c": "unscorable_infra"}),
            ("rejected", "regressed_baseline_tasks", [], ["a"], ["c"], True),
            id="unscorable-infra-with-known-regression",
        ),
        pytest.param(
            dict(solved=("a",), failure_modes={"c": "crash"}),
            dict(solved=("a", "b")),
            (
                "promoted",
                "strict_improvement_without_regression",
                ["b"],
                [],
                [],
                True,
            ),
            id="baseline-crash-ignored",
        ),
        pytest.param(
            dict(solved=("a",)),
            dict(solved=("b",), failure_modes={"a": "verify_timeout"}),
            (
                "invalid_infra",
                "infra_sensitive_verdict",
                ["b"],
                [],
                ["a"],
                False,
            ),
            id="verify-timeout-sensitive",
        ),
    ],
)
def test_decision_classes(
    baseline: dict[str, object],
    candidate: dict[str, object],
    expected: tuple[str, str, list[str], list[str], list[str], bool],
) -> None:
    outcome, reason, new, regressions, invalid, no_rewards = expected
    decision = _decision(
        _experiment("baseline", **baseline),  # type: ignore[arg-type]
        _experiment("candidate", **candidate),  # type: ignore[arg-type]
    )
    _assert_fields(
        decision,
        outcome=outcome,
        reason=reason,
        baseline_solved=sorted(baseline.get("solved", ())),
        candidate_solved=sorted(candidate.get("solved", ())),
        new_solves=new,
        regressions=regressions,
        invalid_infra_tasks=invalid,
        criterion="strict_pareto",
    )
    if no_rewards:
        assert decision.secondary_rewards == []


def test_criterion_compares_only_on_the_candidate_panel() -> None:
    loss = _criterion()(
        baseline=_experiment("baseline", solved=("kept", "flaky")),
        candidate=_experiment("candidate", solved=("kept",)),
    )
    _assert_fields(
        _greedy_decision(loss),
        outcome="rejected",
        baseline_solved=["kept"],
        candidate_solved=["kept"],
        regressions=[],
    )


def test_criterion_rejects_candidate_tasks_the_baseline_never_measured() -> None:
    with pytest.raises(
        ValueError, match="candidate ran tasks the baseline never measured"
    ):
        _criterion()(
            baseline=_experiment("baseline", solved=("task-a",)),
            candidate=_experiment("candidate", solved=("task-a", "off-panel")),
        )


def test_comparison_derives_facts_from_runs_and_resolution() -> None:
    comparison = Comparison(
        baseline=_experiment(
            "baseline",
            solved=("a", "b"),
            steps={"c": 1},
        ),
        candidate=_experiment(
            "candidate",
            solved=("b", "c"),
            failure_modes={"a": "verify_timeout"},
        ),
        candidate_solved=frozenset({"b", "c"}),
        invalid_infra_tasks=frozenset({"a"}),
    )

    assert comparison.baseline_solved == frozenset({"a", "b"})
    assert comparison.regressions == frozenset()
    assert comparison.new_solves == frozenset({"c"})


def test_total_task_solves_promotes_aggregate_gain_despite_regression() -> None:
    loss = _total_task_solves_criterion()(
        baseline=_experiment(
            "baseline",
            solved=("a",),
            steps={"b": 1, "c": 1},
        ),
        candidate=_experiment(
            "candidate",
            solved=("b", "c"),
            steps={"a": 1},
        ),
    )

    assert loss.value == -1.0
    assert loss.report.regressions == frozenset({"a"})
    assert loss.report.new_solves == frozenset({"b", "c"})
    assert loss.report.reason == "higher_total_task_solves"


def test_total_task_solves_inherits_infra_sensitive_verdict_detection() -> None:
    baseline = _experiment("baseline", solved=("a",), steps={"b": 1})
    candidate = _experiment(
        "candidate",
        solved=("b",),
        failure_modes={"a": "verify_timeout"},
    )

    with pytest.raises(UnmeasurableRun) as raised:
        _total_task_solves_criterion()(baseline=baseline, candidate=candidate)

    _assert_fields(
        raised.value.decision,
        outcome="invalid_infra",
        reason="infra_sensitive_verdict",
        regressions=[],
        invalid_infra_tasks=["a"],
        criterion="total_task_solves",
    )


@pytest.mark.parametrize(
    ("baseline", "candidate", "metrics", "reward_name", "comparison", "prior"),
    [
        pytest.param(
            dict(solved=("a", "b"), steps={"a": 10, "b": 8}),
            dict(solved=("a", "b"), steps={"a": 6, "b": 7}),
            GENERIC_SECONDARY_METRICS,
            "steps_used",
            (18, 13, "candidate_better", "promoted"),
            {"invalid_first_attempts": "unavailable"},
            id="fewer-steps-promotes",
        ),
        pytest.param(
            dict(solved=("a",), steps={"a": 5}),
            dict(solved=("a",), steps={"a": 5}),
            GENERIC_SECONDARY_METRICS,
            "steps_used",
            (5, 5, "tied", "rejected"),
            {"invalid_first_attempts": "unavailable"},
            id="same-steps-rejects",
        ),
        pytest.param(
            dict(solved=("a",), steps={"a": 5}),
            dict(solved=("a",), steps={"a": 9}),
            GENERIC_SECONDARY_METRICS,
            "steps_used",
            (5, 9, "baseline_better", "rejected"),
            {"invalid_first_attempts": "unavailable"},
            id="more-steps-rejects",
        ),
        pytest.param(
            dict(solved=("a",), steps={"a": None}),
            dict(solved=("a",), steps={"a": 4}),
            GENERIC_SECONDARY_METRICS,
            "steps_used",
            (None, 4, "unavailable", "rejected"),
            {"invalid_first_attempts": "unavailable"},
            id="missing-steps-rejects",
        ),
        pytest.param(
            dict(solved=("a",), steps={"a": 5}, f2p={"b": 2}),
            dict(solved=("a",), steps={"a": 5}, f2p={"b": 5}),
            _SWE_METRICS,
            "f2p_progress",
            (2, 5, "candidate_better", "promoted"),
            {},
            id="f2p-progress-promotes",
        ),
        pytest.param(
            dict(f2p={"b": 2}),
            dict(f2p={"b": 4}, p2p={"b": 5}),
            _SWE_METRICS,
            "f2p_progress",
            (2, -1, "baseline_better", "rejected"),
            {},
            id="f2p-nets-p2p-breakage",
        ),
        pytest.param(
            dict(solved=("a",), steps={"a": 5}, f2p={"b": 3}),
            dict(solved=("a",), steps={"a": 5}, failure_modes={"b": "crash"}),
            _SWE_METRICS,
            "f2p_progress",
            (0, 0, "tied", "invalid_infra"),
            {},
            id="f2p-ignores-unscorable-task",
        ),
        pytest.param(
            dict(solved=("a",), steps={"a": 5}, validity={"a": (3, 10)}),
            dict(solved=("a",), steps={"a": 5}, validity={"a": (7, 10)}),
            _SWE_METRICS,
            "invalid_first_attempts",
            (7, 3, "candidate_better", "promoted"),
            {"f2p_progress": "tied"},
            id="validity-precedes-steps",
        ),
        pytest.param(
            dict(solved=("a",), steps={"a": 10}, validity={"a": (9, 10)}),
            dict(solved=("a",), steps={"a": 6}, validity={"a": (5, 6)}),
            GENERIC_SECONDARY_METRICS,
            "steps_used",
            (10, 6, "candidate_better", "promoted"),
            {"invalid_first_attempts": "tied"},
            id="equal-invalid-counts-tie-regardless-of-run-length",
        ),
        pytest.param(
            dict(solved=("a",), steps={"a": 5}, validity={"b": (2, 10)}),
            dict(
                solved=("a",),
                steps={"a": 5},
                validity={"b": (8, 10)},
                failure_modes={"b": "verify_timeout"},
            ),
            GENERIC_SECONDARY_METRICS,
            "invalid_first_attempts",
            (8, 2, "candidate_better", "promoted"),
            {},
            id="verify-timeout-remains-scorable",
        ),
    ],
)
def test_secondary_reward_classes(
    baseline: dict[str, object],
    candidate: dict[str, object],
    metrics: tuple[SecondaryRewardMetric, ...],
    reward_name: str,
    comparison: tuple[int | float | None, int | float, str, str],
    prior: dict[str, str],
) -> None:
    decision = _decision(
        _experiment("baseline", **baseline),  # type: ignore[arg-type]
        _experiment("candidate", **candidate),  # type: ignore[arg-type]
        metrics,
    )
    reason = {
        "promoted": "secondary_reward_improvement",
        "rejected": "no_net_improvement",
        "invalid_infra": "infra_sensitive_verdict",
    }[comparison[3]]
    _assert_fields(
        decision,
        outcome=comparison[3],
        reason=reason,
    )
    rewards = {reward.name: reward for reward in decision.secondary_rewards}
    for name, expected_outcome in prior.items():
        assert rewards[name].outcome == expected_outcome
    _assert_fields(
        rewards[reward_name],
        baseline_value=comparison[0],
        candidate_value=comparison[1],
        outcome=comparison[2],
    )


def test_invalid_first_attempts_uses_common_scorable_task_set() -> None:
    comparison = GENERIC_SECONDARY_METRICS[0].compare(
        baseline=_experiment("baseline", validity={"a": (10, 10), "b": (0, 10)}),
        candidate=_experiment(
            "candidate", validity={"b": (5, 10)}, failure_modes={"a": "crash"}
        ),
    )
    _assert_fields(comparison, baseline_value=10, candidate_value=5)


def test_secondary_metric_scorability_ignores_error_text() -> None:
    rollout = _rollout("a", solved=False, validity=(1, 1)).model_copy(
        update={"error": "diagnostic detail"}
    )
    assert GENERIC_SECONDARY_METRICS[0]._scorable(rollout)


def test_invalid_first_attempts_rejects_half_published_metric_pair() -> None:
    baseline = _experiment("baseline", validity={"a": (1, 1)})
    del baseline.tasks["a"].metrics["first_attempt_total"]  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="first_attempt"):
        GENERIC_SECONDARY_METRICS[0].values(
            baseline=baseline,
            candidate=_experiment(
                "candidate", failure_modes={"a": "verified_rejected"}
            ),
        )


def test_f2p_metric_is_not_owned_by_the_loss_module() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import sys\nimport src.trainer.loss\nassert "src.env.swe" not in sys.modules',
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _bare_loss(value: float) -> Loss:
    return Loss(
        value,
        "strict_pareto",
        SimpleNamespace(git_commit_hash="candidate-commit"),  # type: ignore[arg-type]
        SimpleNamespace(git_commit_hash="baseline-commit"),  # type: ignore[arg-type]
        _StubHarness(),  # type: ignore[arg-type]
    )


def test_loss_casts_and_orders_like_scalar() -> None:
    assert float(_bare_loss(-0.5)) == -0.5


def test_loss_backward_deposits_loss_on_harness_grad() -> None:
    loss = _bare_loss(-1.0)
    loss.backward()
    assert loss.harness.grad is loss


def test_result_decision_follows_applied_selection_not_loss_sign() -> None:
    loss = _criterion()(
        baseline=_experiment(
            "baseline", solved=("a",), missing=("b",), crash_reason="panel"
        ),
        candidate=_experiment("candidate", solved=("a", "b")),
    )
    assert loss.value < 0
    loss.harness.applied = "abc123"
    assert StrictPareto().decision(loss).outcome == "rejected"


def test_standalone_criterion_judges_without_trainer_binding() -> None:
    loss = StrictPareto()(
        baseline=_experiment(
            "baseline", solved=("a",), failure_modes={"b": "verified_rejected"}
        ),
        candidate=_experiment("candidate", solved=("a", "b")),
    )
    assert loss.value < 0
    assert loss.report.new_solves == {"b"}
    with pytest.raises(RuntimeError, match="unbound"):
        loss.backward()


def test_promoted_is_undefined_before_step() -> None:
    loss = _bare_loss(-1.0)
    with pytest.raises(RuntimeError, match="optimizer.step"):
        loss.promoted


def test_promoted_reflects_applied_selection() -> None:
    loss = _bare_loss(-1.0)
    loss.harness.applied = "candidate-commit"
    assert loss.promoted
    loss.harness.applied = "abc123"
    assert not loss.promoted


class _OutcomeMetric(SecondaryRewardMetric):
    higher_is_better = True

    def __init__(self, name: str, outcome: SecondaryRewardOutcome) -> None:
        self.name, self.outcome = name, outcome

    def values(
        self, *, baseline: ExperimentResult, candidate: ExperimentResult
    ) -> tuple[None, None]:
        raise NotImplementedError("compare is overridden")

    def compare(
        self, *, baseline: ExperimentResult, candidate: ExperimentResult
    ) -> SecondaryRewardComparison:
        return SecondaryRewardComparison(
            name=self.name, baseline_value=0, candidate_value=0, outcome=self.outcome
        )


@pytest.mark.parametrize(
    ("baseline", "candidate", "outcomes", "value", "reward_names", "verdict"),
    [
        pytest.param(
            dict(solved=("a",)),
            dict(missing=("a",), crash_reason="worker exited"),
            None,
            inf,
            None,
            "rejected",
            id="regression-is-infinity",
        ),
        pytest.param(
            dict(solved=("a",), missing=("b", "c"), crash_reason="panel"),
            dict(solved=("a", "b", "c")),
            None,
            -2.0,
            None,
            "promoted",
            id="solve-delta-is-negative",
        ),
        pytest.param(
            dict(solved=("a",)),
            dict(solved=("a",)),
            ("tied", "candidate_better"),
            -0.25,
            None,
            "promoted",
            id="first-candidate-better",
        ),
        pytest.param(
            dict(solved=("a",)),
            dict(solved=("a",)),
            ("baseline_better", "candidate_better"),
            0.5,
            ["first"],
            "rejected",
            id="baseline-better-short-circuits",
        ),
        pytest.param(
            dict(solved=("a",)),
            dict(solved=("a",)),
            ("tied", "unavailable"),
            0.0,
            None,
            "rejected",
            id="all-nondecisive",
        ),
    ],
)
def test_scalarization_classes(
    baseline: dict[str, object],
    candidate: dict[str, object],
    outcomes: tuple[SecondaryRewardOutcome, ...] | None,
    value: float,
    reward_names: list[str] | None,
    verdict: str,
) -> None:
    metrics = (
        GENERIC_SECONDARY_METRICS
        if outcomes is None
        else tuple(
            _OutcomeMetric(name, outcome)
            for name, outcome in zip(("first", "second"), outcomes, strict=True)
        )
    )
    loss = _criterion(metrics=metrics)(
        baseline=_experiment("baseline", **baseline),  # type: ignore[arg-type]
        candidate=_experiment("candidate", **candidate),  # type: ignore[arg-type]
    )
    assert loss.value == value
    if reward_names is not None:
        assert [reward.name for reward in loss.report.secondary_rewards] == reward_names
    assert _greedy_decision(loss).outcome == verdict


def test_unmeasurable_run_carries_pessimistic_report_and_sensitive_tasks() -> None:
    baseline = _experiment(
        "baseline", solved=("a",), missing=("b",), crash_reason="panel"
    )
    candidate = _experiment(
        "candidate", solved=("b",), failure_modes={"a": "verify_timeout"}
    )
    with pytest.raises(UnmeasurableRun) as raised:
        _criterion()(baseline=baseline, candidate=candidate)
    _assert_fields(
        raised.value.decision,
        outcome="invalid_infra",
        reason="infra_sensitive_verdict",
        regressions=[],
        invalid_infra_tasks=["a"],
        criterion="strict_pareto",
        candidate_solved=["b"],
        baseline_solved=["a"],
    )


def test_dual_eval_agreement_path_forgives_sensitive_regressions() -> None:
    loss = _criterion()(
        baseline=_experiment("baseline", solved=("a", "b")),
        candidate=_experiment(
            "candidate", solved=("b",), failure_modes={"a": "verify_timeout"}
        ),
    )
    _assert_fields(
        _greedy_decision(loss),
        outcome="rejected",
        reason="no_net_improvement",
        regressions=[],
        invalid_infra_tasks=["a"],
        criterion="strict_pareto",
    )


def test_result_decision_preserves_report_facts_and_persisted_shape() -> None:
    decision = _decision(
        _experiment(
            "baseline-1", solved=("task-a", "task-b"), steps={"task-a": 10, "task-b": 8}
        ),
        _experiment(
            "candidate-1", solved=("task-a", "task-b"), steps={"task-a": 6, "task-b": 7}
        ),
    )
    assert decision.model_dump(mode="json") == {
        "outcome": "promoted",
        "reason": "secondary_reward_improvement",
        "candidate_solved": ["task-a", "task-b"],
        "baseline_solved": ["task-a", "task-b"],
        "new_solves": [],
        "regressions": [],
        "secondary_rewards": [
            {
                "name": "invalid_first_attempts",
                "baseline_value": None,
                "candidate_value": None,
                "outcome": "unavailable",
            },
            {
                "name": "steps_used",
                "baseline_value": 18,
                "candidate_value": 13,
                "outcome": "candidate_better",
            },
        ],
        "invalid_infra_tasks": [],
        "criterion": "strict_pareto",
    }


def test_describe_names_criterion_and_active_tiebreakers() -> None:
    assert StrictPareto().describe() == "strict_pareto · tiebreak: none"
    assert _criterion().describe() == (
        "strict_pareto · tiebreak: invalid_first_attempts, steps_used"
    )
