from __future__ import annotations

import pytest

from src.metrics import (
    BaselineComparison,
    TaskMetrics,
    compare_candidate_against_baseline,
    compute_binomial_p_value,
    is_majority_decided,
)


def test_compute_binomial_p_value_matches_regex_log_pool():
    # Matches the Q3 hand-computation: p_hat≈0.475, n=3, observed=1 → p≈0.55.
    p = compute_binomial_p_value(observed_solved=1, observed_total=3, p_hat=0.475)
    assert 0.5 < p <= 1.0
    assert p == pytest.approx(0.99, abs=0.05)
    # Sanity: pmf(0..3) at p=0.475
    # P(X<=1) = (1-p)^3 + 3*p*(1-p)^2 ≈ 0.145 + 0.394 = 0.539
    # P(X>=1) = 1 - P(X=0) ≈ 1 - 0.145 = 0.855
    # 2 * min(0.539, 0.855) = 1.08 → clamped to 1.0
    assert p == pytest.approx(1.0, abs=0.05)


def test_compute_binomial_p_value_extreme_low_tail():
    # Matches the exp-047 openssl hand-computation: p_hat=0.73, n=3, observed=0
    # P(X=0) = 0.27^3 ≈ 0.0197
    # P(X>=0) = 1.0
    # 2 * min(0.0197, 1.0) ≈ 0.039
    p = compute_binomial_p_value(observed_solved=0, observed_total=3, p_hat=0.73)
    assert p == pytest.approx(0.0394, abs=0.005)


def test_compute_binomial_p_value_perfect_consistency():
    # When observed matches expected, p-value should be ~1 (max consistency).
    p = compute_binomial_p_value(observed_solved=2, observed_total=3, p_hat=2 / 3)
    assert 0.9 <= p <= 1.0


def test_compute_binomial_p_value_validates_inputs():
    with pytest.raises(ValueError):
        compute_binomial_p_value(observed_solved=0, observed_total=0, p_hat=0.5)
    with pytest.raises(ValueError):
        compute_binomial_p_value(observed_solved=4, observed_total=3, p_hat=0.5)
    with pytest.raises(ValueError):
        compute_binomial_p_value(observed_solved=1, observed_total=3, p_hat=1.5)


def test_compare_uncompared_when_candidate_has_no_trials():
    verdict = compare_candidate_against_baseline(
        candidate_solved=0,
        candidate_total=0,
        baseline_solved=3,
        baseline_total=9,
    )
    assert verdict.kind == "uncompared"
    assert verdict.p_value is None
    assert verdict.candidate_rate is None
    assert verdict.baseline_rate == pytest.approx(3 / 9)


def test_compare_uncompared_when_both_empty():
    verdict = compare_candidate_against_baseline(
        candidate_solved=0,
        candidate_total=0,
        baseline_solved=0,
        baseline_total=0,
    )
    assert verdict.kind == "uncompared"
    assert verdict.p_value is None
    assert verdict.candidate_rate is None
    assert verdict.baseline_rate is None


def test_compare_no_baseline_frontier_majority_solve_is_improvement():
    # No prior baseline samples; candidate majority-solves the task.
    # 3/4 clears the ceil(4/2) = 2 majority threshold.
    verdict = compare_candidate_against_baseline(
        candidate_solved=3,
        candidate_total=4,
        baseline_solved=0,
        baseline_total=0,
    )
    assert verdict.kind == "improvement"
    assert verdict.p_value is None
    assert verdict.baseline_rate is None


def test_compare_no_baseline_frontier_no_majority_is_unchanged():
    # No prior baseline samples; candidate did not majority-solve.
    # 1/4 falls below the ceil(4/2) = 2 majority threshold.
    verdict = compare_candidate_against_baseline(
        candidate_solved=1,
        candidate_total=4,
        baseline_solved=0,
        baseline_total=0,
    )
    assert verdict.kind == "unchanged"
    assert verdict.p_value is None


def test_compare_no_baseline_frontier_first_solve_ever_is_improvement():
    # Smallest possible no-baseline frontier win: 1/1 solve.
    verdict = compare_candidate_against_baseline(
        candidate_solved=1,
        candidate_total=1,
        baseline_solved=0,
        baseline_total=0,
    )
    assert verdict.kind == "improvement"


def test_compare_baseline_100pct_active_only_flags_regression():
    # Active-baseline-only pattern that drove the exp-006 false positive.
    # The function does what it's told: 3/3 → 3/4 is significant at alpha=0.05.
    # The fix lives at the caller (wider pool); this test pins the math.
    verdict = compare_candidate_against_baseline(
        candidate_solved=3,
        candidate_total=4,
        baseline_solved=3,
        baseline_total=3,
    )
    assert verdict.kind == "regression"
    assert verdict.p_value is not None
    assert verdict.p_value < 0.05


def test_compare_baseline_widened_pool_neutralizes_noise():
    # Same candidate trial counts as the previous test, but the baseline pool
    # now includes uncontaminated history. 3/4 candidate vs 3/9 pool comes out
    # as "unchanged" — exactly the headless-terminal fix.
    verdict = compare_candidate_against_baseline(
        candidate_solved=3,
        candidate_total=4,
        baseline_solved=3,
        baseline_total=9,
    )
    assert verdict.kind == "unchanged"
    assert verdict.p_value is not None
    assert verdict.p_value > 0.05


def test_compare_clear_regression():
    # Baseline 4/4, candidate 0/5: p-value ~= 0, candidate rate well below.
    verdict = compare_candidate_against_baseline(
        candidate_solved=0,
        candidate_total=5,
        baseline_solved=4,
        baseline_total=4,
    )
    assert verdict.kind == "regression"
    assert verdict.p_value == pytest.approx(0.0, abs=1e-9)


def test_compare_clear_improvement():
    # Baseline 3/9 (~33%), candidate 4/4 (100%): rate above, p < 0.05.
    verdict = compare_candidate_against_baseline(
        candidate_solved=4,
        candidate_total=4,
        baseline_solved=3,
        baseline_total=9,
    )
    assert verdict.kind == "improvement"
    assert verdict.p_value is not None
    assert verdict.p_value < 0.05


def test_compare_zero_baseline_with_history_requires_majority_for_improvement():
    # Pool has 9 trials, none solved (the configure-git-webserver pattern).
    # Candidate solves once: under the new rule this stays "unchanged"
    # rather than triggering a noisy single-solve improvement. p_value is
    # None because no binomial test was run.
    verdict = compare_candidate_against_baseline(
        candidate_solved=1,
        candidate_total=5,
        baseline_solved=0,
        baseline_total=9,
    )
    assert verdict.kind == "unchanged"
    assert verdict.p_value is None


def test_compare_zero_baseline_with_history_majority_solve_is_improvement():
    # Same zero-rate pool; candidate now majority-solves (3/5 clears
    # ceil(5/2) = 3). Counts as improvement.
    verdict = compare_candidate_against_baseline(
        candidate_solved=3,
        candidate_total=5,
        baseline_solved=0,
        baseline_total=9,
    )
    assert verdict.kind == "improvement"
    assert verdict.p_value is None


def test_compare_zero_rate_on_both_sides_is_unchanged():
    # 0/4 candidate, 0/5 baseline — both 0%. Zero-baseline branch fires,
    # no majority-solve on the candidate side, kind=unchanged, p=None.
    verdict = compare_candidate_against_baseline(
        candidate_solved=0,
        candidate_total=4,
        baseline_solved=0,
        baseline_total=5,
    )
    assert verdict.kind == "unchanged"
    assert verdict.p_value is None


def test_compare_zero_baseline_cannot_regress():
    # Candidate underperforming a 0%-rate baseline is impossible; verdict
    # should be "unchanged", never "regression".
    verdict = compare_candidate_against_baseline(
        candidate_solved=0,
        candidate_total=8,
        baseline_solved=0,
        baseline_total=12,
    )
    assert verdict.kind == "unchanged"


def test_compare_alpha_boundary_just_below():
    # k=1, n=10, p_hat=0.5 → 2 * P(X<=1) = 2 * 11/1024 ≈ 0.0215 < 0.05.
    verdict = compare_candidate_against_baseline(
        candidate_solved=1,
        candidate_total=10,
        baseline_solved=1,
        baseline_total=2,
    )
    assert verdict.p_value is not None
    assert verdict.p_value < 0.05
    assert verdict.kind == "regression"


def test_compare_alpha_boundary_just_above():
    # k=2, n=10, p_hat=0.5 → 2 * P(X<=2) ≈ 0.1094 > 0.05.
    verdict = compare_candidate_against_baseline(
        candidate_solved=2,
        candidate_total=10,
        baseline_solved=1,
        baseline_total=2,
    )
    assert verdict.p_value is not None
    assert verdict.p_value > 0.05
    assert verdict.kind == "unchanged"


def test_compare_alpha_parameter_overrides_default():
    # Same numbers as the boundary tests above, but a stricter alpha turns
    # a borderline regression into "unchanged".
    verdict_default = compare_candidate_against_baseline(
        candidate_solved=1,
        candidate_total=10,
        baseline_solved=1,
        baseline_total=2,
    )
    verdict_strict = compare_candidate_against_baseline(
        candidate_solved=1,
        candidate_total=10,
        baseline_solved=1,
        baseline_total=2,
        alpha=0.01,
    )
    assert verdict_default.kind == "regression"
    assert verdict_strict.kind == "unchanged"
    assert verdict_default.p_value == verdict_strict.p_value


def test_compare_validates_negative_counts():
    with pytest.raises(ValueError):
        compare_candidate_against_baseline(
            candidate_solved=-1,
            candidate_total=4,
            baseline_solved=0,
            baseline_total=0,
        )
    with pytest.raises(ValueError):
        compare_candidate_against_baseline(
            candidate_solved=0,
            candidate_total=4,
            baseline_solved=0,
            baseline_total=-1,
        )


def test_compare_validates_solved_within_total():
    with pytest.raises(ValueError):
        compare_candidate_against_baseline(
            candidate_solved=5,
            candidate_total=4,
            baseline_solved=0,
            baseline_total=0,
        )
    with pytest.raises(ValueError):
        compare_candidate_against_baseline(
            candidate_solved=0,
            candidate_total=4,
            baseline_solved=10,
            baseline_total=9,
        )


def test_compare_validates_alpha_range():
    with pytest.raises(ValueError):
        compare_candidate_against_baseline(
            candidate_solved=1,
            candidate_total=4,
            baseline_solved=1,
            baseline_total=4,
            alpha=0.0,
        )
    with pytest.raises(ValueError):
        compare_candidate_against_baseline(
            candidate_solved=1,
            candidate_total=4,
            baseline_solved=1,
            baseline_total=4,
            alpha=1.5,
        )


def test_compare_returns_frozen_dataclass():
    verdict = compare_candidate_against_baseline(
        candidate_solved=1,
        candidate_total=4,
        baseline_solved=0,
        baseline_total=0,
    )
    assert isinstance(verdict, BaselineComparison)
    with pytest.raises(AttributeError):
        verdict.kind = "regression"  # type: ignore[misc]


def test_is_majority_decided():
    # k=3: outcome locks in once one side reaches ceil(3/2)=2 or can no longer reach it.
    assert is_majority_decided(solved=0, finished=0, expected_total=3) is False
    assert is_majority_decided(solved=1, finished=1, expected_total=3) is False
    assert is_majority_decided(solved=2, finished=2, expected_total=3) is True
    assert is_majority_decided(solved=0, finished=2, expected_total=3) is True
    assert is_majority_decided(solved=1, finished=2, expected_total=3) is False
    # k=1: any single trial is the final word.
    assert is_majority_decided(solved=1, finished=1, expected_total=1) is True
    # finished >= expected_total is always decided.
    assert is_majority_decided(solved=0, finished=3, expected_total=3) is True


def test_task_metrics_owned_by_metrics_module():
    assert TaskMetrics.__module__ == "src.metrics"
