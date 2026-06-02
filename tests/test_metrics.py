from __future__ import annotations

import pytest

from src.metrics import (
    BaselineComparison,
    TaskMetrics,
    compare_candidate_against_baseline,
    compute_fisher_exact_p_value,
    is_majority_decided,
)


def test_compute_fisher_exact_independence_is_one():
    # Identical arms (1/2 vs 1/2) carry no evidence of difference → p == 1.0.
    p = compute_fisher_exact_p_value(
        candidate_solved=1, candidate_total=2, baseline_solved=1, baseline_total=2
    )
    assert p == pytest.approx(1.0)


def test_compute_fisher_exact_perfect_separation_matches_hypergeometric():
    # 4/4 vs 0/4 is the most extreme split for these margins: the two-sided p
    # is the mass of the two tail tables, 2 / C(8, 4) = 2/70 ≈ 0.02857.
    p = compute_fisher_exact_p_value(
        candidate_solved=4, candidate_total=4, baseline_solved=0, baseline_total=4
    )
    assert p == pytest.approx(2 / 70, abs=1e-6)


def test_compute_fisher_exact_small_high_rate_baseline_is_not_significant():
    # The variance-lottery case: 1/4 candidate vs a 3/3 baseline. A one-sample
    # binomial against rate 1.0 would call this p≈0; Fisher, conditioning on
    # both margins, returns a large p so the small baseline is treated as the
    # weak evidence it is.
    p = compute_fisher_exact_p_value(
        candidate_solved=1, candidate_total=4, baseline_solved=3, baseline_total=3
    )
    assert p > 0.05


def test_compute_fisher_exact_validates_inputs():
    with pytest.raises(ValueError):
        compute_fisher_exact_p_value(
            candidate_solved=0, candidate_total=0, baseline_solved=1, baseline_total=2
        )
    with pytest.raises(ValueError):
        compute_fisher_exact_p_value(
            candidate_solved=1, candidate_total=2, baseline_solved=0, baseline_total=0
        )
    with pytest.raises(ValueError):
        compute_fisher_exact_p_value(
            candidate_solved=4, candidate_total=3, baseline_solved=1, baseline_total=2
        )


def test_compare_small_high_rate_baseline_does_not_false_regress():
    # Regression test for the variance-lottery bug. A below-majority candidate
    # dip against a SMALL high-rate baseline must NOT count as a significant
    # regression: a 3/3 baseline is weak evidence of a ~1.0 true rate, so the
    # comparison has to account for baseline sampling uncertainty (two-sample
    # Fisher), not treat the baseline rate as a known point. The old one-sample
    # binomial flagged each of these as a regression and discarded otherwise
    # solve-positive candidates (this session: password-recovery 1/4 vs 3/3,
    # cobol-modernization 2/5 vs 4/4, etc.).
    for candidate_solved, candidate_total, baseline_solved, baseline_total in (
        (1, 4, 3, 3),
        (2, 5, 4, 4),
        (0, 3, 3, 3),
    ):
        verdict = compare_candidate_against_baseline(
            candidate_solved=candidate_solved,
            candidate_total=candidate_total,
            baseline_solved=baseline_solved,
            baseline_total=baseline_total,
        )
        assert verdict.kind == "unchanged", (
            candidate_solved,
            candidate_total,
            baseline_solved,
            baseline_total,
            verdict.kind,
        )


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


def test_compare_moderate_gap_small_sample_is_unchanged():
    # 3/4 candidate vs 3/9 baseline: rate is higher but the two-sample Fisher
    # test does not reach significance at these small counts, so the verdict is
    # "unchanged" rather than a noisy promotion.
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
    # Baseline 4/4, candidate 0/5: the most extreme split for these margins.
    # Two-sided Fisher p ≈ 0.0079, below alpha, candidate rate well below.
    verdict = compare_candidate_against_baseline(
        candidate_solved=0,
        candidate_total=5,
        baseline_solved=4,
        baseline_total=4,
    )
    assert verdict.kind == "regression"
    assert verdict.p_value is not None
    assert verdict.p_value < 0.05


def test_compare_clear_improvement():
    # Baseline 1/6 (~17%), candidate 6/6 (100%): a large gap at enough trials
    # for Fisher to reach significance (p ≈ 0.0152). Note that smaller gaps or
    # counts (e.g. 4/4 vs 3/9) deliberately stay "unchanged" — at n~3-5 only
    # frontier flips and large separations are statistically detectable.
    verdict = compare_candidate_against_baseline(
        candidate_solved=6,
        candidate_total=6,
        baseline_solved=1,
        baseline_total=6,
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
    # 0/4 vs 4/4: two-sided Fisher p ≈ 0.0286 < 0.05, candidate rate below.
    verdict = compare_candidate_against_baseline(
        candidate_solved=0,
        candidate_total=4,
        baseline_solved=4,
        baseline_total=4,
    )
    assert verdict.p_value is not None
    assert verdict.p_value < 0.05
    assert verdict.kind == "regression"


def test_compare_alpha_boundary_just_above():
    # 0/3 vs 3/3: two-sided Fisher p ≈ 0.10 > 0.05 — one fewer trial each side
    # is no longer enough evidence, so the verdict stays "unchanged".
    verdict = compare_candidate_against_baseline(
        candidate_solved=0,
        candidate_total=3,
        baseline_solved=3,
        baseline_total=3,
    )
    assert verdict.p_value is not None
    assert verdict.p_value > 0.05
    assert verdict.kind == "unchanged"


def test_compare_alpha_parameter_overrides_default():
    # 0/4 vs 4/4 (p ≈ 0.0286) is a regression at the default alpha but a
    # stricter alpha downgrades it to "unchanged". p_value is independent of
    # alpha (alpha only sets the decision threshold).
    verdict_default = compare_candidate_against_baseline(
        candidate_solved=0,
        candidate_total=4,
        baseline_solved=4,
        baseline_total=4,
    )
    verdict_strict = compare_candidate_against_baseline(
        candidate_solved=0,
        candidate_total=4,
        baseline_solved=4,
        baseline_total=4,
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
