"""The graded promotion statistic (Phase 1 step 2): unit + history + simulation.

Three layers, all pure (no git/docker/I/O):

- **Unit**: ``GradedRewardTest.from_task_deltas`` edge cases + a hand-computed
  z, the ``_trial_graded_reward`` binary fallback, and the ``_graded_task_deltas``
  both-arms / valid-trials strata builder.
- **History regression**: the real per-task delta vectors of the one historical
  KEEP (045348) and the borderline discard (151401) -- baked as snapshots so the
  test is self-contained -- must both fail to promote. 045348 is the decisive
  case: the binary CMH promoted it (winner's curse), the graded statistic must
  not.
- **Calibration + power**: a seeded resampling simulation against a representative
  synthetic per-task reward population -- null false-keep <= alpha, and power
  against a sub-threshold +reward effect materially above the binary CMH (which
  is blind to partial-credit movement).
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from src.experiment.record import ExperimentResult, TaskResult, TrialResult
from src.supervisor.policy import (
    GRADED_PROMOTION_P_VALUE_ALPHA,
    GradedRewardTest,
    StratifiedSolveTest,
    _graded_task_deltas,
    _trial_graded_reward,
)

ALPHA = GRADED_PROMOTION_P_VALUE_ALPHA


# --- builders ---------------------------------------------------------------


def _trial(run_id: str, *, solved: bool, reward: float | None = None) -> TrialResult:
    return TrialResult(
        run_id=run_id,
        solved=solved,
        failure_mode="solved" if solved else "verified_rejected",
        verifier_passed=solved,
        reward=reward,
    )


def _task(
    rewards: list[float | None], *, solved: list[bool] | None = None
) -> TaskResult:
    solved = solved if solved is not None else [r == 1.0 for r in rewards]
    trials = [
        _trial(f"r{i}", solved=s, reward=r)
        for i, (r, s) in enumerate(zip(rewards, solved, strict=True))
    ]
    return TaskResult(expected_trial_count=len(trials), trials=trials)


def _exp(tasks: dict[str, TaskResult]) -> ExperimentResult:
    return ExperimentResult(
        experiment_id="exp",
        git_commit_hash="head",
        run_status="completed",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T01:00:00+00:00",
        tasks=tasks,
    )


# --- GradedRewardTest.from_task_deltas --------------------------------------


def test_no_comparable_tasks_is_no_evidence() -> None:
    test = GradedRewardTest.from_task_deltas([])
    assert test == GradedRewardTest(mean_delta=0.0, p_value=1.0, task_count=0)


def test_single_task_has_no_spread_estimate() -> None:
    # One comparable task: a real delta but no way to estimate its spread, so the
    # statistic refuses to claim significance (p_value 1.0 -> never promotes).
    test = GradedRewardTest.from_task_deltas([0.5])
    assert test.task_count == 1
    assert test.mean_delta == pytest.approx(0.5)
    assert test.p_value == 1.0


def test_unanimous_positive_delta_is_decisive() -> None:
    # Every task improved by the same positive amount: zero spread, unanimous
    # direction -> p_value 0.0 (the only degenerate case that promotes).
    test = GradedRewardTest.from_task_deltas([0.1, 0.1, 0.1, 0.1])
    assert test.mean_delta == pytest.approx(0.1)
    assert test.p_value == 0.0


def test_unanimous_nonpositive_delta_never_promotes() -> None:
    assert GradedRewardTest.from_task_deltas([-0.2, -0.2, -0.2]).p_value == 1.0
    # All-zero deltas: no movement, not an improvement.
    assert GradedRewardTest.from_task_deltas([0.0, 0.0, 0.0]).p_value == 1.0


def test_p_value_matches_hand_computed_normal_approx() -> None:
    # deltas [0.0, 0.2, 0.4]: mean 0.2, sd 0.2, se 0.2/sqrt(3), z = 1.732,
    # one-sided p = 1 - Phi(z) ~= 0.0416.
    test = GradedRewardTest.from_task_deltas([0.0, 0.2, 0.4])
    z = 0.2 / (0.2 / math.sqrt(3))
    expected = 0.5 * math.erfc(z / math.sqrt(2.0))
    assert test.mean_delta == pytest.approx(0.2)
    assert test.p_value == pytest.approx(expected)
    assert test.p_value == pytest.approx(0.0416, abs=1e-3)


def test_negative_mean_delta_is_not_an_improvement() -> None:
    test = GradedRewardTest.from_task_deltas([-0.1, 0.0, -0.3, 0.1])
    assert test.mean_delta < 0
    assert test.p_value > 0.5  # wrong direction -> never near significance


# --- _trial_graded_reward (binary fallback) ---------------------------------


def test_graded_reward_uses_recorded_fraction() -> None:
    assert _trial_graded_reward(_trial("r", solved=False, reward=0.6)) == 0.6
    assert _trial_graded_reward(_trial("r", solved=True, reward=1.0)) == 1.0


def test_graded_reward_falls_back_to_binary_when_unrecorded() -> None:
    # Old records (and no-CTRF trials) carry reward=None -> 1.0/0.0 by solved.
    assert _trial_graded_reward(_trial("r", solved=True, reward=None)) == 1.0
    assert _trial_graded_reward(_trial("r", solved=False, reward=None)) == 0.0


# --- _graded_task_deltas (strata builder) -----------------------------------


def test_strata_use_both_arms_valid_trials_only() -> None:
    candidate = _exp(
        {
            "a": _task([1.0, 1.0, 0.8]),
            "b": _task([0.5, 0.5]),  # baseline never ran b -> dropped
        }
    )
    baseline = _exp(
        {
            "a": _task([0.8, 0.6, 0.4]),
            "c": _task([0.2]),  # not in task_ids -> irrelevant
        }
    )
    deltas = _graded_task_deltas(
        candidate=candidate, baseline=baseline, task_ids=frozenset({"a", "b"})
    )
    # Only task "a" is in both arms AND task_ids ("b" is candidate-only): the
    # candidate mean (2.8/3) minus the baseline mean (1.8/3) = 0.3333.
    assert deltas == pytest.approx([2.8 / 3 - 1.8 / 3])


def test_strata_exclude_crash_trials() -> None:
    crash = TrialResult(run_id="c", solved=False, failure_mode="crash", error="boom")
    cand_task = TaskResult(
        expected_trial_count=3,
        trials=[_trial("r0", solved=True, reward=1.0), crash],
    )
    candidate = _exp({"a": cand_task})
    baseline = _exp({"a": _task([0.0, 0.0])})
    deltas = _graded_task_deltas(
        candidate=candidate, baseline=baseline, task_ids=frozenset({"a"})
    )
    # The crash trial is excluded: candidate mean is 1.0 (one valid trial).
    assert deltas == pytest.approx([1.0])


# --- history regression (real per-task delta vectors; self-contained) -------

# Per-task graded-reward deltas (candidate mean - baseline mean) over the 59
# train tasks, recovered from the on-disk CTRF reports. Snapshots so the test
# does not depend on experiments/ artifacts that may be archived.

# exp-20260612-045348 vs baseline exp-20260611-235638: the ONE historical KEEP.
# Binary CMH promoted it (one stratum, pytorch-model-recovery, drove it on a
# lucky baseline draw); the graded statistic must not.
KEEP_045348_DELTAS = [
    0.027778,
    0.0,
    0.0,
    0.0,
    0.083333,
    -0.016667,
    -0.15,
    0.0,
    0.25,
    0.0,
    0.0,
    0.0,
    0.25,
    0.0,
    0.2,
    -0.15,
    0.0,
    0.0,
    0.0,
    -0.25,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.071429,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.177778,
    0.0,
    0.0,
    0.0,
    0.291667,
    -0.125,
    0.0,
    0.0,
    0.0,
    0.25,
    0.0,
    0.0,
    0.25,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.733333,
    0.0,
    -0.25,
    -0.296296,
    0.0,
    0.0,
    0.25,
    0.0,
    0.166667,
    0.0,
]

# exp-20260612-151401 vs baseline exp-20260612-045348: a discard with the
# largest positive aggregate among the discards -- the case the analytic
# Welch-sum spuriously kept. Confirms the task-as-unit statistic does not.
DISCARD_151401_DELTAS = [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.208333,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.25,
    0.0,
    0.0,
    0.0,
    0.0,
    0.75,
    -0.3,
    0.0,
    0.0,
    0.25,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.05,
    -0.05,
    0.0,
    0.0,
    0.694444,
    0.0,
    0.0,
    0.0,
    -0.125,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.0625,
    0.0,
    -0.1,
    0.0,
    0.0,
    0.0,
    0.0625,
    0.0,
    0.0,
    -0.2,
    0.0,
    0.25,
    0.518519,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.194444,
    0.0,
]


def test_graded_statistic_does_not_keep_the_historical_false_keep() -> None:
    test = GradedRewardTest.from_task_deltas(KEEP_045348_DELTAS)
    # Mildly positive aggregate (~+0.024) but not significant against the panel
    # spread: the one big stratum (pytorch +0.73) does not carry 59 tasks.
    assert test.mean_delta > 0  # direction is positive (so it is a real margin call)
    assert test.p_value > ALPHA, f"would keep 045348 at p={test.p_value:.4f}"


def test_graded_statistic_does_not_spuriously_keep_borderline_discard() -> None:
    test = GradedRewardTest.from_task_deltas(DISCARD_151401_DELTAS)
    assert test.p_value > ALPHA, f"would keep 151401 at p={test.p_value:.4f}"


# --- calibration + power simulation (representative synthetic population) ----

# A per-task reward population mirroring the real panel's structure: ~half
# deterministic-solved (always 1.0, no gradient), a partial-credit middle band
# (the tasks graded reward actually discriminates), and hard near-zero tasks.
_DETERMINISTIC = [1.0, 1.0, 1.0, 1.0, 1.0]
_PARTIAL_BANDS = [
    [1.0, 0.8, 1.0, 0.6, 0.8],
    [0.8, 0.6, 0.8, 1.0, 0.4],
    [0.6, 0.4, 0.8, 0.6, 0.6],
    [1.0, 1.0, 0.6, 0.8, 0.4],
]
_HARD = [0.0, 0.0, 0.2, 0.0, 0.0]


def _population() -> list[list[float]]:
    pop: list[list[float]] = []
    for i in range(59):
        if i < 30:
            pop.append(_DETERMINISTIC)
        elif i < 50:
            pop.append(_PARTIAL_BANDS[i % len(_PARTIAL_BANDS)])
        else:
            pop.append(_HARD)
    return pop


POP = _population()
FULL = 5
CYCLES = 600
SEED = 20260613


def _draw(rng: random.Random, pop: list[float], effect: float) -> list[float]:
    return [min(1.0, rng.choice(pop) + effect) for _ in range(FULL)]


def _graded_keep(rng: random.Random, effect: float) -> bool:
    deltas = []
    for pop in POP:
        cand = _draw(rng, pop, effect)
        base = _draw(rng, pop, 0.0)
        deltas.append(statistics.fmean(cand) - statistics.fmean(base))
    test = GradedRewardTest.from_task_deltas(deltas)
    return test.mean_delta > 0 and test.p_value <= ALPHA


def _binary_keep(rng: random.Random, effect: float) -> bool:
    # The binary gate's promotion test on the same draws: solve == full pass.
    strata = []
    for pop in POP:
        cand = _draw(rng, pop, effect)
        base = _draw(rng, pop, 0.0)
        strata.append(
            (
                sum(1 for x in cand if x >= 1.0),
                FULL,
                sum(1 for x in base if x >= 1.0),
                FULL,
            )
        )
    test = StratifiedSolveTest.from_strata(strata)
    return test.delta > 0 and test.p_value <= ALPHA


def _keep_rate(keep_fn, effect: float) -> float:
    rng = random.Random(SEED)
    return sum(keep_fn(rng, effect) for _ in range(CYCLES)) / CYCLES


def test_graded_null_false_keep_within_alpha() -> None:
    # Inert candidate (same reward population both arms): false-keep <= alpha.
    null_rate = _keep_rate(_graded_keep, 0.0)
    assert null_rate <= ALPHA + 0.03, f"graded null false-keep {null_rate:.3f}"


def test_graded_power_exceeds_binary_against_subthreshold_effect() -> None:
    # A +0.05 per-trial reward lift sits mostly below the binary pass threshold,
    # so the binary CMH barely moves while the graded statistic captures it.
    graded_power = _keep_rate(_graded_keep, 0.05)
    binary_power = _keep_rate(_binary_keep, 0.05)
    assert graded_power >= 0.20, f"graded power {graded_power:.3f} too low"
    assert graded_power > 2 * binary_power, (
        f"graded power {graded_power:.3f} not materially above binary {binary_power:.3f}"
    )
