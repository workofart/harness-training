"""Promotion gate acceptance: resampling simulation against the real panel.

Pins the live graded promotion gate's operating characteristics on the actual
panel it runs on -- the finalized SWE headroom panel (39 rich+thin tasks). The
per-task reward vectors are a snapshot of the characterization run
exp-20260614-221243 (valid-trial graded rewards), baked here so the test is
self-contained. Each simulated cycle resamples a candidate and a baseline arm
(full=3 trials, with replacement) from each task's reward vector and runs the
real ``GradedRewardTest`` at the production ``GRADED_PROMOTION_P_VALUE_ALPHA``.

Two properties are pinned:

- **Null calibration**: an inert candidate (drawn from the same reward
  population as the baseline) false-keeps at most the nominal alpha. This is the
  calibration that justified cutting the gate over to graded-sole on this panel
  (and tuning alpha to 0.08): the binary-CMH-era conjunction was needed only on
  the old ceiling-heavy Terminal-Bench panel, whose zero-variance deltas ran the
  one-sample test hot (~0.16). Cutting the ceiling tasks brings it to ~0.08.
- **Power**: a small per-task reward lift -- the partial-credit movement the
  retired binary gate was blind to -- promotes at a materially higher rate.

The RNG is seeded, so the measured rates are deterministic.
"""

from __future__ import annotations

import random
import statistics

from src.supervisor.policy import (
    GRADED_PROMOTION_P_VALUE_ALPHA,
    GradedRewardTest,
)

# Per-task valid-trial graded rewards over the finalized headroom panel
# (snapshot of exp-20260614-221243; the population each arm is resampled from).
CHAR_REWARDS: dict[str, list[float]] = {
    "django__django-10973": [0.8, 0.8],
    "django__django-10999": [0.8333, 0.8333],
    "django__django-11087": [0.9762, 0.9762],
    "django__django-11477": [0.987, 0.987],
    "django__django-11490": [0.9583, 1.0, 0.9583],
    "django__django-11728": [0.9565, 0.9565],
    "django__django-14017": [0.9933, 0.9933],
    "django__django-14034": [0.9231, 0.9231],
    "django__django-14155": [0.9674, 0.9674],
    "django__django-14170": [0.8846, 0.8846],
    "django__django-14315": [0.6364, 0.8182],
    "django__django-14376": [0.6667, 0.6667],
    "django__django-14534": [0.9917, 1.0, 1.0],
    "django__django-14792": [0.92, 0.92],
    "pylint-dev__pylint-4970": [0.9444, 0.9444],
    "pylint-dev__pylint-6386": [0.875, 1.0, 0.875],
    "pylint-dev__pylint-6528": [1.0, 0.9943, 1.0],
    "pylint-dev__pylint-7080": [0.9917, 0.9917],
    "pytest-dev__pytest-10051": [0.875, 1.0, 1.0],
    "pytest-dev__pytest-10081": [1.0, 0.9844, 1.0],
    "pytest-dev__pytest-5840": [0.9623, 0.9623],
    "pytest-dev__pytest-7205": [1.0, 0.6538, 0.6538],
    "pytest-dev__pytest-7490": [0.9875, 1.0, 0.9875],
    "sphinx-doc__sphinx-10323": [0.9756, 0.9756],
    "sphinx-doc__sphinx-10435": [0.9733, 0.9867],
    "sphinx-doc__sphinx-10673": [0.9, 0.5],
    "sphinx-doc__sphinx-7462": [0.9608, 0.9608],
    "sphinx-doc__sphinx-7748": [0.7857, 0.0],
    "sphinx-doc__sphinx-8056": [0.9756, 1.0, 0.9756],
    "sphinx-doc__sphinx-9281": [1.0, 0.9744, 1.0],
    "sphinx-doc__sphinx-9673": [0.92, 0.92],
    "sympy__sympy-12419": [0.9615, 1.0, 1.0],
    "sympy__sympy-13091": [0.989, 0.9451],
    "sympy__sympy-13798": [0.9903, 0.9903],
    "sympy__sympy-13974": [0.8, 0.8],
    "sympy__sympy-17318": [0.9091, 0.9091],
    "sympy__sympy-18763": [0.993, 0.993],
    "sympy__sympy-23950": [0.8, 0.8],
    "sympy__sympy-24443": [0.0, 1.0, 1.0],
}

FULL_TRIAL_COUNT = 3  # config task_trials for the finalized panel
CYCLES = 2000
SEED = 20260614


def _draw(rng: random.Random, rewards: list[float], effect: float) -> list[float]:
    # full=3 trials sampled with replacement; a partial-credit effect lifts each
    # drawn reward (capped at 1.0 -- a ceiling reward cannot move).
    return [min(1.0, rng.choice(rewards) + effect) for _ in range(FULL_TRIAL_COUNT)]


def _simulated_keep(rng: random.Random, effect: float) -> bool:
    deltas = [
        statistics.fmean(_draw(rng, rewards, effect))
        - statistics.fmean(_draw(rng, rewards, 0.0))
        for rewards in CHAR_REWARDS.values()
    ]
    test = GradedRewardTest.from_task_deltas(deltas)
    return test.mean_delta > 0 and test.p_value <= GRADED_PROMOTION_P_VALUE_ALPHA


def _keep_rate(effect: float) -> float:
    rng = random.Random(SEED)
    return sum(_simulated_keep(rng, effect) for _ in range(CYCLES)) / CYCLES


def test_null_false_keep_rate_is_at_most_alpha() -> None:
    # Inert candidate (same reward population both arms): the chance the gate
    # promotes panel noise stays within the nominal alpha. Measured ~0.079 at
    # this seed -- the calibration behind graded-sole + alpha 0.08 on this panel.
    null_rate = _keep_rate(0.0)
    assert null_rate <= 0.10, f"null false-keep rate {null_rate:.3f} > 0.10"


def test_small_partial_credit_effect_promotes_at_a_material_rate() -> None:
    # A +0.05 per-task reward lift -- movement that mostly does NOT cross the
    # binary solve threshold, so the retired CMH was near-blind to it -- promotes
    # at a rate far above the null. Measured ~0.80 at this seed.
    null_rate = _keep_rate(0.0)
    effect_rate = _keep_rate(0.05)
    assert effect_rate >= 0.50, f"+0.05 effect keep rate {effect_rate:.3f} < 0.50"
    assert effect_rate > 3 * null_rate, (
        f"effect rate {effect_rate:.3f} not materially above null {null_rate:.3f}"
    )
