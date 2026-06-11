"""Promotion gate acceptance: resampling simulation against real baseline rates.

Simulates whole candidate cycles against the frozen per-task solve rates of the
last completed baseline (exp-20260609-025307, 59 train tasks: 28 deterministic /
21 flaky / 10 never-solved) and runs each through the real promotion decision
(``_panel_decision``). Two properties are pinned:

- **Null calibration**: an inert candidate (true per-task rates identical to
  the baseline) false-keeps at most ~alpha over >= 1500 simulated cycles.
- **Power**: a synthetic +5% per-task effect promotes at a materially higher
  rate -- the prior aggregate gate (Fisher over majority-solved counts at
  alpha 0.20, needing +7-8/59 tasks) promoted such an effect at ~0.

Candidate arms mirror the existing budgets (deterministic-solved 1 +
confirm-on-fail expand to full, everything else full; the all-fail early-stop
trim is immaterial -- a 0-solve stratum carries no CMH information either way).
The RNG is seeded, so the measured rates are deterministic.
"""

from __future__ import annotations

import random

from src.supervisor.policy import BaselineComparison, _panel_decision

# Per-task (solved, valid_trials) over the train panel of the last completed
# baseline record, exp-20260609-025307 (snapshot; the simulation's population).
BASELINE_RATES: dict[str, tuple[int, int]] = {
    "openssl-selfsigned-cert": (3, 3),
    "regex-log": (3, 4),
    "fix-git": (3, 3),
    "count-dataset-tokens": (3, 3),
    "log-summary-date-ranges": (3, 4),
    "overfull-hbox": (3, 4),
    "git-multibranch": (3, 5),
    "nginx-request-logging": (3, 3),
    "sqlite-db-truncate": (3, 3),
    "multi-source-data-merger": (3, 3),
    "headless-terminal": (3, 3),
    "large-scale-text-editing": (3, 4),
    "pypi-server": (3, 5),
    "sparql-university": (3, 4),
    "mteb-retrieve": (2, 5),
    "hf-model-inference": (3, 3),
    "code-from-image": (3, 5),
    "cancel-async-tasks": (3, 3),
    "extract-elf": (0, 3),
    "modernize-scientific-stack": (3, 3),
    "password-recovery": (3, 3),
    "sanitize-git-repo": (3, 3),
    "git-leak-recovery": (3, 3),
    "build-pmars": (3, 3),
    "build-pov-ray": (3, 3),
    "sqlite-with-gcov": (3, 3),
    "build-cython-ext": (3, 3),
    "cobol-modernization": (3, 4),
    "pytorch-model-recovery": (3, 4),
    "bn-fit-modify": (3, 3),
    "largest-eigenval": (3, 4),
    "chess-best-move": (2, 5),
    "kv-store-grpc": (3, 3),
    "distribution-search": (3, 3),
    "adaptive-rejection-sampler": (3, 3),
    "portfolio-optimization": (3, 3),
    "constraints-scheduling": (3, 5),
    "merge-diff-arc-agi-task": (3, 3),
    "llm-inference-batching-scheduler": (3, 4),
    "dna-insert": (1, 4),
    "sam-cell-seg": (0, 3),
    "torch-tensor-parallelism": (2, 4),
    "circuit-fibsqrt": (3, 3),
    "prove-plus-comm": (3, 3),
    "polyglot-c-py": (0, 3),
    "polyglot-rust-c": (0, 3),
    "write-compressor": (3, 4),
    "feal-linear-cryptanalysis": (3, 3),
    "custom-memory-heap-crash": (3, 5),
    "financial-document-processor": (2, 5),
    "fix-code-vulnerability": (3, 3),
    "feal-differential-cryptanalysis": (3, 3),
    "mteb-leaderboard": (0, 3),
    "path-tracing": (0, 3),
    "make-mips-interpreter": (3, 4),
    "path-tracing-reverse": (0, 3),
    "mcmc-sampling-stan": (0, 3),
    "dna-assembly": (0, 3),
    "qemu-alpine-ssh": (0, 3),
}

FULL_TRIAL_COUNT = 5  # config task_trials at the snapshot
CYCLES = 1500
SEED = 20260610


def _simulated_keep(rng: random.Random, effect: float) -> bool:
    verdicts: dict[str, BaselineComparison] = {}
    for task_id, (solved, total) in BASELINE_RATES.items():
        true_rate = min(1.0, solved / total + effect)
        if 0 < total == solved:  # deterministic-solved: 1 confirming trial
            draws = 1
            candidate_solved = int(rng.random() < true_rate)
            if candidate_solved == 0:  # confirm-on-fail expands to full
                draws = FULL_TRIAL_COUNT
                candidate_solved += sum(
                    rng.random() < true_rate for _ in range(FULL_TRIAL_COUNT - 1)
                )
        else:
            draws = FULL_TRIAL_COUNT
            candidate_solved = sum(rng.random() < true_rate for _ in range(draws))
        verdicts[task_id] = BaselineComparison(
            kind="unchanged",  # _panel_decision reads counts, not kinds
            candidate_solved=candidate_solved,
            candidate_total=draws,
            baseline_solved=solved,
            baseline_total=total,
            p_value=None,
        )
    kind, _ = _panel_decision(verdicts=verdicts, purpose="promotion")
    return kind == "keep"


def _keep_rate(effect: float) -> float:
    rng = random.Random(SEED)
    keeps = sum(_simulated_keep(rng, effect) for _ in range(CYCLES))
    return keeps / CYCLES


def test_null_false_keep_rate_is_at_most_alpha() -> None:
    # Inert candidate: the chance the gate promotes panel noise stays within
    # the one-sided alpha (measured ~0.03 at this seed; the stratified-delta
    # direction requirement makes the test conservative).
    null_rate = _keep_rate(0.0)
    assert null_rate <= 0.10, f"null false-keep rate {null_rate:.3f} > alpha"


def test_synthetic_effect_promotes_at_a_material_rate() -> None:
    # A +5% per-task effect must be promotable at a rate materially above both
    # the null and the prior aggregate gate's ~0 (it needed +7-8/59 tasks).
    # Measured ~0.36 at this seed.
    null_rate = _keep_rate(0.0)
    effect_rate = _keep_rate(0.05)
    assert effect_rate >= 0.20, f"+5% effect keep rate {effect_rate:.3f} < 0.20"
    assert effect_rate > 5 * null_rate, (
        f"effect rate {effect_rate:.3f} not materially above null {null_rate:.3f}"
    )
