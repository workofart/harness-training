from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from src.adapters.llm_base import LlmUsage
    from src.harness.contracts import RawState

# Single source of truth for the per-trial terminal-state bucket persisted as
# `metrics.json.failure_mode` (mirrored in program.md "Post-Run Diagnosis").
# `crash` is an infra failure that exhausted within-trial retries and is
# excluded from evidence; `interrupted` is a trial stopped from the outside (a
# Ctrl-C or a supervisor restart) rather than failing on its own, also excluded
# from evidence; `None` means the trial reached no terminal outcome.
FailureMode = Literal[
    "solved",
    "never_verified",
    "verified_rejected",
    "hit_step_cap",
    "hit_timeout",
    "no_valid_action",
    "interrupted",
    "crash",
]


class TaskMetrics(BaseModel):
    """Per-trial counters and terminal-state summary.

    Mutated in place over a trial by the recorders in ``trace.py`` (which own
    a single instance), then frozen into the persisted ``metrics.json`` and
    carried on ``TaskResult.metrics``. Not hashed anywhere, so it is a plain
    mutable model rather than a frozen target fronted by a builder.
    """

    model_config = ConfigDict(extra="forbid")

    steps_total: int = 0
    run_count: int = 0
    verify_count: int = 0
    action_parse_failure_count: int = 0
    token_input_total: int = 0
    token_output_total: int = 0
    token_reasoning_total: int = 0
    token_cached_input_total: int = 0
    rule_fires: dict[str, int] = Field(default_factory=dict)
    custom_counters: dict[str, int] = Field(default_factory=dict)
    # Trial-final summary fields. Populated by the runner once `final_passed`
    # is known so the supervisor can answer "where did it die" without
    # walking steps.jsonl. `failure_mode` is one of: "solved",
    # "never_verified", "verified_rejected", "hit_step_cap", "hit_timeout",
    # "no_valid_action", "interrupted" (stopped from the outside), or "crash"
    # (an infra failure) -- the last three excluded from evidence -- or None
    # when the trial did not produce a terminal outcome.
    final_action_passed: bool | None = None
    verifier_passed: bool | None = None
    failure_mode: FailureMode | None = None

    def record_action(self, step_index: int, action_name: str) -> None:
        self.steps_total = step_index
        self.final_action_passed = None
        if action_name == "run":
            self.run_count += 1
        if action_name == "verify":
            self.verify_count += 1

    def record_action_parse_failure(self) -> None:
        self.action_parse_failure_count += 1

    def record_completion_usage(self, usage: LlmUsage) -> None:
        if usage.prompt_tokens is not None:
            self.token_input_total += usage.prompt_tokens
        if usage.completion_tokens is not None:
            self.token_output_total += usage.completion_tokens
        if usage.reasoning_tokens is not None:
            self.token_reasoning_total += usage.reasoning_tokens
        if usage.cached_input_tokens is not None:
            self.token_cached_input_total += usage.cached_input_tokens

    def record_step_passed(self, raw_state: RawState) -> None:
        """Attribute the env outcome to the most recent agent-chosen action.
        Called only from `env_step_completed` (in-loop)."""
        if isinstance(raw_state.passed, bool):
            self.final_action_passed = raw_state.passed

    def record_rule_fire(self, rule_name: str) -> None:
        self.rule_fires[rule_name] = self.rule_fires.get(rule_name, 0) + 1

    def set_trial_outcome(
        self,
        *,
        verifier_passed: bool | None,
        failure_mode: FailureMode | None,
    ) -> None:
        self.verifier_passed = verifier_passed
        self.failure_mode = failure_mode

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2) + "\n")


def compute_fisher_exact_p_value(
    *,
    candidate_solved: int,
    candidate_total: int,
    baseline_solved: int,
    baseline_total: int,
) -> float:
    """Two-sided Fisher exact p-value for the 2x2 contingency table::

                  solved                       failed
        candidate candidate_solved             candidate_total - candidate_solved
        baseline  baseline_solved              baseline_total  - baseline_solved

    Unlike a one-sample binomial against the baseline *rate* (which treats that
    rate as a known point and so calls a candidate miss against a 3/3 baseline
    "impossible"), Fisher's exact test conditions on both margins and accounts
    for sampling uncertainty in BOTH arms. A small high-rate baseline is
    therefore correctly treated as weak evidence, which removes the
    small-sample false regressions that discarded solve-positive candidates.

    Two-sided p = sum of the probabilities of every table (with the margins
    fixed) that is no more likely than the observed one, clamped to [0, 1].
    """
    if candidate_total <= 0:
        raise ValueError("candidate_total must be positive")
    if baseline_total <= 0:
        raise ValueError("baseline_total must be positive")
    if not 0 <= candidate_solved <= candidate_total:
        raise ValueError("candidate_solved must be in [0, candidate_total]")
    if not 0 <= baseline_solved <= baseline_total:
        raise ValueError("baseline_solved must be in [0, baseline_total]")

    n = candidate_total + baseline_total
    row1 = candidate_total  # candidate-arm trials
    col1 = candidate_solved + baseline_solved  # total solved across both arms

    def _table_prob(solved_in_candidate: int) -> float:
        # Hypergeometric: P(candidate arm holds `solved_in_candidate` of the
        # `col1` solved trials) with both margins held fixed.
        return (
            math.comb(col1, solved_in_candidate)
            * math.comb(n - col1, row1 - solved_in_candidate)
            / math.comb(n, row1)
        )

    p_observed = _table_prob(candidate_solved)
    k_min = max(0, row1 - (n - col1))
    k_max = min(row1, col1)
    total = sum(
        prob
        for k in range(k_min, k_max + 1)
        if (prob := _table_prob(k)) <= p_observed * (1.0 + 1e-9)
    )
    return min(1.0, total)


# Per-task two-sided alpha used by the promotion gate's Fisher exact test.
# There is no explicit family-wise-error correction across the panel; the gate
# instead relies on Fisher exact being strongly conservative at the small
# per-task trial counts (n~3-5) so chance regressions stay rare per run.
PROMOTION_P_VALUE_ALPHA = 0.05


VerdictKind = Literal["improvement", "regression", "unchanged", "uncompared"]


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    """Single source of truth for "did the candidate beat the baseline on
    this task?". One comparison per task; the caller decides what counts
    as the baseline (typically pooled-control samples).

    Returned by `compare_candidate_against_baseline`. Consumed by the
    promotion gate and persisted experiment evidence.

    `kind` encodes the verdict:
    - "uncompared": candidate produced no trials.
    - "improvement": candidate beat the baseline by a margin significant at
      `alpha`. When the baseline has never been observed solving this task
      (`baseline_solved == 0`, covering both no-baseline frontier with empty
      baseline and tasks with trial history but no solves yet), the
      significance test is replaced by a majority-solve requirement on
      the candidate, since a single noisy solve against a never-solved
      baseline would otherwise read as an improvement on noise alone.
    - "regression": candidate underperformed the baseline at `alpha`.
      Never fires when `baseline_solved == 0` — there is no rate below 0%.
    - "unchanged": neither direction reached significance, or baseline is
      at the 0% floor and the candidate did not majority-solve.

    `p_value` is `None` when no statistical test was run (uncompared, or
    `baseline_solved == 0`).
    """

    kind: VerdictKind
    candidate_solved: int
    candidate_total: int
    baseline_solved: int
    baseline_total: int
    p_value: float | None

    @property
    def candidate_rate(self) -> float | None:
        if self.candidate_total <= 0:
            return None
        return self.candidate_solved / self.candidate_total

    @property
    def baseline_rate(self) -> float | None:
        if self.baseline_total <= 0:
            return None
        return self.baseline_solved / self.baseline_total


def is_majority_solved(*, solved: int, total: int) -> bool:
    """True when `solved` reaches the ceil(total/2) majority threshold."""
    if total <= 0:
        return False
    threshold = (total + 1) // 2
    return solved >= threshold


def is_majority_decided(*, solved: int, finished: int, expected_total: int) -> bool:
    """True when the final majority outcome over `expected_total` trials is
    already locked in by the `solved`/`finished` counts seen so far. Once
    True, remaining trials cannot flip the majority verdict.
    """
    if finished >= expected_total:
        return True
    remaining = expected_total - finished
    final_threshold = (expected_total + 1) // 2
    can_be_true = (solved + remaining) >= final_threshold
    can_be_false = solved < final_threshold
    return not (can_be_true and can_be_false)


def compare_candidate_against_baseline(
    *,
    candidate_solved: int,
    candidate_total: int,
    baseline_solved: int,
    baseline_total: int,
    alpha: float = 0.05,
) -> BaselineComparison:
    """Compare a candidate's per-task trial counts against a baseline pool.

    Returns a `BaselineComparison` carrying the verdict and the numbers
    behind it. This is the only function in the codebase that answers
    "did the candidate beat the baseline?" — gate and evidence both route
    through it.

    Curriculum-frontier (no prior baseline samples for this task) is
    treated as the normal case: a candidate that majority-solves the task
    is an "improvement"; otherwise the comparison stays "unchanged".
    """
    if candidate_solved < 0 or candidate_total < 0:
        raise ValueError("candidate counts must be non-negative")
    if baseline_solved < 0 or baseline_total < 0:
        raise ValueError("baseline counts must be non-negative")
    if candidate_solved > candidate_total:
        raise ValueError("candidate_solved cannot exceed candidate_total")
    if baseline_solved > baseline_total:
        raise ValueError("baseline_solved cannot exceed baseline_total")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")

    if candidate_total == 0:
        return BaselineComparison(
            kind="uncompared",
            candidate_solved=candidate_solved,
            candidate_total=candidate_total,
            baseline_solved=baseline_solved,
            baseline_total=baseline_total,
            p_value=None,
        )

    if baseline_solved == 0:
        # Baseline has never solved this task. Covers two related cases:
        #   - baseline_total == 0: no-baseline frontier; no prior trials.
        #   - baseline_total >  0: task with trial history but no solves.
        # In both, a never-solved baseline cannot be regressed against and a
        # single candidate solve would otherwise read as improvement on noise
        # alone. Require a candidate majority-solve instead.
        kind: VerdictKind = (
            "improvement"
            if is_majority_solved(solved=candidate_solved, total=candidate_total)
            else "unchanged"
        )
        return BaselineComparison(
            kind=kind,
            candidate_solved=candidate_solved,
            candidate_total=candidate_total,
            baseline_solved=baseline_solved,
            baseline_total=baseline_total,
            p_value=None,
        )

    baseline_rate = baseline_solved / baseline_total
    candidate_rate = candidate_solved / candidate_total
    p_value = compute_fisher_exact_p_value(
        candidate_solved=candidate_solved,
        candidate_total=candidate_total,
        baseline_solved=baseline_solved,
        baseline_total=baseline_total,
    )
    if p_value >= alpha or candidate_rate == baseline_rate:
        kind = "unchanged"
    elif candidate_rate > baseline_rate:
        kind = "improvement"
    else:
        kind = "regression"
    return BaselineComparison(
        kind=kind,
        candidate_solved=candidate_solved,
        candidate_total=candidate_total,
        baseline_solved=baseline_solved,
        baseline_total=baseline_total,
        p_value=p_value,
    )
