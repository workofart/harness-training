from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from src.adapters.llm_base import LlmUsage
    from src.harness.contracts import RawState

# Single source of truth for the per-trial terminal-state bucket persisted as
# `metrics.json.failure_mode` (mirrored in program.md "Post-Run Diagnosis").
# `crash` is an infra failure that exhausted within-trial retries and is
# excluded from evidence; `None` means the trial reached no terminal outcome.
FailureMode = Literal[
    "solved",
    "never_verified",
    "verified_rejected",
    "hit_step_cap",
    "hit_timeout",
    "crash",
]


@dataclass(slots=True)
class TaskMetrics:
    """Per-trial counters and terminal-state summary.

    Mutated in place over a trial by the recorders in ``trace.py`` (which own
    a single instance), then frozen into the persisted ``metrics.json`` and
    carried on ``TaskResult.metrics``. Not hashed anywhere, so it is a plain
    mutable dataclass rather than a frozen target fronted by a builder.
    """

    steps_total: int = 0
    run_count: int = 0
    verify_count: int = 0
    action_parse_failure_count: int = 0
    token_input_total: int = 0
    token_output_total: int = 0
    token_reasoning_total: int = 0
    token_cached_input_total: int = 0
    rule_fires: dict[str, int] = field(default_factory=dict)
    custom_counters: dict[str, int] = field(default_factory=dict)
    # Trial-final summary fields. Populated by the runner once `final_passed`
    # is known so the supervisor can answer "where did it die" without
    # walking steps.jsonl. `failure_mode` is one of: "solved",
    # "never_verified", "verified_rejected", "hit_step_cap", "hit_timeout", or
    # "crash" (an infra failure excluded from evidence), or None when the trial
    # did not produce a terminal outcome.
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
        Called only from `env_step_completed` (in-loop) — the forced final
        verify uses a different code path so its rejection does not get
        attributed to the last in-loop action."""
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
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskMetrics":
        fields = dict(payload)
        for counters_field in ("rule_fires", "custom_counters"):
            if counters_field in fields and isinstance(fields[counters_field], dict):
                fields[counters_field] = {
                    str(name): int(count)
                    for name, count in fields[counters_field].items()
                }
        return cls(**fields)


def compute_binomial_p_value(
    *, observed_solved: int, observed_total: int, p_hat: float
) -> float:
    """Two-sided exact binomial p-value for observing `observed_solved`
    successes in `observed_total` trials under H0: success rate is `p_hat`.

    Returns `2 * min(P(X <= obs), P(X >= obs))`, clamped to [0, 1]. This is
    the standard "doubling" definition that's symmetric and well-defined for
    discrete distributions.
    """
    if observed_total <= 0:
        raise ValueError("observed_total must be positive")
    if observed_solved < 0 or observed_solved > observed_total:
        raise ValueError("observed_solved must be in [0, observed_total]")
    if not 0.0 <= p_hat <= 1.0:
        raise ValueError("p_hat must be in [0, 1]")

    n = observed_total
    k = observed_solved

    def _binom_pmf(i: int) -> float:
        return math.comb(n, i) * (p_hat**i) * ((1.0 - p_hat) ** (n - i))

    p_lower = sum(_binom_pmf(i) for i in range(0, k + 1))
    p_upper = sum(_binom_pmf(i) for i in range(k, n + 1))
    return min(1.0, 2.0 * min(p_lower, p_upper))


# Per-task two-sided binomial alpha used by the promotion gate. Family-wise
# error is intentionally uncontrolled.
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
      the candidate, since a single noisy solve against a point-zero rate
      would otherwise trigger improvement on any p_hat==0 boundary.
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
        # In both, p_hat == 0 makes the exact binomial degenerate and a
        # single candidate solve would trigger improvement on noise alone.
        # Require majority-solve instead.
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
    p_value = compute_binomial_p_value(
        observed_solved=candidate_solved,
        observed_total=candidate_total,
        p_hat=baseline_rate,
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
