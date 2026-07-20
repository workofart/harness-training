"""Secondary reward metrics for rollout comparisons."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.rollout.records import (
    UNSCORABLE_FAILURE_MODES,
    ExperimentResult,
    RolloutResult,
    SecondaryRewardComparison,
    solved_task_ids,
)
from src.rollout.episode import STEPS_USED_KEY
from src.rollout.telemetry import FIRST_ATTEMPT_TOTAL_KEY, FIRST_ATTEMPT_VALID_KEY


class SecondaryRewardMetric(ABC):
    """Adding a metric: one frozen-side publish site writes a RolloutResult.metrics key,
    one SecondaryRewardMetric owns that key's scope + absence policy, and the training
    script passes it to the criterion at construction (``benchmark()`` supplies the
    per-benchmark defaults). Shared ledger/merge/Criterion code stays fixed.
    """

    name: str
    higher_is_better: bool

    def compare(
        self, *, baseline: ExperimentResult, candidate: ExperimentResult
    ) -> SecondaryRewardComparison:
        baseline_value, candidate_value = self.values(
            baseline=baseline, candidate=candidate
        )
        if baseline_value is None or candidate_value is None:
            outcome = "unavailable"
        elif candidate_value == baseline_value:
            outcome = "tied"
        elif (candidate_value > baseline_value) == self.higher_is_better:
            outcome = "candidate_better"
        else:
            outcome = "baseline_better"
        return SecondaryRewardComparison(
            name=self.name,
            baseline_value=baseline_value,
            candidate_value=candidate_value,
            outcome=outcome,
        )

    @abstractmethod
    def values(
        self, *, baseline: ExperimentResult, candidate: ExperimentResult
    ) -> tuple[int | float | None, int | float | None]: ...

    @staticmethod
    def _scorable(rollout: RolloutResult | None) -> bool:
        """Whether a rollout carries meaningful cost."""
        return (
            rollout is not None and rollout.failure_mode not in UNSCORABLE_FAILURE_MODES
        )


class InvalidFirstAttemptsMetric(SecondaryRewardMetric):
    # Penalizes invalid attempt-0 tool calls; repair-only leniency does not count.
    # An absolute count, never a valid/total ratio: the ratio's denominator is
    # steps taken, so at equal invalid counts it ranks the shorter run worse.
    name = "invalid_first_attempts"
    higher_is_better = False

    def values(
        self, *, baseline: ExperimentResult, candidate: ExperimentResult
    ) -> tuple[int | None, int | None]:
        scope = frozenset(
            task_id
            for task_id in baseline.tasks.keys() & candidate.tasks.keys()
            if self._scorable(baseline.tasks[task_id])
            and self._scorable(candidate.tasks[task_id])
        )
        return self._invalid(baseline, scope), self._invalid(candidate, scope)

    def _invalid(
        self, experiment: ExperimentResult, task_ids: frozenset[str]
    ) -> int | None:
        valid = total = 0
        published = False
        for task_id in task_ids:
            rollout = experiment.tasks[task_id]
            assert rollout is not None
            has_valid = FIRST_ATTEMPT_VALID_KEY in rollout.metrics
            has_total = FIRST_ATTEMPT_TOTAL_KEY in rollout.metrics
            if has_valid != has_total:
                raise ValueError(
                    f"{task_id}: first_attempt metric numerator/denominator "
                    "must be published together"
                )
            published = published or has_total
            valid += rollout.metrics.get(FIRST_ATTEMPT_VALID_KEY, 0)
            total += rollout.metrics.get(FIRST_ATTEMPT_TOTAL_KEY, 0)
        return total - valid if published else None


class StepsUsedMetric(SecondaryRewardMetric):
    name = "steps_used"
    higher_is_better = False

    def values(
        self, *, baseline: ExperimentResult, candidate: ExperimentResult
    ) -> tuple[int | None, int | None]:
        scope = solved_task_ids(baseline) & solved_task_ids(candidate)
        return self._steps(baseline, scope), self._steps(candidate, scope)

    def _steps(
        self, experiment: ExperimentResult, task_ids: frozenset[str]
    ) -> int | None:
        steps_used = 0
        for task_id in task_ids:
            rollout = experiment.tasks[task_id]
            assert rollout is not None
            steps = rollout.metrics.get(STEPS_USED_KEY)
            if steps is None:
                return None
            steps_used += steps
        return steps_used


# Benchmark-generic tiebreakers, in order; first decisive metric wins.
GENERIC_SECONDARY_METRICS: tuple[SecondaryRewardMetric, ...] = (
    InvalidFirstAttemptsMetric(),
    StepsUsedMetric(),
)
