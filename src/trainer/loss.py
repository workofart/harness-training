"""Criterion contract plus the shipped solved-set promotion criterion.

``StrictPareto`` compares binary task verdicts as sets — set movement (which tasks
regressed, which newly solve) keeps every verdict auditable per task, where a
scalar mean would average a regression away.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import inf
from typing import TYPE_CHECKING, Any

from src.rollout.records import (
    VERDICT_UNTRUSTED_FAILURE_MODES,
    DecisionOutcome,
    ExperimentResult,
    ResultDecision,
    SecondaryRewardComparison,
    solved_task_ids,
)
from src.rollout.metrics import SecondaryRewardMetric

if TYPE_CHECKING:
    from src.trainer.parameter import Parameter


@dataclass(frozen=True, slots=True)
class Loss:
    value: float
    criterion: str
    # The judged evidence: carrying the runs binds the verdict to its candidate
    # commit structurally, so optimizer application cannot desynchronize.
    candidate: ExperimentResult
    baseline: ExperimentResult
    harness: Parameter | None
    # Criterion-owned judgment payload; the framework never reads it.
    report: Any = None

    def __float__(self) -> float:
        return self.value

    def backward(self) -> None:
        """Record this verdict on the harness — no gradients; the harness is a git ref and
        optimizer.step() applies the update."""
        if self.harness is None:
            raise RuntimeError(
                "criterion is unbound: only a Trainer-bound criterion's Loss "
                "can backward()"
            )
        self.harness.grad = self

    @property
    def promoted(self) -> bool:
        """True iff the optimizer applied this verdict's candidate to the harness."""
        applied = None if self.harness is None else self.harness.applied
        if applied is None:
            raise RuntimeError("promoted is undefined before optimizer.step()")
        return applied == self.candidate.git_commit_hash


class UnmeasurableRun(Exception):
    def __init__(self, decision: ResultDecision) -> None:
        super().__init__("infra-sensitive verdict")
        self.decision = decision


class Criterion:
    """Judges a measured (candidate, baseline) pair into a scalar Loss.

    The framework contract: negative loss selects the candidate, ``__call__``
    raises UnmeasurableRun when infra noise makes the verdict undecidable, and
    ``decision()`` renders a judged Loss into the persisted ResultDecision.
    """

    name: str
    # Trainer binds the harness; standalone criteria can judge but cannot backward().
    harness: Parameter | None = None

    def __call__(
        self,
        *,
        candidate: ExperimentResult,
        baseline: ExperimentResult,
    ) -> Loss:
        raise NotImplementedError

    def decision(self, loss: Loss) -> ResultDecision:
        raise NotImplementedError

    def describe(self) -> str:
        """One banner line: what selects the candidate."""
        return self.name


@dataclass(frozen=True, slots=True)
class Comparison:
    """One pessimistic, optimistic, or measured view of a run comparison."""

    baseline: ExperimentResult
    candidate: ExperimentResult
    candidate_solved: frozenset[str]
    invalid_infra_tasks: frozenset[str]
    reason: str = ""
    secondary_rewards: tuple[SecondaryRewardComparison, ...] = ()
    baseline_solved: frozenset[str] = field(init=False)
    regressions: frozenset[str] = field(init=False)
    new_solves: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        baseline_solved = solved_task_ids(self.baseline)
        object.__setattr__(self, "baseline_solved", baseline_solved)
        object.__setattr__(
            self,
            "regressions",
            (baseline_solved - self.candidate_solved) - self.invalid_infra_tasks,
        )
        object.__setattr__(
            self,
            "new_solves",
            self.candidate_solved - baseline_solved,
        )


class StrictPareto(Criterion):
    name = "strict_pareto"

    def __init__(
        self, secondary_metrics: tuple[SecondaryRewardMetric, ...] = ()
    ) -> None:
        self.secondary_metrics = secondary_metrics

    def describe(self) -> str:
        tiebreak = ", ".join(metric.name for metric in self.secondary_metrics)
        return f"{self.name} · tiebreak: {tiebreak or 'none'}"

    def __call__(
        self,
        *,
        candidate: ExperimentResult,
        baseline: ExperimentResult,
    ) -> Loss:
        panel = frozenset(candidate.tasks)
        if not panel <= frozenset(baseline.tasks):
            raise ValueError("candidate ran tasks the baseline never measured")
        baseline = _restrict_to_panel(baseline, panel)
        candidate_solved = solved_task_ids(candidate)
        invalid_infra_tasks = _verdict_sensitive_task_ids(candidate)

        def resolve(
            solved: frozenset[str],
            reported_invalid: frozenset[str],
        ) -> Comparison:
            comparison = Comparison(
                baseline=baseline,
                candidate=candidate,
                candidate_solved=solved,
                invalid_infra_tasks=reported_invalid,
            )
            reason, secondary_rewards = self._compare(comparison)
            return replace(
                comparison, reason=reason, secondary_rewards=secondary_rewards
            )

        if invalid_infra_tasks:
            # Bracket the verdict between the untrusted tasks' worst and best
            # cases; a sign flip means infra noise decided, not the candidate.
            pessimistic_report = resolve(candidate_solved, frozenset())
            optimistic_report = resolve(
                candidate_solved | invalid_infra_tasks, frozenset()
            )
            if (self._loss_value(pessimistic_report) < 0) != (
                self._loss_value(optimistic_report) < 0
            ):
                raise UnmeasurableRun(
                    _decision_from_report(
                        report=pessimistic_report,
                        outcome="invalid_infra",
                        reason="infra_sensitive_verdict",
                        regressions=pessimistic_report.regressions
                        - invalid_infra_tasks,
                        invalid_infra_tasks=invalid_infra_tasks,
                        criterion=self.name,
                    )
                )

        report = resolve(candidate_solved, invalid_infra_tasks)
        return Loss(
            value=self._loss_value(report),
            criterion=self.name,
            candidate=candidate,
            baseline=baseline,
            harness=self.harness,
            report=report,
        )

    def decision(self, loss: Loss) -> ResultDecision:
        report: Comparison = loss.report
        outcome: DecisionOutcome = "promoted" if loss.promoted else "rejected"
        return _decision_from_report(
            report=report,
            outcome=outcome,
            reason=report.reason,
            regressions=report.regressions,
            invalid_infra_tasks=report.invalid_infra_tasks,
            criterion=loss.criterion,
        )

    def _compare(
        self, comparison: Comparison
    ) -> tuple[str, tuple[SecondaryRewardComparison, ...]]:
        secondary_rewards: tuple[SecondaryRewardComparison, ...] = ()
        if comparison.regressions:
            reason = "regressed_baseline_tasks"
        elif len(comparison.candidate_solved) > len(comparison.baseline_solved):
            reason = "strict_improvement_without_regression"
        else:
            # Regressions were rejected above; this branch is an exact solved-set tie.
            reason = "no_net_improvement"
            comparisons: list[SecondaryRewardComparison] = []
            for metric in self.secondary_metrics:
                reward = metric.compare(
                    baseline=comparison.baseline,
                    candidate=comparison.candidate,
                )
                comparisons.append(reward)
                if reward.outcome == "candidate_better":
                    reason = "secondary_reward_improvement"
                    break
                if reward.outcome == "baseline_better":
                    break
            secondary_rewards = tuple(comparisons)

        return reason, secondary_rewards

    def _loss_value(self, report: Comparison) -> float:
        """Scalarize a report; negative values select the candidate."""
        if report.regressions:
            return inf

        primary = -(len(report.candidate_solved) - len(report.baseline_solved))
        if primary != 0:
            return float(primary)

        value = 0.0
        for index, comparison in enumerate(report.secondary_rewards):
            if comparison.outcome == "candidate_better":
                sign = -1
            elif comparison.outcome == "baseline_better":
                sign = 1
            else:
                sign = 0
            value += (2 ** -(index + 1)) * sign
        return value


def _decision_from_report(
    *,
    report: Comparison,
    outcome: DecisionOutcome,
    reason: str,
    regressions: frozenset[str],
    invalid_infra_tasks: frozenset[str],
    criterion: str,
) -> ResultDecision:
    return ResultDecision(
        outcome=outcome,
        reason=reason,
        candidate_solved=sorted(report.candidate_solved),
        baseline_solved=sorted(report.baseline_solved),
        new_solves=sorted(report.new_solves),
        regressions=sorted(regressions),
        secondary_rewards=list(report.secondary_rewards),
        invalid_infra_tasks=sorted(invalid_infra_tasks),
        criterion=criterion,
    )


def _restrict_to_panel(
    experiment: ExperimentResult, panel: frozenset[str]
) -> ExperimentResult:
    return experiment.model_copy(
        update={
            "tasks": {
                task_id: rollout
                for task_id, rollout in experiment.tasks.items()
                if task_id in panel
            }
        }
    )


def _verdict_sensitive_task_ids(experiment: ExperimentResult) -> frozenset[str]:
    # Sensitivity can only invalidate, never promote.
    return frozenset(
        task_id
        for task_id, rollout in experiment.tasks.items()
        if rollout is not None
        and rollout.failure_mode in VERDICT_UNTRUSTED_FAILURE_MODES
    )
