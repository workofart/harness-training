"""Promotion-gate statistics.

Turns a candidate's per-task trial outcomes into ``BaselineComparison`` verdicts
and a keep/discard decision, and builds the pooled-control sample the gate
compares against (active baseline + recent non-conflicting candidates). Pure
functions over the persisted records in ``record.py``; imports nothing from the
runner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

from src.metrics import (
    AGGREGATE_PROMOTION_P_VALUE_ALPHA,
    PER_TASK_VERDICT_P_VALUE_ALPHA,
    BaselineComparison,
    compare_candidate_against_baseline,
    compute_fisher_exact_p_value,
    is_majority_solved,
)

from src.experiment.record import ExperimentRecord, ExperimentStatus, PanelPurpose


class _PanelAggregate(NamedTuple):
    """Panel-level solved-task counts for the promotion decision.

    Each unit is a whole task (majority-solved), not a trial. ``p_value`` is the
    aggregate Fisher exact test, or ``None`` when no baseline tasks were observed.
    """

    candidate_solved: int
    candidate_total: int
    baseline_solved: int
    baseline_total: int
    p_value: float | None


def build_gate_verdicts(
    *,
    candidate: ExperimentRecord,
    pool: Mapping[str, tuple[int, int]],
    panel: str = "train",
) -> dict[str, BaselineComparison]:
    """Single source of truth for per-task verdicts.

    For each task in the candidate's requested panel, compute the
    :class:`BaselineComparison` against the pooled-control samples. The gate
    and persisted evidence both consume this dict so there is exactly one
    verdict per task per candidate.

    Tasks absent from ``pool`` get treated as no-baseline frontier
    (baseline_total == 0) inside
    :func:`compare_candidate_against_baseline`.
    """
    verdicts: dict[str, BaselineComparison] = {}
    for task_id, candidate_trials in candidate.panels[panel].task_results.items():
        baseline_solved, baseline_total = pool.get(task_id, (0, 0))
        verdict = compare_candidate_against_baseline(
            candidate_solved=candidate_trials.solved_count,
            candidate_total=len(candidate_trials.valid_trials),
            baseline_solved=baseline_solved,
            baseline_total=baseline_total,
            alpha=PER_TASK_VERDICT_P_VALUE_ALPHA,
        )
        verdicts[task_id] = _floor_regression_when_candidate_solves(verdict)
    return verdicts


def _floor_regression_when_candidate_solves(
    verdict: BaselineComparison,
) -> BaselineComparison:
    """Never label a still-solving candidate a regression.

    :func:`compare_candidate_against_baseline` is a pure binomial that treats the
    pooled baseline rate as ground truth. A degenerate high-rate pool (e.g. ~1.0
    from an easy task the panel almost always solves) therefore flags a single
    failed extra trial as a significant regression even when the candidate still
    clears the majority-solve bar -- the exact false positive that discarded
    exp-gpt-5-5-high-speed-up-0530 on adaptive-rejection-sampler (pool ~1.0,
    candidate 3/4). Majority-solve is the experiment's own per-task success
    criterion, so a candidate that still reaches it has not regressed.

    This floor only downgrades regression -> unchanged for candidates that still
    majority-solve; binomial power is untouched for candidates that do NOT
    majority-solve, which still surface as regressions. The downgrade keeps the
    gate decision and the persisted evidence label in agreement (the verdict is
    the single source of truth for both).
    """
    if verdict.kind != "regression":
        return verdict
    if is_majority_solved(
        solved=verdict.candidate_solved, total=verdict.candidate_total
    ):
        return replace(verdict, kind="unchanged")
    return verdict


def decide_panel_from_verdicts(
    *,
    candidate: ExperimentRecord,
    verdicts: Mapping[str, BaselineComparison],
    panel: str,
    purpose: PanelPurpose,
) -> tuple[ExperimentStatus, str]:
    """Resolve panel verdicts into a keep/discard decision.

    Promotion is aggregate: the panel solved-count must improve over the frozen
    parent and pass a relaxed Fisher check. Per-task verdicts stay diagnostic,
    not the trigger.
    """
    aggregate = _aggregate_panel_comparison(
        candidate=candidate,
        verdicts=verdicts,
        panel=panel,
    )
    counts = (
        f"{aggregate.candidate_solved}/{aggregate.candidate_total} "
        f"vs {aggregate.baseline_solved}/{aggregate.baseline_total}"
    )
    if purpose == "regression_veto":
        if aggregate.candidate_solved < aggregate.baseline_solved:
            return "discard", f"{panel} aggregate regressed: {counts}"
        return "keep", f"{panel} aggregate did not regress: {counts}"
    if aggregate.candidate_solved <= aggregate.baseline_solved:
        return "discard", f"{panel} aggregate did not improve: {counts}"
    if aggregate.baseline_total == 0:
        return "keep", f"{panel} aggregate improved: {counts}"
    # Both guards above leave a positive denominator, so p_value is set here.
    assert aggregate.p_value is not None
    passed = aggregate.p_value <= AGGREGATE_PROMOTION_P_VALUE_ALPHA
    verb = "improved" if passed else "improvement not significant"
    op = "<=" if passed else ">"
    return (
        "keep" if passed else "discard",
        f"{panel} aggregate {verb}: {counts} "
        f"(p={aggregate.p_value:.3g} {op} {AGGREGATE_PROMOTION_P_VALUE_ALPHA})",
    )


def _aggregate_panel_comparison(
    *,
    candidate: ExperimentRecord,
    verdicts: Mapping[str, BaselineComparison],
    panel: str,
) -> _PanelAggregate:
    # `build_gate_verdicts` covers every panel task, so each in-scope id is in `verdicts`.
    panel_record = candidate.panels[panel]
    in_scope = panel_record.in_scope_task_results
    candidate_total = len(in_scope)
    candidate_solved = panel_record.solved_count
    baseline_total = (
        candidate_total
        if any(verdicts[task_id].baseline_total > 0 for task_id in in_scope)
        else 0
    )
    baseline_solved = sum(
        1
        for task_id in in_scope
        if is_majority_solved(
            solved=verdicts[task_id].baseline_solved,
            total=verdicts[task_id].baseline_total,
        )
    )
    p_value = (
        None
        if baseline_total == 0 or candidate_total == 0
        else compute_fisher_exact_p_value(
            candidate_solved=candidate_solved,
            candidate_total=candidate_total,
            baseline_solved=baseline_solved,
            baseline_total=baseline_total,
        )
    )
    return _PanelAggregate(
        candidate_solved=candidate_solved,
        candidate_total=candidate_total,
        baseline_solved=baseline_solved,
        baseline_total=baseline_total,
        p_value=p_value,
    )


# ----------------------------------------------------------------------------
# Control construction for the promotion gate.
# ----------------------------------------------------------------------------


def load_recent_candidate_records(
    *,
    experiments_root: Path,
    window: int = 20,
) -> list[ExperimentRecord]:
    """Most-recently-finished concluded candidate records, newest first.

    Filters: only records with a parent baseline, only concluded. Caller is responsible for
    further filtering (e.g., excluding the candidate currently being
    evaluated). Used by the supervisor's mechanism-novelty check
    (``control.gates.build_mechanism_novelty_rejection``); the promotion gate
    itself compares only against the frozen active baseline.
    """
    records: list[ExperimentRecord] = []
    if not experiments_root.exists():
        return records
    for child in sorted(experiments_root.iterdir()):
        if not child.is_dir():
            continue
        if not ExperimentRecord.path(child.name, root=experiments_root).exists():
            continue
        try:
            record = ExperimentRecord.load(child.name, root=experiments_root)
        except Exception:
            continue
        if record.is_concluded() and record.parent_baseline_experiment_id is not None:
            records.append(record)
    records.sort(key=lambda r: r.finished_at or r.started_at, reverse=True)
    return records[:window]


def build_baseline_pool(
    *,
    active_baseline: "ExperimentRecord",
    task_ids: Sequence[str],
    panel: str = "train",
) -> dict[str, tuple[int, int]]:
    """Build the (solved, total) control the promotion gate compares against.

    The control is the FROZEN active baseline's own valid trials for each task
    and nothing else. The gate deliberately does not borrow other candidates'
    trials: a candidate-pooled control is non-stationary (its rate depends on
    what earlier candidates happened to do) and couples evaluations, so a
    candidate that records no rule contaminates everyone's pool while
    contaminated pools manufacture false regressions. Comparing only against
    the fixed baseline keeps each evaluation independent and the control
    stationary until a keep re-freezes the baseline.

    Tasks the baseline never ran map to (0, 0), which
    :func:`compare_candidate_against_baseline` treats as no-baseline frontier.
    """
    pool: dict[str, tuple[int, int]] = {}
    for task_id in task_ids:
        trials = active_baseline.panels[panel].task_results.get(task_id)
        if trials is None:
            pool[task_id] = (0, 0)
            continue
        pool[task_id] = (trials.solved_count, len(trials.valid_trials))
    return pool
