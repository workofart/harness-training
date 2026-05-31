"""Promotion-gate statistics.

Turns a candidate's per-task trial outcomes into ``BaselineComparison`` verdicts
and a keep/discard decision, and builds the pooled-control sample the gate
compares against (active baseline + recent non-conflicting candidates). Pure
functions over the persisted records in ``record.py``; imports nothing from the
runner.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from src.control import repo as control_repo
from src.harness.contracts import TaskResult
from src.metrics import (
    PROMOTION_P_VALUE_ALPHA,
    BaselineComparison,
    compare_candidate_against_baseline,
    is_majority_solved,
)

from src.experiment.record import ExperimentRecord, ExperimentStatus


def build_gate_verdicts(
    *,
    candidate: ExperimentRecord,
    pool: Mapping[str, tuple[int, int]],
) -> dict[str, BaselineComparison]:
    """Single source of truth for per-task verdicts.

    For each task in the candidate's train panel, compute the
    :class:`BaselineComparison` against the pooled-control samples. The gate
    and persisted evidence both consume this dict so there is exactly one
    verdict per task per candidate.

    Tasks absent from ``pool`` get treated as no-baseline frontier
    (baseline_total == 0) inside
    :func:`compare_candidate_against_baseline`.
    """
    verdicts: dict[str, BaselineComparison] = {}
    for task_id, candidate_trials in candidate.train_task_results.items():
        baseline_solved, baseline_total = pool.get(task_id, (0, 0))
        verdict = compare_candidate_against_baseline(
            candidate_solved=candidate_trials.solved_count,
            candidate_total=len(candidate_trials.valid_trials),
            baseline_solved=baseline_solved,
            baseline_total=baseline_total,
            alpha=PROMOTION_P_VALUE_ALPHA,
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


def decide_from_verdicts(
    *,
    candidate: ExperimentRecord,
    verdicts: Mapping[str, BaselineComparison],
) -> tuple[ExperimentStatus, str]:
    """Resolve verdicts into a promotion decision.

    Iterates the candidate's panel in ``train_task_ids`` order so the
    decision reason deterministically names the first triggering task.
    Regressions take priority over improvements.
    """
    panel_order = tuple(candidate.train_task_ids)
    for task_id in panel_order:
        verdict = verdicts.get(task_id)
        if verdict is not None and verdict.kind == "regression":
            return "discard", f"train task {task_id} regressed"
    for task_id in panel_order:
        verdict = verdicts.get(task_id)
        if verdict is not None and verdict.kind == "improvement":
            return "keep", f"train task {task_id} improved"
    return "discard", "no train task improvement reached significance"


# ----------------------------------------------------------------------------
# Pool construction for the promotion gate.
# ----------------------------------------------------------------------------

RULE_DIFF_PATHS: tuple[str, ...] = ("src/harness/core.py",)

# Mechanism candidates surface a rule name in the diff three ways:
#   1. Dataclass declarations like ArgumentRule(name="foo", ...)
#   2. Constant assignments like FOO_RULE_NAME = "foo" that get passed to
#      record_rule_fire later in the file (or already-defined elsewhere)
#   3. Direct record_rule_fire("foo") calls with a string literal
RULE_NAME_DIFF_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r'name=["\']([A-Za-z_][A-Za-z0-9_]*)["\']'),
    re.compile(r'_RULE_NAME\s*[:=][\s(]*["\']([A-Za-z_][A-Za-z0-9_]*)["\']'),
    re.compile(r'record_rule_fire\(\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']\s*\)'),
)


def rule_names_from_added_lines(added_lines: list[str]) -> set[str]:
    # Joined-text scan so the _RULE_NAME constant pattern can cross the
    # `FOO_RULE_NAME = (` line into the indented string literal that
    # follows. Per-line scanning misses multi-line constant definitions
    # common in PEP-8 wrapped declarations.
    text = "\n".join(added_lines)
    names: set[str] = set()
    for pattern in RULE_NAME_DIFF_PATTERNS:
        for match in pattern.finditer(text):
            names.add(match.group(1))
    return names


def candidate_new_rule_names(
    *,
    workspace_root: Path,
    record: ExperimentRecord,
) -> tuple[str, ...]:
    """Return rule names added by ``record`` relative to its parent baseline.

    Scoped to RULE_DIFF_PATHS (src/harness/core.py only) so
    test fixtures using ``name="..."`` literals don't pollute the result.
    Returns () when the parent baseline commit is unknown — for pooled-control
    filtering we prefer silence over a guess.
    """
    parent_ref = (
        record.evidence.candidate_change.parent_baseline_commit
        if record.evidence is not None
        else None
    )
    if parent_ref is None:
        return ()
    added_lines = control_repo.git_diff_added_lines_between(
        cwd=workspace_root,
        base_ref=parent_ref,
        head_ref=record.git_commit_hash,
        paths=RULE_DIFF_PATHS,
    )
    return tuple(sorted(rule_names_from_added_lines(added_lines)))


# Decision reasons that mark a record as baseline bookkeeping rather than a
# real candidate. These records must not contribute to pooled-control samples
# via recent-candidate history because they don't represent a proposed mechanism.
BASELINE_DECISION_REASONS: frozenset[str] = frozenset(
    {
        "baseline seed",
        "baseline rerun",
    }
)

# Default lookback for pooled-control aggregation.
POOLED_CONTROL_WINDOW = 20


def load_recent_candidate_records(
    *,
    experiments_root: Path,
    window: int = POOLED_CONTROL_WINDOW,
) -> list[ExperimentRecord]:
    """Most-recently-finished concluded candidate records, newest first.

    Filters: only records with a parent baseline, only concluded. Caller is responsible for
    further filtering (e.g., excluding the candidate currently being
    evaluated).
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


def build_pooled_control_samples(
    *,
    active_baseline: "ExperimentRecord",
    recent_candidates: Sequence["ExperimentRecord"],
    candidate_new_rule_names: Mapping[str, tuple[str, ...]],
    task_ids: Sequence[str],
) -> dict[str, tuple[int, int]]:
    """Build (pooled_solved, pooled_total) per task for the noise model.

    Composition:
    1. Active-baseline trials for the task (always included unless the trial
       carries an infrastructure-error marker).
    2. For each non-crashed record in `recent_candidates` whose
       `decision_reason` is not a baseline-bookkeeping marker, include only
       infrastructure-error-free trials where no rule named in that candidate's
       `new_rule_names` fired. The rule filter removes trials structurally
       affected by the candidate's mechanism; the error filter removes
       crash/skip sentinels. Task-episode timeouts are measured unsolved
       trials and arrive with `error=None`.

    Caller is responsible for providing `recent_candidates` (typically the
    most recent N concluded candidates) and mapping each candidate's
    experiment_id to its newly added rule names (extracted from git diff).
    """
    pool: dict[str, tuple[int, int]] = {task_id: (0, 0) for task_id in task_ids}

    def _add(task_id: str, solved: bool) -> None:
        s, n = pool[task_id]
        pool[task_id] = (s + (1 if solved else 0), n + 1)

    def touched_by_new_rules(
        trial: TaskResult,
        new_rule_names: Sequence[str],
    ) -> bool:
        fires = trial.metrics.rule_fires
        return any(fires.get(name, 0) > 0 for name in new_rule_names)

    for task_id in task_ids:
        baseline_trials = active_baseline.train_task_results.get(task_id)
        if baseline_trials is None:
            continue
        for trial in baseline_trials.valid_trials:
            _add(task_id, trial.solved)

    for record in recent_candidates:
        # The active baseline is itself a concluded candidate with a parent
        # (every promoted record is), so load_recent_candidate_records picks
        # it up alongside older candidates. Without this guard the baseline's
        # trials would be counted once via the loop above and again here.
        if record.experiment_id == active_baseline.experiment_id:
            continue
        if record.status == "crash":
            continue
        if record.decision_reason in BASELINE_DECISION_REASONS:
            continue
        new_rules = candidate_new_rule_names.get(record.experiment_id, ())
        for task_id in task_ids:
            trials = record.train_task_results.get(task_id)
            if trials is None:
                continue
            for trial in trials.valid_trials:
                if touched_by_new_rules(trial, new_rules):
                    continue
                _add(task_id, trial.solved)

    return pool


def build_gate_pool(
    *,
    experiments_root: Path,
    workspace_root: Path,
    active_baseline: "ExperimentRecord",
    candidate_experiment_id: str,
    task_ids: Sequence[str],
    window: int = POOLED_CONTROL_WINDOW,
) -> dict[str, tuple[int, int]]:
    """Assemble the (solved, total) pool the promotion gate compares against.

    Includes the active baseline's own trials and the most recent N
    concluded candidates' trials, filtered inside
    :func:`build_pooled_control_samples` to drop crashes, baseline-bookkeeping
    seeds, and trials touched by each candidate's own new rule. Excludes
    the candidate currently being evaluated.
    """
    recent = [
        record
        for record in load_recent_candidate_records(
            experiments_root=experiments_root,
            window=window,
        )
        if record.experiment_id != candidate_experiment_id
    ]
    rule_names_by_id = {
        record.experiment_id: candidate_new_rule_names(
            workspace_root=workspace_root,
            record=record,
        )
        for record in recent
    }
    return build_pooled_control_samples(
        active_baseline=active_baseline,
        recent_candidates=recent,
        candidate_new_rule_names=rule_names_by_id,
        task_ids=task_ids,
    )
