"""The pure core of the outer loop (plan.md §6/§9).

Every pure function and data type the supervisor needs, with zero I/O: the
``decide(world) -> Command`` transition, the promotion/veto ``gate`` and its
Fisher statistics, ``combine``, the per-task trial ``budget_from_baseline``, the
``validate_candidate`` diff check, and the agent's view constants
(``VISIBLE_PATHS``/``EDITABLE_PATHS``). ``loop.py`` does the reading (builds
``World`` via ``scan()``) and the writing (the command executors); this module
just decides. Depends only on ``record`` and ``contracts``.

The gate folds today's three functions -- restrict the baseline to the panel,
build per-task verdicts, aggregate into keep/discard -- into one call returning a
single ``Decision`` whose ``verdicts`` carry the per-task evidence. The control
is the frozen active baseline's own trials and nothing else: a candidate-pooled
control is non-stationary and couples evaluations, so comparing only against the
fixed baseline keeps each evaluation independent until a keep re-freezes it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.contracts import is_majority_solved
from src.experiment.record import ExperimentResult

# ----------------------------------------------------------------------------
# Gate statistics (moved here from metrics.py; §9).
# ----------------------------------------------------------------------------

# Two-sided alpha for the per-task Fisher exact test in the gate's verdicts.
# These per-task verdicts are diagnostic evidence only; the promotion decision
# uses the aggregate alpha below. The strict 0.05 bar relies on Fisher exact
# being strongly conservative at small per-task trial counts (n~3-5).
PER_TASK_VERDICT_P_VALUE_ALPHA = 0.05

# Two-sided alpha for the aggregate panel test, where each unit is a whole task
# (majority-solved), not a trial. Relaxed relative to the per-task alpha: a
# single test over the panel (no multiplicity to guard) and a panel-wide
# solved-task gain is a weaker per-comparison signal than a per-task rate jump.
AGGREGATE_PROMOTION_P_VALUE_ALPHA = 0.20

Purpose = Literal["promotion", "regression_veto"]
VerdictKind = Literal["improvement", "regression", "unchanged", "uncompared"]


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
    for sampling uncertainty in BOTH arms, so a small high-rate baseline is
    correctly weak evidence -- removing the small-sample false regressions.

    Two-sided p = sum of the probabilities of every table (margins fixed) no
    more likely than the observed one, clamped to [0, 1].
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


class BaselineComparison(BaseModel):
    """One task's candidate-vs-baseline verdict (§5). The gate's per-task
    evidence; persisted nested in ``Decision.verdicts`` in ``loop.json``.

    ``kind``:
    - ``uncompared``: candidate produced no trials.
    - ``improvement``: candidate beat the baseline at ``alpha``. When the
      baseline has never solved this task (``baseline_solved == 0`` -- empty
      frontier or trial-history-but-no-solves), the test is replaced by a
      majority-solve requirement on the candidate, since a single noisy solve
      against a never-solved baseline would otherwise read as improvement alone.
    - ``regression``: candidate underperformed at ``alpha``. Never fires when
      ``baseline_solved == 0`` -- there is no rate below 0%.
    - ``unchanged``: neither direction reached significance, or baseline is at
      the 0% floor and the candidate did not majority-solve.

    ``p_value`` is ``None`` when no test was run (uncompared, or
    ``baseline_solved == 0``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

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


def compare_candidate_against_baseline(
    *,
    candidate_solved: int,
    candidate_total: int,
    baseline_solved: int,
    baseline_total: int,
    alpha: float = PER_TASK_VERDICT_P_VALUE_ALPHA,
) -> BaselineComparison:
    """The only function that answers "did the candidate beat the baseline on
    this task?". Curriculum-frontier (no prior baseline samples) is the normal
    case: a candidate that majority-solves is an improvement, else unchanged.
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
        # Baseline has never solved this task (no prior trials, or trials but no
        # solves). It cannot be regressed against, and a single candidate solve
        # would read as improvement on noise -- require a majority-solve instead.
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


# ----------------------------------------------------------------------------
# The decision a gate produces (§5).
# ----------------------------------------------------------------------------


DecisionKind = Literal["keep", "discard"]


class Decision(BaseModel):
    """One panel's (or the combined run's) keep/discard verdict + its evidence.
    The gate's output and the persisted decision (nested in ``loop.json``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: DecisionKind
    reason: str
    verdicts: dict[str, BaselineComparison]


class LoopResult(BaseModel):
    """One auto cycle, persisted to ``loop.json`` (§5). ``decision is None``
    marks the single pending run prewritten before its orchestrator call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str
    kind: Literal["baseline", "candidate"]
    focus_name: str
    # The one foreign key: a candidate's parent baseline (self-ref to a prior
    # ExperimentResult); null for a baseline cycle. scan() hard-fails at Step 4 if
    # a pending candidate's parent is not the active baseline gate() compares
    # against (§12); decide() only relies on the parent existing.
    parent_baseline_experiment_id: str | None
    decision: Decision | None


# ----------------------------------------------------------------------------
# The gate (pure; §9).
# ----------------------------------------------------------------------------


def _baseline_counts(baseline: ExperimentResult, task_id: str) -> tuple[int, int]:
    # The frozen baseline's own valid trials for this task; (0, 0) for a task it
    # never ran (no-baseline frontier).
    task = baseline.tasks.get(task_id)
    if task is None:
        return (0, 0)
    return (task.solved_count, len(task.valid_trials))


def _task_verdict(
    *, candidate: ExperimentResult, baseline: ExperimentResult, task_id: str
) -> BaselineComparison:
    candidate_task = candidate.tasks.get(task_id)
    candidate_solved = candidate_task.solved_count if candidate_task else 0
    candidate_total = len(candidate_task.valid_trials) if candidate_task else 0
    baseline_solved, baseline_total = _baseline_counts(baseline, task_id)
    verdict = compare_candidate_against_baseline(
        candidate_solved=candidate_solved,
        candidate_total=candidate_total,
        baseline_solved=baseline_solved,
        baseline_total=baseline_total,
    )
    return _floor_regression_when_candidate_solves(verdict)


def _floor_regression_when_candidate_solves(
    verdict: BaselineComparison,
) -> BaselineComparison:
    """Never label a still-solving candidate a regression.

    The per-task Fisher test treats the baseline rate as ground truth, so a
    degenerate high-rate pool (e.g. ~1.0 from an easy task) flags a single
    failed extra trial as a significant regression even when the candidate still
    clears the majority-solve bar. Majority-solve is the experiment's own
    per-task success criterion, so a candidate that still reaches it has not
    regressed. The floor only downgrades regression -> unchanged for candidates
    that still majority-solve; power against candidates that do NOT
    majority-solve is untouched.
    """
    if verdict.kind != "regression":
        return verdict
    if is_majority_solved(
        solved=verdict.candidate_solved, total=verdict.candidate_total
    ):
        return verdict.model_copy(update={"kind": "unchanged"})
    return verdict


def _aggregate_counts(
    verdicts: dict[str, BaselineComparison],
) -> tuple[int, int, int, int]:
    # Each unit is a whole task (majority-solved). Returns
    # (candidate_solved, candidate_total, baseline_solved, baseline_total) in
    # tasks. baseline_total is 0 unless at least one task had baseline trials,
    # which keeps a no-baseline frontier panel out of the Fisher test.
    candidate_total = len(verdicts)
    candidate_solved = sum(
        1
        for verdict in verdicts.values()
        if is_majority_solved(
            solved=verdict.candidate_solved, total=verdict.candidate_total
        )
    )
    has_baseline = any(verdict.baseline_total > 0 for verdict in verdicts.values())
    baseline_total = candidate_total if has_baseline else 0
    baseline_solved = sum(
        1
        for verdict in verdicts.values()
        if is_majority_solved(
            solved=verdict.baseline_solved, total=verdict.baseline_total
        )
    )
    return candidate_solved, candidate_total, baseline_solved, baseline_total


def _panel_decision(
    *,
    verdicts: dict[str, BaselineComparison],
    purpose: Purpose,
) -> tuple[DecisionKind, str]:
    candidate_solved, candidate_total, baseline_solved, baseline_total = (
        _aggregate_counts(verdicts)
    )
    counts = (
        f"{candidate_solved}/{candidate_total} vs {baseline_solved}/{baseline_total}"
    )
    if purpose == "regression_veto":
        # Veto can only block, never promote: discard iff the candidate solves
        # strictly fewer tasks than the frozen baseline did.
        if candidate_solved < baseline_solved:
            return "discard", f"test aggregate regressed: {counts}"
        return "keep", f"test aggregate did not regress: {counts}"
    if candidate_solved <= baseline_solved:
        return "discard", f"train aggregate did not improve: {counts}"
    if baseline_total == 0:
        return "keep", f"train aggregate improved: {counts}"
    # Both guards above leave a positive denominator, so p_value is set here.
    p_value = compute_fisher_exact_p_value(
        candidate_solved=candidate_solved,
        candidate_total=candidate_total,
        baseline_solved=baseline_solved,
        baseline_total=baseline_total,
    )
    passed = p_value <= AGGREGATE_PROMOTION_P_VALUE_ALPHA
    verb = "improved" if passed else "improvement not significant"
    op = "<=" if passed else ">"
    return (
        "keep" if passed else "discard",
        f"train aggregate {verb}: {counts} "
        f"(p={p_value:.3g} {op} {AGGREGATE_PROMOTION_P_VALUE_ALPHA})",
    )


def gate(
    candidate: ExperimentResult,
    baseline: ExperimentResult,
    *,
    task_ids: frozenset[str],
    purpose: Purpose,
) -> Decision:
    """Judge ``candidate`` against the frozen ``baseline`` over ``task_ids``.

    Promotion is aggregate (solved-task count must improve over the frozen
    parent and pass the relaxed Fisher check); regression-veto can only block
    (discard iff fewer tasks solved). Per-task verdicts stay diagnostic, carried
    on the returned ``Decision`` -- the single source of truth for the gate
    decision and the persisted evidence both.
    """
    verdicts = {
        task_id: _task_verdict(candidate=candidate, baseline=baseline, task_id=task_id)
        for task_id in sorted(task_ids)
    }
    kind, reason = _panel_decision(verdicts=verdicts, purpose=purpose)
    return Decision(kind=kind, reason=reason, verdicts=verdicts)


def combine(train: Decision, test: Decision | None) -> Decision:
    """The run's decision: keep iff train kept AND the test panel did not veto.
    Promotion proposes, veto disposes. The merged ``verdicts`` carry both
    panels' evidence (train/test task sets are disjoint, §12)."""
    verdicts = {**train.verdicts, **(test.verdicts if test else {})}
    if train.kind == "discard":
        return Decision(
            kind="discard",
            reason=f"train discarded ({train.reason})",
            verdicts=verdicts,
        )
    if test is not None and test.kind == "discard":
        return Decision(
            kind="discard", reason=f"test vetoed ({test.reason})", verdicts=verdicts
        )
    tail = f"; test passed ({test.reason})" if test is not None else "; no test panel"
    return Decision(
        kind="keep", reason=f"train kept ({train.reason}){tail}", verdicts=verdicts
    )


def budget_from_baseline(
    baseline: ExperimentResult | None,
    *,
    task_ids: frozenset[str],
    full: int,
) -> dict[str, int]:
    """Per-task initial trial budget (#7-decision). A task the frozen baseline
    solved on *every* valid trial starts at a single confirming trial; the
    orchestrator's confirm-on-fail expands it back to ``full`` if that trial
    fails. Every other task starts at ``full``. With ``full == 1`` there is
    nothing to economize, so the deterministic shortcut is inert."""
    budget: dict[str, int] = {}
    for task_id in task_ids:
        baseline_task = None if baseline is None else baseline.tasks.get(task_id)
        deterministic = (
            full > 1
            and baseline_task is not None
            and baseline_task.is_deterministic_solved
        )
        budget[task_id] = 1 if deterministic else full
    return budget


# ----------------------------------------------------------------------------
# The agent's view + candidate diff validation (§7).
# ----------------------------------------------------------------------------

# The only paths a candidate may edit: the harness mechanism and its own test.
EDITABLE_PATHS: frozenset[str] = frozenset(
    {"src/harness/core.py", "tests/harness/test_core.py"}
)
# What the agent sees in its sparse worktree: the editable paths, the brief +
# build/run files, and the import-closure of tests/harness/test_core.py (so it
# can run its own test inside the view). config is deliberately absent -- the
# agent cannot hardcode task names it never sees. Verified by behavior (the
# sparse view runs test_core), not static analysis. EDITABLE c VISIBLE.
VISIBLE_PATHS: frozenset[str] = EDITABLE_PATHS | frozenset(
    {
        "program.md",
        "pyproject.toml",
        "uv.lock",
        "src/__init__.py",
        "src/contracts.py",
        "src/llm/base.py",
        "src/trace.py",
        "src/serialization.py",
        "tests/conftest.py",
    }
)


@dataclass(frozen=True, slots=True)
class CandidateDiff:
    """The candidate's change as pure data: which paths it touched and the lines
    it added under the harness paths. Extracted by ``workspace`` (git); validated
    here without any I/O."""

    changed_paths: tuple[str, ...]
    added_lines: tuple[str, ...]


def _task_id_in_added_lines(
    *, added_lines: tuple[str, ...], task_ids: frozenset[str]
) -> str | None:
    if not task_ids or not added_lines:
        return None
    offenders: dict[str, str] = {}
    for task_id in task_ids:
        pattern = re.compile(r"\b" + re.escape(task_id) + r"\b")
        for line in added_lines:
            if pattern.search(line):
                offenders[task_id] = line.strip()[:120]
                break
    if not offenders:
        return None
    details = "; ".join(
        f"{task_id} -> {sample!r}" for task_id, sample in sorted(offenders.items())
    )
    return (
        "candidate diff embeds literal task ids in harness paths "
        "(program.md task-agnostic rule; use a generic mechanism instead): " + details
    )


def validate_candidate(diff: CandidateDiff, *, task_ids: frozenset[str]) -> str | None:
    """The pure half of candidate validation: the first rejection message, or
    ``None`` if the diff is admissible. The behavioral half (the sparse-view
    ``test_core`` run) lives in ``workspace``. Two checks, in order:

    1. every changed path is in ``EDITABLE_PATHS`` (can't edit what you can't
       see -- ``EDITABLE c VISIBLE``);
    2. no added line embeds a literal configured task id (the task-agnostic
       rule; a backstop -- config being invisible is the primary defense).
    """
    invalid = sorted(path for path in diff.changed_paths if path not in EDITABLE_PATHS)
    if invalid:
        return (
            "candidate modified paths outside the editable allowlist "
            "(src/harness/core.py + tests/harness/test_core.py): " + ", ".join(invalid)
        )
    return _task_id_in_added_lines(added_lines=diff.added_lines, task_ids=task_ids)


# ----------------------------------------------------------------------------
# The outer-loop transition (pure; §6).
# ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PendingRun:
    """The <=1 *live* pending run: ``loop.decision`` is null (prewritten before
    its orchestrator call) AND the run completed. ``scan()`` filters out dead
    pendings -- crashed, killed mid-run, or launched-but-never-recorded
    (``result is None``) -- so they never reach ``decide()`` and a prior crash
    never blocks a manual rerun (§11); hence ``result`` is always a completed
    ``ExperimentResult`` here."""

    loop: LoopResult
    result: ExperimentResult | None


@dataclass(frozen=True, slots=True)
class World:
    """``decide()``'s only input; built by ``scan()`` (the I/O read boundary)."""

    head_commit: str
    primary_dirty: bool
    train_tasks: frozenset[str]
    test_tasks: frozenset[str]
    active_baseline: ExperimentResult | None
    pending: PendingRun | None
    undiagnosed_candidate_id: str | None


@dataclass(frozen=True, slots=True)
class Halt:
    reason: str


@dataclass(frozen=True, slots=True)
class RefreshBaseline:
    pass


@dataclass(frozen=True, slots=True)
class ProposeAndLaunch:
    pass


@dataclass(frozen=True, slots=True)
class RunVeto:
    experiment_id: str


@dataclass(frozen=True, slots=True)
class Conclude:
    experiment_id: str


@dataclass(frozen=True, slots=True)
class Diagnose:
    experiment_id: str


# execute() matches on type (§6).
Command = Halt | RefreshBaseline | ProposeAndLaunch | RunVeto | Conclude | Diagnose


def decide(w: World) -> Command:
    """The outer-loop truth table, executable (§6). First match wins; the order
    is the table. No stored phase -- every command is derived from ``World``.
    A dead pending run (crashed, killed mid-run leaving run_status "running", or
    launched-but-never-recorded) is filtered out of ``World.pending`` by ``scan()``
    (§11), so the pending here is always a completed run a manual rerun can act on
    -- a prior crash never blocks the loop. HEAD-drift safety beyond rule 6's
    ``commit == HEAD`` lives at the one HEAD-moving op (``Conclude``'s
    ``--ff-only``)."""
    if w.primary_dirty:
        return Halt("primary worktree dirty")
    p = w.pending
    if p:  # a completed pending run (scan() filtered out any dead pending)
        result = p.result
        assert result is not None and result.run_status == "completed"
        if p.loop.kind == "baseline":
            return Conclude(result.experiment_id)
        baseline = w.active_baseline
        assert baseline is not None  # a pending candidate always has a parent (§12)
        train = gate(result, baseline, task_ids=w.train_tasks, purpose="promotion")
        if train.kind == "keep" and not (w.test_tasks <= result.tasks.keys()):
            return RunVeto(result.experiment_id)
        return Conclude(result.experiment_id)  # discard at train, or both ran
    if w.undiagnosed_candidate_id:
        return Diagnose(w.undiagnosed_candidate_id)
    if not (w.active_baseline and w.active_baseline.git_commit_hash == w.head_commit):
        return RefreshBaseline()
    return ProposeAndLaunch()
