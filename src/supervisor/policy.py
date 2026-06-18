"""The pure core of the outer loop (plan.md §6/§9).

Every pure function and data type the supervisor needs, with zero I/O: the
``decide(world) -> Command`` transition, the promotion/veto ``gate`` and its
statistics (per-task Fisher exact diagnostics + the graded-reward promotion
test), ``combine``, the per-task trial ``budget_from_baseline``, the
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
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.contracts import is_majority_solved
from src.experiment.record import ExperimentResult, TrialResult

# ----------------------------------------------------------------------------
# Gate statistics (moved here from metrics.py; §9).
# ----------------------------------------------------------------------------

# Two-sided alpha for the per-task Fisher exact test in the gate's verdicts.
# These per-task verdicts are diagnostic evidence only; the promotion decision
# uses the aggregate alpha below. The strict 0.05 bar relies on Fisher exact
# being strongly conservative at small per-task trial counts (n~3-5).
PER_TASK_VERDICT_P_VALUE_ALPHA = 0.05

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


# ----------------------------------------------------------------------------
# The graded promotion statistic (continuous reward; Phase 1 step 2).
# ----------------------------------------------------------------------------

# One-sided alpha for the graded promotion test -- the live promotion bar since
# the cutover from the binary CMH. Set to 0.08, below the nominal 0.10: the
# one-sample test runs slightly hot because a few thin tasks carry near-constant
# reward whose ~zero deltas shrink the variance estimate. A resampling sim over
# the real char reward distributions realizes ~0.12 false-keep at 0.10 vs ~0.09
# at 0.08 with negligible power loss; calibration is pinned by the synthetic
# population (tests/supervisor/test_graded_gate.py) and the real baseline panel
# (tests/supervisor/test_gate_acceptance.py).
GRADED_PROMOTION_P_VALUE_ALPHA = 0.08


def _trial_graded_reward(trial: TrialResult) -> float:
    """The per-trial graded reward the promotion test scores: the fraction of
    verifier tests passed (``TrialResult.reward``, recovered from the CTRF
    report), or the binary solve when no graded reward was recorded -- old
    records (pre-graded-reward) and trials whose verifier wrote no CTRF both fall
    back to 1.0/0.0, so the graded test degrades gracefully to the binary signal
    on any task that carries no partial-credit granularity."""
    if trial.reward is not None:
        return trial.reward
    return 1.0 if trial.solved else 0.0


@dataclass(frozen=True, slots=True)
class GradedRewardTest:
    """One-sided test that the candidate's mean graded reward exceeds the
    baseline's, with the **task as the unit of replication**: per task, the
    candidate's mean per-trial reward minus the baseline's; the panel statistic
    is a one-sample test over those per-task deltas.

    Why task-as-unit (a cluster-robust delta) rather than the per-trial
    stratification the binary CMH uses: graded reward has a few trials per task
    (n~3-5) and many tasks are near-deterministic (within-arm sample variance
    ~0), so an analytic per-stratum variance collapses and false-keeps panel
    noise (measured: it spuriously kept a real historical discard). Using the
    spread of per-task deltas as the variance is robust -- each delta already
    carries its own within-task trial noise -- and stays deterministic +
    closed-form (no RNG in the decision path).

    ``mean_delta`` is the average per-task reward gain (direction: promote iff
    > 0). ``p_value`` is one-sided ``1 - Phi(mean_delta / (sd / sqrt(k)))``, a
    large-K normal approximation (train panels are ~30-60 tasks). Fewer than two
    comparable tasks carry no spread estimate -> p_value 1.0 (no promotion);
    a degenerate all-identical-delta panel is unanimous -> p_value 0.0 iff the
    shared delta is positive. Golden-validated against history + the resampling
    simulation (tests/supervisor/test_graded_gate.py).
    """

    mean_delta: float
    p_value: float
    task_count: int

    @classmethod
    def from_task_deltas(cls, deltas: Iterable[float]) -> GradedRewardTest:
        d = list(deltas)
        k = len(d)
        if k == 0:
            return cls(0.0, 1.0, 0)
        mean_delta = math.fsum(d) / k
        if k < 2:
            # A single comparable task gives no spread -> no significance claim.
            return cls(mean_delta, 1.0, k)
        variance = math.fsum((x - mean_delta) ** 2 for x in d) / (k - 1)
        if variance == 0.0:
            # Every task moved by the same amount: unanimous direction.
            p_value = 0.0 if mean_delta > 0.0 else 1.0
        else:
            z = mean_delta / math.sqrt(variance / k)
            p_value = 0.5 * math.erfc(z / math.sqrt(2.0))
        return cls(mean_delta, p_value, k)

    @property
    def counts(self) -> str:
        return f"mean per-task reward delta {self.mean_delta:+.3f} over {self.task_count} tasks"


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


def _task_counts(result: ExperimentResult, task_id: str) -> tuple[int, int]:
    # One arm's valid trials for this task: (solved, total); (0, 0) for a task it
    # never ran (a no-baseline frontier task on the baseline side).
    task = result.tasks.get(task_id)
    if task is None:
        return (0, 0)
    return (task.solved_count, len(task.valid_trials))


def _task_verdict(
    *, candidate: ExperimentResult, baseline: ExperimentResult, task_id: str
) -> BaselineComparison:
    candidate_solved, candidate_total = _task_counts(candidate, task_id)
    baseline_solved, baseline_total = _task_counts(baseline, task_id)
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


def _graded_task_deltas(
    *,
    candidate: ExperimentResult,
    baseline: ExperimentResult,
    task_ids: frozenset[str],
) -> list[float]:
    """Per-task graded-reward deltas for ``GradedRewardTest``: for every task the
    frozen baseline and the candidate both ran (the same both-arms strata the
    binary CMH uses), the candidate's mean per-trial reward minus the baseline's.
    Scored over valid trials only (crash trials excluded, matching the binary
    gate). A task missing either arm carries no comparison and drops out."""
    deltas: list[float] = []
    for task_id in sorted(task_ids):
        candidate_task = candidate.tasks.get(task_id)
        baseline_task = baseline.tasks.get(task_id)
        if candidate_task is None or baseline_task is None:
            continue
        candidate_rewards = [
            _trial_graded_reward(t) for t in candidate_task.valid_trials
        ]
        baseline_rewards = [_trial_graded_reward(t) for t in baseline_task.valid_trials]
        if not candidate_rewards or not baseline_rewards:
            continue
        deltas.append(
            math.fsum(candidate_rewards) / len(candidate_rewards)
            - math.fsum(baseline_rewards) / len(baseline_rewards)
        )
    return deltas


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


def _regression_veto(
    verdicts: dict[str, BaselineComparison],
) -> tuple[DecisionKind, str]:
    # Veto can only block, never promote: discard iff the candidate solves
    # strictly fewer tasks (majority-solved) than the frozen baseline did. Stays
    # binary on purpose -- "lost a task it should still solve" is a solve fact,
    # and the graded cutover's scope is the promotion decision, not the veto.
    candidate_solved, candidate_total, baseline_solved, baseline_total = (
        _aggregate_counts(verdicts)
    )
    counts = (
        f"{candidate_solved}/{candidate_total} vs {baseline_solved}/{baseline_total}"
    )
    if candidate_solved < baseline_solved:
        return "discard", f"test aggregate regressed: {counts}"
    return "keep", f"test aggregate did not regress: {counts}"


def _graded_promotion(
    *,
    verdicts: dict[str, BaselineComparison],
    deltas: list[float],
) -> tuple[DecisionKind, str]:
    # The promotion decision (cut over from the binary CMH): the per-task
    # graded-reward delta with the task as the unit of replication. It scores
    # the partial-credit movement the binary gate is blind to (a candidate can
    # lift reward without crossing the solve threshold) and the per-task spread,
    # not an analytic per-stratum variance, is what stops a single high-variance
    # task from carrying the panel -- the failure that false-kept on history.
    if not deltas:
        # Pure frontier panel (the baseline ran none of these tasks, so no
        # both-arms delta exists): majority-solved counts are the only evidence.
        candidate_solved, _, baseline_solved, _ = _aggregate_counts(verdicts)
        counts = f"{candidate_solved} vs {baseline_solved} majority-solved tasks"
        if candidate_solved <= baseline_solved:
            return "discard", f"train aggregate did not improve: {counts}"
        return "keep", f"train aggregate improved: {counts}"

    test = GradedRewardTest.from_task_deltas(deltas)
    if test.mean_delta <= 0:
        return "discard", f"train graded reward did not improve: {test.counts}"
    if test.p_value > GRADED_PROMOTION_P_VALUE_ALPHA:
        return (
            "discard",
            f"train improvement not significant: {test.counts} "
            f"(one-sided p={test.p_value:.3g} > {GRADED_PROMOTION_P_VALUE_ALPHA})",
        )
    return (
        "keep",
        f"train graded reward improved: {test.counts} "
        f"(one-sided p={test.p_value:.3g} <= {GRADED_PROMOTION_P_VALUE_ALPHA})",
    )


def gate(
    candidate: ExperimentResult,
    baseline: ExperimentResult,
    *,
    task_ids: frozenset[str],
    purpose: Purpose,
) -> Decision:
    """Judge ``candidate`` against the frozen ``baseline`` over ``task_ids``.

    Promotion is the graded reward test (task as the unit of replication): the
    mean per-task reward delta must be positive and the one-sided test
    significant at ``GRADED_PROMOTION_P_VALUE_ALPHA``. Regression-veto can only
    block (discard iff fewer majority-solved tasks). Per-task verdicts stay
    diagnostic, carried on the returned ``Decision`` -- the single source of
    truth for the gate decision and the persisted evidence both.
    """
    verdicts = {
        task_id: _task_verdict(candidate=candidate, baseline=baseline, task_id=task_id)
        for task_id in sorted(task_ids)
    }
    if purpose == "promotion":
        deltas = _graded_task_deltas(
            candidate=candidate, baseline=baseline, task_ids=task_ids
        )
        kind, reason = _graded_promotion(verdicts=verdicts, deltas=deltas)
    else:
        kind, reason = _regression_veto(verdicts)
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


def _evidence_free_halt(pending: PendingRun) -> Halt:
    """The recovery runbook for a completed pending run with zero valid trials
    (every trial crashed -- provider/infra death, e.g. quota exhaustion). Such a
    run carries no evidence about the code it ran: gating it would record a fake
    discard (or adopt an empty baseline) and relaunch the next run into the same
    dead provider, so the loop stops for a human instead. The trial slots are
    consumed (a crash slot is never re-run), so recovery is archival, not
    resumption; the ref path mirrors ``workspace.candidate_ref`` (pinned by
    test) rather than importing the effectful layer into this pure module."""
    result = pending.result
    assert result is not None
    total = sum(len(task.trials) for task in result.tasks.values())
    first_error = next(
        (
            trial.error.splitlines()[0][:120]
            for task in result.tasks.values()
            for trial in task.trials
            if trial.error
        ),
        None,
    )
    evidence = f"0/{total} valid trials -- every trial crashed"
    if first_error is not None:
        evidence += f" (first error: {first_error})"
    experiment_id = result.experiment_id
    if pending.loop.kind == "candidate":
        return Halt(
            f"pending candidate {experiment_id} has {evidence}. This is an "
            "infra/provider failure, not evidence about the candidate, so the "
            "gate was not run. To recover:\n"
            "  1. fix the provider issue (e.g. wait for the usage-limit window "
            "to reset)\n"
            f"  2. archive the dead run:   mv experiments/{experiment_id} "
            "archived_experiments/\n"
            "  3. drop its candidate ref: git update-ref -d "
            f"refs/experiments/candidate/{experiment_id}\n"
            "  4. restart: uv run auto    (the proposer will generate a fresh "
            "candidate)"
        )
    return Halt(
        f"pending baseline {experiment_id} has {evidence}. This is an "
        "infra/provider failure, not baseline evidence, so it was not adopted. "
        "To recover:\n"
        "  1. fix the provider issue (e.g. wait for the usage-limit window "
        "to reset)\n"
        f"  2. archive the dead run: mv experiments/{experiment_id} "
        "archived_experiments/\n"
        "  3. restart: uv run auto  (the loop will re-run the baseline at HEAD)"
    )


def decide(w: World) -> Command:
    """The outer-loop truth table, executable (§6). First match wins; the order
    is the table. No stored phase -- every command is derived from ``World``.
    A dead pending run (crashed, killed mid-run leaving run_status "running", or
    launched-but-never-recorded) is filtered out of ``World.pending`` by ``scan()``
    (§11), so the pending here is always a completed run a manual rerun can act on
    -- a prior crash never blocks the loop. A completed pending whose every trial
    crashed (zero valid trials -- provider death, e.g. quota exhaustion) Halts
    with a recovery runbook before any gating: an evidence-free run is an infra
    fact, never a verdict. HEAD-drift safety beyond rule 6's
    ``commit == HEAD`` lives at the one HEAD-moving op (``Conclude``'s
    ``--ff-only``)."""
    if w.primary_dirty:
        return Halt("primary worktree dirty")
    p = w.pending
    if p:  # a completed pending run (scan() filtered out any dead pending)
        result = p.result
        assert result is not None and result.run_status == "completed"
        if not any(task.valid_trials for task in result.tasks.values()):
            return _evidence_free_halt(p)
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
