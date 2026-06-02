from __future__ import annotations

import pytest

from src.experiment.gate import (
    build_baseline_pool,
    build_gate_verdicts,
    decide_panel_from_verdicts,
)
from src.experiment.record import ExperimentRecord, PanelRecord, terminal_task_result
from src.harness.contracts import TaskResult


def test_evidence_outcome_follows_gate_verdict_not_majority_flip(tmp_path):
    # Regression: before Phase 4 the persisted outcome label was derived
    # from majority_solved booleans against the single baseline record.
    # That mechanism disagreed with the gate whenever the candidate's
    # rate flipped majority without statistical significance. Now both
    # the gate decision and the persisted label come from one verdict
    # dict, so the disagreement can no longer happen.
    baseline = _make_record(
        experiment_id="baseline",
        parent=None,
        train_ids=["wobbly"],
        k=10,
    )
    # Baseline majority-solves 5/10 -> majority_solved is True (>= ceil(10/2)).
    for solved in [True] * 5 + [False] * 5:
        baseline.record_task_result(
            TaskResult(
                task_name="wobbly",
                reward=1.0 if solved else 0.0,
                solved=solved,
                steps_used=1,
                error=None,
                started_at="2026-04-10T00:00:00+00:00",
                finished_at="2026-04-10T00:00:01+00:00",
            )
        )

    candidate = _make_record(
        experiment_id="candidate",
        parent="baseline",
        train_ids=["wobbly"],
        k=10,
    )
    # Candidate slips to 4/10 -> majority_solved is False. Old majority-flip
    # logic would label this "regression". The gate's binomial at p_hat=0.5
    # does not reject H0 at alpha=0.05 (p_value > 0.5), so the verdict is
    # "unchanged" and the new label is "unchanged_unsolved".
    for solved in [True] * 4 + [False] * 6:
        candidate.record_task_result(
            TaskResult(
                task_name="wobbly",
                reward=1.0 if solved else 0.0,
                solved=solved,
                steps_used=1,
                error=None,
                started_at="2026-04-10T00:00:00+00:00",
                finished_at="2026-04-10T00:00:01+00:00",
            )
        )
    candidate.finalize(status="discard", decision_reason="no train improvement")
    pool = {
        tid: (trials.solved_count, trials.trial_count)
        for tid, trials in _train_results(baseline).items()
    }
    verdicts = build_gate_verdicts(candidate=candidate, pool=pool)
    candidate.refresh_evidence(baseline=baseline, verdicts=verdicts)

    outcome = _train_outcomes(candidate)[0]
    assert outcome.outcome == "unchanged_unsolved"
    assert verdicts["wobbly"].kind == "unchanged"


def test_single_source_of_truth_gate_and_evidence_agree():
    # End-to-end consolidation guarantee: one verdict dict per task drives
    # the promotion decision and the persisted evidence label, so they cannot
    # disagree about what happened on a task.

    def _pool(baseline):
        return {
            tid: (trials.solved_count, trials.trial_count)
            for tid, trials in _train_results(baseline).items()
        }

    # Three tasks crafted to exercise three different verdict kinds:
    #   frontier: pool 0/0  -> no-baseline frontier; candidate majority-solves
    #             -> verdict.kind="improvement", outcome="new_solve"
    #   solid:    pool 10/10 -> pool-100%; candidate 3/10
    #             -> verdict.kind="regression", outcome="regression"
    #   wobbly:   pool 5/10  -> partial; candidate 4/10
    #             -> verdict.kind="unchanged", outcome="unchanged_unsolved"
    baseline = _make_record(
        experiment_id="baseline",
        parent=None,
        train_ids=["frontier", "solid", "wobbly"],
        k=10,
    )
    _record_solves(baseline, "solid", [True] * 10)
    _record_solves(baseline, "wobbly", [True] * 5 + [False] * 5)
    # frontier: no baseline trials.
    candidate = _make_record(
        experiment_id="cand",
        parent="baseline",
        train_ids=["frontier", "solid", "wobbly"],
        k=10,
    )
    _record_solves(candidate, "frontier", [True] * 6 + [False] * 4)
    _record_solves(candidate, "solid", [True] * 3 + [False] * 7)
    _record_solves(candidate, "wobbly", [True] * 4 + [False] * 6)

    pool = _pool(baseline)
    verdicts = build_gate_verdicts(candidate=candidate, pool=pool)

    # Verdict layer
    assert verdicts["frontier"].kind == "improvement"
    assert verdicts["solid"].kind == "regression"
    assert verdicts["wobbly"].kind == "unchanged"

    # Decision layer reads the same verdicts. Regression wins over
    # improvement within a candidate.
    status, reason = decide_panel_from_verdicts(
        candidate=candidate,
        verdicts=verdicts,
        panel="train",
        purpose="promotion",
    )
    assert (status, reason) == ("discard", "train task solid regressed")

    # Evidence layer reads the same verdicts. Labels are the 5-state
    # vocabulary derived from verdict.kind + majority bools.
    candidate.finalize(status=status, decision_reason=reason)
    candidate.refresh_evidence(baseline=baseline, verdicts=verdicts)
    outcomes = {
        outcome.task_id: outcome.outcome for outcome in _train_outcomes(candidate)
    }
    assert outcomes["frontier"] == "new_solve"
    assert outcomes["solid"] == "regression"
    assert outcomes["wobbly"] == "unchanged_unsolved"


def test_build_baseline_pool_counts_only_active_baseline_trials():
    # The promotion control is the frozen baseline's own valid trials and
    # nothing else -- no borrowing from other candidates (that borrowing was
    # the non-stationary, contamination-prone pool we removed). Tasks the
    # baseline never ran map to (0, 0) = no-baseline frontier.
    baseline = _make_record(
        experiment_id="baseline",
        parent=None,
        train_ids=["solid", "frontier"],
        k=3,
    )
    _record_solves(baseline, "solid", [True, True, False])
    # "frontier" has no baseline trials.
    pool = build_baseline_pool(
        active_baseline=baseline,
        task_ids=("solid", "frontier", "never-ran"),
    )
    assert pool == {"solid": (2, 3), "frontier": (0, 0), "never-ran": (0, 0)}


def test_baseline_only_small_high_rate_pool_does_not_false_regress():
    # Decision-level anti-lottery guarantee. Against the fixed baseline a 3/3
    # task whose candidate dips to 1/4 is NOT a significant regression -- the
    # two-sample Fisher test treats the tiny baseline as the weak evidence it
    # is -- so the candidate is not discarded on noise. The old one-sample
    # binomial (baseline rate as a known point) discarded exactly these.
    baseline = _make_record(
        experiment_id="baseline", parent=None, train_ids=["easy"], k=3
    )
    _record_solves(baseline, "easy", [True, True, True])  # 3/3
    candidate = _make_record(
        experiment_id="cand", parent="baseline", train_ids=["easy"], k=4
    )
    _record_solves(candidate, "easy", [True, False, False, False])  # 1/4
    pool = build_baseline_pool(active_baseline=baseline, task_ids=("easy",))

    verdicts = build_gate_verdicts(candidate=candidate, pool=pool)
    assert verdicts["easy"].kind == "unchanged"
    status, _ = decide_panel_from_verdicts(
        candidate=candidate,
        verdicts=verdicts,
        panel="train",
        purpose="promotion",
    )
    assert status == "discard"  # no improvement, but NOT a regression discard


def _make_record(*, experiment_id, parent, train_ids, k):
    return ExperimentRecord.initialize(
        experiment_id=experiment_id,
        git_commit_hash=experiment_id,
        parent_baseline_experiment_id=parent,
        panels=[
            PanelRecord.initialize(
                panel_id="train",
                purpose="promotion",
                task_ids=train_ids,
                expected_trial_count=k,
                lifecycle="active",
            )
        ],
        started_at="2026-05-05T00:00:00+00:00",
    )


def _train_results(record: ExperimentRecord):
    return record.panels["train"].task_results


def _train_outcomes(record: ExperimentRecord):
    assert record.evidence is not None
    return record.evidence.panel_outcomes["train"]


def _record_solves(record, task_id, solves):
    for solved in solves:
        record.record_task_result(
            TaskResult(
                task_name=task_id,
                reward=1.0 if solved else 0.0,
                solved=solved,
                error=None,
                steps_used=1,
                started_at="2026-05-05T00:00:00+00:00",
                finished_at="2026-05-05T00:00:01+00:00",
            )
        )


def _pool_from_baseline(baseline) -> dict[str, tuple[int, int]]:
    """Build the gate control from a baseline record via the production
    `build_baseline_pool`, so these tests exercise the same baseline-only
    path the runner uses."""
    if baseline is None:
        return {}
    return build_baseline_pool(
        active_baseline=baseline,
        task_ids=tuple(_train_results(baseline).keys()),
    )


def _gate(*, candidate, pool) -> tuple[str, str]:
    """Two-step gate composition for tests that only care about the
    decision. Production code does the same two calls in
    ``_run_experiment`` but holds onto the verdict dict so it can thread
    it into evidence; tests below only assert on (status, reason)."""
    verdicts = build_gate_verdicts(candidate=candidate, pool=pool)
    return decide_panel_from_verdicts(
        candidate=candidate,
        verdicts=verdicts,
        panel="train",
        purpose="promotion",
    )


def test_majority_solving_candidate_not_flagged_regression_against_degenerate_pool():
    # An easy task whose baseline rate is ~1.0 (3/3); the candidate runs one
    # extra trial and lands 3/4 -- still a majority-solve. Fisher already does
    # not flag 3/4 vs 3/3 (small baseline), and even if it did the caller-side
    # floor downgrades regression -> unchanged for any still-majority-solving
    # candidate, so this can never read as a regression.
    baseline = _make_record(
        experiment_id="baseline", parent=None, train_ids=["easy"], k=3
    )
    _record_solves(baseline, "easy", [True] * 3)  # pool 3/3 -> rate 1.0
    candidate = _make_record(
        experiment_id="candidate", parent="baseline", train_ids=["easy"], k=4
    )
    _record_solves(candidate, "easy", [True] * 3 + [False])  # 3/4 -> majority-solve
    pool = _pool_from_baseline(baseline)

    verdicts = build_gate_verdicts(candidate=candidate, pool=pool)
    assert verdicts["easy"].kind == "unchanged"
    _, reason = decide_panel_from_verdicts(
        candidate=candidate,
        verdicts=verdicts,
        panel="train",
        purpose="promotion",
    )
    assert "regressed" not in reason


def test_well_separated_regression_is_flagged():
    # Power check: a clear, well-separated regression is still caught. A task
    # the baseline solves 5/5 that the candidate fails 0/5 is a significant
    # Fisher regression (p ~= 0.008). The lottery fix removes false regressions
    # from small-sample noise (1/4 vs 4/4), NOT genuine large-effect drops.
    baseline = _make_record(
        experiment_id="baseline", parent=None, train_ids=["easy"], k=5
    )
    _record_solves(baseline, "easy", [True] * 5)  # pool 5/5 -> rate 1.0
    candidate = _make_record(
        experiment_id="candidate", parent="baseline", train_ids=["easy"], k=5
    )
    _record_solves(candidate, "easy", [False] * 5)  # 0/5 -> clear regression
    pool = _pool_from_baseline(baseline)

    verdicts = build_gate_verdicts(candidate=candidate, pool=pool)
    assert verdicts["easy"].kind == "regression"
    status, reason = decide_panel_from_verdicts(
        candidate=candidate,
        verdicts=verdicts,
        panel="train",
        purpose="promotion",
    )
    assert status == "discard" and "regressed" in reason


def test_evaluate_keeps_candidate_with_significant_train_gain_at_k10():
    baseline = _make_record(
        experiment_id="baseline",
        parent=None,
        train_ids=["frontier"],
        k=10,
    )
    # 1/10 baseline (not 0/10) so the Fisher two-sample path applies (the
    # baseline_solved==0 majority-solve shortcut does not). The candidate needs
    # a large enough separation to reach significance at these counts: 8/10 vs
    # 1/10 is a clear Fisher improvement (p ~= 0.006).
    _record_solves(baseline, "frontier", [True] + [False] * 9)
    candidate = _make_record(
        experiment_id="cand",
        parent="baseline",
        train_ids=["frontier"],
        k=10,
    )
    _record_solves(candidate, "frontier", [True] * 8 + [False] * 2)

    status, reason = _gate(candidate=candidate, pool=_pool_from_baseline(baseline))
    assert status == "keep"
    assert reason.startswith("train task frontier improved")


@pytest.mark.parametrize(
    ("train_ids", "baseline_solves", "candidate_solves"),
    [
        pytest.param(
            ["frontier"],
            {"frontier": [True] * 5 + [False] * 5},
            {"frontier": [True] * 6 + [False] * 4},
            id="small-train-gain",
        ),
        pytest.param(
            ["a", "h"],
            {"a": [True] * 6 + [False] * 4, "h": [True] * 3 + [False] * 7},
            {"a": [True] * 6 + [False] * 4, "h": [True] * 3 + [False] * 7},
            id="identical-candidate",
        ),
        pytest.param(
            ["a"],
            {"a": [True] * 10},
            {},
            id="empty-candidate-panel",
        ),
    ],
)
def test_evaluate_discards_without_significant_train_improvement(
    train_ids, baseline_solves, candidate_solves
):
    baseline = _make_record(
        experiment_id="baseline",
        parent=None,
        train_ids=train_ids,
        k=10,
    )
    for task_id, solves in baseline_solves.items():
        _record_solves(baseline, task_id, solves)
    candidate = _make_record(
        experiment_id="cand",
        parent="baseline",
        train_ids=train_ids,
        k=10,
    )
    for task_id, solves in candidate_solves.items():
        _record_solves(candidate, task_id, solves)

    assert _gate(candidate=candidate, pool=_pool_from_baseline(baseline)) == (
        "discard",
        "no train task improvement reached significance",
    )


def test_evaluate_does_not_regress_on_small_sample_majority_loss():
    # Behavioral change from the lottery fix: a baseline-solved task (3/3) where
    # the candidate slips to 1/3 is no longer a significant regression -- the
    # two-sample Fisher test does not reach alpha against a 3-trial baseline.
    # The candidate is still discarded (no task improved), but NOT vetoed as a
    # regression, which is what previously discarded solve-positive candidates.
    baseline = _make_record(
        experiment_id="baseline",
        parent=None,
        train_ids=["guard"],
        k=3,
    )
    _record_solves(baseline, "guard", [True, True, True])
    candidate = _make_record(
        experiment_id="cand",
        parent="baseline",
        train_ids=["guard"],
        k=3,
    )
    _record_solves(candidate, "guard", [True, False, False])

    assert _gate(candidate=candidate, pool=_pool_from_baseline(baseline)) == (
        "discard",
        "no train task improvement reached significance",
    )


def test_evaluate_uses_rates_not_counts_when_trial_counts_differ():
    baseline = _make_record(
        experiment_id="baseline",
        parent=None,
        train_ids=["frontier"],
        k=47,
    )
    _record_solves(baseline, "frontier", [True] * 25 + [False] * 22)
    candidate = _make_record(
        experiment_id="cand",
        parent="baseline",
        train_ids=["frontier"],
        k=10,
    )
    _record_solves(candidate, "frontier", [True] * 10)

    status, reason = _gate(candidate=candidate, pool=_pool_from_baseline(baseline))
    assert status == "keep"
    assert reason == "train task frontier improved"


def test_evaluate_no_baseline_discards_unsolved_candidate():
    # With an empty pool, every task is treated as no-baseline frontier
    # (baseline_solved == 0). A candidate that fails to majority-solve
    # cannot promote itself, so the gate must discard.
    candidate = _make_record(
        experiment_id="cand",
        parent=None,
        train_ids=["a"],
        k=3,
    )
    _record_solves(candidate, "a", [False, False, False])
    assert _gate(candidate=candidate, pool={}) == (
        "discard",
        "no train task improvement reached significance",
    )


def test_evaluate_train_regression_wins_over_train_improvement():
    # Mixed-panel candidate: one train task improves significantly while
    # another regresses. The regression pass runs first across the whole
    # panel, so regressions must take priority over later improvements.
    baseline = _make_record(
        experiment_id="baseline",
        parent=None,
        train_ids=["winner", "loser"],
        k=10,
    )
    _record_solves(baseline, "winner", [False] * 10)
    _record_solves(baseline, "loser", [True] * 10)
    candidate = _make_record(
        experiment_id="cand",
        parent="baseline",
        train_ids=["winner", "loser"],
        k=10,
    )
    _record_solves(candidate, "winner", [True] * 6 + [False] * 4)
    _record_solves(candidate, "loser", [False] * 10)

    assert _gate(candidate=candidate, pool=_pool_from_baseline(baseline)) == (
        "discard",
        "train task loser regressed",
    )


def test_evaluate_keeps_when_no_baseline_task_solved_with_no_baseline_entry():
    # A task with no baseline train panel result can still appear in a
    # candidate panel after a panel edit. If the candidate majority-solves it,
    # evaluate() must treat that as "improvement" on first measurement.
    baseline = _make_record(
        experiment_id="baseline",
        parent=None,
        train_ids=["b"],
        k=6,
    )
    _record_solves(baseline, "b", [True] * 4 + [False] * 2)
    candidate = _make_record(
        experiment_id="cand",
        parent="baseline",
        train_ids=["b", "c"],  # c has no baseline entry
        k=6,
    )
    _record_solves(candidate, "b", [True] * 4 + [False] * 2)
    _record_solves(candidate, "c", [True] * 5 + [False])

    # Under compare_candidate_against_baseline, a missing pool entry is the
    # baseline_solved == 0 path: a candidate that majority-solves becomes an
    # "improvement". The reason reads like any other train-task improvement.
    assert _gate(candidate=candidate, pool=_pool_from_baseline(baseline)) == (
        "keep",
        "train task c improved",
    )


def test_evaluate_discards_when_no_baseline_task_unsolved():
    # Task with no baseline entry: if the candidate fails
    # to majority-solve it, that does not establish a de-facto baseline and
    # cannot drive promotion.
    baseline = _make_record(
        experiment_id="baseline",
        parent=None,
        train_ids=["b"],
        k=6,
    )
    _record_solves(baseline, "b", [True] * 4 + [False] * 2)
    candidate = _make_record(
        experiment_id="cand",
        parent="baseline",
        train_ids=["b", "c"],
        k=6,
    )
    _record_solves(candidate, "b", [True] * 4 + [False] * 2)
    _record_solves(candidate, "c", [False] * 6)

    assert _gate(candidate=candidate, pool=_pool_from_baseline(baseline)) == (
        "discard",
        "no train task improvement reached significance",
    )


def test_crash_fillers_do_not_change_gate_verdicts():
    # Safety net for the A-relax refactor. The gate reads only valid (error-free)
    # trials, so the crash placeholders `_complete_unfinished_task_results`
    # fabricates for unfinished/never-run tasks are invisible to it. A candidate
    # carrying those fillers must produce byte-identical verdicts to one without
    # them -- exactly the before/after of removing the fillers. If this ever
    # fails, dropping them would move a p-value and the refactor is unsafe.
    train_ids = ["task-a", "task-b", "task-c"]
    pool = {"task-a": (8, 10), "task-b": (2, 10), "task-c": (5, 10)}

    def build(*, with_fillers):
        record = _make_record(
            experiment_id="candidate", parent="baseline", train_ids=train_ids, k=3
        )
        _record_solves(record, "task-a", [True, False])
        _record_solves(record, "task-b", [True])
        # task-c never produced a real trial.
        if with_fillers:
            # The placeholders finalize_crash would append to conclude unfinished
            # slots: crashes carry `error`, so valid_trials excludes them.
            for _ in range(2):
                record.record_task_result(
                    terminal_task_result(task_id="task-b", exc=RuntimeError("boom"))
                )
            for _ in range(3):
                record.record_task_result(
                    terminal_task_result(task_id="task-c", exc=RuntimeError("boom"))
                )
        return record

    filled = build(with_fillers=True)
    relaxed = build(with_fillers=False)

    # The two records genuinely differ in their raw trial lists, so equal
    # verdicts are a real invariant rather than a tautology.
    assert _train_results(filled)["task-c"].trial_count == 3
    assert _train_results(relaxed)["task-c"].trial_count == 0
    assert _train_results(filled)["task-b"].trial_count == 3
    assert _train_results(relaxed)["task-b"].trial_count == 1

    # The gate's only trial inputs -- solved_count and len(valid_trials) -- match,
    # because the fillers carry `error` and are excluded.
    for task_id in train_ids:
        assert (
            _train_results(filled)[task_id].solved_count
            == _train_results(relaxed)[task_id].solved_count
        )
        assert len(_train_results(filled)[task_id].valid_trials) == len(
            _train_results(relaxed)[task_id].valid_trials
        )

    # The full verdict objects (kind, p_value, all four counts) are identical.
    assert build_gate_verdicts(candidate=filled, pool=pool) == build_gate_verdicts(
        candidate=relaxed, pool=pool
    )
