from __future__ import annotations

import pytest

from src.experiment.gate import (
    build_pooled_control_samples,
    build_gate_verdicts,
    decide_panel_from_verdicts,
    rule_names_from_added_lines,
)
from src.experiment.record import ExperimentRecord, PanelRecord, terminal_task_result
from src.harness.contracts import TaskResult
from src.metrics import TaskMetrics


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


def test_evidence_uses_pool_verdict_not_baseline_only_recomputation():
    # Companion to the multi-kind end-to-end test above: that test happens
    # to construct `pool` as an exact mirror of the baseline record, so a
    # future refactor that reverted evidence to compare against the
    # baseline record directly (same function, different inputs) would
    # produce identical labels and slip past the assertions. This test
    # forces the pool to disagree with baseline-only so the false-negative
    # has nowhere to hide.
    #
    # Setup: baseline solved task "X" 1/1. The pool includes 7 additional
    # rule-untouched failing trials from prior candidates -> pool = 1/8
    # (12.5% rate). Current candidate fails 0/1.
    #   pool-based verdict:    candidate 0/1 vs pool 1/8, binomial not
    #                          significant -> kind="unchanged",
    #                          outcome="unchanged_unsolved"
    #   baseline-only verdict: candidate 0/1 vs baseline 1/1, p_hat=1.0,
    #                          p_value=0 -> kind="regression",
    #                          outcome="regression"
    # If evidence silently recomputes from the baseline record, this assertion
    # flips to "regression" and the test fails.
    baseline = _make_record(
        experiment_id="baseline",
        parent=None,
        train_ids=["X"],
        k=1,
    )
    _record_solves(baseline, "X", [True])

    candidate = _make_record(
        experiment_id="cand",
        parent="baseline",
        train_ids=["X"],
        k=1,
    )
    _record_solves(candidate, "X", [False])

    # Pool deliberately differs from baseline alone: baseline contributes
    # 1/1 and seven prior candidates contributed 0/1 each.
    pool = {"X": (1, 8)}

    verdicts = build_gate_verdicts(candidate=candidate, pool=pool)
    assert verdicts["X"].kind == "unchanged"
    # Sanity-check: had the gate been pointed at baseline alone, the verdict
    # would have flipped. Keeping this assertion documents the contrast that
    # the test relies on.
    baseline_only = build_gate_verdicts(
        candidate=candidate,
        pool={"X": (1, 1)},
    )
    assert baseline_only["X"].kind == "regression"

    status, reason = decide_panel_from_verdicts(
        candidate=candidate,
        verdicts=verdicts,
        panel="train",
        purpose="promotion",
    )
    assert (status, reason) == (
        "discard",
        "no train task improvement reached significance",
    )

    candidate.finalize(status=status, decision_reason=reason)
    candidate.refresh_evidence(baseline=baseline, verdicts=verdicts)

    outcomes = {
        outcome.task_id: outcome.outcome for outcome in _train_outcomes(candidate)
    }
    assert outcomes["X"] == "unchanged_unsolved"


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
    """Convert a baseline record's per-task trial counts into the (solved,
    total) pool dict the gate primitives expect. Tests that used to thread
    the baseline record directly into evaluate() now thread it through this
    helper instead."""
    if baseline is None:
        return {}
    return {
        tid: (trials.solved_count, trials.trial_count)
        for tid, trials in _train_results(baseline).items()
    }


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
    # Reproduces the exp-gpt-5-5-high-speed-up-0530 discard artifact: an easy
    # task whose pooled baseline rate is ~1.0 (3/3). The candidate runs one
    # extra trial and lands 3/4 -- still a majority-solve. The pure binomial
    # flags 3/4 vs 3/3 as significant (p~=0), but the candidate still SOLVES the
    # task, so the caller-side floor must downgrade regression -> unchanged.
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


def test_binomial_regression_preserved_when_candidate_below_majority():
    # Power check: a candidate that does NOT majority-solve a task the pool
    # solves robustly still surfaces as a regression. The floor only protects
    # still-solving candidates, so binomial power is intact for the rest.
    baseline = _make_record(
        experiment_id="baseline", parent=None, train_ids=["easy"], k=4
    )
    _record_solves(baseline, "easy", [True] * 4)  # pool 4/4 -> rate 1.0
    candidate = _make_record(
        experiment_id="candidate", parent="baseline", train_ids=["easy"], k=4
    )
    _record_solves(candidate, "easy", [True] + [False] * 3)  # 1/4 -> below majority
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
    # 1/10 baseline (not 0/10) so the binomial path applies — at p_hat=0
    # compare_candidate_against_baseline switches to a majority-solve rule,
    # which 4/10 would not satisfy.
    _record_solves(baseline, "frontier", [True] + [False] * 9)
    candidate = _make_record(
        experiment_id="cand",
        parent="baseline",
        train_ids=["frontier"],
        k=10,
    )
    _record_solves(candidate, "frontier", [True] * 4 + [False] * 6)

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


def test_evaluate_discards_when_baseline_solved_task_loses_majority():
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
        "train task guard regressed",
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


def _pool_record(
    *,
    experiment_id: str,
    parent_baseline_id: str | None,
    train_task_ids: list[str],
    finished_at: str = "2026-05-10T00:00:00+00:00",
    status: str = "discard",
    decision_reason: str = "no train improvement",
):
    record = ExperimentRecord.initialize(
        experiment_id=experiment_id,
        git_commit_hash="cand-" + experiment_id,
        parent_baseline_experiment_id=parent_baseline_id,
        panels=[
            PanelRecord.initialize(
                panel_id="train",
                purpose="promotion",
                task_ids=train_task_ids,
                expected_trial_count=1,
                lifecycle="active",
            )
        ],
        started_at="2026-05-10T00:00:00+00:00",
    )
    record.status = status
    record.decision_reason = decision_reason
    record.finished_at = finished_at
    return record


def _pool_trial(
    task_name: str, *, solved: bool, rule_fires: dict[str, int] | None = None
) -> TaskResult:
    return TaskResult(
        task_name=task_name,
        reward=1.0 if solved else 0.0,
        solved=solved,
        error=None,
        steps_used=1,
        started_at="2026-05-10T00:00:00+00:00",
        finished_at="2026-05-10T00:00:01+00:00",
        metrics=TaskMetrics(rule_fires=dict(rule_fires or {})),
    )


def _populate_pool_trials(record, trials_by_task: dict[str, list[TaskResult]]) -> None:
    for trials in trials_by_task.values():
        for trial in trials:
            record.record_task_result(trial)


def test_rule_names_from_added_lines_matches_three_declaration_styles() -> None:
    added = [
        'ARGUMENT_NORMALIZERS = (ArgumentRule(name="unwrap_input", order=10, apply=fn),)',
        'VERIFY_NUDGE_RULE_NAME = "verify_absent_nudge"',
        '    recorder.metrics.record_rule_fire("direct_literal_rule")',
    ]
    assert rule_names_from_added_lines(added) == {
        "unwrap_input",
        "verify_absent_nudge",
        "direct_literal_rule",
    }


def test_rule_names_from_added_lines_handles_pep8_wrapped_constant() -> None:
    added = [
        "SYSTEM_PROMPT_PRE_VERIFY_VALIDATION_RULE_NAME = (",
        '    "system_prompt_pre_verify_validation"',
        ")",
    ]
    assert rule_names_from_added_lines(added) == {
        "system_prompt_pre_verify_validation",
    }


def test_rule_names_from_added_lines_ignores_plain_string_literals() -> None:
    added = [
        '    """docstring describing a feature without a rule name"""',
        '    return f"trial {task_id}: solved"',
        "    if final_passed is True:",
    ]
    assert rule_names_from_added_lines(added) == set()


def _pool_trial_from_spec(task_name: str, spec) -> TaskResult:
    if isinstance(spec, bool):
        return _pool_trial(task_name, solved=spec)
    return TaskResult(
        task_name=task_name,
        reward=1.0 if spec["solved"] else 0.0,
        solved=spec["solved"],
        error=spec.get("error"),
        steps_used=spec.get("steps_used", 1),
        started_at="2026-05-10T00:00:00+00:00",
        finished_at="2026-05-10T00:00:01+00:00",
        metrics=TaskMetrics(
            rule_fires=dict(spec.get("rule_fires") or {}),
            failure_mode=spec.get("failure_mode"),
        ),
    )


def _pool_record_from_spec(spec):
    record = _pool_record(
        experiment_id=spec["experiment_id"],
        parent_baseline_id=spec.get("parent_baseline_id"),
        train_task_ids=spec.get("train_task_ids", ["t"]),
        status=spec.get("status", "discard"),
        decision_reason=spec.get("decision_reason", "no train improvement"),
    )
    _populate_pool_trials(
        record,
        {
            task_name: [
                _pool_trial_from_spec(task_name, trial_spec)
                for trial_spec in trial_specs
            ]
            for task_name, trial_specs in spec.get("trials", {}).items()
        },
    )
    return record


@pytest.mark.parametrize(
    "pool_case",
    [
        pytest.param(
            {
                "baseline": {
                    "experiment_id": "baseline",
                    "status": "keep",
                    "decision_reason": "baseline seed",
                    "trials": {"t": [True, True, False]},
                },
                "expected": {"t": (2, 3)},
            },
            id="includes-baseline-trials",
        ),
        pytest.param(
            {
                "baseline": {
                    "experiment_id": "baseline",
                    "status": "keep",
                    "decision_reason": "baseline seed",
                    "trials": {"t": [True]},
                },
                "recent": [
                    {
                        "experiment_id": "baseline-rerun",
                        "parent_baseline_id": "baseline",
                        "status": "keep",
                        "decision_reason": "baseline rerun",
                        "trials": {"t": [True, True]},
                    }
                ],
                "rules": {"baseline-rerun": ()},
                "expected": {"t": (1, 1)},
            },
            id="excludes-baseline-bookkeeping",
        ),
        pytest.param(
            {
                "baseline": {
                    "experiment_id": "baseline",
                    "status": "keep",
                    "decision_reason": "baseline seed",
                    "trials": {"t": [True]},
                },
                "recent": [
                    {
                        "experiment_id": "cand-1",
                        "parent_baseline_id": "baseline",
                        "trials": {
                            "t": [
                                {"solved": True, "rule_fires": {"new_rule": 5}},
                                True,
                                False,
                            ]
                        },
                    }
                ],
                "rules": {"cand-1": ("new_rule",)},
                "expected": {"t": (2, 3)},
            },
            id="excludes-mechanism-touched-trials",
        ),
        pytest.param(
            {
                "baseline": {
                    "experiment_id": "baseline",
                    "parent_baseline_id": "prior-baseline",
                    "status": "keep",
                    "trials": {"t": [True, False]},
                },
                "include_active_baseline_as_recent": True,
                "rules": {"baseline": ()},
                "expected": {"t": (1, 2)},
            },
            id="does-not-double-count-active-baseline",
        ),
        pytest.param(
            {
                "baseline": {
                    "experiment_id": "baseline",
                    "status": "keep",
                    "decision_reason": "baseline seed",
                    "trials": {"t": [True]},
                },
                "recent": [
                    {
                        "experiment_id": "crashed",
                        "parent_baseline_id": "baseline",
                        "status": "crash",
                        "decision_reason": "",
                        "trials": {"t": [True, True]},
                    }
                ],
                "rules": {"crashed": ()},
                "expected": {"t": (1, 1)},
            },
            id="excludes-crashed-records",
        ),
        pytest.param(
            {
                "baseline": {
                    "experiment_id": "baseline",
                    "status": "keep",
                    "decision_reason": "baseline seed",
                    "trials": {
                        "t": [
                            {"solved": True, "steps_used": 19},
                            {
                                "solved": False,
                                "failure_mode": "hit_timeout",
                                "steps_used": 13,
                            },
                            {
                                "solved": False,
                                "error": "environment reset/bootstrap timed out",
                                "steps_used": 0,
                            },
                        ]
                    },
                },
                "expected": {"t": (1, 2)},
            },
            id="includes-task-timeouts-excludes-infrastructure-errors",
        ),
        pytest.param(
            {
                # `run_task` produces `error=""` from bare `RuntimeError()`
                # (`str(exc)` is the empty string). A truthiness check on
                # `trial.error` would silently let this slip into the pool.
                "baseline": {
                    "experiment_id": "baseline",
                    "status": "keep",
                    "decision_reason": "baseline seed",
                    "trials": {
                        "t": [
                            True,
                            {"solved": False, "error": "", "steps_used": 30},
                        ]
                    },
                },
                "expected": {"t": (1, 1)},
            },
            id="excludes-trial-with-empty-string-error",
        ),
        pytest.param(
            {
                "baseline": {
                    "experiment_id": "baseline",
                    "status": "keep",
                    "decision_reason": "baseline seed",
                    "trials": {"t": [True]},
                },
                "recent": [
                    {
                        "experiment_id": "cand",
                        "parent_baseline_id": "baseline",
                        "decision_reason": "train regressed",
                        "trials": {
                            "t": [
                                False,
                                {
                                    "solved": False,
                                    "error": "abandoned after supervisor restart",
                                    "steps_used": 0,
                                },
                            ]
                        },
                    }
                ],
                "rules": {"cand": ()},
                "expected": {"t": (1, 2)},
            },
            id="excludes-trials-with-error-marker",
        ),
        pytest.param(
            {
                "baseline": {
                    "experiment_id": "baseline",
                    "status": "keep",
                    "decision_reason": "baseline seed",
                    "trials": {"t": [True]},
                },
                "recent": [
                    {
                        "experiment_id": "cand-1",
                        "parent_baseline_id": "baseline",
                        "trials": {"t": [False]},
                    }
                ],
                "rules": {"cand-1": ()},
                "task_ids": ["t", "unseen-task"],
                "expected": {"t": (1, 2), "unseen-task": (0, 0)},
            },
            id="handles-missing-task",
        ),
    ],
)
def test_build_pooled_control_samples_filters_records_and_trials(pool_case):
    baseline = _pool_record_from_spec(pool_case["baseline"])
    recent_records = [
        _pool_record_from_spec(spec) for spec in pool_case.get("recent", [])
    ]
    if pool_case.get("include_active_baseline_as_recent"):
        recent_records.append(baseline)

    pool = build_pooled_control_samples(
        active_baseline=baseline,
        recent_candidates=recent_records,
        candidate_new_rule_names=pool_case.get("rules", {}),
        task_ids=pool_case.get("task_ids", ["t"]),
    )

    assert pool == pool_case["expected"]


def test_build_pooled_control_samples_skips_recent_record_missing_panel():
    baseline = ExperimentRecord.initialize(
        experiment_id="baseline",
        git_commit_hash="baseline-sha",
        parent_baseline_experiment_id=None,
        panels=[
            PanelRecord.initialize(
                panel_id="train",
                purpose="promotion",
                task_ids=["train-a"],
                expected_trial_count=1,
                lifecycle="finished",
            ),
            PanelRecord.initialize(
                panel_id="test",
                purpose="regression_veto",
                task_ids=["test-a"],
                expected_trial_count=1,
                lifecycle="finished",
            ),
        ],
        started_at="2026-05-10T00:00:00+00:00",
    )
    baseline.status = "keep"
    baseline.decision_reason = "promoted"
    baseline.finished_at = "2026-05-10T00:10:00+00:00"
    baseline.record_task_result(_pool_trial("test-a", solved=True))
    train_only_recent = _pool_record(
        experiment_id="train-only-recent",
        parent_baseline_id="baseline",
        train_task_ids=["train-a"],
        finished_at="2026-05-10T00:09:00+00:00",
    )
    _populate_pool_trials(
        train_only_recent,
        {"train-a": [_pool_trial("train-a", solved=False)]},
    )

    pool = build_pooled_control_samples(
        active_baseline=baseline,
        recent_candidates=[train_only_recent],
        candidate_new_rule_names={},
        task_ids=["test-a"],
        panel="test",
    )

    assert pool == {"test-a": (1, 1)}


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
