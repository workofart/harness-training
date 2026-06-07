"""Tests for src/experiment/orchestrator.py (run_tasks + scheduling).

One scripted stub-executor scenario proves the admission scheduler composes (LPT
order, trial cap, shared heavy gate, confirm-on-fail, majority early-stop,
idempotent slot release); a focused deterministic test pins the one stateful
piece (priority slot reservation); the append test pins the train->veto contract;
the rest are pure-helper unit tests. The executor is never touched -- a stub
``trial_runner`` scripts outcomes directly (plan.md §8).
"""

from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path

import pytest

from src.experiment.orchestrator import (
    _next_trial_admission_count,
    _planned_admission_count,
    _schedule_order,
    _wants_confirmation_expand,
    run_tasks,
)
from src.experiment.record import ExperimentResult, TaskResult, TrialResult


def _trial(run_id: str, *, solved: bool) -> TrialResult:
    return TrialResult(
        run_id=run_id,
        solved=solved,
        failure_mode="solved" if solved else "verified_rejected",
        verifier_passed=solved,
    )


# --- the composition scenario -----------------------------------------------


class _ScriptedExecutor:
    """A stub ``trial_runner`` the test drives completion-by-completion. Each
    trial registers as started, blocks until ``finish(run_id)`` releases it, then
    returns the scripted outcome -- so the scheduler's reactions are
    deterministic. Tracks launch order, peak concurrency, and the heavy gate it
    was handed; double-calls ``slot_release`` to exercise idempotency (opt #3)."""

    def __init__(self) -> None:
        self.started_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self.started_order: list[str] = []
        self.semaphores: list[asyncio.Semaphore] = []
        self.peak = 0
        self._inflight = 0
        self._events: dict[str, asyncio.Event] = {}
        self._results: dict[str, bool] = {}

    async def __call__(
        self,
        task_id: str,
        run_id: str,
        heavy_action_semaphore: asyncio.Semaphore,
        slot_release,
    ) -> TrialResult:
        self.semaphores.append(heavy_action_semaphore)
        self.started_order.append(task_id)
        self._inflight += 1
        self.peak = max(self.peak, self._inflight)
        self._events[run_id] = asyncio.Event()
        await self.started_queue.put((task_id, run_id))
        try:
            await self._events[run_id].wait()
        finally:
            self._inflight -= 1
        # Mimic the executor handing the slot back before teardown, twice, so a
        # non-idempotent release would over-admit and break the peak assertion.
        slot_release()
        slot_release()
        return _trial(run_id, solved=self._results[run_id])

    def finish(self, run_id: str, *, solved: bool) -> None:
        self._results[run_id] = solved
        self._events[run_id].set()


def test_run_tasks_composes_the_admission_scheduler(tmp_path: Path) -> None:
    # alpha (longest, budget 1) fails -> confirm-on-fail expands toward full=3,
    # then majority early-stop cancels the 3rd once two fails decide it.
    # beta/gamma (budget 3) each stop at 2 of 3 once two solves agree.
    priors = {"alpha": 300.0, "beta": 200.0, "gamma": 100.0}
    stub = _ScriptedExecutor()
    outcomes = {
        "alpha": deque([False, False]),
        "beta": deque([True, True]),
        "gamma": deque([True, True]),
    }

    async def scenario() -> ExperimentResult:
        run = asyncio.create_task(
            run_tasks(
                experiment_id="exp-compose",
                git_commit_hash="deadbeef",
                task_ids=["alpha", "beta", "gamma"],
                budget={"alpha": 1, "beta": 3, "gamma": 3},
                full_trial_count=3,
                max_trial_concurrency=2,
                max_heavy_action_concurrency=2,
                trial_runner=stub,
                experiments_root=tmp_path,
                started_at="2026-01-01T00:00:00+00:00",
                duration_priors=priors,
            )
        )
        # Release each started trial with its next scripted outcome; extras
        # (the trial majority-cancels) never get an outcome, so we skip them.
        released = 0
        while released < 6:
            task_id, run_id = await stub.started_queue.get()
            queue = outcomes[task_id]
            if queue:
                stub.finish(run_id, solved=queue.popleft())
                released += 1
        return await run

    result = asyncio.run(scenario())

    # LPT (opt #4): the two longest tasks take the two slots first; gamma waits.
    assert stub.started_order[:2] == ["alpha", "beta"]
    # Trial cap (opt #1) holds even though every trial double-releases its slot
    # (opt #3 idempotency): a non-idempotent release would over-admit past 2.
    assert stub.peak == 2
    # One run-scoped heavy gate threaded to every trial (opt #1 inner / #10).
    assert len({id(sem) for sem in stub.semaphores}) == 1

    # confirm-on-fail (opt #7): budget-1 alpha expanded and ran a 2nd trial,
    # then majority early-stop (opt #6) shrank back to what actually ran.
    alpha = result.tasks["alpha"]
    assert len(alpha.trials) == 2
    assert alpha.solved_count == 0 and alpha.majority_solved is False
    assert alpha.expected_trial_count == 2

    # majority early-stop on agreement: beta/gamma each stop at 2 of 3.
    for task_id in ("beta", "gamma"):
        task = result.tasks[task_id]
        assert len(task.trials) == 2
        assert task.majority_solved is True
        assert task.expected_trial_count == 2

    # mechanical run status finalized.
    assert result.run_status == "completed"
    assert result.finished_at is not None


# --- priority slot reservation (the one stateful piece) ---------------------


def test_run_tasks_reserves_a_freed_slot_for_a_higher_priority_task(
    tmp_path: Path,
) -> None:
    # cap=1. "high" (longest -> priority 0) passes once then fails, so it is not
    # majority-decided until 3 fails of 5 and keeps wanting trials. Each time it
    # frees the single slot the gate must RESERVE it for high's next trial over
    # the waiting lower-priority "low" -- so high runs to its decision (4 trials)
    # before low runs at all. Without speculative reservation, low would steal
    # the slot the instant high releases it (before high re-admits).
    priors = {"high": 100.0, "low": 1.0}
    call_log: list[str] = []
    high_outcomes = iter([True, False, False, False])

    async def trial_runner(task_id, run_id, heavy_action_semaphore, slot_release):
        await asyncio.sleep(0)  # yield so the gate's grant pass interleaves
        call_log.append(task_id)
        solved = next(high_outcomes) if task_id == "high" else True
        return _trial(run_id, solved=solved)

    async def go():
        # wait_for guards the wakeup path: a reserved slot is only re-granted to
        # the parked lower-priority waiter when the higher-priority task's result
        # is appended and re-runs admission (`commit_record_change`). If that
        # post-append re-grant regressed, "low" would never wake -> deadlock; the
        # timeout surfaces it as a failure instead of hanging the suite.
        return await asyncio.wait_for(
            run_tasks(
                experiment_id="exp-reserve",
                git_commit_hash="c0ffee",
                task_ids=["high", "low"],
                budget={"high": 5, "low": 5},
                full_trial_count=5,
                max_trial_concurrency=1,
                max_heavy_action_concurrency=1,
                trial_runner=trial_runner,
                experiments_root=tmp_path,
                started_at="2026-01-01T00:00:00+00:00",
                duration_priors=priors,
            ),
            timeout=5,
        )

    result = asyncio.run(go())

    assert call_log[:4] == ["high"] * 4
    high = result.tasks["high"]
    assert len(high.trials) == 4
    assert high.majority_solved is False


# --- the append (train -> veto) contract ------------------------------------


def test_run_tasks_appends_a_panel_without_disturbing_prior_tasks(
    tmp_path: Path,
) -> None:
    async def always_solve(task_id, run_id, heavy_action_semaphore, slot_release):
        return _trial(run_id, solved=True)

    common = dict(
        experiment_id="exp-append",
        git_commit_hash="abc123",
        full_trial_count=1,
        max_trial_concurrency=4,
        max_heavy_action_concurrency=2,
        trial_runner=always_solve,
        experiments_root=tmp_path,
        started_at="2026-01-01T00:00:00+00:00",
    )

    first = asyncio.run(
        run_tasks(
            task_ids=["train-a", "train-b"],
            budget={"train-a": 1, "train-b": 1},
            **common,
        )
    )
    assert set(first.tasks) == {"train-a", "train-b"}
    assert first.run_status == "completed" and first.finished_at is not None

    # Same experiment_id -> the veto panel appends, preserving the train tasks
    # and transitioning run_status running -> completed again.
    second = asyncio.run(run_tasks(task_ids=["test-a"], budget={"test-a": 1}, **common))
    assert set(second.tasks) == {"train-a", "train-b", "test-a"}
    assert second.run_status == "completed" and second.finished_at is not None
    assert len(second.tasks["train-a"].trials) == 1

    # The single experiment.json was appended in place, not replaced.
    reloaded = ExperimentResult.load("exp-append", root=tmp_path)
    assert set(reloaded.tasks) == {"train-a", "train-b", "test-a"}


# --- pure scheduling helpers ------------------------------------------------


def test_schedule_order_orders_by_descending_duration_prior() -> None:
    priors = {"short": 1.0, "long": 100.0, "mid": 10.0}
    assert _schedule_order(["short", "long", "mid"], priors) == [
        "long",
        "mid",
        "short",
    ]


def test_schedule_order_keeps_config_order_without_priors() -> None:
    assert _schedule_order(["a", "b", "c"], {}) == ["a", "b", "c"]


def test_wants_confirmation_expand_only_for_failed_single_deterministic_trial() -> None:
    fail_single = TaskResult(
        expected_trial_count=1, trials=[_trial("r0", solved=False)]
    )
    assert _wants_confirmation_expand(fail_single, full_trial_count=3) is True
    # full == 1: there is nothing to expand to.
    assert _wants_confirmation_expand(fail_single, full_trial_count=1) is False

    solved_single = TaskResult(
        expected_trial_count=1, trials=[_trial("r0", solved=True)]
    )
    assert _wants_confirmation_expand(solved_single, full_trial_count=3) is False

    multi_budget = TaskResult(
        expected_trial_count=3, trials=[_trial("r0", solved=False)]
    )
    assert _wants_confirmation_expand(multi_budget, full_trial_count=3) is False


@pytest.mark.parametrize(
    ("solved", "finished", "expected_total", "want"),
    [
        (0, 0, 3, 2),  # fresh: admit up to the majority threshold
        (1, 1, 3, 1),  # one solve in: one more can decide
        (1, 2, 3, 1),  # split: the deciding trial
        (2, 2, 3, 0),  # majority-solved locked -> stop
        (0, 2, 3, 0),  # majority-failed locked -> stop
    ],
)
def test_next_trial_admission_count(
    solved: int, finished: int, expected_total: int, want: int
) -> None:
    assert (
        _next_trial_admission_count(
            solved=solved, finished=finished, expected_total=expected_total
        )
        == want
    )


def test_planned_admission_count_honors_pending_confirmation_expand() -> None:
    # The record still reads expected_trial_count == 1, but the failed single
    # deterministic trial pends an expand to full=3, so planned admission targets
    # the larger budget (one more trial, toward the majority threshold).
    fail_single = TaskResult(
        expected_trial_count=1, trials=[_trial("r0", solved=False)]
    )
    assert _planned_admission_count(fail_single, full_trial_count=3) == 1

    # A solved single deterministic trial is finished -> nothing more planned.
    solved_single = TaskResult(
        expected_trial_count=1, trials=[_trial("r0", solved=True)]
    )
    assert _planned_admission_count(solved_single, full_trial_count=3) == 0
