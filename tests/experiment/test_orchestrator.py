"""Tests for src/experiment/orchestrator.py (run_tasks + scheduling).

One scripted stub-executor scenario proves the admission scheduler composes (LPT
order, trial cap, shared heavy gate, confirm-on-fail, majority early-stop,
idempotent slot release); a focused deterministic test pins the one stateful
piece (the LPT-priority slot grant); the append test pins the train->veto
contract; the rest are pure-helper unit tests. The executor is never touched --
a stub ``trial_runner`` scripts outcomes directly (plan.md §8).
"""

from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path

import pytest

from src.experiment.orchestrator import (
    _next_trial_admission_count,
    _planned_admission_count,
    _PriorityTrialGate,
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


# --- the LPT-priority slot grant (the one stateful piece) --------------------


def test_trial_gate_grants_a_freed_slot_by_lpt_priority_not_arrival_order() -> None:
    # A high-priority (longest-task) waiter that queues AFTER two low-priority
    # waiters still wins the next freed slot: that is what lets a long task's
    # next admission wave overtake the short-task backlog. Free slots are never
    # held back for demand that has not queued (no speculative reservation), so
    # the first low waiter takes the slot while nothing better is queued.
    async def scenario() -> list[str]:
        gate = _PriorityTrialGate(capacity=1, priority_by_task={"high": 0, "low": 1})
        order: list[str] = []
        release_holder = asyncio.Event()

        async def holder() -> None:
            await gate.acquire("low")
            order.append("low-holder")
            await release_holder.wait()
            gate.release()

        async def waiter(task_id: str, label: str) -> None:
            await gate.acquire(task_id)
            order.append(label)
            gate.release()

        holding = asyncio.create_task(holder())
        await asyncio.sleep(0)  # low-holder takes the slot (no better waiter)
        low_waiter = asyncio.create_task(waiter("low", "low-waiter"))
        await asyncio.sleep(0)  # low-waiter queues first...
        high_waiter = asyncio.create_task(waiter("high", "high-waiter"))
        await asyncio.sleep(0)  # ...high-waiter queues second
        release_holder.set()
        await asyncio.gather(holding, low_waiter, high_waiter)
        return order

    assert asyncio.run(scenario()) == ["low-holder", "high-waiter", "low-waiter"]


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


# --- the progress hook ------------------------------------------------------


def test_run_tasks_invokes_on_progress_with_the_live_task_mapping(
    tmp_path: Path,
) -> None:
    # on_progress fires from the persist hook with the live task mapping: once on
    # the initial empty record (both tasks present, none finished) and a final
    # time with every task finished. The orchestrator itself prints nothing.
    async def always_solve(task_id, run_id, heavy_action_semaphore, slot_release):
        return _trial(run_id, solved=True)

    snapshots: list[tuple[int, int]] = []

    def on_progress(task_results) -> None:
        done = sum(1 for t in task_results.values() if t.is_finished)
        snapshots.append((done, len(task_results)))

    result = asyncio.run(
        run_tasks(
            experiment_id="exp-progress",
            git_commit_hash="abc123",
            task_ids=["a", "b"],
            budget={"a": 1, "b": 1},
            full_trial_count=1,
            max_trial_concurrency=2,
            max_heavy_action_concurrency=2,
            trial_runner=always_solve,
            experiments_root=tmp_path,
            started_at="2026-01-01T00:00:00+00:00",
            on_progress=on_progress,
        )
    )

    assert result.run_status == "completed"
    assert snapshots, "on_progress never fired"
    assert snapshots[0] == (0, 2)  # initial persist: both tasks present, none done
    assert snapshots[-1] == (2, 2)  # terminal persist: all finished


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


def test_planned_admission_count_targets_the_committed_budget() -> None:
    # A fresh budget-3 task plans up to the majority threshold (2), not the
    # whole budget; an exhausted budget plans nothing. Pending confirmation
    # expands are the admission loop's job (it commits the new budget before
    # planning again), so this helper reads only the committed record.
    fresh = TaskResult(expected_trial_count=3, trials=[])
    assert _planned_admission_count(fresh) == 2

    solved_single = TaskResult(
        expected_trial_count=1, trials=[_trial("r0", solved=True)]
    )
    assert _planned_admission_count(solved_single) == 0
