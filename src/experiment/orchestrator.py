"""Run many trials concurrently -> a raw ``ExperimentResult``.

`run_tasks` is the shared core of both entry points (`uv run exp` and `auto`).
It is **gate-free, baseline-free, decision-free, and git-free** (plan.md §8): it
owns only concurrency, scheduling, aggregation, and persistence via the
``writer``. The per-trial unit it calls is injected as a ``trial_runner`` so the
scheduling can be tested with a scripted stub; the real wiring (env/llm/Harbor +
``executor.run_trial``) is assembled by the caller (the cli, Step 2f).

Scheduling is all here. Its pure decisions -- LPT launch order, the admission
arithmetic, the confirm-on-fail predicate -- are module-level helpers unit-tested
without the async loop. Its one stateful piece, priority admission with
speculative slot reservation, lives in the loop because it reasons about live
free slots. Two concurrency levels are decoupled (plan.md opt #1): the trial cap
(``_PriorityTrialGate``) admits whole trials; the run-scoped heavy-action
semaphore (opt #10), threaded to every trial, bounds container CPU work beneath
it so trials idling on the LLM overlap without oversubscribing cores.
"""

from __future__ import annotations

import asyncio
import heapq
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from src.contracts import is_majority_decided
from src.experiment.record import ExperimentResult, TaskResult, TrialResult
from src.experiment.writer import write_experiment_result

DEFAULT_TASK_DURATION_PRIORS_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "task_duration_priors.json"
)

# The per-trial unit the orchestrator schedules. The real implementation (Step
# 2f) closes over the executor + env/llm factories and calls
# `executor.run_trial`; tests inject a stub that scripts outcomes. Arguments are
# `(task_id, run_id, heavy_action_semaphore, slot_release)`: the semaphore is the
# run-scoped CPU gate to hand to the env; `slot_release` frees the trial slot the
# moment the result is final, before env teardown (opt #3).
TrialRunner = Callable[
    [str, str, asyncio.Semaphore, Callable[[], None]],
    Awaitable[TrialResult],
]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- pure scheduling helpers ------------------------------------------------


def _load_task_duration_priors(
    path: Path = DEFAULT_TASK_DURATION_PRIORS_PATH,
) -> dict[str, float]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {
        str(task_id): float(seconds)
        for task_id, seconds in payload["task_duration_seconds"].items()
    }


def _schedule_order(
    task_names: Sequence[str],
    duration_priors: Mapping[str, float] | None = None,
) -> list[str]:
    """Longest-task-first (LPT) launch order to minimize panel makespan.

    Trials acquire the trial gate FIFO in the order tasks are launched, and
    makespan on a fixed worker pool is tail-dominated: a long task admitted late
    hangs off the end with nothing to overlap it. Ordering by historical
    per-task wall-time (descending) starts the long poles first so short tasks
    fill the tail. Reorders only the launch sequence -- the (task-id-keyed)
    record is order-independent, so outcomes are untouched. Falls back to config
    order for tasks absent from the prior.
    """
    names = list(task_names)
    resolved_priors = (
        _load_task_duration_priors() if duration_priors is None else duration_priors
    )
    # Stable sort (reverse=True keeps equal-cost ties in config order).
    return sorted(
        names, key=lambda task_id: resolved_priors.get(task_id, 0.0), reverse=True
    )


def _next_trial_admission_count(
    *,
    solved: int,
    finished: int,
    expected_total: int,
) -> int:
    if is_majority_decided(
        solved=solved, finished=finished, expected_total=expected_total
    ):
        return 0
    final_threshold = (expected_total + 1) // 2
    failed = finished - solved
    return min(
        expected_total - finished,
        final_threshold - solved,
        final_threshold - failed,
    )


def _wants_confirmation_expand(
    task_result: TaskResult,
    *,
    full_trial_count: int,
) -> bool:
    # A task on the single-trial deterministic budget whose one trial just
    # failed: expand back to the full trial count to confirm the suspected
    # regression. Single definition of the rule -- the admission loop commits the
    # expand to the record, the priority gate reads it speculatively to decide
    # whether a higher-priority task still wants a slot. One copy keeps the two
    # consumers from drifting apart.
    return (
        task_result.expected_trial_count == 1
        and full_trial_count > 1
        and task_result.solved_count == 0
        and len(task_result.valid_trials) == 1
    )


def _effective_expected_count(
    task_result: TaskResult,
    *,
    full_trial_count: int,
) -> int:
    # The budget a task is actually working toward: its committed
    # expected_trial_count, or full_trial_count when a failed single
    # deterministic trial is pending a confirmation expand the admission loop has
    # not committed yet. Folding the pending expand into one rule lets every
    # caller pass only config (full_trial_count).
    if _wants_confirmation_expand(task_result, full_trial_count=full_trial_count):
        return full_trial_count
    return task_result.expected_trial_count


def _planned_admission_count(
    task_result: TaskResult,
    *,
    full_trial_count: int,
) -> int:
    # How many trials to admit toward the effective budget right now: the
    # remaining budget, capped by the majority-threshold rule. Shared by the
    # admission loop and the priority gate so the admission arithmetic lives in
    # one place.
    expected_total = _effective_expected_count(
        task_result, full_trial_count=full_trial_count
    )
    remaining = expected_total - len(task_result.trials)
    if remaining <= 0:
        return 0
    return min(
        remaining,
        _next_trial_admission_count(
            solved=task_result.solved_count,
            finished=len(task_result.valid_trials),
            expected_total=expected_total,
        ),
    )


def _set_trial_budget(task_result: TaskResult, *, expected_trial_count: int) -> bool:
    if expected_trial_count < len(task_result.trials):
        raise ValueError("trial budget cannot be below recorded trials")
    if task_result.expected_trial_count == expected_trial_count:
        return False
    task_result.expected_trial_count = expected_trial_count
    return True


# --- stateful priority admission --------------------------------------------


class _PriorityTrialGate:
    """Trial-concurrency cap (opt #1 outer) with priority slot reservation (opt
    #5): when a higher-priority task still wants a slot, a free slot is reserved
    for it rather than handed to a lower-priority waiter that arrived first.
    Priority is the LPT launch index (0 = longest task)."""

    def __init__(
        self,
        *,
        capacity: int,
        priority_by_task: Mapping[str, int],
        planned_admission_count_by_task: Callable[[str], int],
    ) -> None:
        self._available = capacity
        self._priority_by_task = priority_by_task
        self._task_by_priority = {
            priority: task_id for task_id, priority in priority_by_task.items()
        }
        self._planned_admission_count_by_task = planned_admission_count_by_task
        self._next_sequence = 0
        self._admitted_by_task = {task_id: 0 for task_id in priority_by_task}
        self._waiters: list[tuple[int, int, str, asyncio.Future[None]]] = []
        self._grant_scheduled = False

    async def acquire(self, task_id: str) -> None:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._admitted_by_task[task_id] += 1
        heapq.heappush(
            self._waiters,
            (self._priority_by_task[task_id], self._next_sequence, task_id, future),
        )
        self._next_sequence += 1
        self._schedule_grant()
        try:
            await future
        except BaseException:
            if future.done() and not future.cancelled():
                self.release(task_id)
            else:
                future.cancel()
                self._admitted_by_task[task_id] -= 1
                self._schedule_grant()
            raise

    def release(self, task_id: str) -> None:
        self._admitted_by_task[task_id] -= 1
        self._available += 1
        self._schedule_grant()

    def schedule_grant(self) -> None:
        # Public alias: a record mutation that is not itself a capacity event
        # (a trial result, an expected-count change) can unblock a waiter the
        # gate reserved a slot for, so the loop re-runs admission after every
        # change.
        self._schedule_grant()

    def _schedule_grant(self) -> None:
        if self._grant_scheduled:
            return
        self._grant_scheduled = True
        asyncio.get_running_loop().call_soon(self._grant_waiters)

    def _grant_waiters(self) -> None:
        self._grant_scheduled = False
        while self._available > 0 and self._waiters:
            priority, _, _, future = self._waiters[0]
            if future.done():
                heapq.heappop(self._waiters)
                continue
            if self._has_unmet_higher_priority_admission(priority):
                break
            heapq.heappop(self._waiters)
            self._available -= 1
            future.set_result(None)

    def _has_unmet_higher_priority_admission(self, priority: int) -> bool:
        for higher_priority in range(priority):
            task_id = self._task_by_priority[higher_priority]
            desired = self._planned_admission_count_by_task(task_id)
            if desired > self._admitted_by_task[task_id]:
                return True
        return False


# --- the concurrency loop ---------------------------------------------------


async def _run_panel(
    *,
    task_results: dict[str, TaskResult],
    task_order: Sequence[str],
    trial_runner: TrialRunner,
    max_trial_concurrency: int,
    max_heavy_action_concurrency: int,
    full_trial_count: int,
    persist: Callable[[], None],
) -> None:
    """Schedule the tasks in ``task_order`` to completion, mutating their
    ``TaskResult``s in ``task_results`` in place. ``persist`` writes the
    enclosing ``ExperimentResult`` after each change."""

    def planned_admission_count_by_task(task_id: str) -> int:
        return _planned_admission_count(
            task_results[task_id], full_trial_count=full_trial_count
        )

    trial_gate = _PriorityTrialGate(
        capacity=max_trial_concurrency,
        priority_by_task={task_id: index for index, task_id in enumerate(task_order)},
        planned_admission_count_by_task=planned_admission_count_by_task,
    )
    # One run-scoped heavy-action gate, shared across every trial's env (opt
    # #1 inner / #10): the trial cap can run ahead of host-CPU capacity because
    # this bounds the concurrent heavyweight container work beneath it.
    heavy_action_semaphore = asyncio.Semaphore(max_heavy_action_concurrency)

    def commit_record_change() -> None:
        # Persist and re-run admission together: no capacity event is guaranteed
        # to follow a record change, so every mutation must re-run admission via
        # the gate, else a waiter unblocked purely by the change is never woken.
        persist()
        trial_gate.schedule_grant()

    async def run_one_trial(task_id: str, run_id: str) -> None:
        # Manual acquire/release (not `async with`) so the executor can hand the
        # slot back the moment its result is final -- before its env teardown --
        # via `release_slot`. `release_slot` is idempotent (opt #3): the executor
        # calls it inside its own finally; this `finally` is the safety net for
        # paths that never reach the executor's release (e.g. cancellation
        # before it starts).
        await trial_gate.acquire(task_id)
        slot_released = False

        def release_slot() -> None:
            nonlocal slot_released
            if not slot_released:
                slot_released = True
                trial_gate.release(task_id)

        try:
            trial_result = await trial_runner(
                task_id, run_id, heavy_action_semaphore, release_slot
            )
            task_results[task_id].append(trial_result)
            commit_record_change()
        finally:
            release_slot()

    async def run_task_trials(task_id: str) -> None:
        sequence = 0
        admit_all_remaining = False
        while not task_results[task_id].is_finished:
            task_result = task_results[task_id]
            remaining_budget = task_result.expected_trial_count - len(
                task_result.trials
            )
            if remaining_budget <= 0:
                break
            if admit_all_remaining:
                admission_count = remaining_budget
                admit_all_remaining = False
            else:
                admission_count = _planned_admission_count(
                    task_result, full_trial_count=full_trial_count
                )
            if admission_count <= 0:
                break

            in_flight: set[asyncio.Task[None]] = set()
            for _ in range(admission_count):
                run_id = f"{task_id}-{sequence}"
                sequence += 1
                in_flight.add(asyncio.create_task(run_one_trial(task_id, run_id)))

            try:
                while in_flight:
                    done, in_flight = await asyncio.wait(
                        in_flight, return_when=asyncio.FIRST_COMPLETED
                    )
                    for finished in done:
                        finished.result()
                    task_result = task_results[task_id]
                    if _wants_confirmation_expand(
                        task_result, full_trial_count=full_trial_count
                    ):
                        # A failed single deterministic trial: expand to the full
                        # budget and admit the rest to confirm the regression.
                        _set_trial_budget(
                            task_result, expected_trial_count=full_trial_count
                        )
                        admit_all_remaining = True
                        commit_record_change()
                        break
                    decided = is_majority_decided(
                        solved=task_result.solved_count,
                        finished=len(task_result.valid_trials),
                        expected_total=task_result.expected_trial_count,
                    )
                    if decided and not task_result.is_finished:
                        # Majority locked in early (opt #6): cancel the pending
                        # siblings and shrink the budget to what actually ran.
                        for pending in in_flight:
                            pending.cancel()
                        if in_flight:
                            await asyncio.gather(*in_flight, return_exceptions=True)
                        in_flight = set()
                        _set_trial_budget(
                            task_result,
                            expected_trial_count=len(task_result.trials),
                        )
                        commit_record_change()
            except BaseException:
                for pending in in_flight:
                    pending.cancel()
                if in_flight:
                    await asyncio.gather(*in_flight, return_exceptions=True)
                raise

    await asyncio.gather(*(run_task_trials(task_id) for task_id in task_order))


# --- public entry point -----------------------------------------------------


async def run_tasks(
    *,
    experiment_id: str,
    git_commit_hash: str,
    task_ids: Sequence[str],
    budget: Mapping[str, int],
    full_trial_count: int,
    max_trial_concurrency: int,
    max_heavy_action_concurrency: int,
    trial_runner: TrialRunner,
    experiments_root: Path,
    started_at: str | None = None,
    duration_priors: Mapping[str, float] | None = None,
) -> ExperimentResult:
    """Run ``task_ids`` (each to its ``budget`` trial count) -> ``ExperimentResult``.

    Fresh run: writes a ``running`` record, schedules the panel, finalizes
    ``completed``. Append (the train->veto contract, plan.md §2/§7): an existing
    record at ``experiment_id`` is loaded, reopened ``running``, the new
    ``task_ids`` appended alongside the tasks already there, then finalized
    ``completed`` again. A panel-level exception marks the record ``crashed``
    (mechanical run_status, never a keep/discard decision -- that is the auto
    layer's, in loop.json) and re-raises.
    """
    path = ExperimentResult.path(experiment_id, root=experiments_root)
    if path.exists():
        result = ExperimentResult.load(experiment_id, root=experiments_root)
        # Reopen for the appended panel; the timestamp invariant requires
        # `running` carry no `finished_at`.
        result.run_status = "running"
        result.finished_at = None
    else:
        result = ExperimentResult(
            experiment_id=experiment_id,
            git_commit_hash=git_commit_hash,
            run_status="running",
            started_at=_utcnow_iso() if started_at is None else started_at,
            tasks={},
        )
    for task_id in task_ids:
        result.tasks[task_id] = TaskResult.empty(expected_trial_count=budget[task_id])

    def persist() -> None:
        write_experiment_result(result, root=experiments_root)

    persist()
    try:
        await _run_panel(
            task_results=result.tasks,
            task_order=_schedule_order(task_ids, duration_priors),
            trial_runner=trial_runner,
            max_trial_concurrency=max_trial_concurrency,
            max_heavy_action_concurrency=max_heavy_action_concurrency,
            full_trial_count=full_trial_count,
            persist=persist,
        )
    except BaseException:
        result.run_status = "crashed"
        result.finished_at = _utcnow_iso()
        persist()
        raise
    result.run_status = "completed"
    result.finished_at = _utcnow_iso()
    persist()
    return result
