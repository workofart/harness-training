"""Experiment run orchestration.

Drives a candidate or baseline experiment: prepares task dirs, runs the trial
panel concurrently, applies the promotion gate, and concludes by updating run
state / git refs. The persisted record types live in ``record.py`` and the gate
statistics in ``gate.py``.
"""

from __future__ import annotations

import asyncio
import heapq
import json
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

from src.control import repo as control_repo
from src.experiment.trial import run_task
from src.harness.contracts import TaskResult
from src.metrics import BaselineComparison, is_majority_decided

from src.experiment.gate import (
    build_baseline_pool,
    build_gate_verdicts,
    decide_panel_from_verdicts,
)
from src.experiment.record import (
    ExperimentRecord,
    ExperimentStatus,
    ExperimentState,
    PanelEvaluation,
    PanelLifecycle,
    PanelPurpose,
    PanelRecord,
    TaskTrials,
    failed_experiment_git_ref,
    raise_if_no_valid_evidence,
    terminal_task_result,
)

if TYPE_CHECKING:
    from src.adapters.env import HarborConfig
    from src.harness.config import HarnessConfig, LlmProviderConfig


DEFAULT_TASK_DURATION_PRIORS_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "task_duration_priors.json"
)


@dataclass(frozen=True)
class PanelSpec:
    panel_id: str
    purpose: PanelPurpose
    task_names: tuple[str, ...]
    task_timeout_sec: float
    initial_lifecycle: PanelLifecycle
    requires_baseline: bool
    after_panel: str | None
    when_status: ExperimentStatus | None


def _compile_panel_specs(harness_config: HarnessConfig) -> tuple[PanelSpec, ...]:
    specs: list[PanelSpec] = []
    for panel in harness_config.panels:
        if panel.run.when == "always":
            initial_lifecycle: PanelLifecycle = "active"
            after_panel = None
            when_status = None
        else:
            initial_lifecycle = "pending"
            after_panel = panel.run.after_panel
            when_status = panel.run.when_status

        specs.append(
            PanelSpec(
                panel_id=panel.id,
                purpose=panel.purpose,
                task_names=tuple(panel.task_names),
                task_timeout_sec=panel.task_timeout_sec,
                initial_lifecycle=initial_lifecycle,
                requires_baseline=panel.baseline.required,
                after_panel=after_panel,
                when_status=when_status,
            )
        )
    return tuple(specs)


async def _prepare_task_dirs(
    *,
    trial_harbor_config: HarborConfig,
    task_names: Sequence[str],
) -> dict[str, Path]:
    from src.adapters.env import TaskDirectoryResolver

    return dict(
        await TaskDirectoryResolver(trial_harbor_config).resolve(list(task_names))
    )


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

    Trials acquire the trial semaphore FIFO in the order tasks are launched, and
    makespan on a fixed worker pool is tail-dominated: a long task admitted late
    hangs off the end with nothing to overlap it. Ordering by historical
    per-task wall-time (descending) starts the long poles first so short tasks
    fill the tail. Reorders only the launch sequence — persisted panel task ids, the
    record, and the (task-id-keyed) gate are order-independent, so outcomes are
    untouched. Falls back to config order for tasks absent from the prior.
    """
    names = list(task_names)
    resolved_priors = (
        _load_task_duration_priors() if duration_priors is None else duration_priors
    )

    # Stable sort (reverse=True keeps equal-cost ties in config order).
    return sorted(
        names, key=lambda task_id: resolved_priors.get(task_id, 0.0), reverse=True
    )


def _panel_record_for_spec(
    *,
    spec: PanelSpec,
    expected_trial_count: int,
    lifecycle: PanelLifecycle | None = None,
) -> PanelRecord:
    return PanelRecord.initialize(
        panel_id=spec.panel_id,
        purpose=spec.purpose,
        task_ids=spec.task_names,
        expected_trial_count=expected_trial_count,
        lifecycle=spec.initial_lifecycle if lifecycle is None else lifecycle,
    )


def _record_matches_panel_specs(
    record: ExperimentRecord,
    panel_specs: Sequence[PanelSpec],
) -> bool:
    if set(record.panel_order) != {spec.panel_id for spec in panel_specs}:
        return False
    return all(
        set(record.panels[spec.panel_id].task_ids) == set(spec.task_names)
        for spec in panel_specs
    )


PROGRESS_BAR_WIDTH = 24


class _PriorityTrialGate:
    def __init__(
        self,
        *,
        capacity: int,
        priority_by_task: Mapping[str, int],
        record: ExperimentRecord,
        full_trial_count: int,
    ) -> None:
        self._available = capacity
        self._priority_by_task = priority_by_task
        self._task_by_priority = {
            priority: task_id for task_id, priority in priority_by_task.items()
        }
        self._record = record
        self._full_trial_count = full_trial_count
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
            trials = self._record._task_trials(task_id)
            desired = _planned_admission_count(
                trials, full_trial_count=self._full_trial_count
            )
            if desired > self._admitted_by_task[task_id]:
                return True
        return False


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
    trials: TaskTrials,
    *,
    full_trial_count: int,
) -> bool:
    # A task on the single-trial deterministic budget whose one trial just
    # failed: expand back to the full trial count to confirm the suspected
    # regression. Single definition of the rule -- the admission loop commits
    # the expand to the record, the priority gate reads it speculatively to
    # decide whether a higher-priority task still wants a slot. One copy keeps
    # the two consumers from drifting apart.
    return (
        trials.expected_trial_count == 1
        and full_trial_count > 1
        and trials.solved_count == 0
        and len(trials.valid_trials) == 1
    )


def _effective_expected_count(
    trials: TaskTrials,
    *,
    full_trial_count: int,
) -> int:
    # The budget a task is actually working toward: its committed
    # expected_trial_count, or full_trial_count when a failed single
    # deterministic trial is pending a confirmation expand the admission loop
    # has not committed yet. Folding the pending expand into one rule lets every
    # caller pass only config (full_trial_count) -- no caller has to override
    # the object's own count with a speculative total.
    if _wants_confirmation_expand(trials, full_trial_count=full_trial_count):
        return full_trial_count
    return trials.expected_trial_count


def _planned_admission_count(
    trials: TaskTrials,
    *,
    full_trial_count: int,
) -> int:
    # How many trials to admit toward the effective budget right now: the
    # remaining budget, capped by the majority-threshold rule. Shared by the
    # admission loop and the priority gate so the admission arithmetic lives in
    # one place.
    expected_total = _effective_expected_count(
        trials, full_trial_count=full_trial_count
    )
    remaining = expected_total - len(trials.finished_trials)
    if remaining <= 0:
        return 0
    return min(
        remaining,
        _next_trial_admission_count(
            solved=trials.solved_count,
            finished=len(trials.valid_trials),
            expected_total=expected_total,
        ),
    )


def _format_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def format_panel_progress(
    *,
    tasks_done: int,
    total_tasks: int,
    trials_done: int,
    trials_planned: int,
    solved: int,
    decided: int,
    error_trials: int,
    in_flight: int,
    elapsed_sec: float,
) -> str:
    """Render one progress line for a panel run.

    Anchored on task completion (`tasks_done/total_tasks`) because that
    denominator is fixed and the bar never moves backward. `trials_planned`
    is the dynamic per-task budget sum, which shrinks as the majority decides
    early and grows on candidate confirm-on-fail, so trials are reported as
    detail text rather than driving the bar. `solved`/`decided` mirror the
    record's live counts: `decided` is tasks with at least one valid trial.
    ETA divides remaining tasks by the observed task-completion wall rate,
    which already absorbs the configured concurrency. The progress line is
    event-driven: elapsed/ETA refresh only when a task result is persisted.
    """
    frac = tasks_done / total_tasks if total_tasks else 0.0
    filled = int(frac * PROGRESS_BAR_WIDTH)
    bar = "#" * filled + "-" * (PROGRESS_BAR_WIDTH - filled)
    if tasks_done > 0 and elapsed_sec > 0:
        rate = tasks_done / elapsed_sec
        eta = f"~{_format_hms((total_tasks - tasks_done) / rate)} left"
    else:
        eta = "~-- left"
    return (
        f"[{bar}] {tasks_done}/{total_tasks} tasks ({frac * 100:.0f}%) | "
        f"trials {trials_done}/{trials_planned} | "
        f"solved {solved}/{decided} | errors {error_trials} | active {in_flight} | "
        f"{_format_hms(elapsed_sec)} elapsed, {eta}"
    )


class PanelProgressReporter:
    """Live single-line progress bar for a panel run.

    Active only when the target stream is an interactive TTY. The supervisor
    launches `uv run exp` with captured (piped) output, so this stays silent
    there and renders only for a direct interactive run.
    """

    def __init__(
        self,
        *,
        total_tasks: int,
        max_trial_concurrency: int,
        stream: Any | None = None,
    ) -> None:
        self._stream = sys.stderr if stream is None else stream
        self._total_tasks = total_tasks
        self._max_trial_concurrency = max_trial_concurrency
        self._enabled = bool(getattr(self._stream, "isatty", lambda: False)())
        self._start = time.monotonic()
        self._drawn = False
        self._finalized = False

    def render(
        self,
        record: ExperimentRecord | None = None,
        *,
        task_results: Mapping[str, TaskTrials] | None = None,
        solved_count: int | None = None,
    ) -> None:
        if not self._enabled or self._finalized:
            return
        if record is not None:
            first_panel = record.panels[record.panel_order[0]]
            task_results = first_panel.task_results
            solved_count = first_panel.solved_count
        if task_results is None:
            raise TypeError("task_results is required when record is omitted")
        trials = task_results.values()
        tasks_done = sum(1 for t in trials if t.is_finished)
        trials_done = sum(len(t.finished_trials) for t in trials)
        valid_done = sum(len(t.valid_trials) for t in trials)
        trials_planned = sum(t.expected_trial_count for t in trials)
        decided = sum(1 for t in trials if t.majority_solved is not None)
        in_flight = max(
            0, min(self._max_trial_concurrency, trials_planned - trials_done)
        )
        line = format_panel_progress(
            tasks_done=tasks_done,
            total_tasks=self._total_tasks,
            trials_done=trials_done,
            trials_planned=trials_planned,
            solved=solved_count or 0,
            decided=decided,
            error_trials=trials_done - valid_done,
            in_flight=in_flight,
            elapsed_sec=time.monotonic() - self._start,
        )
        self._stream.write("\r\033[K" + line)
        self._stream.flush()
        self._drawn = True
        if tasks_done >= self._total_tasks:
            self._finalize()

    def close(self) -> None:
        # Terminate the dangling bar line on the crash/cancel path so later
        # stdout lines don't append to it.
        self._finalize()

    def _finalize(self) -> None:
        if self._enabled and self._drawn and not self._finalized:
            self._stream.write("\n")
            self._stream.flush()
        self._finalized = True


async def _run_panel(
    *,
    record: ExperimentRecord,
    experiments_root: Path,
    task_names: Sequence[str],
    task_results: dict[str, TaskTrials],
    task_timeout_sec: float,
    task_dirs: Mapping[str, Path],
    harness_config: HarnessConfig,
    make_llm: Callable[[], Any],
    make_env: Callable[..., Any],
) -> None:
    trial_gate = _PriorityTrialGate(
        capacity=harness_config.max_trial_concurrency,
        priority_by_task={task_id: index for index, task_id in enumerate(task_names)},
        record=record,
        full_trial_count=harness_config.task_trials,
    )
    heavy_action_semaphore = asyncio.Semaphore(
        harness_config.max_heavy_action_concurrency
    )
    reporter = PanelProgressReporter(
        total_tasks=len(task_names),
        max_trial_concurrency=harness_config.max_trial_concurrency,
    )

    def commit_record_change() -> None:
        # Persist the record and re-run gate admission together. The gate
        # reserves slots for higher-priority tasks from their record state
        # (finished/expected counts), and no capacity event is guaranteed to
        # follow a record change -- so every in-panel mutation (a trial result,
        # an expected-count change) must re-run admission via _schedule_grant,
        # else a waiter unblocked purely by the change is never woken. One door
        # so a new mutation site can't write the record but forget the gate.
        record.write(root=experiments_root)
        trial_gate._schedule_grant()

    def persist_task_result(task_result: TaskResult) -> None:
        record.record_task_result(task_result)
        commit_record_change()
        reporter.render(
            task_results=task_results,
            solved_count=sum(
                1 for trials in task_results.values() if trials.majority_solved is True
            ),
        )

    async def run_one_trial(task_id: str) -> None:
        # Manual acquire/release (not `async with`) so `run_task` can hand the
        # slot back the moment its result is final — before its docker teardown
        # — letting the next trial start while this one tears down. `release_slot`
        # is idempotent; the `finally` is the safety net for paths that never
        # reach run_task's own release (e.g. the task_dir KeyError below).
        await trial_gate.acquire(task_id)
        slot_released = False

        def release_slot() -> None:
            nonlocal slot_released
            if not slot_released:
                slot_released = True
                trial_gate.release(task_id)

        try:
            task_dir = task_dirs[task_id]
            env = make_env(
                task_id, task_dir=task_dir, exec_semaphore=heavy_action_semaphore
            )
            try:
                trial_result = await run_task(
                    task_name=task_id,
                    llm=make_llm(),
                    env=env,
                    max_steps=harness_config.max_steps,
                    max_output_retries=harness_config.max_output_retries,
                    task_timeout_sec=task_timeout_sec,
                    env_setup_timeout_sec=harness_config.env_setup_timeout_sec,
                    slot_release=release_slot,
                )
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                persist_task_result(terminal_task_result(task_id=task_id, exc=exc))
                if isinstance(exc, Exception):
                    return
                raise
            persist_task_result(trial_result)
        finally:
            release_slot()

    async def run_task_trials(task_id: str) -> None:
        full_trial_count = harness_config.task_trials
        admit_all_confirmations = False
        while not record._task_trials(task_id).is_finished:
            trials = record._task_trials(task_id)
            expected_total = trials.expected_trial_count
            remaining_budget = expected_total - len(trials.finished_trials)
            if remaining_budget <= 0:
                break
            if admit_all_confirmations:
                admission_count = remaining_budget
                admit_all_confirmations = False
            else:
                admission_count = _planned_admission_count(
                    trials, full_trial_count=full_trial_count
                )
            if admission_count <= 0:
                break
            in_flight: set[asyncio.Task[None]] = {
                asyncio.create_task(run_one_trial(task_id))
                for _ in range(admission_count)
            }
            try:
                while in_flight:
                    done, in_flight = await asyncio.wait(
                        in_flight, return_when=asyncio.FIRST_COMPLETED
                    )
                    for finished in done:
                        finished.result()
                    trials = record._task_trials(task_id)
                    if _wants_confirmation_expand(
                        trials, full_trial_count=full_trial_count
                    ):
                        trials.expected_trial_count = full_trial_count
                        admit_all_confirmations = True
                        commit_record_change()
                        break
                    decided = is_majority_decided(
                        solved=trials.solved_count,
                        finished=len(trials.valid_trials),
                        expected_total=trials.expected_trial_count,
                    )
                    if decided and not trials.is_finished:
                        for pending in in_flight:
                            pending.cancel()
                        if in_flight:
                            await asyncio.gather(*in_flight, return_exceptions=True)
                        in_flight = set()
                        trials.expected_trial_count = len(trials.finished_trials)
                        commit_record_change()
            except BaseException:
                for pending in in_flight:
                    pending.cancel()
                if in_flight:
                    await asyncio.gather(*in_flight, return_exceptions=True)
                raise

    try:
        await asyncio.gather(*(run_task_trials(task_id) for task_id in task_names))
    finally:
        reporter.close()


def _make_llm_for_config(*, config: LlmProviderConfig, api_key: str | None):
    match config.provider:
        case "openrouter":
            if api_key is None:
                raise ValueError("OPENROUTER_API_KEY is not set")
            from src.adapters.open_router import OpenRouter

            return OpenRouter(config=config, api_key=api_key)
        case "chatgpt_codex":
            from src.adapters.chatgpt_codex import ChatGptCodex

            return ChatGptCodex(config=config)


class ExperimentRunner:
    @classmethod
    def run_baseline_at_head(
        cls,
        *,
        harness_config: HarnessConfig,
        harbor_config: HarborConfig,
        api_key: str | None,
        decision_reason: Literal["baseline seed", "baseline rerun"] = "baseline rerun",
        experiment_id: str | None = None,
        started_at: str | None = None,
        repo_root: Path | None = None,
    ) -> ExperimentRecord:
        """Run the current HEAD as the active baseline over the full panel."""
        experiments_root = harbor_config.experiments_dir
        panel_specs = _compile_panel_specs(harness_config)
        state = ExperimentState.load(root=experiments_root)
        baseline_id = state.active_baseline_experiment_id
        baseline = (
            None
            if baseline_id is None
            else ExperimentRecord.load(baseline_id, root=experiments_root)
        )
        if baseline is not None:
            if baseline.status != "keep" or not baseline.is_concluded():
                raise RuntimeError(
                    f"active baseline {baseline.experiment_id} must be a concluded keep record"
                )
        current_id = state.current_experiment_id
        if current_id is not None and current_id != baseline_id:
            current_record = ExperimentRecord.load(current_id, root=experiments_root)
            if not current_record.is_concluded() and any(
                task_trials.trial_count > 0
                for task_trials in current_record._all_task_results()
            ):
                raise RuntimeError(
                    "cannot run baseline while the current experiment has recorded task activity"
                )
        control_repo.require_clean_worktree(cwd=repo_root)
        git_commit_hash = control_repo.get_head_commit(cwd=repo_root)
        if baseline is not None:
            if (
                baseline.git_commit_hash == git_commit_hash
                and _record_matches_panel_specs(baseline, panel_specs)
            ):
                state.current_experiment_id = baseline.experiment_id
                state.updated_at = datetime.now(timezone.utc).isoformat()
                state.save(root=experiments_root)
                return baseline

        baseline_at = (
            datetime.now(timezone.utc).isoformat() if started_at is None else started_at
        )
        resolved_experiment_id = (
            f"baseline-{datetime.fromisoformat(baseline_at).strftime('%Y%m%d-%H%M%S')}"
            if experiment_id is None
            else experiment_id
        )
        if ExperimentRecord.path(
            resolved_experiment_id, root=experiments_root
        ).exists():
            raise RuntimeError(
                f"baseline experiment already exists: {resolved_experiment_id}"
            )
        record = ExperimentRecord.initialize(
            experiment_id=resolved_experiment_id,
            git_commit_hash=git_commit_hash,
            parent_baseline_experiment_id=baseline_id,
            panels=[
                _panel_record_for_spec(
                    spec=spec,
                    expected_trial_count=0,
                    lifecycle="pending",
                )
                for spec in panel_specs
            ],
            focus_name=harness_config.focus_name,
            started_at=baseline_at,
        )
        experiment_dir = experiments_root / resolved_experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=False)
        record.write(root=experiments_root)

        try:
            from src.adapters.env import Harbor

            trial_harbor_config = harbor_config.model_copy(
                update={"experiments_dir": experiment_dir / "tasks"}
            )
            task_dirs = asyncio.run(
                _prepare_task_dirs(
                    trial_harbor_config=trial_harbor_config,
                    task_names=[
                        task_name
                        for spec in panel_specs
                        for task_name in spec.task_names
                    ],
                )
            )
            for spec in panel_specs:
                panel = record.panels[spec.panel_id]
                panel.lifecycle = "active"
                panel.started_at = datetime.now(timezone.utc).isoformat()
                for trials in panel.task_results.values():
                    trials.expected_trial_count = harness_config.task_trials
                record.write(root=experiments_root)
                asyncio.run(
                    _run_panel(
                        record=record,
                        experiments_root=experiments_root,
                        task_names=_schedule_order(spec.task_names),
                        task_results=record.panels[spec.panel_id].task_results,
                        task_timeout_sec=spec.task_timeout_sec,
                        task_dirs=task_dirs,
                        harness_config=harness_config,
                        make_llm=lambda: _make_llm_for_config(
                            config=harness_config.llm_provider_config,
                            api_key=api_key,
                        ),
                        make_env=lambda task_name,
                        *,
                        task_dir,
                        exec_semaphore=None: Harbor(
                            trial_harbor_config,
                            task_name=task_name,
                            task_dir=task_dir,
                            exec_semaphore=exec_semaphore,
                        ),
                    )
                )
                panel.lifecycle = "finished"
                panel.finished_at = datetime.now(timezone.utc).isoformat()
                record.write(root=experiments_root)
                raise_if_no_valid_evidence(record)
        except BaseException as exc:
            record.finalize_crash(exc=exc, baseline=baseline, root=experiments_root)
            if isinstance(exc, Exception):
                return record
            raise

        record.finalize(status="keep", decision_reason=decision_reason)
        record.refresh_evidence(baseline=baseline)
        record.write(root=experiments_root)
        state.active_baseline_experiment_id = record.experiment_id
        state.current_experiment_id = record.experiment_id
        state.updated_at = record.finished_at
        state.save(root=experiments_root)
        return record

    def __init__(
        self,
        *,
        harness_config: HarnessConfig,
        harbor_config: HarborConfig,
        api_key: str | None,
        require_clean_worktree: bool = True,
    ) -> None:
        self.harness_config = harness_config
        self.panel_specs = _compile_panel_specs(harness_config)
        self.harbor_config = harbor_config
        self.api_key = api_key
        if require_clean_worktree:
            control_repo.require_clean_worktree()
        self.experiments_root = self.harbor_config.experiments_dir
        self.experiment_dir = self.experiments_root / self.harness_config.experiment_id
        self.state = ExperimentState.load(root=self.experiments_root)
        self.frozen_baseline_experiment_id = self.state.active_baseline_experiment_id
        self.record = ExperimentRecord.initialize(
            experiment_id=self.harness_config.experiment_id,
            git_commit_hash=control_repo.get_head_commit(),
            parent_baseline_experiment_id=self.frozen_baseline_experiment_id,
            panels=[
                _panel_record_for_spec(
                    spec=spec,
                    expected_trial_count=self.harness_config.task_trials
                    if spec.initial_lifecycle == "active"
                    else 0,
                )
                for spec in self.panel_specs
            ],
            focus_name=self.harness_config.focus_name,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._task_dirs: dict[str, Path] = {}
        # Held for tests and terminal paths that need the parent baseline.
        self.baseline: ExperimentRecord | None = None

    def _trial_harbor_config(self):
        return self.harbor_config.model_copy(
            update={"experiments_dir": self.experiment_dir / "tasks"}
        )

    def _make_env(
        self,
        task_name: str,
        task_dir: Path,
        exec_semaphore: asyncio.Semaphore | None = None,
    ):
        from src.adapters.env import Harbor

        return Harbor(
            self._trial_harbor_config(),
            task_name=task_name,
            task_dir=task_dir,
            exec_semaphore=exec_semaphore,
        )

    def _make_llm(self):
        return _make_llm_for_config(
            config=self.harness_config.llm_provider_config,
            api_key=self.api_key,
        )

    def _set_panel_trial_budget(
        self,
        *,
        panel: PanelRecord,
        baseline_panel: PanelRecord | None,
    ) -> bool:
        changed = False
        for task_id, trials in panel.task_results.items():
            expected_trial_count = self.harness_config.task_trials
            baseline_trials = (
                None
                if baseline_panel is None
                else baseline_panel.task_results.get(task_id)
            )
            if (
                self.harness_config.task_trials > 1
                and baseline_trials is not None
                and baseline_trials.is_deterministic_solved
            ):
                expected_trial_count = 1
            if trials.expected_trial_count != expected_trial_count:
                trials.expected_trial_count = expected_trial_count
                changed = True
        return changed

    def run(self) -> ExperimentRecord:
        self.experiment_dir.mkdir(parents=True, exist_ok=False)
        baseline_id = self.frozen_baseline_experiment_id
        baseline = (
            None
            if baseline_id is None
            else ExperimentRecord.load(baseline_id, root=self.experiments_root)
        )
        self.baseline = baseline

        try:
            self._validate_setup_contract(
                state=self.state,
                candidate=self.record,
                baseline=baseline,
            )
            self._task_dirs = asyncio.run(
                _prepare_task_dirs(
                    trial_harbor_config=self._trial_harbor_config(),
                    task_names=[
                        task_name
                        for spec in self.panel_specs
                        if spec.initial_lifecycle == "active"
                        for task_name in spec.task_names
                    ],
                )
            )
            self.record.started_at = datetime.now(timezone.utc).isoformat()
            self.record.write(root=self.experiments_root)
            self.state.current_experiment_id = self.record.experiment_id
            self.state.updated_at = self.record.started_at
            self.state.save(root=self.experiments_root)
            return asyncio.run(self._run_experiment(baseline=baseline))
        except BaseException as exc:
            self.record.finalize_crash(
                exc=exc, baseline=baseline, root=self.experiments_root
            )
            self._conclude_experiment()
            if isinstance(exc, Exception):
                return self.record
            raise

    async def _run_experiment(
        self,
        *,
        baseline: ExperimentRecord | None,
    ) -> ExperimentRecord:
        try:
            status: ExperimentStatus | None = None
            decision_reason = ""
            verdicts: dict[str, BaselineComparison] = {}
            for spec in self.panel_specs:
                if spec.after_panel is None:
                    should_run = True
                else:
                    upstream_evaluation = self.record.panels[
                        spec.after_panel
                    ].evaluation
                    should_run = (
                        upstream_evaluation is not None
                        and upstream_evaluation.status == spec.when_status
                    )
                if not should_run:
                    panel = self.record.panels[spec.panel_id]
                    panel.lifecycle = "skipped"
                    panel.skip_reason = (
                        f"{spec.after_panel} panel did not {spec.when_status}"
                    )
                    self.record.write(root=self.experiments_root)
                    continue
                if spec.requires_baseline and baseline is None:
                    raise RuntimeError(
                        f"{spec.panel_id} panel requires an active baseline"
                    )
                panel = self.record.panels[spec.panel_id]
                baseline_panel = (
                    None
                    if baseline is None or spec.panel_id not in baseline.panels
                    else baseline.panels[spec.panel_id]
                )
                panel.lifecycle = "active"
                panel.skip_reason = ""
                if panel.started_at is None:
                    panel.started_at = datetime.now(timezone.utc).isoformat()
                self._set_panel_trial_budget(
                    panel=panel,
                    baseline_panel=baseline_panel,
                )
                self.record.write(root=self.experiments_root)
                missing_task_dirs = [
                    task_name
                    for task_name in spec.task_names
                    if task_name not in self._task_dirs
                ]
                if missing_task_dirs:
                    self._task_dirs.update(
                        await _prepare_task_dirs(
                            trial_harbor_config=self._trial_harbor_config(),
                            task_names=missing_task_dirs,
                        )
                    )
                await _run_panel(
                    record=self.record,
                    experiments_root=self.experiments_root,
                    task_names=_schedule_order(spec.task_names),
                    task_results=panel.task_results,
                    task_timeout_sec=spec.task_timeout_sec,
                    task_dirs=self._task_dirs,
                    harness_config=self.harness_config,
                    make_llm=self._make_llm,
                    make_env=self._make_env,
                )
                panel.lifecycle = "finished"
                panel.finished_at = datetime.now(timezone.utc).isoformat()
                self.record.write(root=self.experiments_root)
                self._validate_record_ready_for_evaluation(self.record)
                panel_pool = self._build_gate_pool(
                    baseline=baseline,
                    panel=spec.panel_id,
                )
                panel_verdicts = build_gate_verdicts(
                    candidate=self.record,
                    pool=panel_pool,
                    panel=spec.panel_id,
                )
                panel_status, panel_reason = decide_panel_from_verdicts(
                    candidate=self.record,
                    verdicts=panel_verdicts,
                    panel=spec.panel_id,
                    purpose=spec.purpose,
                )
                panel.evaluation = PanelEvaluation(
                    status=panel_status,
                    decision_reason=panel_reason,
                    verdicts=panel_verdicts,
                )
                verdicts = {**verdicts, **panel_verdicts}
                if spec.purpose == "promotion" or panel_status == "discard":
                    status = panel_status
                    decision_reason = panel_reason
                self.record.write(root=self.experiments_root)
            if status is None:
                raise RuntimeError("candidate produced no panel decision")
            self.record.finalize(status=status, decision_reason=decision_reason)
            # Thread the gate's verdict dict into the persisted evidence so
            # downstream readers, including supervisor artifact selection,
            # read the same per-task verdict the promotion decision used.
            self.record.refresh_evidence(baseline=baseline, verdicts=verdicts)
            self.record.write(root=self.experiments_root)
            self._conclude_experiment()
        except Exception as exc:
            self.record.finalize_crash(
                exc=exc, baseline=baseline, root=self.experiments_root
            )
            self._conclude_experiment()

        return self.record

    def _build_gate_pool(
        self,
        *,
        baseline: ExperimentRecord | None,
        panel: str,
    ) -> dict[str, tuple[int, int]]:
        # No baseline yet: no-baseline frontier path for every task.
        # compare_candidate_against_baseline handles (0,0) entries by
        # requiring a candidate majority-solve for improvement.
        if baseline is None:
            return {}
        return build_baseline_pool(
            active_baseline=baseline,
            task_ids=tuple(self.record.panels[panel].task_ids),
            panel=panel,
        )

    def _conclude_experiment(self) -> None:
        status = self.record.status
        if status is None:
            raise RuntimeError(
                "terminal outcome requires a finalized experiment status"
            )
        if self.record.finished_at is None:
            raise RuntimeError("terminal outcome requires a finished timestamp")
        self.state.current_experiment_id = self.record.experiment_id
        self.state.updated_at = self.record.finished_at
        match status:
            case "keep":
                self.state.active_baseline_experiment_id = self.record.experiment_id
                self.state.save(root=self.experiments_root)
                return
            case _:
                preserved_git_ref = failed_experiment_git_ref(self.record.experiment_id)
                control_repo.update_ref(preserved_git_ref, self.record.git_commit_hash)
                self.state.save(root=self.experiments_root)
                return

    def _validate_setup_contract(
        self,
        *,
        state: ExperimentState,
        candidate: ExperimentRecord,
        baseline: ExperimentRecord | None,
    ) -> None:
        if (
            candidate.parent_baseline_experiment_id
            != state.active_baseline_experiment_id
        ):
            raise ValueError(
                "candidate parent baseline must match the frozen active baseline"
            )
        for panel_id in candidate.panel_order:
            panel = candidate.panels[panel_id]
            if set(panel.task_ids) != set(panel.task_results):
                raise ValueError(
                    f"candidate {panel_id} panel task results must cover exactly the configured task ids"
                )
        if baseline is None:
            for spec in self.panel_specs:
                if spec.requires_baseline:
                    raise ValueError(
                        f"{spec.panel_id} panel requires an active baseline"
                    )
            return
        if candidate.panel_order != baseline.panel_order:
            raise ValueError(
                "candidate panels must match the frozen baseline panel order"
            )
        for panel_id in candidate.panel_order:
            if (
                candidate.panels[panel_id].task_ids
                != baseline.panels[panel_id].task_ids
            ):
                raise ValueError(
                    f"candidate {panel_id} panel must match the frozen baseline {panel_id} panel"
                )

    def _validate_record_ready_for_evaluation(self, record: ExperimentRecord) -> None:
        if any(not trials.is_finished for trials in record._all_task_results()):
            raise RuntimeError("all task results must be finished before evaluation")
        if record.error:
            raise RuntimeError(
                "experiment record must not carry a top-level error before evaluation"
            )
        raise_if_no_valid_evidence(record)
