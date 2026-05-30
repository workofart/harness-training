"""Experiment run orchestration.

Drives a candidate or baseline experiment: prepares task dirs, runs the trial
panel concurrently, applies the promotion gate, and concludes by updating run
state / git refs. The persisted record types live in ``record.py`` and the gate
statistics in ``gate.py``.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

from src.control import repo as control_repo
from src.experiment.trial import run_task
from src.harness.contracts import TaskResult
from src.metrics import is_majority_decided

from src.experiment.gate import (
    build_gate_pool,
    build_gate_verdicts,
    decide_from_verdicts,
)
from src.experiment.record import (
    ExperimentRecord,
    ExperimentState,
    failed_experiment_git_ref,
    raise_if_no_valid_evidence,
    terminal_task_result,
)

if TYPE_CHECKING:
    from src.adapters.env import HarborConfig
    from src.harness.config import HarnessConfig, LlmProviderConfig


def _prepare_task_dirs(
    *,
    trial_harbor_config: HarborConfig,
    task_names: Sequence[str],
) -> dict[str, Path]:
    from src.adapters.env import TaskDirectoryResolver

    return dict(TaskDirectoryResolver(trial_harbor_config).resolve(list(task_names)))


PROGRESS_BAR_WIDTH = 24


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
        max_concurrency: int,
        stream: Any | None = None,
    ) -> None:
        self._stream = sys.stderr if stream is None else stream
        self._total_tasks = total_tasks
        self._max_concurrency = max_concurrency
        self._enabled = bool(getattr(self._stream, "isatty", lambda: False)())
        self._start = time.monotonic()
        self._drawn = False
        self._finalized = False

    def render(self, record: ExperimentRecord) -> None:
        if not self._enabled or self._finalized:
            return
        trials = record.train_task_results.values()
        tasks_done = sum(1 for t in trials if t.is_finished)
        trials_done = sum(len(t.finished_trials) for t in trials)
        valid_done = sum(len(t.valid_trials) for t in trials)
        trials_planned = sum(t.expected_trial_count for t in trials)
        decided = sum(1 for t in trials if t.majority_solved is not None)
        in_flight = max(0, min(self._max_concurrency, trials_planned - trials_done))
        line = format_panel_progress(
            tasks_done=tasks_done,
            total_tasks=self._total_tasks,
            trials_done=trials_done,
            trials_planned=trials_planned,
            solved=record.train_solved_count or 0,
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
    task_dirs: Mapping[str, Path],
    harness_config: HarnessConfig,
    make_llm: Callable[[], Any],
    make_env: Callable[..., Any],
) -> None:
    semaphore = asyncio.Semaphore(harness_config.max_concurrency)
    reporter = PanelProgressReporter(
        total_tasks=len(task_names),
        max_concurrency=harness_config.max_concurrency,
    )

    def persist_task_result(task_result: TaskResult) -> None:
        record.record_task_result(task_result)
        record.write(root=experiments_root)
        reporter.render(record)

    async def run_one_trial(task_id: str) -> None:
        async with semaphore:
            task_dir = task_dirs[task_id]
            try:
                trial_result = await run_task(
                    task_name=task_id,
                    llm=make_llm(),
                    env=make_env(task_id, task_dir=task_dir),
                    max_steps=harness_config.max_steps,
                    max_output_retries=harness_config.max_output_retries,
                    task_timeout_sec=harness_config.task_timeout_sec,
                )
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                persist_task_result(terminal_task_result(task_id=task_id, exc=exc))
                if isinstance(exc, Exception):
                    return
                raise
            persist_task_result(trial_result)

    async def run_task_trials(task_id: str) -> None:
        full_trial_count = harness_config.task_trials
        while not record._task_trials(task_id).is_finished:
            trials = record._task_trials(task_id)
            remaining_budget = trials.expected_trial_count - len(trials.finished_trials)
            if remaining_budget <= 0:
                break
            in_flight: set[asyncio.Task[None]] = {
                asyncio.create_task(run_one_trial(task_id))
                for _ in range(remaining_budget)
            }
            try:
                while in_flight:
                    done, in_flight = await asyncio.wait(
                        in_flight, return_when=asyncio.FIRST_COMPLETED
                    )
                    for finished in done:
                        finished.result()
                    trials = record._task_trials(task_id)
                    if (
                        trials.expected_trial_count == 1
                        and full_trial_count > 1
                        and trials.solved_count == 0
                        and len(trials.valid_trials) == 1
                    ):
                        trials.expected_trial_count = full_trial_count
                        record.write(root=experiments_root)
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
                        record.write(root=experiments_root)
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
                for task_trials in current_record.train_task_results.values()
            ):
                raise RuntimeError(
                    "cannot run baseline while the current experiment has recorded task activity"
                )
        control_repo.require_clean_worktree(cwd=repo_root)
        git_commit_hash = control_repo.get_head_commit(cwd=repo_root)
        if baseline is not None:
            if baseline.git_commit_hash == git_commit_hash and set(
                harness_config.train_task_names
            ) == set(baseline.train_task_ids):
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
            train_task_ids=list(harness_config.train_task_names),
            focus_name=harness_config.focus_name,
            started_at=baseline_at,
            expected_trial_count=harness_config.task_trials,
        )
        experiment_dir = experiments_root / resolved_experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=False)
        record.write(root=experiments_root)

        try:
            from src.adapters.env import Harbor

            trial_harbor_config = harbor_config.model_copy(
                update={"experiments_dir": experiment_dir / "tasks"}
            )
            task_dirs = _prepare_task_dirs(
                trial_harbor_config=trial_harbor_config,
                task_names=harness_config.train_task_names,
            )
            asyncio.run(
                _run_panel(
                    record=record,
                    experiments_root=experiments_root,
                    task_names=harness_config.train_task_names,
                    task_dirs=task_dirs,
                    harness_config=harness_config,
                    make_llm=lambda: _make_llm_for_config(
                        config=harness_config.llm_provider_config,
                        api_key=api_key,
                    ),
                    make_env=lambda task_name, *, task_dir: Harbor(
                        trial_harbor_config,
                        task_name=task_name,
                        task_dir=task_dir,
                    ),
                )
            )
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
            train_task_ids=list(self.harness_config.train_task_names),
            focus_name=self.harness_config.focus_name,
            started_at=datetime.now(timezone.utc).isoformat(),
            expected_trial_count=self.harness_config.task_trials,
        )
        self._task_dirs: dict[str, Path] = {}
        # Held for tests and terminal paths that need the parent baseline.
        self.baseline: ExperimentRecord | None = None

    def _trial_harbor_config(self):
        return self.harbor_config.model_copy(
            update={"experiments_dir": self.experiment_dir / "tasks"}
        )

    def _make_env(self, task_name: str, task_dir: Path):
        from src.adapters.env import Harbor

        return Harbor(
            self._trial_harbor_config(),
            task_name=task_name,
            task_dir=task_dir,
        )

    def _make_llm(self):
        return _make_llm_for_config(
            config=self.harness_config.llm_provider_config,
            api_key=self.api_key,
        )

    def _apply_baseline_derived_trial_counts(self, baseline: ExperimentRecord) -> None:
        # Where the baseline shows a task as deterministic-solved (every
        # observed trial passed), budget the candidate for a single trial.
        # If that trial fails, run_task_trials' confirm-on-fail expands the
        # budget back to task_trials to verify the suspected regression.
        if self.harness_config.task_trials <= 1:
            return
        changed = False
        for task_id, baseline_trials in baseline.train_task_results.items():
            candidate_trials = self.record.train_task_results.get(task_id)
            if candidate_trials is None:
                continue
            if baseline_trials.is_deterministic_solved:
                candidate_trials.expected_trial_count = 1
                changed = True
        if changed:
            self.record.write(root=self.experiments_root)

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
            self._task_dirs = _prepare_task_dirs(
                trial_harbor_config=self._trial_harbor_config(),
                task_names=self.harness_config.train_task_names,
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
            if baseline is not None:
                self._apply_baseline_derived_trial_counts(baseline)
            await _run_panel(
                record=self.record,
                experiments_root=self.experiments_root,
                task_names=list(self.harness_config.train_task_names),
                task_dirs=self._task_dirs,
                harness_config=self.harness_config,
                make_llm=self._make_llm,
                make_env=self._make_env,
            )
            self._validate_record_ready_for_evaluation(self.record)
            pool = self._build_gate_pool(baseline=baseline)
            verdicts = build_gate_verdicts(candidate=self.record, pool=pool)
            status, decision_reason = decide_from_verdicts(
                candidate=self.record, verdicts=verdicts
            )
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
    ) -> dict[str, tuple[int, int]]:
        # No baseline yet: no-baseline frontier path for every task.
        # compare_candidate_against_baseline handles (0,0) entries by
        # requiring a candidate majority-solve for improvement.
        if baseline is None:
            return {}
        return build_gate_pool(
            experiments_root=self.experiments_root,
            workspace_root=Path.cwd(),
            active_baseline=baseline,
            candidate_experiment_id=self.record.experiment_id,
            task_ids=tuple(self.harness_config.train_task_names),
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
        if set(candidate.train_task_ids) != set(candidate.train_task_results):
            raise ValueError(
                "candidate task results must cover exactly the configured task ids"
            )
        if baseline is None:
            return
        if candidate.train_task_ids != baseline.train_task_ids:
            raise ValueError(
                "candidate train panel must match the frozen baseline train panel"
            )

    def _validate_record_ready_for_evaluation(self, record: ExperimentRecord) -> None:
        if any(not trials.is_finished for trials in record.train_task_results.values()):
            raise RuntimeError("all task results must be finished before evaluation")
        if record.error:
            raise RuntimeError(
                "experiment record must not carry a top-level error before evaluation"
            )
        raise_if_no_valid_evidence(record)
