from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import DEFAULT_HARNESS_CONFIG_PATH

if TYPE_CHECKING:
    from src.env.harbor import HarborConfig
    from src.config import HarnessConfig
    from src.experiment.record import TaskResult


def _quiet_http_request_logs() -> None:
    """Pin the HTTP stack's loggers to WARNING. httpx logs one INFO line per LLM
    call (a run makes thousands) and httpcore/huggingface_hub are similarly noisy
    at dataset load -- together they bury the progress bar. A dependency sets the
    root logger to INFO at import; a named logger's own level takes precedence, so
    this silences the per-request chatter without touching our logs. Errors still
    surface at WARNING+."""
    import logging

    for name in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)


def load_strict_runtime_config(
    *,
    harbor_config_path: Path,
    harness_config_path: Path,
) -> tuple[HarborConfig, HarnessConfig]:
    from src.env.harbor import HarborConfig
    from src.config import HarnessConfig

    harbor_config = HarborConfig.from_toml(harbor_config_path)
    harness_config = HarnessConfig.model_validate_json(harness_config_path.read_text())
    return harbor_config, harness_config


def _load_llm_provider_secret(
    *,
    harness_config: HarnessConfig,
    dotenv_path: str | Path = ".env",
) -> str | None:
    if harness_config.llm_provider_config.provider == "openrouter":
        from src.llm.openrouter import load_openrouter_api_key

        return load_openrouter_api_key(dotenv_path=dotenv_path)
    return None


def load_runtime_config(
    *, experiments_dir: str | None = None
) -> tuple[HarborConfig, HarnessConfig, str | None]:
    from src.env.harbor import DEFAULT_HARBOR_CONFIG_PATH

    harbor_config, harness_config = load_strict_runtime_config(
        harbor_config_path=DEFAULT_HARBOR_CONFIG_PATH,
        harness_config_path=DEFAULT_HARNESS_CONFIG_PATH,
    )
    harbor_config = _apply_experiments_dir_override(harbor_config, experiments_dir)
    api_key = _load_llm_provider_secret(harness_config=harness_config)
    return harbor_config, harness_config, api_key


def _apply_experiments_dir_override(
    harbor_config: HarborConfig, override: str | None
) -> HarborConfig:
    """Honor `--experiments-dir` (plan.md §12 path anchoring): when `auto` runs
    `exp` inside a throwaway candidate worktree it passes the absolute
    `<main_repo>/experiments` so trial artifacts, `experiment.json`, and the shared
    verifier-context cache land in the one canonical dir -- never relative to the
    worktree cwd. Re-validates so the cache dir re-derives under the new root.
    A no-op (returns the same config) when not given, e.g. a standalone
    `uv run exp` in the primary repo."""
    if not override:
        return harbor_config
    experiments_dir = Path(override)
    if not experiments_dir.is_absolute():  # §12: never relative to cwd
        raise ValueError(
            f"--experiments-dir must be an absolute path, got: {override!r}"
        )
    payload = harbor_config.model_dump()
    payload["experiments_dir"] = experiments_dir
    payload["verifier_contexts_dir"] = None  # re-derive under the new experiments dir
    return type(harbor_config).model_validate(payload)


def _require_clean_worktree_for_exp() -> bool:
    return os.getenv("EXP_ALLOW_DIRTY_WORKTREE", "").lower() not in {
        "1",
        "true",
        "yes",
    }


def _configured_panels(harness_config) -> list:
    panels = [harness_config.train]
    if harness_config.test is not None:
        panels.append(harness_config.test)
    return panels


def _selected_task_ids(harness_config, tasks: str | None) -> list[str]:
    # exp runs a task SET (plan.md §2): `--tasks` (comma-separated) selects a
    # subset -- how `auto` drives train, then test, as separate calls -- and the
    # default is every configured task. exp itself stays decision-free; the §12
    # asserts that `auto` only ever passes sanctioned shapes live in the loop.
    if tasks is not None and tasks.strip():
        selected = [task_id.strip() for task_id in tasks.split(",") if task_id.strip()]
    else:
        selected = [
            t for panel in _configured_panels(harness_config) for t in panel.task_names
        ]
    return list(dict.fromkeys(selected))


def _selected_experiment_id(experiment_id: str | None) -> str:
    # `--experiment-id` names an existing dir to append into (auto's shared id
    # across its train and test calls); a standalone run gets a fresh id (§2).
    if experiment_id is not None and experiment_id.strip():
        return experiment_id.strip()
    return f"exp-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def _selected_trial_budget(
    *, task_ids, harness_config, trial_budget: str | None
) -> dict[str, int]:
    # `auto` passes `--trial-budget` (a JSON task->count map from
    # `budget_from_baseline`, §9) so a candidate run gets the deterministic-baseline
    # single-trial shortcut across the subprocess seam; a standalone `uv run exp`
    # omits it -> uniform-full. The loop is the only writer, so a budget that
    # does not cover the selected task set exactly is a bug -> fail fast (§12 strict
    # interfaces) rather than silently mis-budget.
    if trial_budget is None or not trial_budget.strip():
        return {task_id: harness_config.task_trials for task_id in task_ids}
    budget = json.loads(trial_budget)
    if set(budget) != set(task_ids):
        raise ValueError(
            f"--trial-budget keys {sorted(budget)} != selected task ids "
            f"{sorted(task_ids)}"
        )
    return {task_id: int(budget[task_id]) for task_id in task_ids}


def _parse_exp_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="exp",
        description=(
            "Run the configured task set once and write a raw experiment record. "
            "All flags are optional; `auto` passes them to drive train/test runs."
        ),
    )
    parser.add_argument(
        "--tasks",
        help="comma-separated task subset (default: every configured task)",
    )
    parser.add_argument(
        "--experiment-id",
        help="existing experiment dir to append into (default: fresh exp-<timestamp>)",
    )
    parser.add_argument(
        "--experiments-dir",
        help="absolute experiments dir to anchor artifacts to (plan.md §12)",
    )
    parser.add_argument(
        "--trial-budget",
        help="per-task trial-count JSON map, e.g. '{\"task-id\": 1}'",
    )
    return parser.parse_args(argv)


def _panel_seconds_by_task(harness_config, field_name: str) -> dict[str, float]:
    # Each task carries its owning panel's wall budgets (task_timeout_sec /
    # verify_timeout_sec); tasks are disjoint across panels (config §12), so
    # this is unambiguous for any selected subset.
    return {
        task_id: getattr(panel, field_name)
        for panel in _configured_panels(harness_config)
        for task_id in panel.task_names
    }


def _make_llm_for_config(*, config, api_key: str | None):
    match config.provider:
        case "openrouter":
            if api_key is None:
                raise ValueError("OPENROUTER_API_KEY is not set")
            from src.llm.openrouter import OpenRouter

            return OpenRouter(config=config, api_key=api_key)
        case "chatgpt_codex":
            from src.llm.codex import ChatGptCodex

            return ChatGptCodex(config=config)


async def _run_trial_with_env(
    *,
    env,
    task_id,
    run_id,
    harness_config,
    api_key,
    task_timeout_sec,
    verify_timeout_sec,
    slot_release,
):
    """Drive one prepared env through `executor.run_trial` with this task's panel
    wall budget -- the call shared by every backend's trial_runner."""
    from src.experiment.executor import run_trial

    return await run_trial(
        task_id=task_id,
        run_id=run_id,
        llm=_make_llm_for_config(
            config=harness_config.llm_provider_config, api_key=api_key
        ),
        env=env,
        max_steps=harness_config.max_steps,
        max_output_retries=harness_config.max_output_retries,
        task_timeout_sec=task_timeout_sec,
        env_setup_timeout_sec=harness_config.env_setup_timeout_sec,
        verify_timeout_sec=verify_timeout_sec,
        slot_release=slot_release,
    )


def _make_trial_runner(*, harness_config, api_key, build_env):
    # The trial_runner closure shared by every backend: look up each task's panel
    # wall budgets once, then drive `build_env(task_id, run_id, heavy_semaphore)`
    # through `_run_trial_with_env`. Each builder supplies only its env factory.
    task_timeout_sec = _panel_seconds_by_task(harness_config, "task_timeout_sec")
    verify_timeout_sec = _panel_seconds_by_task(harness_config, "verify_timeout_sec")

    async def trial_runner(task_id, run_id, heavy_action_semaphore, slot_release):
        return await _run_trial_with_env(
            env=build_env(task_id, run_id, heavy_action_semaphore),
            task_id=task_id,
            run_id=run_id,
            harness_config=harness_config,
            api_key=api_key,
            task_timeout_sec=task_timeout_sec[task_id],
            verify_timeout_sec=verify_timeout_sec[task_id],
            slot_release=slot_release,
        )

    return trial_runner


async def _build_harbor_trial_runner(
    *, harness_config, harbor_config, api_key: str | None, task_ids, experiment_id
):
    """Terminal Bench backend: resolve each selected task's directory once, then
    build a per-trial `Harbor` handed the run-scoped heavy gate."""
    from src.env.harbor import Harbor, TaskDirectoryResolver

    trial_harbor_config = harbor_config.model_copy(
        update={
            "experiments_dir": harbor_config.experiments_dir / experiment_id / "tasks"
        }
    )
    task_dirs = dict(
        await TaskDirectoryResolver(trial_harbor_config).resolve(list(task_ids))
    )
    return _make_trial_runner(
        harness_config=harness_config,
        api_key=api_key,
        build_env=lambda task_id, _run_id, sem: Harbor(
            trial_harbor_config,
            task_name=task_id,
            task_dir=task_dirs[task_id],
            exec_semaphore=sem,
        ),
    )


async def _build_swe_trial_runner(
    *, harness_config, harbor_config, api_key: str | None, task_ids, experiment_id
):
    """SWE-bench-Verified backend: resolve each instance id to a dataset row once,
    then build a per-trial `SweEnv` (offline container, handed the run-scoped
    heavy gate). Artifacts mirror Harbor's tasks/<task_id>/<run_id> layout."""
    from src.env.swe import SweEnv, load_rows

    rows = load_rows(list(task_ids))
    tasks_root = harbor_config.experiments_dir / experiment_id / "tasks"
    return _make_trial_runner(
        harness_config=harness_config,
        api_key=api_key,
        build_env=lambda task_id, run_id, sem: SweEnv(
            row=rows[task_id],
            artifacts_dir=str(tasks_root / task_id / run_id),
            heavy_semaphore=sem,
        ),
    )


async def _build_trial_runner(
    *, harness_config, harbor_config, api_key: str | None, task_ids, experiment_id
):
    """Return the `run_tasks` trial_runner for the configured env backend -- the
    seam the orchestrator schedules. Each builder resolves its task source once
    and returns a `(task_id, run_id, heavy_action_semaphore, slot_release)`
    closure that builds the per-trial env + llm and calls `executor.run_trial`."""
    builder = (
        _build_swe_trial_runner
        if harness_config.env_backend == "swe"
        else _build_harbor_trial_runner
    )
    return await builder(
        harness_config=harness_config,
        harbor_config=harbor_config,
        api_key=api_key,
        task_ids=task_ids,
        experiment_id=experiment_id,
    )


# --- uv run exp live progress bar -------------------------------------------

_PROGRESS_BAR_WIDTH = 24


def _format_hms(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_exp_progress(
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
    """Render one `uv run exp` progress line.

    Anchored on task completion (`tasks_done/total_tasks`): that denominator is
    fixed so the bar never moves backward, whereas `trials_planned` shifts as the
    majority decides early or a candidate confirms-on-fail, so trials are detail
    text. `solved/decided` are task counts off the live record (`solved` = tasks
    the majority verdict has solved, `decided` = tasks with at least one valid
    trial). ETA divides remaining tasks by the observed completion rate.
    """
    frac = tasks_done / total_tasks if total_tasks else 0.0
    filled = int(frac * _PROGRESS_BAR_WIDTH)
    bar = "#" * filled + "-" * (_PROGRESS_BAR_WIDTH - filled)
    if tasks_done > 0 and elapsed_sec > 0:
        remaining = _format_hms((total_tasks - tasks_done) / (tasks_done / elapsed_sec))
        eta = f"~{remaining} left"
    else:
        eta = "~-- left"
    return (
        f"[{bar}] {tasks_done}/{total_tasks} tasks ({frac * 100:.0f}%) | "
        f"trials {trials_done}/{trials_planned} | "
        f"solved {solved}/{decided} | errors {error_trials} | active {in_flight} | "
        f"{_format_hms(elapsed_sec)} elapsed, {eta}"
    )


class _ExpProgressBar:
    """Live single-line `uv run exp` progress bar on stderr.

    Active only when stderr is a TTY -- a direct interactive run, or an `auto`
    cycle (the loop runs `uv run exp` under a PTY, so the bar streams through
    live). Event-driven: `render` fires from `run_tasks`'s persist hook, once per
    task state change, and counts only this run's `task_ids` (an `auto` veto run
    appends to a record that already holds the train tasks). `close` caps the
    dangling line with a newline so later stdout never appends to it.
    """

    def __init__(
        self,
        *,
        task_ids: Iterable[str],
        max_trial_concurrency: int,
        stream=None,
    ) -> None:
        self._stream = sys.stderr if stream is None else stream
        self._task_ids = list(task_ids)
        self._max_trial_concurrency = max_trial_concurrency
        self._enabled = bool(getattr(self._stream, "isatty", lambda: False)())
        self._start = time.monotonic()
        self._drawn = False

    def render(self, task_results: Mapping[str, "TaskResult"]) -> None:
        if not self._enabled:
            return
        scoped = [task_results[t] for t in self._task_ids if t in task_results]
        trials_done = sum(len(t.trials) for t in scoped)
        valid_done = sum(len(t.valid_trials) for t in scoped)
        trials_planned = sum(t.expected_trial_count for t in scoped)
        line = format_exp_progress(
            tasks_done=sum(1 for t in scoped if t.is_finished),
            total_tasks=len(self._task_ids),
            trials_done=trials_done,
            trials_planned=trials_planned,
            solved=sum(1 for t in scoped if t.majority_solved is True),
            decided=sum(1 for t in scoped if t.majority_solved is not None),
            error_trials=trials_done - valid_done,
            in_flight=max(
                0, min(self._max_trial_concurrency, trials_planned - trials_done)
            ),
            elapsed_sec=time.monotonic() - self._start,
        )
        self._stream.write("\r\033[K" + line)
        self._stream.flush()
        self._drawn = True

    def close(self) -> None:
        if self._enabled and self._drawn:
            self._stream.write("\n")
            self._stream.flush()
            self._drawn = False


async def run_experiment(
    *,
    harness_config,
    harbor_config,
    git_commit_hash,
    task_ids,
    experiment_id,
    trial_runner,
    budget=None,
    on_progress: Callable[[Mapping[str, "TaskResult"]], None] | None = None,
):
    """Run the selected `task_ids` into `experiment_id` -> a raw `ExperimentResult`
    (plan.md §2). `budget` is the per-task trial count (the gate's
    `budget_from_baseline` for an `auto` candidate, via `--trial-budget`);
    `None` defaults to uniform-full (a standalone `exp`/baseline). One `run_tasks`
    call per `uv run exp` invocation; an existing `experiment_id` is appended to
    (auto's train-then-test across two calls). No baseline, gate, decision, or
    git -- the loop's."""
    from src.experiment.orchestrator import run_tasks

    if budget is None:
        budget = {task_id: harness_config.task_trials for task_id in task_ids}
    return await run_tasks(
        experiment_id=experiment_id,
        git_commit_hash=git_commit_hash,
        task_ids=task_ids,
        budget=budget,
        full_trial_count=harness_config.task_trials,
        max_trial_concurrency=harness_config.max_trial_concurrency,
        max_heavy_action_concurrency=harness_config.max_heavy_action_concurrency,
        trial_runner=trial_runner,
        experiments_root=harbor_config.experiments_dir,
        on_progress=on_progress,
    )


def main_exp(argv: Sequence[str] | None = None) -> int:
    import asyncio

    from src.llm.codex import (
        CODEX_CREDENTIALS_EXPIRED_EXIT_CODE,
        ChatGptCodexCredentialsExpiredError,
    )
    from src.env.harbor import DEFAULT_HARBOR_CONFIG_PATH
    from src.repo import get_head_commit, require_clean_worktree

    _quiet_http_request_logs()
    args = _parse_exp_args(argv)
    harbor_config, harness_config, api_key = load_runtime_config(
        experiments_dir=args.experiments_dir
    )
    task_ids = _selected_task_ids(harness_config, args.tasks)
    experiment_id = _selected_experiment_id(args.experiment_id)
    budget = _selected_trial_budget(
        task_ids=task_ids,
        harness_config=harness_config,
        trial_budget=args.trial_budget,
    )
    print(f"experiment: {experiment_id}")
    print(f"tasks ({len(task_ids)}): {', '.join(task_ids)}")
    print(f"harbor config: {DEFAULT_HARBOR_CONFIG_PATH}")
    print(f"harness config: {DEFAULT_HARNESS_CONFIG_PATH}")
    if _require_clean_worktree_for_exp():
        require_clean_worktree()
    git_commit_hash = get_head_commit()
    bar = _ExpProgressBar(
        task_ids=task_ids,
        max_trial_concurrency=harness_config.max_trial_concurrency,
    )

    async def _go():
        trial_runner = await _build_trial_runner(
            harness_config=harness_config,
            harbor_config=harbor_config,
            api_key=api_key,
            task_ids=task_ids,
            experiment_id=experiment_id,
        )
        return await run_experiment(
            harness_config=harness_config,
            harbor_config=harbor_config,
            git_commit_hash=git_commit_hash,
            task_ids=task_ids,
            experiment_id=experiment_id,
            trial_runner=trial_runner,
            budget=budget,
            on_progress=bar.render,
        )

    try:
        result = asyncio.run(_go())
    except ChatGptCodexCredentialsExpiredError as exc:
        bar.close()
        print(str(exc), file=sys.stderr)
        return CODEX_CREDENTIALS_EXPIRED_EXIT_CODE
    bar.close()
    print(f"run status: {result.run_status}")
    print("run complete")
    return 0


def _install_runner_teardown_signal_handler() -> None:
    """On SIGINT (Ctrl-C) / SIGTERM, reap the in-flight ``uv run exp`` subtree
    before exiting. The active child is blocked deep inside ``run_streamed``'s
    select loop with no handle reachable from here, so we reap via the module-level
    registry instead. CRITICAL: a second Ctrl-C during cleanup would otherwise raise
    ``KeyboardInterrupt`` straight through and abort the reap, leaving ``exp``
    orphaned -- so we mask SIGINT (``SIG_IGN``) for the duration of cleanup."""
    import signal

    from src.supervisor.subproc import terminate_live_children

    def handler(signum, _frame):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        name = signal.Signals(signum).name
        print(
            f"\n{name} received -- shutting down runner, please wait, "
            "do NOT press Ctrl-C again.",
            file=sys.stderr,
            flush=True,
        )
        terminate_live_children()
        print("Runner shutdown complete.", file=sys.stderr, flush=True)
        raise SystemExit(130)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def main_auto() -> int:
    import contextlib

    from src.env.harbor import DEFAULT_HARBOR_CONFIG_PATH
    from src.llm.codex import (
        CODEX_CREDENTIALS_EXPIRED_EXIT_CODE,
        ChatGptCodexCredentialsExpiredError,
    )
    from src.supervisor.agent_backend import create_backend, supervisor_root_for_repo
    from src.supervisor.loop import LoopContext, run_auto

    _quiet_http_request_logs()
    agent_type = "codex"
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg in ("--agent", "-a") and i < len(sys.argv) - 1:
            agent_type = sys.argv[i + 1]
            break
        if arg.startswith("--agent="):
            agent_type = arg.split("=", 1)[1]
            break

    repo_root = Path(__file__).resolve().parents[1]
    # §12: experiments_dir must anchor to <main_repo>, but HarborConfig resolves a
    # relative experiments_dir against cwd -- so load with cwd at repo_root, making
    # `uv run auto` cwd-independent (an absolute config path is unaffected; cwd is
    # restored on exit, and the loop's I/O all uses explicit paths, never cwd).
    with contextlib.chdir(repo_root):
        harbor_config, harness_config = load_strict_runtime_config(
            harbor_config_path=DEFAULT_HARBOR_CONFIG_PATH,
            harness_config_path=DEFAULT_HARNESS_CONFIG_PATH,
        )
    ctx = LoopContext(
        repo_root=repo_root,
        # HarborConfig resolves experiments_dir to absolute (§12 path anchoring); the
        # loop hands it to every `uv run exp` via --experiments-dir. Throwaway
        # candidate worktrees live in a sibling dir so they never dirty the primary.
        experiments_dir=harbor_config.experiments_dir,
        worktree_root=supervisor_root_for_repo(repo_root) / "worktrees",
        config=harness_config,
        backend=create_backend(agent_type),
        program_md_path=repo_root / "program.md",
    )
    _install_runner_teardown_signal_handler()
    try:
        halt = run_auto(ctx)
    except ChatGptCodexCredentialsExpiredError as exc:
        print(f"\nSupervisor halted. {exc}", file=sys.stderr)
        return CODEX_CREDENTIALS_EXPIRED_EXIT_CODE
    except KeyboardInterrupt:
        return 130
    # run_auto only returns at a Halt (needs a human, §6); a LoopCorruption
    # propagates uncaught (an impossible control state -- not a normal transition).
    print(f"\nSupervisor halted: {halt.reason}", file=sys.stderr)
    return 0
