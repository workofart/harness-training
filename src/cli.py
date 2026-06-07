from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import DEFAULT_HARNESS_CONFIG_PATH

if TYPE_CHECKING:
    from src.env.harbor import HarborConfig
    from src.config import HarnessConfig


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


def load_runtime_config() -> tuple[HarborConfig, HarnessConfig, str | None]:
    from src.env.harbor import DEFAULT_HARBOR_CONFIG_PATH

    harbor_config, harness_config = load_strict_runtime_config(
        harbor_config_path=DEFAULT_HARBOR_CONFIG_PATH,
        harness_config_path=DEFAULT_HARNESS_CONFIG_PATH,
    )
    harbor_config = _apply_experiments_dir_override(harbor_config)
    api_key = _load_llm_provider_secret(harness_config=harness_config)
    return harbor_config, harness_config, api_key


def _apply_experiments_dir_override(harbor_config: HarborConfig) -> HarborConfig:
    """Honor `EXP_EXPERIMENTS_DIR` (plan.md §12 path anchoring): when `auto` runs
    `exp` inside a throwaway candidate worktree it passes the absolute
    `<main_repo>/experiments` so trial artifacts, `experiment.json`, and the shared
    verifier-context cache land in the one canonical dir -- never relative to the
    worktree cwd. Re-validates so the cache dir re-derives under the new root.
    A no-op (returns the same config) when the var is unset, e.g. a standalone
    `uv run exp` in the primary repo."""
    override = os.getenv("EXP_EXPERIMENTS_DIR")
    if not override:
        return harbor_config
    experiments_dir = Path(override)
    if not experiments_dir.is_absolute():  # §12: never relative to cwd
        raise ValueError(
            f"EXP_EXPERIMENTS_DIR must be an absolute path, got: {override!r}"
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


def _selected_task_ids(harness_config) -> list[str]:
    # exp runs a task SET (plan.md §2): `EXP_TASK_IDS` (comma-separated) selects a
    # subset -- how `auto` drives train, then test, as separate calls -- and the
    # default is every configured task. exp itself stays decision-free; the §12
    # asserts that `auto` only ever passes sanctioned shapes live in the loop.
    raw = os.getenv("EXP_TASK_IDS")
    if raw is not None and raw.strip():
        selected = [task_id.strip() for task_id in raw.split(",") if task_id.strip()]
    else:
        selected = [t for panel in harness_config.panels for t in panel.task_names]
    return list(dict.fromkeys(selected))


def _selected_experiment_id() -> str:
    # `EXP_EXPERIMENT_ID` names an existing dir to append into (auto's shared id
    # across its train and test calls); a standalone run gets a fresh id (§2).
    existing = os.getenv("EXP_EXPERIMENT_ID")
    if existing is not None and existing.strip():
        return existing.strip()
    return f"exp-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"


def _task_timeout_by_task(harness_config) -> dict[str, float]:
    # Each task carries its owning panel's wall budget; tasks are disjoint across
    # panels (config §12), so this is unambiguous for any selected subset.
    return {
        task_id: panel.task_timeout_sec
        for panel in harness_config.panels
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


async def _resolve_task_dirs(*, trial_harbor_config, task_names):
    from src.env.harbor import TaskDirectoryResolver

    return dict(
        await TaskDirectoryResolver(trial_harbor_config).resolve(list(task_names))
    )


async def _build_trial_runner(
    *, harness_config, harbor_config, api_key: str | None, task_ids, experiment_id
):
    """Resolve the selected tasks' directories once, then return the `run_tasks`
    trial_runner -- the seam the orchestrator schedules. It builds each trial's
    Harbor (handed the run-scoped heavy gate) + llm and calls `executor.run_trial`
    with that task's panel wall budget."""
    from src.env.harbor import Harbor
    from src.experiment.executor import run_trial

    trial_harbor_config = harbor_config.model_copy(
        update={
            "experiments_dir": harbor_config.experiments_dir / experiment_id / "tasks"
        }
    )
    task_dirs = await _resolve_task_dirs(
        trial_harbor_config=trial_harbor_config, task_names=task_ids
    )
    task_timeout_sec = _task_timeout_by_task(harness_config)

    async def trial_runner(task_id, run_id, heavy_action_semaphore, slot_release):
        env = Harbor(
            trial_harbor_config,
            task_name=task_id,
            task_dir=task_dirs[task_id],
            exec_semaphore=heavy_action_semaphore,
        )
        return await run_trial(
            task_id=task_id,
            run_id=run_id,
            llm=_make_llm_for_config(
                config=harness_config.llm_provider_config, api_key=api_key
            ),
            env=env,
            max_steps=harness_config.max_steps,
            max_output_retries=harness_config.max_output_retries,
            task_timeout_sec=task_timeout_sec[task_id],
            env_setup_timeout_sec=harness_config.env_setup_timeout_sec,
            slot_release=slot_release,
        )

    return trial_runner


async def run_experiment(
    *,
    harness_config,
    harbor_config,
    git_commit_hash,
    task_ids,
    experiment_id,
    trial_runner,
):
    """Run the selected `task_ids` at the uniform full budget into `experiment_id`
    -> a raw `ExperimentResult` (plan.md §2). One `run_tasks` call per `uv run exp`
    invocation; an existing `experiment_id` is appended to (auto's train-then-test
    across two calls). No baseline, gate, decision, or git -- the loop's (Step 5)."""
    from src.experiment.orchestrator import run_tasks

    return await run_tasks(
        experiment_id=experiment_id,
        git_commit_hash=git_commit_hash,
        task_ids=task_ids,
        budget={task_id: harness_config.task_trials for task_id in task_ids},
        full_trial_count=harness_config.task_trials,
        max_trial_concurrency=harness_config.max_trial_concurrency,
        max_heavy_action_concurrency=harness_config.max_heavy_action_concurrency,
        trial_runner=trial_runner,
        experiments_root=harbor_config.experiments_dir,
    )


async def _run_exp_async(
    *, harness_config, harbor_config, api_key, git_commit_hash, task_ids, experiment_id
):
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
    )


def main_exp() -> int:
    import asyncio
    import sys

    from src.llm.codex import (
        CODEX_CREDENTIALS_EXPIRED_EXIT_CODE,
        ChatGptCodexCredentialsExpiredError,
    )
    from src.env.harbor import DEFAULT_HARBOR_CONFIG_PATH
    from src.repo import get_head_commit, require_clean_worktree

    harbor_config, harness_config, api_key = load_runtime_config()
    task_ids = _selected_task_ids(harness_config)
    experiment_id = _selected_experiment_id()
    print(f"experiment: {experiment_id}")
    print(f"tasks ({len(task_ids)}): {', '.join(task_ids)}")
    print(f"harbor config: {DEFAULT_HARBOR_CONFIG_PATH}")
    print(f"harness config: {DEFAULT_HARNESS_CONFIG_PATH}")
    if _require_clean_worktree_for_exp():
        require_clean_worktree()
    git_commit_hash = get_head_commit()
    try:
        result = asyncio.run(
            _run_exp_async(
                harness_config=harness_config,
                harbor_config=harbor_config,
                api_key=api_key,
                git_commit_hash=git_commit_hash,
                task_ids=task_ids,
                experiment_id=experiment_id,
            )
        )
    except ChatGptCodexCredentialsExpiredError as exc:
        print(str(exc), file=sys.stderr)
        return CODEX_CREDENTIALS_EXPIRED_EXIT_CODE
    print(f"run status: {result.run_status}")
    print("run complete")
    return 0


def main_auto() -> int:
    import sys

    from src.llm.codex import (
        CODEX_CREDENTIALS_EXPIRED_EXIT_CODE,
        ChatGptCodexCredentialsExpiredError,
    )
    from src.supervisor.agent_backend import create_backend
    from src.control.supervisor import run_supervisor_loop

    agent_type = "codex"
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg in ("--agent", "-a") and i < len(sys.argv) - 1:
            agent_type = sys.argv[i + 1]
            break
        if arg.startswith("--agent="):
            agent_type = arg.split("=", 1)[1]
            break

    backend = create_backend(agent_type)
    try:
        run_supervisor_loop(
            repo_root=Path(__file__).resolve().parents[1],
            backend=backend,
        )
    except ChatGptCodexCredentialsExpiredError as exc:
        print(f"\nSupervisor halted. {exc}", file=sys.stderr)
        return CODEX_CREDENTIALS_EXPIRED_EXIT_CODE
    except KeyboardInterrupt:
        return 130
    return 0
