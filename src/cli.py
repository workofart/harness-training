from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from src.harness.config import DEFAULT_HARNESS_CONFIG_PATH

if TYPE_CHECKING:
    from src.adapters.env import HarborConfig
    from src.harness.config import HarnessConfig


def load_strict_runtime_config(
    *,
    harbor_config_path: Path,
    harness_config_path: Path,
) -> tuple[HarborConfig, HarnessConfig]:
    from src.adapters.env import HarborConfig
    from src.harness.config import HarnessConfig

    harbor_config = HarborConfig.from_toml(harbor_config_path)
    harness_config = HarnessConfig.model_validate_json(harness_config_path.read_text())
    return harbor_config, harness_config


def _load_llm_provider_secret(
    *,
    harness_config: HarnessConfig,
    dotenv_path: str | Path = ".env",
) -> str | None:
    if harness_config.llm_provider_config.provider == "openrouter":
        from src.adapters.open_router import load_openrouter_api_key

        return load_openrouter_api_key(dotenv_path=dotenv_path)
    return None


def load_runtime_config() -> tuple[HarborConfig, HarnessConfig, str | None]:
    from src.adapters.env import DEFAULT_HARBOR_CONFIG_PATH

    harbor_config, harness_config = load_strict_runtime_config(
        harbor_config_path=DEFAULT_HARBOR_CONFIG_PATH,
        harness_config_path=DEFAULT_HARNESS_CONFIG_PATH,
    )
    api_key = _load_llm_provider_secret(harness_config=harness_config)
    return harbor_config, harness_config, api_key


def _require_clean_worktree_for_exp() -> bool:
    return os.getenv("EXP_ALLOW_DIRTY_WORKTREE", "").lower() not in {
        "1",
        "true",
        "yes",
    }


def _panel_task_summary(harness_config: HarnessConfig) -> str:
    summaries = []
    for panel in harness_config.panels:
        task_label = "task" if len(panel.task_names) == 1 else "tasks"
        summaries.append(
            f"{panel.id}({panel.purpose}): {len(panel.task_names)} {task_label}"
        )
    return "; ".join(summaries)


def main_exp() -> int:
    import sys

    from src.adapters.chatgpt_codex import (
        CODEX_CREDENTIALS_EXPIRED_EXIT_CODE,
        ChatGptCodexCredentialsExpiredError,
    )
    from src.adapters.env import DEFAULT_HARBOR_CONFIG_PATH
    from src.experiment.runner import ExperimentRunner

    harbor_config, harness_config, api_key = load_runtime_config()
    print(f"experiment: {harness_config.experiment_id}")
    print(f"panels: {_panel_task_summary(harness_config)}")
    print(f"harbor config: {DEFAULT_HARBOR_CONFIG_PATH}")
    print(f"harness config: {DEFAULT_HARNESS_CONFIG_PATH}")
    try:
        record = ExperimentRunner(
            harness_config=harness_config,
            harbor_config=harbor_config,
            api_key=api_key,
            require_clean_worktree=_require_clean_worktree_for_exp(),
        ).run()
    except ChatGptCodexCredentialsExpiredError as exc:
        print(str(exc), file=sys.stderr)
        return CODEX_CREDENTIALS_EXPIRED_EXIT_CODE
    print(f"evaluation: {record.status}")
    print("run complete")
    return 0


def main_auto() -> int:
    import sys

    from src.adapters.chatgpt_codex import (
        CODEX_CREDENTIALS_EXPIRED_EXIT_CODE,
        ChatGptCodexCredentialsExpiredError,
    )
    from src.control.agent_backend import create_backend
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
