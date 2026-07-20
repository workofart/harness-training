"""Core-owned environment step execution seam.

Core wraps ``provision`` in the setup timeout, ``step_run`` and
``prepare_submit`` in the agent deadline, and ``step_verify`` in the verify cap
plus backstop. Implementations may spend those budgets but never extend them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from src.config import RunConfig
from src.env.base import (
    RunAction,
    StepResult,
    TaskEnv,
    VerifyAction,
    VerifyWrapper,
    execute_env_action,
)

if TYPE_CHECKING:
    from src.rollout.records import ExperimentResult
    from src.rollout.store import RunStore

_LOGGER = logging.getLogger(__name__)


class ExecutionDriftError(RuntimeError):
    """Replayed execution contradicted this rollout; caller must retry live-only.

    ``diagnostic`` is an opaque, JSON-ready payload built by the raising
    regime; core writes it to the drift artifact without knowing its fields.
    """

    def __init__(self, *, action_index: int, diagnostic: dict[str, Any]) -> None:
        super().__init__(
            "execution drift during catch-up: "
            f"action {action_index} no longer reproduces its recorded observation; "
            "the replayed prefix is contaminated"
        )
        self.action_index = action_index
        self.diagnostic = diagnostic


class StepExecutor(Protocol):
    """Execute env steps without owning or extending the core's time budgets."""

    async def provision(self) -> None: ...

    async def step_run(self, action: RunAction) -> StepResult: ...

    async def prepare_submit(self) -> StepResult | None: ...

    async def step_verify(self) -> StepResult: ...


class LiveStepExecutor:
    """Identity executor: every action runs directly against the live env."""

    def __init__(self, env: TaskEnv) -> None:
        self._env = env

    async def provision(self) -> None:
        await self._env.provision()

    async def step_run(self, action: RunAction) -> StepResult:
        return await execute_env_action(self._env, action)

    async def prepare_submit(self) -> StepResult | None:
        return None

    async def step_verify(self) -> StepResult:
        return await execute_env_action(self._env, VerifyAction())


class Execution(Protocol):
    """One measurement-execution regime: identity, certification, step execution.

    Fully async: sync drivers enter via asyncio.run at their boundary, and no
    implementation may start an event loop of its own.
    """

    verify_wrapper: VerifyWrapper | None

    async def fingerprint(self, task_ids: Sequence[str]) -> list[str]: ...

    async def certify(
        self,
        *,
        tracker: RunStore,
        baseline: ExperimentResult,
        log: Callable[[str], None],
        inherited: ExperimentResult | None = None,
    ) -> tuple[ExperimentResult, tuple[str, ...]]: ...

    async def step_executor(
        self, *, content_id: str | None, env: TaskEnv, force_live: bool = False
    ) -> StepExecutor: ...


class EagerExecution:
    """Identity regime: every action runs live, certification passes through."""

    verify_wrapper: VerifyWrapper | None = None

    async def fingerprint(self, task_ids: Sequence[str]) -> list[str]:
        return [f"{task_id}:live" for task_id in task_ids]

    async def certify(
        self,
        *,
        tracker: RunStore,
        baseline: ExperimentResult,
        log: Callable[[str], None],
        inherited: ExperimentResult | None = None,
    ) -> tuple[ExperimentResult, tuple[str, ...]]:
        del tracker, log, inherited
        return baseline, tuple(baseline.tasks)

    async def step_executor(
        self, *, content_id: str | None, env: TaskEnv, force_live: bool = False
    ) -> StepExecutor:
        del content_id, force_live
        return LiveStepExecutor(env)


def resolve_execution(run_config: RunConfig) -> Execution:
    """The single composition point; execution flags are read nowhere else.

    Config validation guarantees a replay config's provider is deterministic.
    The cache store is the one runtime dependency: losing it degrades loudly
    to eager, and the eager fingerprint keeps the identity digest honest.
    """
    if run_config.plugins.execution != "replay":
        return EagerExecution()

    from src.plugins.caching import store as cache

    if cache.disabled():
        _LOGGER.warning(
            'plugins.execution "replay" degraded to eager: cache store disabled'
        )
        return EagerExecution()

    from src.plugins.replay import ReplayExecution

    return ReplayExecution(run_config)
