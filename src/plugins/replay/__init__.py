"""Certified-replay execution plugin.

``ReplayExecution`` is composed exclusively by
``src/rollout/execution.py:resolve_execution``. Regime rules: src/plugins/README.md.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from src.config import RunConfig
from src.env.base import TaskEnv, VerifyWrapper, benchmark
from src.plugins.replay import audit, contract, step_cache, verify_cache
from src.rollout.execution import LiveStepExecutor, StepExecutor

if TYPE_CHECKING:
    from src.rollout.records import ExperimentResult
    from src.rollout.store import RunStore


class ReplayExecution:
    """Certified-replay regime: recorded steps replay, drift audits certify."""

    def __init__(self, run_config: RunConfig) -> None:
        self._run_config = run_config
        # Injected so the verify cache stays out of env.
        self.verify_wrapper: VerifyWrapper | None = verify_cache.cache_wrapper

    async def fingerprint(self, task_ids: Sequence[str]) -> list[str]:
        taskset = await benchmark(self._run_config.environment.kind).load_tasks(
            task_ids=task_ids,
            environment=self._run_config.environment,
            verify_wrapper=verify_cache.cache_wrapper,
        )
        lines = []
        for task_id in task_ids:
            scope = await contract.resolve_scope(taskset.content_id(task_id))
            if scope is None:
                lines.append(f"{task_id}:live")
                continue
            namespace, epoch = scope
            lines.append(f"{namespace}:{epoch}")
        return lines

    async def certify(
        self,
        *,
        tracker: RunStore,
        baseline: ExperimentResult,
        log: Callable[[str], None],
        inherited: ExperimentResult | None = None,
    ) -> tuple[ExperimentResult, tuple[str, ...]]:
        return await audit.exclude_nondeterministic_tasks(
            run_config=self._run_config,
            tracker=tracker,
            baseline=baseline,
            log=log,
            inherited=inherited,
        )

    async def step_executor(
        self, *, content_id: str | None, env: TaskEnv, force_live: bool = False
    ) -> StepExecutor:
        if force_live:
            return LiveStepExecutor(env)
        replay = await step_cache.make_replay_cache(content_id=content_id, env=env)
        return LiveStepExecutor(env) if replay is None else replay
