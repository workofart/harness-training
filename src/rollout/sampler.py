"""Experiment orchestration: fan out one rollout per task, with infra-retry.

``run_experiment`` is the framework-owned sampler -- it loads the task set once,
then each retry attempt constructs its own env and cached backend before running a
``RolloutRunner``. ``_run_rollout_with_infra_retries`` wraps one rollout with the
deterministic-provider-only infra retry envelope and its diagnostic artifacts.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from git import Repo

from src.concurrency import CpuConcurrencyLimiter, cross_process_task_lock
from src.config import RunConfig
from src.env.base import (
    benchmark,
    TaskEnv,
    TaskHandle,
)
from src.rollout.certification import resolve_measurement_identity
from src.rollout.episode import RolloutRunner
from src.rollout.execution import ExecutionDriftError, Execution, resolve_execution
from src.rollout.records import (
    ExperimentResult,
    RolloutResult,
    error_text,
)
from src.rollout.store import RunObserver, RunStore
from src.llm.backend import make_backend
import src.rollout.telemetry as tm

# Backstops outside run_rollout budgets; the trainer process is the final watchdog.
ENV_FACTORY_SETUP_TIMEOUT_SEC = 900.0

# Docker execs hold to_thread workers for full command duration.
TO_THREAD_POOL_WORKERS = 64


def _save_run_ref(repo: Repo, experiment_id: str, commit_hash: str) -> None:
    """Persist a run-local git ref for the commit named in ``experiment.json``."""
    ref = f"refs/experiments/runs/{experiment_id}"
    repo.git.update_ref(ref, repo.commit(commit_hash).hexsha)


async def run_experiment(
    *,
    run_config: RunConfig,
    tracker: RunStore,
    observer: RunObserver,
    experiment_id: str | None = None,
) -> ExperimentResult:
    # One experiment per process: 64 must exceed rollout concurrency × Docker execs per rollout.
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=TO_THREAD_POOL_WORKERS)
    )
    task_ids = tuple(run_config.environment.task_names)
    experiment_id = experiment_id or _new_experiment_id()
    repo = Repo(Path.cwd())
    commit_hash = repo.head.commit.hexsha
    _save_run_ref(repo, experiment_id, commit_hash)
    execution = resolve_execution(run_config)
    measurement_identity = await resolve_measurement_identity(run_config, execution)
    result = ExperimentResult(
        experiment_id=experiment_id,
        git_commit_hash=commit_hash,
        measurement_identity=measurement_identity,
        git_dirty=repo.is_dirty(),
        config_path=run_config.config_path,
        started_at=datetime.now(UTC),
        tasks={task_id: None for task_id in task_ids},
    )
    tracker.save_experiment(result)
    # Report early so a parent watchdog can finalize a wedged child as crashed.
    observer.experiment_started(experiment_id)

    try:
        taskset = await asyncio.wait_for(
            benchmark(run_config.environment.kind).load_tasks(
                task_ids=task_ids,
                environment=run_config.environment,
                verify_wrapper=execution.verify_wrapper,
            ),
            timeout=ENV_FACTORY_SETUP_TIMEOUT_SEC,
        )
        handles = {
            task_id: taskset.task(
                task_id, timeout_multiplier=run_config.agent_timeout_multiplier
            )
            for task_id in task_ids
        }
        rollout_limiter = CpuConcurrencyLimiter(run_config.max_rollout_concurrency)

        async def run_one(task_id: str) -> None:
            async with rollout_limiter:
                # Fixed same-task Docker subnets collide across processes; lock host-wide outside rollout budgets.
                async with cross_process_task_lock(
                    f"{run_config.environment.kind}:{task_id}",
                    on_wait=lambda: observer.log(
                        f"{task_id}: held by a concurrent experiment; waiting"
                    ),
                ):
                    rollout_dir = tracker.task_dir(experiment_id, task_id)
                    trace_path = tracker.trace_path(experiment_id, task_id)
                    rollout_result = await _run_rollout_with_infra_retries(
                        task=handles[task_id],
                        rollout_dir=rollout_dir,
                        trace_path=trace_path,
                        run_config=run_config,
                        execution=execution,
                        log=observer.log,
                    )
            tracker.log_rollout(experiment_id, rollout_result)
            observer.task_finished(task_id, rollout_result.failure_mode)

        tasks = [asyncio.create_task(run_one(task_id)) for task_id in task_ids]
        try:
            await asyncio.gather(*tasks)
        finally:
            # Cancel siblings, then drain so every exception is retrieved.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    except BaseException as exc:
        tracker.mark_crashed(experiment_id, reason=error_text(exc))
        raise

    loaded = tracker.load_experiment(experiment_id)
    completed = loaded.model_copy(
        update={
            "finished_at": datetime.now(UTC),
            "config": run_config.model_dump(mode="json"),
        }
    )
    tracker.save_experiment(completed)
    return completed


async def _run_rollout_with_infra_retries(
    *,
    task: TaskHandle,
    rollout_dir: Path,
    trace_path: Path,
    run_config: RunConfig,
    execution: Execution,
    log: Callable[[str], None],
) -> RolloutResult:
    task_id = task.task_id
    infra_retries: list[dict[str, Any]] = []
    deterministic = run_config.llm_provider_config.is_deterministic

    async def run_attempt() -> RolloutResult:
        env: TaskEnv = task.make_env(rollout_dir)
        executor = await execution.step_executor(
            content_id=task.content_id,
            env=env,
            force_live=drift_retried,
        )
        return await RolloutRunner(
            task_id=task_id,
            llm=make_backend(
                run_config.llm_provider_config,
                cache=run_config.plugins.llm_cache,
            ),
            env=env,
            executor=executor,
            run_config=run_config,
            telemetry=tm.RolloutTelemetry(
                rollout_dir=rollout_dir,
                trace_path=trace_path,
            ),
            agent_timeout_sec=task.agent_timeout_sec,
        ).run()

    drift_retried = False
    slow_llm_retried = False
    while True:
        try:
            rollout_result = await run_attempt()
        except ExecutionDriftError as exc:
            drift_retried = True
            artifact_path = _write_execution_drift_artifact(rollout_dir, exc)
            archived_trace_path = _archive_trace(trace_path, "execution-drift")
            infra_retries.append(
                {
                    "kind": "execution_drift",
                    "action_index": exc.action_index,
                    "artifact_path": str(artifact_path),
                    "archived_trace_path": str(archived_trace_path),
                }
            )
            log(
                f"{task_id}: execution_drift at action "
                f"{exc.action_index}; retrying live-only"
            )
            continue
        # Nondeterministic retry would resample, forging the measurement.
        retry = (
            None
            if not deterministic
            else _slow_llm_timeout_retry(
                rollout_result,
                agent_timeout_sec=task.agent_timeout_sec,
            )
        )
        if retry is not None and not slow_llm_retried:
            slow_llm_retried = True
            archived_trace_path = _archive_trace(trace_path, "slow-llm-timeout")
            retry["archived_trace_path"] = str(archived_trace_path)
            infra_retries.append(retry)
            log(f"{task_id}: slow_llm_timeout; retrying once")
            continue
        break

    if infra_retries:
        update: dict[str, Any] = {"infra_retries": infra_retries}
        if (
            drift_retried
            and rollout_result.failure_mode == "crash"
            and rollout_result.failure_origin == "env"
        ):
            update["failure_mode"] = "unscorable_infra"
            update["failure_origin"] = None
        rollout_result = rollout_result.model_copy(update=update)
    return rollout_result


def _archive_trace(trace_path: Path, suffix: str) -> Path:
    archived_trace_path = trace_path.with_name(
        f"{trace_path.stem}.{suffix}{trace_path.suffix}"
    )
    trace_path.replace(archived_trace_path)
    return archived_trace_path


def _write_execution_drift_artifact(
    rollout_dir: Path, exc: ExecutionDriftError
) -> Path:
    path = rollout_dir / "infra" / "execution_drift.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "execution_drift",
        "action_index": exc.action_index,
        **exc.diagnostic,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _slow_llm_timeout_retry(
    rollout_result: RolloutResult,
    *,
    agent_timeout_sec: float,
) -> dict[str, Any] | None:
    metrics = rollout_result.metrics
    if (
        rollout_result.failure_mode != "hit_timeout"
        or metrics.get(tm.LIVE_LLM_CALLS_KEY, 0) < 3
        or tm.P25_LIVE_OUTPUT_TOKENS_PER_SEC_KEY not in metrics
    ):
        return None
    if not (
        metrics[tm.SUM_LIVE_LLM_LATENCY_SEC_KEY] >= 0.75 * agent_timeout_sec
        and metrics[tm.MEDIAN_LIVE_LLM_LATENCY_SEC_KEY] >= 20.0
        and metrics[tm.P25_LIVE_OUTPUT_TOKENS_PER_SEC_KEY] < 10.0
    ):
        return None
    return {
        "kind": "slow_llm_timeout",
        tm.LIVE_LLM_CALLS_KEY: metrics[tm.LIVE_LLM_CALLS_KEY],
        tm.SUM_LIVE_LLM_LATENCY_SEC_KEY: metrics[tm.SUM_LIVE_LLM_LATENCY_SEC_KEY],
        tm.MEDIAN_LIVE_LLM_LATENCY_SEC_KEY: metrics[tm.MEDIAN_LIVE_LLM_LATENCY_SEC_KEY],
        tm.P25_LIVE_OUTPUT_TOKENS_PER_SEC_KEY: metrics[
            tm.P25_LIVE_OUTPUT_TOKENS_PER_SEC_KEY
        ],
    }


def _new_experiment_id() -> str:
    return "exp-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
