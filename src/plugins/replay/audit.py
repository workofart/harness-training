"""Determinism audit: re-execute recorded chains and exclude tasks that fork."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from src.concurrency import CpuConcurrencyLimiter
from src.config import RunConfig
from src.env.base import (
    RunAction,
    TaskEnv,
    TaskHandle,
    VERIFY_BACKSTOP_SLACK_SEC,
    VerifyAction,
    benchmark,
    execute_env_action,
    scrub_step_result,
)
from src.plugins.replay import verify_cache
from src.plugins.replay.contract import apply_replay_scope, namespace_for
from src.rollout.certification import (
    DETERMINISM_CHAIN_RELPATH,
    scrubbed_hash,
    verdict_summary,
)
from src.rollout.records import (
    ExperimentResult,
    TaskCertification,
)
from src.rollout.store import RunStore

_FORK_ARTIFACT_RELPATH = "infra/determinism_fork.json"


async def exclude_nondeterministic_tasks(
    *,
    run_config: RunConfig,
    tracker: RunStore,
    baseline: ExperimentResult,
    log: Callable[[str], None],
    inherited: ExperimentResult | None = None,
) -> tuple[ExperimentResult, tuple[str, ...]]:
    """Certify the baseline's recorded chains and return its deterministic panel."""
    cached = baseline.determinism_certification is not None
    inherited_tasks: set[str] = set()
    if not cached:
        certification: dict[str, TaskCertification] = {}
        if (
            inherited is not None
            and inherited.determinism_certification is not None
            and inherited.measurement_identity.digest
            == baseline.measurement_identity.digest
        ):
            for task_id in baseline.tasks:
                prior = inherited.determinism_certification.get(task_id)
                if prior is None:
                    continue
                _rows, digest, has_timeout = _load_chain(
                    tracker.task_dir(baseline.experiment_id, task_id)
                )
                if has_timeout:
                    continue
                if digest == prior.chain_digest:
                    certification[task_id] = prior
                    inherited_tasks.add(task_id)
        audit_task_ids = tuple(
            task_id for task_id in baseline.tasks if task_id not in inherited_tasks
        )
        if audit_task_ids:
            certification.update(
                await _certify_tasks(
                    run_config=run_config,
                    tracker=tracker,
                    baseline=baseline,
                    task_ids=audit_task_ids,
                    log=log,
                )
            )
        baseline = baseline.model_copy(
            update={"determinism_certification": certification}
        )
        tracker.save_experiment(baseline)

    certification = baseline.determinism_certification
    assert certification is not None
    excluded = {
        task_id: item.verdict
        for task_id, item in certification.items()
        if item.verdict != "deterministic"
    }
    task_ids = tuple(task_id for task_id in baseline.tasks if task_id not in excluded)
    summary = f"excluded {len(excluded)}/{len(baseline.tasks)} nondeterministic tasks"
    if cached:
        log(f"determinism: cached ({baseline.experiment_id}, {summary})")
    elif inherited_tasks:
        log(
            f"determinism: inherited {len(inherited_tasks)}/{len(baseline.tasks)} "
            f"task certificates ({summary})"
        )
    else:
        log(f"determinism: {summary}")
    for task_id, reason in excluded.items():
        detail = f"excluded: {task_id} ({reason})"
        # Inherited fork artifacts remain under the baseline that computed them.
        if reason == "forked" and task_id not in inherited_tasks:
            artifact = (
                tracker.task_dir(baseline.experiment_id, task_id)
                / _FORK_ARTIFACT_RELPATH
            )
            detail += f" {artifact}"
        log(detail)
    if not task_ids:
        raise RuntimeError("determinism check excluded every task")
    return baseline, task_ids


# Leave one core free so audit contention cannot create false timeout forks.
_AUDIT_MAX_CONCURRENCY = max(1, (os.cpu_count() or 2) - 1)


def _chain_digest(rows: Sequence[dict[str, Any]]) -> str:
    action_keys = (
        json.dumps(row["action"], sort_keys=True, separators=(",", ":")) for row in rows
    )
    return hashlib.sha256("\n".join(action_keys).encode()).hexdigest()


def _load_chain(task_dir: Path) -> tuple[list[dict[str, Any]] | None, str, bool]:
    """One task's recorded chain: (rows, digest, has_timeout); rows None = no file."""
    path = task_dir / DETERMINISM_CHAIN_RELPATH
    try:
        rows: list[dict[str, Any]] | None = [
            json.loads(line) for line in path.read_text().splitlines()
        ]
    except FileNotFoundError:
        rows = None
    digest = _chain_digest(rows or [])
    has_timeout = rows is not None and any(row.get("timed_out") is True for row in rows)
    return rows, digest, has_timeout


async def _certify_tasks(
    *,
    run_config: RunConfig,
    tracker: RunStore,
    baseline: ExperimentResult,
    task_ids: Sequence[str],
    log: Callable[[str], None],
) -> dict[str, TaskCertification]:
    taskset = await benchmark(run_config.environment.kind).load_tasks(
        task_ids=task_ids,
        environment=run_config.environment,
        verify_wrapper=verify_cache.cache_wrapper,
    )
    limiter = CpuConcurrencyLimiter(
        min(run_config.max_rollout_concurrency, _AUDIT_MAX_CONCURRENCY)
    )
    handles = {
        task_id: taskset.task(
            task_id, timeout_multiplier=run_config.agent_timeout_multiplier
        )
        for task_id in task_ids
    }

    async def check_one(task_id: str) -> tuple[str, TaskCertification]:
        rows, chain_digest, has_timeout = _load_chain(
            tracker.task_dir(baseline.experiment_id, task_id)
        )
        if rows is None or has_timeout:
            # A timed-out chain is not replayable, even by truncating it.
            return task_id, TaskCertification(
                chain_digest=chain_digest, verdict="no_chain"
            )
        handle = handles[task_id]
        async with limiter:
            namespace = namespace_for(handle.content_id)
            reason = await _check_task(
                handle=handle,
                rows=rows,
                tracker=tracker,
                baseline=baseline,
                replay_namespace=namespace,
            )
        if reason == "forked" and namespace is not None:
            log(
                f"fork hint: {task_id} — stale recording? bump: uv run python -m "
                f"src.plugins.replay.bump_epoch '{namespace}'; forks persisting after "
                "re-record are real nondeterminism (fix the scrub, not the epoch)"
            )
        return task_id, TaskCertification(
            chain_digest=chain_digest,
            verdict="deterministic" if reason is None else reason,
        )

    pairs = await asyncio.gather(*(check_one(task_id) for task_id in task_ids))
    return dict(pairs)


async def _check_task(
    *,
    handle: TaskHandle,
    rows: list[dict[str, Any]],
    tracker: RunStore,
    baseline: ExperimentResult,
    replay_namespace: str | None,
) -> Literal["forked"] | None:
    task_id = handle.task_id
    env = handle.make_env(
        tracker.task_dir(baseline.experiment_id, task_id) / "determinism_check",
    )
    try:
        # Pin replay scope before reset builds the container setup command.
        await apply_replay_scope(content_id=handle.content_id, env=env)
        async with asyncio.timeout(env.setup_timeout_sec):
            await env.reset()
            await env.provision()
        # Match live budgets: runs share the agent deadline; verify uses its grader deadline.
        run_deadline = asyncio.get_running_loop().time() + handle.agent_timeout_sec
        for action_index, row in enumerate(rows, start=1):
            action = row["action"]
            row_timeout = (
                asyncio.timeout_at(run_deadline)
                if action["kind"] == "run"
                else asyncio.timeout(env.verify_timeout_sec + VERIFY_BACKSTOP_SLACK_SEC)
            )
            async with row_timeout:
                mismatch = await _compare_action(env, row)
            if mismatch is not None:
                _write_fork(
                    tracker=tracker,
                    baseline=baseline,
                    task_id=task_id,
                    payload={
                        "task": task_id,
                        "replay_namespace": replay_namespace,
                        "action_index": action_index,
                        "action": action,
                        **mismatch,
                    },
                )
                return "forked"
    finally:
        await env.close()
    return None


async def _compare_action(env: TaskEnv, row: dict[str, Any]) -> dict[str, Any] | None:
    action = row["action"]
    if action["kind"] == "run":
        env_action: RunAction | VerifyAction = RunAction(
            command=action["command"],
            cwd=action["cwd"],
            timeout_sec=action["timeout_sec"],
        )
    else:
        env_action = VerifyAction()
    result = await execute_env_action(env, env_action)
    command = env_action.command if isinstance(env_action, RunAction) else None
    live = scrub_step_result(result, command=command)
    if isinstance(env_action, RunAction):
        live_hash = scrubbed_hash(live, command=command)
        if live_hash == row["audit_hash"]:
            return None
        return {
            "recorded_audit_hash": row["audit_hash"],
            "live": dataclasses.asdict(live),
        }

    live_verdict = verdict_summary(live)
    if live_verdict == row["verdict"]:
        return None
    return {
        "recorded_audit_hash": None,
        "live": dataclasses.asdict(live),
        "verdict_mismatch": {
            "recorded": row["verdict"],
            "live": live_verdict,
        },
    }


def _write_fork(
    *,
    tracker: RunStore,
    baseline: ExperimentResult,
    task_id: str,
    payload: dict[str, Any],
) -> None:
    path = tracker.task_dir(baseline.experiment_id, task_id) / _FORK_ARTIFACT_RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
