"""Provider preflight and isolated experiment process supervision."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import docker

from src.config import RunConfig
from src.llm.backend import Completion, CompletionRequest, make_backend
from src.policy.base import SUBMIT_ACTION_NAME
from src.rollout.records import ExperimentResult
from src.rollout.store import RunObserver, RunStore


class PreflightError(RuntimeError):
    """A launch precondition (docker, provider reachability) is unavailable."""


# Fail fast on unreachable rollout providers before a full run can crash deep inside.
_SMOKE_TIMEOUT_SEC = 120.0
_SMOKE_MESSAGES = [
    {
        "role": "user",
        "content": "Call the submit tool now. Do not answer with text.",
    }
]

# Use a frozen tool spec so preflight never imports candidate-editable policy.
_SMOKE_TOOL_SPEC: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": SUBMIT_ACTION_NAME,
            "description": "Submit the smoke-check response.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }
]


def _assert_llm_provider_reachable(run_config: RunConfig) -> None:
    """Send one minimal completion to the rollout provider; raise if it does not answer.

    Builds the exact backend the rollouts use (same endpoint, auth, and request
    shaping) via the shared factory, so a green check means the real path works.
    """
    try:
        completion = asyncio.run(_smoke_complete(run_config))
    except Exception as exc:  # connection, auth, timeout, SDK, missing-env errors
        raise PreflightError(
            f"rollout LLM provider smoke request failed: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        tool_call = completion.tool_calls[0]
        if tool_call.name != SUBMIT_ACTION_NAME:
            raise ValueError(
                f"expected {SUBMIT_ACTION_NAME} tool call, got {tool_call.name!r}"
            )
    except Exception as exc:
        raise PreflightError(
            "rollout LLM provider smoke response did not return a valid submit "
            f"tool call: {type(exc).__name__}: {exc}; "
            f"completion={_format_smoke_completion(completion)}"
        ) from exc


async def _smoke_complete(run_config: RunConfig) -> Completion:
    backend = make_backend(run_config.llm_provider_config)
    try:
        async with asyncio.timeout(_SMOKE_TIMEOUT_SEC):
            return await backend.complete(
                CompletionRequest(messages=_SMOKE_MESSAGES, tools=_SMOKE_TOOL_SPEC)
            )
    finally:
        await backend.close()


def _format_smoke_completion(completion: Completion) -> str:
    content = repr(completion.content)
    if len(content) > 200:
        content = content[:197] + "..."
    return (
        f"finish_reason={completion.finish_reason!r}, "
        f"content={content}, "
        f"tool_calls={completion.tool_calls!r}"
    )


def preflight(configs: Sequence[RunConfig]) -> None:
    try:
        client = docker.from_env(timeout=10)
        try:
            client.ping()
        finally:
            client.close()
    except (OSError, docker.errors.DockerException) as exc:
        raise PreflightError(
            "Docker daemon is not reachable -- start Docker/OrbStack"
        ) from exc

    # A missing api_key_env fails inside the smoke request with the KeyError
    # naming the variable; no separate env check needed.
    unique: dict[str, RunConfig] = {}
    for config in configs:
        unique.setdefault(config.llm_provider_config.model_dump_json(), config)
    for config in unique.values():
        _assert_llm_provider_reachable(config)


_WATCHDOG_POLL_SEC = 2.0
WATCHDOG_INACTIVITY_SEC = 1800.0
_CHILD_SHUTDOWN_GRACE_SEC = 30.0


def _terminate_experiment_process(process: subprocess.Popen[bytes]) -> int:
    pid = process.pid
    if process.poll() is None:
        try:
            process.wait(timeout=_CHILD_SHUTDOWN_GRACE_SEC)
        except (KeyboardInterrupt, subprocess.TimeoutExpired):
            pass
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=_CHILD_SHUTDOWN_GRACE_SEC)
        except (KeyboardInterrupt, subprocess.TimeoutExpired):
            process.kill()
            process.wait()
    exitcode = process.returncode
    assert exitcode is not None
    from src.env.docker_shell import DockerShellSession

    DockerShellSession.sweep_owner_resources(pid)
    return exitcode


class MeasurementError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        result: ExperimentResult | None,
    ) -> None:
        super().__init__(message)
        self.result = result


def _load_crashed_result(
    *,
    tracker: RunStore,
    observer: RunObserver,
    experiment_id: str | None,
    reason: str,
) -> ExperimentResult | None:
    if experiment_id is None:
        return None
    try:
        return tracker.mark_crashed(experiment_id, reason=reason)
    except Exception as exc:
        observer.log(f"measurement: could not finalize {experiment_id} crashed: {exc}")
        return None


def run_isolated_experiment(
    *,
    config_path: str,
    tracker: RunStore,
    observer: RunObserver,
    measure_root: Path,
    task_ids: tuple[str, ...],
    experiment_id: str | None = None,
) -> ExperimentResult:
    """The subprocess loads config from ``config_path`` itself; the parent's
    in-memory config is deliberately not trusted across the candidate boundary."""
    from src.plugins.caching.store import DB_PATH

    experiments_root = tracker.root.resolve()
    experiments_root.mkdir(parents=True, exist_ok=True)
    receive, send = multiprocessing.Pipe(duplex=False)
    try:
        command = [
            sys.executable,
            "-m",
            "src.worker",
            config_path,
            str(experiments_root),
            str(send.fileno()),
            json.dumps(task_ids),
            json.dumps(experiment_id),
        ]
        process = subprocess.Popen(
            command,
            cwd=measure_root,
            env={**os.environ, "FRAMEWORK_CACHE_DB": str(DB_PATH)},
            pass_fds=(send.fileno(),),
        )
    except BaseException:
        receive.close()
        send.close()
        raise
    send.close()

    reported_id: str | None = None
    recent: deque[str] = deque(maxlen=40)
    last_activity = time.monotonic()
    killed_by_watchdog = False

    def forward_event() -> bool:
        nonlocal reported_id
        try:
            data = receive.recv_bytes()
        except EOFError:
            return False
        # JSON, never recv(): the worker runs candidate-editable code, and this
        # trusted process must not unpickle bytes it produced.
        match json.loads(data):
            case {"event": "heartbeat"}:
                observer.measurement_heartbeat()
            case ["started", str(payload)]:
                recent.append(payload)
                reported_id = payload
                observer.experiment_started(payload)
            case ["task", str(task_id), str(failure_mode)]:
                recent.append(f"{task_id}: {failure_mode}")
                observer.task_finished(task_id, failure_mode)
            case ["log", str(payload)]:
                recent.append(payload)
                observer.log(payload)
            case event:
                raise ValueError(f"invalid measurement event: {event!r}")
        return True

    exitcode: int
    try:
        # With worker heartbeats, silence means a dead or blocked worker, not a long phase.
        while process.poll() is None:
            if receive.poll(_WATCHDOG_POLL_SEC):
                if not forward_event():
                    break
                last_activity = time.monotonic()
            elif time.monotonic() - last_activity > WATCHDOG_INACTIVITY_SEC:
                killed_by_watchdog = True
                process.kill()
                process.wait()
                break
        while receive.poll():
            if not forward_event():
                break
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        raise
    finally:
        receive.close()
        exitcode = _terminate_experiment_process(process)

    if killed_by_watchdog or exitcode != 0:
        message = (
            f"measurement process killed by watchdog after "
            f"{WATCHDOG_INACTIVITY_SEC:.0f}s with no output (wedged):\n"
            if killed_by_watchdog
            else f"measurement process failed (exitcode={exitcode}):\n"
        ) + "\n".join(recent)
        raise MeasurementError(
            message,
            result=_load_crashed_result(
                tracker=tracker,
                observer=observer,
                experiment_id=reported_id,
                reason=message,
            ),
        )
    if reported_id is None:
        raise MeasurementError(
            "measurement process did not report an experiment id:\n"
            + "\n".join(recent),
            result=None,
        )
    result = tracker.load_experiment(reported_id)
    if result.finished_at is None:
        message = f"experiment {reported_id} did not complete: still running"
        raise MeasurementError(
            message,
            result=_load_crashed_result(
                tracker=tracker,
                observer=observer,
                experiment_id=reported_id,
                reason=message,
            ),
        )
    if result.crash_reason is not None:
        raise MeasurementError(
            f"experiment {reported_id} did not complete: crashed: {result.crash_reason}",
            result=result,
        )
    # Validate the requested panel trusted-side; omissions could hide regressions.
    if set(result.tasks) != set(task_ids):
        raise RuntimeError(
            f"experiment {reported_id} ran tasks {sorted(result.tasks)} "
            f"but {sorted(task_ids)} were requested"
        )
    return result
