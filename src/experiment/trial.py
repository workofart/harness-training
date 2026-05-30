from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import src.trace as trace_module
from src.adapters.llm_base import BaseLlm
from src.harness.contracts import HarnessEnv, TaskResult
from src.harness.core import NoValidActionError, TaskLoopProgress, run_task_loop
from src.metrics import FailureMode


def _failure_mode(
    *,
    solved: bool,
    final_passed: bool | None,
    steps_used: int,
    max_steps: int,
) -> FailureMode | None:
    # Classifies a trial that completed its loop without an infrastructure
    # failure. Infra failures (`error is not None`) are labeled "crash" where
    # they are caught, not here.
    if solved:
        return "solved"
    if final_passed is False:
        return "verified_rejected"
    if steps_used >= max_steps:
        return "hit_step_cap"
    return "never_verified"


async def _close_resources(
    *,
    llm: BaseLlm,
    env: HarnessEnv,
    recorder: trace_module.HarnessRecorder,
) -> None:
    cleanup = asyncio.gather(llm.close(), env.close(), return_exceptions=True)
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None:
                current.uncancel()
    for component, result in zip(("llm", "env"), cleanup.result(), strict=True):
        if isinstance(result, BaseException):
            recorder.cleanup_failed(component=component, exc=result)


def _recorder_for_paths(
    *,
    trace_path: str | None,
    metrics_path: str | None,
) -> trace_module.HarnessRecorder:
    return trace_module.HarnessRecorder.create(
        trace_path=trace_path,
        metrics_path=metrics_path,
    )


def _existing_artifact_path(path: str | None) -> str | None:
    if path is None:
        return None
    if not Path(path).exists():
        return None
    return path


def _recorded_steps_used(
    recorder: trace_module.HarnessRecorder,
    *,
    fallback: int,
) -> int:
    return max(fallback, recorder.build_metrics().steps_total)


def _recover_artifact_paths(
    env: HarnessEnv,
    *,
    trial_dir: str | None,
    verifier_stdout_path: str | None,
) -> tuple[str | None, str | None]:
    recovered_trial_dir = env.trial_dir
    if recovered_trial_dir is not None:
        trial_dir = recovered_trial_dir

    recovered_verifier_stdout_path = env.verifier_stdout_path
    if recovered_verifier_stdout_path is not None:
        verifier_stdout_path = recovered_verifier_stdout_path

    return trial_dir, verifier_stdout_path


async def run_task(
    *,
    task_name: str,
    llm: BaseLlm,
    env: HarnessEnv,
    max_steps: int,
    max_output_retries: int = 2,
    task_timeout_sec: float | None = None,
    env_setup_timeout_sec: float | None = None,
    trace_path: str | None = None,
    slot_release: Callable[[], None] | None = None,
) -> TaskResult:
    setup_timeout_ctx = None
    agent_timeout_ctx = None
    started_at = datetime.now(timezone.utc).isoformat()
    finished_at: str | None = None
    trial_dir = env.trial_dir
    verifier_stdout_path = env.verifier_stdout_path
    metrics_path: str | None = None
    recorder = trace_module.NOOP_HARNESS_RECORDER
    reset_completed = False

    reward: float | None = 0.0
    solved = False
    error: str | None = None
    failure_mode: str | None = None
    steps_used = 0
    final_passed: bool | None = None
    progress = TaskLoopProgress()

    async def execute_trial() -> TaskResult:
        nonlocal error
        nonlocal final_passed
        nonlocal finished_at
        nonlocal metrics_path
        nonlocal recorder
        nonlocal reset_completed
        nonlocal reward
        nonlocal solved
        nonlocal steps_used
        nonlocal setup_timeout_ctx
        nonlocal agent_timeout_ctx
        nonlocal trial_dir
        nonlocal verifier_stdout_path

        # Environment setup (docker start + bootstrap) is budgeted separately
        # from the agent loop so a slow/hung bootstrap fails fast as a crash
        # without consuming the agent's step budget or being misreported as a
        # task timeout. asyncio.timeout(None) never fires, preserving the
        # unbounded behavior when a budget is not configured (single-trial use,
        # tests).
        async with asyncio.timeout(env_setup_timeout_sec) as setup_timeout_ctx:
            reset_state = await env.reset()
        reset_completed = True
        trial_dir = env.trial_dir
        verifier_stdout_path = env.verifier_stdout_path
        if trial_dir is None:
            raise RuntimeError("environment reset must expose trial_dir")
        if trace_path is None:
            trace_path_for_trial, metrics_path = trace_module.task_artifact_paths(
                trial_dir
            )
        else:
            trace_path_for_trial = trace_path
        recorder = _recorder_for_paths(
            trace_path=trace_path_for_trial,
            metrics_path=metrics_path,
        )
        recorder.task_started(
            task_name=task_name,
            instruction=reset_state.instruction,
            working_dir=reset_state.working_dir,
        )
        async with asyncio.timeout(task_timeout_sec) as agent_timeout_ctx:
            outcome = await run_task_loop(
                task_name=task_name,
                llm=llm,
                env=env,
                reset_state=reset_state,
                max_steps=max_steps,
                max_output_retries=max_output_retries,
                recorder=recorder,
                progress=progress,
            )
        reward = outcome.reward
        solved = outcome.solved
        steps_used = outcome.steps_used
        final_passed = outcome.final_passed
        verifier_stdout_path = env.verifier_stdout_path
        recorder.set_trial_outcome(
            verifier_passed=final_passed,
            failure_mode=_failure_mode(
                solved=solved,
                final_passed=final_passed,
                steps_used=steps_used,
                max_steps=max_steps,
            ),
        )
        final_metrics = recorder.build_metrics()
        recorder.write_metrics()
        finished_at = datetime.now(timezone.utc).isoformat()
        recorder.task_finished(
            task_name=task_name,
            reward=reward,
            solved=solved,
            error=error,
            steps_used=steps_used,
            final_passed=final_passed,
            forced_final_verify=False,
        )
        return TaskResult(
            task_name=task_name,
            reward=reward,
            solved=solved,
            error=error,
            steps_used=steps_used,
            trial_dir=trial_dir,
            trace_path=trace_path_for_trial,
            metrics_path=metrics_path,
            verifier_stdout_path=_existing_artifact_path(verifier_stdout_path),
            metrics=final_metrics,
            started_at=started_at,
            finished_at=finished_at,
        )

    try:
        return await execute_trial()
    except TimeoutError as exc:
        reward = progress.reward
        steps_used = progress.steps_used
        final_passed = progress.final_passed
        trial_dir, verifier_stdout_path = _recover_artifact_paths(
            env,
            trial_dir=trial_dir,
            verifier_stdout_path=verifier_stdout_path,
        )
        if trial_dir is None:
            raise RuntimeError("environment must expose trial_dir before reset failure")
        if trace_path is None:
            trace_path, metrics_path = trace_module.task_artifact_paths(trial_dir)
        if recorder.trace is None:
            recorder = _recorder_for_paths(
                trace_path=trace_path, metrics_path=metrics_path
            )
        # Each stage has its own timeout context; whichever expired tells us
        # which stage timed out. The agent context only exists once reset has
        # completed, so its expiry unambiguously means the agent loop ran out of
        # time (hit_timeout); a setup-context expiry means reset/bootstrap did.
        agent_expired = agent_timeout_ctx is not None and agent_timeout_ctx.expired()
        setup_expired = setup_timeout_ctx is not None and setup_timeout_ctx.expired()
        if agent_expired:
            error = None
            failure_mode = "hit_timeout"
            detail = f"task timed out after {task_timeout_sec} seconds"
        elif setup_expired:
            error = (
                "environment reset/bootstrap timed out after "
                f"{env_setup_timeout_sec} seconds"
            )
            failure_mode = "crash"
            detail = error
        else:
            error = str(exc) or type(exc).__name__
            failure_mode = "crash"
            detail = error
        finished_at = datetime.now(timezone.utc).isoformat()
        recorder.task_failed(exc=exc, detail=detail)
        steps_used = _recorded_steps_used(recorder, fallback=steps_used)
    except Exception as exc:
        reward = progress.reward
        steps_used = progress.steps_used
        final_passed = progress.final_passed
        trial_dir, verifier_stdout_path = _recover_artifact_paths(
            env,
            trial_dir=trial_dir,
            verifier_stdout_path=verifier_stdout_path,
        )
        if trial_dir is None:
            if reset_completed:
                raise RuntimeError("environment reset must expose trial_dir") from exc
            raise RuntimeError(
                "environment must expose trial_dir before reset failure"
            ) from exc
        if trace_path is None:
            trace_path, metrics_path = trace_module.task_artifact_paths(trial_dir)
        if recorder.trace is None:
            recorder = _recorder_for_paths(
                trace_path=trace_path, metrics_path=metrics_path
            )
        error = str(exc) or type(exc).__name__
        # A model that never emits a parseable action is an agent failure, not a
        # broken environment: label it distinctly. `error` stays set, so it is
        # still excluded from the gate's valid trials (an empty/refused response
        # is not a fair measure of capability) -- just not lumped under `crash`.
        failure_mode = (
            "no_valid_action" if isinstance(exc, NoValidActionError) else "crash"
        )
        finished_at = datetime.now(timezone.utc).isoformat()
        recorder.task_failed(exc=exc, detail=error)
        steps_used = _recorded_steps_used(recorder, fallback=steps_used)
    finally:
        # The trial's result is already finalized at this point (happy path
        # returned it; error paths captured state into the nonlocals the bottom
        # return packages). Free the concurrency slot *before* the docker
        # teardown so the next trial's `compose up` overlaps this trial's
        # `compose down` — teardown still runs here, in this same task.
        if slot_release is not None:
            slot_release()
        await _close_resources(llm=llm, env=env, recorder=recorder)

    recorder.set_trial_outcome(
        verifier_passed=None,
        failure_mode=failure_mode,
    )
    final_metrics = recorder.build_metrics()
    recorder.write_metrics()
    return TaskResult(
        task_name=task_name,
        reward=reward,
        solved=False,
        error=error,
        steps_used=steps_used,
        trial_dir=trial_dir,
        trace_path=trace_path,
        metrics_path=metrics_path,
        verifier_stdout_path=_existing_artifact_path(verifier_stdout_path),
        metrics=final_metrics,
        started_at=started_at,
        finished_at=finished_at,
    )
