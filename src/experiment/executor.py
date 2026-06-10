"""One trial, end to end -> a `TrialResult`.

`run_trial` is the only place a `TrialResult` is built. It owns:
- the two independent timeouts -- env setup (docker start + bootstrap) vs the
  agent loop -- so a hung bootstrap fails fast as a `crash` without consuming the
  agent's step budget or being misread as a task timeout (#8);
- the verify-ceiling: a ceiling-enforcing `HarnessEnv` wrapper injected around
  the agent loop, so `core.py`'s `VerifyAction` stays a bare `env.verify()` and
  Harbor stays trace-free; a hung grader is cut to a terminal non-passing state
  plus a trial-level `verify_timeout` fire (#9);
- terminal failure classification into the `failure_mode` buckets;
- resource cleanup with the concurrency slot released *before* docker teardown,
  so the next trial's `compose up` overlaps this trial's `compose down` (#3).

Concurrency, scheduling, aggregation, and gating are the orchestrator's job.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import src.trace as trace_module
from src.contracts import EnvExecWorkload, FailureMode, HarnessEnv, RawState
from src.experiment.record import TrialResult
from src.harness.core import NoValidActionError, TaskLoopState, run_task_loop
from src.llm.base import BaseLlm

# Wall ceiling for a single graded verify (run in a separate build+test env). A
# looping/deadlocked deliverable otherwise hangs the verifier until the whole-task
# timeout (observed: 11-28 min hangs on tasks that normally verify in seconds); the
# ceiling fails such a grader fast and terminally. Set above the longest verify seen
# to complete, so only a hang is cut; keyed on wall time alone, not the task. Lives
# here (off `harness/core`) so grading infra stays off the candidate-editable
# surface (plan.md §13).
VERIFY_TIMEOUT_SEC: float = 900.0
VERIFY_TIMEOUT_NOTICE = (
    "Grading did not complete within the time limit and was stopped. A correct "
    "solution is graded quickly; a solution that loops or blocks during grading "
    "is treated as failing."
)


class _VerifyCeilingEnv:
    """Wraps a `HarnessEnv` to bound `verify()` by `VERIFY_TIMEOUT_SEC`.

    On expiry it records a trial-level `verify_timeout` fire and returns the
    terminal non-passing `RawState` -- the unsolved verdict the task timeout
    would reach anyway, sooner. Only the inner ceiling is caught; an outer
    task-timeout cancellation passes through as `CancelledError` for upstream
    classification. Every other call delegates to the inner env, so Harbor stays
    trace-free and `core.py`'s `VerifyAction` is a bare `env.verify()`.
    """

    def __init__(self, inner: HarnessEnv, recorder: trace_module.Recorder) -> None:
        self._inner = inner
        self._recorder = recorder

    @property
    def trial_dir(self) -> str | None:
        return self._inner.trial_dir

    @property
    def verifier_stdout_path(self) -> str | None:
        return self._inner.verifier_stdout_path

    async def reset(self) -> RawState:
        return await self._inner.reset()

    async def exec(
        self,
        *,
        command: str,
        cwd: str | None = None,
        timeout_sec: int | None = None,
        workload: EnvExecWorkload = "heavy",
    ) -> RawState:
        return await self._inner.exec(
            command=command, cwd=cwd, timeout_sec=timeout_sec, workload=workload
        )

    async def verify(self) -> RawState:
        ceiling = None
        try:
            async with asyncio.timeout(VERIFY_TIMEOUT_SEC) as ceiling:
                return await self._inner.verify()
        except TimeoutError:
            # Only the ceiling's own expiry is a graded-verify timeout. A
            # `TimeoutError` raised by the verifier itself (infra failure) leaves
            # `ceiling.expired()` False -- propagate it so `run_trial` classifies
            # a `crash`, not a scorable `verified_rejected`.
            if ceiling is None or not ceiling.expired():
                raise
            self._recorder.rule_fired("verify_timeout")
            return RawState(reward=0.0, passed=False, stdout=VERIFY_TIMEOUT_NOTICE)

    async def close(self) -> None:
        await self._inner.close()


def _existing_artifact_path(path: str | None) -> str | None:
    if path is None or not Path(path).exists():
        return None
    return path


def _completed_failure_mode(
    *,
    solved: bool,
    final_passed: bool | None,
    steps_used: int,
    max_steps: int,
) -> FailureMode:
    # Classifies a trial that finished its loop without an infrastructure
    # failure. Infra failures (`error is not None`) are classified at the catch
    # site, not here.
    if solved:
        return "solved"
    if final_passed is False:
        return "verified_rejected"
    if steps_used >= max_steps:
        return "hit_step_cap"
    return "never_verified"


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


async def _close_resources(
    *,
    llm: BaseLlm,
    env: HarnessEnv,
    recorder: trace_module.Recorder,
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


async def run_trial(
    *,
    task_id: str,
    run_id: str,
    llm: BaseLlm,
    env: HarnessEnv,
    max_steps: int,
    max_output_retries: int = 2,
    task_timeout_sec: float | None = None,
    env_setup_timeout_sec: float | None = None,
    trace_path: str | None = None,
    slot_release: Callable[[], None] | None = None,
) -> TrialResult:
    setup_timeout_ctx = None
    agent_timeout_ctx = None
    started_at = datetime.now(timezone.utc).isoformat()
    finished_at: str | None = None
    trial_dir = env.trial_dir
    verifier_stdout_path = env.verifier_stdout_path
    metrics_path: str | None = None
    recorder = trace_module.NOOP_RECORDER
    reset_completed = False

    # Trial outcome, finalized by exactly one of the try/except branches and
    # assembled into the single TrialResult after cleanup.
    solved = False
    verifier_passed: bool | None = None
    error: str | None = None
    failure_mode: FailureMode = "crash"
    state = TaskLoopState()

    try:
        # `asyncio.timeout(None)` never fires, preserving the unbounded behavior
        # when a budget is not configured (single-trial use, tests).
        async with asyncio.timeout(env_setup_timeout_sec) as setup_timeout_ctx:
            reset_state = await env.reset()
        reset_completed = True
        trial_dir = env.trial_dir
        verifier_stdout_path = env.verifier_stdout_path
        if trial_dir is None:
            raise RuntimeError("environment reset must expose trial_dir")
        if trace_path is None:
            trace_path, metrics_path = trace_module.task_artifact_paths(trial_dir)
        recorder = trace_module.Recorder.create(
            trace_path=trace_path,
            metrics_path=metrics_path,
        )
        recorder.task_started(
            task_name=task_id,
            instruction=reset_state.instruction,
            working_dir=reset_state.working_dir,
        )
        async with asyncio.timeout(task_timeout_sec) as agent_timeout_ctx:
            await run_task_loop(
                llm=llm,
                env=_VerifyCeilingEnv(env, recorder),
                reset_state=reset_state,
                max_steps=max_steps,
                max_output_retries=max_output_retries,
                recorder=recorder,
                state=state,
            )
        solved = state.solved
        verifier_passed = state.final_passed
        verifier_stdout_path = env.verifier_stdout_path
        failure_mode = _completed_failure_mode(
            solved=solved,
            final_passed=verifier_passed,
            steps_used=state.steps_used,
            max_steps=max_steps,
        )
        finished_at = datetime.now(timezone.utc).isoformat()
        recorder.task_finished(
            task_name=task_id,
            reward=state.reward,
            solved=solved,
            error=None,
            steps_used=state.steps_used,
            final_passed=verifier_passed,
        )
    except Exception as exc:
        # Timeout and crash exits recover the same partial state, then classify.
        # `asyncio.timeout` cancels `run_task_loop` mid-flight so it never returns
        # normally; the env's dirs are re-read for the artifact paths.
        solved = False
        verifier_passed = None
        trial_dir, verifier_stdout_path = _recover_artifact_paths(
            env, trial_dir=trial_dir, verifier_stdout_path=verifier_stdout_path
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
            recorder = trace_module.Recorder.create(
                trace_path=trace_path, metrics_path=metrics_path
            )
        # Classification stays explicit: the exits carry different result
        # semantics (error set-or-None, failure_mode, detail).
        if isinstance(exc, TimeoutError):
            # Each stage has its own timeout context; whichever expired tells us
            # which stage timed out. The agent context exists only once reset has
            # completed, so its expiry unambiguously means the agent loop ran out
            # of time (hit_timeout); a setup-context expiry means reset/bootstrap.
            agent_expired = (
                agent_timeout_ctx is not None and agent_timeout_ctx.expired()
            )
            setup_expired = (
                setup_timeout_ctx is not None and setup_timeout_ctx.expired()
            )
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
        elif isinstance(exc, NoValidActionError):
            # A model that never emits a parseable action is an agent failure,
            # not a broken environment. `error` stays set, so it is excluded from
            # the gate's valid trials (an empty/refused response is not a fair
            # measure of capability).
            error = str(exc) or type(exc).__name__
            failure_mode = "no_valid_action"
            detail = error
        else:
            error = str(exc) or type(exc).__name__
            failure_mode = "crash"
            detail = error
        finished_at = datetime.now(timezone.utc).isoformat()
        recorder.task_failed(exc=exc, detail=detail)
    finally:
        # The trial outcome is finalized by this point (both branches captured it
        # into locals). Free the concurrency slot *before* the docker teardown so
        # the next trial's `compose up` overlaps this trial's `compose down` --
        # teardown still runs here, in this same task.
        if slot_release is not None:
            slot_release()
        await _close_resources(llm=llm, env=env, recorder=recorder)

    recorder.set_trial_outcome(
        verifier_passed=verifier_passed, failure_mode=failure_mode
    )
    recorder.write_metrics()
    return TrialResult(
        run_id=run_id,
        solved=solved,
        failure_mode=failure_mode,
        verifier_passed=verifier_passed,
        error=error,
        trial_dir=trial_dir,
        trace_path=trace_path,
        metrics_path=metrics_path,
        verifier_stdout_path=_existing_artifact_path(verifier_stdout_path),
        started_at=started_at,
        finished_at=finished_at,
    )
