"""One measured rollout: the frozen episode loop and its outcome classifier.

``run_task_loop`` is the framework-owned episode loop -- budget/stall/verify
timeouts and telemetry live here, structurally out of reach of the editable agent.
``RolloutRunner`` wires the config-resolved policy to one env + backend, runs the
loop, and classifies the outcome into a ``RolloutResult``.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from src.config import RunConfig
from src.env.base import (
    FrameworkEnvError,
    RawEnvOutput,
    RunAction,
    StepExecutionError,
    StepResult,
    TaskEnv,
    UnscorableInfraError,
    VERIFY_BACKSTOP_SLACK_SEC,
    VerifyAction,
    scrub_step_result,
)
from src.llm.backend import (
    CompletionBackend,
    CompletionInfraError,
    FrameworkError,
    backend_class,
)
from src.rollout.certification import (
    DETERMINISM_CHAIN_RELPATH,
    ChainStep,
    write_chain,
)
from src.rollout.execution import (
    ExecutionDriftError,
    StepExecutor,
)
from src.rollout.records import (
    FailureMode,
    FailureOrigin,
    RolloutResult,
    error_text,
)
from src.policy.base import (
    NoValidActionError,
    Policy,
    RepeatedLengthCutoffError,
    SUBMIT_ACTION_NAME,
    PolicyEventName,
)
from src.rollout.telemetry import (
    InstrumentedLlm,
    RolloutTelemetry,
)

STEPS_USED_KEY = "steps_used"

# Backstop outside the rollout budget; the trainer process is the final watchdog.
TRIAL_CLOSE_TIMEOUT_SEC = 180.0

# Slack beyond worst-case act(): all parse attempts may exhaust their LLM budgets.
STALL_TIMEOUT_SLACK_SEC = 120.0


class _PolicyBoundaryError(Exception):
    pass


class _EnvBoundaryError(Exception):
    pass


@contextmanager
def _policy_boundary():
    """Convert only editable-policy failures to a typed rollout failure."""
    try:
        yield
    except CompletionInfraError as exc:
        raise _EnvBoundaryError from (exc.__cause__ or exc)
    except FrameworkError as exc:
        raise exc.__cause__ or exc
    except (_TaskTimeoutError, _StallTimeoutError):
        raise
    except Exception as exc:
        raise _PolicyBoundaryError from exc


@contextmanager
def _env_boundary():
    """Convert only external env/backend failures to a typed rollout failure."""
    try:
        yield
    except (
        ExecutionDriftError,
        FrameworkEnvError,
        UnscorableInfraError,
        _EnvSetupTimeoutError,
    ):
        raise
    except Exception as exc:
        raise _EnvBoundaryError from exc


@dataclass(frozen=True, slots=True)
class LoopOutcome:
    """Why the episode ended, plus the data classification needs.

    Loop endings carry a step count; failure endings leave it None,
    matching the discarded partial progress."""

    end: Literal[
        "submitted",
        "budget",
        "step_cap",
        "gave_up",
        "setup_timeout",
        "task_timeout",
        "stall_timeout",
        "unscorable_infra",
        "crash",
    ]
    final: StepResult | None = None
    steps_taken: int | None = None
    origin: FailureOrigin | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.end in ("submitted", "budget") and self.final is None:
            raise ValueError(
                f"LoopOutcome end={self.end!r} requires a final transition"
            )


def _boundary_crash(exc: Exception) -> LoopOutcome:
    origin = "policy" if isinstance(exc, _PolicyBoundaryError) else "env"
    return LoopOutcome(
        end="crash", origin=origin, error=error_text(exc.__cause__ or exc)
    )


# Mirrors the policy default repair budget; defensive assumption, not coupling.
_STALL_ATTEMPT_BUDGET = 3


def _stall_timeout_sec(run_config: RunConfig) -> float:
    """Wall ceiling on time since the last executed action, enforced around act().

    Example: a truncation loop -- every completion streams healthily to
    max_tokens but parses to no action. Transport timeouts (transport.py)
    never fire: bytes keep flowing within each attempt. The agent deadline
    fires only after the whole task allowance is burned. This wall, refreshed
    only by executed actions, kills the rollout after roughly one doomed turn:
    max_tokens=6000, retries=2 -> 3 attempts x 360s healthy-attempt bound
    (complete_duration_bound_sec, provider-owned) + 120s slack = 1200s."""
    cfg = run_config.llm_provider_config
    per_attempt = backend_class(cfg.provider).complete_duration_bound_sec(
        cfg.max_tokens
    )
    return _STALL_ATTEMPT_BUDGET * per_attempt + STALL_TIMEOUT_SLACK_SEC


async def run_task_loop(
    *,
    policy: Policy,
    build_env_action: Callable[[Any], RunAction],
    max_steps: int,
    env: TaskEnv,
    executor: StepExecutor,
    telemetry: RolloutTelemetry,
    agent_timeout_sec: float,
    stall_timeout_sec: float,
    action_chain: list[ChainStep],
) -> LoopOutcome:
    """Run one episode to a total LoopOutcome.

    Every classifiable ending returns; only ExecutionDriftError (sampler
    infra, retried live-only) and framework defects escape."""
    try:
        with _env_boundary():
            timeout = asyncio.timeout(env.setup_timeout_sec)
            try:
                async with timeout:
                    initial_env_output = await env.reset()
                    await executor.provision()
            except TimeoutError:
                if not timeout.expired():
                    raise
                raise _EnvSetupTimeoutError from None

        with _policy_boundary():
            policy.reset(initial_env_output)
        now = asyncio.get_running_loop().time()
        agent_deadline = now + agent_timeout_sec
        stall_deadline = now + stall_timeout_sec
        final: StepResult | None = None
        step_index = 0

        while step_index < max_steps:
            with _policy_boundary():
                timeout = asyncio.timeout_at(min(agent_deadline, stall_deadline))
                try:
                    async with timeout:
                        # Materialized inside the boundary: consuming the
                        # harness's returned iterable runs harness code.
                        actions = tuple(await policy.act())
                except NoValidActionError:
                    actions = ()
                except RepeatedLengthCutoffError as exc:
                    return LoopOutcome(
                        end="gave_up",
                        final=final,
                        steps_taken=step_index,
                        error=str(exc),
                    )
                except TimeoutError:
                    if not timeout.expired():
                        raise
                    if asyncio.get_running_loop().time() >= agent_deadline:
                        raise _TaskTimeoutError() from None
                    raise _StallTimeoutError(
                        f"no action executed for {stall_timeout_sec:.0f}s; "
                        "aborting stalled rollout"
                    ) from None

            previous_step_index = step_index
            # Executed first, observed second: observe() runs harness code that
            # build_env_action reads, so a batch must render against the state
            # the policy had when it chose the batch.
            transitions: list[tuple[Any, StepResult]] = []
            batch_terminal = False
            submit_requested = False
            for action in actions:
                if step_index >= max_steps:
                    break
                with _policy_boundary():
                    submit_requested = action.name == SUBMIT_ACTION_NAME
                if submit_requested:
                    env_action = VerifyAction()
                    prepare = asyncio.timeout_at(agent_deadline)
                    try:
                        async with prepare:
                            served = await executor.prepare_submit()
                    except TimeoutError as exc:
                        if not prepare.expired():
                            raise _EnvBoundaryError from exc
                        raise _TaskTimeoutError() from None
                    timeout = asyncio.timeout(env.verify_timeout_sec)
                    verify_backstop = asyncio.timeout(
                        env.verify_timeout_sec + VERIFY_BACKSTOP_SLACK_SEC
                    )
                else:
                    # Rendered frozen-side so replay keys include the harness's trainable rendered command.
                    with _policy_boundary():
                        env_action = build_env_action(action)
                        # Prevent the harness's renderer from invoking the grader outside the frozen submit route.
                        if type(env_action) is not RunAction:
                            raise ValueError(
                                f"action route for {action.name!r} produced "
                                f"{type(env_action).__name__}; expected RunAction"
                            )
                    # Renderer-selected timeouts must not extend the operator's agent deadline.
                    timeout = asyncio.timeout_at(agent_deadline)
                    verify_backstop = None
                timed_out = False
                try:
                    async with verify_backstop or nullcontext(), timeout:
                        if submit_requested:
                            if served is None:
                                step_result = await executor.step_verify()
                            else:
                                step_result = served
                        else:
                            step_result = await executor.step_run(env_action)
                except TimeoutError as exc:
                    if not timeout.expired() and (
                        verify_backstop is None or not verify_backstop.expired()
                    ):
                        raise _EnvBoundaryError from exc
                    step_result = StepResult(
                        raw_env_output=RawEnvOutput(stderr="timed out"),
                        reward=0.0,
                        terminated=False,
                        truncated=True,
                    )
                    timed_out = True
                step_result = scrub_step_result(
                    step_result,
                    command=env_action.command
                    if isinstance(env_action, RunAction)
                    else None,
                )
                action_chain.append(
                    ChainStep(
                        env_action=env_action,
                        step_result=step_result,
                        timed_out=timed_out,
                    )
                )
                step_index += 1
                transitions.append((action, step_result))
                if step_result.terminated or step_result.truncated:
                    batch_terminal = True
                    break

            if not actions:
                step_index += 1
                telemetry.on_no_valid_action_step(step_index=step_index)

            for call_index, (action, transition) in enumerate(transitions):
                terminal = transition.terminated or transition.truncated
                telemetry.on_step_completed(
                    step_index=previous_step_index + call_index + 1,
                    call_index=call_index,
                    step_result=transition,
                    terminal=terminal,
                )
                with _policy_boundary():
                    policy.observe(action, transition)
                final = transition
            if actions:
                stall_deadline = asyncio.get_running_loop().time() + stall_timeout_sec
            if batch_terminal:
                return LoopOutcome(
                    end="submitted" if submit_requested else "budget",
                    final=final,
                    steps_taken=step_index,
                )
        return LoopOutcome(
            end="step_cap",
            final=final,
            steps_taken=step_index,
        )
    except _EnvSetupTimeoutError:
        return LoopOutcome(end="setup_timeout")
    except _StallTimeoutError as exc:
        return LoopOutcome(end="stall_timeout", error=str(exc))
    except _TaskTimeoutError:
        return LoopOutcome(end="task_timeout")
    except UnscorableInfraError as exc:
        return LoopOutcome(end="unscorable_infra", error=error_text(exc))
    except FrameworkEnvError as exc:
        # The harness can induce grader corruption; score it as an env crash, not unscorable infra.
        return LoopOutcome(end="crash", origin="env", error=error_text(exc))
    except StepExecutionError as exc:
        return LoopOutcome(
            end="crash", origin="env", error=error_text(exc.__cause__ or exc)
        )
    except (_PolicyBoundaryError, _EnvBoundaryError) as exc:
        return _boundary_crash(exc)


@dataclass(frozen=True, slots=True)
class _Outcome:
    """One rollout's classified outcome. Fields default to None so each return
    names only what its failure mode populates."""

    failure_mode: FailureMode
    origin: FailureOrigin | None = None
    reward: float | None = None
    error: str | None = None


@dataclass(slots=True)
class RolloutRunner:
    task_id: str
    llm: CompletionBackend
    env: TaskEnv
    executor: StepExecutor
    run_config: RunConfig
    telemetry: RolloutTelemetry
    agent_timeout_sec: float

    async def run(self) -> RolloutResult:
        started_at = datetime.now(UTC)
        action_chain: list[ChainStep] = []

        try:
            # Keep candidate imports inside classification/cleanup; instrument before policy receives the backend.
            try:
                with _policy_boundary():
                    # Resolved here, not at module scope: the import binds to the
                    # measured worktree (worker cwd), and a broken candidate module
                    # classifies as a policy crash.
                    target = importlib.import_module(
                        self.run_config.training_target.module
                    )

                    def policy_event(event: PolicyEventName, /, **fields: Any) -> None:
                        try:
                            self.telemetry.on_policy_event(event, **fields)
                        except Exception as exc:
                            raise FrameworkError from exc

                    # Pass only policy-owned scalars; keep task panel and execution mode frozen-side.
                    llm_cfg = self.run_config.llm_provider_config
                    policy = target.build_policy(
                        InstrumentedLlm(self.llm, self.telemetry),
                        policy_event,
                        max_context_length=llm_cfg.max_context_length,
                        max_completion_tokens=llm_cfg.max_tokens,
                        thinking_toggleable=llm_cfg.enable_thinking is not None,
                        tokenizer_name=llm_cfg.tokenizer_name,
                        model_name=llm_cfg.model_name,
                    )
                    if not isinstance(policy, Policy):
                        raise TypeError(
                            "build_policy returned "
                            f"{type(policy).__name__}, not a Policy"
                        )
                    build_env_action = target.build_env_action
            except (_PolicyBoundaryError, _EnvBoundaryError) as exc:
                loop_outcome = _boundary_crash(exc)
            else:
                loop_outcome = await run_task_loop(
                    policy=policy,
                    build_env_action=build_env_action,
                    max_steps=self.run_config.max_steps,
                    env=self.env,
                    executor=self.executor,
                    telemetry=self.telemetry,
                    agent_timeout_sec=self.agent_timeout_sec,
                    stall_timeout_sec=_stall_timeout_sec(self.run_config),
                    action_chain=action_chain,
                )
        finally:
            # Cleanup must also run for cancellation, drift, and framework defects.
            close_error = await self._close_resources()
            write_chain(
                self.telemetry.rollout_dir / DETERMINISM_CHAIN_RELPATH,
                action_chain,
            )

        outcome = self._classify_loop_result(loop_outcome)
        failure_mode: FailureMode = outcome.failure_mode
        failure_origin = outcome.origin
        error = outcome.error
        if close_error is not None:
            failure_mode = "crash"
            failure_origin = "env"
            error = close_error if error is None else f"{error}; {close_error}"

        final = loop_outcome.final
        episode_metrics: dict[str, int | float] = {}
        if loop_outcome.steps_taken is not None:
            episode_metrics[STEPS_USED_KEY] = loop_outcome.steps_taken
        if outcome.reward is not None:
            episode_metrics["reward"] = outcome.reward
        metrics = _merge_metrics(
            episode_metrics,
            self.telemetry.metrics(),
            {} if final is None else dict(final.metrics),
        )
        return RolloutResult(
            task_id=self.task_id,
            failure_mode=failure_mode,
            failure_origin=failure_origin,
            error=error,
            metrics=metrics,
            rollout_dir=str(self.telemetry.rollout_dir),
            trace_path=str(self.telemetry.trace_path),
            started_at=started_at,
            finished_at=datetime.now(UTC),
        )

    def _classify_loop_result(
        self,
        outcome: LoopOutcome,
    ) -> _Outcome:
        final = outcome.final
        if outcome.end == "submitted":
            if final.truncated:
                return _Outcome("verify_timeout", reward=final.reward)
            verdict = final.verdict
            if verdict is None:
                raise RuntimeError("submit produced no verifier verdict")
            if not verdict.completed:
                return _Outcome("crash", origin="env", error=verdict.error)
            return _Outcome(
                "solved" if verdict.passed else "verified_rejected",
                reward=final.reward,
            )
        if outcome.end == "budget":
            return _Outcome("hit_timeout", reward=final.reward)
        if outcome.end == "gave_up":
            return _Outcome("no_valid_action", error=outcome.error)
        if outcome.end == "step_cap":
            if final is None:
                return _Outcome("no_valid_action")
            return _Outcome("hit_step_cap", reward=final.reward)
        if outcome.end == "setup_timeout":
            return _Outcome("crash", origin="env")
        if outcome.end in ("task_timeout", "stall_timeout"):
            return _Outcome("hit_timeout", error=outcome.error)
        if outcome.end == "unscorable_infra":
            return _Outcome("unscorable_infra", error=outcome.error)
        return _Outcome("crash", origin=outcome.origin, error=outcome.error)

    async def _close_resources(self) -> str | None:
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    self.env.close(),
                    self.llm.close(),
                    return_exceptions=True,
                ),
                timeout=TRIAL_CLOSE_TIMEOUT_SEC,
            )
        except TimeoutError:
            # Bound close so one stuck rollout becomes a scorable crash, not a wedged experiment.
            return f"resource close timed out after {TRIAL_CLOSE_TIMEOUT_SEC:.0f}s"
        errors = [
            error_text(result) for result in results if isinstance(result, Exception)
        ]
        if not errors:
            return None
        return "resource close failed: " + "; ".join(errors)


class _EnvSetupTimeoutError(Exception):
    pass


class _TaskTimeoutError(Exception):
    pass


class _StallTimeoutError(Exception):
    """No action executed within the stall window.

    Timeout classification is keyed on dedicated framework exceptions or
    truncated StepResult data; bare TimeoutError is foreign and crashes.
    """


def _merge_metrics(*sources: dict[str, int | float]) -> dict[str, int | float]:
    merged: dict[str, int | float] = {}
    for source in sources:
        duplicates = merged.keys() & source.keys()
        if duplicates:
            raise ValueError(f"duplicate rollout metric keys: {sorted(duplicates)}")
        merged.update(source)
    return merged
