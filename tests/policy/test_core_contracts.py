"""Measurement-integrity tests for the rollout loop."""

from __future__ import annotations

import asyncio
from typing import Any, cast

from conftest import _StubEnv

from src.env.docker_shell import MAX_CAPTURED_STREAM_BYTES
from src.env.base import (
    RawEnvOutput,
    RunAction,
    TaskEnv,
    VerifyAction,
    VerifyOutcome,
    VerifyVerdict,
)
from src.env.base import StepResult
from src.policy import core
from src.policy.base import NoValidActionError
from src.policy.core import (
    COMMAND_TIMEOUT_SEC,
    Action,
    RunArgs,
    SubmitArgs,
    _RequestBuilder,
)
from src.rollout.episode import (
    LoopOutcome,
    run_task_loop,
)
from src.rollout.execution import LiveStepExecutor


def _env(env: _StubEnv) -> TaskEnv:
    return cast(TaskEnv, env)


class _TraceEvents:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def on_no_valid_action_step(self, **fields: Any) -> None:
        self.steps_taken = fields["step_index"]
        self.events.append(("no_valid_action_step", fields))

    def on_step_completed(self, **fields: Any) -> None:
        self.steps_taken = fields["step_index"]
        self.events.append(("step_completed", fields))

    @property
    def step_completed_events(self) -> list[dict[str, Any]]:
        return [fields for name, fields in self.events if name == "step_completed"]


class _FakeAgent:
    def __init__(self, batches: list[tuple[Action, ...]]) -> None:
        self._batches = list(batches)
        self.reset_env_outputs: list[RawEnvOutput] = []
        self.act_calls = 0
        self.observe_calls: list[tuple[Action, StepResult]] = []

    def reset(self, raw_env_output: RawEnvOutput) -> None:
        self.reset_env_outputs.append(raw_env_output)

    async def act(self) -> tuple[Action, ...]:
        self.act_calls += 1
        if not self._batches:
            raise AssertionError("agent acted after scripted batches were exhausted")
        return self._batches.pop(0)

    def observe(self, action: Action, step_result: StepResult) -> None:
        self.observe_calls.append((action, step_result))


class _SlowAgent:
    def reset(self, raw_env_output: RawEnvOutput) -> None:
        del raw_env_output

    async def act(self) -> tuple[Action, ...]:
        await asyncio.sleep(60)
        return ()

    def observe(self, action: Action, step_result: StepResult) -> None:
        del action, step_result


class _AlwaysInvalidAgent:
    def __init__(self) -> None:
        self.act_calls = 0

    def reset(self, raw_env_output: RawEnvOutput) -> None:
        del raw_env_output

    async def act(self) -> tuple[Action, ...]:
        self.act_calls += 1
        raise NoValidActionError("failed to parse a valid action call")

    def observe(self, action: Action, step_result: StepResult) -> None:
        del action, step_result


class _NoStepEnv:
    setup_timeout_sec = 600.0
    verify_timeout_sec = 1800.0

    async def reset(self) -> RawEnvOutput:
        return RawEnvOutput()

    async def provision(self) -> None:
        pass

    async def execute(self, action: RunAction) -> RawEnvOutput:
        raise AssertionError("env.execute must not run for a no-valid-action turn")

    async def verify(self) -> VerifyOutcome:
        raise AssertionError("env.verify must not run for a no-valid-action turn")

    async def close(self) -> None:
        pass


class _ProtocolProbeEnv:
    """Records attribute access so tests can pin protocol-only env use."""

    def __init__(self, verify_result: StepResult) -> None:
        self.accessed: set[str] = set()
        self._verify_result = verify_result
        self.setup_timeout_sec = 600.0
        self.verify_timeout_sec = 1800.0

    def __getattribute__(self, name: str) -> Any:
        if not name.startswith("__"):
            object.__getattribute__(self, "accessed").add(name)
        return object.__getattribute__(self, name)

    async def reset(self) -> RawEnvOutput:
        return RawEnvOutput(instruction="probe", working_dir="/work")

    async def provision(self) -> None:
        pass

    async def execute(self, action: RunAction) -> RawEnvOutput:
        del action
        return RawEnvOutput(exit_code=0)

    async def verify(self) -> VerifyOutcome:
        result = object.__getattribute__(self, "_verify_result")
        if result.verdict is None:
            raise AssertionError("probe verify result must include a verdict")
        return VerifyOutcome(
            verdict=result.verdict,
            output=result.raw_env_output,
            reward=result.reward,
            info=result.info,
        )

    async def close(self) -> None:
        pass


def _run_loop(
    policy,
    env: TaskEnv,
    *,
    max_steps: int = 5,
    agent_timeout_sec: float | None = 30.0,
    build_env_action=core.build_env_action,
) -> tuple[Any, _TraceEvents]:
    events = _TraceEvents()
    result = asyncio.run(
        run_task_loop(
            policy=policy,
            build_env_action=build_env_action,
            max_steps=max_steps,
            env=env,
            executor=LiveStepExecutor(env),
            telemetry=events,
            agent_timeout_sec=agent_timeout_sec,
            stall_timeout_sec=30.0,
            action_chain=[],
        )
    )
    return result, events


def test_capture_cap_dwarfs_observation_clip_budget() -> None:
    assert MAX_CAPTURED_STREAM_BYTES >= 10 * _RequestBuilder.RESULT_CHAR_LIMIT


def test_episode_scrubs_raw_env_output_before_the_policy_observes_it() -> None:
    seen_actions: list[RunAction | VerifyAction] = []

    class _Env:
        setup_timeout_sec = 600.0
        verify_timeout_sec = 900.0

        async def reset(self) -> RawEnvOutput:
            return RawEnvOutput()

        async def provision(self) -> None:
            pass

        async def execute(self, action: RunAction) -> RawEnvOutput:
            seen_actions.append(action)
            return RawEnvOutput(stdout="<Exists object at 0x7ffffd51dd30>")

        async def verify(self) -> VerifyOutcome:
            raise AssertionError("a run action must not reach the grader")

        async def close(self) -> None:
            pass

    agent = _FakeAgent([(Action(name="run", args=RunArgs(command="pytest")),)])
    result, _events = _run_loop(agent, cast(TaskEnv, _Env()), max_steps=1)

    # A non-terminal run action exhausts the step cap rather than ending the episode.
    assert result.end == "step_cap"
    [(action, step_result)] = agent.observe_calls
    assert action == Action(name="run", args=RunArgs(command="pytest"))
    assert seen_actions == [
        RunAction(command="pytest", timeout_sec=COMMAND_TIMEOUT_SEC)
    ]
    assert "0x" not in step_result.raw_env_output.stdout


def test_episode_rejects_non_submit_verify_renderer() -> None:
    def _verify_renderer(_action: Action) -> VerifyAction:
        return VerifyAction()

    class _Env:
        setup_timeout_sec = 600.0
        verify_timeout_sec = 900.0

        async def reset(self) -> RawEnvOutput:
            return RawEnvOutput()

        async def provision(self) -> None:
            pass

        async def execute(self, action: RunAction) -> RawEnvOutput:
            raise AssertionError("non-submit VerifyAction reached env")

        async def verify(self) -> VerifyOutcome:
            raise AssertionError("non-submit VerifyAction reached the grader")

        async def close(self) -> None:
            pass

    agent = _FakeAgent([(Action(name="run", args=RunArgs(command="pytest")),)])
    result, _events = _run_loop(
        agent,
        cast(TaskEnv, _Env()),
        max_steps=1,
        build_env_action=_verify_renderer,
    )

    # The rejection is a policy-boundary failure: a scorable policy crash,
    # not a framework defect that escapes the loop.
    assert result.end == "crash"
    assert result.origin == "policy"
    assert "action route for 'run' produced VerifyAction" in result.error


def test_run_task_loop_consumes_step_and_continues_after_no_valid_action() -> None:
    agent = _AlwaysInvalidAgent()
    result, events = _run_loop(agent, cast(TaskEnv, _NoStepEnv()), max_steps=3)

    assert result is not None
    assert result.end == "step_cap"
    assert agent.act_calls == 3
    assert [name for name, _ in events.events] == ["no_valid_action_step"] * 3
    assert [fields["step_index"] for _, fields in events.events] == [1, 2, 3]


def test_run_task_loop_uses_initial_output_and_step_protocol_only():
    probe = _ProtocolProbeEnv(
        verify_result=StepResult(
            raw_env_output=RawEnvOutput(stdout="verified\n"),
            reward=1.0,
            terminated=True,
            truncated=False,
            verdict=VerifyVerdict(completed=True, passed=True, error=None),
        ),
    )
    agent = _FakeAgent(
        [
            (Action(name="run", args=RunArgs(command="pwd")),),
            (Action(name="submit", args=SubmitArgs()),),
        ]
    )
    _run_loop(agent, cast(TaskEnv, probe))

    accessed = object.__getattribute__(probe, "accessed")
    assert "step_index" not in accessed
    assert "step" not in accessed
    assert "reset" in accessed
    assert "execute" in accessed
    assert "verify" in accessed
    assert "verify_timeout_sec" in accessed
    assert "close" not in accessed
    assert [action.name for action, _step_result in agent.observe_calls] == [
        "run",
        "submit",
    ]


def test_run_task_loop_emits_terminal_submit_step_with_reward():
    agent = _FakeAgent([(Action(name="submit", args=SubmitArgs()),)])
    initial_env_output = RawEnvOutput(instruction="do the thing", working_dir="/work")
    env = _StubEnv(reset_env_output=initial_env_output)
    result, events = _run_loop(agent, _env(env))
    assert env.verify_calls == 1
    assert result is not None
    assert result.end == "submitted"
    assert [name for name, _fields in events.events] == [
        "step_completed",
    ]
    [event] = events.step_completed_events
    assert event["step_index"] == 1
    assert event["call_index"] == 0
    assert event["step_result"].reward == 1.0
    assert event["step_result"].terminated is True
    assert event["terminal"] is True
    assert agent.reset_env_outputs == [initial_env_output]
    assert agent.observe_calls == [
        (Action(name="submit", args=SubmitArgs()), event["step_result"])
    ]


def test_run_task_loop_allows_env_owned_verify():
    agent = _FakeAgent([(Action(name="submit", args=SubmitArgs()),)])
    seen_actions: list[RunAction | VerifyAction] = []

    class _OwnedVerifyEnv:
        setup_timeout_sec = 600.0
        verify_timeout_sec = 1800.0

        async def reset(self) -> RawEnvOutput:
            return RawEnvOutput(instruction="probe", working_dir="/work")

        async def provision(self) -> None:
            pass

        async def execute(self, action: RunAction) -> RawEnvOutput:
            raise AssertionError(f"unexpected execute: {action}")

        async def verify(self) -> VerifyOutcome:
            seen_actions.append(VerifyAction())
            return VerifyOutcome(
                output=RawEnvOutput(stdout="verified\n"),
                reward=1.0,
                verdict=VerifyVerdict(completed=True, passed=True, error=None),
                info={},
            )

        async def close(self) -> None:
            pass

    env = _OwnedVerifyEnv()

    result, _events = _run_loop(agent, env)

    assert result is not None
    assert result.end == "submitted"
    assert seen_actions == [VerifyAction()]


def test_run_task_loop_unsolved_when_max_steps_reached_without_submit():
    agent = _FakeAgent(
        [(Action(name="run", args=RunArgs(command="pwd")),) for _ in range(3)]
    )
    env = _StubEnv()
    result, events = _run_loop(agent, _env(env), max_steps=3)
    assert env.verify_calls == 0
    assert result is not None
    assert result.end == "step_cap"
    assert [event["step_index"] for event in events.step_completed_events] == [
        1,
        2,
        3,
    ]
    assert [action.name for action, _step_result in agent.observe_calls] == [
        "run",
        "run",
        "run",
    ]
    assert [event["call_index"] for event in events.step_completed_events] == [
        0,
        0,
        0,
    ]
    assert [event["terminal"] for event in events.step_completed_events] == [
        False,
        False,
        False,
    ]
    assert agent.act_calls == 3
    assert len(agent.observe_calls) == 3


def test_run_task_loop_stops_batch_after_submit():
    agent = _FakeAgent(
        [
            (
                Action(name="submit", args=SubmitArgs()),
                Action(name="run", args=RunArgs(command="pwd")),
            )
        ]
    )
    env = _StubEnv()
    result, events = _run_loop(agent, _env(env))
    assert env.verify_calls == 1
    assert env.exec_calls == []
    assert result is not None
    assert result.end == "submitted"
    assert len(events.step_completed_events) == 1
    assert events.step_completed_events[0]["call_index"] == 0
    assert events.step_completed_events[0]["terminal"] is True
    assert agent.observe_calls[0][0].name == "submit"
    assert len(agent.observe_calls) == 1


def test_run_task_loop_executes_batch_without_intermediate_agent_turn():
    agent = _FakeAgent(
        [
            (
                Action(name="run", args=RunArgs(command="pwd")),
                Action(name="run", args=RunArgs(command="ls")),
            )
        ]
    )
    env = _StubEnv()
    result, events = _run_loop(agent, _env(env), max_steps=2)

    assert result is not None
    assert result.end == "step_cap"
    assert agent.act_calls == 1
    assert [call["command"] for call in env.exec_calls] == ["pwd", "ls"]
    assert [action.name for action, _step_result in agent.observe_calls] == [
        "run",
        "run",
    ]
    assert [event["call_index"] for event in events.step_completed_events] == [0, 1]
    assert len(agent.observe_calls) == 2


def test_run_task_loop_truncates_a_batch_that_overruns_the_step_cap():
    # A batch may be longer than the steps left; the cap is enforced per action,
    # so the tail never reaches the env and never reaches the policy.
    agent = _FakeAgent(
        [
            (
                Action(name="run", args=RunArgs(command="first")),
                Action(name="run", args=RunArgs(command="second")),
                Action(name="run", args=RunArgs(command="third")),
            )
        ]
    )
    env = _StubEnv()
    result, events = _run_loop(agent, _env(env), max_steps=2)

    assert result.end == "step_cap"
    assert result.steps_taken == 2
    assert [call["command"] for call in env.exec_calls] == ["first", "second"]
    assert len(agent.observe_calls) == 2
    assert [event["step_index"] for event in events.step_completed_events] == [1, 2]


def test_run_task_loop_stops_on_truncated_transition():
    agent = _FakeAgent(
        [
            (
                Action(name="run", args=RunArgs(command="sleep 10")),
                Action(name="run", args=RunArgs(command="pwd")),
            )
        ]
    )

    class _SlowEnv(_StubEnv):
        async def execute(self, action: RunAction) -> RawEnvOutput:
            self.step_calls.append(action)
            self.exec_calls.append(
                {
                    "command": action.command,
                    "cwd": action.cwd,
                    "timeout_sec": action.timeout_sec,
                }
            )
            await asyncio.sleep(60)
            return RawEnvOutput(stdout="finished too late\n")

    env = _SlowEnv()

    result, events = _run_loop(agent, _env(env), agent_timeout_sec=0.001)

    assert result is not None
    assert result.end == "budget"
    assert [call["command"] for call in env.exec_calls] == ["sleep 10"]
    assert len(events.step_completed_events) == 1
    assert events.step_completed_events[0]["call_index"] == 0
    assert events.step_completed_events[0]["terminal"] is True
    transition = events.step_completed_events[0]["step_result"]
    assert transition.truncated is True
    assert transition.raw_env_output.stderr == "timed out"
    assert agent.observe_calls == [
        (Action(name="run", args=RunArgs(command="sleep 10")), transition)
    ]


def test_run_task_loop_times_out_waiting_for_agent_action():
    async def run() -> LoopOutcome:
        return await run_task_loop(
            policy=_SlowAgent(),
            build_env_action=core.build_env_action,
            max_steps=5,
            env=(env := _env(_StubEnv())),
            executor=LiveStepExecutor(env),
            telemetry=_TraceEvents(),
            agent_timeout_sec=0.01,
            stall_timeout_sec=30.0,
            action_chain=[],
        )

    outcome = asyncio.run(run())
    assert outcome.end == "task_timeout"
