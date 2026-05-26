"""Tests for src/experiment/trial.py."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from src.adapters.llm_base import BaseLlm, LlmCompletion, LlmToolCall
from src.experiment.trial import run_task
from src.harness.contracts import RawState


class _StubLlm(BaseLlm):
    def __init__(self, completions: list[LlmCompletion]) -> None:
        self._completions = list(completions)
        self.calls: list[list[dict[str, Any]]] = []
        self.closed = False

    @property
    def max_context_length(self) -> int:
        return 1000

    async def complete(self, *, messages, tools=None, reasoning_effort=None):
        del tools, reasoning_effort
        self.calls.append([dict(m) for m in messages])
        return self._completions.pop(0)

    def get_token_count(self, messages, *, tools=None) -> int:
        del messages, tools
        return 1

    async def close(self) -> None:
        self.closed = True


class _StubEnv:
    def __init__(
        self,
        *,
        reset_state: RawState | None = None,
        exec_states: list[RawState] | None = None,
        verify_state: RawState | None = None,
        trial_dir: str = "/tmp/trial",
        verifier_stdout_path: str | None = None,
    ) -> None:
        self._reset_state = reset_state or RawState(
            instruction="do the thing", working_dir="/work"
        )
        self._exec_states = list(exec_states or [])
        self._verify_state = verify_state or RawState(
            done=True, passed=True, reward=1.0
        )
        self.trial_dir: str | None = trial_dir
        self.verifier_stdout_path: str | None = verifier_stdout_path
        self.exec_calls: list[dict[str, Any]] = []
        self.verify_calls = 0
        self.closed = False

    async def reset(self) -> RawState:
        return self._reset_state

    async def exec(self, *, command, cwd=None, timeout_sec=None):
        self.exec_calls.append(
            {"command": command, "cwd": cwd, "timeout_sec": timeout_sec}
        )
        if self._exec_states:
            return self._exec_states.pop(0)
        return RawState(return_code=0)

    async def verify(self) -> RawState:
        self.verify_calls += 1
        return self._verify_state

    async def close(self) -> None:
        self.closed = True


def _tool_call(name: str, **args: Any) -> LlmToolCall:
    return LlmToolCall(name=name, arguments=json.dumps(args))


def _completion(*calls: LlmToolCall, content: str | None = None) -> LlmCompletion:
    return LlmCompletion(tool_calls=tuple(calls), content=content)


def test_run_task_populates_trial_dir_and_verifier_stdout_path(tmp_path):
    trial_dir = tmp_path / "myrun"
    verifier_stdout = trial_dir / "stdout.txt"
    verifier_stdout.parent.mkdir(parents=True, exist_ok=True)
    verifier_stdout.write_text("ok\n")
    llm = _StubLlm([_completion(_tool_call("verify"))])
    env = _StubEnv(trial_dir=str(trial_dir), verifier_stdout_path=str(verifier_stdout))

    result = asyncio.run(run_task(task_name="t", llm=llm, env=env, max_steps=5))

    assert result.trial_dir == str(trial_dir)
    assert result.verifier_stdout_path == str(verifier_stdout)


def test_run_task_populates_lifecycle_and_canonical_artifacts(tmp_path):
    trial_dir = tmp_path / "trial"
    llm = _StubLlm([_completion(_tool_call("verify"))])
    env = _StubEnv(
        trial_dir=str(trial_dir),
        verifier_stdout_path=str(trial_dir / "verifier" / "test-stdout.txt"),
    )

    result = asyncio.run(run_task(task_name="t", llm=llm, env=env, max_steps=5))

    assert result.started_at is not None
    assert result.finished_at is not None
    assert result.trial_dir == str(trial_dir)
    assert result.trace_path == str(trial_dir / "agent" / "steps.jsonl")
    assert result.metrics_path == str(trial_dir / "agent" / "metrics.json")
    assert Path(result.trace_path).exists()
    assert Path(result.metrics_path).exists()
    metrics = json.loads(Path(result.metrics_path).read_text())
    assert metrics["steps_total"] == 1
    assert metrics["verify_count"] == 1
    assert metrics["verifier_passed"] is True


def test_run_task_closes_llm_and_env_in_finally():
    llm = _StubLlm([_completion(_tool_call("verify"))])
    env = _StubEnv()

    asyncio.run(run_task(task_name="t", llm=llm, env=env, max_steps=5))

    assert llm.closed is True
    assert env.closed is True


def test_run_task_closes_resources_even_when_act_raises():
    llm = _StubLlm([])
    env = _StubEnv()

    result = asyncio.run(run_task(task_name="t", llm=llm, env=env, max_steps=5))

    assert result.solved is False
    assert result.error == "pop from empty list"
    assert llm.closed is True
    assert env.closed is True


def test_run_task_writes_jsonl_when_trace_path_provided(tmp_path):
    trace = tmp_path / "trace.jsonl"
    llm = _StubLlm([_completion(_tool_call("verify"))])
    env = _StubEnv()

    result = asyncio.run(
        run_task(
            task_name="t",
            llm=llm,
            env=env,
            max_steps=5,
            trace_path=str(trace),
        )
    )

    assert result.trace_path == str(trace)
    events = [json.loads(line)["event"] for line in trace.read_text().splitlines()]
    assert "task_started" in events
    assert "action_chosen" in events
    assert "env_step_completed" in events
    assert "task_finished" in events


def test_run_task_skips_trace_when_trace_path_is_none():
    llm = _StubLlm([_completion(_tool_call("verify"))])
    env = _StubEnv()

    result = asyncio.run(
        run_task(task_name="t", llm=llm, env=env, max_steps=5, trace_path=None)
    )

    assert result.solved is True


def test_run_task_timeout_preserves_artifact_paths_and_attempted_steps(tmp_path):
    class _SlowVerifyEnv(_StubEnv):
        async def verify(self) -> RawState:
            self.verify_calls += 1
            await asyncio.sleep(1)
            return RawState(done=True, passed=True, reward=1.0)

    trial_dir = tmp_path / "trial"
    llm = _StubLlm(
        [
            _completion(_tool_call("run", command="true")),
            _completion(_tool_call("verify")),
        ]
    )
    env = _SlowVerifyEnv(trial_dir=str(trial_dir))

    result = asyncio.run(
        run_task(
            task_name="t",
            llm=llm,
            env=env,
            max_steps=5,
            task_timeout_sec=0.05,
        )
    )

    assert result.solved is False
    assert result.error == "task timed out after 0.05 seconds"
    assert result.steps_used == 2
    assert result.started_at is not None
    assert result.finished_at is not None
    assert result.trial_dir == str(trial_dir)
    assert result.trace_path == str(trial_dir / "agent" / "steps.jsonl")
    assert result.metrics_path == str(trial_dir / "agent" / "metrics.json")
    assert Path(result.trace_path).exists()
    assert Path(result.metrics_path).exists()
    metrics = json.loads(Path(result.metrics_path).read_text())
    assert metrics["steps_total"] == result.steps_used


def test_run_task_timeout_omits_missing_verifier_stdout_path(tmp_path):
    class _SlowVerifyEnv(_StubEnv):
        async def verify(self) -> RawState:
            self.verify_calls += 1
            await asyncio.sleep(1)
            return RawState(done=True, passed=True, reward=1.0)

    trial_dir = tmp_path / "trial"
    verifier_stdout_path = trial_dir / "verifier" / "test-stdout.txt"
    llm = _StubLlm([_completion(_tool_call("verify"))])
    env = _SlowVerifyEnv(
        trial_dir=str(trial_dir),
        verifier_stdout_path=str(verifier_stdout_path),
    )

    result = asyncio.run(
        run_task(
            task_name="t",
            llm=llm,
            env=env,
            max_steps=5,
            task_timeout_sec=0.05,
        )
    )

    assert result.error == "task timed out after 0.05 seconds"
    assert not verifier_stdout_path.exists()
    assert result.verifier_stdout_path is None
