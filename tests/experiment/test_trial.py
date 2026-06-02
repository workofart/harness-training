"""Tests for src/experiment/trial.py."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.experiment.trial import run_task
from src.harness.contracts import RawState

from conftest import _StubLlm, _StubEnv, _tool_call, _completion


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


def test_run_task_releases_slot_before_closing_env():
    # The concurrency slot must be handed back *before* teardown so the next
    # trial's container startup overlaps this trial's `compose down`.
    events: list[str] = []

    class _OrderEnv(_StubEnv):
        async def close(self) -> None:
            events.append("close")
            await super().close()

    llm = _StubLlm([_completion(_tool_call("verify"))])
    env = _OrderEnv()

    asyncio.run(
        run_task(
            task_name="t",
            llm=llm,
            env=env,
            max_steps=5,
            slot_release=lambda: events.append("release"),
        )
    )

    assert events == ["release", "close"]
    assert env.closed is True


def test_run_task_releases_slot_before_close_even_when_act_raises():
    events: list[str] = []

    class _OrderEnv(_StubEnv):
        async def close(self) -> None:
            events.append("close")
            await super().close()

    llm = _StubLlm([])  # empty -> act() raises on first completion
    env = _OrderEnv()

    result = asyncio.run(
        run_task(
            task_name="t",
            llm=llm,
            env=env,
            max_steps=5,
            slot_release=lambda: events.append("release"),
        )
    )

    assert result.error == "pop from empty list"
    assert events == ["release", "close"]


def test_run_task_closes_resources_even_when_act_raises():
    llm = _StubLlm([])
    env = _StubEnv()

    result = asyncio.run(run_task(task_name="t", llm=llm, env=env, max_steps=5))

    assert result.solved is False
    assert result.error == "pop from empty list"
    assert llm.closed is True
    assert env.closed is True


def test_run_task_marks_crash_with_exception_type_for_blank_error():
    # A bare exception (e.g. httpx.ReadError("")) has an empty str(); the trial
    # must still record a non-blank error (its type) and failure_mode "crash".
    class _BareErrorLlm(_StubLlm):
        async def complete(self, *, messages, tools=None, reasoning_effort=None):
            raise RuntimeError()

    llm = _BareErrorLlm([])
    env = _StubEnv()

    result = asyncio.run(run_task(task_name="t", llm=llm, env=env, max_steps=5))

    assert result.solved is False
    assert result.error == "RuntimeError"
    assert result.metrics.failure_mode == "crash"


def test_run_task_propagates_credentials_expired_instead_of_recording_crash():
    # A dead-credentials error must escape run_task's `except Exception`
    # containment (it inherits BaseException) so it bubbles up to halt the loop
    # rather than being recorded as a crashed trial the gate would discard. The
    # finally-block cleanup (slot release + resource close) must still run.
    from src.adapters.chatgpt_codex import ChatGptCodexCredentialsExpiredError

    class _DeadCredentialsLlm(_StubLlm):
        async def complete(self, *, messages, tools=None, reasoning_effort=None):
            raise ChatGptCodexCredentialsExpiredError("refresh token rejected")

    llm = _DeadCredentialsLlm([])
    env = _StubEnv()

    with pytest.raises(ChatGptCodexCredentialsExpiredError):
        asyncio.run(run_task(task_name="t", llm=llm, env=env, max_steps=5))

    assert llm.closed is True
    assert env.closed is True


def test_run_task_classifies_no_valid_action_distinct_from_crash():
    # When the model never emits a parseable tool call (e.g. empty/refused
    # completions), the trial is labeled an agent failure (no_valid_action), not
    # an infra crash. error stays set so it remains excluded from the gate's
    # valid trials -- an empty/refused response is not a fair capability signal.
    llm = _StubLlm([_completion(content="no tool call")] * 3)
    env = _StubEnv()

    result = asyncio.run(run_task(task_name="t", llm=llm, env=env, max_steps=5))

    assert result.metrics.failure_mode == "no_valid_action"
    assert result.error == "failed to parse a valid action call"
    assert result.solved is False


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
    assert result.error is None
    assert result.steps_used == 2
    assert result.metrics.failure_mode == "hit_timeout"
    assert result.started_at is not None
    assert result.finished_at is not None
    assert result.trial_dir == str(trial_dir)
    assert result.trace_path == str(trial_dir / "agent" / "steps.jsonl")
    assert result.metrics_path == str(trial_dir / "agent" / "metrics.json")
    assert Path(result.trace_path).exists()
    assert Path(result.metrics_path).exists()
    metrics = json.loads(Path(result.metrics_path).read_text())
    assert metrics["steps_total"] == result.steps_used


def test_run_task_env_setup_timeout_is_crash_distinct_from_hit_timeout(tmp_path):
    # A slow/hung reset (docker start + bootstrap) must fail fast against the
    # separate env_setup_timeout_sec as a crash -- not a hit_timeout -- and must
    # not have consumed any agent steps.
    class _SlowResetEnv(_StubEnv):
        async def reset(self) -> RawState:
            await asyncio.sleep(1)
            return self._reset_state

    trial_dir = tmp_path / "trial"
    llm = _StubLlm([_completion(_tool_call("verify"))])
    env = _SlowResetEnv(trial_dir=str(trial_dir))

    result = asyncio.run(
        run_task(
            task_name="t",
            llm=llm,
            env=env,
            max_steps=5,
            task_timeout_sec=10.0,
            env_setup_timeout_sec=0.05,
        )
    )

    assert result.metrics.failure_mode == "crash"
    assert result.error == "environment reset/bootstrap timed out after 0.05 seconds"
    assert result.steps_used == 0


def test_run_task_agent_timeout_independent_of_env_setup_budget(tmp_path):
    # With a generous setup budget and an instant reset, exhausting only the
    # agent budget still registers as hit_timeout -- proving the two budgets are
    # decoupled and a fast reset does not borrow from / lend to the agent loop.
    class _SlowVerifyEnv(_StubEnv):
        async def verify(self) -> RawState:
            self.verify_calls += 1
            await asyncio.sleep(1)
            return RawState(done=True, passed=True, reward=1.0)

    trial_dir = tmp_path / "trial"
    llm = _StubLlm([_completion(_tool_call("verify"))])
    env = _SlowVerifyEnv(trial_dir=str(trial_dir))

    result = asyncio.run(
        run_task(
            task_name="t",
            llm=llm,
            env=env,
            max_steps=5,
            task_timeout_sec=0.05,
            env_setup_timeout_sec=10.0,
        )
    )

    assert result.error is None
    assert result.metrics.failure_mode == "hit_timeout"


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

    assert result.error is None
    assert result.metrics.failure_mode == "hit_timeout"
    assert not verifier_stdout_path.exists()
    assert result.verifier_stdout_path is None
