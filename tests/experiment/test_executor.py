"""Tests for src/experiment/executor.py (one trial -> a TrialResult).

Covers the verify-ceiling env wrapper (#9), terminal failure classification,
the two independent timeouts (#8), and slot-release-before-teardown (#3).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.contracts import RawState
from src.experiment.executor import (
    VERIFY_TIMEOUT_NOTICE,
    _BudgetStampingEnv,
    _VerifyCeilingEnv,
    run_trial,
)
from src.trace import Recorder

from conftest import _StubLlm, _StubEnv, _completion, _tool_call


def _run_one(*, llm, env, **kwargs):
    return asyncio.run(
        run_trial(
            task_id="task-a",
            run_id="r1",
            llm=llm,
            env=env,
            max_steps=kwargs.pop("max_steps", 5),
            **kwargs,
        )
    )


# --- verify-ceiling env wrapper (#9) ----------------------------------------


def test_verify_ceiling_caps_a_hung_grader() -> None:
    # A graded verify exceeding the ceiling is stopped: the wrapper returns a
    # terminal non-passing state and records a trial-level `verify_timeout` fire.
    class _HangingVerifyEnv(_StubEnv):
        async def verify(self) -> RawState:
            self.verify_calls += 1
            await asyncio.sleep(30)  # never returns within the ceiling
            return RawState(passed=True, reward=1.0)

    inner = _HangingVerifyEnv()
    recorder = Recorder.create()
    wrapped = _VerifyCeilingEnv(inner, recorder, timeout_sec=0.05)

    # wait_for turns a non-firing ceiling into a clear failure, not a hang.
    raw = asyncio.run(asyncio.wait_for(wrapped.verify(), timeout=2))

    assert inner.verify_calls == 1  # the verifier was invoked, then stopped
    assert raw.passed is False and raw.reward == 0.0
    assert VERIFY_TIMEOUT_NOTICE in (raw.stdout or "")
    assert recorder.metrics.rule_fires.get("verify_timeout") == 1


def test_verify_ceiling_passes_through_a_prompt_grader() -> None:
    inner = _StubEnv(verify_state=RawState(passed=True, reward=1.0))
    recorder = Recorder.create()
    wrapped = _VerifyCeilingEnv(inner, recorder)

    raw = asyncio.run(wrapped.verify())

    assert raw.passed is True and raw.reward == 1.0
    assert recorder.metrics.rule_fires == {}


def test_verify_ceiling_propagates_inner_verifier_timeout() -> None:
    # A TimeoutError from the verifier itself (infra failure) is NOT the wall
    # ceiling firing: it must propagate for crash classification, not be masked
    # as a scorable verified_rejected verdict.
    class _InnerTimeoutEnv(_StubEnv):
        async def verify(self) -> RawState:
            self.verify_calls += 1
            raise TimeoutError("inner verifier timeout")

    inner = _InnerTimeoutEnv()
    recorder = Recorder.create()
    wrapped = _VerifyCeilingEnv(inner, recorder)

    with pytest.raises(TimeoutError, match="inner verifier timeout"):
        asyncio.run(wrapped.verify())
    assert recorder.metrics.rule_fires == {}  # the ceiling did not fire


# --- terminal classification ------------------------------------------------


def test_run_trial_solved_happy_path(tmp_path: Path) -> None:
    llm = _StubLlm([_completion(_tool_call("verify"))])
    env = _StubEnv(
        trial_dir=str(tmp_path / "trial"),
        verify_state=RawState(passed=True, reward=1.0),
    )

    result = _run_one(llm=llm, env=env)

    assert result.run_id == "r1"
    assert result.solved is True
    assert result.failure_mode == "solved"
    assert result.verifier_passed is True
    assert result.error is None
    assert result.metrics_path is not None and Path(result.metrics_path).exists()
    assert env.closed and llm.closed


def test_run_trial_classifies_verified_rejected(tmp_path: Path) -> None:
    llm = _StubLlm([_completion(_tool_call("verify"))])
    env = _StubEnv(
        trial_dir=str(tmp_path / "trial"),
        verify_state=RawState(passed=False, reward=0.0),
    )

    result = _run_one(llm=llm, env=env)

    assert result.solved is False
    assert result.failure_mode == "verified_rejected"
    assert result.verifier_passed is False
    assert result.error is None  # a real verdict is a valid (scorable) trial


def test_verify_timeout_within_run_trial_is_verified_rejected(tmp_path: Path) -> None:
    # End-to-end #9: a hung grader is cut by the ceiling to a terminal
    # non-passing verdict (`verified_rejected`, NOT a new failure_mode), and the
    # `verify_timeout` fire is recorded at trial level in metrics.json.
    class _HangingVerifyEnv(_StubEnv):
        async def verify(self) -> RawState:
            self.verify_calls += 1
            await asyncio.sleep(30)
            return RawState(passed=True, reward=1.0)

    llm = _StubLlm([_completion(_tool_call("verify"))])
    env = _HangingVerifyEnv(trial_dir=str(tmp_path / "trial"))

    result = _run_one(llm=llm, env=env, verify_timeout_sec=0.05)

    assert result.failure_mode == "verified_rejected"
    assert result.solved is False and result.verifier_passed is False
    assert result.error is None  # the ceiling produces a real (scorable) verdict
    assert result.metrics_path is not None
    metrics = json.loads(Path(result.metrics_path).read_text())
    assert metrics["rule_fires"].get("verify_timeout") == 1


def test_inner_verifier_timeout_is_a_crash(tmp_path: Path) -> None:
    # End-to-end: an infra TimeoutError from verify propagates to a `crash`
    # (error set => excluded from scoring), NOT a scorable verified_rejected.
    class _InnerTimeoutEnv(_StubEnv):
        async def verify(self) -> RawState:
            self.verify_calls += 1
            raise TimeoutError("inner verifier timeout")

    llm = _StubLlm([_completion(_tool_call("verify"))])
    env = _InnerTimeoutEnv(trial_dir=str(tmp_path / "trial"))

    result = _run_one(llm=llm, env=env)

    assert result.failure_mode == "crash"
    assert result.error is not None
    assert result.solved is False


def test_run_trial_no_valid_action(tmp_path: Path) -> None:
    # A model that never emits a parseable action is an agent failure: classified
    # `no_valid_action` with `error` set (excluded from the gate's valid trials).
    llm = _StubLlm([_completion(content="no tool call") for _ in range(6)])
    env = _StubEnv(trial_dir=str(tmp_path / "trial"))

    result = _run_one(llm=llm, env=env)

    assert result.failure_mode == "no_valid_action"
    assert result.solved is False
    assert result.error is not None


# --- the two independent timeouts (#8) --------------------------------------


def test_env_setup_timeout_is_a_crash(tmp_path: Path) -> None:
    class _HangingResetEnv(_StubEnv):
        async def reset(self) -> RawState:
            await asyncio.sleep(30)
            return await super().reset()

    llm = _StubLlm([_completion(_tool_call("verify"))])
    env = _HangingResetEnv(trial_dir=str(tmp_path / "trial"))

    result = _run_one(llm=llm, env=env, env_setup_timeout_sec=0.05)

    # A hung bootstrap fails fast as a crash (error set => excluded from scoring),
    # NOT a task timeout.
    assert result.failure_mode == "crash"
    assert result.error is not None and "reset/bootstrap timed out" in result.error
    assert result.solved is False


def test_agent_timeout_is_hit_timeout(tmp_path: Path) -> None:
    class _HangingLlm(_StubLlm):
        async def complete(self, *, messages, tools=None, reasoning_effort=None):
            await asyncio.sleep(30)  # the agent loop never returns in time
            return await super().complete(
                messages=messages, tools=tools, reasoning_effort=reasoning_effort
            )

    llm = _HangingLlm([_completion(_tool_call("verify"))])
    env = _StubEnv(trial_dir=str(tmp_path / "trial"))

    result = _run_one(llm=llm, env=env, task_timeout_sec=0.05)

    # The agent ran out of time: a valid unsolved sample (`error is None`),
    # distinct from a setup crash.
    assert result.failure_mode == "hit_timeout"
    assert result.error is None
    assert result.solved is False


def test_agent_work_timeout_is_hit_timeout(tmp_path: Path) -> None:
    # A non-verify WORK step (here an `exec`/run) that outruns the task deadline
    # with NO verify in flight is still hard-cut to `hit_timeout`. The task
    # timeout bounds the agent's work; only the terminal verify is exempt.
    class _HangingExecEnv(_StubEnv):
        async def exec(self, **kwargs):
            await asyncio.sleep(30)  # the work step never returns in time
            return await super().exec(**kwargs)

    llm = _StubLlm([_completion(_tool_call("run", command="loop"))])
    env = _HangingExecEnv(trial_dir=str(tmp_path / "trial"))

    result = _run_one(llm=llm, env=env, task_timeout_sec=0.05)

    assert result.failure_mode == "hit_timeout"
    assert result.error is None
    assert result.solved is False


def test_late_verify_runs_to_completion_not_hit_timeout(tmp_path: Path) -> None:
    # Bug repro: the agent calls verify with little task budget left. The verify
    # duration (0.3s) exceeds the remaining task budget (~0.05s) but is well
    # under the verify ceiling (5s). The terminal verify must run to completion
    # and yield a REAL verdict -- it is bounded by `verify_timeout_sec` ALONE,
    # not by however much `task_timeout_sec` remains. On the buggy code the outer
    # task timeout cancels the grade mid-flight -> `hit_timeout`/`passed=None`.
    class _SlowVerifyEnv(_StubEnv):
        async def verify(self) -> RawState:
            self.verify_calls += 1
            await asyncio.sleep(0.3)  # outlasts the task deadline, under ceiling
            return self._verify_state

    llm = _StubLlm([_completion(_tool_call("verify"))])
    env = _SlowVerifyEnv(
        trial_dir=str(tmp_path / "trial"),
        verify_state=RawState(passed=True, reward=1.0),
    )

    result = _run_one(llm=llm, env=env, task_timeout_sec=0.05, verify_timeout_sec=5.0)

    assert env.verify_calls == 1
    assert result.failure_mode == "solved"
    assert result.solved is True
    assert result.verifier_passed is True
    assert result.error is None


def test_late_verify_rejected_runs_to_completion(tmp_path: Path) -> None:
    # Same late-verify scoping, but the verifier returns a non-passing verdict:
    # the trial must classify `verified_rejected` (a real, scorable verdict),
    # NOT `hit_timeout`.
    class _SlowVerifyEnv(_StubEnv):
        async def verify(self) -> RawState:
            self.verify_calls += 1
            await asyncio.sleep(0.3)
            return self._verify_state

    llm = _StubLlm([_completion(_tool_call("verify"))])
    env = _SlowVerifyEnv(
        trial_dir=str(tmp_path / "trial"),
        verify_state=RawState(passed=False, reward=0.0),
    )

    result = _run_one(llm=llm, env=env, task_timeout_sec=0.05, verify_timeout_sec=5.0)

    assert result.failure_mode == "verified_rejected"
    assert result.solved is False
    assert result.verifier_passed is False
    assert result.error is None


def test_verify_ceiling_still_caps_a_late_hung_grader(tmp_path: Path) -> None:
    # The verify exemption is bounded by `verify_timeout_sec` alone: a grader
    # that hangs past its ceiling is still cut to a terminal non-passing verdict
    # with a `verify_timeout` fire, even when the task deadline already passed.
    class _HangingVerifyEnv(_StubEnv):
        async def verify(self) -> RawState:
            self.verify_calls += 1
            await asyncio.sleep(30)
            return RawState(passed=True, reward=1.0)

    llm = _StubLlm([_completion(_tool_call("verify"))])
    env = _HangingVerifyEnv(trial_dir=str(tmp_path / "trial"))

    result = _run_one(llm=llm, env=env, task_timeout_sec=0.05, verify_timeout_sec=0.1)

    assert result.failure_mode == "verified_rejected"
    assert result.solved is False and result.verifier_passed is False
    assert result.error is None
    metrics = json.loads(Path(result.metrics_path).read_text())
    assert metrics["rule_fires"].get("verify_timeout") == 1


# --- budget stamping (PR1: time_remaining_sec) -------------------------------


class _FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_budget_stamping_env_first_stamp_is_full_budget() -> None:
    clock = _FakeClock()
    wrapped = _BudgetStampingEnv(_StubEnv(), budget_sec=60.0, clock=clock)
    wrapped.start_clock()
    state = asyncio.run(wrapped.reset())
    assert state.time_remaining_sec == 60.0


def test_budget_stamping_env_unbounded_leaves_states_unstamped() -> None:
    # No budget configured (single-trial use, tests): time_remaining_sec stays
    # None, and stamping needs no clock anchor.
    wrapped = _BudgetStampingEnv(_StubEnv(), budget_sec=None, clock=_FakeClock())
    state = asyncio.run(wrapped.exec(command="x"))
    assert state.time_remaining_sec is None


def test_budget_countdown_excludes_bootstrap_and_decreases(tmp_path: Path) -> None:
    # End-to-end through run_trial with an injected clock: a slow bootstrap
    # (500s before the agent phase) consumes none of the 100s budget; the
    # stamps then strictly decrease as exec time accrues.
    clock = _FakeClock()

    class _SlowBootstrapEnv(_StubEnv):
        async def reset(self) -> RawState:
            clock.now += 500.0  # slow docker start + bootstrap
            return await super().reset()

        async def exec(self, **kwargs):
            clock.now += 10.0
            return await super().exec(**kwargs)

    llm = _StubLlm(
        [
            _completion(_tool_call("run", command="a")),
            _completion(_tool_call("run", command="b")),
            _completion(_tool_call("verify")),
        ]
    )
    env = _SlowBootstrapEnv(
        trial_dir=str(tmp_path / "trial"),
        verify_state=RawState(passed=True, reward=1.0),
    )

    result = _run_one(llm=llm, env=env, task_timeout_sec=100.0, clock=clock)

    assert result.solved is True
    events = [
        json.loads(line) for line in Path(result.trace_path).read_text().splitlines()
    ]
    stamps = [
        event["fields"]["time_remaining_sec"]
        for event in events
        if event["event"] == "env_step_completed"
    ]
    # Bootstrap's 500s never counted: the first exec stamp reflects only its
    # own 10s. The verify stamp repeats the elapsed total (verify is instant).
    assert stamps == [90.0, 80.0, 80.0]


# --- cleanup ordering (#3) --------------------------------------------------


def test_slot_release_runs_before_env_teardown(tmp_path: Path) -> None:
    order: list[str] = []

    class _OrderEnv(_StubEnv):
        async def close(self) -> None:
            order.append("close")
            await super().close()

    llm = _StubLlm([_completion(_tool_call("verify"))])
    env = _OrderEnv(
        trial_dir=str(tmp_path / "trial"),
        verify_state=RawState(passed=True, reward=1.0),
    )

    _run_one(
        llm=llm,
        env=env,
        slot_release=lambda: order.append("slot_release"),
    )

    # The slot frees before docker teardown so the next trial's setup overlaps.
    assert order == ["slot_release", "close"]
