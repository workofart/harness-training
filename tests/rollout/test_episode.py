"""One measured rollout: RolloutRunner classification and the frozen episode loop."""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
import src.policy.core as core_module
from src.config import LlmProviderConfig
from src.env.base import (
    MODEL_PATCH_INFO_KEY,
    RawEnvOutput,
    RunAction,
    TaskEnv,
    UnscorableInfraError,
    VerifyVerdict,
)
from src.env.base import StepResult
from src.env.swebench_verify import VerifierCorruptError
from src.llm.backend import (
    Completion,
    CompletionBackend,
    CompletionInfraError,
    CompletionRequestError,
    FrameworkError,
    ProviderRejectedToolCallError,
    backend_class,
)
import src.rollout.episode as episode
from src.plugins.replay.step_cache import ReplayCache
from src.rollout.episode import LoopOutcome, RolloutRunner
from src.rollout.certification import scrubbed_hash
from src.rollout.execution import LiveStepExecutor
from src.rollout.telemetry import (
    LIVE_LLM_CALLS_KEY,
    MEDIAN_LIVE_LLM_LATENCY_SEC_KEY,
    SUM_LIVE_LLM_LATENCY_SEC_KEY,
)

from conftest import _completion, _tool_call
from _rollout_fixtures import (
    _FakeEnv,
    _FakeLlm,
    _MODEL_PATCH,
    _telemetry,
    _rollout_config,
)


def _run_rollout(
    tmp_path: Path,
    llm: CompletionBackend,
    env: TaskEnv,
    *,
    config=None,
    executor=None,
    telemetry=None,
    agent_timeout_sec: float = 1.0,
):
    """Run the real measured boundary with only invariant wiring defaulted."""
    config = _rollout_config() if config is None else config
    telemetry = _telemetry(tmp_path) if telemetry is None else telemetry
    executor = LiveStepExecutor(env) if executor is None else executor
    runner = RolloutRunner(
        "task-a", llm, env, executor, config, telemetry, agent_timeout_sec
    )
    return asyncio.run(runner.run())


def _assert_closed(env: _FakeEnv, llm: _FakeLlm) -> None:
    assert env.closed is True
    assert llm.closed is True


def _chain_rows(path: Path) -> list[dict[str, Any]]:
    chain = path / "infra/determinism_chain.jsonl"
    return [json.loads(line) for line in chain.read_text().splitlines()]


def _assert_empty_chain(path: Path) -> None:
    assert (path / "infra/determinism_chain.jsonl").read_text() == ""


def test_rollout_runner_signature_keeps_task_timeout_out_of_config() -> None:
    expected = "task_id llm env executor run_config telemetry agent_timeout_sec"
    assert set(inspect.signature(RolloutRunner).parameters) == set(expected.split())


def test_run_rollout_crashes_and_closes_when_build_policy_returns_non_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def build_policy(llm, events, **scalars):
        del llm, events, scalars
        return object()

    monkeypatch.setattr(core_module, "build_policy", build_policy)
    llm = _FakeLlm([])
    env = _FakeEnv()

    result = _run_rollout(tmp_path, llm, env)

    assert result.failure_mode == "crash"
    assert result.failure_origin == "policy"
    assert result.error == "TypeError: build_policy returned object, not a Policy"
    _assert_closed(env, llm)


def test_run_rollout_passes_policy_event_callback_not_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Policy receives the event callback, never trusted telemetry methods.
    captured: list[Any] = []
    real_build_policy = core_module.build_policy

    def build_policy(llm, events, **scalars):
        captured.append(events)
        return real_build_policy(llm, events, **scalars)

    monkeypatch.setattr(core_module, "build_policy", build_policy)
    llm = _FakeLlm([_completion(_tool_call("submit"))])
    env = _FakeEnv()
    telemetry = _telemetry(tmp_path)

    _run_rollout(tmp_path, llm, env, telemetry=telemetry)

    [events] = captured
    assert callable(events)
    assert not isinstance(events, episode.RolloutTelemetry)
    events("policy_rule", step_index=99, rule="probe", fired=False)
    rows = [json.loads(line) for line in telemetry.trace_path.read_text().splitlines()]
    assert {
        "event": "policy_rule",
        "step_index": 99,
        "rule": "probe",
        "fired": False,
    }.items() <= rows[-1].items()


def _captured_policy_scalars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config
) -> dict[str, Any]:
    captured: list[dict[str, Any]] = []
    real_build_policy = core_module.build_policy

    def build_policy(llm, events, **scalars):
        captured.append(scalars)
        return real_build_policy(llm, events, **scalars)

    monkeypatch.setattr(core_module, "build_policy", build_policy)
    llm = _FakeLlm([_completion(_tool_call("submit"))])
    _run_rollout(tmp_path, llm, _FakeEnv(), config=config)
    [scalars] = captured
    return scalars


def test_run_rollout_hands_policy_only_these_scalars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The harness receives only these scalars; value drift forks rendered requests and the LLM cache.
    scalars = _captured_policy_scalars(tmp_path, monkeypatch, _rollout_config())

    assert scalars == {
        "max_context_length": 16384,
        "max_completion_tokens": 8192,
        "thinking_toggleable": False,
        "tokenizer_name": None,
        "model_name": "gpt-test",
    }


def test_run_rollout_marks_explicit_thinking_channel_toggleable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An explicit thinking channel (even False) is toggleable.
    thinking = _rollout_config().model_copy(
        update={
            "llm_provider_config": LlmProviderConfig(
                model_name="gpt-test",
                base_url="http://127.0.0.1:18000/v1",
                api_key_env="OPENAI_API_KEY",
                max_context_length=16384,
                max_tokens=8192,
                enable_thinking=False,
            )
        }
    )
    scalars = _captured_policy_scalars(tmp_path, monkeypatch, thinking)
    assert scalars["thinking_toggleable"] is True


def test_run_rollout_records_solved_submit_and_closes_backends(tmp_path: Path) -> None:
    llm = _FakeLlm([_completion(_tool_call("submit"))])
    env = _FakeEnv(
        verify_result=StepResult(
            raw_env_output=RawEnvOutput(stdout="ok\n"),
            reward=1.0,
            terminated=True,
            truncated=False,
            info={MODEL_PATCH_INFO_KEY: _MODEL_PATCH},
            metrics={"fail_to_pass_passed": 3, "pass_to_pass_failed": 0},
            verdict=VerifyVerdict(completed=True, passed=True, error=None),
        )
    )
    telemetry = _telemetry(tmp_path)

    result = _run_rollout(tmp_path, llm, env, telemetry=telemetry)

    assert result.task_id == "task-a"
    assert result.failure_mode == "solved"
    assert result.failure_origin is None
    assert result.error is None
    live_latency = result.metrics[SUM_LIVE_LLM_LATENCY_SEC_KEY]
    assert live_latency >= 0.0
    assert result.metrics == {
        "reward": 1.0,
        "steps_used": 1,
        "first_attempt_valid": 1,
        "first_attempt_total": 1,
        LIVE_LLM_CALLS_KEY: 1,
        SUM_LIVE_LLM_LATENCY_SEC_KEY: live_latency,
        MEDIAN_LIVE_LLM_LATENCY_SEC_KEY: live_latency,
        "fail_to_pass_passed": 3,
        "pass_to_pass_failed": 0,
    }
    assert result.rollout_dir == str(tmp_path)
    assert result.trace_path == str(tmp_path / "agent" / "steps.jsonl")
    assert result.started_at is not None
    assert result.finished_at is not None
    assert env.reset_calls == 1
    assert env.verify_calls == 1
    _assert_closed(env, llm)
    assert _trace_events(telemetry.trace_path) == [
        "completion_received",
        "step_completed",
    ]


def test_run_rollout_writes_ordered_canonical_action_chain(tmp_path: Path) -> None:
    env = _FakeEnv()
    result = _run_rollout(
        tmp_path,
        _FakeLlm(
            [
                _completion(
                    _tool_call("run", command="pwd", cwd="/work", timeout_sec=3.0)
                ),
                _completion(_tool_call("submit")),
            ]
        ),
        env,
    )

    assert result.failure_mode == "solved"
    rows = _chain_rows(tmp_path)
    assert rows == [
        {
            "action": {
                "kind": "run",
                "command": "pwd",
                "cwd": "/work",
                "timeout_sec": 3.0,
            },
            "audit_hash": scrubbed_hash(
                StepResult(
                    raw_env_output=RawEnvOutput(exit_code=0, stdout="ran\n"),
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                )
            ),
            "timed_out": False,
        },
        {
            "action": {"kind": "verify"},
            "verdict": {"passed": True, "reward": 1.0},
            "timed_out": False,
        },
    ]


_SUBMIT_CASES = {
    "rejected": (
        StepResult(
            raw_env_output=RawEnvOutput(stdout="not solved\n"),
            reward=0.0,
            terminated=True,
            truncated=False,
            verdict=VerifyVerdict(completed=True, passed=False, error=None),
        ),
        ("verified_rejected", None, None, True),
    ),
    "incomplete": (
        StepResult(
            raw_env_output=RawEnvOutput(),
            reward=0.0,
            terminated=True,
            truncated=False,
            info={MODEL_PATCH_INFO_KEY: _MODEL_PATCH},
            verdict=VerifyVerdict(
                completed=False,
                passed=None,
                error="official SWE-bench evaluation did not complete",
            ),
        ),
        ("crash", "env", "official SWE-bench evaluation did not complete", False),
    ),
}


@pytest.mark.parametrize("case", _SUBMIT_CASES.values(), ids=_SUBMIT_CASES)
def test_run_rollout_classifies_submit_result(tmp_path: Path, case) -> None:
    verify_result, expected = case
    result = _run_rollout(
        tmp_path,
        _FakeLlm([_completion(_tool_call("submit"))]),
        _FakeEnv(verify_result=verify_result),
    )

    assert (
        result.failure_mode,
        result.failure_origin,
        result.error,
        "reward" in result.metrics,
    ) == expected
    if expected[-1]:
        assert result.metrics["reward"] == 0.0


def test_rollout_metric_merge_rejects_duplicate_publishers() -> None:
    with pytest.raises(ValueError, match="duplicate rollout metric keys: \\['x'\\]"):
        episode._merge_metrics({"x": 1}, {"x": 2})


def test_run_rollout_propagates_framework_telemetry_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_telemetry(self, **kwargs):
        del self, kwargs
        raise RuntimeError("telemetry persistence failed")

    monkeypatch.setattr(episode.RolloutTelemetry, "on_step_completed", fail_telemetry)

    with pytest.raises(RuntimeError, match="telemetry persistence failed"):
        _run_rollout(
            tmp_path, _FakeLlm([_completion(_tool_call("submit"))]), _FakeEnv()
        )


def test_run_rollout_rejects_submit_without_verdict(
    tmp_path: Path,
) -> None:
    env = _FakeEnv(
        verify_result=StepResult(
            raw_env_output=RawEnvOutput(stdout="forged\n"),
            reward=1.0,
            terminated=True,
            truncated=False,
            info={MODEL_PATCH_INFO_KEY: _MODEL_PATCH},
        ),
    )

    with pytest.raises(RuntimeError, match="submit produced no verifier verdict"):
        _run_rollout(tmp_path, _FakeLlm([_completion(_tool_call("submit"))]), env)


_LOOP_END_CASES = {
    "step-cap-after-run": (
        [_completion(_tool_call("run", command="pwd"))],
        {"max_steps": 1},
        ("hit_step_cap", None, {"reward": 0.0, "steps_used": 1}, ["pwd"], None),
    ),
    # Invalid turns consume steps but are output failure, not useful work.
    "only-invalid-turns": (
        [Completion(content="no tool call") for _ in range(6)],
        {"max_steps": 2},
        (
            "no_valid_action",
            None,
            {
                "steps_used": 2,
                "first_attempt_total": 2,
                "first_attempt_valid": 0,
                "parse_failures.MissingToolCall": 6,
            },
            [],
            (
                ["completion_received", "action_parse_failed"] * 3
                + ["no_valid_action_step"]
            )
            * 2,
        ),
    ),
    "repeated-length-cutoff": (
        [
            Completion(content="runaway reasoning", finish_reason="length"),
            _completion(_tool_call("run", command="pwd")),
            Completion(content="runaway again", finish_reason="length"),
            Completion(content="runaway still", finish_reason="length"),
        ],
        {"max_steps": 10, "max_completion_tokens": 8192},
        (
            "no_valid_action",
            "repeated length-truncated model outputs",
            {"steps_used": 1},
            ["pwd"],
            [
                "completion_received",
                "action_parse_failed",
                "completion_received",
                "step_completed",
                "completion_received",
                "action_parse_failed",
                "completion_received",
                "action_parse_failed",
            ],
        ),
    ),
    # Real work before the final invalid turn still exhausts the work budget.
    "work-then-invalid-turn": (
        [
            _completion(_tool_call("run", command="pwd")),
            *[Completion(content="no tool call") for _ in range(3)],
        ],
        {"max_steps": 2},
        ("hit_step_cap", None, {"reward": 0.0, "steps_used": 2}, ["pwd"], None),
    ),
}


@pytest.mark.parametrize("case", _LOOP_END_CASES.values(), ids=_LOOP_END_CASES)
def test_run_rollout_classifies_non_submit_loop_end(tmp_path: Path, case) -> None:
    completions, config_updates, expected = case
    failure_mode, error, expected_metrics, exec_calls, trace_events = expected
    telemetry = _telemetry(tmp_path)
    llm = _FakeLlm(completions)
    env = _FakeEnv()

    result = _run_rollout(
        tmp_path,
        llm,
        env,
        config=_rollout_config(**config_updates),
        telemetry=telemetry,
    )

    assert result.failure_mode == failure_mode
    assert result.error == error
    for name, value in expected_metrics.items():
        assert result.metrics[name] == value
    assert env.exec_calls == exec_calls
    assert env.verify_calls == 0
    _assert_closed(env, llm)
    if trace_events is not None:
        assert _trace_events(telemetry.trace_path) == trace_events


def test_provider_rejected_first_attempt_counts_as_invalid(tmp_path: Path) -> None:
    # A provider-rejected tool call never yields a completion, but it is still
    # the decision's attempt 0; the repaired retry must not score as a valid
    # first attempt nor be mislabeled attempt_index=0.
    class _RejectThenSubmitLlm(_FakeLlm):
        def __init__(self) -> None:
            super().__init__([_completion(_tool_call("submit"))])
            self._rejected = False

        async def _complete(self, request: Any) -> Completion:
            if not self._rejected:
                self._rejected = True
                raise ProviderRejectedToolCallError("invalid tool call arguments")
            return await super()._complete(request)

    telemetry = _telemetry(tmp_path)
    llm = _RejectThenSubmitLlm()
    env = _FakeEnv()

    result = _run_rollout(tmp_path, llm, env, telemetry=telemetry)

    assert result.failure_mode == "solved"
    assert result.metrics["first_attempt_total"] == 1
    assert result.metrics["first_attempt_valid"] == 0
    assert _trace_events(telemetry.trace_path) == [
        "completion_rejected",
        "action_parse_failed",
        "completion_received",
        "step_completed",
    ]
    [received] = [
        row
        for row in map(json.loads, telemetry.trace_path.read_text().splitlines())
        if row["event"] == "completion_received"
    ]
    assert (received["step_index"], received["attempt_index"]) == (1, 1)


def test_completion_step_labels_track_loop_steps_across_failed_turns(
    tmp_path: Path,
) -> None:
    # Failed turns never observe(), so labels must follow loop steps, not trajectory.
    telemetry = _telemetry(tmp_path)
    llm = _FakeLlm(
        [
            *[Completion(content="no call") for _ in range(6)],
            _completion(_tool_call("run", command="pwd")),
            _completion(_tool_call("submit")),
        ]
    )
    env = _FakeEnv()

    result = _run_rollout(
        tmp_path,
        llm,
        env,
        config=_rollout_config(max_steps=10),
        telemetry=telemetry,
    )

    assert result.failure_mode == "solved"
    labels = [
        (row["step_index"], row["attempt_index"])
        for row in map(json.loads, telemetry.trace_path.read_text().splitlines())
        if row["event"] == "completion_received"
    ]
    assert labels == [(step, attempt) for step in (1, 2) for attempt in range(3)] + [
        (3, 0),
        (4, 0),
    ]
    assert result.metrics["first_attempt_total"] == 4
    assert result.metrics["first_attempt_valid"] == 2


_TIMEOUT_CASES = {
    "task-budget-during-run": (
        [_completion(_tool_call("run", command="sleep 10"))],
        {"step_delay_sec": 0.05},
        {"agent_timeout_sec": 0.01},
        ("hit_timeout", None, ["sleep 10"], 0, True),
    ),
    "env-setup-timeout": (
        [_completion(_tool_call("submit"))],
        {"reset_delay_sec": 0.05, "setup_timeout_sec": 0.01},
        {},
        ("crash", "env", [], 0, None),
    ),
    # Live provisioning shares the setup budget, outside the agent clock.
    "provision-timeout": (
        [_completion(_tool_call("submit"))],
        {"provision_delay_sec": 0.05, "setup_timeout_sec": 0.01},
        {},
        ("crash", "env", [], 0, None),
    ),
    "verify-timeout": (
        [_completion(_tool_call("submit"))],
        {"verify_delay_sec": 0.05, "verify_timeout_sec": 0.01},
        {"agent_timeout_sec": 1.0},
        ("verify_timeout", None, [], 1, True),
    ),
    # Once submit starts, grading owns its timeout even if task time expires.
    "late-verify-uses-verify-budget": (
        [_completion(_tool_call("submit"))],
        {"verify_delay_sec": 0.05},
        {"agent_timeout_sec": 0.01},
        ("solved", None, [], 1, None),
    ),
}


@pytest.mark.parametrize("case", _TIMEOUT_CASES.values(), ids=_TIMEOUT_CASES)
def test_run_rollout_classifies_owned_timeouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case
) -> None:
    completions, env_kwargs, config_updates, expected = case
    failure_mode, failure_origin, exec_calls, verify_calls, chain_timed_out = expected
    monkeypatch.setattr(episode, "VERIFY_BACKSTOP_SLACK_SEC", 1.0)
    llm = _FakeLlm(completions)
    env = _FakeEnv(**env_kwargs)

    result = _run_rollout(tmp_path, llm, env, **config_updates)

    assert result.failure_mode == failure_mode
    assert result.failure_origin == failure_origin
    assert result.error is None
    assert env.reset_calls == 1
    assert env.exec_calls == exec_calls
    assert env.verify_calls == verify_calls
    _assert_closed(env, llm)
    if chain_timed_out is not None:
        [row] = _chain_rows(tmp_path)
        assert row["timed_out"] is chain_timed_out


def test_live_rollout_provisions_env_on_the_setup_budget(tmp_path: Path) -> None:
    env = _FakeEnv()

    result = _run_rollout(tmp_path, _FakeLlm([_completion(_tool_call("submit"))]), env)

    assert result.failure_mode == "solved"
    assert env.provision_calls == 1


def test_replayed_executor_never_provisions_when_no_step_goes_live(
    tmp_path: Path,
) -> None:
    """Cache-hit rollouts must stay infra-free; live transitions provision lazily."""
    env = _FakeEnv()

    class _ReplayedVerify(LiveStepExecutor):
        async def provision(self) -> None:
            pass

        async def prepare_submit(self) -> StepResult:
            return StepResult(
                raw_env_output=RawEnvOutput(stdout="verified\n"),
                reward=1.0,
                terminated=True,
                truncated=False,
                verdict=VerifyVerdict(completed=True, passed=True, error=None),
            )

    result = _run_rollout(
        tmp_path,
        _FakeLlm([_completion(_tool_call("submit"))]),
        env,
        executor=_ReplayedVerify(env),
    )

    assert result.failure_mode == "solved"
    assert env.provision_calls == 0


class _ForeignTimeoutEnv(_FakeEnv):
    async def execute(self, action: RunAction) -> RawEnvOutput:
        del action
        raise TimeoutError("foreign timeout")


class _UnscorableSetupEnv(_FakeEnv):
    async def execute(self, action: RunAction) -> RawEnvOutput:
        del action
        raise UnscorableInfraError("required setup command failed")


class _CorruptGraderEnv(_FakeEnv):
    async def verify(self):
        raise VerifierCorruptError("official report missing")


class _FailingBackend(_FakeLlm):
    async def _complete(self, request):
        del request
        try:
            raise ConnectionError("transport failed")
        except ConnectionError as exc:
            raise CompletionInfraError from exc


class _InvalidRequestBackend(_FakeLlm):
    async def _complete(self, request):
        del request
        raise CompletionRequestError("invalid completion request")


def _run_action_llm() -> _FakeLlm:
    return _FakeLlm([_completion(_tool_call("run", command="pwd"))])


def _failing_backend() -> _FailingBackend:
    return _FailingBackend([])


def _invalid_request_backend() -> _InvalidRequestBackend:
    return _InvalidRequestBackend([])


# Every pre-action failure must close both backends and leave an empty determinism chain.
_EXCEPTION_SOURCE_CASES = {
    # Foreign TimeoutError from env.execute is a crash, not a framework timeout
    # (wide task budget so the deadline cannot claim the failure).
    "foreign-execute-timeout": (
        _run_action_llm,
        _ForeignTimeoutEnv,
        30.0,
        ("crash", "env", "TimeoutError: foreign timeout"),
    ),
    # Backend transport failure (CompletionInfraError) surfaces as an env-plane
    # crash, reported from its underlying transport cause.
    "completion-backend-failure": (
        _failing_backend,
        _FakeEnv,
        30.0,
        ("crash", "env", "ConnectionError: transport failed"),
    ),
    # A bad completion request (CompletionRequestError) is the editable policy's
    # fault, so it is a policy-plane crash.
    "completion-request-failure": (
        _invalid_request_backend,
        _FakeEnv,
        30.0,
        ("crash", "policy", "CompletionRequestError: invalid completion request"),
    ),
    # UnscorableInfraError from required setup is unscorable (origin stays unset).
    "unscorable-infra-setup-failure": (
        _run_action_llm,
        _UnscorableSetupEnv,
        30.0,
        (
            "unscorable_infra",
            None,
            "UnscorableInfraError: required setup command failed",
        ),
    ),
    # Harness-inducible grader corruption is a scorable rollout crash, not amnesty or a whole-run failure.
    "corrupt-grader-report": (
        lambda: _FakeLlm([_completion(_tool_call("submit"))]),
        _CorruptGraderEnv,
        30.0,
        ("crash", "env", "VerifierCorruptError: official report missing"),
    ),
}


@pytest.mark.parametrize(
    "case", _EXCEPTION_SOURCE_CASES.values(), ids=_EXCEPTION_SOURCE_CASES
)
def test_run_rollout_classifies_exception_source(tmp_path: Path, case) -> None:
    make_llm, make_env, agent_timeout_sec, expected = case
    llm = make_llm()
    env = make_env()
    result = _run_rollout(
        tmp_path,
        llm,
        env,
        agent_timeout_sec=agent_timeout_sec,
    )

    assert (result.failure_mode, result.failure_origin, result.error) == expected
    _assert_closed(env, llm)
    _assert_empty_chain(tmp_path)


class _ScriptedPolicy:
    """Contract-shaped policy with per-test act; no core internals touched."""

    def __init__(self, act_fn) -> None:
        self._act = act_fn

    def reset(self, raw_env_output) -> None:
        pass

    async def act(self):
        return await self._act()

    def observe(self, action, step_result) -> None:
        pass


def _install_scripted_policy(monkeypatch: pytest.MonkeyPatch, act_fn) -> None:
    monkeypatch.setattr(
        core_module,
        "build_policy",
        lambda llm, events, **scalars: _ScriptedPolicy(act_fn),
    )


def test_run_rollout_classifies_foreign_policy_timeout_as_policy_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def act():
        raise TimeoutError("policy bug")

    _install_scripted_policy(monkeypatch, act)
    result = _run_rollout(
        tmp_path,
        _FakeLlm([]),
        _FakeEnv(),
        agent_timeout_sec=30.0,
    )

    assert result.failure_mode == "crash"
    assert result.failure_origin == "policy"
    assert result.error == "TimeoutError: policy bug"


def test_run_rollout_classifies_nameless_action_as_policy_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The frozen loop reads action.name; a malformed candidate action is a
    # scorable policy crash, not a framework defect that aborts the experiment.
    async def act():
        return (object(),)

    _install_scripted_policy(monkeypatch, act)
    llm = _FakeLlm([])
    env = _FakeEnv()

    result = _run_rollout(tmp_path, llm, env, agent_timeout_sec=30.0)

    assert result.failure_mode == "crash"
    assert result.failure_origin == "policy"
    assert result.error == ("AttributeError: 'object' object has no attribute 'name'")
    _assert_closed(env, llm)


def test_run_rollout_classifies_non_iterable_act_result_as_policy_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def act():
        return None

    _install_scripted_policy(monkeypatch, act)
    llm = _FakeLlm([])
    env = _FakeEnv()

    result = _run_rollout(tmp_path, llm, env, agent_timeout_sec=30.0)

    assert result.failure_mode == "crash"
    assert result.failure_origin == "policy"
    assert result.error == "TypeError: 'NoneType' object is not iterable"
    _assert_closed(env, llm)


def test_run_rollout_classifies_invalid_env_action_route_as_policy_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Action:
        name = "run"

    async def act():
        return (_Action(),)

    _install_scripted_policy(monkeypatch, act)
    monkeypatch.setattr(core_module, "build_env_action", lambda action: object())
    llm = _FakeLlm([])
    env = _FakeEnv()

    result = _run_rollout(tmp_path, llm, env, agent_timeout_sec=30.0)

    assert result.failure_mode == "crash"
    assert result.failure_origin == "policy"
    assert result.error == (
        "ValueError: action route for 'run' produced object; expected RunAction"
    )
    _assert_closed(env, llm)


def test_run_rollout_propagates_framework_completion_failure(tmp_path: Path) -> None:
    # A FrameworkError (backend defect) re-raises its cause past classification.
    class BrokenBackend(_FakeLlm):
        async def _complete(self, request):
            del request
            try:
                raise AssertionError("backend invariant failed")
            except AssertionError as exc:
                raise FrameworkError from exc

    with pytest.raises(AssertionError, match="backend invariant failed"):
        _run_rollout(tmp_path, BrokenBackend([]), _FakeEnv())


def test_run_rollout_writes_executed_chain_prefix_when_later_action_crashes(
    tmp_path: Path,
) -> None:
    class SecondActionCrashes(_FakeEnv):
        async def execute(self, action: RunAction) -> RawEnvOutput:
            if self.exec_calls:
                raise RuntimeError("second action crashed")
            return await super().execute(action)

    result = _run_rollout(
        tmp_path,
        _FakeLlm(
            [
                _completion(_tool_call("run", command="first")),
                _completion(_tool_call("run", command="second")),
            ]
        ),
        SecondActionCrashes(),
    )

    assert result.failure_mode == "crash"
    rows = _chain_rows(tmp_path)
    assert [row["action"]["command"] for row in rows] == ["first"]


def test_run_rollout_aborts_stalled_rollout_before_agent_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(episode, "_stall_timeout_sec", lambda config: 0.05)
    llm = _FakeLlm([_completion() for _ in range(3)], completion_delay_sec=0.2)
    env = _FakeEnv()

    result = _run_rollout(
        tmp_path,
        llm,
        env,
        config=_rollout_config(max_steps=10),
        agent_timeout_sec=30.0,
    )

    assert result.failure_mode == "hit_timeout"
    assert result.error is not None and "aborting stalled rollout" in result.error
    _assert_closed(env, llm)
    _assert_empty_chain(tmp_path)


def test_run_rollout_executed_actions_refresh_the_stall_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each 0.1s turn fits 0.18s, but all three exceed it; actions must refresh it.
    monkeypatch.setattr(episode, "_stall_timeout_sec", lambda config: 0.18)
    llm = _FakeLlm(
        [
            _completion(_tool_call("run", command="ls")),
            _completion(_tool_call("run", command="ls")),
            _completion(_tool_call("submit")),
        ],
        completion_delay_sec=0.1,
    )
    env = _FakeEnv()

    result = _run_rollout(
        tmp_path,
        llm,
        env,
        config=_rollout_config(max_steps=10),
        agent_timeout_sec=30.0,
    )

    assert result.failure_mode == "solved"
    assert result.error is None


def test_stall_timeout_covers_one_full_act_of_llm_attempts() -> None:
    config = _rollout_config()
    cfg = config.llm_provider_config
    per_attempt = backend_class(cfg.provider).complete_duration_bound_sec(
        cfg.max_tokens
    )
    assert episode._stall_timeout_sec(config) == pytest.approx(
        3 * per_attempt + episode.STALL_TIMEOUT_SLACK_SEC
    )


def _trace_events(path: Path) -> list[str]:
    return [
        row["event"]
        for line in path.read_text().splitlines()
        if line.strip() and (row := json.loads(line))["event"] != "policy_rule"
    ]


def test_run_rollout_passes_original_env_to_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def run_loop(**kwargs: Any) -> LoopOutcome:
        captured.update(kwargs)
        return LoopOutcome(end="step_cap", final=None, steps_taken=0)

    monkeypatch.setattr(episode, "run_task_loop", run_loop)
    env = _FakeEnv()

    _run_rollout(tmp_path, _FakeLlm([]), env)

    expected = "policy build_env_action max_steps env executor telemetry agent_timeout_sec stall_timeout_sec action_chain"
    assert set(captured) == set(expected.split())
    assert captured["build_env_action"] is core_module.build_env_action
    assert captured["env"] is env
    assert isinstance(captured["executor"], LiveStepExecutor)
    assert captured["max_steps"] == 3
    assert isinstance(captured["telemetry"], episode.RolloutTelemetry)
    # The policy's completions flow through the frozen measurement boundary.
    assert isinstance(captured["policy"].llm, episode.InstrumentedLlm)
    assert captured["agent_timeout_sec"] == 1.0


def test_run_rollout_scores_close_timeout_as_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Bound hanging close() so concurrency-1 experiments do not stall.
    class _HangingCloseEnv(_FakeEnv):
        async def close(self) -> None:
            await asyncio.sleep(60)

    monkeypatch.setattr(episode, "TRIAL_CLOSE_TIMEOUT_SEC", 0.05)
    llm = _FakeLlm([_completion(_tool_call("submit"))])

    result = _run_rollout(tmp_path, llm, _HangingCloseEnv())

    assert result.failure_mode == "crash"
    assert result.error is not None
    assert "resource close timed out" in result.error
    assert llm.closed is True


def test_run_rollout_closes_resources_when_cancelled(tmp_path: Path) -> None:
    # CancelledError bypasses except clauses, but must still close both resources.
    env = _FakeEnv(reset_delay_sec=30.0)
    llm = _FakeLlm([])

    async def scenario() -> None:
        task = asyncio.create_task(
            RolloutRunner(
                "task-a",
                llm,
                env,
                LiveStepExecutor(env),
                _rollout_config(),
                _telemetry(tmp_path),
                30.0,
            ).run()
        )
        while env.reset_calls == 0:
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    _assert_closed(env, llm)
    _assert_empty_chain(tmp_path)


@pytest.fixture
def _cache_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("src.plugins.caching.store._DISABLED", False)


def _replay_cache(env: TaskEnv):
    return ReplayCache(namespace="ns", epoch=0, env=env)


def _run_then_submit_llm() -> _FakeLlm:
    return _FakeLlm(
        [
            _completion(_tool_call("run", command="pwd")),
            _completion(_tool_call("submit")),
        ]
    )


def _run_warm_replay(path: Path, env: _FakeEnv, agent_timeout_sec: float):
    return _run_rollout(
        path,
        _run_then_submit_llm(),
        env,
        executor=_replay_cache(env),
        agent_timeout_sec=agent_timeout_sec,
    )


def _prime_verify_miss(path: Path) -> None:
    result = _run_warm_replay(
        path,
        _FakeEnv(verify_delay_sec=0.3, verify_timeout_sec=0.05),
        10.0,
    )
    assert result.failure_mode == "verify_timeout"


def test_replay_storage_failure_aborts_framework(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _cache_enabled
) -> None:
    async def fail_put(key: str, value: str) -> None:
        del key, value
        raise RuntimeError("replay storage failed")

    monkeypatch.setattr("src.plugins.replay.step_cache.cache.put", fail_put)
    env = _FakeEnv()
    llm = _FakeLlm([_completion(_tool_call("run", command="pwd"))])

    with pytest.raises(RuntimeError, match="replay storage failed"):
        _run_rollout(tmp_path, llm, env, executor=_replay_cache(env))

    _assert_closed(env, llm)


def test_replay_env_failure_is_env_origin_crash(tmp_path: Path, _cache_enabled) -> None:
    class BrokenEnv(_FakeEnv):
        async def execute(self, action: RunAction) -> RawEnvOutput:
            del action
            raise RuntimeError("replayed env failed")

    env = BrokenEnv()
    result = _run_rollout(
        tmp_path,
        _FakeLlm([_completion(_tool_call("run", command="pwd"))]),
        env,
        executor=_replay_cache(env),
    )

    assert result.failure_mode == "crash"
    assert result.failure_origin == "env"
    assert result.error == "RuntimeError: replayed env failed"


def test_warm_replay_charges_materialization_to_task_budget_not_verify_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _cache_enabled,
) -> None:
    monkeypatch.setattr(episode, "VERIFY_BACKSTOP_SLACK_SEC", 0.0)

    _prime_verify_miss(tmp_path / "r1")

    env2 = _FakeEnv(step_delay_sec=0.3)
    result2 = _run_warm_replay(tmp_path / "r2", env2, 10.0)
    assert result2.failure_mode == "solved"
    assert env2.exec_calls == ["pwd"]  # verify-miss catch-up ran the prefix live
    assert env2.verify_calls == 1

    env3 = _FakeEnv()
    result3 = _run_warm_replay(tmp_path / "r3", env3, 10.0)
    assert result3.failure_mode == "solved"
    assert env3.exec_calls == []
    assert env3.verify_calls == 0


def test_materialization_past_agent_deadline_classifies_hit_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _cache_enabled,
) -> None:
    monkeypatch.setattr(episode, "VERIFY_BACKSTOP_SLACK_SEC", 0.0)
    _prime_verify_miss(tmp_path / "r1")

    env2 = _FakeEnv(step_delay_sec=0.4)
    result2 = _run_warm_replay(tmp_path / "r2", env2, 0.15)
    assert result2.failure_mode == "hit_timeout"
    assert env2.closed is True
