"""Frozen measurement boundary: the per-rollout telemetry event sink."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest

import src.rollout.telemetry as telemetry_module
from src.env.base import RawEnvOutput, StepResult, VerifyVerdict
from src.llm.backend import (
    Completion,
    CompletionBackend,
    CompletionInfraError,
    CompletionRequest,
    ContextWindowExceededError,
    FrameworkError,
    ProviderRejectedToolCallError,
    ToolCall,
    Usage,
)
from src.policy.base import PolicyEventName
from src.rollout.telemetry import (
    LIVE_LLM_CALLS_KEY,
    MEDIAN_LIVE_LLM_LATENCY_SEC_KEY,
    P25_LIVE_OUTPUT_TOKENS_PER_SEC_KEY,
    SUM_LIVE_LLM_LATENCY_SEC_KEY,
    InstrumentedLlm,
    RolloutTelemetry,
)

from _rollout_fixtures import (
    _telemetry,
)


def _record_call(
    telemetry: RolloutTelemetry,
    completion: Completion,
    *,
    llm_latency_sec: float,
) -> None:
    # Request context does not affect folded live-LLM metrics; use empty values.
    telemetry.on_completion_received(
        request_messages=[],
        request_tools=[],
        completion=completion,
        llm_latency_sec=llm_latency_sec,
    )


def test_rollout_telemetry_records_allowed_generic_policy_events(
    tmp_path: Path,
) -> None:
    telemetry = _telemetry(tmp_path)
    cases: list[tuple[PolicyEventName, dict[str, Any]]] = [
        (
            "context_window_retrimmed",
            {
                "step_index": 2,
                "attempt_index": 0,
                "limit": 100,
                "requested": 120,
                "calibration": 1.2,
            },
        ),
        (
            "policy_rule",
            {"step_index": 3, "rule": "budget_reminder", "fired": False},
        ),
        ("observation_clipped", {"step_index": 4, "field": "stdout"}),
        ("context_groups_dropped", {"step_index": 5, "dropped": 3}),
    ]

    for event, fields in cases:
        telemetry.on_policy_event(event, **fields)

    rows = [
        json.loads(line)
        for line in (tmp_path / "agent" / "steps.jsonl").read_text().splitlines()
    ]
    assert [
        {key: value for key, value in row.items() if key != "t_sec"} for row in rows
    ] == [{"event": event, **fields} for event, fields in cases]


def test_rollout_telemetry_metrics_omit_unmeasured_signals(tmp_path: Path) -> None:
    telemetry = _telemetry(tmp_path)

    assert telemetry.metrics() == {}

    telemetry.on_policy_event(
        "action_parse_failed",
        step_index=0,
        error="MissingToolCall",
        detail="no tool call",
    )

    assert telemetry.metrics() == {"parse_failures.MissingToolCall": 1}


def test_rollout_telemetry_rejects_unknown_policy_event(tmp_path: Path) -> None:
    # The closed vocabulary: the harness cannot mint trusted-looking rows (e.g.
    # "completion_received") through the policy-event channel.
    telemetry = _telemetry(tmp_path)

    with pytest.raises(ValueError, match="unknown policy event"):
        telemetry.on_policy_event(cast(Any, "completion_received"), step_index=1)

    assert not (tmp_path / "agent" / "steps.jsonl").exists()


@pytest.mark.parametrize(
    "error",
    [
        ContextWindowExceededError(
            "context overflow",
            limit=32768,
            requested=32770,
            input_tokens=24578,
        ),
        CompletionInfraError("provider unavailable"),
        FrameworkError("backend defect"),
        ProviderRejectedToolCallError("invalid tool call arguments"),
    ],
)
def test_instrumented_llm_preserves_typed_backend_error(
    tmp_path: Path, error: Exception
) -> None:
    class FailingBackend(CompletionBackend):
        async def _complete(self, request: CompletionRequest) -> Completion:
            del request
            raise error

    backend = InstrumentedLlm(FailingBackend(), _telemetry(tmp_path))

    with pytest.raises(type(error)) as raised:
        asyncio.run(backend.complete(CompletionRequest(messages=[])))

    assert raised.value is error


def test_rollout_telemetry_writes_slimmed_completion_and_step_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ticks = iter([100.0, 100.1256, 100.5])
    monkeypatch.setattr(telemetry_module.time, "monotonic", lambda: next(ticks))
    telemetry = _telemetry(tmp_path)

    telemetry.on_completion_received(
        request_messages=[
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "fix"},
        ],
        request_tools=[{"type": "function", "function": {"name": "run"}}],
        completion=Completion(
            tool_calls=(
                ToolCall(
                    name="run",
                    arguments='{"command": "pwd", "timeout_sec": 3}',
                ),
                ToolCall(name="submit", arguments="{not-json"),
            ),
            content="",
            finish_reason="tool_calls",
            usage=Usage(
                prompt_tokens=7,
                completion_tokens=11,
                reasoning_tokens=2,
                cached_input_tokens=3,
            ),
            reasoning_content={"blocks": ["think"]},
            response={"id": "resp-1"},
        ),
        llm_latency_sec=0.25,
    )
    telemetry.on_step_completed(
        step_index=1,
        call_index=0,
        step_result=StepResult(
            raw_env_output=RawEnvOutput(
                instruction="do",
                working_dir="/work",
                exit_code=0,
                stdout="ok\n",
                stderr="",
            ),
            reward=1.0,
            terminated=True,
            truncated=False,
            info={"passed": True, "patch": "diff"},
            metrics={"fail_to_pass_passed": 2, "pass_to_pass_failed": 0},
            verdict=VerifyVerdict(
                completed=True,
                passed=True,
                error=None,
            ),
        ),
        terminal=True,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "agent" / "steps.jsonl").read_text().splitlines()
    ]
    assert rows == [
        {
            "event": "completion_received",
            "t_sec": 0.1256,
            "step_index": 1,
            "attempt_index": 0,
            "request_messages": [
                {"role": "system", "content": "policy", "sha": "e14ae89ab76c"},
                {"role": "user", "content": "fix", "sha": "55f85ef0ae71"},
            ],
            "request_tools": {
                "tools": [{"type": "function", "function": {"name": "run"}}],
                "sha": "c950326cab43",
            },
            "completion": {
                "content": "",
                "finish_reason": "tool_calls",
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 11,
                    "reasoning_tokens": 2,
                    "cached_input_tokens": 3,
                },
                "reasoning_content": {"blocks": ["think"]},
                "response": {"id": "resp-1"},
                "served_from_cache": False,
                "calls": [
                    {
                        "call_index": 0,
                        "name": "run",
                        "args": {"command": "pwd", "timeout_sec": 3},
                    },
                    {
                        "call_index": 1,
                        "name": "submit",
                        "raw_arguments": "{not-json",
                        "args_error": "Expecting property name enclosed in double "
                        "quotes: line 1 column 2 (char 1)",
                    },
                ],
            },
            "llm_latency_sec": 0.25,
        },
        {
            "event": "step_completed",
            "t_sec": 0.5,
            "step_index": 1,
            "call_index": 0,
            "step_result": {
                "raw_env_output": {
                    "instruction": "do",
                    "working_dir": "/work",
                    "exit_code": 0,
                    "stdout": "ok\n",
                    "stderr": "",
                },
                "reward": 1.0,
                "terminated": True,
                "truncated": False,
                "info": {"passed": True, "patch": "diff"},
                "metrics": {"fail_to_pass_passed": 2, "pass_to_pass_failed": 0},
                "verdict": {
                    "completed": True,
                    "passed": True,
                    "error": None,
                },
            },
            "terminal": True,
        },
    ]
    assert "action_name" not in rows[1]


def test_rollout_telemetry_dedups_repeated_request_messages_within_rollout(
    tmp_path: Path,
) -> None:
    telemetry = _telemetry(tmp_path)
    system = {"role": "system", "content": "policy"}
    tools = [{"name": "run"}]

    def send(step_index: int, *messages: dict[str, str]) -> None:
        telemetry.steps_taken = step_index - 1
        telemetry.on_completion_received(
            request_messages=list(messages),
            request_tools=tools,
            completion=Completion(),
            llm_latency_sec=0.1,
        )

    send(1, system, {"role": "user", "content": "task"})
    send(
        2,
        system,
        {"role": "user", "content": "task"},
        {"role": "tool", "content": "rc=0"},
    )

    first, second = [
        json.loads(line)
        for line in (tmp_path / "agent" / "steps.jsonl").read_text().splitlines()
    ]
    assert [m["content"] for m in first["request_messages"]] == ["policy", "task"]
    assert first["request_tools"]["tools"] == tools
    assert second["request_messages"][0] == {
        "role": "system",
        "sha": first["request_messages"][0]["sha"],
    }
    assert second["request_messages"][1] == {
        "role": "user",
        "sha": first["request_messages"][1]["sha"],
    }
    assert second["request_messages"][2]["content"] == "rc=0"
    assert second["request_tools"] == {"sha": first["request_tools"]["sha"]}

    full_by_sha = {
        m["sha"]: {k: v for k, v in m.items() if k != "sha"}
        for m in first["request_messages"] + second["request_messages"]
        if "content" in m
    }
    resolved = [
        full_by_sha[m["sha"]]
        if "content" not in m
        else {k: v for k, v in m.items() if k != "sha"}
        for m in second["request_messages"]
    ]
    assert resolved == [
        system,
        {"role": "user", "content": "task"},
        {"role": "tool", "content": "rc=0"},
    ]

    other = _telemetry(tmp_path / "other")
    other.on_completion_received(
        request_messages=[system],
        request_tools=tools,
        completion=Completion(),
        llm_latency_sec=0.1,
    )
    [other_row] = [
        json.loads(line)
        for line in (tmp_path / "other" / "agent" / "steps.jsonl")
        .read_text()
        .splitlines()
    ]
    assert other_row["request_messages"][0]["content"] == "policy"


def test_rollout_telemetry_folds_live_llm_metrics(tmp_path: Path) -> None:
    telemetry = _telemetry(tmp_path)

    _record_call(
        telemetry, Completion(usage=Usage(completion_tokens=10)), llm_latency_sec=1.0
    )
    _record_call(
        telemetry, Completion(usage=Usage(completion_tokens=4)), llm_latency_sec=0.5
    )

    metrics = telemetry.metrics()
    assert metrics[LIVE_LLM_CALLS_KEY] == 2
    assert metrics[SUM_LIVE_LLM_LATENCY_SEC_KEY] == 1.5
    assert metrics[MEDIAN_LIVE_LLM_LATENCY_SEC_KEY] == 0.75
    assert P25_LIVE_OUTPUT_TOKENS_PER_SEC_KEY not in metrics

    _record_call(
        telemetry, Completion(usage=Usage(completion_tokens=3)), llm_latency_sec=0.25
    )

    assert telemetry.metrics()[P25_LIVE_OUTPUT_TOKENS_PER_SEC_KEY] == 9.0


def test_rollout_telemetry_omits_cached_calls_from_live_metrics(
    tmp_path: Path,
) -> None:
    telemetry = _telemetry(tmp_path)

    _record_call(
        telemetry,
        Completion(usage=Usage(completion_tokens=10), served_from_cache=True),
        llm_latency_sec=0.25,
    )

    assert telemetry.metrics() == {}


def test_rollout_telemetry_caller_cannot_overwrite_canonical_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    times = iter((10.0, 10.25))
    monkeypatch.setattr(telemetry_module.time, "monotonic", lambda: next(times))
    telemetry = _telemetry(tmp_path)

    telemetry.event("canonical", event="forged", t_sec=999)

    row = json.loads((tmp_path / "agent" / "steps.jsonl").read_text())
    assert row["event"] == "canonical"
    assert row["t_sec"] == 0.25
