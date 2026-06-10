from __future__ import annotations

import json

from src.llm.base import LlmCompletion, LlmUsage
from src.contracts import RawState, TaskMetrics
import src.trace as trace_module


def test_trace_artifact_paths_are_canonical():
    assert trace_module.STEP_TRACE_FILENAME == "steps.jsonl"
    assert trace_module.TASK_METRICS_FILENAME == "metrics.json"
    assert trace_module.task_artifact_paths("/tmp/trial") == (
        "/tmp/trial/agent/steps.jsonl",
        "/tmp/trial/agent/metrics.json",
    )


def test_trace_module_owns_recording_types():
    assert trace_module.task_artifact_paths.__module__ == "src.trace"
    assert trace_module.Recorder.__module__ == "src.trace"


def test_task_metrics_records_actions_usage_rules_and_outcome(tmp_path):
    metrics = TaskMetrics()

    metrics.record_action(1, "run")
    metrics.record_action(2, "verify")
    metrics.record_action_parse_failure()
    metrics.record_completion_usage(
        LlmUsage(
            prompt_tokens=10,
            completion_tokens=4,
            reasoning_tokens=2,
            cached_input_tokens=3,
        )
    )
    metrics.record_completion_usage(LlmUsage(prompt_tokens=None, completion_tokens=5))
    metrics.final_action_passed = RawState(passed=True).action_passed
    metrics.record_rule_fire("direct_literal_rule")
    metrics.record_rule_fire("direct_literal_rule")
    metrics.set_trial_outcome(verifier_passed=False, failure_mode="verified_rejected")

    assert metrics == TaskMetrics(
        steps_total=2,
        run_count=1,
        verify_count=1,
        action_parse_failure_count=1,
        token_input_total=10,
        token_output_total=9,
        token_reasoning_total=2,
        token_cached_input_total=3,
        rule_fires={"direct_literal_rule": 2},
        final_action_passed=True,
        verifier_passed=False,
        failure_mode="verified_rejected",
    )

    metrics_path = tmp_path / "agent" / "metrics.json"
    metrics.write(metrics_path)
    payload = json.loads(metrics_path.read_text())
    assert payload["rule_fires"] == {"direct_literal_rule": 2}
    assert payload["failure_mode"] == "verified_rejected"


def test_recorder_writes_sanitized_completion_jsonl_and_tool_registry(tmp_path):
    trace_path = tmp_path / "agent" / "steps.jsonl"
    writer = trace_module.TraceWriter(trace_path)
    metrics = TaskMetrics()
    recorder = trace_module.Recorder(trace=writer, metrics=metrics)
    request_tools = [
        {"function": {"name": "verify"}, "schema_path": tmp_path / "tool.json"},
        {"name": "run"},
    ]
    completion = LlmCompletion(
        finish_reason="tool_calls",
        usage=LlmUsage(
            prompt_tokens=7,
            completion_tokens=5,
            reasoning_tokens=2,
            cached_input_tokens=1,
        ),
        response={"id": "response-1"},
    )

    recorder.completion_received(
        step_index=3,
        attempt_index=0,
        request_messages=[{"role": "user", "content": tmp_path / "workspace"}],
        request_tools=request_tools,
        completion=completion,
    )
    recorder.completion_received(
        step_index=3,
        attempt_index=1,
        request_messages=[
            {"role": "user", "content": tmp_path / "workspace"},
            {"role": "assistant", "content": "continuing"},
        ],
        request_tools=request_tools,
        completion=completion,
    )

    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "tools_registered",
        "completion_received",
        "completion_received",
    ]
    registry = events[0]["fields"]
    assert registry["tool_names"] == ["run", "verify"]
    assert len(registry["fingerprint"]) == 12
    first_completion = events[1]["fields"]
    assert first_completion["step_index"] == 3
    assert first_completion["request_tools_fp"] == registry["fingerprint"]
    assert first_completion["request_messages_delta"] == [
        {"role": "user", "content": str(tmp_path / "workspace")}
    ]
    assert events[2]["fields"]["request_messages_delta_reuse"] == 1
    assert events[2]["fields"]["request_messages_delta"] == [
        {"role": "assistant", "content": "continuing"}
    ]
    assert metrics.token_output_total == 10


def test_recorder_env_step_derives_passed_and_attributes_it_to_the_last_action(
    tmp_path,
):
    # `passed` is derived (`RawState.action_passed`): an exec state carries only
    # `return_code`; a verify state carries the explicit judgment, which wins.
    trace_path = tmp_path / "agent" / "steps.jsonl"
    metrics = TaskMetrics()
    recorder = trace_module.Recorder(
        trace=trace_module.TraceWriter(trace_path), metrics=metrics
    )

    recorder.env_step_completed(
        step_index=1,
        action_name="run",
        raw_state=RawState(return_code=0, stdout="ok"),
    )
    assert metrics.final_action_passed is True
    recorder.env_step_completed(
        step_index=2,
        action_name="verify",
        raw_state=RawState(passed=False, reward=0.0),
    )

    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [(e["fields"]["step_index"], e["fields"]["passed"]) for e in events] == [
        (1, True),
        (2, False),
    ]
    assert all("done" not in e["fields"] for e in events)
    assert metrics.final_action_passed is False
