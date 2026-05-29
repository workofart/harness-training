from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.adapters.llm_base import LlmCompletion
from src.harness.contracts import RawState
from src.metrics import FailureMode, TaskMetrics
from src.serialization import json_safe

STEP_TRACE_FILENAME = "steps.jsonl"
TASK_METRICS_FILENAME = "metrics.json"


def task_artifact_paths(trial_dir: str | Path) -> tuple[str, str]:
    agent_dir = Path(trial_dir) / "agent"
    return (
        str(agent_dir / STEP_TRACE_FILENAME),
        str(agent_dir / TASK_METRICS_FILENAME),
    )


# Trace event `tools_registered` is emitted once per fingerprint/run and
# carries `fingerprint` + `tool_names` only; per-step `completion_received`
# events reference the registry by `request_tools_fp`. The full tool
# schemas are constant per experiment and recoverable from
# `src.harness.core.build_tool_specs()` at the experiment's
# `git_commit_hash`. The model identity lives in
# `config/harness_config.json`, not the per-trial trace.
# `request_messages_delta` carries only the messages added since the
# previous completion, so a reader replays deltas to reconstruct the full
# prompt history.


@dataclass(frozen=True, slots=True)
class TraceEvent:
    ts: str
    event: str
    fields: dict[str, Any]


def _tool_name(entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return None
    fn = entry.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        if isinstance(name, str):
            return name
    name = entry.get("name")
    return name if isinstance(name, str) else None


class TraceWriter:
    def __init__(self, trace_path: Path) -> None:
        self.trace_path = trace_path
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_request_messages: list[Any] | None = None
        self._tool_specs_seen: set[str] = set()

    def write(self, event: str, **fields: Any) -> None:
        payload = TraceEvent(
            ts=datetime.now(timezone.utc).isoformat(),
            event=event,
            fields=fields,
        )
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(payload)) + "\n")

    def consume_message_delta(
        self,
        request_messages: list[dict[str, Any]] | None,
    ) -> tuple[int, list[Any]]:
        """Encode this turn's request_messages against the previous one.

        Returns `(reuse, tail)` where `reuse` is how many leading messages
        from the previous turn's stored list are reused verbatim, and
        `tail` is everything from that offset onward.

        Replay protocol:

            messages = []
            for event in completion_received_events:
                messages = messages[:event.request_messages_delta_reuse] + event.request_messages_delta

        With the multi-turn message shape the common case is `reuse == len(prev)`,
        so `tail` is the strict append since the previous turn. When the
        latest tool result transitions to the historical char-limit on a
        subsequent turn — or when compaction drops the oldest steps —
        `reuse` is shorter than `len(prev)` and `tail` carries everything
        from the first changed position. Without recording `reuse`, replay
        would either duplicate the mutated entries (naive concat) or drop
        the changes (overwrite by length).
        """
        current = list(request_messages or [])
        prev = self._last_request_messages
        self._last_request_messages = current
        if prev is None:
            return 0, current
        reuse = 0
        for a, b in zip(prev, current):
            if a == b:
                reuse += 1
            else:
                break
        return reuse, current[reuse:]

    def maybe_register_tools(
        self,
        request_tools: list[dict[str, Any]] | None,
    ) -> str | None:
        if not request_tools:
            return None
        safe_tools = json_safe(request_tools)
        fingerprint = hashlib.sha1(
            json.dumps(safe_tools, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        if fingerprint not in self._tool_specs_seen:
            self._tool_specs_seen.add(fingerprint)
            # Emit fingerprint + tool names only. Full schemas are constant
            # across the experiment and recoverable from
            # `src.harness.core.build_tool_specs()` at the experiment's
            # `git_commit_hash` — see experiment.json.
            tool_names = sorted(
                name
                for name in (_tool_name(entry) for entry in (safe_tools or []))
                if name is not None
            )
            self.write(
                "tools_registered",
                fingerprint=fingerprint,
                tool_names=tool_names,
            )
        return fingerprint


def _completion_trace_fields(
    completion: LlmCompletion,
    *,
    request_messages_delta: list[Any] | None = None,
    request_messages_delta_reuse: int = 0,
    request_tools_fp: str | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "finish_reason": completion.finish_reason,
        "token_input": completion.usage.prompt_tokens,
        "token_output": completion.usage.completion_tokens,
        "token_reasoning": completion.usage.reasoning_tokens,
        "token_cached_input": completion.usage.cached_input_tokens,
        "request_messages_delta": json_safe(request_messages_delta or []),
        "request_messages_delta_reuse": request_messages_delta_reuse,
        "request_tools_fp": request_tools_fp,
        "response": completion.response,
    }
    if completion.reasoning_content is not None:
        fields["reasoning_content"] = completion.reasoning_content
    return fields


@dataclass(slots=True)
class StepRecorder:
    trace: TraceWriter | None = None
    metrics: TaskMetrics | None = None
    step_index: int | None = None

    def _write(self, event: str, **fields: Any) -> None:
        if self.trace is None:
            return
        if self.step_index is not None:
            fields = {**fields, "step_index": self.step_index}
        self.trace.write(event, **fields)

    def rule_fired(self, rule_name: str) -> None:
        if self.metrics is not None:
            self.metrics.record_rule_fire(rule_name)

    def completion_received(
        self,
        *,
        attempt_index: int,
        request_messages: list[dict[str, Any]],
        request_tools: list[dict[str, Any]] | None,
        completion: LlmCompletion,
    ) -> None:
        if self.metrics is not None:
            self.metrics.record_completion_usage(completion.usage)
        writer = self.trace
        if writer is not None:
            tools_fp = writer.maybe_register_tools(request_tools)
            delta_reuse, messages_delta = writer.consume_message_delta(request_messages)
        else:
            tools_fp = None
            delta_reuse = 0
            messages_delta = list(request_messages or [])
        self._write(
            "completion_received",
            attempt_index=attempt_index,
            **_completion_trace_fields(
                completion,
                request_messages_delta=messages_delta,
                request_messages_delta_reuse=delta_reuse,
                request_tools_fp=tools_fp,
            ),
        )

    def action_parse_failed(self, *, error: str, detail: str) -> None:
        if self.metrics is not None:
            self.metrics.record_action_parse_failure()
        self._write("action_parse_failed", error=error, detail=detail)

    def action_chosen(
        self,
        *,
        action_name: str,
        action_summary: dict[str, Any],
    ) -> None:
        if self.metrics is not None and self.step_index is not None:
            self.metrics.record_action(self.step_index, action_name)
        self._write(
            "action_chosen",
            action_name=action_name,
            action_summary=action_summary,
        )

    def env_step_completed(
        self,
        *,
        action_name: str,
        action_summary: dict[str, Any],
        raw_state: RawState,
    ) -> None:
        if self.metrics is not None:
            self.metrics.record_step_passed(raw_state)
        self._write(
            "env_step_completed",
            action_name=action_name,
            return_code=raw_state.return_code,
            done=raw_state.done,
            passed=raw_state.passed,
        )


@dataclass(slots=True)
class HarnessRecorder:
    trace: TraceWriter | None = None
    metrics: TaskMetrics | None = None
    metrics_path: Path | None = None

    @classmethod
    def create(
        cls,
        *,
        trace_path: str | Path | None = None,
        metrics_path: str | Path | None = None,
    ) -> "HarnessRecorder":
        writer = None if trace_path is None else TraceWriter(Path(trace_path))
        resolved_metrics_path = None
        if metrics_path is not None:
            resolved_metrics_path = Path(metrics_path)
        return cls(
            trace=writer,
            metrics=TaskMetrics(),
            metrics_path=resolved_metrics_path,
        )

    def _write(self, event: str, **fields: Any) -> None:
        if self.trace is None:
            return
        self.trace.write(event, **fields)

    def for_step(self, step_index: int) -> StepRecorder:
        return StepRecorder(self.trace, self.metrics, step_index)

    def task_started(
        self,
        *,
        task_name: str,
        instruction: str,
        working_dir: str | None,
    ) -> None:
        self._write(
            "task_started",
            task_name=task_name,
            instruction=instruction,
            working_dir=working_dir,
        )

    def final_verify_started(
        self,
        *,
        step_limit_reached: bool,
        steps_used: int,
    ) -> None:
        self._write(
            "final_verify_started",
            step_limit_reached=step_limit_reached,
            steps_used=steps_used,
        )

    def final_verify_completed(
        self,
        *,
        raw_state: RawState,
        step_limit_reached: bool,
        steps_used: int,
    ) -> None:
        self._write(
            "final_verify_completed",
            done=raw_state.done,
            passed=raw_state.passed,
            reward=raw_state.reward,
            step_limit_reached=step_limit_reached,
            steps_used=steps_used,
        )

    def task_finished(
        self,
        *,
        task_name: str,
        reward: float | None,
        solved: bool,
        error: str | None,
        steps_used: int,
        final_passed: bool | None,
        forced_final_verify: bool,
    ) -> None:
        self._write(
            "task_finished",
            task_name=task_name,
            reward=reward,
            solved=solved,
            error=error,
            steps_used=steps_used,
            final_passed=final_passed,
            forced_final_verify=forced_final_verify,
        )

    def task_failed(self, *, exc: BaseException, detail: str) -> None:
        self._write("task_failed", error=type(exc).__name__, detail=detail)

    def cleanup_failed(self, *, component: str, exc: BaseException) -> None:
        self._write(
            "cleanup_failed",
            component=component,
            error=type(exc).__name__,
            detail=str(exc),
        )

    def write_metrics(self, path: str | Path | None = None) -> None:
        if self.metrics is None:
            return
        resolved_path = self.metrics_path if path is None else Path(path)
        if resolved_path is None:
            return
        self.metrics.write(resolved_path)

    def build_metrics(self) -> TaskMetrics:
        if self.metrics is None:
            return TaskMetrics()
        return self.metrics

    def set_trial_outcome(
        self,
        *,
        verifier_passed: bool | None,
        failure_mode: FailureMode | None,
    ) -> None:
        if self.metrics is None:
            return
        self.metrics.set_trial_outcome(
            verifier_passed=verifier_passed,
            failure_mode=failure_mode,
        )


NOOP_STEP_RECORDER = StepRecorder()
NOOP_HARNESS_RECORDER = HarnessRecorder()
