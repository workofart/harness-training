"""Frozen per-rollout measurement boundary: the event sink and the instrumented LLM.

Why this boundary exists: src/rollout/README.md ("Evidence integrity").
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from statistics import median, quantiles
import time
from typing import Any, get_args

from src.env.base import StepResult
from src.llm.backend import (
    Completion,
    CompletionBackend,
    CompletionRequest,
    FrameworkError,
    ProviderRejectedToolCallError,
)
from src.policy.base import PolicyEventName

FIRST_ATTEMPT_VALID_KEY = "first_attempt_valid"
FIRST_ATTEMPT_TOTAL_KEY = "first_attempt_total"
LIVE_LLM_CALLS_KEY = "live_llm_calls"
SUM_LIVE_LLM_LATENCY_SEC_KEY = "sum_live_llm_latency_sec"
MEDIAN_LIVE_LLM_LATENCY_SEC_KEY = "median_live_llm_latency_sec"
P25_LIVE_OUTPUT_TOKENS_PER_SEC_KEY = "p25_live_output_tokens_per_sec"

_POLICY_EVENT_NAMES = frozenset(get_args(PolicyEventName))


@dataclass(slots=True)
class RolloutTelemetry:
    """Central event sink and metrics fold for one rollout.

    Every trace row passes through here, and every derived rollout metric is folded
    from that same ordered event stream in this one class -- emitters announce facts
    at their boundaries and never count. Step/attempt labels for completions are
    derived here too: a completion belongs to the step after the last concluded one,
    and its attempt index is its ordinal within that step -- provider-rejected
    calls occupy ordinals like completions, so attempt indices match the
    policy's repair loop.
    """

    rollout_dir: Path
    trace_path: Path
    parse_failures: dict[str, int] = field(default_factory=dict, init=False)
    first_attempt_valid: int = field(default=0, init=False)
    first_attempt_total: int = field(default=0, init=False)
    steps_taken: int = field(default=0, init=False)
    _pending_completions: int = field(default=0, init=False)
    _live_llm_latencies: list[float] = field(default_factory=list, init=False)
    _live_rates: list[float] = field(default_factory=list, init=False)
    _seen_request_shas: set[str] = field(default_factory=set, init=False)
    _start: float = field(init=False)

    def __post_init__(self) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._start = time.monotonic()

    def event(self, name: str, **fields: Any) -> None:
        # Trace-only wall time supports latency analysis; it is never fed to the model.
        record = {
            **fields,
            "event": name,
            "t_sec": round(time.monotonic() - self._start, 4),
        }
        with self.trace_path.open("a") as handle:
            handle.write(json.dumps(record))
            handle.write("\n")

    def _conclude_decision(self, *, valid: bool) -> None:
        # A zero-completion decision (only non-LLM test agents) is not a
        # first-try sample; a crash mid-decision never concludes.
        if self._pending_completions == 0:
            return
        self.first_attempt_total += 1
        if valid and self._pending_completions == 1:
            self.first_attempt_valid += 1
        self._pending_completions = 0

    def on_policy_event(self, event: PolicyEventName, /, **fields: Any) -> None:
        # Reject unknown events so the harness cannot mint trusted-looking trace rows.
        if event not in _POLICY_EVENT_NAMES:
            raise ValueError(f"unknown policy event: {event}")
        if event == "action_parse_failed":
            error = fields["error"]
            self.parse_failures[error] = self.parse_failures.get(error, 0) + 1
        self.event(event, **fields)

    def on_completion_received(
        self,
        *,
        request_messages: list[dict[str, Any]],
        request_tools: list[dict[str, Any]],
        completion: Completion,
        llm_latency_sec: float,
    ) -> None:
        attempt_index = self._pending_completions
        self._pending_completions += 1
        if not completion.served_from_cache:
            self._live_llm_latencies.append(llm_latency_sec)
            completion_tokens = completion.usage.completion_tokens
            if completion_tokens is not None and llm_latency_sec > 0:
                self._live_rates.append(completion_tokens / llm_latency_sec)
        fields = _slim_completion_event(
            {
                "step_index": self.steps_taken + 1,
                "attempt_index": attempt_index,
                "request_messages": request_messages,
                "request_tools": request_tools,
                "completion": dataclasses.asdict(completion),
                "llm_latency_sec": llm_latency_sec,
            },
            self._seen_request_shas,
        )
        self.event("completion_received", **fields)

    def on_completion_rejected(self, *, error: str) -> None:
        # A provider-rejected tool call is a real model attempt that produced no
        # completion; counting it keeps first-try validity honest.
        attempt_index = self._pending_completions
        self._pending_completions += 1
        self.event(
            "completion_rejected",
            step_index=self.steps_taken + 1,
            attempt_index=attempt_index,
            error=error,
        )

    def on_no_valid_action_step(self, *, step_index: int) -> None:
        self._conclude_decision(valid=False)
        self.steps_taken = step_index
        self.event("no_valid_action_step", step_index=step_index)

    def on_step_completed(
        self,
        *,
        step_index: int,
        call_index: int,
        step_result: StepResult,
        terminal: bool,
    ) -> None:
        if call_index == 0:
            self._conclude_decision(valid=True)
        self.steps_taken = step_index
        self.event(
            "step_completed",
            step_index=step_index,
            call_index=call_index,
            step_result=dataclasses.asdict(step_result),
            terminal=terminal,
        )

    def metrics(self) -> dict[str, int | float]:
        metrics: dict[str, int | float] = {}
        if self.first_attempt_total:
            metrics[FIRST_ATTEMPT_VALID_KEY] = self.first_attempt_valid
            metrics[FIRST_ATTEMPT_TOTAL_KEY] = self.first_attempt_total
        if latencies := self._live_llm_latencies:
            metrics[LIVE_LLM_CALLS_KEY] = len(latencies)
            metrics[SUM_LIVE_LLM_LATENCY_SEC_KEY] = round(sum(latencies), 4)
            metrics[MEDIAN_LIVE_LLM_LATENCY_SEC_KEY] = round(median(latencies), 4)
        if len(rates := self._live_rates) >= 3:
            metrics[P25_LIVE_OUTPUT_TOKENS_PER_SEC_KEY] = round(
                quantiles(rates, n=4, method="inclusive")[0],
                4,
            )
        for error, count in self.parse_failures.items():
            metrics[f"parse_failures.{error}"] = count
        return metrics


class InstrumentedLlm(CompletionBackend):
    """Frozen per-rollout measurement boundary around the backend.

    Times each call and announces the completion together with the exact rendered
    request the model read -- ground truth the editable agent cannot forge, suppress,
    or mislabel. All counting and labeling is folded centrally in RolloutTelemetry.
    """

    def __init__(self, inner: CompletionBackend, telemetry: RolloutTelemetry) -> None:
        self._inner = inner
        self._telemetry = telemetry

    async def complete(self, request: CompletionRequest) -> Completion:
        started = time.monotonic()
        try:
            completion = await self._inner.complete(request)
        except ProviderRejectedToolCallError as rejection:
            try:
                self._telemetry.on_completion_rejected(error=str(rejection))
            except Exception as exc:
                raise FrameworkError from exc
            raise
        try:
            self._telemetry.on_completion_received(
                request_messages=request.messages,
                request_tools=request.tools,
                completion=completion,
                llm_latency_sec=round(time.monotonic() - started, 4),
            )
        except Exception as exc:
            raise FrameworkError from exc
        return completion

    async def close(self) -> None:
        await self._inner.close()


def _slim_completion_event(
    fields: dict[str, Any], seen_shas: set[str]
) -> dict[str, Any]:
    # Deduplicate requests so traces grow with novel text, not O(steps²).
    slim = dict(fields)
    slim["request_messages"] = [
        _dedup_request_blob(message, seen_shas, ref_keys=("role",))
        for message in slim["request_messages"]
    ]
    slim["request_tools"] = _dedup_request_blob(
        {"tools": slim["request_tools"]}, seen_shas, ref_keys=()
    )
    slim["completion"] = _completion_with_parsed_calls(slim["completion"])
    return slim


def _dedup_request_blob(
    blob: dict[str, Any], seen_shas: set[str], *, ref_keys: tuple[str, ...]
) -> dict[str, Any]:
    sha = hashlib.sha256(
        json.dumps(blob, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    if sha in seen_shas:
        return {**{key: blob[key] for key in ref_keys}, "sha": sha}
    seen_shas.add(sha)
    return {**blob, "sha": sha}


def _completion_with_parsed_calls(completion: dict[str, Any]) -> dict[str, Any]:
    raw = dict(completion)
    calls = []
    for index, call in enumerate(raw.pop("tool_calls", ())):
        parsed_call = {"call_index": index, "name": call["name"]}
        try:
            parsed_call["args"] = json.loads(call["arguments"] or "{}")
        except json.JSONDecodeError as exc:
            parsed_call["raw_arguments"] = call["arguments"]
            parsed_call["args_error"] = str(exc)
        calls.append(parsed_call)
    raw["calls"] = calls
    return raw
