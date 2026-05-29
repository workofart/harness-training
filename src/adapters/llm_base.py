from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


ReasoningEffort = Literal["none", "low", "medium", "high"]


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None


@dataclass(frozen=True, slots=True)
class LlmToolCall:
    name: str
    arguments: str  # raw JSON string emitted by the model


@dataclass(frozen=True, slots=True)
class LlmUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_input_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class LlmCompletion:
    """Adapter-normalized completion. Replaces the dict/object polymorphism
    on the wire response shape: every adapter unpacks its provider type
    once into these typed fields, so the harness and trace never reach
    through `getattr`/`dict.get` chains.
    """

    tool_calls: tuple[LlmToolCall, ...] = ()
    content: str | None = None
    finish_reason: str | None = None
    usage: LlmUsage = field(default_factory=LlmUsage)
    # JSON-safe reasoning text/blocks, or None if the provider did not
    # surface any. Written verbatim to the trace event.
    reasoning_content: Any | None = None
    # JSON-safe slim envelope written under the trace event's `response`
    # field. Built by the adapter so trace.py does not need to know about
    # provider-specific message shapes.
    response: dict[str, Any] = field(default_factory=dict)


class BaseLlm(ABC):
    @property
    @abstractmethod
    def max_context_length(self) -> int:
        pass

    @abstractmethod
    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> LlmCompletion:
        pass

    def get_token_count(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        """Provider-agnostic token estimate (~3 serialized chars per token).

        Concrete because every provider's heuristic was byte-identical; a
        provider with a real tokenizer may override it.
        """
        payload = {"messages": messages, "tools": tools or []}
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return max(1, (len(serialized) + 2) // 3)

    async def close(self) -> None:
        pass
