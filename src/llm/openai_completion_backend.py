from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, ClassVar

import httpx
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai import (
    AsyncOpenAI,
    BadRequestError,
)

from src.config import LlmProviderConfig
from src.llm.backend import (
    Completion,
    CompletionBackend,
    CompletionRequest,
    ContextWindowExceededError,
    MalformedProviderStreamError,
    ToolCall,
    Usage,
)
from src.llm.token_counter import resolve_token_counter
from src.llm.transport import (
    REQUEST_TIMEOUT_SECONDS,
    stream_stall_timeout_seconds,
    transient_retrying,
)

logger = logging.getLogger(__name__)

# Recount periodically to avoid O(n^2); slack keeps final clipping deterministic.
_BUDGET_RECOUNT_CHARS = 512
_BUDGET_SLACK_TOKENS = 64


def _reasoning_extension_field(value: Any) -> str | None:
    # sglang/vLLM extension fields outside the SDK types.
    return getattr(value, "reasoning_content", None) or getattr(
        value, "reasoning", None
    )


@dataclass(slots=True)
class _ReasoningBudgetWatch:
    budget: int
    token_counter: Any
    answered: bool = False
    reasoning: str = ""
    recount_at: int = 0

    def overrun(self, delta: Any) -> bool:
        if self.answered:
            return False
        # An open answer channel means thinking closed on its own.
        if delta.content or delta.tool_calls:
            self.answered = True
            return False
        self.reasoning += _reasoning_extension_field(delta) or ""
        if len(self.reasoning) < self.recount_at:
            return False
        self.recount_at = len(self.reasoning) + _BUDGET_RECOUNT_CHARS
        # Overrun by a slack: the clip then depends only on the generated
        # reasoning, not on where the stream was cut.
        return self.token_counter.count(self.reasoning) >= (
            self.budget + _BUDGET_SLACK_TOKENS
        )


class OpenAICompletionBackend(CompletionBackend):
    """Plain OpenAI-compatible chat-completions transport.

    Deterministic only against a seed-pinned deterministic endpoint."""

    @classmethod
    def complete_duration_bound_sec(cls, max_tokens: int) -> float:
        """Same window `_client()` sets as the httpx read timeout, so the bound
        published to the stall watchdog cannot drift from the enforced one."""
        return stream_stall_timeout_seconds(max_tokens)

    _CTX_LIMIT_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"context length of (\d+)", re.IGNORECASE
    )
    _CTX_TOTAL_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"total of (\d+) tokens", re.IGNORECASE
    )
    _CTX_INPUT_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"(\d+) tokens? from the input", re.IGNORECASE
    )

    def __init__(self, *, config: LlmProviderConfig) -> None:
        self.config = config
        self._client_obj: AsyncOpenAI | None = None
        self._budget_counter = (
            resolve_token_counter(config.tokenizer_name, config.model_name)
            if config.thinking_budget_tokens is not None
            else None
        )
        if config.thinking_budget_tokens is not None and self._budget_counter is None:
            raise ValueError(
                "thinking_budget_tokens requires a resolvable tokenizer "
                "(set tokenizer_name or use a Hub-id-shaped model_name)"
            )

    async def _complete(self, request: CompletionRequest) -> Completion:
        body = self._build_request_body(request)
        thinking_budget = self._thinking_budget(request.enable_thinking)
        client = self._client()

        async def _create_and_read() -> Completion:
            try:
                stream = await client.chat.completions.create(
                    **body,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                async with stream:
                    return await self._completion_from_stream(
                        stream, thinking_budget=thinking_budget
                    )
            except BadRequestError as exc:
                context_error = self._context_window_error(exc)
                if context_error is not None:
                    raise context_error from exc
                raise

        # One retry layer covers create() plus stream failures: dropped sockets, SSE errors.
        return await transient_retrying(logger)(_create_and_read)

    def _resolved_thinking(self, enable_thinking: bool | None) -> bool | None:
        """Single resolution point for both the request body and the abort
        budget, so the two cannot drift."""
        if enable_thinking is None:
            return self.config.enable_thinking
        if self.config.enable_thinking is None:
            raise ValueError(
                "enable_thinking override passed but this deployment declares no "
                "thinking channel (config enable_thinking is unset)"
            )
        return enable_thinking

    def _thinking_budget(self, enable_thinking: bool | None) -> int | None:
        """None when thinking is off: a thinking-off retry is never budget-forced."""
        thinking = self._resolved_thinking(enable_thinking)
        return self.config.thinking_budget_tokens if thinking else None

    def _budget_aborted_completion(self, reasoning: str) -> Completion:
        """Shaped like a natural runaway so the caller's existing thinking-off
        recovery fires, minus the decode to max_tokens. Salvaging the partial
        reasoning via continue_final_message was tried and lost: every observed
        continuation ran reasoning in the content channel to a second full
        budget.

        Clipping at exactly the budget is what lets the caching wrapper store
        the result under the request's key like any completed call."""
        clipped = self._budget_counter.truncate(
            reasoning, self.config.thinking_budget_tokens
        )
        return Completion(
            finish_reason="length",
            # No usage chunk exists; the clipped output is exactly the budget.
            usage=Usage(
                completion_tokens=self.config.thinking_budget_tokens,
                reasoning_tokens=self.config.thinking_budget_tokens,
            ),
            reasoning_content=clipped,
            response={"budget_aborted": True},
        )

    async def close(self) -> None:
        if self._client_obj is not None:
            await self._client_obj.close()
            self._client_obj = None

    def _api_key(self) -> str:
        """Read per request so building a backend to shape or key a request
        needs no environment."""
        try:
            return os.environ[self.config.api_key_env]
        except KeyError as exc:
            raise RuntimeError(
                f"environment variable {self.config.api_key_env} is not set "
                "(named by llm_provider_config.api_key_env)"
            ) from exc

    def _client(self) -> AsyncOpenAI:
        """Cached for the backend's life so rollout steps reuse one kept-alive
        connection instead of reconnecting per step."""
        if self._client_obj is None:
            self._client_obj = AsyncOpenAI(
                api_key=self._api_key(),
                base_url=self.config.base_url.rstrip("/"),
                # Scale the read timeout for buffered tool calls; clamping it can
                # cause retry livelock.
                timeout=httpx.Timeout(
                    REQUEST_TIMEOUT_SECONDS,
                    read=stream_stall_timeout_seconds(self.config.max_tokens),
                ),
                # complete()'s tenacity layer owns all retries; a nonzero value
                # here would nest budgets multiplicatively.
                max_retries=0,
            )
        return self._client_obj

    def cache_key(self, request: CompletionRequest) -> str:
        """Hash the exact body the provider would receive (minus stream/auth).

        Keying off the same builder `complete` sends makes the key complete
        (model, messages, tools, sampling, reasoning, routing) and unable to drift."""
        body = self._build_request_body(request)
        payload = json.dumps(
            {
                **body,
                "__client": {
                    "thinking_budget_tokens": self._thinking_budget(
                        request.enable_thinking
                    )
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _build_request_body(self, request: CompletionRequest) -> dict[str, Any]:
        config = self.config
        body: dict[str, Any] = {
            "model": config.model_name,
            "messages": request.messages,
        }
        if config.temperature is not None:
            body["temperature"] = config.temperature
        if config.top_p is not None:
            body["top_p"] = config.top_p
        if config.seed is not None:
            body["seed"] = config.seed
        if config.reasoning_effort is not None:
            body["reasoning_effort"] = config.reasoning_effort
        body["max_tokens"] = config.max_tokens
        extra_body: dict[str, Any] = {}
        if config.provider_only is not None:
            extra_body["provider"] = {"only": config.provider_only}
        thinking = self._resolved_thinking(request.enable_thinking)
        if thinking is not None:
            extra_body["chat_template_kwargs"] = {"enable_thinking": thinking}
        if extra_body:
            body["extra_body"] = extra_body

        if request.tools:
            body["tools"] = request.tools
            body["tool_choice"] = "auto"
        return body

    def _context_window_error(
        self, exc: BadRequestError
    ) -> ContextWindowExceededError | None:
        message = exc.message
        if (
            exc.code != "context_length_exceeded"
            and "context length" not in message.lower()
        ):
            return None
        limit_match = self._CTX_LIMIT_RE.search(message)
        requested_match = self._CTX_TOTAL_RE.search(message)
        input_tokens_match = self._CTX_INPUT_RE.search(message)
        return ContextWindowExceededError(
            message,
            limit=int(limit_match.group(1)) if limit_match else None,
            requested=int(requested_match.group(1)) if requested_match else None,
            input_tokens=int(input_tokens_match.group(1))
            if input_tokens_match
            else None,
        )

    async def _completion_from_stream(
        self, stream: Any, *, thinking_budget: int | None
    ) -> Completion:
        # SDK parsers reject finish_reason="length"; accumulate manually because
        # policy uses it for recovery.
        stream_state = ChatCompletionStreamState()
        watch = (
            _ReasoningBudgetWatch(
                budget=thinking_budget,
                token_counter=self._budget_counter,
            )
            if thinking_budget is not None
            else None
        )
        async for chunk in stream:
            try:
                stream_state.handle_chunk(chunk)
            except IndexError as exc:
                raise MalformedProviderStreamError(
                    "malformed tool_call stream: out-of-range tool_call index"
                ) from exc
            delta = chunk.choices[0].delta if chunk.choices else None
            if watch is not None and delta is not None and watch.overrun(delta):
                return self._budget_aborted_completion(watch.reasoning)
        response = stream_state.current_completion_snapshot
        choice = response.choices[0]
        message = choice.message
        # OpenRouter/upstreams may omit the best-effort usage chunk.
        usage = response.usage
        completion_details = (
            usage.completion_tokens_details if usage is not None else None
        )
        prompt_details = usage.prompt_tokens_details if usage is not None else None
        return Completion(
            tool_calls=tuple(
                ToolCall(
                    name=raw_call.function.name,
                    arguments=raw_call.function.arguments or "{}",
                )
                for raw_call in message.tool_calls or ()
            ),
            content=message.content,
            finish_reason=choice.finish_reason,
            usage=Usage(
                prompt_tokens=usage.prompt_tokens if usage is not None else None,
                completion_tokens=usage.completion_tokens
                if usage is not None
                else None,
                reasoning_tokens=completion_details.reasoning_tokens
                if completion_details is not None
                else None,
                cached_input_tokens=prompt_details.cached_tokens
                if prompt_details is not None
                else None,
            ),
            reasoning_content=_reasoning_extension_field(message),
            response={
                "id": response.id,
                "model": response.model,
                "system_fingerprint": response.system_fingerprint,
            },
        )
