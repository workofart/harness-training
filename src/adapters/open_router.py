from __future__ import annotations

import asyncio
import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from openrouter import OpenRouter as OpenRouterClient
from openrouter import errors
from dotenv import load_dotenv

from src.adapters.llm_base import (
    BaseLlm,
    LlmCompletion,
    LlmToolCall,
    LlmUsage,
    ReasoningEffort,
    _int_or_none,
)
from src.serialization import json_safe

if TYPE_CHECKING:
    from src.harness.config import OpenRouterConfig


OPENROUTER_MAX_ATTEMPTS = 3
OPENROUTER_TIMEOUT_GRACE_SECONDS = 5.0

logger = logging.getLogger(__name__)


class OpenRouterEmbeddedProviderError(RuntimeError):
    def __init__(self, *, status_code: int, message: str) -> None:
        super().__init__(f"OpenRouter embedded provider error {status_code}: {message}")
        self.status_code = status_code


def load_openrouter_api_key(
    *,
    dotenv_path: str | Path = ".env",
) -> str:
    load_dotenv(dotenv_path=dotenv_path, override=True)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")
    return api_key


def _normalize_openrouter_model_name(model_name: str) -> str:
    return model_name.removeprefix("openrouter/")


@lru_cache(maxsize=64)
def _fetch_openrouter_model_endpoints(
    *,
    base_url: str,
    model_name: str,
) -> tuple[dict[str, Any], ...]:
    model_id = _normalize_openrouter_model_name(model_name)
    author, slug = model_id.split("/", maxsplit=1)
    try:
        with OpenRouterClient(server_url=base_url, timeout_ms=10_000) as client:
            payload = client.endpoints.list(author=author, slug=slug)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load OpenRouter endpoint metadata for model {model_name!r}."
        ) from exc

    return tuple(
        endpoint.model_dump(mode="json") for endpoint in payload.data.endpoints
    )


def _is_retryable_openrouter_error(exc: Exception) -> bool:
    return isinstance(
        exc,
        (
            TimeoutError,
            httpx.TimeoutException,
            httpx.TransportError,
            errors.NoResponseError,
            errors.BadGatewayResponseError,
            errors.EdgeNetworkTimeoutResponseError,
            errors.InternalServerResponseError,
            errors.OpenRouterDefaultError,
            errors.ProviderOverloadedResponseError,
            errors.RequestTimeoutResponseError,
            errors.ServiceUnavailableResponseError,
            errors.TooManyRequestsResponseError,
            OpenRouterEmbeddedProviderError,
        ),
    ) and getattr(exc, "status_code", 500) in {408, 429, 500, 502, 503, 524, 529}


def _extract_reasoning_payload(message: Any) -> Any:
    if message is None:
        return None
    reasoning_content = _field(message, "reasoning_content")
    if reasoning_content is not None:
        return json_safe(reasoning_content)

    thinking_blocks = _field(message, "thinking_blocks")
    if thinking_blocks is not None:
        return json_safe(thinking_blocks)

    provider_specific_fields = _field(message, "provider_specific_fields")
    if isinstance(provider_specific_fields, dict):
        for key in ("reasoning_content", "thinking", "reasoning"):
            if key in provider_specific_fields:
                return json_safe(provider_specific_fields[key])

    return None


def _build_response_envelope(completion: Any, message: Any) -> dict[str, Any]:
    """Build the slim JSON-safe response envelope written under the trace
    event's `response` field. Keeps only the bits a replay needs: the
    message body (content / tool_calls / reasoning), plus the
    provider-side ids when present.
    """
    if message is None:
        return json_safe(completion) if isinstance(completion, dict) else {}

    raw_message = json_safe(message)
    if not isinstance(raw_message, dict):
        return {}

    slim_message: dict[str, Any] = {}
    for key in ("content", "tool_calls"):
        value = raw_message.get(key)
        if value is not None:
            slim_message[key] = value

    reasoning = raw_message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        slim_message["reasoning"] = reasoning
    else:
        details = raw_message.get("reasoning_details")
        if isinstance(details, list):
            text = "".join(
                part.get("text", "")
                for part in details
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
            if text:
                slim_message["reasoning"] = text

    reasoning_content = raw_message.get("reasoning_content")
    if reasoning_content is not None and "reasoning" not in slim_message:
        slim_message["reasoning_content"] = reasoning_content

    response: dict[str, Any] = {"message": slim_message}
    for envelope_key in ("id", "created", "system_fingerprint"):
        value = getattr(completion, envelope_key, None)
        if value is None and isinstance(completion, dict):
            value = completion.get(envelope_key)
        if value is not None:
            response[envelope_key] = json_safe(value)
    return response


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _to_llm_completion(completion: Any) -> LlmCompletion:
    """Normalize a provider response (real ChatResult, dict, or test fake)
    into the typed LlmCompletion the harness and trace consume.
    """
    choices = _field(completion, "choices") or []
    if not choices:
        return LlmCompletion()
    choice = choices[0]

    message = _field(choice, "message")
    finish_reason = _field(choice, "finish_reason")
    if not isinstance(finish_reason, str):
        finish_reason = None

    tool_calls: list[LlmToolCall] = []
    for raw_call in _field(message, "tool_calls") or []:
        function = _field(raw_call, "function")
        if function is None:
            continue
        name = _field(function, "name")
        if not isinstance(name, str):
            continue
        arguments = _field(function, "arguments")
        if not isinstance(arguments, str) or not arguments:
            arguments = "{}"
        tool_calls.append(LlmToolCall(name=name, arguments=arguments))

    content = _field(message, "content")
    if not isinstance(content, str):
        content = None

    usage_obj = _field(completion, "usage")
    completion_details = _field(usage_obj, "completion_tokens_details")
    prompt_details = _field(usage_obj, "prompt_tokens_details")
    usage = LlmUsage(
        prompt_tokens=_int_or_none(_field(usage_obj, "prompt_tokens")),
        completion_tokens=_int_or_none(_field(usage_obj, "completion_tokens")),
        reasoning_tokens=_int_or_none(_field(completion_details, "reasoning_tokens")),
        cached_input_tokens=_int_or_none(_field(prompt_details, "cached_tokens")),
    )

    return LlmCompletion(
        tool_calls=tuple(tool_calls),
        content=content,
        finish_reason=finish_reason,
        usage=usage,
        reasoning_content=_extract_reasoning_payload(message),
        response=_build_response_envelope(completion, message),
    )


def _embedded_provider_error(completion: Any) -> OpenRouterEmbeddedProviderError | None:
    try:
        message = completion.choices[0].message
        provider_specific_fields = message.provider_specific_fields
    except (AttributeError, IndexError, TypeError):
        return None
    if not isinstance(provider_specific_fields, dict):
        return None
    error = provider_specific_fields.get("error")
    if not isinstance(error, dict):
        return None
    status_code = error.get("code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    message = error.get("message", "provider returned an embedded error")
    if not isinstance(message, str):
        message = "provider returned an embedded error"
    return OpenRouterEmbeddedProviderError(status_code=status_code, message=message)


class OpenRouter(BaseLlm):
    def __init__(
        self,
        *,
        config: OpenRouterConfig,
        api_key: str,
    ) -> None:
        self.config = config
        self._max_context_length: int | None = None
        self._client = OpenRouterClient(
            api_key=api_key,
            server_url=config.base_url,
            timeout_ms=int(config.timeout_seconds * 1000),
        )

    @property
    def max_context_length(self) -> int:
        if self._max_context_length is not None:
            return self._max_context_length
        endpoints = _fetch_openrouter_model_endpoints(
            base_url=self.config.base_url,
            model_name=self.config.model_name,
        )
        provider = self.config.provider_kwargs.provider
        provider_order = () if provider is None else provider.order
        if not provider_order:
            selected = tuple(endpoints)
        else:
            selected = tuple(
                endpoint
                for endpoint in endpoints
                if any(
                    (normalized_tag := endpoint["tag"].lower()) == candidate.lower()
                    or normalized_tag.startswith(f"{candidate.lower()}/")
                    for candidate in provider_order
                )
            )
        if not selected:
            available_tags = sorted(endpoint["tag"] for endpoint in endpoints)
            raise RuntimeError(
                "No OpenRouter endpoints matched configured provider order "
                f"{list(provider_order)!r}. Available endpoint tags: {available_tags!r}."
            )
        self._max_context_length = min(
            endpoint["context_length"]
            if endpoint["max_prompt_tokens"] is None
            else endpoint["max_prompt_tokens"]
            for endpoint in selected
        )
        return self._max_context_length

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> LlmCompletion:
        async def attempt_completion(
            request_messages: list[dict[str, Any]],
        ) -> Any:
            completion_kwargs: dict[str, Any] = dict(
                self.config.provider_kwargs.extra_body
            )
            if tools:
                completion_kwargs["tools"] = tools
            provider = self.config.provider_kwargs.provider
            provider_kwargs = {}
            if provider is not None:
                provider_kwargs.update(provider.model_dump(mode="json"))
            if self.config.provider_kwargs.require_parameters is not None:
                provider_kwargs["require_parameters"] = (
                    self.config.provider_kwargs.require_parameters
                )
            if provider_kwargs:
                completion_kwargs["provider"] = provider_kwargs
            effective_reasoning_effort = (
                self.config.reasoning_effort
                if reasoning_effort is None
                else reasoning_effort
            )
            if (
                reasoning_effort is not None
                or "reasoning_effort" in self.config.model_fields_set
            ):
                completion_kwargs["reasoning"] = {"effort": effective_reasoning_effort}
            request_kwargs = {
                "model": _normalize_openrouter_model_name(self.config.model_name),
                "messages": request_messages,
                "stream": False,
            }
            if "max_output_tokens" in self.config.model_fields_set:
                request_kwargs["max_tokens"] = self.config.max_output_tokens
            if "temperature" in self.config.model_fields_set:
                request_kwargs["temperature"] = self.config.temperature
            if "top_p" in self.config.model_fields_set:
                request_kwargs["top_p"] = self.config.top_p
            if "seed" in self.config.model_fields_set:
                request_kwargs["seed"] = self.config.seed
            if "service_tier" in self.config.model_fields_set:
                request_kwargs["service_tier"] = self.config.service_tier
            if "timeout_seconds" in self.config.model_fields_set:
                request_kwargs["timeout_ms"] = int(self.config.timeout_seconds * 1000)
            request_kwargs.update(completion_kwargs)
            request_kwargs = {
                key: value for key, value in request_kwargs.items() if value is not None
            }
            request_kwargs["retries"] = None
            try:
                return await asyncio.wait_for(
                    self._client.chat.send_async(**request_kwargs),
                    timeout=(
                        self.config.timeout_seconds + OPENROUTER_TIMEOUT_GRACE_SECONDS
                    ),
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    "OpenRouter completion timed out after "
                    f"{self.config.timeout_seconds} seconds."
                ) from exc

        async def run_completion() -> LlmCompletion:
            for attempt in range(1, OPENROUTER_MAX_ATTEMPTS + 1):
                attempt_started_at = time.perf_counter()
                try:
                    completion = await attempt_completion(list(messages))
                    if embedded_error := _embedded_provider_error(completion):
                        raise embedded_error
                except Exception as exc:
                    elapsed_sec = time.perf_counter() - attempt_started_at
                    retryable = _is_retryable_openrouter_error(exc)
                    logger.warning(
                        "OpenRouter attempt failed attempt=%s/%s elapsed_sec=%.3f "
                        "retryable=%s tools=%s reasoning_effort=%s error_type=%s error=%s",
                        attempt,
                        OPENROUTER_MAX_ATTEMPTS,
                        elapsed_sec,
                        retryable,
                        tools is not None,
                        self.config.reasoning_effort
                        if reasoning_effort is None
                        else reasoning_effort,
                        type(exc).__name__,
                        exc,
                    )
                    # Terminal failures propagate from the attempt that observed them.
                    if not retryable or attempt == OPENROUTER_MAX_ATTEMPTS:
                        raise
                    await asyncio.sleep(float(attempt))
                    continue

                elapsed_sec = time.perf_counter() - attempt_started_at
                usage = getattr(completion, "usage", None)
                completion_details = (
                    None
                    if usage is None
                    else getattr(usage, "completion_tokens_details", None)
                )
                logger.info(
                    "OpenRouter attempt completed attempt=%s/%s elapsed_sec=%.3f "
                    "tools=%s reasoning_effort=%s finish_reason=%s prompt_tokens=%s "
                    "completion_tokens=%s reasoning_tokens=%s",
                    attempt,
                    OPENROUTER_MAX_ATTEMPTS,
                    elapsed_sec,
                    tools is not None,
                    self.config.reasoning_effort
                    if reasoning_effort is None
                    else reasoning_effort,
                    getattr(completion.choices[0], "finish_reason", None),
                    None if usage is None else getattr(usage, "prompt_tokens", None),
                    None
                    if usage is None
                    else getattr(usage, "completion_tokens", None),
                    None
                    if completion_details is None
                    else getattr(completion_details, "reasoning_tokens", None),
                )

                return _to_llm_completion(completion)

        return await run_completion()

    async def close(self) -> None:
        await self._client.__aexit__(None, None, None)
        self._client.__exit__(None, None, None)
