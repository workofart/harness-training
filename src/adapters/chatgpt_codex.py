from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from src.adapters.infra_retry import INFRA_RETRY_BUDGET, retry_transient
from src.adapters.llm_base import (
    BaseLlm,
    LlmCompletion,
    LlmToolCall,
    LlmUsage,
    ReasoningEffort,
    _int_or_none,
)

if TYPE_CHECKING:
    from src.harness.config import ChatGptCodexConfig


CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_JWT_AUTH_CLAIM = "https://api.openai.com/auth"
TOKEN_REFRESH_SKEW_SECONDS = 60

RETRYABLE_STATUS_CODES = frozenset({408, 500, 502, 503, 524, 529})

logger = logging.getLogger(__name__)


class ChatGptCodexError(RuntimeError):
    pass


class ChatGptCodexUnauthorizedError(ChatGptCodexError):
    pass


class ChatGptCodexResponseError(ChatGptCodexError):
    """A Codex completion returned a non-success HTTP status other than 401."""

    def __init__(self, *, status_code: int, message: str) -> None:
        super().__init__(f"Codex completion failed: HTTP {status_code}: {message}")
        self.status_code = status_code


def _is_retryable_infra_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, ChatGptCodexResponseError):
        return exc.status_code in RETRYABLE_STATUS_CODES
    return False


@dataclass(frozen=True, slots=True)
class _CodexAuth:
    access_token: str
    id_token: str
    refresh_token: str
    account_id: str


class _CodexAuthStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = _default_auth_path() if path is None else path

    def load(self) -> _CodexAuth:
        payload = json.loads(self.path.expanduser().read_text())
        tokens = payload["tokens"]
        access_token = tokens["access_token"]
        id_token = tokens["id_token"]
        refresh_token = tokens["refresh_token"]
        account_id = (
            tokens.get("account_id")
            or _extract_account_id(id_token)
            or _extract_account_id(access_token)
        )
        if not isinstance(account_id, str) or not account_id:
            raise ChatGptCodexError("Codex auth.json is missing chatgpt_account_id")
        return _CodexAuth(
            access_token=access_token,
            id_token=id_token,
            refresh_token=refresh_token,
            account_id=account_id,
        )

    async def refresh(self, *, http_client: httpx.AsyncClient) -> _CodexAuth:
        current = self.load()
        response = await http_client.post(
            CODEX_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": current.refresh_token,
                "client_id": CODEX_CLIENT_ID,
            },
        )
        if response.status_code >= 400:
            raise ChatGptCodexError(
                f"Codex token refresh failed: HTTP {response.status_code}"
            )
        refreshed = response.json()
        auth = _CodexAuth(
            access_token=refreshed["access_token"],
            id_token=refreshed.get("id_token", current.id_token),
            refresh_token=refreshed.get("refresh_token", current.refresh_token),
            account_id=(
                _extract_account_id(refreshed.get("id_token", current.id_token))
                or current.account_id
            ),
        )
        self._write(auth)
        return auth

    def _write(self, auth: _CodexAuth) -> None:
        path = self.path.expanduser()
        payload = json.loads(path.read_text())
        tokens = payload["tokens"]
        tokens["access_token"] = auth.access_token
        tokens["id_token"] = auth.id_token
        tokens["refresh_token"] = auth.refresh_token
        tokens["account_id"] = auth.account_id
        payload["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


class ChatGptCodex(BaseLlm):
    def __init__(
        self,
        *,
        config: ChatGptCodexConfig,
        auth_store: Any | None = None,
        http_client: Any | None = None,
    ) -> None:
        self.config = config
        self._auth_store = (
            _CodexAuthStore(
                None if config.auth_file is None else Path(config.auth_file)
            )
            if auth_store is None
            else auth_store
        )
        self._owns_client = http_client is None
        self._client = (
            httpx.AsyncClient(timeout=config.timeout_seconds)
            if http_client is None
            else http_client
        )

    @property
    def max_context_length(self) -> int:
        return self.config.max_context_length

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> LlmCompletion:
        body = _build_request_body(
            config=self.config,
            messages=messages,
            tools=tools,
            reasoning_effort=reasoning_effort,
        )

        def _log_infra_retry(retry: int, exc: Exception) -> None:
            logger.warning(
                "Codex completion failed; retrying (%d/%d): %s: %s",
                retry,
                INFRA_RETRY_BUDGET,
                type(exc).__name__,
                exc,
            )

        return await retry_transient(
            lambda: self._complete_with_auth(body=body),
            is_transient=_is_retryable_infra_error,
            on_retry=_log_infra_retry,
        )

    async def _complete_with_auth(self, *, body: dict[str, Any]) -> LlmCompletion:
        """One completion attempt. Refreshes the access token once on a 401 and
        retries that single attempt; a repeated 401 is terminal. Transport
        failures (including during a proactive token refresh in `_current_auth`)
        propagate so the bounded retry in `complete()` can handle them."""
        auth = await self._current_auth()
        try:
            return await self._post_completion(body=body, auth=auth)
        except ChatGptCodexUnauthorizedError:
            auth = await self._auth_store.refresh(http_client=self._client)
            return await self._post_completion(body=body, auth=auth)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _current_auth(self) -> _CodexAuth:
        auth = self._auth_store.load()
        if _token_expires_soon(auth.access_token):
            return await self._auth_store.refresh(http_client=self._client)
        return auth

    async def _post_completion(
        self,
        *,
        body: dict[str, Any],
        auth: _CodexAuth,
    ) -> LlmCompletion:
        async with self._client.stream(
            "POST",
            _codex_responses_url(self.config.base_url),
            json=body,
            headers=_build_headers(auth, originator="codex_cli_rs"),
        ) as response:
            if response.status_code == 401:
                raise ChatGptCodexUnauthorizedError("Codex backend returned HTTP 401")
            if response.status_code >= 400:
                text = await _read_response_text(response)
                raise ChatGptCodexResponseError(
                    status_code=response.status_code,
                    message=text,
                )
            return _completion_from_events(
                [event async for event in _iter_sse_json(response)]
            )


def _default_auth_path() -> Path:
    return Path(os.getenv("CODEX_HOME", "~/.codex")).expanduser() / "auth.json"


def _build_request_body(
    *,
    config: ChatGptCodexConfig,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    reasoning_effort: ReasoningEffort | None,
) -> dict[str, Any]:
    instructions, input_items = _convert_messages(messages)
    body: dict[str, Any] = {
        "model": config.model_name,
        "store": False,
        "stream": True,
        "input": input_items,
        "include": ["reasoning.encrypted_content"],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "text": {"verbosity": config.text_verbosity},
    }
    if instructions:
        body["instructions"] = instructions
    effective_reasoning = (
        config.reasoning_effort if reasoning_effort is None else reasoning_effort
    )
    if effective_reasoning != "none":
        body["reasoning"] = {"effort": effective_reasoning}
    if tools:
        body["tools"] = [_convert_tool(tool) for tool in tools]
    if config.service_tier is not None:
        body["service_tier"] = config.service_tier
    if config.prompt_cache_key is not None:
        body["prompt_cache_key"] = config.prompt_cache_key
    return body


def _convert_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "system":
            instructions.append(_message_text(message))
            continue
        if role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": _message_text(message),
                }
            )
            continue
        if role == "assistant" and message.get("tool_calls"):
            content = _message_text(message)
            if content:
                input_items.append(_message_item("assistant", content))
            for raw_call in message["tool_calls"]:
                function = raw_call["function"]
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": raw_call["id"],
                        "name": function["name"],
                        "arguments": function.get("arguments") or "{}",
                    }
                )
            continue
        input_items.append(_message_item(role, _message_text(message)))
    return "\n\n".join(instructions) or None, input_items


def _message_item(role: str, text: str) -> dict[str, Any]:
    content_type = "output_text" if role == "assistant" else "input_text"
    return {
        "type": "message",
        "role": role,
        "content": [{"type": content_type, "text": text}],
    }


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return str(content)


def _convert_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool["function"]
    converted = {
        "type": "function",
        "name": function["name"],
        "description": function.get("description", ""),
        "parameters": function["parameters"],
        "strict": False,
    }
    return {key: value for key, value in converted.items() if value != ""}


def _build_headers(auth: _CodexAuth, *, originator: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {auth.access_token}",
        "chatgpt-account-id": auth.account_id,
        "originator": originator,
        "OpenAI-Beta": "responses=experimental",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }


def _codex_responses_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/codex/responses"


async def _iter_sse_json(response: Any):
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data_lines:
                event = _parse_sse_data(data_lines)
                if event is not None:
                    yield event
            data_lines = []
            continue
        if line.startswith("data:"):
            data = line.removeprefix("data:").strip()
            if not data_lines:
                event = _parse_sse_data([data])
                if event is not None:
                    yield event
                continue
            data_lines.append(data)
    if data_lines:
        event = _parse_sse_data(data_lines)
        if event is not None:
            yield event


def _parse_sse_data(data_lines: list[str]) -> dict[str, Any] | None:
    data = "\n".join(data_lines)
    if data == "[DONE]":
        return None
    return json.loads(data)


def _completion_from_events(events: list[dict[str, Any]]) -> LlmCompletion:
    final_response: dict[str, Any] | None = None
    calls: dict[str, dict[str, Any]] = {}
    content_parts: list[str] = []
    finish_reason: str | None = None

    for event in events:
        event_type = event.get("type")
        if event_type == "response.output_item.added":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call":
                item_id = str(item.get("id") or item.get("call_id"))
                calls[item_id] = {
                    "name": item.get("name"),
                    "arguments": item.get("arguments") or "",
                    "call_id": item.get("call_id"),
                }
        elif event_type == "response.function_call_arguments.delta":
            item_id = str(event.get("item_id"))
            if item_id in calls and isinstance(event.get("delta"), str):
                calls[item_id]["arguments"] += event["delta"]
        elif event_type == "response.function_call_arguments.done":
            item_id = str(event.get("item_id"))
            if item_id in calls and isinstance(event.get("arguments"), str):
                calls[item_id]["arguments"] = event["arguments"]
        elif event_type == "response.output_text.delta":
            if isinstance(event.get("delta"), str):
                content_parts.append(event["delta"])
        elif event_type == "response.output_item.done":
            _merge_done_item(event.get("item"), calls, content_parts)
        elif event_type == "response.completed":
            response = event.get("response")
            if isinstance(response, dict):
                final_response = response
                finish_reason = "completed"

    if final_response is not None:
        final_calls, final_content = _extract_final_output(final_response)
        if final_calls:
            calls = final_calls
        if final_content:
            content_parts = [final_content]

    tool_calls = tuple(
        LlmToolCall(
            name=call["name"],
            arguments=call["arguments"] or "{}",
        )
        for call in calls.values()
        if isinstance(call.get("name"), str)
    )
    content = "".join(content_parts) or None
    return LlmCompletion(
        tool_calls=tool_calls,
        content=content,
        finish_reason=finish_reason,
        usage=_extract_usage(final_response),
        reasoning_content=_extract_reasoning(final_response),
        response=_response_envelope(final_response),
    )


def _merge_done_item(
    item: Any,
    calls: dict[str, dict[str, Any]],
    content_parts: list[str],
) -> None:
    if not isinstance(item, dict):
        return
    if item.get("type") == "function_call":
        item_id = str(item.get("id") or item.get("call_id"))
        calls[item_id] = {
            "name": item.get("name"),
            "arguments": item.get("arguments") or "{}",
            "call_id": item.get("call_id"),
        }
    elif item.get("type") == "message":
        text = _text_from_content(item.get("content"))
        if text:
            content_parts.append(text)


def _extract_final_output(
    response: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], str | None]:
    calls: dict[str, dict[str, Any]] = {}
    content_parts: list[str] = []
    output = response.get("output")
    if not isinstance(output, list):
        return calls, None
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            item_id = str(item.get("id") or item.get("call_id"))
            calls[item_id] = {
                "name": item.get("name"),
                "arguments": item.get("arguments") or "{}",
                "call_id": item.get("call_id"),
            }
        elif item.get("type") == "message":
            text = _text_from_content(item.get("content"))
            if text:
                content_parts.append(text)
    return calls, "\n".join(content_parts) or None


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        part["text"]
        for part in content
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )


def _extract_usage(response: dict[str, Any] | None) -> LlmUsage:
    if response is None:
        return LlmUsage()
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return LlmUsage()
    input_details = usage.get("input_tokens_details")
    if not isinstance(input_details, dict):
        input_details = {}
    output_details = usage.get("output_tokens_details")
    if not isinstance(output_details, dict):
        output_details = {}
    return LlmUsage(
        prompt_tokens=_int_or_none(usage.get("input_tokens")),
        completion_tokens=_int_or_none(usage.get("output_tokens")),
        reasoning_tokens=_int_or_none(output_details.get("reasoning_tokens")),
        cached_input_tokens=_int_or_none(input_details.get("cached_tokens")),
    )


def _extract_reasoning(response: dict[str, Any] | None) -> Any | None:
    if response is None:
        return None
    output = response.get("output")
    if not isinstance(output, list):
        return None
    reasoning = [item for item in output if _is_reasoning_item(item)]
    return reasoning or None


def _response_envelope(response: dict[str, Any] | None) -> dict[str, Any]:
    if response is None:
        return {}
    envelope: dict[str, Any] = {}
    for key in ("id", "model", "output", "usage"):
        if key in response:
            envelope[key] = response[key]
    return envelope


def _is_reasoning_item(item: Any) -> bool:
    return isinstance(item, dict) and item.get("type") == "reasoning"


def _token_expires_soon(token: str) -> bool:
    exp = _jwt_payload(token).get("exp")
    return (
        isinstance(exp, (int, float))
        and exp <= time.time() + TOKEN_REFRESH_SKEW_SECONDS
    )


def _extract_account_id(token: str) -> str | None:
    claim = _jwt_payload(token).get(CODEX_JWT_AUTH_CLAIM)
    if not isinstance(claim, dict):
        return None
    account_id = claim.get("chatgpt_account_id")
    return account_id if isinstance(account_id, str) else None


def _jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    padded = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        payload = base64.urlsafe_b64decode(padded)
    except ValueError:
        return {}
    decoded = json.loads(payload)
    return decoded if isinstance(decoded, dict) else {}


async def _read_response_text(response: Any) -> str:
    if hasattr(response, "aread"):
        body = await response.aread()
        if isinstance(body, bytes):
            return body.decode(errors="replace")
    return ""
