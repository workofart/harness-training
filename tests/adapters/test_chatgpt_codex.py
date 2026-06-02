from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

import src.adapters.chatgpt_codex as chatgpt_codex_module
from src.adapters.infra_retry import INFRA_RETRY_BUDGET
from src.harness.config import ChatGptCodexConfig


def _jwt(payload: dict[str, Any]) -> str:
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode()
    return "header." + encoded.rstrip("=") + ".sig"


def test_codex_auth_reads_account_id_from_id_token(tmp_path: Path):
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": _jwt({"exp": 4_102_444_800}),
                    "id_token": _jwt(
                        {
                            "https://api.openai.com/auth": {
                                "chatgpt_account_id": "acct_123"
                            }
                        }
                    ),
                    "refresh_token": "refresh-token",
                },
            }
        )
    )

    auth = chatgpt_codex_module._CodexAuthStore(auth_path).load()

    assert auth.account_id == "acct_123"
    assert auth.access_token.startswith("header.")


def test_build_request_body_converts_chat_messages_and_tools():
    body = chatgpt_codex_module._build_request_body(
        config=ChatGptCodexConfig(
            model_name="gpt-5.5",
            max_context_length=200_000,
            reasoning_effort="low",
        ),
        messages=[
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "read the file"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0001",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"README.md"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0001",
                "content": "file contents",
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
        reasoning_effort=None,
    )

    assert body["model"] == "gpt-5.5"
    assert body["instructions"] == "system rules"
    assert body["store"] is False
    assert body["stream"] is True
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["reasoning"] == {"effort": "low"}
    assert body["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "read the file"}],
        },
        {
            "type": "function_call",
            "call_id": "call_0001",
            "name": "read_file",
            "arguments": '{"path":"README.md"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_0001",
            "output": "file contents",
        },
    ]
    assert body["tools"] == [
        {
            "type": "function",
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            "strict": False,
        }
    ]


def test_complete_posts_codex_sse_and_normalizes_tool_call():
    captured: dict[str, Any] = {}
    lines = [
        'data: {"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","call_id":"call_x","name":"read_file","arguments":""}}',
        'data: {"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\\"path\\""}',
        'data: {"type":"response.function_call_arguments.done","item_id":"fc_1","arguments":"{\\"path\\":\\"README.md\\"}"}',
        'data: {"type":"response.completed","response":{"id":"resp_1","usage":{"input_tokens":10,"output_tokens":5,"input_tokens_details":{"cached_tokens":3},"output_tokens_details":{"reasoning_tokens":2}}}}',
        "data: [DONE]",
    ]

    class _FakeResponse:
        status_code = 200

        async def aiter_lines(self):
            for line in lines:
                yield line

    class _FakeStream:
        async def __aenter__(self):
            return _FakeResponse()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class _FakeClient:
        def stream(self, method, url, *, json, headers):
            captured["method"] = method
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _FakeStream()

        async def aclose(self):
            captured["closed"] = True

    class _FakeAuthStore:
        def load(self):
            return chatgpt_codex_module._CodexAuth(
                access_token=_jwt({"exp": 4_102_444_800}),
                id_token=_jwt({}),
                refresh_token="refresh-token",
                account_id="acct_123",
            )

        async def refresh(self, *, http_client):
            raise AssertionError("fresh token should not refresh")

    llm = chatgpt_codex_module.ChatGptCodex(
        config=ChatGptCodexConfig(
            model_name="gpt-5.5",
            max_context_length=200_000,
            timeout_seconds=10.0,
        ),
        auth_store=_FakeAuthStore(),
        http_client=_FakeClient(),
    )

    completion = asyncio.run(
        llm.complete(messages=[{"role": "user", "content": "read README"}])
    )

    assert captured["method"] == "POST"
    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
    assert captured["headers"]["Authorization"].startswith("Bearer header.")
    assert captured["headers"]["chatgpt-account-id"] == "acct_123"
    assert captured["headers"]["OpenAI-Beta"] == "responses=experimental"
    assert captured["json"]["store"] is False
    assert completion.tool_calls[0].name == "read_file"
    assert completion.tool_calls[0].arguments == '{"path":"README.md"}'
    assert completion.usage.prompt_tokens == 10
    assert completion.usage.completion_tokens == 5
    assert completion.usage.cached_input_tokens == 3
    assert completion.usage.reasoning_tokens == 2


# --- Transient-failure retry policy -----------------------------------------

_GOOD_SSE_LINES = [
    'data: {"type":"response.completed","response":{"id":"resp_ok",'
    '"output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}]}}',
    "data: [DONE]",
]


class _Response:
    """A streamed Codex response: a status code plus SSE lines or error body."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        lines: list[str] | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.status_code = status_code
        self._lines = lines if lines is not None else _GOOD_SSE_LINES
        self.headers = headers or {}
        self.request = httpx.Request(
            "POST", "https://chatgpt.com/backend-api/codex/responses"
        )

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return "\n".join(self._lines).encode()


class _Stream:
    """Async context manager mimicking `httpx.AsyncClient.stream`. Either raises
    a transport error on entry or hands back a `_Response`."""

    def __init__(
        self,
        *,
        response: _Response | None = None,
        enter_error: Exception | None = None,
    ):
        self._response = response
        self._enter_error = enter_error

    async def __aenter__(self) -> _Response:
        if self._enter_error is not None:
            raise self._enter_error
        assert self._response is not None
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _SequencedClient:
    """Returns one queued outcome per `stream()` call. A queued `Exception` is
    raised on stream entry (transport failure); a `_Response` is served."""

    def __init__(self, steps: list[Exception | _Response]):
        self._steps = list(steps)
        self.stream_calls = 0

    def stream(self, method, url, *, json, headers) -> _Stream:
        del method, url, json, headers
        self.stream_calls += 1
        step = self._steps.pop(0)
        if isinstance(step, Exception):
            return _Stream(enter_error=step)
        return _Stream(response=step)

    async def aclose(self) -> None:
        return None


class _StaticAuthStore:
    """Loads a never-expiring token and counts refreshes."""

    def __init__(self):
        self.refresh_count = 0

    def load(self):
        return chatgpt_codex_module._CodexAuth(
            access_token=_jwt({"exp": 4_102_444_800}),
            id_token=_jwt({}),
            refresh_token="refresh-token",
            account_id="acct_123",
        )

    async def refresh(self, *, http_client):
        del http_client
        self.refresh_count += 1
        return self.load()


def _make_llm(*, client: _SequencedClient, auth_store: _StaticAuthStore):
    return chatgpt_codex_module.ChatGptCodex(
        config=ChatGptCodexConfig(
            model_name="gpt-5.5",
            max_context_length=200_000,
            timeout_seconds=10.0,
        ),
        auth_store=auth_store,
        http_client=client,
    )


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    """Collapse the bounded-retry backoff so tests don't actually sleep."""

    async def _sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", _sleep)


def test_complete_retries_transport_error_then_succeeds():
    client = _SequencedClient([httpx.RemoteProtocolError("peer reset"), _Response()])
    llm = _make_llm(client=client, auth_store=_StaticAuthStore())

    completion = asyncio.run(llm.complete(messages=[{"role": "user", "content": "go"}]))

    assert completion.content == "ok"
    assert client.stream_calls == 2


def test_complete_retries_retryable_status_then_succeeds():
    client = _SequencedClient(
        [_Response(status_code=503, lines=["overloaded"]), _Response()]
    )
    llm = _make_llm(client=client, auth_store=_StaticAuthStore())

    completion = asyncio.run(llm.complete(messages=[{"role": "user", "content": "go"}]))

    assert completion.content == "ok"
    assert client.stream_calls == 2


def test_complete_retries_http_429_then_succeeds():
    client = _SequencedClient(
        [
            _Response(status_code=429, lines=['{"detail":"Rate limit exceeded"}']),
            _Response(),
        ]
    )
    llm = _make_llm(client=client, auth_store=_StaticAuthStore())

    completion = asyncio.run(llm.complete(messages=[{"role": "user", "content": "go"}]))

    assert completion.content == "ok"
    assert client.stream_calls == 2


def test_complete_uses_retry_after_header_for_retry_delay(monkeypatch):
    sleeps: list[float] = []

    async def _record_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", _record_sleep)
    client = _SequencedClient(
        [
            _Response(
                status_code=429,
                lines=['{"detail":"Rate limit exceeded"}'],
                headers={"Retry-After": "7"},
            ),
            _Response(),
        ]
    )
    llm = _make_llm(client=client, auth_store=_StaticAuthStore())

    completion = asyncio.run(llm.complete(messages=[{"role": "user", "content": "go"}]))

    assert completion.content == "ok"
    assert sleeps == [7.0]


def test_complete_exhausts_budget_on_persistent_transport_error():
    client = _SequencedClient(
        [httpx.ReadError("eof") for _ in range(INFRA_RETRY_BUDGET + 1)]
    )
    llm = _make_llm(client=client, auth_store=_StaticAuthStore())

    with pytest.raises(httpx.ReadError):
        asyncio.run(llm.complete(messages=[{"role": "user", "content": "go"}]))

    assert client.stream_calls == INFRA_RETRY_BUDGET + 1


def test_complete_does_not_retry_http_400():
    client = _SequencedClient([_Response(status_code=400, lines=["bad request"])])
    llm = _make_llm(client=client, auth_store=_StaticAuthStore())

    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        asyncio.run(llm.complete(messages=[{"role": "user", "content": "go"}]))

    assert excinfo.value.response.status_code == 400
    assert "Codex completion failed: HTTP 400: bad request" in str(excinfo.value)
    assert client.stream_calls == 1


def test_complete_refreshes_once_on_401_then_succeeds():
    client = _SequencedClient([_Response(status_code=401), _Response()])
    auth_store = _StaticAuthStore()
    llm = _make_llm(client=client, auth_store=auth_store)

    completion = asyncio.run(llm.complete(messages=[{"role": "user", "content": "go"}]))

    assert completion.content == "ok"
    assert auth_store.refresh_count == 1
    assert client.stream_calls == 2


def test_complete_repeated_401_is_terminal():
    client = _SequencedClient([_Response(status_code=401), _Response(status_code=401)])
    auth_store = _StaticAuthStore()
    llm = _make_llm(client=client, auth_store=auth_store)

    with pytest.raises(chatgpt_codex_module.ChatGptCodexUnauthorizedError):
        asyncio.run(llm.complete(messages=[{"role": "user", "content": "go"}]))

    # One refresh after the first 401; the second 401 is not retried.
    assert auth_store.refresh_count == 1
    assert client.stream_calls == 2


def _write_dead_auth_file(tmp_path: Path) -> Path:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": _jwt({"exp": 4_102_444_800}),
                    "id_token": _jwt(
                        {
                            "https://api.openai.com/auth": {
                                "chatgpt_account_id": "acct_123"
                            }
                        }
                    ),
                    "refresh_token": "dead-refresh-token",
                },
            }
        )
    )
    return auth_path


class _TokenResponse:
    def __init__(self, *, status_code: int):
        self.status_code = status_code

    def json(self):
        return {"error": "invalid_grant"}


class _RefreshClient:
    """Backend stream() always 401s (token rejected); the token-endpoint post()
    returns the configured status, simulating a refresh attempt against an
    expired/revoked refresh token."""

    def __init__(self, *, token_status: int):
        self._token_status = token_status
        self.stream_calls = 0
        self.post_calls = 0

    def stream(self, method, url, *, json, headers) -> _Stream:
        del method, url, json, headers
        self.stream_calls += 1
        return _Stream(response=_Response(status_code=401))

    async def post(self, url, *, data) -> _TokenResponse:
        del url, data
        self.post_calls += 1
        return _TokenResponse(status_code=self._token_status)

    async def aclose(self) -> None:
        return None


def test_credentials_expired_error_is_base_exception_not_exception():
    # Inherits BaseException (not Exception) on purpose: it must escape the
    # runner's per-trial and per-experiment `except Exception` containment so
    # the subprocess exits non-zero and the supervisor halts for `codex login`,
    # instead of finalizing a crash record and advancing into the same wall.
    err = chatgpt_codex_module.ChatGptCodexCredentialsExpiredError("x")
    assert isinstance(err, BaseException)
    assert not isinstance(err, Exception)


@pytest.mark.parametrize("token_status", [400, 401])
def test_refresh_raises_credentials_expired_on_dead_token(tmp_path: Path, token_status):
    # OAuth `invalid_grant` is HTTP 400; a 401 is `invalid_client`. Both mean
    # the stored credentials are dead and only re-login fixes them.
    store = chatgpt_codex_module._CodexAuthStore(_write_dead_auth_file(tmp_path))
    client = _RefreshClient(token_status=token_status)

    with pytest.raises(chatgpt_codex_module.ChatGptCodexCredentialsExpiredError):
        asyncio.run(store.refresh(http_client=client))

    assert client.post_calls == 1


def test_refresh_raises_generic_error_on_transient_status(tmp_path: Path):
    # A transient token-endpoint failure must NOT be treated as dead credentials
    # (that would hard-halt the loop on a recoverable blip).
    store = chatgpt_codex_module._CodexAuthStore(_write_dead_auth_file(tmp_path))
    client = _RefreshClient(token_status=503)

    with pytest.raises(chatgpt_codex_module.ChatGptCodexError) as excinfo:
        asyncio.run(store.refresh(http_client=client))

    assert not isinstance(
        excinfo.value, chatgpt_codex_module.ChatGptCodexCredentialsExpiredError
    )


def test_complete_halts_when_refresh_token_is_dead(tmp_path: Path):
    # End-to-end repro of the outage: backend 401 -> refresh -> dead token ->
    # the credentials error propagates out of complete() rather than being
    # swallowed by the bounded `except Exception` retry.
    llm = chatgpt_codex_module.ChatGptCodex(
        config=ChatGptCodexConfig(
            model_name="gpt-5.5",
            max_context_length=200_000,
            timeout_seconds=10.0,
        ),
        auth_store=chatgpt_codex_module._CodexAuthStore(
            _write_dead_auth_file(tmp_path)
        ),
        http_client=_RefreshClient(token_status=400),
    )

    with pytest.raises(chatgpt_codex_module.ChatGptCodexCredentialsExpiredError):
        asyncio.run(llm.complete(messages=[{"role": "user", "content": "go"}]))
