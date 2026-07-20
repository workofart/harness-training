from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import APIError, AuthenticationError, BadRequestError, RateLimitError
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

import src.llm.openai_completion_backend as openai_backend_module
import src.llm.transport as transport_module
from src.config import LlmProviderConfig
from src.llm.backend import (
    CompletionInfraError,
    CompletionRequest,
    ContextWindowExceededError,
    ProviderRejectedToolCallError,
)


def _cfg(**overrides: Any) -> LlmProviderConfig:
    return LlmProviderConfig.model_validate(
        {
            "model_name": "gpt-5.5",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
            "max_context_length": 131_072,
            "max_tokens": 8192,
            **overrides,
        }
    )


def _qwen_cfg(**overrides: Any) -> LlmProviderConfig:
    return _cfg(
        **{
            "model_name": "Qwen/Qwen3.6-35B-A3B",
            "base_url": "http://127.0.0.1:18000/v1",
            "api_key_env": "TEST_API_KEY",
            "enable_thinking": True,
            **overrides,
        }
    )


def _body(config: LlmProviderConfig, **kwargs):
    backend = openai_backend_module.OpenAICompletionBackend(config=config)
    return backend._build_request_body(CompletionRequest(**kwargs))


def _use_client(monkeypatch, fake_client) -> None:
    """Route the backend's lazily-built AsyncOpenAI to a fake. Injection was removed
    from the production constructor, so behavior tests patch the real dependency
    boundary (the AsyncOpenAI class) instead of handing a client in."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    monkeypatch.setattr(
        openai_backend_module, "AsyncOpenAI", lambda **_kwargs: fake_client
    )


def _complete(monkeypatch, responses, *, config=None, **kwargs):
    client = _FakeOpenAIClient(responses)
    _use_client(monkeypatch, client)
    backend = openai_backend_module.OpenAICompletionBackend(config=config or _cfg())
    completion = asyncio.run(
        backend.complete(
            CompletionRequest(messages=[{"role": "user", "content": "go"}], **kwargs)
        )
    )
    return completion, client


def test_openai_backend_forwards_standard_params_and_tools():
    body = _body(
        _cfg(temperature=0.0, top_p=1.0, seed=123, max_tokens=4096),
        messages=[{"role": "user", "content": "read the file"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "read_file", "parameters": {"type": "object"}},
            }
        ],
    )
    assert body["model"] == "gpt-5.5"
    assert body["temperature"] == 0.0
    assert body["top_p"] == 1.0
    assert body["seed"] == 123
    assert body["max_tokens"] == 4096
    assert body["tool_choice"] == "auto"
    assert body["tools"][0]["function"]["name"] == "read_file"
    assert "extra_body" not in body


def test_openai_backend_omits_unset_sampling_and_provider_policy():
    body = _body(_cfg(), messages=[{"role": "user", "content": "go"}])
    for key in (
        "temperature",
        "top_p",
        "seed",
        "extra_body",
        "reasoning_effort",
        "tools",
    ):
        assert key not in body
    assert body["max_tokens"] == 8192


def test_openai_backend_forwards_explicit_reasoning_effort():
    body = _body(
        _cfg(reasoning_effort="high"),
        messages=[{"role": "user", "content": "go"}],
    )

    assert body["reasoning_effort"] == "high"


def test_openai_backend_request_body_tracks_serialized_config():
    """Configs that serialize identically must produce identical request
    bodies and cache keys, or measurement identity dedup breaks."""
    omitted = _cfg()
    # Round-tripping marks every field explicitly set; the body must not care.
    round_tripped = LlmProviderConfig.model_validate(omitted.model_dump(mode="json"))
    assert omitted.model_dump(mode="json") == round_tripped.model_dump(mode="json")

    request = CompletionRequest(messages=[{"role": "user", "content": "go"}])
    backend_a = openai_backend_module.OpenAICompletionBackend(config=omitted)
    backend_b = openai_backend_module.OpenAICompletionBackend(config=round_tripped)
    assert backend_a._build_request_body(request) == backend_b._build_request_body(
        request
    )
    assert backend_a.cache_key(request) == backend_b.cache_key(request)


def test_openai_backend_forwards_provider_only_routing():
    body = _body(
        _cfg(provider_only=["wandb"]),
        messages=[{"role": "user", "content": "go"}],
    )

    assert body["extra_body"] == {"provider": {"only": ["wandb"]}}


@pytest.mark.parametrize(
    ("configured", "override", "expected"),
    [
        pytest.param(True, None, True, id="enabled"),
        pytest.param(False, None, False, id="disabled"),
        pytest.param(True, False, False, id="per-call-override"),
    ],
)
def test_thinking_policy_is_sent_as_chat_template_kwarg(
    configured: bool, override: bool | None, expected: bool
):
    body = _body(
        _qwen_cfg(enable_thinking=configured),
        messages=[{"role": "user", "content": "go"}],
        enable_thinking=override,
    )
    assert body["extra_body"] == {"chat_template_kwargs": {"enable_thinking": expected}}


def test_thinking_override_rejected_when_config_does_not_drive_thinking():
    with pytest.raises(ValueError, match="declares no thinking channel"):
        _body(
            _cfg(),
            messages=[{"role": "user", "content": "go"}],
            enable_thinking=False,
        )


def test_cache_key_distinguishes_thinking_override():
    backend = openai_backend_module.OpenAICompletionBackend(
        config=_qwen_cfg(enable_thinking=True)
    )
    messages = [{"role": "user", "content": "go"}]
    base = backend.cache_key(CompletionRequest(messages=messages))
    assert (
        backend.cache_key(CompletionRequest(messages=messages, enable_thinking=None))
        == base
    )
    assert (
        backend.cache_key(CompletionRequest(messages=messages, enable_thinking=False))
        != base
    )


def test_complete_uses_configured_api_key_env(monkeypatch):
    captured: dict[str, Any] = {}

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured["built"] = captured.get("built", 0) + 1
            captured.update(kwargs)
            self.chat = _FakeOpenAIClient([_fake_response(), _fake_response()]).chat

        async def close(self):
            captured["closed"] = captured.get("closed", 0) + 1

    monkeypatch.setenv("LOCAL_API_KEY", "local-test-key")
    monkeypatch.setattr(openai_backend_module, "AsyncOpenAI", _FakeAsyncOpenAI)
    backend = openai_backend_module.OpenAICompletionBackend(
        config=_cfg(
            base_url="http://127.0.0.1:8000/v1",
            api_key_env="LOCAL_API_KEY",
        )
    )
    completion = asyncio.run(
        backend.complete(
            CompletionRequest(messages=[{"role": "user", "content": "hi"}])
        )
    )
    asyncio.run(
        backend.complete(
            CompletionRequest(messages=[{"role": "user", "content": "again"}])
        )
    )

    assert captured["built"] == 1
    assert captured["api_key"] == "local-test-key"
    assert str(captured["base_url"]) == "http://127.0.0.1:8000/v1"
    # The request timeout bounds connect/write/pool; stream read silence is
    # governed by the stall window, never clamped down to it.
    assert captured["timeout"] == httpx.Timeout(
        transport_module.REQUEST_TIMEOUT_SECONDS,
        read=transport_module.stream_stall_timeout_seconds(8192),
    )
    # complete()'s tenacity layer owns all retries; nested SDK retries would
    # multiply budgets on a persistently failing request.
    assert captured["max_retries"] == 0
    assert "closed" not in captured
    assert completion.content == "PONG"
    asyncio.run(backend.close())
    assert captured["closed"] == 1


def test_complete_requests_streaming_and_normalizes_streamed_chunks(monkeypatch):
    completion, client = _complete(
        monkeypatch,
        _FakeStream(
            [
                _fake_chunk(
                    content="PO",
                    reasoning_content="think ",
                    tool_calls=[
                        _fake_tool_call_delta(
                            index=0,
                            name="run",
                            arguments='{"cmd"',
                        )
                    ],
                ),
                _fake_chunk(
                    content="NG",
                    reasoning_content="more",
                    tool_calls=[_fake_tool_call_delta(index=0, arguments=':"pwd"}')],
                ),
                _fake_chunk(finish_reason="tool_calls"),
                _fake_chunk(choices=[], usage=_fake_usage(), system_fingerprint=None),
            ]
        ),
        config=_cfg(temperature=0.0, seed=123),
    )

    assert client.create_calls[0]["stream"] is True
    assert client.create_calls[0]["stream_options"] == {"include_usage": True}
    assert client.create_calls[0]["temperature"] == 0.0
    assert client.create_calls[0]["seed"] == 123
    assert completion.content == "PONG"
    assert completion.reasoning_content == "think more"
    assert completion.finish_reason == "tool_calls"
    assert completion.tool_calls[0].name == "run"
    assert completion.tool_calls[0].arguments == '{"cmd":"pwd"}'
    assert completion.usage.prompt_tokens == 10
    assert completion.usage.completion_tokens == 5
    assert completion.usage.reasoning_tokens == 2
    assert completion.usage.cached_input_tokens == 4
    assert completion.response == {
        "id": "chatcmpl_1",
        "model": "openai/gpt-oss-20b",
        "system_fingerprint": None,
    }


def test_complete_preserves_length_finish_reason_from_stream(monkeypatch):
    completion, _client = _complete(
        monkeypatch,
        _FakeStream(
            [
                _fake_chunk(content="partial", finish_reason="length"),
                _fake_chunk(choices=[], usage=_fake_usage()),
            ]
        ),
    )

    assert completion.content == "partial"
    assert completion.finish_reason == "length"


def test_complete_tolerates_stream_without_usage_chunk(monkeypatch):
    # Some OpenRouter upstreams omit the best-effort usage chunk; preserve the
    # completion without usage.
    completion, _client = _complete(
        monkeypatch,
        _FakeStream(
            [
                _fake_chunk(content="PONG", finish_reason="stop"),
            ]
        ),
    )
    assert completion.content == "PONG"
    assert completion.finish_reason == "stop"
    assert completion.usage.prompt_tokens is None
    assert completion.usage.completion_tokens is None
    assert completion.usage.reasoning_tokens is None
    assert completion.usage.cached_input_tokens is None


def test_complete_retries_malformed_tool_call_index_stream(monkeypatch):
    # Some providers skip tool-call index zero, which the SDK cannot accumulate.
    # Treat that malformed stream as transient instead of crashing the task.
    _no_backoff(monkeypatch)
    completion, client = _complete(
        monkeypatch,
        [
            _FakeStream(
                [
                    _fake_chunk(
                        tool_calls=[
                            _fake_tool_call_delta(index=1, name="run", arguments="{}")
                        ],
                    ),
                ]
            ),
            _fake_response(content="PONG"),
        ],
    )

    assert completion.content == "PONG"
    assert len(client.create_calls) == 2


class _WordCounter:
    """Token counter fake: one word = one token."""

    def count(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, max_tokens: int) -> str:
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens]) + "\n...[truncated]"


def _budget_backend(monkeypatch, *, counter: Any = None, **overrides: Any):
    monkeypatch.setattr(
        openai_backend_module,
        "resolve_token_counter",
        lambda _tokenizer_name, _model_name: counter,
    )
    return openai_backend_module.OpenAICompletionBackend(
        config=_qwen_cfg(thinking_budget_tokens=4, **overrides)
    )


def _runaway_chunk(words: int = 100):
    return _fake_chunk(reasoning_content=" ".join(f"w{i}" for i in range(words)))


def _run_budget(monkeypatch, responses, **kwargs):
    """Run complete() through a thinking-budget backend (budget=4, word counter)."""
    client = _FakeOpenAIClient(responses)
    _use_client(monkeypatch, client)
    backend = _budget_backend(monkeypatch, counter=_WordCounter())
    completion = asyncio.run(
        backend.complete(
            CompletionRequest(messages=[{"role": "user", "content": "go"}], **kwargs)
        )
    )
    return completion, client


def test_thinking_budget_aborts_runaway_as_length(monkeypatch):
    runaway = _FakeStream([_runaway_chunk(), _fake_chunk(content="never read")])
    completion, client = _run_budget(monkeypatch, [runaway])

    # Surface an aborted stream as length so thinking-off recovery fires without
    # retrying.
    assert runaway.closed is True
    assert len(client.create_calls) == 1
    assert completion.finish_reason == "length"
    assert completion.tool_calls == ()
    assert completion.content is None
    # The reasoning is clipped at exactly the budget, so the stored completion
    # does not depend on where the stream happened to be cut.
    assert completion.reasoning_content == "w0 w1 w2 w3\n...[truncated]"
    assert completion.response["budget_aborted"] is True
    # No server usage arrived; only the exact budget counts are known.
    assert completion.usage.completion_tokens == 4
    assert completion.usage.reasoning_tokens == 4
    assert completion.usage.prompt_tokens is None


def test_thinking_budget_never_fires_after_answer_starts(monkeypatch):
    # Once content starts, accumulated reasoning cannot trigger the budget.
    monkeypatch.setattr(openai_backend_module, "_BUDGET_SLACK_TOKENS", 200)
    completion, client = _run_budget(
        monkeypatch,
        _FakeStream(
            [
                _runaway_chunk(),
                _fake_chunk(content="PONG", finish_reason="stop"),
                _runaway_chunk(),
                _fake_chunk(choices=[], usage=_fake_usage()),
            ]
        ),
    )
    assert len(client.create_calls) == 1
    assert completion.content == "PONG"
    assert "budget_aborted" not in completion.response


def test_thinking_budget_skipped_on_thinking_off_calls(monkeypatch):
    # The thinking-off repair retry decodes in the answer channel; budget
    # forcing must never touch it.
    completion, client = _run_budget(
        monkeypatch,
        _FakeStream(
            [
                _runaway_chunk(),
                _fake_chunk(finish_reason="length"),
                _fake_chunk(choices=[], usage=_fake_usage()),
            ]
        ),
        enable_thinking=False,
    )
    assert len(client.create_calls) == 1
    assert completion.finish_reason == "length"
    assert "budget_aborted" not in completion.response
    assert completion.usage.prompt_tokens == 10
    assert completion.usage.completion_tokens == 5
    assert completion.usage.reasoning_tokens == 2
    assert completion.usage.cached_input_tokens == 4


def test_thinking_budget_requires_resolvable_tokenizer(monkeypatch):
    with pytest.raises(ValueError, match="resolvable tokenizer"):
        _budget_backend(monkeypatch, counter=None)


def test_thinking_budget_changes_cache_key_without_changing_request_body(monkeypatch):
    messages = [{"role": "user", "content": "go"}]
    with_budget = _budget_backend(monkeypatch, counter=_WordCounter())
    without = openai_backend_module.OpenAICompletionBackend(config=_qwen_cfg())
    request = CompletionRequest(messages=messages)
    assert with_budget._build_request_body(request) == without._build_request_body(
        request
    )
    assert with_budget.cache_key(request) != without.cache_key(request)


def _no_backoff(monkeypatch) -> None:
    """Neutralize the tenacity retry backoff so retry tests don't really sleep."""

    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(transport_module.asyncio, "sleep", _instant)


def _remote_drop():
    return _FakeStream(
        [
            _fake_chunk(content="partial"),
            httpx.RemoteProtocolError("incomplete chunked read"),
        ]
    )


def _read_stall():
    return _FakeStream(
        [
            _fake_chunk(content="partial"),
            httpx.ReadTimeout("no chunk within the stall window"),
        ]
    )


def _streamed_api_error():
    return _FakeStream(
        [
            _fake_chunk(content="partial"),
            APIError(
                "Response payload is not completed: TransferEncodingError",
                request=httpx.Request(
                    "POST", "https://api.openai.com/v1/chat/completions"
                ),
                body=None,
            ),
        ]
    )


def _rate_limit_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return RateLimitError(
        "rate limited", response=httpx.Response(429, request=request), body=None
    )


@pytest.mark.parametrize(
    "failure_factory",
    [
        pytest.param(_remote_drop, id="remote-protocol"),
        pytest.param(_read_stall, id="read-stall"),
        pytest.param(_streamed_api_error, id="streamed-api-error"),
        pytest.param(_rate_limit_error, id="rate-limit"),
    ],
)
def test_complete_retries_transient_stream_failures(monkeypatch, failure_factory):
    _no_backoff(monkeypatch)
    completion, client = _complete(
        monkeypatch, [failure_factory(), _fake_response(content="PONG")]
    )
    assert completion.content == "PONG"
    assert len(client.create_calls) == 2


def test_complete_routes_rejected_tool_args_without_transport_retry(monkeypatch):
    error = APIError(
        "Upstream error from Groq: Failed to parse tool call arguments as JSON",
        request=httpx.Request("POST", "https://openrouter.ai/api/v1"),
        body=None,
    )
    client = _FakeOpenAIClient([error, _fake_response(content="PONG")])
    _use_client(monkeypatch, client)
    backend = openai_backend_module.OpenAICompletionBackend(config=_cfg())

    with pytest.raises(ProviderRejectedToolCallError) as exc_info:
        asyncio.run(
            backend.complete(
                CompletionRequest(messages=[{"role": "user", "content": "go"}])
            )
        )

    assert exc_info.value.__cause__ is error
    assert len(client.create_calls) == 1


def test_complete_classifies_remote_protocol_error_after_budget_exhausted(monkeypatch):
    _no_backoff(monkeypatch)
    budget = transport_module.OPENAI_CLIENT_RETRY_BUDGET
    client = _FakeOpenAIClient(
        [
            _FakeStream(
                [
                    httpx.RemoteProtocolError(
                        "peer closed connection without sending complete message body "
                        "(incomplete chunked read)"
                    )
                ]
            )
            for _ in range(budget + 1)
        ]
    )
    _use_client(monkeypatch, client)
    backend = openai_backend_module.OpenAICompletionBackend(config=_cfg())
    with pytest.raises(CompletionInfraError) as exc_info:
        asyncio.run(
            backend.complete(
                CompletionRequest(messages=[{"role": "user", "content": "go"}])
            )
        )
    assert isinstance(exc_info.value.__cause__, httpx.RemoteProtocolError)
    assert len(client.create_calls) == budget + 1


def test_stream_stall_window_covers_max_tokens_buffered_tool_call():
    # Cover a fully buffered tool call at ~22 tok/s plus first-token wait; the
    # 240s floor alone was insufficient.
    window = transport_module.stream_stall_timeout_seconds(8192)
    assert window > 8192 / 22 + 30
    # Small caps keep the legacy floor for first-token waits.
    floor = transport_module.STREAM_STALL_TIMEOUT_FLOOR_SECONDS
    assert transport_module.stream_stall_timeout_seconds(1000) == floor


def test_complete_applies_exponential_backoff_between_stream_retries(monkeypatch):
    slept: list[float] = []

    async def _record(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(transport_module.asyncio, "sleep", _record)
    client = _FakeOpenAIClient(
        [
            _FakeStream([httpx.RemoteProtocolError("drop 1")]),
            _FakeStream([httpx.RemoteProtocolError("drop 2")]),
            _fake_response(content="PONG"),
        ]
    )
    _use_client(monkeypatch, client)
    backend = openai_backend_module.OpenAICompletionBackend(config=_cfg())

    completion = asyncio.run(
        backend.complete(
            CompletionRequest(messages=[{"role": "user", "content": "go"}])
        )
    )

    assert completion.content == "PONG"
    assert len(client.create_calls) == 3
    # Exponential 4 * 2^n, sized so the full budget's sleep (105s) spans a
    # serverless worker-restart window (60-110s observed).
    assert slept == [4.0, 8.0]


def test_complete_classifies_auth_error_without_retry(monkeypatch):
    # Permanent 4xx errors surface without consuming the retry budget.
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    auth_error = AuthenticationError(
        "invalid api key",
        response=httpx.Response(401, request=request),
        body=None,
    )
    client = _FakeOpenAIClient([auth_error, _fake_response(content="PONG")])
    _use_client(monkeypatch, client)
    backend = openai_backend_module.OpenAICompletionBackend(config=_cfg())
    with pytest.raises(CompletionInfraError) as exc_info:
        asyncio.run(
            backend.complete(
                CompletionRequest(messages=[{"role": "user", "content": "go"}])
            )
        )
    assert exc_info.value.__cause__ is auth_error
    assert len(client.create_calls) == 1


@pytest.mark.parametrize(
    ("body", "expected_input_tokens"),
    [
        pytest.param(
            {
                "code": "context_length_exceeded",
                "message": (
                    "Requested token count exceeds the model's maximum context length "
                    "of 32768 tokens. You requested a total of 33131 tokens: 29035 "
                    "tokens from the input messages and 4096 tokens for the completion."
                ),
            },
            29035,
            id="openai-code-with-input-count",
        ),
        pytest.param(
            {
                "message": (
                    "Requested token count exceeds the model's maximum context length "
                    "of 32768 tokens. You requested a total of 33131 tokens."
                )
            },
            None,
            id="prose-only",
        ),
    ],
)
def test_context_window_errors_are_normalized(monkeypatch, body, expected_input_tokens):
    client = _FakeOpenAIClient(_bad_request_error(body=body))
    _use_client(monkeypatch, client)
    backend = openai_backend_module.OpenAICompletionBackend(config=_cfg())
    with pytest.raises(ContextWindowExceededError) as exc_info:
        asyncio.run(
            backend.complete(
                CompletionRequest(messages=[{"role": "user", "content": "go"}])
            )
        )
    err = exc_info.value
    assert (err.limit, err.requested, err.input_tokens) == (
        32768,
        33131,
        expected_input_tokens,
    )
    assert len(client.create_calls) == 1


class _FakeOpenAIClient:
    def __init__(self, response):
        self.create_calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=self)
        self._responses = response if isinstance(response, list) else [response]

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if kwargs.get("stream") and not isinstance(response, _FakeStream):
            return _fake_stream_from_response(response)
        return response


def _fake_response(
    *,
    content: str | None = "PONG",
    tool_calls: list[Any] | None = None,
    reasoning_content: str | None = None,
):
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "created": 1,
            "model": "openai/gpt-oss-20b",
            "system_fingerprint": "fp_test",
            "choices": [
                {"index": 0, "message": message, "finish_reason": "stop"},
            ],
            "usage": _fake_usage(),
        }
    )


class _FakeStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        await self.close()

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for chunk in self._chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk

    async def close(self):
        self.closed = True


def _fake_stream_from_response(response: Any) -> _FakeStream:
    choice = response.choices[0]
    message = choice.message
    message_data = message.to_dict()
    tool_calls = [
        _fake_tool_call_delta(
            index=index,
            name=call.function.name,
            arguments=call.function.arguments,
        )
        for index, call in enumerate(message.tool_calls or ())
    ]
    chunk = _fake_chunk(
        id=response.id,
        model=response.model,
        system_fingerprint=response.system_fingerprint,
        content=message.content,
        reasoning_content=message_data.get("reasoning_content"),
        tool_calls=tool_calls,
        finish_reason=choice.finish_reason,
    )
    usage_chunk = _fake_chunk(
        id=response.id,
        model=response.model,
        system_fingerprint=None,
        choices=[],
        usage=response.usage,
    )
    return _FakeStream([chunk, usage_chunk])


def _fake_chunk(
    *,
    id: str = "chatcmpl_1",
    model: str = "openai/gpt-oss-20b",
    system_fingerprint: str | None = "fp_test",
    content: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
    choices: list[Any] | None = None,
    usage: Any | None = None,
):
    if choices is None:
        delta: dict[str, Any] = {}
        if content is not None:
            delta["content"] = content
        if reasoning_content is not None:
            delta["reasoning_content"] = reasoning_content
        if tool_calls is not None:
            delta["tool_calls"] = tool_calls
        choices = [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    payload: dict[str, Any] = {
        "id": id,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": choices,
    }
    payload["system_fingerprint"] = system_fingerprint
    if usage is not None:
        payload["usage"] = usage
    return ChatCompletionChunk.model_validate(payload)


def _fake_tool_call_delta(
    *,
    index: int,
    name: str | None = None,
    arguments: str | None = None,
):
    function: dict[str, Any] = {}
    if name is not None:
        function["name"] = name
    if arguments is not None:
        function["arguments"] = arguments
    payload: dict[str, Any] = {"index": index, "function": function}
    if name is not None:
        payload["id"] = f"call_{index}"
        payload["type"] = "function"
    return payload


def _fake_usage():
    return {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "completion_tokens_details": {"reasoning_tokens": 2},
        "prompt_tokens_details": {"cached_tokens": 4},
    }


def _bad_request_error(*, body: object) -> BadRequestError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(400, request=request)
    message = body.get("message") if isinstance(body, dict) else "bad request"
    return BadRequestError(str(message or "bad request"), response=response, body=body)
