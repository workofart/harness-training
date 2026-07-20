"""Tests for the cross-run LLM completion cache (src/plugins/caching/).

No network: a counting fake backend stands in for the provider, and the OpenAI
backend's `cache_key` is exercised on its pure request-body builder. Each test that
touches storage isolates the process-global singleton onto a tmp SQLite file.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Any

import pytest

from src.plugins.caching import store as cache
import src.plugins.caching.llm_cache as cc
from src.config import LlmProviderConfig
from src.llm.backend import (
    Completion,
    CompletionBackend,
    CompletionInfraError,
    CompletionRequest,
    CompletionRequestError,
    ContextWindowExceededError,
    FrameworkError,
    ToolCall,
    Usage,
)
from src.llm.openai_completion_backend import OpenAICompletionBackend
from src.rollout.telemetry import InstrumentedLlm, RolloutTelemetry


def _completion(content: str = "ok") -> Completion:
    return Completion(
        tool_calls=(ToolCall(name="submit", arguments='{"answer":"x"}'),),
        content=content,
        finish_reason="tool_calls",
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=5,
            reasoning_tokens=2,
            cached_input_tokens=4,
        ),
        reasoning_content="because",
        response={"id": "resp-1", "model": "m"},
    )


class _CountingBackend(CompletionBackend):
    """Returns a canned completion, counts live calls, exposes a fixed cache key."""

    def __init__(self, completion: Completion, *, key: str = "k") -> None:
        self._completion = completion
        self._key = key
        self.calls = 0
        self.closed = False

    async def _complete(self, request):
        del request
        self.calls += 1
        return self._completion

    def cache_key(self, request):
        del request
        return self._key

    async def close(self) -> None:
        self.closed = True


_MSGS = [{"role": "user", "content": "hi"}]
_REQUEST = CompletionRequest(messages=_MSGS)


def test_miss_then_hit_serves_from_cache(store_env):
    inner = _CountingBackend(_completion(), key="abc")
    backend = cc.CachingCompletionBackend(inner, revision="rev")
    first = asyncio.run(backend.complete(_REQUEST))
    second = asyncio.run(backend.complete(_REQUEST))
    assert inner.calls == 1
    assert first == _completion()
    assert first.served_from_cache is False
    assert second.served_from_cache is True
    assert dataclasses.replace(second, served_from_cache=False) == _completion()


@pytest.mark.parametrize(
    ("keys", "revisions"),
    [
        pytest.param(("ka", "kb"), ("r", "r"), id="backend-key"),
        pytest.param(("k", "k"), ("v1", "v2"), id="provider-revision"),
    ],
)
def test_cache_identity_dimensions_do_not_collide(store_env, keys, revisions):
    a = _CountingBackend(_completion(content="A"), key=keys[0])
    b = _CountingBackend(_completion(content="B"), key=keys[1])
    results = [
        asyncio.run(
            cc.CachingCompletionBackend(inner, revision=revision).complete(_REQUEST)
        )
        for inner, revision in zip((a, b), revisions, strict=True)
    ]
    assert [result.content for result in results] == ["A", "B"]
    assert (a.calls, b.calls) == (1, 1)


def test_thinking_override_reaches_inner_key_and_call(store_env):
    # A dropped override would silently collide the degraded retry's cache
    # entry with the normal call's -- both derivations must see it.
    seen: dict[str, bool | None] = {}

    class _Recorder(_CountingBackend):
        def cache_key(self, request):
            seen["key"] = request.enable_thinking
            return "k"

        async def _complete(self, request):
            seen["call"] = request.enable_thinking
            return await super()._complete(request)

    backend = cc.CachingCompletionBackend(_Recorder(_completion()), revision="r")
    asyncio.run(
        backend.complete(CompletionRequest(messages=_MSGS, enable_thinking=False))
    )
    assert seen == {"key": False, "call": False}


def test_corrupt_row_falls_through_to_live(store_env):
    inner = _CountingBackend(_completion(content="LIVE"), key="k")
    backend = cc.CachingCompletionBackend(inner, revision="r")
    asyncio.run(cache.put("c:r:k", "{not valid json"))
    result = asyncio.run(backend.complete(_REQUEST))
    assert result.content == "LIVE"
    assert inner.calls == 1


@pytest.mark.parametrize(
    "error",
    [
        CompletionRequestError("bad request"),
        ContextWindowExceededError("too long", limit=100, requested=101),
        CompletionInfraError("provider unavailable"),
        FrameworkError("backend defect"),
    ],
)
def test_backend_failure_type_passes_through_cache(store_env, error):
    class _FailingBackend(_CountingBackend):
        async def _complete(self, request):
            del request
            raise error

    backend = cc.CachingCompletionBackend(_FailingBackend(_completion()), revision="r")

    with pytest.raises(type(error)) as raised:
        asyncio.run(backend.complete(_REQUEST))

    assert raised.value is error


def test_non_json_completion_is_framework_owned(store_env):
    # Serialization failure is a framework defect; preserve complete()'s typed contract.
    completion = dataclasses.replace(_completion(), response={"bad": object()})
    backend = cc.CachingCompletionBackend(_CountingBackend(completion), revision="r")

    with pytest.raises(FrameworkError):
        asyncio.run(backend.complete(_REQUEST))


def test_close_closes_inner_not_store(store_env):
    inner = _CountingBackend(_completion(), key="k")
    backend = cc.CachingCompletionBackend(inner, revision="r")
    asyncio.run(backend.complete(_REQUEST))
    asyncio.run(backend.close())
    assert inner.closed
    assert asyncio.run(cache.get("missing")) is None
    assert asyncio.run(cache.get("c:r:k")) is not None


def test_instrumented_llm_outside_cache_records_cache_hits(store_env, tmp_path):
    inner = _CountingBackend(_completion(content="cached"), key="telemetry")
    trace_path = tmp_path / "steps.jsonl"
    backend = InstrumentedLlm(
        cc.CachingCompletionBackend(inner, revision="ordering-pin"),
        RolloutTelemetry(rollout_dir=tmp_path, trace_path=trace_path),
    )

    asyncio.run(backend.complete(_REQUEST))
    rows_before = len(trace_path.read_text().splitlines())
    second = asyncio.run(backend.complete(_REQUEST))

    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    hit_rows = rows[rows_before:]
    assert inner.calls == 1
    assert second.content == "cached"
    assert [row["event"] for row in hit_rows] == ["completion_received"]
    payload = hit_rows[0]
    assert payload["completion"]["content"] == "cached"
    assert payload["completion"]["served_from_cache"] is True
    assert isinstance(payload["llm_latency_sec"], float)


def test_put_is_idempotent_first_writer_wins(store_env):
    store = cache.store()
    asyncio.run(store.put("k", "v1"))
    asyncio.run(store.put("k", "v2"))
    assert asyncio.run(store.get("k")) == "v1"


def test_maybe_wrap_disabled_returns_inner(monkeypatch):
    monkeypatch.setattr(cache, "_DISABLED", True)
    inner = _CountingBackend(_completion(), key="k")
    assert cc.maybe_wrap(inner, revision="provider:m") is inner


def test_maybe_wrap_enabled_uses_resolved_revision(store_env):
    inner = _CountingBackend(_completion(), key="k")
    wrapped = cc.maybe_wrap(inner, revision="provider:m")
    assert isinstance(wrapped, cc.CachingCompletionBackend)
    assert wrapped._revision == "provider:m"


# OpenAI cache-key derivation


def _cfg(**overrides: Any) -> LlmProviderConfig:
    return LlmProviderConfig.model_validate(
        {
            "model_name": "m",
            "base_url": "http://x/v1",
            "api_key_env": "OPENAI_API_KEY",
            "max_context_length": 1000,
            "max_tokens": 256,
            "seed": 1,
            **overrides,
        }
    )


def test_openai_cache_key_bytes_pinned():
    # These legacy literals keep existing cache rows addressable; key changes require an explicit FRAMEWORK_LLM_CACHE_REV fork.
    backend = OpenAICompletionBackend(
        config=_cfg(
            model_name="test-model",
            base_url="http://localhost:1234/v1",
            api_key_env="K",
            max_tokens=64,
            max_context_length=8192,
            temperature=0.0,
            seed=7,
        )
    )
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "run", "parameters": {}}}]
    assert (
        backend.cache_key(CompletionRequest(messages=messages, tools=tools))
        == "03711aba9981bcbf146c4fe92aded85d314b62dc91409921c3fc08618c83f422"
    )
    assert (
        backend.cache_key(CompletionRequest(messages=messages))
        == "63f81cbb07c855386f17e9a5738c805896a7790d4c0f8ba5c3e0d0417ebaa005"
    )


def test_openai_cache_key_stable_and_sensitive():
    backend = OpenAICompletionBackend(config=_cfg())
    key = backend.cache_key(_REQUEST)
    assert key == backend.cache_key(_REQUEST)
    assert OpenAICompletionBackend(config=_cfg(seed=2)).cache_key(_REQUEST) != key
    other = backend.cache_key(
        CompletionRequest(messages=[{"role": "user", "content": "bye"}])
    )
    assert other != key
