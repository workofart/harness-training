from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import get_args

import pytest
import httpx
from openai import APIError, BadRequestError

import src.plugins.caching.llm_cache as completion_cache
from src.config import RunConfig, LlmProviderConfig
from src.llm import openai_completion_backend as openai_backend_module
from src.llm.backend import (
    AgentBackend,
    Completion,
    CompletionBackend,
    CompletionInfraError,
    CompletionRequest,
    CompletionRequestError,
    ContextWindowExceededError,
    FrameworkError,
    TurnResult,
    backend_class,
    make_backend,
)


class _FakeLlm(CompletionBackend):
    async def _complete(self, request: CompletionRequest) -> Completion:
        del request
        return Completion()


class _FakeAgent(AgentBackend):
    def run_turn(self, *, prompt, repo_root: Path, emit, thread_id=None) -> TurnResult:
        del prompt, repo_root, emit, thread_id
        return TurnResult(thread_id="t1", progress_summary="-")


def test_completion_backend_contract_surface_is_transport_only():
    assert not inspect.isabstract(CompletionBackend)
    with pytest.raises(CompletionInfraError):
        asyncio.run(CompletionBackend().complete(CompletionRequest(messages=[])))
    assert inspect.iscoroutinefunction(CompletionBackend.complete)
    assert inspect.iscoroutinefunction(CompletionBackend.close)
    assert isinstance(_FakeLlm(), CompletionBackend)


class _FailingLlm(CompletionBackend):
    def __init__(self, error: BaseException) -> None:
        self._error = error

    async def _complete(self, request: CompletionRequest) -> Completion:
        del request
        raise self._error


def test_completion_backend_classifies_bad_request_as_request_error():
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    error = BadRequestError(
        "invalid request",
        response=httpx.Response(400, request=request),
        body=None,
    )

    with pytest.raises(CompletionRequestError) as raised:
        asyncio.run(_FailingLlm(error).complete(CompletionRequest(messages=[])))

    assert str(raised.value) == "invalid request"
    assert raised.value.__cause__ is error


@pytest.mark.parametrize(
    "error",
    [
        httpx.HTTPError("connection failed"),
        APIError(
            "stream failed",
            httpx.Request("POST", "https://example.test/v1/chat/completions"),
            body=None,
        ),
        OSError("socket failed"),
    ],
)
def test_completion_backend_classifies_transport_failures_as_infra(error):
    with pytest.raises(CompletionInfraError) as raised:
        asyncio.run(_FailingLlm(error).complete(CompletionRequest(messages=[])))

    assert raised.value.__cause__ is error


def test_completion_backend_classifies_arbitrary_exception_as_infra():
    # Unrecognized failure types (foreign SDK errors, adapter defects) must stay
    # per-task scorable crashes; a fatal default aborts the whole experiment.
    error = RuntimeError("normalization failed")

    with pytest.raises(CompletionInfraError) as raised:
        asyncio.run(_FailingLlm(error).complete(CompletionRequest(messages=[])))

    assert raised.value.__cause__ is error


@pytest.mark.parametrize(
    "error",
    [
        ContextWindowExceededError("too long", limit=100, requested=101),
        CompletionInfraError("provider unavailable"),
        FrameworkError("backend defect"),
    ],
)
def test_completion_backend_preserves_classified_failures(error):
    with pytest.raises(type(error)) as raised:
        asyncio.run(_FailingLlm(error).complete(CompletionRequest(messages=[])))

    assert raised.value is error
    assert raised.value.__cause__ is None


def test_completion_backend_does_not_classify_cancellation():
    error = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError) as raised:
        asyncio.run(_FailingLlm(error).complete(CompletionRequest(messages=[])))

    assert raised.value is error


def test_determinism_is_resolved_from_provider_configuration():
    assert LlmProviderConfig.model_validate(
        {**_OPENAI_PAYLOAD, "seed": 7}
    ).is_deterministic
    assert not LlmProviderConfig.model_validate(_OPENAI_PAYLOAD).is_deterministic


def test_agent_backend_contract_surface_is_one_turn():
    assert set(AgentBackend.__abstractmethods__) == {"run_turn"}
    assert isinstance(_FakeAgent(), AgentBackend)


def _config(provider_payload: dict) -> RunConfig:
    return RunConfig.model_validate(
        {
            "schema_version": 13,
            "training_target": {"module": "src.policy.core"},
            "environment": {"kind": "swe", "task_names": ["task-a"]},
            "llm_provider_config": provider_payload,
        }
    )


_OPENAI_PAYLOAD = {
    "model_name": "gpt-5.5",
    "base_url": "http://127.0.0.1:18000/v1",
    "api_key_env": "OPENAI_API_KEY",
    "max_context_length": 200000,
    "max_tokens": 8192,
}


def test_make_backend_builds_provider_backend_with_config():
    run_config = _config(_OPENAI_PAYLOAD)
    backend = make_backend(run_config.llm_provider_config)
    try:
        assert type(backend) is openai_backend_module.OpenAICompletionBackend
        assert backend.config is run_config.llm_provider_config
    finally:
        asyncio.run(backend.close())


def test_make_backend_uses_resolved_provider_revision(monkeypatch):
    captured: list[str] = []

    def _wrap(inner, *, revision: str):
        captured.append(revision)
        return inner

    monkeypatch.setattr(completion_cache, "maybe_wrap", _wrap)
    run_config = _config({**_OPENAI_PAYLOAD, "seed": 1})

    backend = make_backend(run_config.llm_provider_config, cache=True)

    try:
        assert captured == ["openai_compatible:http://127.0.0.1:18000/v1:gpt-5.5"]
    finally:
        asyncio.run(backend.close())


def test_make_backend_does_not_cache_nondeterministic_provider(monkeypatch):
    def _wrap(inner, *, revision: str):
        del inner, revision
        raise AssertionError("nondeterministic providers must not be cached")

    monkeypatch.setattr(completion_cache, "maybe_wrap", _wrap)
    run_config = _config(_OPENAI_PAYLOAD)

    backend = make_backend(run_config.llm_provider_config, cache=True)

    try:
        assert type(backend) is openai_backend_module.OpenAICompletionBackend
    finally:
        asyncio.run(backend.close())


def test_backend_class_resolves_provider_classes():
    assert (
        backend_class("openai_compatible")
        is openai_backend_module.OpenAICompletionBackend
    )


def test_provider_registry_surfaces_do_not_drift():
    providers = set(get_args(LlmProviderConfig.model_fields["provider"].annotation))
    assert providers == {"openai_compatible"}

    for provider in providers:
        cls = backend_class(provider)
        assert (
            "complete_duration_bound_sec" in vars(cls)
            or cls.complete_duration_bound_sec
            is not CompletionBackend.complete_duration_bound_sec
        )
        assert not hasattr(cls, "supports_determinism")
        assert cls.complete_duration_bound_sec(8192) > 0

    with pytest.raises(KeyError):
        backend_class("unknown_provider")


def test_provider_revision_appends_operator_override(monkeypatch):
    config = LlmProviderConfig.model_validate({**_OPENAI_PAYLOAD, "seed": 1})
    monkeypatch.setenv("FRAMEWORK_LLM_CACHE_REV", "weights-v2")
    assert config.provider_revision == (
        "openai_compatible:http://127.0.0.1:18000/v1:gpt-5.5:weights-v2"
    )
