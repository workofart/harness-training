from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass

from openrouter import components, utils

import src.llm.openrouter as open_router_module
from src.config import OpenRouterConfig


@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    function: _FakeFunction


@dataclass
class _FakeMessage:
    content: str | None
    tool_calls: list[_FakeToolCall] | None
    provider_specific_fields: dict[str, object] | None = None


@dataclass
class _FakeChoice:
    message: _FakeMessage
    finish_reason: str | None = None


@dataclass
class _FakeCompletion:
    choices: list[_FakeChoice]


async def _fast_sleep(delay):
    del delay


def test_to_llm_completion_normalizes_dataclass_shape():
    completion = open_router_module._to_llm_completion(
        _FakeCompletion(
            choices=[
                _FakeChoice(
                    message=_FakeMessage(
                        content=None,
                        tool_calls=[
                            _FakeToolCall(
                                function=_FakeFunction(
                                    name="read_file",
                                    arguments='{"path":"README.md"}',
                                )
                            )
                        ],
                    )
                )
            ]
        )
    )

    assert len(completion.tool_calls) == 1
    assert completion.tool_calls[0].name == "read_file"
    assert completion.tool_calls[0].arguments == '{"path":"README.md"}'


def test_to_llm_completion_normalizes_openrouter_dict_shape():
    completion = open_router_module._to_llm_completion(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "fc_1",
                                "index": 0,
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"README.md"}',
                                },
                            }
                        ],
                    },
                }
            ]
        }
    )

    assert len(completion.tool_calls) == 1
    assert completion.tool_calls[0].name == "read_file"
    assert completion.tool_calls[0].arguments == '{"path":"README.md"}'


def test_openrouter_complete_omits_unset_optional_top_level_kwargs(monkeypatch):
    captured_client_kwargs: dict[str, object] = {}
    captured_kwargs: dict[str, object] = {}

    class _FakeChat:
        async def send_async(self, **kwargs):
            captured_kwargs.update(kwargs)
            return _FakeCompletion(
                choices=[
                    _FakeChoice(
                        message=_FakeMessage(
                            content=None,
                            tool_calls=[],
                        )
                    )
                ]
            )

    class _FakeOpenRouterClient:
        def __init__(self, **kwargs):
            captured_client_kwargs.update(kwargs)
            self.chat = _FakeChat()

    monkeypatch.setattr(open_router_module, "OpenRouterClient", _FakeOpenRouterClient)

    llm = open_router_module.OpenRouter(
        config=OpenRouterConfig(
            model_name="openrouter/openai/gpt-oss-20b",
            seed=None,
        ),
        api_key="test-key",
    )

    completion = asyncio.run(
        llm.complete(messages=[{"role": "user", "content": "do the thing"}])
    )

    assert isinstance(completion, open_router_module.LlmCompletion)
    assert captured_client_kwargs["api_key"] == "test-key"
    assert captured_kwargs["model"] == "openai/gpt-oss-20b"
    assert captured_kwargs["retries"] is None
    assert "max_tokens" not in captured_kwargs
    assert "reasoning" not in captured_kwargs
    assert "temperature" not in captured_kwargs
    assert "seed" not in captured_kwargs
    assert "top_p" not in captured_kwargs
    assert "service_tier" not in captured_kwargs


def test_openrouter_complete_forwards_explicit_optional_top_level_kwargs(monkeypatch):
    captured_kwargs: dict[str, object] = {}

    class _FakeChat:
        async def send_async(self, **kwargs):
            captured_kwargs.update(kwargs)
            return _FakeCompletion(
                choices=[
                    _FakeChoice(
                        message=_FakeMessage(
                            content=None,
                            tool_calls=[],
                        )
                    )
                ]
            )

    class _FakeOpenRouterClient:
        def __init__(self, **kwargs):
            del kwargs
            self.chat = _FakeChat()

    monkeypatch.setattr(open_router_module, "OpenRouterClient", _FakeOpenRouterClient)

    llm = open_router_module.OpenRouter(
        config=OpenRouterConfig(
            model_name="openrouter/openai/gpt-oss-20b",
            max_output_tokens=128,
            reasoning_effort="low",
            temperature=0.1,
            top_p=0.9,
            seed=7,
            service_tier="flex",
        ),
        api_key="test-key",
    )

    asyncio.run(llm.complete(messages=[{"role": "user", "content": "do the thing"}]))

    assert captured_kwargs["max_tokens"] == 128
    assert captured_kwargs["reasoning"] == {"effort": "low"}
    assert captured_kwargs["temperature"] == 0.1
    assert captured_kwargs["top_p"] == 0.9
    assert captured_kwargs["seed"] == 7
    assert captured_kwargs["service_tier"] == "flex"


def test_openrouter_complete_forwards_configured_service_tier(monkeypatch):
    captured_kwargs: dict[str, object] = {}

    class _FakeChat:
        async def send_async(self, **kwargs):
            captured_kwargs.update(kwargs)
            return _FakeCompletion(
                choices=[
                    _FakeChoice(
                        message=_FakeMessage(
                            content=None,
                            tool_calls=[],
                        )
                    )
                ]
            )

    class _FakeOpenRouterClient:
        def __init__(self, **kwargs):
            del kwargs
            self.chat = _FakeChat()

    monkeypatch.setattr(open_router_module, "OpenRouterClient", _FakeOpenRouterClient)

    llm = open_router_module.OpenRouter(
        config=OpenRouterConfig(
            model_name="openrouter/openai/gpt-oss-20b",
            service_tier="flex",
        ),
        api_key="test-key",
    )

    asyncio.run(llm.complete(messages=[{"role": "user", "content": "do the thing"}]))

    assert captured_kwargs["service_tier"] == "flex"


def test_openrouter_complete_forwards_provider_require_parameters(monkeypatch):
    captured_kwargs: dict[str, object] = {}

    class _FakeChat:
        async def send_async(self, **kwargs):
            captured_kwargs.update(kwargs)
            return _FakeCompletion(
                choices=[
                    _FakeChoice(
                        message=_FakeMessage(
                            content=None,
                            tool_calls=[],
                        )
                    )
                ]
            )

    class _FakeOpenRouterClient:
        def __init__(self, **kwargs):
            del kwargs
            self.chat = _FakeChat()

    monkeypatch.setattr(open_router_module, "OpenRouterClient", _FakeOpenRouterClient)

    llm = open_router_module.OpenRouter(
        config=OpenRouterConfig(
            model_name="openrouter/openai/gpt-oss-20b",
            provider_kwargs={"require_parameters": True},
        ),
        api_key="test-key",
    )

    asyncio.run(llm.complete(messages=[{"role": "user", "content": "do the thing"}]))

    assert captured_kwargs["provider"] == {"require_parameters": True}


def test_openrouter_complete_allows_explicit_extra_body_tool_choice(monkeypatch):
    captured_kwargs: dict[str, object] = {}

    class _FakeChat:
        async def send_async(self, **kwargs):
            captured_kwargs.update(kwargs)
            return _FakeCompletion(
                choices=[
                    _FakeChoice(
                        message=_FakeMessage(
                            content=None,
                            tool_calls=[],
                        )
                    )
                ]
            )

    class _FakeOpenRouterClient:
        def __init__(self, **kwargs):
            del kwargs
            self.chat = _FakeChat()

    monkeypatch.setattr(open_router_module, "OpenRouterClient", _FakeOpenRouterClient)

    llm = open_router_module.OpenRouter(
        config=OpenRouterConfig(
            model_name="openrouter/openai/gpt-oss-20b",
            provider_kwargs={"extra_body": {"tool_choice": "required"}},
        ),
        api_key="test-key",
    )

    asyncio.run(
        llm.complete(
            messages=[{"role": "user", "content": "do the thing"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "verify",
                        "description": "verify",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        )
    )

    assert captured_kwargs["tool_choice"] == "required"


def test_openrouter_config_builds_valid_sdk_request(monkeypatch):
    from src.harness import core

    captured_kwargs: dict[str, object] = {}
    send_signature = inspect.signature(
        open_router_module.OpenRouterClient(api_key="test-key").chat.send_async
    )

    class _FakeChat:
        async def send_async(self, **kwargs):
            send_signature.bind(**kwargs)
            components.ChatRequest(
                messages=utils.get_pydantic_model(
                    kwargs["messages"], list[components.ChatMessages]
                ),
                max_tokens=kwargs["max_tokens"],
                model=kwargs["model"],
                provider=utils.get_pydantic_model(
                    kwargs["provider"], components.ProviderPreferences | None
                ),
                reasoning=utils.get_pydantic_model(
                    kwargs["reasoning"], components.Reasoning | None
                ),
                stream=kwargs["stream"],
                tools=utils.get_pydantic_model(
                    kwargs["tools"], list[components.ChatFunctionTool] | None
                ),
            )
            captured_kwargs.update(kwargs)
            return _FakeCompletion(
                choices=[
                    _FakeChoice(
                        message=_FakeMessage(
                            content=None,
                            tool_calls=[],
                        )
                    )
                ]
            )

    class _FakeOpenRouterClient:
        def __init__(self, **kwargs):
            del kwargs
            self.chat = _FakeChat()

    monkeypatch.setattr(open_router_module, "OpenRouterClient", _FakeOpenRouterClient)
    llm = open_router_module.OpenRouter(
        config=OpenRouterConfig(
            model_name="deepseek/deepseek-v4-flash",
            max_output_tokens=16384,
            reasoning_effort="low",
            timeout_seconds=300.0,
            base_url="https://openrouter.ai/api/v1",
            provider_kwargs={
                "require_parameters": True,
                "provider": {
                    "order": ["gmicloud"],
                    "allow_fallbacks": True,
                    "ignore": ["siliconflow", "parasail", "morph", "venice"],
                },
            },
        ),
        api_key="test-key",
    )

    asyncio.run(
        llm.complete(
            messages=[{"role": "user", "content": "do the thing"}],
            tools=core.build_tool_specs(),
        )
    )

    assert captured_kwargs["provider"] == {
        "order": ["gmicloud"],
        "allow_fallbacks": True,
        "ignore": ["siliconflow", "parasail", "morph", "venice"],
        "require_parameters": True,
    }
    assert "max_tokens" in captured_kwargs
    assert "max_completion_tokens" not in captured_kwargs
    assert "tool_choice" not in captured_kwargs
    assert "parallel_tool_calls" not in captured_kwargs
    assert "require_parameters" not in captured_kwargs
    assert "temperature" not in captured_kwargs


def test_openrouter_complete_retries_embedded_provider_error(monkeypatch):
    completions = [
        _FakeCompletion(
            choices=[
                _FakeChoice(
                    message=_FakeMessage(
                        content=None,
                        tool_calls=None,
                        provider_specific_fields={
                            "error": {
                                "code": 502,
                                "message": (
                                    "Upstream error from Groq: Tool choice is "
                                    "required, but model did not call a tool"
                                ),
                            }
                        },
                    ),
                    finish_reason="stop",
                )
            ]
        ),
        _FakeCompletion(
            choices=[
                _FakeChoice(
                    message=_FakeMessage(
                        content=None,
                        tool_calls=[
                            _FakeToolCall(
                                function=_FakeFunction(
                                    name="verify",
                                    arguments="{}",
                                )
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ]
        ),
    ]
    captured_kwargs: list[dict[str, object]] = []

    class _FakeChat:
        async def send_async(self, **kwargs):
            captured_kwargs.append(kwargs)
            return completions.pop(0)

    class _FakeOpenRouterClient:
        def __init__(self, **kwargs):
            del kwargs
            self.chat = _FakeChat()

    monkeypatch.setattr(open_router_module, "OpenRouterClient", _FakeOpenRouterClient)
    monkeypatch.setattr(open_router_module.asyncio, "sleep", _fast_sleep)

    llm = open_router_module.OpenRouter(
        config=OpenRouterConfig(
            model_name="openrouter/openai/gpt-oss-20b",
            reasoning_effort="low",
        ),
        api_key="test-key",
    )

    completion = asyncio.run(
        llm.complete(
            messages=[{"role": "user", "content": "do the thing"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "verify",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )
    )

    assert completion.finish_reason == "tool_calls"
    assert len(captured_kwargs) == 2
