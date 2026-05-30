from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import src.adapters.chatgpt_codex as chatgpt_codex_module
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
