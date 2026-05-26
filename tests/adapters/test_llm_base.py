from __future__ import annotations

import inspect

from src.adapters.llm_base import BaseLlm


class _FakeLlm(BaseLlm):
    @property
    def max_context_length(self) -> int:
        return 1000

    async def complete(self, *, messages, tools=None, reasoning_effort=None):
        del messages, tools, reasoning_effort
        return None

    def get_token_count(self, messages, *, tools=None) -> int:
        del messages, tools
        return 42


def test_base_llm_exposes_token_count_contract():
    llm = _FakeLlm()

    assert llm.get_token_count([{"role": "user", "content": "hi"}], tools=[]) == 42


def test_base_llm_contract_surface_is_transport_only():
    assert set(BaseLlm.__abstractmethods__) == {
        "max_context_length",
        "complete",
        "get_token_count",
    }
    assert inspect.iscoroutinefunction(BaseLlm.close)
