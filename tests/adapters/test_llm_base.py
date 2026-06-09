from __future__ import annotations

import inspect

from src.llm.base import BaseLlm, LlmCompletion


class _FakeLlm(BaseLlm):
    async def complete(self, *, messages, tools=None, reasoning_effort=None):
        del messages, tools, reasoning_effort
        return LlmCompletion()


def test_base_llm_contract_surface_is_transport_only():
    # The harness only ever calls complete()/close(): `complete` is the sole
    # abstract hook, so a fake implementing just it is instantiable.
    assert set(BaseLlm.__abstractmethods__) == {"complete"}
    assert inspect.iscoroutinefunction(BaseLlm.close)
    assert isinstance(_FakeLlm(), BaseLlm)
