"""Content-addressed, cross-run cache for deterministic LLM completions.

Memoizes the ``CompletionBackend.complete`` boundary. ``maybe_wrap`` is the seam,
applied by the completion-backend factory when the sampler asks for caching; policy
stays unaware while measurement reads provenance. A hit returns the byte-identical
policy payload; only provenance and measured ``llm_latency_sec`` differ. Writes are
idempotent (``INSERT OR IGNORE``), so concurrent writers of the same key store
identical bytes -- no conflict, no lost update.

Key composition and the fail-open contract: src/plugins/README.md.
"""

from __future__ import annotations

import dataclasses
import json

from src.llm.backend import (
    Completion,
    CompletionBackend,
    CompletionRequest,
    FrameworkError,
    ToolCall,
    Usage,
)
from src.plugins.caching import store as cache


def _deserialize(text: str) -> Completion:
    data = json.loads(text)
    return Completion(
        tool_calls=tuple(ToolCall(**call) for call in data.get("tool_calls", ())),
        content=data.get("content"),
        finish_reason=data.get("finish_reason"),
        usage=Usage(**data.get("usage", {})),
        reasoning_content=data.get("reasoning_content"),
        response=data.get("response", {}),
        served_from_cache=True,
    )


class CachingCompletionBackend(CompletionBackend):
    """Memoizes ``inner.complete`` on the backend's own request identity.

    Storage goes through the shared fail-open facade (``cache.get``/``put``); its
    own ``close`` only closes the wrapped backend -- the store is shared and
    closed at process exit.
    """

    def __init__(self, inner: CompletionBackend, *, revision: str) -> None:
        self._inner = inner
        self._revision = revision

    async def complete(self, request: CompletionRequest) -> Completion:
        key = f"c:{self._revision}:{self._inner.cache_key(request)}"
        hit = await cache.get(key)
        if hit is not None:
            try:
                return _deserialize(hit)
            except Exception:
                pass  # corrupt row -> fall through to a live call
        completion = await self._inner.complete(request)
        try:
            text = json.dumps(dataclasses.asdict(completion), separators=(",", ":"))
        except Exception as exc:
            # Serialization failure is a framework defect, not a fail-open cache error.
            raise FrameworkError(str(exc)) from exc
        await cache.put(key, text)
        return completion

    async def close(self) -> None:
        await self._inner.close()


def maybe_wrap(inner: CompletionBackend, *, revision: str) -> CompletionBackend:
    """Wrap ``inner`` with the completion cache unless disabled by env."""
    if cache.disabled():
        return inner
    return CachingCompletionBackend(inner, revision=revision)
