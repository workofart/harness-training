"""Bounded retry for transient infrastructure operations inside one trial."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

# Number of retries (beyond the first attempt) for one transient infra operation.
INFRA_RETRY_BUDGET = 2


_T = TypeVar("_T")


def _linear_backoff(retries: int, exc: Exception) -> float:
    del exc
    return float(retries)


async def retry_transient(
    operation: Callable[[], Awaitable[_T]],
    *,
    is_transient: Callable[[Exception], bool],
    on_retry: Callable[[int, Exception], None],
    budget: int = INFRA_RETRY_BUDGET,
    delay: Callable[[int, Exception], float] = _linear_backoff,
) -> _T:
    retries = 0
    while True:
        try:
            return await operation()
        except Exception as exc:
            retries += 1
            if retries > budget or not is_transient(exc):
                raise
            on_retry(retries, exc)
            await asyncio.sleep(delay(retries, exc))
