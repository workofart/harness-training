"""Tests for src/adapters/infra_retry.py."""

from __future__ import annotations

import asyncio

import pytest

from src.adapters import infra_retry
from src.adapters.infra_retry import retry_transient


def _no_sleep(monkeypatch) -> None:
    async def _sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", _sleep)


def test_retries_transient_until_success(monkeypatch):
    _no_sleep(monkeypatch)
    attempts = 0
    retries: list[int] = []

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("flaky")
        return "ok"

    result = asyncio.run(
        retry_transient(
            operation,
            is_transient=lambda _exc: True,
            on_retry=lambda n, _exc: retries.append(n),
        )
    )

    assert result == "ok"
    assert attempts == 3  # one initial attempt plus INFRA_RETRY_BUDGET (2) retries
    assert retries == [1, 2]


def test_raises_after_budget_exhausted(monkeypatch):
    _no_sleep(monkeypatch)
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("always")

    with pytest.raises(ValueError, match="always"):
        asyncio.run(
            retry_transient(
                operation,
                is_transient=lambda _exc: True,
                on_retry=lambda _n, _exc: None,
            )
        )

    assert attempts == infra_retry.INFRA_RETRY_BUDGET + 1


def test_non_transient_error_is_not_retried(monkeypatch):
    _no_sleep(monkeypatch)
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("fatal")

    def _fail(_n, _exc):
        raise AssertionError("a non-transient error must not trigger on_retry")

    with pytest.raises(ValueError, match="fatal"):
        asyncio.run(
            retry_transient(
                operation,
                is_transient=lambda _exc: False,
                on_retry=_fail,
            )
        )

    assert attempts == 1


def test_budget_override(monkeypatch):
    _no_sleep(monkeypatch)
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("x")

    with pytest.raises(ValueError):
        asyncio.run(
            retry_transient(
                operation,
                is_transient=lambda _exc: True,
                on_retry=lambda _n, _exc: None,
                budget=1,
            )
        )

    assert attempts == 2
