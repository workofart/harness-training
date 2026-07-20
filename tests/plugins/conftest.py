"""Shared scaffolding for the cache-plugin tests."""

from __future__ import annotations

import pytest

from src.plugins.caching import store as cache


@pytest.fixture
def store_env(monkeypatch):
    """Force caching enabled for tests that exercise the store.

    Isolation onto a tmp db (and closing it afterwards) already comes from the
    autouse ``_isolate_cache`` fixture in ``tests/conftest.py``; this only lifts
    the ``FRAMEWORK_CACHE=0`` opt-out so the cache path actually runs.
    """
    monkeypatch.setattr(cache, "_DISABLED", False)
