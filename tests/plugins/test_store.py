"""Contract tests for the shared cache facade (src/cache.py): store
initialization failures propagate; operations on an initialized store fail
open with a logged warning."""

from __future__ import annotations

import asyncio
import gc
import logging
import sqlite3
import warnings

import pytest

from src.plugins.caching import store as cache


def test_public_roundtrip(store_env):
    assert asyncio.run(cache.get("k")) is None
    asyncio.run(cache.put("k", "v"))
    assert asyncio.run(cache.get("k")) == "v"
    assert asyncio.run(cache.get_counter("env:epoch:ns")) == 0


def test_store_close_releases_worker_connections(tmp_path):
    gc.collect()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        store = cache._SqliteStore(tmp_path / "cache.db")
        asyncio.run(store.get("missing"))
        store.close()
        del store
        gc.collect()

    assert not [warning for warning in caught if warning.category is ResourceWarning]


def test_store_init_failure_is_hard(tmp_path, monkeypatch):
    # A substrate that cannot open must crash, not impersonate a cold cache:
    # an all-miss measurement would silently skew promotion decisions.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    monkeypatch.setattr(cache, "DB_PATH", blocker / "sub" / "llm_cache.db")
    monkeypatch.setattr(cache, "_DISABLED", False)
    monkeypatch.setattr(cache, "_STORE", None)

    with pytest.raises(OSError):
        asyncio.run(cache.get("k"))
    with pytest.raises(OSError):
        cache.store()


def test_ops_fail_open_and_warn_on_storage_errors(store_env, monkeypatch, caplog):
    async def boom(*args):
        raise sqlite3.OperationalError("disk I/O error")

    store = cache.store()
    monkeypatch.setattr(store, "get", boom)
    monkeypatch.setattr(store, "put", boom)
    monkeypatch.setattr(store, "get_counter", boom)

    with caplog.at_level(logging.WARNING, logger="src.plugins.caching.store"):
        assert asyncio.run(cache.get("k")) is None
        asyncio.run(cache.put("k", "v"))
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            asyncio.run(cache.get_counter("k"))
    warnings = [
        r.getMessage() for r in caplog.records if "failed open" in r.getMessage()
    ]
    assert len(warnings) == 2


def test_disabled_cache_never_touches_storage(monkeypatch):
    monkeypatch.setattr(cache, "_DISABLED", True)
    monkeypatch.setattr(cache, "_STORE", None)

    assert asyncio.run(cache.get("k")) is None
    asyncio.run(cache.put("k", "v"))
    assert asyncio.run(cache.get_counter("k")) == 0
    assert cache._STORE is None


def test_get_counter_reads_externally_managed_epochs(store_env):
    # Namespace epochs have no in-process writer; operators bump them directly
    # in SQLite, and existing nonzero epochs must stay readable.
    asyncio.run(cache.put("seed", "row"))
    conn = sqlite3.connect(cache.DB_PATH)
    conn.execute("INSERT INTO counters(key, value) VALUES('env:epoch:ns', 3)")
    conn.commit()
    conn.close()

    assert asyncio.run(cache.get_counter("env:epoch:ns")) == 3
