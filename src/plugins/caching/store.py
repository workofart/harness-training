"""Fail-open SQLite cache substrate shared across framework caches.

Storage is one WAL-mode SQLite file shared by every process on the host. The "pool"
is a small executor with a thread-local connection per worker: WAL gives lock-free
concurrent reads plus a single serialized writer, and ``busy_timeout`` absorbs the
rare cross-process write contention.

Failure contract: store initialization errors propagate. Value reads/writes fail open;
epoch counter reads fail closed because an invented zero changes measurement identity.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Keep cache data outside agent-read artifacts; worktrees share it via FRAMEWORK_CACHE_DB.
DB_PATH = (
    Path(os.environ["FRAMEWORK_CACHE_DB"])
    if "FRAMEWORK_CACHE_DB" in os.environ
    else Path(__file__).resolve().parents[3] / "cache" / "llm_cache.db"
)
_DISABLED = os.environ.get("FRAMEWORK_CACHE", "1") == "0"
_POOL_WORKERS = 4
_BUSY_TIMEOUT_MS = 30_000
_LOGGER = logging.getLogger(__name__)

# Owned here so every counters writer (store init, bump_epoch) shares one schema.
COUNTERS_DDL = (
    "CREATE TABLE IF NOT EXISTS counters (key TEXT PRIMARY KEY, value INTEGER NOT NULL)"
)


class _SqliteStore:
    """Process-global WAL-mode key/value store; one thread-local connection per worker."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._connections_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=_POOL_WORKERS, thread_name_prefix="llm-cache"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        # WAL is file-persistent; set it once and let worker connections inherit it.
        conn = sqlite3.connect(path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.execute(COUNTERS_DDL)
            conn.commit()
        finally:
            conn.close()

    def _conn(self) -> sqlite3.Connection:
        """The calling worker thread's own connection, opened once and reused.

        Queries stay on the creating worker, so no shared-connection locking is
        needed. The owner only crosses threads to close connections after joining
        every worker; WAL + busy_timeout handle query concurrency.
        """
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                self._path,
                timeout=_BUSY_TIMEOUT_MS / 1000,
                check_same_thread=False,
            )
            conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
            with self._connections_lock:
                self._connections.append(conn)
        return conn

    async def get(self, key: str) -> str | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, self._get, key)

    def _get(self, key: str) -> str | None:
        row = (
            self._conn()
            .execute("SELECT value FROM cache WHERE key=?", (key,))
            .fetchone()
        )
        return None if row is None else row[0]

    async def put(self, key: str, value: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._pool, self._put, key, value)

    def _put(self, key: str, value: str) -> None:
        # Same-key writes are identical; first commit wins, later writers no-op.
        conn = self._conn()
        conn.execute(
            "INSERT OR IGNORE INTO cache(key, value) VALUES(?, ?)",
            (key, value),
        )
        conn.commit()

    async def get_counter(self, key: str) -> int:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, self._get_counter, key)

    def _get_counter(self, key: str) -> int:
        row = (
            self._conn()
            .execute("SELECT value FROM counters WHERE key=?", (key,))
            .fetchone()
        )
        return 0 if row is None else int(row[0])

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        for conn in self._connections:
            conn.close()
        self._connections.clear()


_STORE: _SqliteStore | None = None
_STORE_LOCK = threading.Lock()


def store() -> _SqliteStore:
    """The process-global store, created on first use and closed at process exit."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = _SqliteStore(DB_PATH)
            atexit.register(_STORE.close)
    return _STORE


def disabled() -> bool:
    return _DISABLED


def _enabled_store() -> _SqliteStore | None:
    return None if _DISABLED else store()


async def get(key: str) -> str | None:
    """Fail-open read: disabled or erroring storage reads as a miss."""
    store = _enabled_store()
    if store is None:
        return None
    try:
        return await store.get(key)
    except Exception as exc:
        _LOGGER.warning("cache get failed open (miss): %r", exc)
        return None


async def put(key: str, value: str) -> None:
    """Fail-open write: a cache write must never fail the caller."""
    store = _enabled_store()
    if store is None:
        return
    try:
        await store.put(key, value)
    except Exception as exc:
        _LOGGER.warning("cache put failed open (dropped): %r", exc)


async def get_counter(key: str) -> int:
    """Read a cache namespace counter; storage failures are measurement failures."""
    store = _enabled_store()
    if store is None:
        return 0
    return await store.get_counter(key)
