"""CPU-aware task admission and cross-process task exclusivity."""

from __future__ import annotations

import asyncio
import fcntl
import os
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import psutil

CPU_TARGET_PERCENT = 85.0
CPU_SAMPLE_INTERVAL_SEC = 0.5
# CPU gating misses daemon-bound Docker creates; stagger cold starts without affecting long-rollout throughput.
ADMISSION_INTERVAL_SEC = 5


class CpuConcurrencyLimiter:
    def __init__(self, ceiling: int) -> None:
        self._slots = asyncio.Semaphore(ceiling)
        self._admission = asyncio.Lock()
        self._active = 0
        self._last_admission = 0.0

    async def __aenter__(self) -> None:
        await self._slots.acquire()
        try:
            async with self._admission:
                while self._active and (
                    await asyncio.to_thread(psutil.cpu_percent, CPU_SAMPLE_INTERVAL_SEC)
                    >= CPU_TARGET_PERCENT
                ):
                    pass
                loop = asyncio.get_running_loop()
                delay = self._last_admission + ADMISSION_INTERVAL_SEC - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                self._active += 1
                self._last_admission = loop.time()
        except BaseException:
            self._slots.release()
            raise

    async def __aexit__(self, *args: object) -> None:
        self._active -= 1
        self._slots.release()


_TASK_LOCK_DIR = Path.home() / ".cache" / "harness-experiment" / "task-locks"


@asynccontextmanager
async def cross_process_task_lock(
    key: str, *, on_wait: Callable[[], None] | None = None
) -> AsyncIterator[None]:
    """Hold a host-wide exclusive lock for one task rollout.

    Terminal-Bench pins each task to a fixed docker subnet (deterministic
    container IPs), so the same task rolled out by two overlapping experiment
    processes would collide on network creation. The flock spans the caller's
    whole env lifetime and is released at process exit even after a kill.
    ``on_wait`` fires once if another process already holds the key.
    """
    _TASK_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    path = _TASK_LOCK_DIR / (re.sub(r"[^A-Za-z0-9._-]", "_", key) + ".lock")
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            if on_wait is not None:
                on_wait()
            await asyncio.to_thread(fcntl.flock, fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)
