from __future__ import annotations

import asyncio

import pytest

import src.concurrency as concurrency


def test_cpu_limiter_waits_for_cpu_headroom(monkeypatch) -> None:
    samples = iter([90.0, 84.0])
    intervals: list[float] = []

    def cpu_percent(interval: float) -> float:
        intervals.append(interval)
        return next(samples)

    monkeypatch.setattr(concurrency.psutil, "cpu_percent", cpu_percent)
    monkeypatch.setattr(concurrency, "ADMISSION_INTERVAL_SEC", 0)

    async def run() -> None:
        limiter = concurrency.CpuConcurrencyLimiter(2)
        entered = asyncio.Event()

        async def enter_second() -> None:
            async with limiter:
                entered.set()

        async with limiter:
            task = asyncio.create_task(enter_second())
            await asyncio.wait_for(entered.wait(), timeout=1.0)
        await task

    asyncio.run(run())

    assert intervals == [
        concurrency.CPU_SAMPLE_INTERVAL_SEC,
        concurrency.CPU_SAMPLE_INTERVAL_SEC,
    ]


def test_cpu_limiter_staggers_cold_start_admissions(monkeypatch) -> None:
    # CPU idle at cold start, so only the admission ramp bounds the burst.
    monkeypatch.setattr(concurrency.psutil, "cpu_percent", lambda interval: 0.0)
    monkeypatch.setattr(concurrency, "ADMISSION_INTERVAL_SEC", 0.05)

    async def run() -> list[float]:
        limiter = concurrency.CpuConcurrencyLimiter(10)
        loop = asyncio.get_running_loop()
        admitted_at: list[float] = []

        async def admit() -> None:
            async with limiter:
                admitted_at.append(loop.time())
                await asyncio.sleep(0.5)

        await asyncio.gather(*(admit() for _ in range(4)))
        return admitted_at

    admitted_at = asyncio.run(run())
    gaps = [b - a for a, b in zip(admitted_at, admitted_at[1:])]
    assert all(gap >= 0.05 * 0.9 for gap in gaps), gaps


def test_cross_process_task_lock_serializes_same_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(concurrency, "_TASK_LOCK_DIR", tmp_path)

    async def run() -> None:
        waits: list[str] = []
        held = asyncio.Event()
        release = asyncio.Event()
        contender_entered = asyncio.Event()

        async def holder() -> None:
            async with concurrency.cross_process_task_lock("terminal_bench:task/a"):
                held.set()
                await release.wait()

        async def contender() -> None:
            async with concurrency.cross_process_task_lock(
                "terminal_bench:task/a", on_wait=lambda: waits.append("waited")
            ):
                contender_entered.set()

        holder_task = asyncio.create_task(holder())
        await asyncio.wait_for(held.wait(), timeout=2.0)
        contender_task = asyncio.create_task(contender())
        await asyncio.sleep(0.1)
        assert waits == ["waited"]
        assert not contender_entered.is_set()
        release.set()
        await asyncio.wait_for(contender_task, timeout=2.0)
        await holder_task

    asyncio.run(run())
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "terminal_bench_task_a.lock"
    ]


def test_cross_process_task_lock_distinct_keys_do_not_block(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(concurrency, "_TASK_LOCK_DIR", tmp_path)

    async def run() -> None:
        async with concurrency.cross_process_task_lock(
            "terminal_bench:task-a",
            on_wait=lambda: pytest.fail("distinct keys must not contend"),
        ):
            async with concurrency.cross_process_task_lock(
                "terminal_bench:task-b",
                on_wait=lambda: pytest.fail("distinct keys must not contend"),
            ):
                pass

    asyncio.run(run())


def test_cpu_limiter_enforces_hard_ceiling(monkeypatch) -> None:
    monkeypatch.setattr(concurrency.psutil, "cpu_percent", lambda interval: 0.0)
    monkeypatch.setattr(concurrency, "ADMISSION_INTERVAL_SEC", 0)

    async def run() -> None:
        limiter = concurrency.CpuConcurrencyLimiter(1)
        entered = asyncio.Event()

        async def enter_second() -> None:
            async with limiter:
                entered.set()

        async with limiter:
            task = asyncio.create_task(enter_second())
            await asyncio.sleep(0)
            assert not entered.is_set()
        await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(run())
