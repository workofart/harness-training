"""Measurement subprocess entrypoint."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from collections.abc import Callable
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import src
from src.config import RunConfig
from src.rollout.store import RunStore

_NOISY_DEPENDENCY_LOGGERS = ("httpx", "httpcore")
HEARTBEAT_INTERVAL_SEC = 60.0


async def _emit_heartbeats(observer: _PipeObserver) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)
        observer.measurement_heartbeat()


def _assert_src_from_cwd() -> None:
    assert Path(src.__file__).resolve().parents[1] == Path.cwd().resolve()


def _assert_cache_ready(run_config: RunConfig) -> None:
    """Spawn contract: the parent resolves the shared cache DB for this child
    (measurement sets FRAMEWORK_CACHE_DB), and the substrate must open before any
    measurement -- a broken cache dies here instead of running a silent
    all-miss experiment that skews promotion decisions."""
    assert "FRAMEWORK_CACHE_DB" in os.environ
    plugins = run_config.plugins
    if not (plugins.llm_cache or plugins.execution == "replay"):
        return

    from src.plugins.caching import store as cache

    if not cache.disabled():
        cache.store()


def _install_run_log(path: Path) -> None:
    """Route the process's logging to the run's log file; stdout stays event-only."""
    handler = logging.FileHandler(path)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().handlers = [handler]


class _PipeObserver:
    """Forwards observer events as JSON bytes: this process runs
    candidate-editable code, so the trusted parent must never unpickle its
    pipe data."""

    def __init__(self, events: Connection, tracker: RunStore) -> None:
        self._events = events
        self._tracker = tracker

    def log(self, line: str) -> None:
        self._events.send_bytes(json.dumps(["log", line]).encode())

    def experiment_started(self, experiment_id: str) -> None:
        _install_run_log(self._tracker.run_log_path(experiment_id))
        self._events.send_bytes(json.dumps(["started", experiment_id]).encode())

    def task_finished(self, task_id: str, failure_mode: str) -> None:
        self._events.send_bytes(json.dumps(["task", task_id, failure_mode]).encode())

    def measurement_heartbeat(self) -> None:
        self._events.send_bytes(json.dumps({"event": "heartbeat"}).encode())


def _make_quiet_unraisablehook(
    previous: Callable[[Any], Any],
) -> Callable[[Any], None]:
    def hook(unraisable: Any) -> None:
        exc = getattr(unraisable, "exc_value", None)
        obj = getattr(unraisable, "object", None)
        if (
            isinstance(exc, ValueError)
            and "I/O operation on closed file" in str(exc)
            and type(obj).__module__.split(".", 1)[0] == "urllib3"
        ):
            return
        previous(unraisable)

    return hook


def _experiment_worker(
    config_path: str,
    experiments_root: str,
    events: Connection,
    task_ids: tuple[str, ...],
    experiment_id: str | None,
) -> None:
    root_logger = logging.getLogger()
    root_logger.handlers = [logging.NullHandler()]
    root_logger.setLevel(logging.WARNING)
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    try:
        from src.rollout.sampler import run_experiment

        for logger_name in _NOISY_DEPENDENCY_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        sys.unraisablehook = _make_quiet_unraisablehook(sys.unraisablehook)

        tracker = RunStore(Path(experiments_root))
        run_config = RunConfig.load(config_path)
        _assert_cache_ready(run_config)
        run_config = run_config.with_task_panel(task_ids)
        observer = _PipeObserver(events, tracker)

        async def sample() -> None:
            heartbeat = asyncio.create_task(_emit_heartbeats(observer))
            try:
                await run_experiment(
                    run_config=run_config,
                    tracker=tracker,
                    observer=observer,
                    experiment_id=experiment_id,
                )
            finally:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)

        # Cleanup is owned by docker_shell's atexit hook and the parent's post-reap sweep.
        asyncio.run(sample())
    finally:
        events.close()


if __name__ == "__main__":
    _assert_src_from_cwd()
    (
        config_path,
        experiments_root,
        events_fd,
        serialized_task_ids,
        serialized_experiment_id,
    ) = sys.argv[1:]
    _experiment_worker(
        config_path,
        experiments_root,
        Connection(int(events_fd)),
        tuple(json.loads(serialized_task_ids)),
        json.loads(serialized_experiment_id),
    )
