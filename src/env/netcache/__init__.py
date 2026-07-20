"""Terminal-Bench host network-cache bring-up."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path


CACHE_PROXY_SETUP_SCRIPT = Path(__file__).with_name("setup-terminal-bench-container.sh")
_HOST_CACHES_COMPOSE_FILE = Path(__file__).with_name("docker-compose.caches.yml")
_HOST_CACHE_LOCK = asyncio.Lock()
_host_caches_started = False
_NETWORK_CACHE_SCOPE_ENV = "FRAMEWORK_NETWORK_CACHE_SCOPE"


class ContainerNetworkCache:
    """Everything one Docker env needs from the host network cache.

    Owns host cache bring-up, the pinned snapshot token, and the setup-command
    fragment the in-container proxy script consumes. The token is opaque here:
    callers decide what a snapshot is. The proxy script itself is injected
    even when the cache is disabled: without the exports it probes and
    self-disables, keeping container setup a pure function of task content
    plus snapshot.
    """

    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled
        self._scope_token: str | None = None

    def pin(self, token: str) -> bool:
        """Freeze the network snapshot; False (no-op) when the cache is off."""
        if not self._enabled:
            return False
        self._scope_token = token
        return True

    async def ensure_host_caches(self) -> None:
        if self._enabled:
            await _ensure_host_caches_started()

    def setup_command(self, *, base_command: str) -> str:
        cache_exports = ""
        if self._enabled:
            require_caches = "1" if self._scope_token is not None else "0"
            scope = (
                ""
                if self._scope_token is None
                else f"export {_NETWORK_CACHE_SCOPE_ENV}={self._scope_token}\n"
            )
            cache_exports = f"export FRAMEWORK_REQUIRE_CACHES={require_caches}\n{scope}"
        return f"{base_command}\n{cache_exports}" + CACHE_PROXY_SETUP_SCRIPT.read_text()


async def _ensure_host_caches_started() -> None:
    """Bring up the shared apt/PyPI cache containers once per process.

    `docker compose up -d` is idempotent, and the caches are deliberately left
    running after the run: their volumes persist regardless, and an always-up
    cache keeps the in-container probe outcome (proxy vs direct mirror)
    deterministic across runs.
    """
    global _host_caches_started
    async with _HOST_CACHE_LOCK:
        if _host_caches_started:
            return
        await asyncio.to_thread(_compose_up_host_caches)
        _host_caches_started = True


def _compose_up_host_caches() -> None:
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(_HOST_CACHES_COMPOSE_FILE),
            "up",
            "-d",
            # Avoid proxy restarts when per-worktree bind paths change the Compose hash.
            "--no-recreate",
            "--wait",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Terminal-Bench host cache startup failed: {detail}")
