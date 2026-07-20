from __future__ import annotations

import hashlib

from src.env.terminal_bench import _env_fingerprint
from src.env.netcache import CACHE_PROXY_SETUP_SCRIPT as _SETUP_SCRIPT


def test_terminal_bench_setup_script_sha256() -> None:
    # Re-pinning requires an explicit replay-epoch/schema decision.
    assert hashlib.sha256(_SETUP_SCRIPT.read_bytes()).hexdigest() == (
        "de140795c5febe15d2579db4a3f7bdc7a1f2afaac126d0255894992d24ae7bc2"
    )


def test_terminal_bench_env_fingerprint() -> None:
    assert _env_fingerprint() == (
        "2fd12b88aafdd04a52c298e3940bcb189f9766d6:17df03da5634"
    )
