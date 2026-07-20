#!/usr/bin/env python3
"""Read-through mirror for uv managed-Python artifacts."""

from __future__ import annotations

import contextlib
import os
import posixpath
import shutil
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_UPSTREAM_BASE_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download"
)
DEFAULT_CACHE_DIR = "/var/cache/uv-python-mirror"
DEFAULT_PORT = 8000


def make_handler(
    *,
    cache_dir: Path,
    upstream_base_url: str,
) -> type[BaseHTTPRequestHandler]:
    cache_root = cache_dir.resolve()
    upstream = upstream_base_url.rstrip("/")
    locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)

    class UvPythonMirrorHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if urllib.parse.urlsplit(self.path).path == "/healthz":
                self._send_bytes(200, b"ok\n")
                return

            try:
                cache_key = _cache_key(self.path)
                target = (cache_root / cache_key).resolve()
                target.relative_to(cache_root)

                with locks[cache_key]:
                    # os.replace makes cross-process first-download races atomic.
                    if not target.exists():
                        _fetch_to_cache(
                            upstream_url=f"{upstream}/{cache_key}",
                            target=target,
                        )
                self._send_file(target)
            except urllib.error.HTTPError as exc:
                self._send_bytes(exc.code, str(exc).encode())
            except urllib.error.URLError as exc:
                self._send_bytes(502, str(exc).encode())
            except ValueError as exc:
                self._send_bytes(400, str(exc).encode())

        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _send_file(self, path: Path) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as source:
                shutil.copyfileobj(source, self.wfile)

        def _send_bytes(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return UvPythonMirrorHandler


def _cache_key(request_path: str) -> str:
    raw_path = urllib.parse.urlsplit(request_path).path
    parts = [part for part in raw_path.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError(f"unsupported mirror path: {request_path!r}")
    return posixpath.join(*parts)


def _fetch_to_cache(*, upstream_url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            with urllib.request.urlopen(upstream_url, timeout=300) as response:
                if response.status != 200:
                    raise urllib.error.HTTPError(
                        upstream_url,
                        response.status,
                        response.reason,
                        response.headers,
                        response,
                    )
                shutil.copyfileobj(response, tmp_file)
        os.replace(tmp_name, target)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def main() -> None:
    cache_dir = Path(os.environ.get("UV_PYTHON_MIRROR_CACHE_DIR", DEFAULT_CACHE_DIR))
    cache_dir.mkdir(parents=True, exist_ok=True)
    handler = make_handler(
        cache_dir=cache_dir,
        upstream_base_url=DEFAULT_UPSTREAM_BASE_URL,
    )
    port = int(os.environ.get("UV_PYTHON_MIRROR_PORT", str(DEFAULT_PORT)))
    with ThreadingHTTPServer(("", port), handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
