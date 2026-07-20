from __future__ import annotations

import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.env.netcache.uv_python_mirror import make_handler


def test_uv_python_mirror_caches_successful_artifact(tmp_path: Path) -> None:
    with _mirror_scenario(
        cache_dir=tmp_path,
        artifacts={"/20240713/cpython.tar.gz": b"python-build"},
        path="/20240713/cpython.tar.gz",
    ) as (upstream, url):
        assert urllib.request.urlopen(url, timeout=2).read() == b"python-build"
        first_hits = upstream.hits

    with _mirror_scenario(
        cache_dir=tmp_path,
        artifacts={},
        path="/20240713/cpython.tar.gz",
    ) as (upstream, url):
        assert urllib.request.urlopen(url, timeout=2).read() == b"python-build"
        assert upstream.hits == 0

    assert first_hits == 1


def test_uv_python_mirror_follows_release_redirect(tmp_path: Path) -> None:
    with _mirror_scenario(
        cache_dir=tmp_path,
        artifacts={
            "/20240713/cpython.tar.gz": (302, b"", "/objects/cpython.tar.gz"),
            "/objects/cpython.tar.gz": (200, b"redirected-python", None),
        },
        path="/20240713/cpython.tar.gz",
    ) as (_upstream, url):
        assert urllib.request.urlopen(url, timeout=2).read() == b"redirected-python"

    assert (tmp_path / "20240713" / "cpython.tar.gz").read_bytes() == (
        b"redirected-python"
    )


def test_uv_python_mirror_does_not_cache_negative_responses(
    tmp_path: Path,
) -> None:
    with _mirror_scenario(
        cache_dir=tmp_path,
        artifacts={},
        path="/missing.tar.gz",
    ) as (upstream, url):
        for _ in range(2):
            try:
                urllib.request.urlopen(url, timeout=2)
            except urllib.error.HTTPError as exc:
                assert exc.code == 404
            else:
                raise AssertionError("expected missing artifact to stay uncached")

        assert upstream.hits == 2

    assert list(tmp_path.rglob("*")) == []


def test_uv_python_mirror_relays_non_standard_upstream_status(
    tmp_path: Path,
) -> None:
    with _mirror_scenario(
        cache_dir=tmp_path,
        artifacts={"/cdn.tar.gz": (530, b"cdn unavailable", None)},
        path="/cdn.tar.gz",
    ) as (_upstream, url):
        try:
            urllib.request.urlopen(url, timeout=2)
        except urllib.error.HTTPError as exc:
            assert exc.code == 530
            assert b"HTTP Error 530" in exc.read()
        else:
            raise AssertionError("expected CDN error to be relayed")

    assert list(tmp_path.rglob("*")) == []


@contextmanager
def _mirror_scenario(
    *,
    cache_dir: Path,
    artifacts: dict[str, bytes | tuple[int, bytes, str | None]],
    path: str,
) -> Iterator[tuple[ThreadingHTTPServer, str]]:
    with _upstream_server(artifacts) as upstream:
        with _mirror_server(
            cache_dir=cache_dir,
            upstream_base_url=f"http://127.0.0.1:{upstream.server_address[1]}",
        ) as server:
            yield upstream, f"http://127.0.0.1:{server.server_address[1]}{path}"


@contextmanager
def _mirror_server(
    *,
    cache_dir: Path,
    upstream_base_url: str,
) -> Iterator[ThreadingHTTPServer]:
    handler = make_handler(cache_dir=cache_dir, upstream_base_url=upstream_base_url)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    with _running_server(server):
        yield server


@contextmanager
def _upstream_server(
    artifacts: dict[str, bytes | tuple[int, bytes, str | None]],
) -> Iterator[ThreadingHTTPServer]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.server.hits += 1
            artifact = artifacts.get(self.path)
            if artifact is None:
                status, body, location = 404, b"", None
            elif isinstance(artifact, tuple):
                status, body, location = artifact
            else:
                status, body, location = 200, artifact, None

            self.send_response(status)
            if location is not None:
                self.send_header("Location", location)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.hits = 0
    with _running_server(server):
        yield server


@contextmanager
def _running_server(server: ThreadingHTTPServer) -> Iterator[ThreadingHTTPServer]:
    # Tight polling avoids the default 0.5s shutdown delay.
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
