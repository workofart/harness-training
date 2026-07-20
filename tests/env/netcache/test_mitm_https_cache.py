from __future__ import annotations

import base64
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.env.netcache.mitm_https_cache as mitm_cache

_SCOPE = "task-a-epoch-0"
_GET_URL = "https://huggingface.co/api/models?author=alice"
_POST_URL = "https://huggingface.co/api/datasets/search"
_INSTALL_URL = "https://astral.sh/uv/0.9.5/install.sh"
_DEB_URL = "http://deb.debian.org/debian/pool/main/h/hello/hello_2.10-3_amd64.deb"


def _store(
    cache_dir: Path,
    *,
    method: str = "GET",
    url: str = _INSTALL_URL,
    request_headers: list[tuple[str, str]] | None = None,
    request_body: bytes = b"",
    cache_scope: str = "global",
    status: int = 200,
    headers: list[tuple[str, str]] | None = None,
    body: bytes = b"body",
) -> bool:
    return mitm_cache.store_cacheable_response(
        cache_dir,
        method=method,
        url=url,
        request_headers=request_headers,
        request_body=request_body,
        cache_scope=cache_scope,
        status_code=status,
        headers=headers or [],
        body=body,
    )


def _load(
    cache_dir: Path,
    *,
    method: str = "GET",
    url: str = _INSTALL_URL,
    request_body: bytes = b"",
    cache_scope: str = "global",
):
    return mitm_cache.load_cached_response(
        cache_dir,
        method=method,
        url=url,
        request_body=request_body,
        cache_scope=cache_scope,
    )


def _flow(
    *,
    method: str = "GET",
    url: str = _GET_URL,
    request_headers: dict[str, str] | None = None,
    request_body: bytes | None = None,
    response: bool = False,
    response_headers: dict[str, str] | None = None,
    response_body: bytes = b"live",
    status: int = 200,
    metadata: dict | None = None,
    client_conn=None,
):
    # Real mitmproxy requests always expose raw_content; None means a missing or
    # streamed body.
    request = SimpleNamespace(
        method=method,
        pretty_url=url,
        url=url,
        headers=dict(request_headers or {}),
        raw_content=request_body,
    )
    live_response = None
    if response:
        live_response = SimpleNamespace(
            status_code=status,
            raw_content=response_body,
            content=response_body,
            headers=dict(response_headers or {}),
            stream=False,
        )
    return SimpleNamespace(
        request=request,
        response=live_response,
        metadata={} if metadata is None else metadata,
        client_conn=client_conn,
    )


@pytest.fixture
def addon(tmp_path: Path, monkeypatch) -> mitm_cache.HttpsCacheAddon:
    class FakeResponse:
        @staticmethod
        def make(status_code, body, headers):
            return SimpleNamespace(
                status_code=status_code,
                raw_content=body,
                content=body,
                headers=dict(headers),
            )

    monkeypatch.setattr(
        mitm_cache, "http", SimpleNamespace(Response=FakeResponse), raising=False
    )
    value = mitm_cache.HttpsCacheAddon()
    value.cache_dir = tmp_path
    return value


def _scoped_flow(**kwargs):
    return _flow(metadata={"proxyauth": ("framework", _SCOPE)}, **kwargs)


def test_store_round_trip_filters_transport_headers(tmp_path: Path) -> None:
    assert _store(
        tmp_path,
        headers=[
            ("Content-Type", "text/x-shellscript"),
            ("Content-Length", "7"),
            ("Connection", "close"),
        ],
        body=b"install",
    )
    cached = _load(tmp_path)
    assert (cached.status_code, cached.headers, cached.body) == (
        200,
        [("Content-Type", "text/x-shellscript")],
        b"install",
    )


@pytest.mark.parametrize(
    ("method", "status", "request_headers", "response_headers"),
    [
        pytest.param("GET", 404, None, None, id="non-success"),
        pytest.param("HEAD", 200, None, None, id="unsupported-method"),
        pytest.param("GET", 200, [("Range", "bytes=10-")], None, id="range"),
        pytest.param("GET", 200, None, [("Vary", "User-Agent")], id="vary"),
        pytest.param("GET", 200, None, [("Content-Encoding", "gzip")], id="encoded"),
    ],
)
def test_non_cacheable_responses_leave_no_entry(
    tmp_path: Path,
    method: str,
    status: int,
    request_headers: list[tuple[str, str]] | None,
    response_headers: list[tuple[str, str]] | None,
) -> None:
    assert not _store(
        tmp_path,
        method=method,
        status=status,
        request_headers=request_headers,
        headers=response_headers,
    )
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize(
    ("method", "first", "second", "first_body", "second_body"),
    [
        pytest.param(
            "GET",
            _GET_URL,
            _GET_URL.replace("alice", "bob"),
            b"alice-models",
            b"bob-models",
            id="full-get-url",
        ),
        pytest.param(
            "POST",
            b'{"query":"alpha"}',
            b'{"query":"beta"}',
            b"alpha",
            b"beta",
            id="post-body",
        ),
        pytest.param(
            "SCOPE",
            "task-a-epoch-0",
            "task-b-epoch-0",
            b"task-a",
            b"task-b",
            id="scope",
        ),
    ],
)
def test_cache_key_dimensions_do_not_collide(
    tmp_path: Path,
    method: str,
    first,
    second,
    first_body: bytes,
    second_body: bytes,
) -> None:
    if method == "GET":
        assert _store(tmp_path, url=first, body=first_body)
        assert _store(tmp_path, url=second, body=second_body)
        actual = [_load(tmp_path, url=url).body for url in (first, second)]
    elif method == "POST":
        assert _store(
            tmp_path, method=method, url=_POST_URL, request_body=first, body=first_body
        )
        assert _store(
            tmp_path,
            method=method,
            url=_POST_URL,
            request_body=second,
            body=second_body,
        )
        actual = [
            _load(tmp_path, method=method, url=_POST_URL, request_body=body).body
            for body in (first, second)
        ]
    else:
        assert _store(tmp_path, url=_GET_URL, cache_scope=first, body=first_body)
        assert _store(tmp_path, url=_GET_URL, cache_scope=second, body=second_body)
        actual = [
            _load(tmp_path, url=_GET_URL, cache_scope=scope).body
            for scope in (first, second)
        ]
    assert actual == [first_body, second_body]


def test_signed_github_object_urls_ignore_expiring_query(tmp_path: Path) -> None:
    path = (
        "https://objects.githubusercontent.com/github-production-release-asset-2e65be/"
        "1234/uv-x86_64-unknown-linux-gnu.tar.gz"
    )
    assert _store(tmp_path, url=f"{path}?X-Amz-Signature=old", body=b"uv-binary")
    assert _load(tmp_path, url=f"{path}?X-Amz-Signature=new").body == b"uv-binary"


def test_debian_pool_request_is_pinned_to_configured_mirror(monkeypatch) -> None:
    monkeypatch.setenv(mitm_cache._PINNED_DEBIAN_MIRROR_HOST_ENV, "ftp.us.debian.org")
    flow = _scoped_flow(method="HEAD", url=_DEB_URL)
    mitm_cache.HttpsCacheAddon().request(flow)
    assert flow.metadata[mitm_cache._PINNED_UPSTREAM_ORIGINAL_URL_MARKER] == _DEB_URL
    assert flow.request.url == _DEB_URL.replace("deb.debian.org", "ftp.us.debian.org")
    assert flow.request.headers["host"] == "ftp.us.debian.org"


def test_debian_pin_defaults_to_fixed_mirror_and_downgrades_https() -> None:
    assert mitm_cache._pinned_upstream_url(
        "https://deb.debian.org/debian/pool/main/h/hello/hello_2.10-3_amd64.deb"
    ) == ("http://debian.osuosl.org/debian/pool/main/h/hello/hello_2.10-3_amd64.deb")


def test_pinned_debian_response_is_stored_under_original_url(
    tmp_path: Path, addon: mitm_cache.HttpsCacheAddon
) -> None:
    flow = _flow(
        url=_DEB_URL.replace("deb.debian.org", "debian.osuosl.org"),
        response=True,
        response_body=b"deb",
        metadata={
            mitm_cache._PINNED_UPSTREAM_ORIGINAL_URL_MARKER: _DEB_URL,
            mitm_cache._CACHE_SCOPE_MARKER: _SCOPE,
        },
    )
    addon.response(flow)
    assert _load(tmp_path, url=_DEB_URL, cache_scope=_SCOPE).body == b"deb"


def test_first_writer_wins_even_when_entry_directory_already_exists(
    tmp_path: Path,
) -> None:
    url = "https://sourceforge.net/projects/povray/files/"
    assert _store(tmp_path, url=url, body=b"first")
    assert not _store(tmp_path, url=url, body=b"second")
    assert _load(tmp_path, url=url).body == b"first"

    cache_url = mitm_cache._cache_url(
        method="GET",
        url=_GET_URL,
        request_headers=[],
        request_body=b"",
        response_headers=[],
    )
    entry = mitm_cache._cache_entry_dir(
        tmp_path, cache_scope="global", method="GET", url=cache_url
    )
    entry.mkdir(parents=True)
    assert not _store(tmp_path, url=_GET_URL, body=b"lost-race")


def test_accept_encoding_vary_is_cacheable_after_identity_normalization(
    tmp_path: Path,
) -> None:
    assert _store(
        tmp_path,
        url=_GET_URL,
        headers=[("Vary", "Accept-Encoding"), ("Content-Type", "application/json")],
        body=b'{"models":[]}',
    )
    assert _load(tmp_path, url=_GET_URL).body == b'{"models":[]}'


def test_request_forces_identity_accept_encoding() -> None:
    flow = _scoped_flow(request_headers={"Accept-Encoding": "gzip, deflate, br"})
    mitm_cache.HttpsCacheAddon().request(flow)
    assert flow.request.headers["Accept-Encoding"] == "identity"


def test_cache_hit_is_immutable_and_replays_headers(
    tmp_path: Path, addon: mitm_cache.HttpsCacheAddon
) -> None:
    assert _store(
        tmp_path,
        cache_scope=_SCOPE,
        headers=[("Content-Type", "text/x-shellscript")],
        body=b"install",
    )
    [body_path] = list(tmp_path.rglob("body"))
    before = body_path.stat().st_mtime_ns
    flow = _scoped_flow(url=_INSTALL_URL)
    addon.request(flow)
    addon.response(flow)
    assert flow.metadata[mitm_cache._CACHE_HIT_MARKER] is True
    assert flow.response.headers == {b"Content-Type": b"text/x-shellscript"}
    assert body_path.stat().st_mtime_ns == before


def test_post_hit_uses_request_body_hash(
    tmp_path: Path, addon: mitm_cache.HttpsCacheAddon
) -> None:
    body = b'{"query":"alpha"}'
    assert _store(
        tmp_path,
        method="POST",
        url=_POST_URL,
        request_body=body,
        cache_scope=_SCOPE,
        headers=[("Content-Type", "application/json")],
        body=b"alpha-results",
    )
    flow = _scoped_flow(method="POST", url=_POST_URL, request_body=body)
    addon.request(flow)
    assert flow.metadata[mitm_cache._CACHE_HIT_MARKER] is True
    assert flow.response.raw_content == b"alpha-results"


@pytest.mark.parametrize("source", ["metadata", "header"])
def test_request_scope_sources_select_scoped_cache(
    tmp_path: Path, addon: mitm_cache.HttpsCacheAddon, source: str
) -> None:
    assert _store(tmp_path, url=_GET_URL, cache_scope=_SCOPE, body=b"task-a")
    token = base64.b64encode(f"framework:{_SCOPE}".encode()).decode()
    kwargs = (
        {"metadata": {"proxyauth": ("framework", _SCOPE)}}
        if source == "metadata"
        else {"request_headers": {"Proxy-Authorization": f"Basic {token}"}}
    )
    flow = _flow(**kwargs)
    addon.request(flow)
    assert flow.response.raw_content == b"task-a"
    assert "Proxy-Authorization" not in flow.request.headers


def test_connect_scope_is_reused_by_client(
    tmp_path: Path, addon: mitm_cache.HttpsCacheAddon
) -> None:
    assert _store(tmp_path, url=_GET_URL, cache_scope=_SCOPE, body=b"task-a")
    client = SimpleNamespace(id="client-1")
    token = base64.b64encode(f"framework:{_SCOPE}".encode()).decode()
    connect = _flow(
        request_headers={"Proxy-Authorization": f"Basic {token}"}, client_conn=client
    )
    request = _flow(client_conn=client)
    addon.http_connect(connect)
    addon.request(request)
    assert request.response.raw_content == b"task-a"
    assert "Proxy-Authorization" not in connect.request.headers


def test_unscoped_request_passes_through_without_storing(
    tmp_path: Path, addon: mitm_cache.HttpsCacheAddon
) -> None:
    flow = _flow(response=True)
    addon.request(flow)
    addon.response(flow)
    assert _load(tmp_path, url=_GET_URL) is None


def test_unscoped_request_is_not_served_scoped_cache(
    tmp_path: Path, addon: mitm_cache.HttpsCacheAddon
) -> None:
    assert _store(tmp_path, url=_GET_URL, cache_scope=_SCOPE, body=b"task-a")
    flow = _flow()
    addon.request(flow)
    assert flow.response is None


def test_flow_logs_pass_store_and_hit(
    addon: mitm_cache.HttpsCacheAddon, caplog
) -> None:
    caplog.set_level(logging.INFO)
    miss = _scoped_flow(
        url=_INSTALL_URL,
        response=True,
        response_headers={"Content-Type": "text/x-shellscript"},
        response_body=b"install",
    )
    addon.request(miss)
    addon.response(miss)
    addon.request(_scoped_flow(url=_INSTALL_URL))
    assert all(
        f"framework-https-cache {event} " in caplog.text
        for event in ("pass", "store", "hit")
    )


def test_logging_does_not_depend_on_mitm_context(monkeypatch, caplog) -> None:
    monkeypatch.setattr(mitm_cache, "ctx", SimpleNamespace(), raising=False)
    caplog.set_level(logging.INFO)
    message = "framework-https-cache pass GET https://example.com/"
    mitm_cache._log(message)
    assert message in caplog.text


@pytest.mark.parametrize(
    ("method", "headers", "streams"),
    [
        pytest.param("HEAD", {}, True, id="head"),
        pytest.param("GET", {}, False, id="cacheable"),
        pytest.param("GET", {"Content-Encoding": "gzip"}, True, id="encoded"),
    ],
)
def test_only_non_cacheable_responses_stream(
    addon: mitm_cache.HttpsCacheAddon,
    method: str,
    headers: dict[str, str],
    streams: bool,
) -> None:
    flow = _flow(
        method=method,
        url=_INSTALL_URL,
        response=True,
        response_headers=headers,
        metadata={mitm_cache._CACHE_SCOPE_MARKER: _SCOPE},
    )
    addon.responseheaders(flow)
    assert flow.response.stream is streams


def test_store_drops_volatile_but_preserves_identity_headers(tmp_path: Path) -> None:
    assert _store(
        tmp_path,
        headers=[
            ("Content-Type", "text/x-shellscript"),
            ("ETag", '"abc123"'),
            ("Age", "35"),
            ("Date", "Fri, 03 Jul 2026 23:26:32 GMT"),
            ("X-Served-By", "cache-bfi"),
            ("X-Request-Id", "request-id"),
        ],
    )
    assert _load(tmp_path).headers == [
        ("Content-Type", "text/x-shellscript"),
        ("ETag", '"abc123"'),
    ]


def test_live_response_strips_volatile_headers_but_cache_hit_does_not(
    addon: mitm_cache.HttpsCacheAddon,
) -> None:
    live = _flow(
        response=True,
        response_headers={
            "content-type": "application/json",
            "age": "35",
            "x-served-by": "cache-ams",
            "x-request-id": "request-id",
        },
        metadata={mitm_cache._CACHE_SCOPE_MARKER: _SCOPE},
    )
    hit = _flow(
        response=True,
        response_headers={"age": "35"},
        metadata={mitm_cache._CACHE_HIT_MARKER: True},
    )
    addon.responseheaders(live)
    addon.responseheaders(hit)
    assert live.response.headers == {"content-type": "application/json"}
    assert hit.response.headers == {"age": "35"}
