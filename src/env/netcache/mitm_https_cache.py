#!/usr/bin/env python3
"""mitmproxy addon for first-write-wins network freezing.

Deliberately not a general HTTP cache: no TTLs, no revalidation, no eviction.
Successful non-variant GET responses are frozen by URL, and successful POST
responses are frozen by URL plus request-body hash. Staleness is deliberately
traded for run-to-run determinism at the network boundary. The freeze scope is
the task namespace/epoch, not an action index: repeated identical requests
within one rollout also replay the first response. Stateful polling endpoints
must make state part of the URL/body or bypass this proxy.

Only requests carrying a framework network-cache scope are frozen. The bundled
compose proxy challenges for proxy auth so mitmproxy exposes that scope; if the
addon is run without that gate, unscoped clients pass through live. Range
requests and non-encoding negotiated variants are refused up front. Upstream
requests are forced to ``Accept-Encoding: identity``, so ``Vary:
Accept-Encoding`` responses collapse to one unencoded body; actual
``Content-Encoding`` responses are still refused. Signed GitHub release asset
URLs are canonicalized without their expiring query strings; every other GET
keeps its full query in the key.
"""

import dataclasses
import hashlib
import errno
import json
import logging
import os
import re
import shutil
import tempfile
from base64 import b64decode
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

try:
    from mitmproxy import http
except ModuleNotFoundError:  # Unit tests exercise the storage helpers locally.
    http = None

DEFAULT_CACHE_DIR = "/var/cache/framework-https-cache"
_CACHE_HIT_MARKER = "framework_https_cache_hit"
_CACHE_SCOPE_MARKER = "framework_https_cache_scope"
_PINNED_UPSTREAM_ORIGINAL_URL_MARKER = "framework_pinned_upstream_original_url"
_CACHE_ENTRY_META = "meta.json"
_CACHE_ENTRY_BODY = "body"
_PROXY_AUTH_USER = "framework"
# Strip signed release-asset queries; the path identifies the immutable asset.
_GITHUB_RELEASE_HOSTS = {
    "objects.githubusercontent.com",
    "github-releases.githubusercontent.com",
    # github.com /releases/download/ 302s here now.
    "release-assets.githubusercontent.com",
}
# Debian pool filenames pin exact package versions.
_DEBIAN_POOL_PATH = re.compile(r"^/debian/pool/.+")
_PINNED_DEBIAN_MIRROR_HOST_ENV = "FRAMEWORK_DEBIAN_MIRROR_HOST"
_DEFAULT_PINNED_DEBIAN_MIRROR_HOST = "debian.osuosl.org"
_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
# Remove request telemetry that can fork trajectories; retain content-identity
# headers.
_VOLATILE_RESPONSE_HEADERS = {
    "age",
    "date",
    "via",
    "server-timing",
    "x-served-by",
    "x-cache",
    "x-cache-hits",
    "x-timer",
    "x-request-id",
    "x-amzn-trace-id",
    "x-amz-request-id",
    "x-amz-id-2",
    "x-amz-cf-id",
    "x-amz-cf-pop",
    "x-fastly-request-id",
    "cf-ray",
    "x-github-request-id",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
    "x-ratelimit-used",
}


@dataclasses.dataclass(kw_only=True)
class CachedResponse:
    status_code: int
    headers: list[tuple[str, str]]
    body: bytes


def load_cached_response(
    cache_dir: Path,
    *,
    method: str,
    url: str,
    request_headers: list[tuple[str, str]] | None = None,
    request_body: bytes = b"",
    cache_scope: str,
) -> CachedResponse | None:
    cache_url = _cache_url(
        method=method,
        url=url,
        request_headers=request_headers or [],
        request_body=request_body,
        response_headers=[],
    )
    if cache_url is None:
        return None
    entry_dir = _cache_entry_dir(
        cache_dir,
        cache_scope=cache_scope,
        method=method,
        url=cache_url,
    )
    meta_path = entry_dir / _CACHE_ENTRY_META
    body_path = entry_dir / _CACHE_ENTRY_BODY
    if not meta_path.exists() or not body_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        return CachedResponse(
            status_code=meta["status_code"],
            headers=[tuple(header) for header in meta["headers"]],
            body=body_path.read_bytes(),
        )
    except (OSError, TypeError, ValueError, KeyError):
        return None


def store_cacheable_response(
    cache_dir: Path,
    *,
    method: str,
    url: str,
    request_headers: list[tuple[str, str]] | None = None,
    request_body: bytes = b"",
    cache_scope: str,
    status_code: int,
    headers: list[tuple[str, str]],
    body: bytes,
) -> bool:
    if status_code != 200:
        return False
    cache_url = _cache_url(
        method=method,
        url=url,
        request_headers=request_headers or [],
        request_body=request_body,
        response_headers=headers,
    )
    if cache_url is None:
        return False

    entry_dir = _cache_entry_dir(
        cache_dir,
        cache_scope=cache_scope,
        method=method,
        url=cache_url,
    )
    entry_dir.parent.mkdir(parents=True, exist_ok=True)
    if entry_dir.exists():
        return False
    tmp_dir = Path(tempfile.mkdtemp(dir=entry_dir.parent, prefix=f".{entry_dir.name}."))
    try:
        (tmp_dir / _CACHE_ENTRY_BODY).write_bytes(body)
        (tmp_dir / _CACHE_ENTRY_META).write_text(
            json.dumps(
                {
                    "status_code": status_code,
                    "headers": _cacheable_headers(headers),
                },
                sort_keys=True,
            )
        )
        os.rename(tmp_dir, entry_dir)
    except OSError as exc:
        if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY, errno.EISDIR}:
            raise
        return False
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return True


def _cache_url(
    *,
    method: str,
    url: str,
    request_headers: list[tuple[str, str]],
    request_body: bytes,
    response_headers: list[tuple[str, str]],
) -> str | None:
    # Range needs 206 semantics. Other Vary dimensions need request-keyed
    # variants, but Accept-Encoding is normalized to identity before upstream.
    if _has_header(request_headers, "range"):
        return None
    if _has_uncacheable_vary(response_headers) or _has_header(
        response_headers, "content-encoding"
    ):
        return None

    method = method.upper()
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if method == "GET":
        if host in _GITHUB_RELEASE_HOSTS or (
            host == "github.com" and "/releases/download/" in parsed.path
        ):
            return _canonical_url(parsed, keep_query=False)
        return _canonical_url(parsed)
    if method == "POST":
        body_digest = hashlib.sha256(request_body).hexdigest()
        return f"{_canonical_url(parsed)}#body-sha256={body_digest}"
    return None


def _cache_identity_url(flow: Any) -> str:
    return flow.metadata.get(
        _PINNED_UPSTREAM_ORIGINAL_URL_MARKER,
        flow.request.pretty_url,
    )


def _pinned_upstream_url(url: str) -> str | None:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    if host == "deb.debian.org" and _DEBIAN_POOL_PATH.fullmatch(parsed.path):
        mirror_host = os.environ.get(
            _PINNED_DEBIAN_MIRROR_HOST_ENV,
            _DEFAULT_PINNED_DEBIAN_MIRROR_HOST,
        )
        return urlunsplit(
            (
                "http",
                mirror_host,
                parsed.path,
                parsed.query,
                "",
            )
        )
    return None


def _force_identity_accept_encoding(flow: Any) -> None:
    headers = flow.request.headers
    for name in list(headers.keys()):
        if name.lower() == "accept-encoding":
            del headers[name]
    headers["Accept-Encoding"] = "identity"


def _canonical_url(parsed: SplitResult, *, keep_query: bool = True) -> str:
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            parsed.query if keep_query else "",
            "",
        )
    )


def _has_header(headers: list[tuple[str, str]], name: str) -> bool:
    return any(header_name.lower() == name for header_name, _value in headers)


def _has_uncacheable_vary(headers: list[tuple[str, str]]) -> bool:
    for name, value in headers:
        if name.lower() != "vary":
            continue
        tokens = [token.strip().lower() for token in value.split(",")]
        if any(token and token != "accept-encoding" for token in tokens):
            return True
    return False


def _cache_entry_dir(
    cache_dir: Path, *, cache_scope: str, method: str, url: str
) -> Path:
    digest = hashlib.sha256(
        f"{cache_scope}\0{method.upper()} {url}".encode()
    ).hexdigest()
    return cache_dir / digest[:2] / digest


def _cacheable_headers(headers: list[tuple[str, str]]) -> list[tuple[str, str]]:
    dropped = _HOP_BY_HOP_HEADERS | _VOLATILE_RESPONSE_HEADERS
    return [(name, value) for name, value in headers if name.lower() not in dropped]


def _log(message: str) -> None:
    # mitmproxy 12 removed ctx.log; stdlib logging reaches mitmdump output.
    logging.info(message)


def _scope_from_proxy_authorization(value: str | None) -> str | None:
    if not value:
        return None
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "basic" or not token:
        return None
    try:
        decoded = b64decode(token, validate=True).decode()
    except (ValueError, UnicodeDecodeError):
        return None
    user, separator, scope = decoded.partition(":")
    if user == _PROXY_AUTH_USER and separator and scope:
        return scope
    return None


def _client_scope_key(flow: Any) -> object:
    client_conn = getattr(flow, "client_conn", None)
    return getattr(client_conn, "id", None) or id(client_conn)


class HttpsCacheAddon:
    def __init__(self) -> None:
        self.cache_dir = Path(
            os.environ.get("FRAMEWORK_HTTPS_CACHE_DIR", DEFAULT_CACHE_DIR)
        )
        self._client_scopes: dict[object, str] = {}

    def _cache_scope(self, flow: Any) -> str | None:
        user, scope = flow.metadata.get("proxyauth", ("", ""))
        if user == _PROXY_AUTH_USER and scope:
            return scope
        header_scope = _scope_from_proxy_authorization(
            flow.request.headers.get("Proxy-Authorization")
        )
        if header_scope is not None:
            return header_scope
        return self._client_scopes.get(_client_scope_key(flow))

    def http_connect(self, flow: Any) -> None:
        cache_scope = _scope_from_proxy_authorization(
            flow.request.headers.get("Proxy-Authorization")
        )
        flow.request.headers.pop("Proxy-Authorization", None)
        if cache_scope is not None:
            self._client_scopes[_client_scope_key(flow)] = cache_scope

    def request(self, flow: Any) -> None:
        cache_scope = self._cache_scope(flow)
        flow.request.headers.pop("Proxy-Authorization", None)
        if cache_scope is None:
            _log(
                f"framework-https-cache pass unscoped "
                f"{flow.request.method} {flow.request.pretty_url}"
            )
            return
        flow.metadata[_CACHE_SCOPE_MARKER] = cache_scope
        _force_identity_accept_encoding(flow)
        request_url = flow.request.pretty_url
        cached = load_cached_response(
            self.cache_dir,
            method=flow.request.method,
            url=request_url,
            request_headers=list(flow.request.headers.items()),
            request_body=flow.request.raw_content or b"",
            cache_scope=cache_scope,
        )
        if cached is not None:
            flow.response = http.Response.make(  # type: ignore[union-attr]
                cached.status_code,
                cached.body,
                # mitmproxy 12 requires byte header tuples; HTTP/1.1 uses latin-1.
                [
                    (name.encode("latin-1"), value.encode("latin-1"))
                    for name, value in cached.headers
                ],
            )
            # Local responses also hit response(); mark cache hits to avoid re-storing.
            flow.metadata[_CACHE_HIT_MARKER] = True
            _log(
                f"framework-https-cache hit {flow.request.method} {flow.request.pretty_url}"
            )
        else:
            pinned_url = _pinned_upstream_url(request_url)
            if pinned_url is not None:
                flow.metadata[_PINNED_UPSTREAM_ORIGINAL_URL_MARKER] = request_url
                # url and Host header must move together.
                flow.request.url = pinned_url
                flow.request.headers["host"] = urlsplit(pinned_url).netloc
                _log(
                    "framework-https-cache pin "
                    f"{flow.request.method} {request_url} -> {pinned_url}"
                )
            else:
                _log(f"framework-https-cache pass {flow.request.method} {request_url}")

    def responseheaders(self, flow: Any) -> None:
        if flow.response is None:
            return
        if flow.metadata.get(_CACHE_SCOPE_MARKER) is None:
            flow.response.stream = True
            return
        # Strip upstream telemetry before clients or cache entries observe it; hits
        # are already frozen.
        if flow.metadata.get(_CACHE_HIT_MARKER) is None:
            for name in _VOLATILE_RESPONSE_HEADERS:
                flow.response.headers.pop(name, None)
        # Stream non-cacheable pass-through bodies to avoid buffering large downloads.
        cache_url = _cache_url(
            method=flow.request.method,
            url=_cache_identity_url(flow),
            request_headers=list(flow.request.headers.items()),
            request_body=flow.request.raw_content or b"",
            response_headers=list(flow.response.headers.items()),
        )
        if cache_url is None:
            flow.response.stream = True

    def response(self, flow: Any) -> None:
        if flow.response is None or flow.metadata.get(_CACHE_HIT_MARKER) is True:
            return
        cache_scope = flow.metadata.get(_CACHE_SCOPE_MARKER)
        if cache_scope is None:
            return
        stored = store_cacheable_response(
            self.cache_dir,
            method=flow.request.method,
            url=_cache_identity_url(flow),
            request_headers=list(flow.request.headers.items()),
            request_body=flow.request.raw_content or b"",
            cache_scope=cache_scope,
            status_code=flow.response.status_code,
            headers=list(flow.response.headers.items()),
            body=flow.response.raw_content or b"",
        )
        if stored:
            _log(
                "framework-https-cache store "
                f"{flow.request.method} {flow.response.status_code} {flow.request.pretty_url}"
            )


addons = [HttpsCacheAddon()] if http is not None else []
