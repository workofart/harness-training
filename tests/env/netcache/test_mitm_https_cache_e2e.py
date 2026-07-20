from __future__ import annotations

import json
import re
import subprocess
import textwrap
import uuid
from pathlib import Path

import pytest


_NETCACHE_DIR = Path(__file__).resolve().parents[3] / "src" / "env" / "netcache"
_COMPOSE_FILE = _NETCACHE_DIR / "docker-compose.caches.yml"
_ADDON_FILE = _NETCACHE_DIR / "mitm_https_cache.py"
_POVRAY_URL = (
    "https://www.povray.org/ftp/pub/povray/Old-Versions/Official-2.2/POVSRC.TAR.Z"
)


@pytest.mark.integration
def test_https_cache_compose_smoke_stores_then_hits_povray_artifact(
    tmp_path: Path,
) -> None:
    compose_file = _write_smoke_compose(tmp_path)
    project = f"httpscache{uuid.uuid4().hex[:8]}"
    compose = ["docker", "compose", "-p", project, "-f", str(compose_file)]

    try:
        _run(
            [
                *compose,
                "up",
                "-d",
                "--wait",
                "https-cache",
                "https-cache-ca",
            ],
            timeout=90,
        )
        client = _run(
            [*compose, "run", "--rm", "client", "python", "-c", _CLIENT_SCRIPT],
            timeout=120,
        )
        logs = _run([*compose, "logs", "--no-color", "https-cache"], timeout=30)
    finally:
        _run([*compose, "down", "-v", "--remove-orphans"], timeout=60, check=False)

    result = json.loads(client.stdout)

    assert result["lengths"] == [575233, 575233]
    assert result["sha256"][0] == result["sha256"][1]
    assert result["prefixes"] == ["1f9d9073", "1f9d9073"]
    assert logs.stdout.count(f"framework-https-cache store GET 200 {_POVRAY_URL}") == 1
    assert logs.stdout.count(f"framework-https-cache hit GET {_POVRAY_URL}") == 1
    assert "Addon error" not in logs.stdout
    assert "Header fields must be bytes" not in logs.stdout


def _write_smoke_compose(tmp_path: Path) -> Path:
    compose_file = tmp_path / "docker-compose.https-cache-smoke.yml"
    cache_dir = tmp_path / "cache"
    cert_dir = tmp_path / "certs"
    cache_dir.mkdir()
    cert_dir.mkdir()
    compose_file.write_text(
        textwrap.dedent(
            f"""
            services:
              https-cache:
                image: {_mitmproxy_image()}
                environment:
                  FRAMEWORK_HTTPS_CACHE_DIR: /var/cache/framework-https-cache
                volumes:
                  - {_volume(_ADDON_FILE, "/opt/mitm_https_cache.py", read_only=True)}
                  - {_volume(cache_dir, "/var/cache/framework-https-cache")}
                  - {_volume(cert_dir, "/home/mitmproxy/.mitmproxy")}
                command:
                  [
                    "sh",
                    "-c",
                    "chmod 0777 /var/cache/framework-https-cache /home/mitmproxy/.mitmproxy && exec mitmdump --listen-host 0.0.0.0 --listen-port 8080 --set confdir=/home/mitmproxy/.mitmproxy --set upstream_cert=false --set connection_strategy=lazy --set proxyauth=any -s /opt/mitm_https_cache.py",
                  ]
                healthcheck:
                  test:
                    [
                      "CMD",
                      "python",
                      "-c",
                      "import socket; s=socket.create_connection(('127.0.0.1', 8080), 2); s.close()",
                    ]
                  interval: 5s
                  timeout: 3s
                  retries: 12

              https-cache-ca:
                image: python:3.13-alpine
                volumes:
                  - {_volume(cert_dir, "/var/lib/mitmproxy", read_only=True)}
                command:
                  [
                    "sh",
                    "-c",
                    "while [ ! -f /var/lib/mitmproxy/mitmproxy-ca-cert.pem ]; do sleep 1; done; mkdir -p /var/run/framework-ca; cp /var/lib/mitmproxy/mitmproxy-ca-cert.pem /var/run/framework-ca/mitmproxy-ca-cert.pem; python -m http.server 8000 -d /var/run/framework-ca",
                  ]
                depends_on:
                  - https-cache
                healthcheck:
                  test:
                    [
                      "CMD",
                      "python",
                      "-c",
                      "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/mitmproxy-ca-cert.pem', timeout=2).read()",
                    ]
                  interval: 5s
                  timeout: 3s
                  retries: 12

              client:
                image: python:3.13-alpine
                depends_on:
                  - https-cache
                  - https-cache-ca
            """
        )
    )
    return compose_file


def _mitmproxy_image() -> str:
    match = re.search(
        r"^\s*image:\s*(mitmproxy/mitmproxy@\S+)\s*$",
        _COMPOSE_FILE.read_text(),
        flags=re.MULTILINE,
    )
    assert match is not None
    return match.group(1)


def _volume(source: Path, target: str, *, read_only: bool = False) -> str:
    mode = ":ro" if read_only else ""
    return json.dumps(f"{source}:{target}{mode}")


def _run(
    command: list[str],
    *,
    timeout: int,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"command failed: {' '.join(command)}\n"
            f"exit code: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


_CLIENT_SCRIPT = f"""
import hashlib
import json
import ssl
import urllib.request
from pathlib import Path

ca_path = Path("/tmp/framework-https-cache-ca.pem")
ca_path.write_bytes(
    urllib.request.urlopen(
        "http://https-cache-ca:8000/mitmproxy-ca-cert.pem",
        timeout=20,
    ).read()
)
context = ssl.create_default_context(cafile=ca_path)
opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({{"https": "http://framework:e2e-scope@https-cache:8080"}}),
    urllib.request.HTTPSHandler(context=context),
)

results = []
for _ in range(2):
    body = opener.open({_POVRAY_URL!r}, timeout=60).read()
    results.append(
        {{
            "length": len(body),
            "prefix": body[:4].hex(),
            "sha256": hashlib.sha256(body).hexdigest(),
        }}
    )

print(
    json.dumps(
        {{
            "lengths": [result["length"] for result in results],
            "prefixes": [result["prefix"] for result in results],
            "sha256": [result["sha256"] for result in results],
        }}
    )
)
"""
