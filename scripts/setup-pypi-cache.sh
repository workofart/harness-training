#!/usr/bin/env bash
# Start (or restart) a host-side PyPI cache for harness task verifiers.
#
# Task verifiers run `uv run`, which re-resolves and re-downloads the full
# dependency tree from PyPI on every verify. For the ML tasks that is torch plus
# the CUDA stack (~2GB) fetched fresh each trial -- the dominant share of verify
# wall-time. proxpi is a caching PyPI proxy: it caches every fetched wheel/index
# on the host and coalesces concurrent requests for the same file into one
# upstream download, so only the first trial ever leaves the host.
#
# The bootstrap preamble (src/adapters/env.py) auto-detects this cache on
# host.docker.internal:3141 and points uv/pip at it (/etc/uv/uv.toml and
# /etc/pip.conf), falling back to direct PyPI when it is absent. Running this is
# therefore optional, but strongly recommended for any multi-trial run that
# touches the ML tasks. Note: the apt cache (setup-apt-cache.sh) cannot cover
# PyPI because PyPI is served over HTTPS -- hence this separate index-level
# cache that the container reaches over plain HTTP.
#
# Idempotent, and persists across Docker/OrbStack restarts via --restart=always
# plus a named volume for the cache contents.
set -euo pipefail

NAME="${PYPI_CACHE_NAME:-harness-pypi-cache}"
# Host port. proxpi listens on 5000 in-container; 5000 is taken by AirPlay on
# macOS, so default the host side to 3141 (next to the apt cache's 3142).
PORT="${PYPI_CACHE_PORT:-3141}"
VOLUME="${PYPI_CACHE_VOLUME:-harness-pypi-cache}"
IMAGE="${PYPI_CACHE_IMAGE:-epicwink/proxpi:latest}"
# Hold the full ML dependency set (torch + CUDA across versions) without
# evicting; proxpi defaults to 5GB, which the CUDA wheels alone can exceed.
CACHE_SIZE="${PYPI_CACHE_SIZE:-21474836480}"

if docker ps --filter "name=^/${NAME}$" --filter "status=running" \
    --format '{{.Names}}' | grep -q .; then
  echo "PyPI cache '${NAME}' already running on host port ${PORT}."
  exit 0
fi

docker rm -f "${NAME}" >/dev/null 2>&1 || true
docker run -d \
  --name "${NAME}" \
  --restart always \
  -p "${PORT}:5000" \
  -v "${VOLUME}:/var/cache/proxpi" \
  -e PROXPI_CACHE_DIR=/var/cache/proxpi \
  -e PROXPI_CACHE_SIZE="${CACHE_SIZE}" \
  "${IMAGE}" >/dev/null

echo "Started PyPI cache '${NAME}' on host port ${PORT} (volume '${VOLUME}', image '${IMAGE}')."
echo "Verifier uv/pip installs reach it at http://host.docker.internal:${PORT}/index/"
echo "and fall back to direct PyPI if it is down. Remove with: docker rm -f ${NAME}"
