#!/usr/bin/env bash
# Start (or restart) a host-side apt cache for harness bootstraps.
#
# Task images are minimal (FROM ubuntu:24.04, no python3), so every trial's
# bootstrap runs `apt-get install python3` against archive.ubuntu.com. Under
# concurrent load that external fetch stalls and is the dominant cause of
# bootstrap timeouts. apt-cacher-ng caches every fetched .deb/index on the host
# and coalesces concurrent requests for the same file into one upstream
# download, so only the first trial ever leaves the host.
#
# The bootstrap preamble (src/adapters/env.py) auto-detects this cache on
# host.docker.internal:3142 and routes apt through it, falling back to the
# direct mirror when it is absent. Running this is therefore optional, but
# strongly recommended for any multi-trial run.
#
# Idempotent, and persists across Docker/OrbStack restarts via --restart=always
# plus a named volume for the cache contents.
set -euo pipefail

NAME="${APT_CACHE_NAME:-harness-apt-cache}"
PORT="${APT_CACHE_PORT:-3142}"
VOLUME="${APT_CACHE_VOLUME:-harness-apt-cache}"
IMAGE="${APT_CACHE_IMAGE:-sameersbn/apt-cacher-ng:latest}"

if docker ps --filter "name=^/${NAME}$" --filter "status=running" \
    --format '{{.Names}}' | grep -q .; then
  echo "apt cache '${NAME}' already running on host port ${PORT}."
  exit 0
fi

docker rm -f "${NAME}" >/dev/null 2>&1 || true
docker run -d \
  --name "${NAME}" \
  --restart always \
  -p "${PORT}:3142" \
  -v "${VOLUME}:/var/cache/apt-cacher-ng" \
  "${IMAGE}" >/dev/null

echo "Started apt cache '${NAME}' on host port ${PORT} (volume '${VOLUME}', image '${IMAGE}')."
echo "Bootstraps reach it at http://host.docker.internal:${PORT} and fall back"
echo "to the direct mirror if it is down. Remove with: docker rm -f ${NAME}"
