# netcache

*Read this when the cache services misbehave or you're changing them; normal
runs manage them automatically.*

Host-side network cache for Terminal-Bench containers: an mitm HTTPS cache plus
PyPI and uv-Python mirrors, wired into every task container by
`setup-terminal-bench-container.sh` (apt is not mirrored — it routes through the
same mitm proxy). It exists so container network
observations are a frozen, replayable function of the task — the last
uncontrolled input to an otherwise deterministic environment. SWE-bench
ignores it (those containers have no network at all).

With the default `environment.host_netcache: true`, `terminal_bench` runs
bring the stack up automatically (`docker compose up -d --no-recreate
--wait`) and leave it running, so the two runs a training epoch compares —
the baseline harness and the candidate change — see the same warm state.
Misses persist to Docker volumes; hits are served from them.
Normal runs start the services automatically; shutdown and cache cleanup are
manual.

> **Before quickstart evaluation:** Docker preflight only pings the daemon. It
> does not verify Docker Compose, host architecture, required ports, or this
> stack. [Quickstart evaluation](../../../config/quickstart_eval.yaml) enables
> netcache, so its first run starts these services.
>
> Training leaves four `restart: always` services running and four persistent
> volumes behind. The PyPI cache alone is configured for up to
> [20 GiB](docker-compose.caches.yml#L27-L29). Stop the services while retaining
> cached data with:
>
>     docker compose -f src/env/netcache/docker-compose.caches.yml down
>
> Add `-v` to delete all four cache volumes and reclaim their disk space.

## Host contract

Four services, all bound to loopback and reached from task containers as
`host.docker.internal:<port>`:

| service | host port | serves |
| --- | --- | --- |
| `pypi-cache` | 3141 | PyPI index and wheels |
| `uv-python-mirror` | 3143 | uv managed-Python builds |
| `https-cache` | 3144 | mitm HTTPS proxy (apt routes here too) |
| `https-cache-ca` | 3145 | the mitm CA certificate |

## Runbook

- If you edit `docker-compose.caches.yml`, `mitm_https_cache.py`, or another
  implementation file while the services are up, recreate the affected
  service before relying on it. For the HTTPS cache:

      docker compose -f src/env/netcache/docker-compose.caches.yml up -d --force-recreate --wait https-cache https-cache-ca

- `plugins.execution: "replay"` refuses to run with `host_netcache: false` —
  an unfrozen network can't be part of a certified replay (certification is
  explained in `src/plugins/README.md`).
