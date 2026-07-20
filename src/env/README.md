# env

*Read this to add a benchmark or understand how tasks run and get graded.*

Task environments: what the harness acts on and what grades it.

## The contract (`base.py`)

`TaskEnv` is a small protocol — `reset`, `provision`, `execute`, `verify`,
`close` — plus frozen dataclasses for everything crossing it (`RawEnvOutput`,
`StepResult`, `VerifyOutcome`). `reset` must stay cheap; `provision` does the
expensive start. The split lets cache-served re-runs skip container
infrastructure entirely.

A new benchmark subclasses `DockerTaskEnv` and implements three things: the
`_task_workdir` ClassVar (where actions run), `_build_solve_env()` (pick the
container) and `verify()` (grade). Use `TerminalBenchEnv` and `SweEnv` as the
concrete implementations; registration is one branch in `benchmark()`.
Each task record also carries `agent_timeout_sec` (its wall budget) and
`replay_id` — a fingerprint of the task's content, so cached step results are
keyed to exactly this version of the environment. A task that can't be
replayed soundly sets `replay_id = None` and always runs live.

## Built-in benchmarks

| module | tasks from | graded by | network |
| --- | --- | --- | --- |
| `terminal_bench.py` | pinned clone auto-synced to `~/.cache/harness-experiment/terminal-bench-2-1/`, prebuilt `alexgshaw/<task>` images | the task's own `tests/test.sh` → `reward.txt` | allowed; can be frozen via `netcache/` |
| `swe.py` | SWE-bench Verified (HuggingFace), official instance images | the official grader (`swebench_verify.py`) | none (`network_mode="none"`) |

The Terminal-Bench loader reads only the image, cpu/memory, and the three
timeouts from `task.toml` — tasks needing compose or GPUs are not filtered out
and fail at run time. Reward parsing is exactly `reward.txt` ∈ {0, 1} —
anything else is an error, not a zero.

Every task container runs as `linux/amd64`; ARM hosts therefore need x86
emulation. Terminal-Bench's default host cache also requires
`host.docker.internal` to resolve inside task containers. Docker Desktop and
OrbStack provide it; standalone Docker Engine on Linux does not add it
automatically, and the harness does not currently inject a `host-gateway`
mapping. SWE-bench does not use the host cache.

The env owns the verify deadline: `verify_timeout_sec` is a per-task fact
(Terminal-Bench takes `[verifier].timeout_sec` from task.toml and honors it
up to the declared value; SWE uses the official 1800s grader budget). The
episode adds only an outer backstop on top of it — no global verify constant
second-guesses the env (`src/rollout/README.md`).

`docker_shell.py` is the shared substrate: one long-lived container per task
driven over `docker exec`, with in-container timeouts, bounded output capture,
and PID-scoped cleanup that survives Ctrl-C. The 256 KiB output cap bounds
**model-visible** observations only; internal control reads
(`DockerShellSession.run(..., lossless=True)`) are lossless — grading inputs like
extracted patches are never truncated — and data crossing a shell boundary
internally uses NUL-safe transport (`-z` + literal pathspecs), not
whitespace-split text.

## Determinism

An env owns its own determinism. `src/determinism.py` holds the source-level
pins (fixed hostname, hash seeds, `TZ=UTC`, pinned git dates) and the
*scrubbing* — deleting run-varying tokens like memory addresses and tempfile
names from observations before the model or any cache key sees them. Each
Terminal-Bench task gets a fixed docker subnet, and network observations can
be frozen through the host cache stack in `netcache/` (see its README).
