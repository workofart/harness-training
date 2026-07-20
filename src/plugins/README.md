# plugins

*Read this to understand what makes re-runs fast and what "certified" means.*

Determinism is the prerequisite. 
If the LLM model or the environment isn't deterministic, a cache just replays the first divergence faster.

## The three caches

| boundary            | plugin                   | key                                                     |
| ------------------- | ------------------------ | ------------------------------------------------------- |
| policy <> model      | `caching/llm_cache.py`   | provider revision + exact rendered request              |
| agent <> environment | `replay/step_cache.py`   | rolling hash over the action chain, seeded with schema + task content + epoch |
| solution <> grade    | `replay/verify_cache.py` | schema + installed `swebench` version + (instance, diff) |

The *rendered request* is the exact JSON body sent to the provider; the
*provider revision* is `provider:base_url:model_name`, so a different
endpoint or checkpoint can never serve a stale hit. Action chain is defined
in `src/rollout/README.md`. Scrubbing (`src/env/README.md`) does not enter
these keys — it governs the output comparison that detects drift. Upgrading
the `swebench` dependency intentionally invalidates every verify-cache row.

Replay serves a *prefix*: the first miss re-executes the already-replayed
actions live to bring the env to chain state, and a scrubbed mismatch there is
`ExecutionDriftError` — the rollout fails and retries live-only rather than
grading a forked trajectory.

A cache key is the complete input set of the function it memoizes — no less,
no more. The commit is never a key component (it would kill reuse for zero
correctness), and each cache deliberately excludes facts the layer above
cares about: whether two runs are comparable is measurement identity's job
(`src/rollout/README.md`), not the caches'. Undeclared external drift is out
of scope — a provider silently swapping weights under the same declared
revision is operator-managed (`provider_revision`, `FRAMEWORK_LLM_CACHE_REV`),
and the determinism audit catches it behaviorally; there is no detection
machinery for it.

## Turning them on

Two flags (`PluginsConfig` in `src/config.py`):

    "plugins": { "llm_cache": true, "execution": "eager" | "replay" }

`llm_cache` is independent and on by default. `execution` picks the regime:
`eager` (the default) runs every environment step live; `replay` turns on the
step cache, the verify cache, and certification as one unit — "cache on,
certification off" is deliberately not expressible. The verify cache wraps env's
pure grader (env defines the grade identity; the wrapper never lives in core).
Replay is a demand, not a
hint: config validation rejects it without a seeded deterministic provider
and, on terminal_bench, without `environment.host_netcache` (which lives in
`src/env/netcache/` — env-owned substrate, not a plugin).

## Certification

Before a candidate is compared against a baseline, every task must be
*certified* (`replay/audit.py`): the baseline's recorded action chain is
re-executed in a fresh environment, and every step must reproduce the same
scrubbed output. A task that doesn't reproduce has *forked* and is excluded
from the comparison — environment noise must not masquerade as a candidate
effect.

The verdict is per-task: `deterministic`, `forked`, or `no_chain` (no chain was
recorded, or it timed out — neither certifies the task). Only an observed
semantic mismatch makes a task `forked`; an audit infra failure raises
immediately — it never silently excludes or certifies. A certificate is inherited
by a later run only when both the measurement-identity digest and the task's
chain digest match; otherwise re-audit.

## The replay epoch

The epoch counter (`env:epoch:<namespace>` in the cache DB) is the blessed
operator lever for invalidating a bad recording (e.g. stale apt). It keys
both the env-step replay digest and the Terminal-Bench network-freeze scope
token. A counter that has never been bumped reads as 0 — epoch 0 is the
initial scope, not an error. Bump one namespace via
`python -m src.plugins.replay.bump_epoch '<namespace>'` (a fork's rejection
message prints the namespace to pass); because the epoch is inside the
replay-regime digest, a bump automatically invalidates baseline reuse — there is
no operator-memory "remember to re-baseline" step.

## Storage

One WAL-mode SQLite file, `cache/llm_cache.db` (`caching/store.py`). Cache reads
and writes are fail-open — a broken store reads as a miss, never as a wrong hit —
but opening the store and reading the epoch counter are not: those failures are
measurement failures, because a silently-zero epoch would serve stale recordings.
`FRAMEWORK_CACHE=0` disables all caching; `FRAMEWORK_CACHE_DB` points the store
elsewhere.
