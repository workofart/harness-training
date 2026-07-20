# rollout

*Read this to understand what happens during a measurement and what a run
leaves on disk. You rarely call anything here directly — the Trainer and
`scripts/evaluate.py` drive it.*

The measured middle: everything between "here is a task panel" and "here is a
graded result". This layer is framework-owned — the policy being trained
cannot reach the budgets, the telemetry, or the grade.

Both drivers enter through `src/measurement.py`, which preflights and then runs
every measurement in an isolated `python -m src.worker` subprocess, supervised
over an event pipe. That is why this README talks about "the worker" and "the
parent": a measurement never shares a process with the training loop.

## Vocabulary

| term | meaning |
| --- | --- |
| rollout | one measured episode: the policy driving one task to a grade |
| run | one rollout per task across the panel; `experiment` in artifact names means the same thing |
| panel | the task list a config measures (`environment.task_names`) |
| action chain | the ordered record of every environment action a rollout executed, stored as `infra/determinism_chain.jsonl`; certification re-executes it |
| measurement identity | fingerprint of *how* a run was measured: config digest + provider revision + execution regime, persisted as a small frozen record on the result and compared by digest equality. The commit is recorded alongside it, not folded into it. It gates exactly two things: baseline reuse and determinism-certification inheritance |

Anything inside the repo tree is already identified by the commit — pin
external inputs in code (e.g. the SWE dataset `revision=`) instead of
measuring them into identity downstream.

## Files

| file | role |
| --- | --- |
| `episode.py` | the frozen per-task loop wiring the config-resolved policy (`build_policy`, checked against the `Policy` protocol) to one env; owns step/wall budgets, classifies the outcome, writes the action chain |
| `sampler.py` | fans one run out over the panel; CPU-aware admission, cross-process task locks |
| `telemetry.py` | the measurement boundary: every trace row and completion passes through it, so the editable policy can't forge, suppress, or mislabel a metric |
| `records.py` / `store.py` | persisted shapes (`RolloutResult`, `ExperimentResult`, `FailureMode`) and the filesystem contract |
| `certification.py` | computes measurement identity and rollout provenance |
| `execution.py` | the step-execution seam: `EagerExecution` runs everything live; `resolve_execution` swaps in replay (`src/plugins/README.md`) |
| `metrics.py` | secondary rewards — tie-breaker metrics like first-try-valid rate and steps used |

## Failure taxonomy

Every failure has exactly one owner, expressed as values on existing records —
never new wrapper types:

- **Policy failure** — the candidate's fault; scorable against it.
- **Infra failure** — environment/transport; non-scorable, retry or exclude.
- **Framework defect** — a bug in the frozen framework code; aborts the experiment.
  Framework exceptions propagate; they are never converted into a task grade.

A crash is never a scorable candidate result. A crash carries
`failure_origin ∈ {policy, env}` (set exactly when `failure_mode == "crash"`),
and relabeling to `unscorable_infra` is allowed only when the origin is `env`.
Exception classification happens at exactly two typed boundaries in the
episode — policy calls and env/backend calls — not via one broad catch; a
foreign `TimeoutError` is attributed to a deadline only if the bound timeout
context actually expired.

## Deadlines

The env owns the verify deadline (`TaskEnv.verify_timeout_sec`, a per-task
fact — `src/env/README.md`); the episode adds only an outer backstop of that
deadline + 600s slack. There is no global verify constant that second-guesses
the env. The parent watchdog is pure liveness, not deadline math: the worker
heartbeats every 60s on the event pipe, and silence means a dead or blocked
worker. Long verifies legitimately occupy a slot for hours — if wall-clock
binds, the lever is panel composition, not a shorter timeout.

## Evidence integrity

Records that grade candidates must not be able to misstate what ran:

- Telemetry writes the canonical `event`/`t_sec` fields last; caller fields
  cannot overwrite them.
- `ExperimentResult` enforces finality: finished + non-crashed ⇒ every
  requested task populated and each nested `task_id` equal to its map key.
- The **effective** config payload (post panel-fold) is what gets persisted
  and digested; the config path is provenance only.
- The index row is constructed and validated before the experiment is
  written; JSONL rows are appended in a single write.
- Records normalize at the producing boundary and are strict inside
  (`extra="forbid"`, typed literals — no `getattr` probing).
- Clean break on persisted records: no legacy-field defaults, no schema
  versioning, no backward-compatible loaders. Old artifacts are debug-only —
  the framework never loads them. Envs may produce different artifact trees;
  that asymmetry is fine.

## On disk

Root: `experiments/` for training, `evals/` for evaluation.

    runs.jsonl                      append-only index, one row per run
    learning.md                     the learning memo (training only)
    <run_id>/
      experiment.json               the full result — rewards live here
      run.log
      tasks/<task_id>/
        agent/steps.jsonl           full trace: every request, completion, action, result
        infra/determinism_chain.jsonl
        terminal_bench_verifier/    grader artifacts (SWE-bench writes its own set)
