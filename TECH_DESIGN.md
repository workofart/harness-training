# Technical Design

![Infrastructure for experiment and self-improvement loop](https://www.henrypan.com/blog/assets/images/ml/harness-design/self-improvement-loop-infrastructure.png)

## Scope

Two entry points with **disjoint, non-overlapping** purposes:

- `uv run exp` — one-off run / smoke test. Runs a task set against **whatever code is checked out** and writes one **raw** experiment record (`experiment.json`). It does not look up a baseline, compare, gate, promote, or touch git.
- `uv run auto` — the self-improving loop. Drives an outer coding agent that proposes one harness mechanism, measures it against the current baseline with a pure gate, keeps or discards it, diagnoses the result, and repeats. It *uses* the same orchestrator as `exp`.

The design rests on four moves: **derive control state from durable artifacts + git** (no parallel state store); **one writer per fact** (anything derivable is computed on demand, never stored twice); **the run orchestrator is gate-free, baseline-free, decision-free** (all promotion logic lives in `supervisor/`, in types the orchestrator's layer cannot import); and **`HEAD` only advances** (candidates live as a ref + ephemeral worktree; the primary repo is read-only except a fast-forward on keep).

## Layered architecture

A lower layer never imports a higher one. This keeps the surface-under-test isolated, the orchestrator decision-free, and the pure decision logic unit-testable without git/docker.

```
Layer 5  supervisor/   (auto only — the outer loop)
   policy   PURE: World, Command, Decision, LoopResult, BaselineComparison,
            CandidateDiff; decide/gate/combine/budget/validate + Fisher stats. NO I/O.
   loop     EFFECTS: run_auto driver, scan()->World, execute(cmd), thread-id memo
   workspace  ephemeral worktree + candidate ref      agent  proposer backends
        │ uses ▼  (supervisor/* → experiment/*, never the reverse)
Layer 4  experiment/   (one run = tasks → RAW ExperimentResult)
   orchestrator  run_tasks: concurrency + scheduling + aggregate
   executor  run_trial (one trial)   record  dumb models + .load()   writer  write-only persist
        │ uses ▼
Layer 3  harness/      (THE SURFACE UNDER TEST — candidate edits here)
   core   agent loop, 8-action vocab, tool specs, prompts
        │ uses ▼
Layer 2  env/  (HarnessEnv impl: harbor, docker)     llm/  (BaseLlm impl: base, openrouter, codex)
        │ uses ▼
Layer 1  foundation
   contracts (RawState, HarnessEnv, EnvExecWorkload, TaskMetrics, FailureMode,
              is_majority_solved/decided)   ·   trace · config · repo · retry · serialization
```

Hard rules the layering enforces:

- `experiment/*` must not import `supervisor/*` — the orchestrator cannot know about baselines, gates, decisions, or worktrees. `LoopResult`/`Decision`/`BaselineComparison` live in `supervisor/`, so the orchestrator physically cannot write a decision; the exp/auto decoupling is an architectural guarantee, not a discipline.
- `harness/core` imports only `contracts`, the LLM base, and `trace` — the surface-under-test cannot reach into experiment or supervisor machinery.

## The surface under test: `src/harness/core.py`

The harness is an LLM shell agent. `core.py` owns the agent policy loop: build prompts from the trajectory, ask the configured LLM for tool calls, validate/repair model output, execute actions through `HarnessEnv`, render observations, and decide light-vs-heavy workload per action.

The model-facing action vocabulary is eight typed dataclasses: `list_dir`, `find_files`, `search_text`, `read_file`, `write_file`, `edit_file`, `run`, `verify`. `ACTION_CLASSES` is the source of truth — each class declares its model-facing description, and `build_tool_specs()` derives required/optional keys and JSON scalar types from dataclass fields/type hints.

`core.py` chooses each action's workload (`light` for file/list/search; `run` light or heavy by timeout; `verify` is a bare `await env.verify()`); the **enforcer** (the heavy-action semaphore) lives in `env/harbor`. The **verify-timeout ceiling lives in the executor**, which injects a ceiling-enforcing `HarnessEnv` wrapper into the loop, keeping grading infra off the candidate-editable surface.

This is the only file the outer agent may edit, plus its test (`tests/harness/test_core.py`).

## `uv run exp` — one run → a raw `ExperimentResult`

```
load config → run_tasks(task_ids, budget) → write experiment.json → print → exit
```

`run_tasks(config, task_ids, budget) -> ExperimentResult` is the shared core of both entry points. Two roles, zero shared implementation:

- **`experiment/orchestrator`** runs many trials concurrently, applies all execution-level optimizations and scheduling, aggregates, and persists via `writer`. It never touches an env.
- **`experiment/executor`** (`run_trial`) runs one trial: env reset, two independent timeouts (env-setup vs agent), verify-ceiling enforcement (the env wrapper), failure classification, slot-release-before-teardown. It never touches concurrency.

The results nest (`TrialResult ⊂ TaskResult ⊂ ExperimentResult`); the execution does not, so they stay two files with separate test seams (fake `env`+`llm` for the executor; a stub executor for the orchestrator).

Two optional inputs select what runs without changing measurement: `EXP_TASK_IDS` (a task subset) and `EXP_EXPERIMENT_ID` (an existing dir to append into) let `auto` drive train and test as separate calls into one experiment dir; `EXP_EXPERIMENTS_DIR` anchors artifacts to `<main_repo>/experiments` (absolute) so a run inside a throwaway worktree is byte-for-byte equivalent to a primary run. A standalone user passes none and gets all-tasks / fresh-id. `exp` stays decision-free either way.

Runtime optimizations: decoupled two-level concurrency (trial cap vs heavy-action cap); light/heavy action split; slot-release-before-teardown; LPT scheduling from `task_duration_priors.json`; priority admission with speculative slot reservation; majority early-stop (`is_majority_decided`); deterministic-solved single-trial fast path + confirm-on-fail; separate env-setup vs agent timeout; verify-timeout ceiling (executor wrapper); CPU fanout cap from the per-task budget.

## State model & single source of truth

**Control state is derived from the experiment directories + git by `scan()`.** Each artifact has exactly one writer; anything derivable (verdict labels, success rates, evidence rows) is computed on demand.

| Type | File / Layer | SSOT for | Serialized to |
|---|---|---|---|
| `TrialResult` | `record.py` / exp | one trial: `solved`, `error`, `failure_mode`, `verifier_passed`, artifact paths (incl. `metrics_path`), timestamps | `experiment.json` (nested) |
| `TaskResult` | `record.py` / exp | one task = `trials: [TrialResult]` + `expected_trial_count` + derived (`solved_count`, `majority_solved`, `representative`, `is_finished`) | `experiment.json` (nested) |
| `ExperimentResult` | `record.py` / exp | the run: id, commit, `run_status`, timestamps, `tasks: {task_id → TaskResult}` | `experiment.json` |
| `TaskMetrics` | `contracts.py` / foundation | live per-trial **telemetry**: counters, tokens, `rule_fires` | `tasks/.../agent/metrics.json` (sole owner) |
| `BaselineComparison` | `policy.py` / auto | one task's candidate-vs-baseline **verdict** (`kind`, counts, `p_value`) | `loop.json` (nested in `Decision.verdicts`) |
| `Decision` | `policy.py` / auto | one cycle's **decision**: `kind` (keep/discard), `reason`, `verdicts: {task_id → BaselineComparison}` | `loop.json` (nested in `LoopResult.decision`) |
| `LoopResult` | `policy.py` / auto | the auto cycle: `experiment_id`, `kind`, `focus_name`, `parent_baseline_experiment_id`, `decision: Decision \| null` | `loop.json` |

**Outcome vs telemetry — no field has two owners.** A trial's *outcome* is first-class on `TrialResult` (`solved` is the gate's SSOT; `error` distinguishes infra crash/interrupt from a valid measurement; `failure_mode` is the 8-way categorical *why*). `TaskMetrics` holds only telemetry and is referenced by `metrics_path`, never embedded — so `experiment.json` supports run-level triage on its own. Invariants enforced at construction: `solved ⟺ failure_mode == "solved"`, and `error is not None ⟹ not solved`.

### `run_status` vs `decision` — two questions, two owners

| Fact | File | Owner | Values | Question |
|---|---|---|---|---|
| `run_status` | `experiment.json` | orchestrator | running / completed / crashed | did the run finish? (mechanical) |
| `decision` | `loop.json` | auto | keep / discard / null | what did the gate judge? |

`experiment.json` (written by `exp`/the orchestrator) is mode-agnostic: no panels, no focus, no parent, no decision. `loop.json` (written by `auto`) is **prewritten with `decision: null` before the run** and filled by `Conclude` after — so the dir is stamped as `auto`'s the instant the run launches (a crash never strands a completed run as an indistinguishable `exp` one-off), and `decision == null ⟺ pending` is the one fact the loop routes on. `decision` is the only nullable; it is never prewritten `keep`. The only foreign key is `LoopResult.parent_baseline_experiment_id` (self-reference to a prior `ExperimentResult`) — `ExperimentResult` itself holds no baseline reference, which is the layering restated as a schema invariant.

### Derived facts (computed by `scan()`, consumed by `decide`)

- `active_baseline` = the `loop.decision.kind == "keep"` run with the newest `ExperimentResult.finished_at` (the ordering authority).
- `baseline_ok` = a single conjunct: `active_baseline.git_commit_hash == HEAD`. Everything that affects measurement (code, timeouts, trial budget, model, reasoning_effort, panel sets) lives in the committed tree, so git's commit hash *is* the protocol fingerprint and `commit == HEAD` is a complete staleness check. Task-set and protocol consistency are enforced by hard-fail asserts at load rather than control-flow branches.
- `pending` = the (≤1) **live** run: `loop.decision == null` AND `run_status == completed`. **Dead** pendings (crashed, killed mid-run leaving `running`, or launch-incomplete with no `experiment.json`) are filtered out here, never surfaced.
- `primary_dirty`, `undiagnosed_candidate_id` (a concluded *candidate* with no `diagnosis.md`; baselines are never diagnosed).

`thread_id` (codex/claude conversation resume) is the only persisted ephemeral state — a thin cache (`{phase, thread_id, experiment_id}`). Losing it just starts a fresh agent turn; it is never an authority for `decide`.

## `uv run auto` — the outer loop

```python
# supervisor/loop.py
while True:
    world = scan(experiments_dir, repo)   # I/O read boundary -> World
    cmd   = decide(world)                  # pure (policy)
    execute(cmd)                           # the only side effects
```

Every auto run follows one lifecycle — **prewrite `loop{decision:null}` → run (1+ orchestrator calls) → `Conclude`** — so no completed run is ever lost and each command has one honest cost.

`decide(world) -> Command` is pure. `Command` is a discriminated union of frozen dataclasses; `execute()` matches on type:

| # | Condition | Command | Cost |
|---|---|---|---|
| 1 | `primary_dirty` | `Halt(reason)` | — |
| 2 | live `pending` baseline, **or** candidate with `gate(train)==discard` **or** test already run | `Conclude(exp)` | cheap/pure |
| 3 | live `pending` candidate, `gate(train)==keep`, test not yet run | `RunVeto(exp)` | expensive |
| 4 | `undiagnosed_candidate` | `Diagnose(exp)` | cheap |
| 5 | `not baseline_ok` | `RefreshBaseline()` | expensive |
| 6 | else | `ProposeAndLaunch()` (run train) | expensive |

First match wins. A dead pending is **not** a row — `scan()` excludes it before `decide()` runs, so a prior crash never blocks a manual rerun. `gate(train)` is pure, so `decide()` calls it freely to route Conclude-vs-RunVeto; `Conclude` recomputes it to write. `Halt` fires only on a dirty primary or a genuine `LoopCorruption` (an impossible disk state — `scan()` hard-fails on >1 live pending, a candidate whose parent isn't the active baseline, a baseline that didn't run all tasks, etc.); keep/discard are normal autonomous transitions.

Outer-agent backends (`supervisor/agent_backend.py`), selected by `--agent` (default `codex`):

- `CodexBackend`: `codex exec --json --dangerously-bypass-approvals-and-sandbox`
- `ClaudeBackend`: `claude -p --output-format stream-json --verbose --dangerously-skip-permissions`

### Candidate isolation: a ref + ephemeral worktrees

A candidate survives as a git ref `refs/experiments/candidate/<id>` → commit `C`, from the moment it commits until `Conclude`, so its code outlives any worktree. Each orchestrator call runs in a **fresh, throwaway worktree** at that ref. The agent edits in a **sparse** worktree — a restricted view that omits the run machinery and `config` (which carries the literal task names); run worktrees are full.

The agent's view is two nested constants in `policy.py` (pure data):

```text
EDITABLE_PATHS = { "src/harness/core.py", "tests/harness/test_core.py" }
VISIBLE_PATHS  = EDITABLE_PATHS
               + "program.md", "pyproject.toml", "uv.lock"                   # brief + build/run
               + "src/__init__.py", "src/contracts.py", "src/llm/base.py",
                 "src/trace.py", "src/serialization.py", "tests/conftest.py" # = import-closure(test_core)
```

`VISIBLE_PATHS` is the transitive import closure of `tests/harness/test_core.py` (so the agent can run its own test in the view) plus the brief/build files — verified by *behavior* (a test builds the sparse view at `HEAD` and runs `test_core.py`), not static analysis. `EDITABLE ⊆ VISIBLE`, and because `config` is not visible, the agent cannot hardcode task names it never sees — making task-agnosticism structural rather than a check.

Prelaunch is a capped feedback loop: propose (agent turn, resuming `thread_id`, returns `focus_name`) → `validate_candidate(diff, *, task_ids)` (pure: every changed path ∈ `EDITABLE_PATHS`; no literal task ids in added lines) → run `test_core` in the sparse view (red ⟹ re-prompt) → commit `C`, set the candidate ref → prewrite `loop.json{kind:candidate, focus_name, parent, decision:null}` → full worktree → `uv run exp` (train subset). `focus_name` is captured from the proposal turn and lives on `LoopResult` — `config/harness_config.json` neither drives it nor is visible to the agent.

`Conclude` is ordered for crash-safety and is idempotent: keep ⟹ `git merge --ff-only C` onto the primary (also the only HEAD-drift guard — a diverged HEAD fails the FF and Halts rather than 3-way merging); discard ⟹ `refs/experiments/failed/<id>`; then drop the candidate ref; then persist `decision` last. A crash at any point re-enters via rule 2/4 and replays cleanly. The primary repo is read-only except the single FF on keep.

## The gate (pure)

The gate lives in `supervisor/policy` as pure functions over two loaded `ExperimentResult`s — the loop does the loading and sequences the two panels via commands; the orchestrator never gates.

```python
def gate(candidate, baseline, *, task_ids, purpose) -> Decision      # purpose = promotion | regression_veto
def combine(train, test) -> Decision                                  # keep iff train.keep AND (test is None or test.keep)
def budget_from_baseline(baseline, *, task_ids, full) -> dict[str,int]
```

`train` is the **promotion** panel (aggregate Fisher-exact improvement at α, per-task `BaselineComparison`s as diagnostic evidence, a majority-solve floor); `test` is **regression-veto** (can only block, never promote). Promotion proposes, veto disposes. The flow: `ProposeAndLaunch` runs train → `gate(train, purpose="promotion")`; discard ⟹ `Conclude` writes `combine(train, None)` (test never runs); keep ⟹ `RunVeto` runs test → `Conclude` writes `combine(train, gate(test, "regression_veto"))`. A still-majority-solved task floors a statistical regression to unchanged; tasks with no baseline samples are no-baseline frontier tasks. The gate being pure and swappable means its statistics can be revised as a one-module change.

The per-task **budget is an input** (uniform-full for `exp`/baseline; baseline-derived for candidates via `budget_from_baseline` — the deterministic-solved single-trial fast path). It crosses the `uv run exp` seam as `EXP_TRIAL_BUDGET` (JSON), which is the auto→exp transport of a value *derived* from the committed `task_trials` + the measured baseline — a scheduling optimization, not an independent measurement knob, so `commit == HEAD` stays a complete staleness check.

**Evidence is derived on demand.** The diagnosis prompt hands the agent the raw `experiment.json` (plus its trial dirs and `learning.md`) and it reasons over those directly; the per-task `BaselineComparison` verdicts persisted in `loop.json` are the gate's own per-task evidence.

## The cumulative memo

Two concerns, split:

- **Raw log — `experiments/<id>/diagnosis.md`**: write-only, immutable, one per cycle. `Diagnose` is resumable for free — "done" = `diagnosis.md` exists.
- **Curated view — `experiments/learning.md`**: the agent emits a full fresh rewrite to `learning.draft.md` (input = current `learning.md` + this cycle's `diagnosis.md`); the loop validates it (non-empty, within a line budget) and **atomically swaps** (`os.replace`).

Because the live `learning.md` is only ever replaced atomically or left untouched, it is never half-written; condensation is non-lossy because the raw `diagnosis.md` log persists.

## Crash handling — no auto-recovery of broken work

The supervisor never auto-recovers, resumes, or re-runs *broken* work. A run that dies mid-flight leaves a **dead pending** (`loop.json` decision==null + a `crashed`/`running` `experiment.json`, or none at all); the crashing `uv run auto` invocation **dies in place** (the `exp` subprocess exits nonzero → `_run_exp` raises, uncaught). The next manual `uv run auto` **filters** that dead pending out of `World.pending`, so it is never adopted (decision is null ⟹ not a keep), never acted on, and never blocks forward progress — its artifacts stay on disk for inspection. This is deliberate non-interference, not recovery.

| Failure | Scope | Handling |
|---|---|---|
| One trial's infra failure (docker hiccup) | trial | tolerated: `TrialResult.error` set, excluded from solved/gate; the run continues |
| Every trial crashed in an active task set | experiment | `run_status = crashed` → `exp`: nonzero exit + visible record; `auto`: invocation dies, the record is a dead pending — filtered next scan |
| Run died mid-run / launch-incomplete | experiment / launch | dead pending → filtered on the next scan; a manual rerun proceeds |
| auto died after a completed run, before `Conclude` | recoverable tail | a completed pending is **live** → routed to `Conclude` (cheap, idempotent) |
| auto died after train (kept), before veto | forward step | `RunVeto` — train results preserved, only the test panel runs |
| primary worktree dirty | supervisor | `Halt` (never auto-clean) |

`Halt` prints a human-readable report (what is inconsistent, which dir/ref/worktree to inspect) and stops. Because every run is `prewrite → run → Conclude`, **no completed run is ever lost** even though broken runs are never auto-rerun.

## Configuration

`uv run exp` and `uv run auto` load [config/harbor_config.toml](./config/harbor_config.toml) and [config/harness_config.json](./config/harness_config.json), load OpenRouter credentials when configured, and require a clean worktree unless `EXP_ALLOW_DIRTY_WORKTREE=1`.

`HarnessConfig` (`src/config.py`) is strict `schema_version: 2`. Key fields:

- `panels`: the task panels (see contract below). `train_tasks` / `test_tasks` are **derived properties** over the promotion / regression-veto panels — the loop reasons in train/test, the config file still validates panels.
- `experiment_id`: manual `uv run exp` record id; supervisor-owned during `auto`.
- `max_steps`, `task_trials`: per-trial action budget and independent trials per task.
- `max_trial_concurrency`, `max_heavy_action_concurrency`: live trial bound and reset/run/verify bound.
- `env_setup_timeout_sec`, `max_output_retries`: reset/bootstrap timeout and invalid-output repair budget.
- `llm_provider_config`: harness model provider.
- `focus_name` exists in the schema but `auto` does not read it — the live mechanism label is `LoopResult.focus_name`, captured from the proposal turn.

Panel contract (validated at load): exactly one `purpose: "promotion"` panel and at most one `purpose: "regression_veto"`; both panel task sets non-empty and disjoint (a task cannot sit in both); panel task sets and excluded task groups disjoint.

## LLM providers

Supported harness providers (`src/llm/`):

- `openrouter` (`src/llm/openrouter.py`): OpenRouter transport, timeout/retry, token accounting; uses `OPENROUTER_API_KEY`.
- `chatgpt_codex` (`src/llm/codex.py`): Codex/ChatGPT OAuth transport; requires `codex login` and explicit `model_name` + `max_context_length`; converts harness requests to Responses calls and parses streaming events.
- `src/llm/base.py`: the `BaseLlm` adapter interface.

Per-step flow: task instructions + trajectory + current observation are replayed into model-facing messages; request/response metadata and model-emitted tool calls / reasoning are persisted under trial artifacts via `trace.py`.

## Terminal Bench environment

Only Terminal-Bench is implemented today; `HarnessEnv` (`contracts.py`) abstracts the environment so more can be added. `src/env/harbor.py` implements it:

- resolves tasks (checking `task_overrides/<task_id>/` before the Harbor registry), starts Harbor-backed Docker task environments
- executes `run` actions via `HarnessEnv.exec` and the authoritative verifier via `verify`, returning stdout/stderr/return-code observations
- holds the heavy-action semaphore (light actions bypass) and writes per-trial artifacts under the configured experiments dir

`src/env/docker.py` and the bootstrap preamble (apt/pypi proxy wiring, `no_proxy` sanitize, apt-shim restore) support it; the verifier context/image cache is anchored under `experiments_dir`. Harbor stays trace-free — the verify-ceiling and telemetry live in the executor.

## Artifact layout

Experiment-level (one writer each):

```text
experiments/
├── learning.md                  # AGENT (atomic swap) — curated memo
└── <experiment_id>/
    ├── experiment.json          # ORCHESTRATOR — raw run (ExperimentResult)
    ├── loop.json                # AUTO — decision/verdict (LoopResult), prewritten decision:null
    ├── diagnosis.md             # AGENT — write-only per-cycle raw log
    └── tasks/
```

Trial-level:

```text
experiments/<experiment_id>/tasks/<task_id>/<run_id>/
├── agent/
│   ├── exec.log
│   ├── metrics.json
│   └── steps.jsonl
├── artifacts/
├── bootstrap/
│   ├── bootstrap.sh
│   └── return-code.txt
└── verifier/
    ├── ctrf.json
    ├── reward.txt
    └── test-stdout.txt
```

Supervisor-level (working dirs only — control state is derived, not persisted here):

```text
../harness-experiment_supervisor/
├── codex-home/                  # Codex home: symlinks to user auth/config
└── harness-experiment-<hash>/   # per-repo
    ├── worktrees/               # ephemeral candidate/run worktrees
    └── workspace/               # sparse worktree used by the outer agent
```

## Module ownership

- `src/cli.py`: console entrypoints (`main_exp`, `main_auto`) and runtime config loading
- `src/config.py`: strict runtime config models (`HarnessConfig`, `HarborConfig`)
- `src/contracts.py`: foundation vocabulary — env boundary (`RawState`/`HarnessEnv`), trial telemetry (`TaskMetrics`/`FailureMode`), majority helpers
- `src/trace.py`: trace writer and stable artifact filenames
- `src/repo.py`, `src/retry.py`, `src/serialization.py`: git wrapper, retry policy, (de)serialization helpers
- `src/harness/core.py`: **the harness policy/action loop — the only file the outer agent may modify (plus its unit test)**
- `src/env/harbor.py`, `src/env/docker.py`: Harbor/Docker task environment + heavy-action gating
- `src/llm/openrouter.py`, `src/llm/codex.py`, `src/llm/base.py`: provider adapters + interface
- `src/experiment/orchestrator.py`: many-trial concurrency, scheduling, aggregation → raw `ExperimentResult`
- `src/experiment/executor.py`: one trial (`run_trial`) — env lifecycle, timeouts, verify-ceiling, classification
- `src/experiment/record.py`: dumb models `TrialResult`/`TaskResult`/`ExperimentResult` (+ `.load()`)
- `src/experiment/writer.py`: write-only atomic persist of `experiment.json`
- `src/supervisor/policy.py`: **pure** — `decide`/`gate`/`combine`/`budget_from_baseline`/`validate_candidate`, the `World`/`Command`/`Decision`/`LoopResult`/`BaselineComparison`/`CandidateDiff` types, the `VISIBLE_PATHS`/`EDITABLE_PATHS` constants, Fisher stats
- `src/supervisor/loop.py`: `run_auto`, `scan()→World`, command executors, `loop.json` writes, thread-id memo
- `src/supervisor/workspace.py`: candidate ref + ephemeral worktree lifecycle, sparse view, diff extraction, the `test_core` gate, FF-on-keep, failed-ref
- `src/supervisor/agent.py`: codex/claude resume ids, prompts, the `validate_candidate` feedback loop
- `src/supervisor/agent_backend.py`: Codex/Claude subprocess adapters
