# Technical Design

![Infrastructure for experiment and self-improvement loop](https://www.henrypan.com/blog/assets/images/ml/harness-design/self-improvement-loop-infrastructure.png)

## Scope

Two loops exist:

- `uv run exp`: orange loop. Run the current committed harness once and persist an experiment record.
- `uv run auto`: blue loop. Run an outer coding agent that proposes one harness mechanism, launches `uv run exp`, diagnoses the result, and repeats.

## Agent

Primary modules:

- `src/control/supervisor.py`
- `src/control/agent_backend.py`
- `src/control/gates.py`
- `src/control/supervisor_state.py`
- `src/control/repo.py`

`uv run auto` creates the selected outer-agent backend and delegates to `run_supervisor_loop`.

- default: `codex`
- optional: `--agent claude`

Outer agent backends:

- `CodexBackend`: runs `codex exec --json --dangerously-bypass-approvals-and-sandbox`
- `ClaudeBackend`: runs `claude -p --output-format stream-json --verbose --dangerously-skip-permissions`

The autonomous loop is intentionally narrow. Candidate patches must express harness behavior through `src/harness/core.py`, may add focused tests in `tests/harness/test_core.py`, and may update only `focus_name` in `config/harness_config.json`. The supervisor assigns `experiment_id`.

Supervisor state lives outside the repo under `../harness-experiment_supervisor/<repo-fingerprint>/` by default:

- `state.json`: current supervisor phase and resumable thread metadata
- `events.jsonl`: append-only supervisor event log
- `workspace/`: sparse git worktree used by the outer agent
- `codex-home/`: Codex home provisioned with symlinks to user auth/config

Loop lifecycle:

1. load runtime snapshot
2. ensure sparse workspace
3. recover interrupted launch/postrun state if needed
4. clean orphaned experiment artifacts
5. abandon any unfinished current candidate as `crash`
6. ensure the active baseline matches clean committed `HEAD`
7. run post-run diagnosis for any concluded candidate that still needs it
8. sync sparse workspace to current `HEAD`
9. run prelaunch agent turn
10. validate candidate patch
11. commit candidate in sparse workspace
12. hard-reset primary repo to candidate commit
13. launch `uv run exp`
14. hard-reset primary repo back to baseline on `discard`/`crash`
15. run post-run diagnosis and repeat

Prelaunch gates:

- candidate must change tracked files
- changed paths must be within supervisor-editable paths
- candidate must change `src/harness/core.py`
- `config/harness_config.json` changes are limited to `focus_name`
- added harness/test lines must not contain literal promotion task ids
- mechanism must not duplicate a recent discarded candidate
- if the agent accidentally edits the primary repo instead of the sparse workspace, the supervisor resets the primary repo and sends feedback

Post-run diagnosis gates:

- agent may update only `experiments/learning.md`
- `experiment.json` is restored if the diagnosis turn mutates it
- learning memo must be non-empty and changed

## Harness Code

Primary modules:

- `src/harness/core.py`
- `src/harness/config.py`
- `src/harness/contracts.py`
- `src/experiment/trial.py`
- `src/experiment/runner.py`
- `src/experiment/record.py`
- `src/experiment/gate.py`
- `src/metrics.py`
- `src/trace.py`

`uv run exp` loads runtime config and constructs `ExperimentRunner`.

- loads [config/harbor_config.toml](./config/harbor_config.toml)
- loads [config/harness_config.json](./config/harness_config.json)
- loads provider credentials for OpenRouter when configured
- requires a clean worktree unless `EXP_ALLOW_DIRTY_WORKTREE=1`

`HarnessConfig` in `src/harness/config.py` is strict `schema_version: 2`.

Key fields:

- `experiment_id`: manual `uv run exp` record id; supervisor-owned during `auto`
- `focus_name`: short mechanism label; candidate-editable during `auto`
- `panels`: ordered task panels
- `max_steps`: per-trial action budget
- `max_trial_concurrency`: live trial concurrency
- `max_heavy_action_concurrency`: concurrent reset/run/verify bound
- `env_setup_timeout_sec`: reset/bootstrap timeout
- `max_output_retries`: invalid model-output repair budget
- `task_trials`: independent trials per task
- `llm_provider_config`: harness model provider

Panel contract:

- exactly one `purpose: "promotion"` panel
- at most one `purpose: "regression_veto"` panel
- promotion panel must run `{"when": "always"}` and not require a baseline
- regression-veto panel must appear after the promotion panel, require a baseline, and run only after the promotion panel reaches `keep`
- panel task sets and excluded task groups must be disjoint

The model-facing action vocabulary is eight typed dataclasses:

- `list_dir`
- `find_files`
- `search_text`
- `read_file`
- `write_file`
- `edit_file`
- `run`
- `verify`

`ACTION_CLASSES` is the source of truth. `build_tool_specs()` derives tool schemas from the dataclass fields, while descriptions and integer-typed fields are declared separately.

Per-task execution:

1. `src.experiment.trial.run_task()` creates artifact paths, resets the Harbor environment, installs tracing/metrics, and enforces timeouts.
2. `run_task_loop()` builds prompts from the trajectory, asks the configured LLM for tool calls, validates/repairs model output, executes actions through `HarnessEnv`, and updates `TaskLoopState`.
3. `verify` returns the authoritative task judgment and ends the trial when the environment is done.
4. `TaskResult` crosses back to the experiment runner with reward, solved flag, error, step count, metrics, and artifact paths.

Contracts:

- environment adapters implement `src.harness.contracts.HarnessEnv`
- LLM adapters implement `src.adapters.llm_base.BaseLlm`
- trial outputs use `src.harness.contracts.TaskResult`

## LLM

Supported harness providers:

- `openrouter`: uses `OPENROUTER_API_KEY`
- `chatgpt_codex`: experimental Codex/ChatGPT OAuth transport; requires `codex login` and explicit `model_name` plus `max_context_length`

Primary modules:

- `src/adapters/open_router.py`
  - OpenRouter transport, timeout/retry handling, token accounting
- `src/adapters/chatgpt_codex.py`
  - experimental Codex/ChatGPT transport
  - reads Codex auth
  - converts harness requests to Responses calls
  - parses streaming events
- `src/adapters/llm_base.py`
  - LLM adapter interface

Flow labels:

- `Task Execution Prompts`: prompt replay from task instructions, trajectory, and current observation into model-facing messages
- `LLM call traces`: request/response metadata captured through tracing and metrics
- `Tool Call + Reasoning traces`: model-emitted tool calls plus reasoning/response events persisted under trial artifacts

## Terminal Bench Environment

We are currently only supporting terminal bench. Maybe more environments in the future.

Primary module: `src/adapters/env.py`.

The environment adapter:

- resolves tasks through `TaskDirectoryResolver`
- checks `task_overrides/<task_id>/` before Harbor registry tasks
- starts Harbor-backed Docker task environments
- executes terminal commands from harness `run` actions
- runs the authoritative verifier from harness `verify` actions
- writes per-trial task artifacts under the configured experiments dir
- shares a semaphore for heavyweight reset/run/verify work

Flow labels:

- `Executes terminal commands`: harness `run` actions enter the container through `HarnessEnv.exec`
- `Terminal Exec results`: stdout/stderr/return-code observations return to the harness loop
- `Terminal Exec Logs metrics`: execution logs, timing, verifier output, and failure-mode metrics are persisted for diagnosis

## Artifacts for Self-Improvement

Durable experiment files:

- `experiments/learning.md`: cumulative diagnosis memo written by the outer agent
- `experiments/state.json`: active baseline id, current experiment id, update time
- `experiments/<experiment_id>/experiment.json`: full experiment record
- `experiments/<experiment_id>/tasks/<task_id>/<run_id>/...`: per-trial artifacts

Important record types:

- `ExperimentState`: active/current experiment index
- `ExperimentRecord`: experiment metadata, status, panels, evidence
- `PanelRecord`: panel lifecycle, task ids, task results, panel evaluation
- `TaskTrials`: per-task trial aggregation and majority-solved result
- `ExperimentEvidence`: candidate commit plus per-panel `TaskOutcomeEvidence`

`TaskResult` is the trial-output contract from `src/harness/contracts.py`; `TaskTrials` stores those values inside the experiment record.

Statuses:

- `keep`: candidate or baseline is accepted
- `discard`: candidate is rejected
- `crash`: experiment-level failure; available evidence is still written

Crash handling separates trial-level and experiment-level failures. A terminal trial with `error` set is preserved for diagnosis and excluded from solve counts/gate evidence. A run with no valid trials in an active panel becomes an experiment-level `crash`.

Evidence links in `experiment.json.evidence.panel_outcomes` point to representative task artifacts for new solves, regressions, unsolved tasks, or crashes.

## Experiment And Gate

`ExperimentRunner.run()` lifecycle:

1. create `experiments/<experiment_id>/`
2. load frozen active baseline from `experiments/state.json`
3. validate candidate panel order/task sets against the frozen baseline when a baseline exists
4. resolve active panel task dirs
5. persist an initialized `ExperimentRecord`
6. run panels in configured order
7. evaluate each completed panel
8. refresh evidence
9. finalize `keep`, `discard`, or `crash`
10. update `experiments/state.json`
11. preserve discarded/crashed candidate commits under `refs/experiments/failed/<experiment_id>`

Panel execution:

- tasks are launched longest-prior-first using [config/task_duration_priors.json](./config/task_duration_priors.json) when available
- `max_trial_concurrency` bounds live trials
- `max_heavy_action_concurrency` bounds reset/run/verify actions
- majority decisions can stop extra trials early
- candidate tasks that were deterministically solved by the baseline can start with a single trial and expand to `task_trials` on suspected failure

Baseline execution:

- `ExperimentRunner.run_baseline_at_head()` runs the current `HEAD` as a full kept baseline
- `uv run auto` calls it when no baseline exists, when `HEAD` has advanced beyond the active baseline, or when committed panel order/task sets no longer match the active baseline
- a baseline run that crashes is not installed

Gate rules:

- candidate task results compare against the frozen active baseline for the same panel
- prior candidate trials are not pooled into the control
- per-task comparison uses `compare_candidate_against_baseline()` from `src/metrics.py`
- alpha is `PROMOTION_P_VALUE_ALPHA`
- a regression verdict discards the panel
- a promotion panel keeps only if at least one task significantly improves and no task regresses
- a regression-veto panel can only block; it cannot promote by itself
- if a candidate still majority-solves a task, a statistical regression verdict is floored to unchanged
- tasks with no baseline samples are treated as no-baseline frontier tasks

The final experiment status comes from the promotion panel unless a later regression-veto panel discards.

## Artifact Layout

Experiment-level:

```text
experiments/
├── state.json
├── learning.md
└── <experiment_id>/
    ├── experiment.json
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

Supervisor-level:

```text
../harness-experiment_supervisor/<repo-fingerprint>/
├── events.jsonl
├── state.json
└── workspace/
```

## Module Ownership

- `src/cli.py`: console entrypoints and runtime config loading
- `src/harness/config.py`: strict runtime config models
- `src/harness/core.py`: **The main harness policy/action loop (agent is only allowed to modify this and the associated unit test)**
- `src/harness/contracts.py`: typed environment/result contracts
- `src/adapters/env.py`: Harbor/Docker task environment
- `src/adapters/open_router.py`: OpenRouter provider adapter
- `src/adapters/chatgpt_codex.py`: experimental Codex/ChatGPT provider adapter
- `src/adapters/llm_base.py`: LLM adapter interface
- `src/trace.py`: trace writer and stable artifact filenames
- `src/metrics.py`: task metrics, failure modes, baseline comparison statistics
- `src/experiment/trial.py`: per-trial lifecycle
- `src/experiment/record.py`: durable experiment/state models
- `src/experiment/gate.py`: promotion/regression-veto decisions
- `src/experiment/runner.py`: panel orchestration and experiment conclusion
- `src/control/supervisor.py`: autonomous loop orchestration
- `src/control/gates.py`: supervisor candidate/diagnosis policy checks
- `src/control/supervisor_state.py`: supervisor state and events
- `src/control/agent_backend.py`: Codex/Claude subprocess adapters
- `src/control/repo.py`: git wrapper operations
