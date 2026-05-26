## Technical Details

### Entrypoints

Configured in `pyproject.toml`:

- `exp` -> `src.cli:main_exp`
  - loads `config/harbor_config.toml`, `config/harness_config.json`, and `OPENROUTER_API_KEY`
  - runs one tracked experiment via `ExperimentRunner`
  - when an active baseline exists, fails fast if the candidate train panel does not match it
- `auto` -> `src.cli:main_auto`
  - creates the selected agent backend
  - runs the supervisor loop
  - default backend is Codex; `--agent claude` selects Claude Code

### Key Data Structures

- `HarnessConfig`
  - runtime experiment config from `config/harness_config.json`
  - owns `experiment_id`, `focus_name`, `train_task_names`, task budgets, and LLM provider config
- `ExperimentRecord`
  - persisted experiment record at `experiments/<experiment_id>/experiment.json`
  - aggregates task trials, status, decision reason, git commit, and evidence
- `ExperimentState`
  - persisted experiment index at `experiments/state.json`
  - tracks active baseline and current experiment ids
- `TaskResult`
  - one task trial result returned by `run_task`
  - carries reward/solved/error plus artifact paths for trace, metrics, trial dir, and verifier stdout
- `TaskTrials`
  - per-task aggregation of one or more `TaskResult` values
  - owns majority-solved selection and representative trial selection
- `ExperimentEvidence`, `CandidateChangeEvidence`, `TaskOutcomeEvidence`
  - persisted comparison evidence and task-level artifact links
- `SupervisorState`
  - persisted supervisor recovery state under `../<repo>_supervisor/.../state.json`
- `RuntimeSnapshot`
  - in-memory snapshot of repo config, experiment state, active baseline, and current candidate
- `PreparedCandidate`
  - prelaunch output: agent thread id, experiment id, changed paths, and candidate harness config

### Module Ownership Boundaries

- `src/harness/core.py`
  - active harness behavior surface
  - action models, prompt building, LLM output parsing/repair policy, and per-task episode loop
- `src/harness/config.py`
  - runtime config models and validation
- `src/harness/contracts.py`
  - shared typed contracts between harness, environment, trace, and runner
- `src/adapters/env.py`
  - Harbor-backed environment adapter and task-directory resolution
- `src/adapters/open_router.py`
  - OpenRouter transport, token counting, provider retry/shutdown
- `src/adapters/llm_base.py`
  - abstract LLM interface and typed completion dataclasses
- `src/trace.py`
  - trace writer, task metrics, stable artifact filenames
- `src/metrics.py`
  - derived task metrics and failure-mode classification
- `src/experiment/trial.py`
  - per-task trial lifecycle around `run_task_loop`
  - timeout, cleanup, artifact recovery, and `TaskResult`
- `src/experiment/runner.py`
  - experiment persistence, baseline/candidate evaluation, gate decisions, and panel orchestration
- `src/control/supervisor.py`
  - autonomous supervisor loop, sparse workspace management, candidate validation, launch/recovery, and post-run diagnosis
- `src/control/agent_backend.py`
  - Codex/Claude subprocess backends
  - agent event parsing, terminal formatting, and default supervisor Codex home
- `src/control/repo.py`
  - git command wrapper for worktree, sparse checkout, reset, refs, and dirty checks
- `src/cli.py`
  - console entrypoints for `uv run exp` and `uv run auto`
  - command-line agent selection for the supervisor loop

### Task Overrides

`TaskDirectoryResolver` in `src/adapters/env.py` checks `./task_overrides/<task_id>/` before falling back to the Harbor registry. Drop a valid Harbor task layout there (verified via `TaskPaths.is_valid()`) to either:

- run the harness against a task that is not in the published Terminal-Bench 2 dataset, or
- shadow a published task locally (e.g., to patch a flaky bootstrap or test).

The directory is empty by default and is not required for normal use. Path is configurable through `task_overrides_dir` in [`config/harbor_config.toml`](./config/harbor_config.toml); set it to `null` to disable overrides entirely.
