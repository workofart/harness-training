# Harness Experiment

**Background, motivation, and learnings**: [blog post](https://www.henrypan.com/blog/2026-05-25-self-improvement-harness/).

![Infrastructure for experiment and self-improvement loop](https://www.henrypan.com/blog/assets/images/ml/harness-design/self-improvement-loop-infrastructure.png)

A self-improvement experiment in two layers:

- **Inner — the harness** (`src/harness/core.py`): an LLM shell agent that solves [Terminal-Bench 2](https://www.tbench.ai/benchmarks/terminal-bench-2) tasks.
- **Outer — the supervisor** (`uv run auto`): a coding agent (Codex or Claude Code) that proposes one focused change to the inner harness per cycle, measures it against the current baseline, and keeps or discards it on statistical evidence.

Architecture: [TECH_DESIGN.md](./TECH_DESIGN.md). Operating policy the outer agent follows: [program.md](./program.md).

## Setup

Requires Python 3.13+, Docker, `uv`, and a source checkout.

```bash
uv sync
source .venv/bin/activate
```

Configure the harness in [config/harness_config.json](./config/harness_config.json); [config/harness_config.template.json](./config/harness_config.template.json) is the commented reference. The committed default is a bounded smoke run, not benchmark-quality evidence.

Credentials, per layer:

- **Task-solving LLM** (required for everything — every task step calls a model):
  - `openrouter`: set `OPENROUTER_API_KEY` in your shell or a repo-local `.env`. If the checked-in provider routing is unavailable to your account, adjust `llm_provider_config.provider_kwargs.provider`.
  - `chatgpt_codex`: run `codex login` once; auth is read from `CODEX_HOME/auth.json` or `~/.codex/auth.json`.
  - Local or other models: implement `src/llm/base.py`.
- **Outer agent** (only for `uv run auto`): an authenticated `codex` (default) or `claude` CLI.

## Run a one-off experiment

`uv run exp` runs the configured task panel against whatever code is checked out and writes one raw record to `experiments/<id>/experiment.json` — no baseline, no gating.

It uses paid/quota-limited LLM calls, needs Docker running, and requires a clean worktree (`EXP_ALLOW_DIRTY_WORKTREE=1` bypasses, for throwaway runs only).

```bash
uv run exp
```

Progress streams to the terminal:

```
[####--------------------] 10/49 tasks (20%) | trials 28/105 | solved 9/12 | errors 0 | active 8 | 1m12s elapsed, ~4m40s left
```

## Run the self-improvement loop

```bash
uv run auto                 # codex agent (default)
uv run auto --agent claude
```

Each cycle: the outer agent edits `src/harness/core.py` in a sparse worktree → the candidate runs in a throwaway sibling worktree → a statistical gate keeps (fast-forwards onto the primary) or discards → the agent writes a diagnosis.

Use a dedicated checkout. `uv run auto`:

- has **no budget cap** — it runs until Ctrl+C or an error, spending on both the outer agent and the task-solving LLM
- runs agent CLIs **without permission prompts**
- requires a clean, committed `HEAD`; working dirs (worktrees, an isolated codex-home symlinked to your `~/.codex` auth) live under `../harness-experiment_supervisor/`

It streams the agent's thinking and tool calls, then the same per-run progress bar as `exp`:

```
[supervisor] loop_iteration_started
[claude] I'll read the authoritative files to understand the current state...
  [toolcall] read .../workspace/program.md
  [toolcall] read .../workspace/config/harness_config.json
```

## Results

- `experiments/learning.md` — the cumulative agent-written diagnosis memo. Start here.
- `experiments/<id>/experiment.json` — a run's raw outcome; `loop.json` next to it — the keep/discard decision with per-task evidence.
- Full artifact tree (per-trial logs, verifier output, supervisor dirs): [TECH_DESIGN.md → Artifact layout](./TECH_DESIGN.md#artifact-layout).

## Optional

<details>
<summary>Local package caches</summary>

For multi-trial runs, [scripts/setup-apt-cache.sh](./scripts/setup-apt-cache.sh) reduces repeated `apt` bootstrap stalls. For ML-heavy tasks, [scripts/setup-pypi-cache.sh](./scripts/setup-pypi-cache.sh) reduces repeated verifier dependency downloads.

</details>

<details>
<summary>Task overrides</summary>

Drop a valid Harbor task layout under `task_overrides/<task_id>/` to shadow a Terminal-Bench task or run a local task. Configure or disable overrides in [config/harbor_config.toml](./config/harbor_config.toml).

</details>
