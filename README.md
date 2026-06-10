# Harness Experiment

Experimental harness for improving an LLM shell agent on Terminal-Bench 2 tasks.

The harness is an LLM shell agent. `uv run auto` lets an automation agent propose one focused change, test it against the current baseline, and keep or discard it based on measured evidence.

Technical details: [TECH_DESIGN.md](./TECH_DESIGN.md). Operating policy for the self-improvement run: [program.md](./program.md).

Blog Post: [Background, motivation and learnings of this project](https://www.henrypan.com/blog/2026-05-25-self-improvement-harness/).

## Setup

Requires Python 3.13+, Docker, `uv`, and a source checkout.

```bash
uv sync
source .venv/bin/activate
```

Configure the harness in [config/harness_config.json](./config/harness_config.json). Use [config/harness_config.template.json](./config/harness_config.template.json) as the commented reference.

The committed default is a bounded smoke run, not benchmark-quality evidence.

There are two layers here.
- task-solving LLM
  - A LLM provider is required because every task step sends model requests. Unless you're running a local LLM, in which case, you can extend the `src/llm/base.py` to support that yourself. Currently we support two types of LLM providers:
    - <details>
        <summary>OpenRouter setup</summary>

        Set `OPENROUTER_API_KEY` in your shell or a repo-local `.env`.

        Use an OpenRouter provider config in `llm_provider_config`. If the checked-in provider routing is unavailable in your account, update `llm_provider_config.provider_kwargs.provider`.

    </details>

    - <details>
        <summary>Codex subscription setup</summary>

        Run `codex login` once. The `chatgpt_codex` LLM provider (`src/llm/codex.py`) reads Codex auth from `CODEX_HOME/auth.json` or `~/.codex/auth.json`

    </details>
- outer self-improving Agent
  - codex (already logged in)
  - claude code (already logged in)



## Run Experiments

Prerequisites: Docker is running, provider credentials are configured, [config/harness_config.json](./config/harness_config.json) has the intended panels/budgets, and the worktree is clean.

Warnings: `uv run exp` uses paid/quota-limited LLM calls; it writes to `experiments/`; repeated manual runs need a new `experiment_id`. For throwaway local runs only, `EXP_ALLOW_DIRTY_WORKTREE=1 uv run exp` bypasses the clean-worktree gate.

```bash
uv run exp
```

It should print out a progress bar indicating the progress of this one-off experiment. Something like this:

```
[------------------------] 0/49 tasks (0%) | trials 3/105 | solved 2/3 | errors 0 | active 8 | 14s elapsed, ~-- left
[####--------------------] 10/49 tasks (20%) | trials 28/105 | solved 9/12 | errors 0 | active 8 | 1m12s elapsed, ~4m40s left
[################--------] 33/49 tasks (67%) | trials 78/105 | solved 27/35 | errors 1 | active 8 | 5m02s elapsed, ~2m26s left
```

## Run Self-Improvement

Prerequisites: all experiment prerequisites, a clean committed `HEAD`, and an authenticated automation CLI (`codex` by default, or `claude` with `--agent claude`).

For `uv run auto`, Codex is the default agent. The run provisions a separate Codex home under `../harness-experiment_supervisor/codex-home` with symlinks to your `~/.codex/auth.json` and `~/.codex/config.toml`.

Warnings: `uv run auto` has no built-in budget cap, keeps running until Ctrl+C or an error, uses paid/quota-limited calls for both the self-improvement agent and task-solving LLM, runs agent CLIs without permission prompts, and runs each baseline/candidate in a throwaway sibling worktree (a kept candidate lands on the primary via fast-forward). Use a dedicated checkout.

```bash
uv run auto
uv run auto --agent codex
uv run auto --agent claude
```

Once it starts, it should print out the self-improvement agent's thinking and tool call traces in real-time:

```bash
> uv run auto --agent claude
[supervisor] loop_iteration_started
[claude] thread 98f1d6ff-1e3e-4988-9b11-e34d5e47a250
[claude] I'll read the authoritative files to understand the current state and what's needed for this prelaunch phase.
  [toolcall] read /Users/henry/Git/harness-experiment_supervisor/harness-experiment-e930d607ca91/workspace/program.md
  [toolcall] read /Users/henry/Git/harness-experiment_supervisor/harness-experiment-e930d607ca91/workspace/config/harness_config.json
  [toolcall] read /Users/henry/Git/harness-experiment_supervisor/harness-experiment-e930d607ca91/workspace/experiments/learning.md
  [toolcall] read experiments/exp-20260601-160651/experiment.json

...

[------------------------] 0/49 tasks (0%) | trials 3/105 | solved 2/3 | errors 0 | active 8 | 14s elapsed, ~-- left
[####--------------------] 10/49 tasks (20%) | trials 28/105 | solved 9/12 | errors 0 | active 8 | 1m12s elapsed, ~4m40s left
[################--------] 33/49 tasks (67%) | trials 78/105 | solved 27/35 | errors 1 | active 8 | 5m02s elapsed, ~2m26s left
```

## Results

After you've ran `uv run auto` for a while to complete a couple of iterations, you can start to examine:
- `experiments/learning.md`: the cumulative agent-generated human-readable diagnosis memo.
- `experiments/<experiment_id>/experiment.json` for a run's outcome, and its `loop.json` for the keep/discard decision. The current baseline and any in-flight run are derived by scanning these files plus git.

- `experiment.json` carries `run_status` (`running`, `completed`, or `crashed`); the keep/discard decision lives alongside it in `loop.json`, whose per-task verdicts carry the gate's evidence.
- Per-trial artifacts live under `experiments/<experiment_id>/tasks/<task>/<run_id>/`.
- Automation working dirs (`codex-home/`, per-repo `worktrees/`, agent thread cache) live under `../harness-experiment_supervisor/`.

<details>
<summary>Result examples</summary>

```text
experiments/
└── exp-.../
    ├── experiment.json
    ├── loop.json
    └── tasks/
        └── regex-log/
            └── 20260525-215314-213b2141/
                ├── agent/
                │   ├── exec.log
                │   ├── metrics.json
                │   └── steps.jsonl
                ├── artifacts/
                ├── bootstrap/
                └── verifier/
                    ├── ctrf.json
                    ├── reward.txt
                    └── test-stdout.txt
```

```text
../harness-experiment_supervisor/
├── codex-home/
└── harness-experiment-<hash>/
    ├── worktrees/
    └── workspace/
```

</details>

## Optional

<details>
<summary>Local package caches</summary>

For multi-trial runs, [scripts/setup-apt-cache.sh](./scripts/setup-apt-cache.sh) reduces repeated `apt` bootstrap stalls. For ML-heavy tasks, [scripts/setup-pypi-cache.sh](./scripts/setup-pypi-cache.sh) reduces repeated verifier dependency downloads.

</details>

<details>
<summary>Task overrides</summary>

Drop a valid Harbor task layout under `task_overrides/<task_id>/` to shadow a Terminal-Bench task or run a local task. Configure or disable overrides in [config/harbor_config.toml](./config/harbor_config.toml).

</details>
