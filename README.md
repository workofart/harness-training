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
  - A LLM provider is required because every task step sends model requests. Unless you're running a local LLM, in which case, you can extend the `src/adapters/llm_base.py` to support that yourself. Currently we support two types of LLM providers:
    - <details>
        <summary>OpenRouter setup</summary>

        Set `OPENROUTER_API_KEY` in your shell or a repo-local `.env`.

        Use an OpenRouter provider config in `llm_provider_config`. If the checked-in provider routing is unavailable in your account, update `llm_provider_config.provider_kwargs.provider`.

    </details>

    - <details>
        <summary>Codex subscription setup</summary>

        Run `codex login` once. The `chatgpt_codex` LLM provider (`src/adapters/chatgpt_codex.py`) reads Codex auth from `CODEX_HOME/auth.json` or `~/.codex/auth.json`

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
[------------------------] 0/49 tasks (0%) | trials 1/105 | solved 1/1 | errors 0
[------------------------] 0/49 tasks (0%) | trials 2/105 | solved 2/2 | errors 0
[------------------------] 0/49 tasks (0%) | trials 3/105 | solved 2/2 | errors 0
```

## Run Self-Improvement

Prerequisites: all experiment prerequisites, a clean committed `HEAD`, and an authenticated automation CLI (`codex` by default, or `claude` with `--agent claude`).

For `uv run auto`, Codex is the default agent. The run provisions a separate Codex home under `../harness-experiment_supervisor/codex-home` with symlinks to your `~/.codex/auth.json` and `~/.codex/config.toml`.

Warnings: `uv run auto` has no built-in budget cap, keeps running until Ctrl+C or an error, uses paid/quota-limited calls for both the self-improvement agent and task-solving LLM, runs agent CLIs without permission prompts, and hard-resets the primary worktree between baseline/candidate commits. Use a dedicated checkout.

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

[------------------------] 0/49 tasks (0%) | trials 1/105 | solved 1/1 | errors 0
[------------------------] 0/49 tasks (0%) | trials 2/105 | solved 2/2 | errors 0
[------------------------] 0/49 tasks (0%) | trials 3/105 | solved 2/2 | errors 0
```

## Results

After you've ran `uv run auto` for a while to complete a couple of iterations, you can start to examine:
- `experiments/learning.md`: the cumulative agent-generated human-readable diagnosis memo. Then read 
- `experiments/state.json` for the current baseline/current experiment and `experiments/<experiment_id>/experiment.json` for the verdict.

- In `experiment.json`, `status` is the run decision (`keep`, `discard`, or `crash`) and `evidence.panel_outcomes` links the verdict to representative task artifacts.
- Per-trial artifacts live under `experiments/<experiment_id>/tasks/<task>/<run_id>/`.
- Automation state/events live under `../harness-experiment_supervisor/<repo-fingerprint>/`.

<details>
<summary>Result examples</summary>

```json
{
  "active_baseline_experiment_id": "baseline-20260525-190114",
  "current_experiment_id": "exp-20260525-191311",
  "updated_at": "2026-05-25T19:20:06.438753+00:00"
}
```

```text
experiments/
└── exp-.../
    ├── experiment.json
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
└── harness-experiment-<hash>/
    ├── events.jsonl
    ├── state.json
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
