# Harness Experiment

An experimental project that explores the possibility of agent-driven self-improving harness.

Here's the blog post that might bring more color: [Link](https://www.henrypan.com/blog/2026-05-25-self-improvement-harness/)


## Key Concepts

- **The harness** ([`src/harness/core.py`](./src/harness/core.py)): an LLM-driven shell agent with an 8-action vocabulary (`list_dir`, `find_files`, `search_text`, `read_file`, `write_file`, `edit_file`, `run`, `verify`) that attempts [Terminal-Bench 2](https://www.tbench.ai/benchmarks/terminal-bench-2) tasks. Tasks are fetched through the [Harbor](https://pypi.org/project/harbor/) registry and run inside Docker containers.
- **The self-improvement loop** ([`src/control/supervisor.py`](./src/control/supervisor.py)): an *outer* coding agent — Codex by default, or Claude Code with `--agent claude` — reads [`program.md`](./program.md), proposes one mechanism-scoped patch to the harness, then `uv run exp` measures the candidate against the active baseline on a fixed train panel. The supervisor decides `keep` or `discard` using a per-task two-sided binomial test, commits or reverts accordingly, and loops.

[`program.md`](./program.md) is the operating policy the outer agent reads on every iteration. It defines what files the agent may edit, the promotion rule, and the post-run diagnosis protocol. If you want to understand the system, read it after this README.


## Before You Run

This repository is intended to be run from a source checkout, not from an installed wheel. The commands below assume you are in the repo root so `config/`, `program.md`, and local experiment state are available.

Important behavior:

- **`uv run exp` uses paid/quota-limited LLM calls.** The harness calls the configured LLM provider on every task step.
- **`uv run auto` adds a second paid/quota-limited agent loop.** The supervisor calls Codex or Claude while the harness calls its configured LLM provider. There is no built-in budget cap; stop it with Ctrl+C.
- **`uv run auto` moves your git HEAD.** The supervisor `git reset --hard`s your primary worktree between baseline and candidate commits as it evaluates patches (see [`promote_workspace_commit_to_repo`](./src/control/supervisor.py)). Run `auto` in a dedicated checkout, not on a branch you care about.
- **The supervisor agent CLIs run without permission prompts.** The Codex backend uses `--dangerously-bypass-approvals-and-sandbox`; the Claude backend uses `--dangerously-skip-permissions`.


## How to Run

1. Install the project:

- If you do not have `uv`, install it from the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).
- Requires Python 3.13 or newer. If your system Python is older, `uv sync` will download a compatible interpreter automatically.

```bash
uv sync
source .venv/bin/activate
```

2. Prerequisites for running Terminal-Bench tasks:

- LLM credentials: for OpenRouter, set `OPENROUTER_API_KEY` in the shell or in a repo-local `.env`; for experimental `chatgpt_codex`, run `codex login` first so `~/.codex/auth.json` exists.
- Docker: install and start Docker. Harbor starts one container per task trial; `max_trial_concurrency` in [`config/harness_config.json`](./config/harness_config.json) caps trials in flight, and `max_env_concurrency` caps how many run a container command at once (host-CPU bound).
- Optional but recommended for multi-trial runs — local apt cache: run [`scripts/setup-apt-cache.sh`](./scripts/setup-apt-cache.sh) once. Task images are minimal and reinstall `python3` via `apt` on every trial; the cache serves those repeats locally so concurrent bootstraps don't stall on the public mirror (the dominant infra failure). The bootstrap auto-detects it on `host.docker.internal:3142` and falls back to the direct mirror when absent.

3. Prerequisites for the self-improvement loop:

- Install and authenticate one agent CLI:
  - [`codex`](https://github.com/openai/codex) (default backend) — sign in once with `codex login`. The supervisor reads `~/.codex/auth.json` and `~/.codex/config.toml`.
  - [`claude`](https://github.com/anthropics/claude-code) (optional, selected with `--agent claude`) — sign in once with `claude`.

4. Configure the run in [`config/harness_config.json`](./config/harness_config.json):

- The committed default is a one-task smoke run: `log-summary-date-ranges`, `max_trial_concurrency: 1`, `task_trials: 1`, `max_steps: 100`. This is meant to validate local setup with bounded Docker/API usage, not to produce benchmark-quality evidence.
- The default OpenRouter config includes provider routing used by the checked-in tests. If that provider is unavailable in your OpenRouter account, update `llm_provider_config.provider_kwargs.provider`. Experimental `chatgpt_codex` requires explicit `model_name` and `max_context_length`.
- For serious comparisons, expand `train_task_names`, raise budgets deliberately, and commit the config before running `uv run auto`.
- See [`config/harness_config.template.json`](./config/harness_config.template.json) for a commented template.

5. Baseline and worktree rules:

- Experiments compare a candidate against the active baseline recorded in `experiments/state.json`.
- Commit the harness you want measured before running `uv run exp` or `uv run auto`. Both commands assume a clean worktree so results map to a specific git commit.
- `uv run auto` refreshes the baseline on a clean worktree when no active baseline exists, or when `HEAD` has advanced past the active baseline commit.
- `uv run exp` runs one candidate against the current active baseline. Its train panel must match the active baseline's train panel. If you need to change the task panel, use `uv run auto` on a clean committed `HEAD` so it can refresh the baseline first.
- Change `experiment_id` before each repeated manual `uv run exp`; experiment directories are not overwritten.
- For throwaway local runs only, `EXP_ALLOW_DIRTY_WORKTREE=1 uv run exp` bypasses the clean-worktree gate for `exp`. Do not use that override for results you intend to compare or promote.

6. Run one tracked experiment:

- This can be helpful when you don't want the agent to get involved, and you just want to test one change in an ad-hoc fashion

```bash
uv run exp
```

7. Run the self-improvement loop:

- This keeps looping until you stop it with Ctrl+C or a runtime error occurs.

```bash
uv run auto # default is codex
uv run auto --agent claude # claude -p might start charging credits, so be careful
uv run auto --agent codex
```


## Where to See Experiment Output Artifacts

Experiment-level artifacts:

- `experiments/state.json`
```json
// Example
{
  "active_baseline_experiment_id": "baseline-20260525-190114",
  "current_experiment_id": "exp-20260525-191311",
  "updated_at": "2026-05-25T19:20:06.438753+00:00"
}
```

- `experiments/learning.md`
  - cumulative post-run diagnosis notes written by the agent
- `experiments/<experiment_id>/experiment.json`
```json
// Example
{
  "experiment_id": "exp-20260525-191311",
  "parent_baseline_experiment_id": "baseline-20260525-190114",
  "git_commit_hash": "...",
  "focus_name": "usually the hypothesis the agent is testing",
  "train_task_ids": [
    "configure-git-webserver",
    // ...
  ],
  // keep: candidate is promoted as the active baseline
  // discard: candidate is not promoted; its commit is preserved under `refs/experiments/failed/<experiment_id>`
  // crash: run failed; record and available artifacts are still written
  "status": "usually 'discard' or 'keep' or 'crash'",
  "train_solved_count": ...,
  "decision_reason": "train task configure-git-webserver regressed",
  "error": "",
  "started_at": "...",
  "finished_at": "...",
  "train_task_results": {
    "configure-git-webserver": {
      "task_name": "configure-git-webserver",
      "expected_trial_count": 1,
      "trials": [
        {
          "task_name": "configure-git-webserver",
          "reward": 0.0,
          "solved": false,
          "error": "...",
          "steps_used": 79,
          "trial_dir": "...",
          "trace_path": "...",
          "metrics_path": "...",
          "metrics": {
            ...
          },
          "started_at": "...",
          "finished_at": "..."
        }
      ]
    },
  },
  "evidence": {}
}
```

**Task-level artifacts:**
```text
experiments/
└── exp-.../                                # one experiment run
    ├── experiment.json                     # experiment-level config/metadata
    └── tasks/                              # per-task results
        └── regex-log/                      # task name
            └── 20260525-215314-213b2141/   # one trial: YYYYMMDD-HHMMSS-<8hex>
                ├── agent/                  # agent execution traces
                │   ├── exec.log            # raw command/session log
                │   ├── metrics.json        # run metrics/timing/token stats
                │   └── steps.jsonl         # step-by-step agent events
                ├── artifacts/              # files produced/collected from run
                ├── bootstrap/              # setup phase records
                │   ├── bootstrap.sh        # setup script used for task env
                │   └── return-code.txt     # setup script exit code
                └── verifier/               # grading/test outputs
                    ├── ctrf.json           # structured test report
                    ├── reward.txt          # scalar verifier score/reward
                    └── test-stdout.txt     # verifier stdout
```

**Supervisor artifacts:**

Supervisor state lives outside the clone in a sibling directory named after the repo directory with `_supervisor` appended. For this checkout, that directory is `../harness-experiment_supervisor`; the default Codex home is `../harness-experiment_supervisor/codex-home`.

```text
../harness-experiment_supervisor/
└── repo-.../         # one repo fingerprint supervisor record
    ├── events.jsonl  # append-only event log; JSONL supervisor events/timestamps
    └── state.json    # latest supervisor state snapshot: phase, thread id, update time, postrun payload/log
```

## Technical Design

See [TECH_DESIGN.md](./TECH_DESIGN.md)
