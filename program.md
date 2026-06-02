# program.md

Operating policy for the autonomous harness-search loop.

## Objective

Improve the harness within the current experiment.
Treat Harbor tasks as black-box evaluations.
Increase per-task solve rates on the train panel without significantly regressing any baseline-solved train task.

This is a narrow mechanism-search loop, not a broad repo-wide self-improvement loop.
If this file and code disagree, code is source of truth.

## Source-of-truth boundary

Main behavior surface:

- `src/harness/core.py`

Allowed support edits during prelaunch:

- `config/harness_config.json`
- `tests/harness/test_core.py`

The candidate patch must produce a behavioral change in `src/harness/core.py`.

`config/harness_config.json` may change only candidate identity fields:

- `focus_name`

Do not change frozen runtime contract fields:

- `experiment_id` (supervisor-owned)
- `panels[].task_names` and panel order (changing a panel's task set or order triggers a full baseline rerun on the next `uv run auto`)
- model/provider config
- max steps, timeouts, concurrency
- retry budgets
- `task_trials`

Do not edit other tracked files visible in the sparse workspace. They are read-only context, except for `experiments/learning.md` during post-run diagnosis (see `Post-Run Diagnosis`).
Do not edit runner state by hand:

- `experiments/state.json`
- `experiments/<experiment_id>/experiment.json`

## Promotion

A candidate is evaluated against the active baseline with per-task two-sided Fisher exact tests at alpha = 0.05, comparing the candidate's solved/total counts against the baseline's solved/total counts.

1. A per-trial infrastructure failure is retried within the trial on a bounded internal budget. If it still fails, the trial concludes as a terminal `crash`, excluded from all evidence, with no effect on the experiment status or other trials (and it is not re-run). Experiment-level failures still make the experiment `crash`: setup, task resolution, evaluation, or a run that produced zero valid trials across the whole panel. A baseline run that crashes is not promoted; no baseline is installed and the next `uv run auto` reruns it.
2. If there is no baseline yet, `uv run auto` first runs the current panel as a kept baseline, then starts candidate search.
3. Otherwise `keep` iff some train task significantly improved (p < 0.05, candidate rate above baseline) AND no train task significantly regressed (p < 0.05, candidate rate below baseline). A task with no baseline samples counts as "improved" if the candidate majority-solves it. Else `discard`.

There is no mean-reward promotion rule. Family-wise error is intentionally uncontrolled.

## Panel changes

The train panel is fixed by default. If the user edits a panel's `task_names`
(or changes panel order) and commits that change, the next `uv run auto` treats
it as a new baseline regime and reruns the full current panel set before
candidate search continues.

## Evidence

Use:

- task instructions
- normal command stdout/stderr
- `agent/steps.jsonl`
- `agent/metrics.json`
- `agent/exec.log`
- verifier stdout captured in task artifacts
- `experiment.json.evidence.panel_outcomes` (per-panel lists of per-task outcome evidence)
- local experiment records
- preserved failed refs
- repo-local code and git history

Do not use hidden tests, verifier implementation details, or reverse-engineered reward logic.

Diagnosis may read task-specific evidence. Harness code and tests must remain task-agnostic: no hard-coded task ids, dataset/repo names, expected answers, commit hashes, task filenames, README phrases, or benchmark-instance constants. Encode only generic state/action/policy mechanisms.

## Prelaunch Process

1. Read `program.md`, `config/harness_config.json`, `experiments/learning.md`, active baseline record, latest candidate record when present, and surfaced evidence artifacts.
2. Review `experiments/learning.md` sections `Current bottleneck`, `Exhausted mechanisms`, and `Research leads`.
3. Choose one small mechanism-scoped hypothesis that is structurally different from recently discarded candidates. Treat unsolved baseline train tasks as the frontier.
4. Before changing `src/harness/core.py`, write one neutral lifecycle-contract test in `tests/harness/test_core.py` that fails before the change and passes only for the generic contract.
5. Set `focus_name` to a short mechanism label. The supervisor assigns `experiment_id`.
6. Run the focused lifecycle-contract test and directly affected tests.
7. Stop when the candidate is ready for tracked launch. The supervisor commits and launches.

The supervisor enforces every rule above and will reject candidates that violate them.

## Post-Run Diagnosis

1. Read the concluded experiment record, surfaced evidence artifacts, and `experiments/learning.md`.
2. Use trace evidence (`agent/steps.jsonl`, `agent/metrics.json`) to reason about whether the candidate's mechanism reached its intended state transitions. Start with `metrics.json.rule_fires` (mechanism instrumentation) and `metrics.json.failure_mode` (per-trial terminal-state bucket: `solved | verified_rejected | never_verified | hit_step_cap | hit_timeout | no_valid_action | interrupted | crash`, where `crash` is an infra failure that exhausted internal retries and `interrupted` is a trial stopped from the outside by Ctrl-C or a supervisor restart -- both excluded from evidence).
3. Update only `experiments/learning.md`. Keep it concise, cumulative, generic, and organized by stable harness-design domains.
4. Stop when the memo update is complete.

Unexpected blockers should fail normally; the supervisor handles runtime errors.
