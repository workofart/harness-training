# program.md

Operating policy for the autonomous harness-search loop.

## Objective

Improve the harness within the current experiment.
Treat Harbor tasks as black-box evaluations.
Increase aggregate solved-task count on the train panel while preserving task-level evidence about improvements and regressions.

This is a narrow mechanism-search loop, not a broad repo-wide self-improvement loop.
If this file and code disagree, code is source of truth.

## Source-of-truth boundary

Main behavior surface, and the only files you may edit:

- `src/harness/core.py` — the harness mechanism
- `tests/harness/test_core.py` — its contract test

The candidate patch must produce a behavioral change in `src/harness/core.py`.

Every other file visible in your workspace is read-only context. The measurement
protocol — the task panels, model/provider, step caps, timeouts, concurrency,
retry budgets, and trial count — lives in committed config that is **not** in your
view: you neither see nor edit it. Because the protocol is in the committed tree,
the supervisor uses the commit hash as its fingerprint; changing the protocol is
the user's job, not a candidate's.

Do not hand-edit supervisor state. The experiment record
(`experiments/<id>/experiment.json`), the decision record
(`experiments/<id>/loop.json`), and git refs are all supervisor-owned. The
supervisor assigns the `experiment_id`, commits the candidate, manages its refs,
and launches every run.

To name a candidate, write a single-line mechanism label to `.candidate_focus` in
the workspace root during prelaunch. It is git-excluded — never committed — and is
how the supervisor captures the candidate's `focus_name`.

## Benchmark protocol boundary

Verifier/tests are final grading. Do not:

- run final verifier/tests mid-solve or more than once
- show verifier output/reward/diagnostics to the agent before score
- make failed final verification retryable
- expose/reverse-engineer hidden tests, verifier scripts, or reward logic

Single-step TB/Harbor: agent phase, then one final score. Multi-step: only
task-declared step checks.

## Promotion

A candidate is evaluated against the frozen active baseline on pooled per-trial solves, stratified by task. Per-task two-sided Fisher exact verdicts at alpha = 0.05 are still computed as task-level evidence for diagnosis and self-improvement; they do not directly keep or discard a promotion panel.

1. A per-trial infrastructure failure is retried within the trial on a bounded internal budget. If it still fails, the trial concludes as a terminal `crash`, excluded from all evidence, with no effect on the run status or other trials (and it is not re-run). Experiment-level failures still make the run `crash`: setup, task resolution, evaluation, or a run that produced zero valid trials across the whole panel. A baseline run that crashes is not installed; the next `uv run auto` reruns it.
2. If there is no baseline at the current commit, `uv run auto` first runs all configured tasks as a kept baseline, then starts candidate search.
3. Otherwise the promotion panel returns `keep` only if both hold; else `discard`:
   - the stratified solve delta over the train panel is strictly positive (the candidate out-solves each task's own pooled expectation, summed over tasks with trials in both arms — raw pooled rates are confounded by the deterministic-tier single-trial budget);
   - a one-sided Cochran-Mantel-Haenszel test, stratified by task, is significant at alpha = 0.10.
   If the frozen baseline has no panel samples at all (a pure frontier), a higher majority-solved task count is the bar instead of the stratified test.
4. Per-task verdicts (`improvement`, `regression`, `unchanged`, `uncompared`) label task-level evidence for the self-improving agent. They are diagnostic signal, not the promotion trigger. A candidate that still majority-solves a task is never labelled a regression on it (the still-solving floor).
5. A regression-veto panel runs only after a promotion `keep`; it can only block. It discards iff the candidate's aggregate solved-task count drops below the baseline's.

The candidate is kept only if the promotion panel keeps **and** the veto panel does not block. There is no mean-reward promotion rule. No family-wise correction is applied to diagnostic per-task verdicts.

## Panel changes

The task panels are fixed by default and are not yours to change. If the user edits
a panel and commits it, the commit moves, so the active baseline no longer matches
the current commit and the next `uv run auto` reruns the full configured set as a
new baseline before candidate search continues.

## Evidence

Use:

- task instructions
- normal command stdout/stderr
- the remaining wall-clock budget: every environment state carries `time_remaining_sec` (stamped by the executor; `None` means unbounded)
- `agent/steps.jsonl`
- `agent/metrics.json`
- `agent/exec.log`
- verifier stdout captured in task artifacts
- the experiment record `experiments/<id>/experiment.json` (run status + per-task trial results)
- the decision record `experiments/<id>/loop.json` (the gate's keep/discard + per-task verdicts)
- preserved failed candidate refs (`refs/experiments/failed/<id>`)
- `experiments/learning.md`
- repo-local code and git history

Do not use hidden tests, verifier implementation details, or reverse-engineered reward logic.

Diagnosis may read task-specific evidence. Harness code and tests must remain task-agnostic: no hard-coded task ids, dataset/repo names, expected answers, commit hashes, task filenames, README phrases, or benchmark-instance constants. Encode only generic state/action/policy mechanisms.

## Prelaunch Process

1. Read `program.md`, `experiments/learning.md`, and the evidence artifacts the supervisor surfaces.
2. Review `experiments/learning.md` sections `Current bottleneck`, `Exhausted mechanisms`, and `Research leads`.
3. Choose one small mechanism-scoped hypothesis that is structurally different from recently discarded candidates. Treat unsolved baseline train tasks as the frontier.
4. Before changing `src/harness/core.py`, write one neutral lifecycle-contract test in `tests/harness/test_core.py` that fails before the change and passes only for the generic contract.
5. Write a short mechanism label to `.candidate_focus` in the workspace root.
6. Run the focused lifecycle-contract test and directly affected tests.
7. Stop when the candidate is ready for tracked launch. The supervisor commits and launches.

The supervisor enforces every rule above and will reject candidates that violate them.

## Post-Run Diagnosis

1. Read the concluded experiment record (`experiment.json`), its decision record (`loop.json`), the surfaced evidence artifacts, and `experiments/learning.md`.
2. Use trace evidence (`agent/steps.jsonl`, `agent/metrics.json`) to reason about whether the candidate's mechanism reached its intended state transitions. Start with `metrics.json.rule_fires` (mechanism instrumentation) and `metrics.json.failure_mode` (per-trial terminal-state bucket: `solved | verified_rejected | never_verified | hit_step_cap | hit_timeout | no_valid_action | interrupted | crash`, where `crash` is an infra failure that exhausted internal retries and `interrupted` is a trial stopped from the outside by Ctrl-C or a supervisor restart -- both excluded from evidence).
3. Root-cause at least one still-unsolved frontier task mechanistically before drawing any frontier conclusion. Read its instruction and the agent's trace; name the exact criterion that failed; decide whether that criterion was stated in, or derivable from, the instruction and the files the task provides; and check whether the agent's own pre-submit validation actually covered it. Separate "the agent could not have known" (the task withheld the information) from "the agent had the information but did not enforce it" (a validation/policy gap a harness mechanism may close). Diagnosis is task-specific and mechanistic; the harness change it motivates stays task-agnostic.
4. Write the raw per-cycle diagnosis to the `diagnosis.md` path the supervisor gives you (write-only, immutable, one per cycle -- the durable log). Then emit a full rewritten curated memo to the `learning.draft.md` path: the supervisor validates it (non-empty, within the enforced length budget) and atomically swaps it over `experiments/learning.md`; an over-budget draft is rejected and reported. Rewrite and condense; do not append. Record only transferable, generic knowledge: what each mechanism does, why it fired or stayed inert, and durable frontier root-causes from step 3. Do not record per-cycle solve-count deltas or cross-experiment "variance"/"mean-reversion" narration -- each iteration is a different harness, so cross-run rate comparisons are confounded and are not valid signal. An "exhausted"/"closed" claim must cite the evidence behind it and stays revisable, not a permanent rule.
5. Stop when `diagnosis.md` and a valid `learning.draft.md` are written.

Unexpected blockers should fail normally; the supervisor handles runtime errors.
