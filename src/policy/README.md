# policy

*Read this to understand the harness being trained — and what a candidate is
allowed to touch.*

The harness under training — the parameter in the PyTorch analogy.

## Two files, two sets of rules

- `base.py` — frozen. The contract between the rollout loop
  (`src/rollout/episode.py`) and any policy: the `Policy` protocol (`reset` /
  `act` / `observe`), the failure signals (`NoValidActionError`,
  `RepeatedLengthCutoffError`), and the fixed set of event names a policy may
  emit — telemetry counts them, and a policy can't invent new ones. The loop
  checks the protocol at runtime and imports nothing from `core.py`'s
  internals.
- `core.py` — editable. The config's `training_target` block names it as the
  trained module (`module: "src.policy.core"`), and the rollout loop resolves
  it from there: the module must export `build_policy` and `build_env_action`.
  Every candidate must modify this file, and may touch nothing else except the
  config's `extra_patch_paths` (here `tests/policy/test_core_impl.py`). That's
  the patch surface, enforced by `validate_candidate` in
  `src/trainer/parameter.py`, not by convention.

## Inside `core.py`

A caveat before the tour: `core.py` is the parameter being trained, so this
section describes the current substrate, not an invariant. Promotions rewrite
the file — tools, prompts, and the repair ladder can all change over a
training run — and when this list and the code disagree, the code wins. These
READMEs are for humans only: the proposer never sees them (its worktree
excludes `src/**/README.md`; `program.md` is its rulebook).

One file on purpose: the whole search space fits in a single diff. Five
sections, in reading order:

1. Prompt surface — the system, initial, and repair prompts.
2. Action space — two tools: `run` renders down to a shell command; `submit` is
   terminal and never reaches the env.
3. `LlmAgent`, the step policy, with its repair ladder: reasoning runaway
   (the model burns its whole output budget thinking without emitting a tool
   call) → disable thinking and steer; an oversized tool call cut off
   mid-stream → one steered retry; repeated length cutoffs → abort the
   rollout. `REMINDER_RULES` and `ACTION_GUARDS` are empty tuples by default —
   seams where a candidate can add a step-triggered nudge or rewrite a parsed
   action without touching the loop. In the checked-in substrate they stay empty;
   during training, a promoted candidate fast-forwards HEAD, so its edits to
   this file become the new baseline (rejected candidates stay parked under
   `refs/candidates/`).
4. Request assembly and context-window fitting.
5. Completion parsing and repair.

## Rules of the game

`program.md` at the repo root is the rulebook the estimator reads before
editing this file. What it must not do — forge metrics, touch grading — is
enforced outside this directory (`src/rollout/telemetry.py`, candidate
validation), not requested politely in a prompt.
