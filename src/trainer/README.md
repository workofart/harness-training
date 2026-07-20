# trainer

*Read this to write or customize a training loop: Trainer, Estimator,
Criterion, Optimizer.*

The PyTorch analogy and the loop sketch are in the [root README](../../README.md).
This doc is the vocabulary, the four contracts, and the epoch order. The user
owns the loop; there is no `fit()`.

## Vocabulary

| term | meaning |
| --- | --- |
| baseline | the committed harness at HEAD, and its measured result |
| candidate | baseline + one proposed diff, captured as a child commit |
| patch surface | the files a candidate may change, declared by the config's `training_target`: the trained module's own file (required; derived from `training_target.module`) plus `extra_patch_paths` |
| proposer visibility | `training_target.proposer_visible` — sparse-checkout patterns for what the proposer can *read*; distinct from the patch surface, which is what it may *write* |
| learning memo | `experiments/learning.md` — the estimator's running diagnosis, rewritten after every verdict and read back before the next proposal |
| promotion | the optimizer fast-forwards HEAD to the candidate commit. Every candidate is parked under `refs/candidates/` at capture time, so rejected work stays reachable |

## The four contracts

| file | contract | shipped implementation |
| --- | --- | --- |
| `parameter.py` | `Parameter`: the handle over the harness repo; scratch worktrees; candidate capture and patch-surface validation | — |
| `estimator.py` | `Estimator.propose()` gets a worktree plus the resolved training target and leaves a change in it; `diagnose()` writes the learning memo. Nothing in the contract says "agent" — a scripted patch queue or a human diff satisfies it | `AgenticEstimator`: one `claude`/`codex` turn to propose, up to two to diagnose (a rejected memo draft is fed back once), steered by `program.md`; the target facts are injected into its prompts from config, never restated in the doc |
| `loss.py` | `Criterion` judges two runs into a `Loss`; negative selects the candidate. Raises `UnmeasurableRun` when infra noise could flip the verdict; `decision()` renders the persisted verdict. What the primary signal *is* belongs to the concrete criterion | `StrictPareto` on solved sets: any regression = `inf`; exact ties fall to secondary rewards (`src/rollout/metrics.py`) |
| `optim.py` | `Optimizer.step()` ff-only merges the selected commit into the harness | `GreedyMonotonic`: candidate iff `loss < 0` |

## One epoch, in order (`trainer.py`)

1. Reuse the baseline run for HEAD if one exists with the same measurement
   identity (`src/rollout/README.md`); measure one otherwise.
2. Certify it — replay configs re-execute each task's action chain and drop
   tasks that fork (`src/plugins/README.md`).
3. `propose()` in a sparse worktree (only estimator-visible files are checked
   out).
4. Capture the estimator's diff as a commit; validate it against the patch
   surface.
5. Measure the candidate in its own worktree (one retry on infra failure);
   judge it — the criterion runs once, inside the funnel, so an
   `UnmeasurableRun` verdict can trigger the retry.
6. Yield the judged `Loss` (carrying `candidate` and `baseline`) to your
   loop: backward → step.
7. Record the decision, run `diagnose()`, zero `harness.grad`.

An epoch *skips* — no verdict, nothing measured — when the proposal never
reaches measurement: `no_candidate` (the proposer changed nothing) or
`invalid_candidate` (the diff broke a guardrail: bad ancestry, a path outside
the patch surface, or a candidate that fails the repo's own test suite). The
cause is reported when it happens. Two consecutive skips raise
`CircuitBreakerTripped` rather than let a broken estimator spin.

## Candidate identity is trainer-owned

Custom `Estimator`s are a kept extension point — which is exactly why the
trainer trusts none of them. The estimator contract is "mutate this worktree",
nothing else. The trainer snapshots the epoch baseline SHA before the propose
turn, captures the complete baseline→workspace delta itself
(`capture_candidate`: soft-reset to base, one direct-child commit), and
validates ancestry (`validate_candidate`: sole parent == base). The identity
chain is asserted before anything is measured or persisted — candidate.base
== epoch baseline, measured SHA == validated candidate SHA — and violations
are `RuntimeError`s, not warnings. The pending grad is cleared in a `finally`
around post-decision diagnosis. Single trainer per repo is an assumption, not
enforced: no locks, no per-session worktree paths.

Worktrees live in a sibling directory (`<repo>-worktrees/`). Only committed
code is measured — dirty files are warned about and ignored — and the config
must be git-tracked because it is part of the measurement identity.
