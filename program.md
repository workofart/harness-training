# program.md

Operating policy for the harness training loop. Two fresh agents read this file each cycle: the **proposer** owns steps 1-2, the **diagnoser** owns step 4, and the outer-loop driver runs step 3 between them. The driver's prompt says which steps are yours and names the editable files — this file never names them; each step below is written as instructions to its owner, and the other role's steps are context. This is policy, read the in-scope files yourself, and if this file and the code disagree, the code wins.

## Objective

Improve the harness — the editable policy surface your prompt names — so the agent solves more tasks in a given environment over time. Solving more tasks is the goal. When two candidates solve the same set, the driver breaks the tie on secondary signals — progress toward a solve, then a valid tool call on the first try, then fewer steps — so a correct change that does not yet flip a task can still promote. These are tie-breakers, not targets. Make ONE bounded, measurable change per cycle — one mechanism, never several bundled (a bundled win cannot be attributed). A flat result is information, not failure. Make sure to confirm the change's trigger actually fired.

## What the proposer may edit

Exactly the files your prompt names: the editable policy surface (a candidate MUST change it) and its companion test file (cover your change). All of the surface is tunable. The measurement-integrity contract is **frozen** and lives outside it. The driver rejects any patch touching any other file, so the boundary is structural, not just a rule. The measurement protocol (task panel, model, step caps, timeouts, rollout count) is committed config: read it, but changing it is a new baseline, the user's job.

## Reward-safety (non-negotiable)

The verifier/tests are final grading. A candidate must NOT:

- run the final verifier mid-solve, or more than once;
- show verifier output / reward / diagnostics to the solving agent before the score;
- make failed final verification retryable;
- depend on hidden tests, verifier internals, task ids, gold patches, or benchmark-specific constants.

Stay task-agnostic: generic state/action/policy only. You may *read* task-specific evidence to diagnose; the change it motivates may not *encode* it.

## One cycle (proposer: steps 1-2; driver: step 3; diagnoser: step 4)
1. **Propose.** Read `experiments/learning.md` and the last run's `experiment.json` plus the traces it points to. If `runs.jsonl` records runs newer than the memo covers, the memo is stale — diagnose those runs from their artifacts yourself (per "Diagnostic principles" below) before trusting its leads. The last run may be a rejected candidate: weigh it against the baseline and recent runs, not in isolation. Then pick ONE bounded change and be able to say which tasks it could convert and why. Use the memo's mechanism ledger to steer the search: prefer an untried mechanism or part of the surface over another variation on a spent one, though revisiting or recombining a flat mechanism is fair game. Before re-targeting anything a recent cycle already targeted, attribute that cycle's failure first (per "Diagnostic principles"); never spend a cycle on a variant of an unattributed rejection — record the finding and pick a different lever.

2. **Patch.** Edit the named files; run `uv run python -m pytest` (exactly this form — plain `pytest` imports the wrong checkout in a worktree) and keep the whole suite green before stopping. Your checkout holds every test your change can break; the suite was green before your patch, so any failure is your patch's doing, and since only the named files are editable the fix always goes through them. The driver re-runs the same suite from a trusted checkout and rejects a red candidate. Treat the frozen tests, especially `test_core_contracts.py`, as the machine-checked spec of what your change must preserve. Don't run git — the driver commits once the patch clears the guardrails. Comments state only lasting constraints the code cannot show — never per-cycle evidence (panel statistics, solve-safety proofs, target tasks), which fossilizes and misleads later cycles. Pin no-op guarantees as tests; put panel evidence in your proposal summary for the memo.

3. **Driver.** Commits the candidate, runs the repo test suite from a trusted checkout (a red suite rejects the candidate before measurement, with the failing output in the rejection), runs the deterministic experiment against the baseline, and promotes if it solves more tasks with no regression, else if it wins the secondary tie-break; otherwise it rolls back (the rollback lands after Step 4's turn concludes).

4. **Diagnose.** The diagnoser — a *separate* agent from the proposer, in a fresh turn after the run concludes — rewrites `experiments/learning.md`: the deliverable is specced in "The learning memo" below, the method in "Diagnostic principles". Condense, never append, within the driver's line budget. During this turn the candidate's commit is still checked out (`git show HEAD` is its patch); change nothing outside the memo draft.

## The learning memo (the diagnoser's deliverable)

The memo carries three things, each cited to artifacts and kept revisable:

- the **current bottleneck** — what stays unsolved across runs, with the causally-attributed reason per blocker;
- the **mechanism ledger** — one entry per failure mechanism (not per cycle): interventions tried, what each changed and aimed to fix, its attributed result, and status (open / palliated / spent / load-bearing);
- the **research leads** — the next levers, including recombinations, each with the signal that would confirm it.

Three further duties:

- **Friction, not coaching** — for agent-side failures, look for distribution mismatch: places where the model's habitual output collides with what the harness expects (schema strictness, output budget, feedback the model ignores and re-emits against). Quantify each collision in steps lost; prefer removing the collision over coaching around it.
- **Account for the frontier** — when a rejected run's artifacts prove something the baseline does not (a task demonstrably convertible, a gain the gate discarded), that is a first-class bottleneck entry with its number attached.
- **Escalate measurement faults** — when artifacts show the measurement itself decided an outcome (infra event scored as regression, cache drift, latency flipping a task), the finding is the measurement fault, written for the user; a policy workaround is the wrong move, and the fault stands until the user changes the protocol.

## Diagnostic principles (govern Step 4 and any run-reading in Step 1)

- **Artifacts, not labels.** Diagnose from what the agent actually produced, never from failure-mode labels or counts. Many failures emit no error event — read what the task was doing when it ended. A task is substrate-bound only when its diff shows the agent never understood the fix; "unsolved across runs" is not "unsolvable."
- **Attribute causally before concluding.** For any regression, unconverted task, or solved-to-unsolved flip: diff the candidate's trajectory against the baseline as the agent actually saw it — each trace row's `request_messages` is exactly that rendering — find the first diverging step, and confirm the change's own mechanism fired there. If the mechanism never executed on that task, or the divergence is on an unrelated path, it's an incidental fork (or infra) — record it as "this diff perturbed task X," not as a verdict on the mechanism. A rejection not caused by the mechanism is evidence about the measurement, not the mechanism.
- **No mechanism is closed by a flat result.** Record flat/negative outcomes as "tried this shape, got this result." The same mechanism may still convert tasks when recombined with another change — recombination toward a global optimum is where later wins often come from.

## Evidence

Read each run's `experiment.json` — and one example task end-to-end — before re-deriving anything from traces. `crash` is infra noise — exclude it; everything else (incl. `no_valid_action`) is a scorable unsolved attempt. `verify_timeout` is unsolved for scoring, but because the grade itself never completed the gate invalidates the run (`invalid_infra`) instead when the verdict would hinge on it.

Determinism: the model is a deterministic function of its *rendered* input, but the environment is not — a volatile token (timing, seed, address) can slip past the scrubber and fork a trajectory. Before grounding a proposal in a solved-to-unsolved flip, attribute it causally first (per "Diagnostic principles"); a flip whose first diverging step the change's mechanism never touched is not the change's work.
