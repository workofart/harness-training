"""Deterministic end-to-end auto cycle (plan.md §15 Step 5; Step-0 golden).

Drives ONE full ``scan -> decide -> execute`` cycle of the supervisor loop on the
golden's ``stub_config``, with two seams replaced:

- the agent backend is **scripted** -- the prelaunch turn makes a fixed
  behavioral edit to ``core.py`` + writes a focus label; the diagnosis turn writes
  ``diagnosis.md`` + a ``learning.draft.md``;
- the ``uv run exp`` subprocess is replaced by an **in-process call to the REAL
  orchestrator** (``run_tasks``) over a scripted pass/fail ``trial_runner``.

Everything below the subprocess boundary is real: ``scan`` + the §12 hard-fails,
``decide`` (§6 truth table), the gate (§9), ``budget_from_baseline`` (the
deterministic-baseline single-trial shortcut), the candidate worktree/ref dance,
and the orchestrator's scheduling (LPT / majority early-stop / confirm-on-fail).
So the cycle reproduces, for each golden scenario, the ``decide()`` command
sequence, the per-task solved-sets + trial-counts, and the keep/discard decision
-- the semantic equivalence Step 0 pinned (grounded in test_orchestrator.py).

This is NOT the real 2-task smoke (config/smoke_config.json, docker + a live
model) -- that is plumbing-only, asserts no outcome parity, and is run manually.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import repo
from src.experiment.orchestrator import run_tasks
from src.experiment.record import ExperimentResult, TrialResult
from src.supervisor import loop as loop_mod, workspace
from src.supervisor.agent_backend import TurnResult
from src.supervisor.loop import (
    LoopContext,
    execute,
    learning_memo_path,
    read_learning_memo,
    scan,
)
from src.supervisor.policy import Diagnose, Halt, decide

_GOLDEN = json.loads(
    (
        Path(__file__).resolve().parents[1] / "goldens" / "auto_cycle_golden.json"
    ).read_text()
)
_STUB = _GOLDEN["stub_config"]
_SCENARIOS = {scenario["name"]: scenario for scenario in _GOLDEN["scenarios"]}

_BASELINE_ID = "baseline-e2e"
_CANDIDATE_ID = "cand-e2e"


def _stub_config() -> SimpleNamespace:
    # The golden's stub_config (3 train + 2 test synthetic tasks, sized so the
    # promotion Fisher is exercisable). Concurrency is pinned to 1: the golden's
    # trial counts rest on *staggered* completions (the majority watcher cancels
    # the surplus trial before it runs), which a non-blocking in-process
    # trial_runner only reproduces one trial at a time. Concurrent scheduling is
    # test_orchestrator.py's job; here we validate decisions/solved-sets/counts.
    return SimpleNamespace(
        train_tasks=frozenset(_STUB["train_tasks"]),
        test_tasks=frozenset(_STUB["test_tasks"]),
        task_trials=_STUB["task_trials"],
        max_trial_concurrency=1,
        max_heavy_action_concurrency=1,
    )


class _CycleBackend:
    """Scripts both agent turns. The prelaunch turn (run in the sparse edit
    worktree) makes a fixed behavioral ``core.py`` edit + writes the focus label;
    the diagnosis turn writes the raw log + the curated memo draft. Dispatch is on
    the prompt header, the same way a real agent reads its phase from program.md."""

    def __init__(self, experiments_dir: Path) -> None:
        self.experiments_dir = experiments_dir

    def run_turn(self, *, prompt, repo_root, thread_id=None, timeout_sec=3600.0):
        worktree = Path(repo_root)
        if "prelaunch" in prompt:
            (worktree / "src" / "harness" / "core.py").write_text(
                "VALUE = 99  # candidate mechanism\n"
            )
            (worktree / workspace.FOCUS_FILE).write_text("e2e-mechanism\n")
        else:  # post-run diagnosis
            (
                self.experiments_dir / _CANDIDATE_ID / loop_mod.DIAGNOSIS_FILENAME
            ).write_text("raw per-cycle diagnosis\n")
            (self.experiments_dir / loop_mod.LEARNING_DRAFT_FILENAME).write_text(
                "curated memo rewrite\n"
            )
        return TurnResult(thread_id="thread-e2e")


def _make_run_exp(scenario, config):
    """An in-process ``run_exp`` that calls the REAL orchestrator over a scripted
    trial_runner. The phase (baseline / candidate train / candidate veto) is read
    from the task set -- baseline runs train|test, the candidate runs train then
    test as separate calls -- so the right golden pass/fail map keys the outcomes.
    The golden's per-(phase, task) sequences are homogeneous, so a single outcome
    per task is the task's scripted verdict regardless of trial scheduling order."""
    train, test = config.train_tasks, config.test_tasks
    configured = train | test
    pass_fail = scenario["given"]["stub_pass_fail"]

    def run_exp(*, worktree, experiment_id, task_ids, experiments_dir, trial_budget):
        task_ids = frozenset(task_ids)
        if task_ids == configured:
            phase = "baseline"
        elif task_ids == train:
            phase = "candidate_train"
        elif task_ids == test:
            phase = "candidate_test"
        else:
            raise AssertionError(f"unexpected task set: {sorted(task_ids)}")
        outcomes = pass_fail[phase]

        async def trial_runner(task_id, run_id, heavy_action_semaphore, slot_release):
            # Yield before returning so a trial that completes a majority lets the
            # watcher cancel the surplus expansion trial (still parked) before it
            # appends -- reproducing the golden's "staggered completions" basis with
            # max_trial_concurrency=1 instead of a manually-driven blocking stub.
            await asyncio.sleep(0)
            solved = outcomes[task_id][0] == "pass"  # homogeneous per task
            return TrialResult(
                run_id=run_id,
                solved=solved,
                failure_mode="solved" if solved else "verified_rejected",
                verifier_passed=solved,
            )

        asyncio.run(
            run_tasks(
                experiment_id=experiment_id,
                git_commit_hash=repo.get_head_commit(cwd=worktree),
                task_ids=sorted(task_ids),
                budget=dict(trial_budget),
                full_trial_count=config.task_trials,
                max_trial_concurrency=config.max_trial_concurrency,
                max_heavy_action_concurrency=config.max_heavy_action_concurrency,
                trial_runner=trial_runner,
                experiments_root=experiments_dir,
                started_at="2026-01-01T00:00:00+00:00",
            )
        )

    return run_exp


def _drive_one_cycle(ctx: LoopContext) -> list[str]:
    """``scan -> decide -> execute`` until the cycle's terminal ``Diagnose`` (or a
    ``Halt``); returns the command class names in order. run_auto itself never
    stops on its own (it would propose a second candidate), so the cycle boundary
    is the first ``Diagnose``."""
    commands: list[str] = []
    for _ in range(20):
        world = scan(
            experiments_dir=ctx.experiments_dir,
            repo_root=ctx.repo_root,
            config=ctx.config,
        )
        command = decide(world)
        commands.append(type(command).__name__)
        if isinstance(command, Halt):
            break
        execute(command, world, ctx)
        if isinstance(command, Diagnose):
            break
    return commands


def _assert_task_counts(task_result, spec: dict) -> None:
    assert len(task_result.valid_trials) == spec["trial_count"]
    assert task_result.solved_count == spec["solved_count"]
    assert task_result.majority_solved is spec["majority_solved"]


@pytest.mark.parametrize("scenario_name", list(_SCENARIOS))
def test_auto_cycle_matches_the_golden(
    scenario_name: str,
    repo_root: Path,
    experiments_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _SCENARIOS[scenario_name]
    expect = scenario["expect"]
    config = _stub_config()
    # The cheap pre-experiment test_core gate would shell out to `uv run pytest`
    # in the sparse view; the scripted edit keeps it green, so stub it.
    monkeypatch.setattr(
        workspace, "run_test_core", lambda _wt: workspace.TestCoreResult(True, "ok")
    )
    experiment_ids = iter([_BASELINE_ID, _CANDIDATE_ID])
    ctx = LoopContext(
        repo_root=repo_root,
        experiments_dir=experiments_dir,
        worktree_root=repo_root.parent / "worktrees",
        config=config,
        backend=_CycleBackend(experiments_dir),
        program_md_path=repo_root / "program.md",
        run_exp=_make_run_exp(scenario, config),
        new_experiment_id=lambda _dir: next(experiment_ids),
    )

    head_before = repo.get_head_commit(cwd=repo_root)
    commands = _drive_one_cycle(ctx)

    # 1. the decide() command sequence (§6).
    assert commands == expect["command_sequence"]

    # 2. baseline solved-sets + trial-counts (orchestrator §8), when pinned.
    baseline = ExperimentResult.load(_BASELINE_ID, root=experiments_dir)
    for task_id, spec in expect.get("baseline", {}).items():
        _assert_task_counts(baseline.tasks[task_id], spec)

    # 3. candidate solved-sets + trial-counts, incl. the deterministic-baseline
    #    single-trial budget + confirm-on-fail expansion crossing the seam.
    candidate = ExperimentResult.load(_CANDIDATE_ID, root=experiments_dir)
    # The veto scenario's golden details only candidate_test (train counts match
    # the keep scenario); RunVeto in its command sequence already pins train kept.
    for task_id, spec in expect.get("candidate_train", {}).items():
        _assert_task_counts(candidate.tasks[task_id], spec)
    candidate_test = expect["candidate_test"]
    if isinstance(candidate_test, dict):
        for task_id, spec in candidate_test.items():
            _assert_task_counts(candidate.tasks[task_id], spec)
    else:
        # discard-at-train: the veto never ran, so no test task is recorded.
        assert config.test_tasks.isdisjoint(candidate.tasks)

    # 4. the keep/discard decision (gate §9) + its HEAD/ref side effects (§7).
    concluded = loop_mod.load_loop(_CANDIDATE_ID, root=experiments_dir)
    assert concluded.decision is not None
    assert concluded.decision.kind == expect["decision"]

    head_after = repo.get_head_commit(cwd=repo_root)
    assert (head_after != head_before) is expect["head_advances"]
    assert repo.git_ref_exists(
        cwd=repo_root, ref=workspace.failed_ref(_CANDIDATE_ID)
    ) is expect.get("failed_ref_written", False)

    # 5. the diagnosis memo was swapped in (§10).
    assert (experiments_dir / _CANDIDATE_ID / loop_mod.DIAGNOSIS_FILENAME).exists()
    assert read_learning_memo(experiments_dir) == "curated memo rewrite\n"
    assert not (experiments_dir / loop_mod.LEARNING_DRAFT_FILENAME).exists()
    assert learning_memo_path(experiments_dir).exists()
