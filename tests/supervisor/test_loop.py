"""Tests for src/supervisor/loop.py -- the read + write boundaries (plan.md §6/§12).

``scan()`` derives ``decide()``'s ``World`` from the experiment dirs + git (the read
boundary + the §12 corruption hard-fails); ``execute()`` performs the one command
``decide()`` returns (the write boundary -- the candidate worktree/ref lifecycle, the
``uv run exp`` seam, the atomic-swap memo); ``run_auto`` drives them to ``Halt``. The
read tests seed real ``loop.json`` + ``experiment.json`` files; the write tests run a
real temp git repo with the ``uv run exp`` seam + agent backend stubbed, so the git
side effects (commit/ref/fast-forward) are exercised for real. The pure ``decide()``
truth table is covered in test_policy.py.
"""

from __future__ import annotations

import subprocess
from collections.abc import Collection
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.contracts import FailureMode
from src.experiment.record import ExperimentResult, TaskResult, TrialResult
from src.experiment.writer import write_experiment_result
from src.supervisor import loop as loop_mod, workspace
from src.supervisor.agent_backend import TurnResult
from src.supervisor.loop import (
    LoopContext,
    LoopCorruption,
    build_diagnosis_prompt,
    execute,
    learning_draft_path,
    learning_memo_path,
    read_learning_memo,
    run_auto,
    scan,
    swap_learning_draft,
)
from src.supervisor.policy import (
    Conclude,
    Decision,
    Diagnose,
    Halt,
    LoopResult,
    PendingRun,
    ProposeAndLaunch,
    RefreshBaseline,
    RunVeto,
    World,
)

_KEEP = Decision(kind="keep", reason="kept", verdicts={})
_DISCARD = Decision(kind="discard", reason="discarded", verdicts={})


# repo_root + experiments_dir fixtures are shared via tests/supervisor/conftest.py.


def _config(train, test, *, task_trials: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        train_tasks=frozenset(train),
        test_tasks=frozenset(test),
        task_trials=task_trials,
    )


def _seed(
    experiments_dir: Path,
    experiment_id: str,
    *,
    kind: str,
    decision: Decision | None,
    parent: str | None = None,
    run_status: str | None = None,
    tasks: tuple[str, ...] = (),
    finished_at: str = "2026-01-01T00:00:00",
    diagnosis: bool = False,
    commit: str = "c0",
) -> None:
    """Write one experiment dir: loop.json always; experiment.json iff
    ``run_status`` is given; an empty diagnosis.md iff ``diagnosis``."""
    loop_mod.write_loop(
        LoopResult(
            experiment_id=experiment_id,
            kind=kind,
            focus_name="f",
            parent_baseline_experiment_id=parent,
            decision=decision,
        ),
        root=experiments_dir,
    )
    if run_status is not None:
        write_experiment_result(
            ExperimentResult(
                experiment_id=experiment_id,
                git_commit_hash=commit,
                run_status=run_status,
                started_at="2026-01-01T00:00:00",
                finished_at=None if run_status == "running" else finished_at,
                tasks={t: TaskResult.empty(expected_trial_count=1) for t in tasks},
            ),
            root=experiments_dir,
        )
    if diagnosis:
        (experiments_dir / experiment_id / loop_mod.DIAGNOSIS_FILENAME).write_text(
            "d\n"
        )


# --- happy derivations ------------------------------------------------------


def test_scan_empty_experiments_is_a_blank_world(
    repo_root: Path, experiments_dir: Path
) -> None:
    world = scan(
        experiments_dir=experiments_dir,
        repo_root=repo_root,
        config=_config({"a"}, {"b"}),
    )
    assert world.active_baseline is None
    assert world.pending is None
    assert world.undiagnosed_candidate_id is None
    assert not world.primary_dirty
    assert world.head_commit  # the real HEAD sha
    assert world.train_tasks == frozenset({"a"})
    assert world.test_tasks == frozenset({"b"})


def test_scan_picks_the_newest_kept_baseline(
    repo_root: Path, experiments_dir: Path
) -> None:
    cfg = _config({"a"}, {"b"})
    _seed(
        experiments_dir,
        "exp-old",
        kind="baseline",
        decision=_KEEP,
        run_status="completed",
        tasks=("a", "b"),
        finished_at="2026-01-01T00:00:00",
    )
    _seed(
        experiments_dir,
        "exp-new",
        kind="baseline",
        decision=_KEEP,
        run_status="completed",
        tasks=("a", "b"),
        finished_at="2026-01-02T00:00:00",
    )
    world = scan(experiments_dir=experiments_dir, repo_root=repo_root, config=cfg)
    assert world.active_baseline is not None
    assert world.active_baseline.experiment_id == "exp-new"  # newest finished_at


def test_scan_ignores_discarded_and_pending_for_the_baseline(
    repo_root: Path, experiments_dir: Path
) -> None:
    cfg = _config({"a"}, {"b"})
    _seed(
        experiments_dir,
        "exp-keep",
        kind="baseline",
        decision=_KEEP,
        run_status="completed",
        tasks=("a", "b"),
    )
    _seed(
        experiments_dir,
        "exp-discard",
        kind="candidate",
        decision=_DISCARD,
        parent="exp-keep",
        run_status="completed",
        tasks=("a",),
        diagnosis=True,
    )
    world = scan(experiments_dir=experiments_dir, repo_root=repo_root, config=cfg)
    assert world.active_baseline is not None
    assert world.active_baseline.experiment_id == "exp-keep"


def test_scan_surfaces_a_pending_candidate_with_its_result(
    repo_root: Path, experiments_dir: Path
) -> None:
    cfg = _config({"a"}, {"b"})
    _seed(
        experiments_dir,
        "exp-base",
        kind="baseline",
        decision=_KEEP,
        run_status="completed",
        tasks=("a", "b"),
    )
    # Pending: decision is None; recorded train only (awaiting the veto run).
    _seed(
        experiments_dir,
        "exp-cand",
        kind="candidate",
        decision=None,
        parent="exp-base",
        run_status="completed",
        tasks=("a",),
    )
    world = scan(experiments_dir=experiments_dir, repo_root=repo_root, config=cfg)
    assert world.pending is not None
    assert world.pending.loop.experiment_id == "exp-cand"
    assert world.pending.result is not None
    assert world.pending.result.run_status == "completed"


def test_scan_filters_a_launch_incomplete_pending(
    repo_root: Path, experiments_dir: Path
) -> None:
    # loop.json prewritten, experiment.json never produced (the launch died before
    # recording). A dead pending (result is None) is filtered out of World.pending
    # (§11) so a manual rerun proceeds instead of halting; the dir stays on disk.
    _seed(experiments_dir, "exp-cand", kind="candidate", decision=None, parent=None)
    world = scan(
        experiments_dir=experiments_dir,
        repo_root=repo_root,
        config=_config({"a"}, {"b"}),
    )
    assert world.pending is None


def test_scan_filters_a_crashed_pending(repo_root: Path, experiments_dir: Path) -> None:
    # A run that crash-finalized (run_status "crashed") is a dead pending: filtered
    # so a leftover crash never wedges a manual `uv run auto` (§11).
    _seed(
        experiments_dir,
        "exp-cand",
        kind="candidate",
        decision=None,
        parent=None,
        run_status="crashed",
    )
    world = scan(
        experiments_dir=experiments_dir,
        repo_root=repo_root,
        config=_config({"a"}, {"b"}),
    )
    assert world.pending is None


def test_scan_filters_a_dead_running_pending(
    repo_root: Path, experiments_dir: Path
) -> None:
    # A process killed mid-run leaves run_status stuck at "running" (the crash
    # handler never ran). On a fresh invocation the prior process is gone, so it is
    # a dead pending -> filtered (§11).
    _seed(
        experiments_dir,
        "exp-cand",
        kind="candidate",
        decision=None,
        parent=None,
        run_status="running",
    )
    world = scan(
        experiments_dir=experiments_dir,
        repo_root=repo_root,
        config=_config({"a"}, {"b"}),
    )
    assert world.pending is None


def test_scan_filters_a_crashed_candidate_with_a_wrong_parent(
    repo_root: Path, experiments_dir: Path
) -> None:
    # The pending-contract asserts (§12) guard a run we ACT on; a dead pending is
    # only ignored, so its now-moot parent lineage is never checked -- it is
    # filtered, not a corruption fault (contrast the completed-pending case below).
    cfg = _config({"a"}, {"b"})
    _seed(
        experiments_dir,
        "exp-base",
        kind="baseline",
        decision=_KEEP,
        run_status="completed",
        tasks=("a", "b"),
    )
    _seed(
        experiments_dir,
        "exp-cand",
        kind="candidate",
        decision=None,
        parent="exp-stale",  # would raise LoopCorruption if this run were live
        run_status="crashed",
        tasks=("a",),
    )
    world = scan(experiments_dir=experiments_dir, repo_root=repo_root, config=cfg)
    assert world.pending is None


def test_scan_keeps_a_live_pending_beside_a_dead_one(
    repo_root: Path, experiments_dir: Path
) -> None:
    # A leftover crashed pending must neither shadow a live completed one nor trip
    # the >1-pending fault: the live pending is routed, the dead corpse ignored.
    cfg = _config({"a"}, {"b"})
    _seed(
        experiments_dir,
        "exp-base",
        kind="baseline",
        decision=_KEEP,
        run_status="completed",
        tasks=("a", "b"),
    )
    _seed(
        experiments_dir,
        "exp-dead",
        kind="candidate",
        decision=None,
        parent="exp-base",
        run_status="crashed",
        tasks=("a",),
    )
    _seed(
        experiments_dir,
        "exp-live",
        kind="candidate",
        decision=None,
        parent="exp-base",
        run_status="completed",
        tasks=("a",),
    )
    world = scan(experiments_dir=experiments_dir, repo_root=repo_root, config=cfg)
    assert world.pending is not None
    assert world.pending.loop.experiment_id == "exp-live"


def test_scan_flags_an_undiagnosed_decided_candidate(
    repo_root: Path, experiments_dir: Path
) -> None:
    cfg = _config({"a"}, {"b"})
    _seed(
        experiments_dir,
        "exp-base",
        kind="baseline",
        decision=_KEEP,
        run_status="completed",
        tasks=("a", "b"),
    )
    _seed(
        experiments_dir,
        "exp-cand",
        kind="candidate",
        decision=_DISCARD,
        parent="exp-base",
        run_status="completed",
        tasks=("a",),
        diagnosis=False,
    )
    world = scan(experiments_dir=experiments_dir, repo_root=repo_root, config=cfg)
    assert world.undiagnosed_candidate_id == "exp-cand"


def test_scan_does_not_flag_a_diagnosed_candidate_or_a_baseline(
    repo_root: Path, experiments_dir: Path
) -> None:
    cfg = _config({"a"}, {"b"})
    # A kept baseline (decision set, no diagnosis) is never "undiagnosed":
    # only candidates get a diagnosis memo.
    _seed(
        experiments_dir,
        "exp-base",
        kind="baseline",
        decision=_KEEP,
        run_status="completed",
        tasks=("a", "b"),
    )
    _seed(
        experiments_dir,
        "exp-cand",
        kind="candidate",
        decision=_DISCARD,
        parent="exp-base",
        run_status="completed",
        tasks=("a",),
        diagnosis=True,
    )
    world = scan(experiments_dir=experiments_dir, repo_root=repo_root, config=cfg)
    assert world.undiagnosed_candidate_id is None


def test_scan_reports_a_dirty_primary(repo_root: Path, experiments_dir: Path) -> None:
    (repo_root / "scratch.txt").write_text("uncommitted\n")
    world = scan(
        experiments_dir=experiments_dir,
        repo_root=repo_root,
        config=_config({"a"}, {"b"}),
    )
    assert world.primary_dirty


def test_scan_ignores_dirs_without_a_loop_json(
    repo_root: Path, experiments_dir: Path
) -> None:
    # A plain `uv run exp` one-off writes experiment.json but no loop.json -- it
    # is not an auto cycle and must not be mistaken for a baseline/candidate.
    write_experiment_result(
        ExperimentResult(
            experiment_id="exp-oneoff",
            git_commit_hash="c0",
            run_status="completed",
            started_at="2026-01-01T00:00:00",
            finished_at="2026-01-01T01:00:00",
            tasks={"a": TaskResult.empty(expected_trial_count=1)},
        ),
        root=experiments_dir,
    )
    world = scan(
        experiments_dir=experiments_dir,
        repo_root=repo_root,
        config=_config({"a"}, {"b"}),
    )
    assert world.active_baseline is None
    assert world.pending is None


# --- §12 corruption hard-fails ----------------------------------------------


def test_scan_rejects_an_empty_test_panel(
    repo_root: Path, experiments_dir: Path
) -> None:
    # The feedback-3 obligation: an empty test panel must hard-fail *before*
    # World is built, else decide()'s `test_tasks <= tasks` is vacuously true
    # and the regression veto silently never runs.
    with pytest.raises(LoopCorruption, match="non-empty"):
        scan(
            experiments_dir=experiments_dir,
            repo_root=repo_root,
            config=_config({"a"}, set()),
        )


def test_scan_rejects_an_empty_train_panel(
    repo_root: Path, experiments_dir: Path
) -> None:
    with pytest.raises(LoopCorruption, match="non-empty"):
        scan(
            experiments_dir=experiments_dir,
            repo_root=repo_root,
            config=_config(set(), {"b"}),
        )


def test_scan_rejects_overlapping_panels(
    repo_root: Path, experiments_dir: Path
) -> None:
    with pytest.raises(LoopCorruption, match="overlap"):
        scan(
            experiments_dir=experiments_dir,
            repo_root=repo_root,
            config=_config({"a", "x"}, {"b", "x"}),
        )


def test_scan_rejects_more_than_one_live_pending(
    repo_root: Path, experiments_dir: Path
) -> None:
    # Two *completed* pendings the sequential loop never concluded is corruption.
    _seed(
        experiments_dir,
        "exp-1",
        kind="candidate",
        decision=None,
        parent=None,
        run_status="completed",
        tasks=("a",),
    )
    _seed(
        experiments_dir,
        "exp-2",
        kind="candidate",
        decision=None,
        parent=None,
        run_status="completed",
        tasks=("a",),
    )
    with pytest.raises(LoopCorruption, match="more than one live pending"):
        scan(
            experiments_dir=experiments_dir,
            repo_root=repo_root,
            config=_config({"a"}, {"b"}),
        )


def test_scan_tolerates_multiple_dead_pendings(
    repo_root: Path, experiments_dir: Path
) -> None:
    # Stacked corpses (crashed twice without a rerun between) are all filtered, not
    # a >1-pending fault -- a manual rerun still proceeds (§11).
    _seed(
        experiments_dir,
        "exp-1",
        kind="candidate",
        decision=None,
        parent=None,
        run_status="crashed",
    )
    _seed(
        experiments_dir,
        "exp-2",
        kind="candidate",
        decision=None,
        parent=None,
        run_status="crashed",
    )
    world = scan(
        experiments_dir=experiments_dir,
        repo_root=repo_root,
        config=_config({"a"}, {"b"}),
    )
    assert world.pending is None


def test_scan_rejects_a_pending_candidate_with_the_wrong_parent(
    repo_root: Path, experiments_dir: Path
) -> None:
    cfg = _config({"a"}, {"b"})
    _seed(
        experiments_dir,
        "exp-base",
        kind="baseline",
        decision=_KEEP,
        run_status="completed",
        tasks=("a", "b"),
    )
    # Parent points at a stale baseline, not the active one (exp-base).
    _seed(
        experiments_dir,
        "exp-cand",
        kind="candidate",
        decision=None,
        parent="exp-stale",
        run_status="completed",
        tasks=("a",),
    )
    with pytest.raises(LoopCorruption, match="parent"):
        scan(experiments_dir=experiments_dir, repo_root=repo_root, config=cfg)


def test_scan_rejects_a_pending_candidate_recording_a_foreign_task_set(
    repo_root: Path, experiments_dir: Path
) -> None:
    cfg = _config({"a"}, {"b"})
    _seed(
        experiments_dir,
        "exp-base",
        kind="baseline",
        decision=_KEEP,
        run_status="completed",
        tasks=("a", "b"),
    )
    # Completed candidate recorded neither train ({a}) nor configured ({a,b}).
    _seed(
        experiments_dir,
        "exp-cand",
        kind="candidate",
        decision=None,
        parent="exp-base",
        run_status="completed",
        tasks=("a", "z"),
    )
    with pytest.raises(LoopCorruption, match="recorded"):
        scan(experiments_dir=experiments_dir, repo_root=repo_root, config=cfg)


def test_scan_accepts_a_pending_candidate_recording_the_full_panel(
    repo_root: Path, experiments_dir: Path
) -> None:
    cfg = _config({"a"}, {"b"})
    _seed(
        experiments_dir,
        "exp-base",
        kind="baseline",
        decision=_KEEP,
        run_status="completed",
        tasks=("a", "b"),
    )
    # train|test is valid (both panels ran before the loop concluded).
    _seed(
        experiments_dir,
        "exp-cand",
        kind="candidate",
        decision=None,
        parent="exp-base",
        run_status="completed",
        tasks=("a", "b"),
    )
    world = scan(experiments_dir=experiments_dir, repo_root=repo_root, config=cfg)
    assert world.pending is not None


def test_scan_rejects_a_pending_baseline_recording_a_partial_panel(
    repo_root: Path, experiments_dir: Path
) -> None:
    # A pending (decision None) baseline that completed must have run all
    # configured tasks, not a subset.
    _seed(
        experiments_dir,
        "exp-base",
        kind="baseline",
        decision=None,
        run_status="completed",
        tasks=("a",),
    )
    with pytest.raises(LoopCorruption, match="all configured"):
        scan(
            experiments_dir=experiments_dir,
            repo_root=repo_root,
            config=_config({"a"}, {"b"}),
        )


def test_scan_rejects_an_active_baseline_at_head_missing_a_configured_task(
    repo_root: Path, experiments_dir: Path
) -> None:
    # §12 (plan.md:695,698): the active baseline gate() compares against must have
    # run all configured tasks. A kept baseline AT HEAD (current protocol) missing
    # the test panel is corruption -- gate(test) would compare against a 0-trial
    # frontier and the veto could never fire. The check is guarded by commit==HEAD
    # ("subsumed by commit==HEAD"): the live config's task set only applies to a
    # baseline at the current protocol.
    head = loop_mod.repo.get_head_commit(cwd=repo_root)
    _seed(
        experiments_dir,
        "exp-base",
        kind="baseline",
        decision=_KEEP,
        run_status="completed",
        tasks=("a",),  # ran train only -- missing the configured test task 'b'
        commit=head,
    )
    with pytest.raises(LoopCorruption, match="configured"):
        scan(
            experiments_dir=experiments_dir,
            repo_root=repo_root,
            config=_config({"a"}, {"b"}),
        )


def test_scan_tolerates_a_stale_active_baseline_with_a_different_task_set(
    repo_root: Path, experiments_dir: Path
) -> None:
    # A kept baseline whose commit != HEAD is legitimately stale (the panels changed
    # -> the commit moved); it ran the OLD configured set, so its task set need not
    # match the live config. scan must NOT hard-fail -- decide() routes to
    # RefreshBaseline. Locks the commit==HEAD guard so it can't over-fire.
    _seed(
        experiments_dir,
        "exp-base",
        kind="baseline",
        decision=_KEEP,
        run_status="completed",
        tasks=("a",),
        commit="stale-commit-not-head",
    )
    world = scan(
        experiments_dir=experiments_dir,
        repo_root=repo_root,
        config=_config({"a"}, {"b"}),
    )
    assert world.active_baseline is not None
    assert world.active_baseline.experiment_id == "exp-base"


# ============================================================================
# The write boundary: the memo, the executors, the driver (plan.md §6/§7/§10).
# ============================================================================


def _trial(*, solved: bool) -> TrialResult:
    mode: FailureMode = "solved" if solved else "verified_rejected"
    return TrialResult(run_id="r0", solved=solved, failure_mode=mode)


def _task(*, solved: bool) -> TaskResult:
    return TaskResult(expected_trial_count=1, trials=[_trial(solved=solved)])


def _result(
    experiment_id: str,
    *,
    commit: str,
    solved_by_task: dict[str, bool],
    run_status: str = "completed",
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        git_commit_hash=commit,
        run_status=run_status,
        started_at="2026-01-01T00:00:00",
        finished_at=None if run_status == "running" else "2026-01-01T01:00:00",
        tasks={t: _task(solved=s) for t, s in solved_by_task.items()},
    )


class _ScriptedBackend:
    """An ``AgentBackend`` whose every turn runs ``action(worktree)`` (the
    side effect a real agent would make: edit core.py + write focus, or write the
    diagnosis + memo draft) then returns a fixed thread id."""

    def __init__(self, action) -> None:
        self.action = action
        self.turns = 0

    def run_turn(self, *, prompt, repo_root, thread_id=None, timeout_sec=3600.0):
        self.turns += 1
        self.action(Path(repo_root))
        return TurnResult(thread_id="thread-1")


class _FakeExp:
    """A stand-in for the ``uv run exp`` subprocess: records each call and writes
    (or appends to) the experiment.json the real orchestrator would produce, so the
    executors' git/loop side effects can be tested without a container run."""

    def __init__(self, *, run_status: str = "completed", solved: bool = False) -> None:
        self.calls: list[tuple[str, frozenset[str], Path]] = []
        self.budgets: list[dict[str, int]] = []
        self._run_status = run_status
        self._solved = solved

    def __call__(
        self, *, worktree, experiment_id, task_ids, experiments_dir, trial_budget
    ):
        self.calls.append((experiment_id, frozenset(task_ids), Path(worktree)))
        self.budgets.append(dict(trial_budget))
        commit = loop_mod.repo.get_head_commit(cwd=worktree)
        path = ExperimentResult.path(experiment_id, root=experiments_dir)
        solved_by_task = {t: self._solved for t in task_ids}
        if path.exists():  # RunVeto appends the test panel into the same record
            prior = ExperimentResult.load(experiment_id, root=experiments_dir)
            solved_by_task = {
                **{t: tr.trials[0].solved for t, tr in prior.tasks.items()},
                **solved_by_task,
            }
        write_experiment_result(
            _result(
                experiment_id,
                commit=commit,
                solved_by_task=solved_by_task,
                run_status=self._run_status,
            ),
            root=experiments_dir,
        )
        if self._run_status == "crashed":
            # The real orchestrator crash-finalizes the record AND the subprocess
            # exits nonzero, so `_run_exp` raises (loop.py:457). Mirror both: the
            # crashing process dies rather than the driver looping on the record.
            raise RuntimeError(f"uv run exp crashed for {experiment_id}")


def _ctx(
    repo_root: Path,
    experiments_dir: Path,
    *,
    train,
    test,
    backend=None,
    run_exp=None,
    experiment_id: str = "exp-cand",
) -> LoopContext:
    return LoopContext(
        repo_root=repo_root,
        experiments_dir=experiments_dir,
        worktree_root=repo_root.parent / "worktrees",
        config=_config(train, test),
        backend=backend or _ScriptedBackend(lambda _wt: None),
        program_md_path=repo_root / "program.md",
        run_exp=run_exp or _FakeExp(),
        new_experiment_id=lambda _dir: experiment_id,
    )


def _make_candidate(repo_root: Path, worktree_root: Path, experiment_id: str) -> str:
    """Commit a real change on top of HEAD and publish it as the candidate ref,
    mirroring what ProposeAndLaunch leaves behind, so Conclude has a ref to act on."""
    with workspace.full_worktree(
        repo_root, worktree_root / f"{experiment_id}-mk", ref="HEAD"
    ) as view:
        (view / "src" / "harness" / "core.py").write_text("VALUE = 99  # candidate\n")
        commit = workspace.commit_candidate(view, experiment_id=experiment_id)
    workspace.set_candidate_ref(repo_root, experiment_id=experiment_id, commit=commit)
    return commit


def _green_test_core(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        workspace, "run_test_core", lambda _wt: workspace.TestCoreResult(True, "ok")
    )


# --- the cumulative memo (§10) ----------------------------------------------


def test_read_learning_memo_is_empty_when_absent(tmp_path: Path) -> None:
    assert read_learning_memo(tmp_path) == ""


def test_swap_learning_draft_rejects_a_missing_draft(tmp_path: Path) -> None:
    message = swap_learning_draft(tmp_path)
    assert message is not None and "learning.draft.md" in message


def test_swap_learning_draft_rejects_an_empty_draft(tmp_path: Path) -> None:
    learning_draft_path(tmp_path).write_text("   \n")
    message = swap_learning_draft(tmp_path)
    assert message is not None and "empty" in message


def test_swap_learning_draft_rejects_an_over_budget_draft(tmp_path: Path) -> None:
    over = loop_mod.LEARNING_MEMO_MAX_LINES + 1
    learning_draft_path(tmp_path).write_text(
        "\n".join(f"line {i}" for i in range(over))
    )
    message = swap_learning_draft(tmp_path)
    assert message is not None and "budget" in message
    # rejected: the live memo is untouched, the draft remains for the re-condense
    assert not learning_memo_path(tmp_path).exists()
    assert learning_draft_path(tmp_path).exists()


def test_swap_learning_draft_atomically_replaces_the_live_memo(tmp_path: Path) -> None:
    learning_memo_path(tmp_path).write_text("old memo\n")
    learning_draft_path(tmp_path).write_text("fresh curated memo\n")
    assert swap_learning_draft(tmp_path) is None
    assert read_learning_memo(tmp_path) == "fresh curated memo\n"
    assert not learning_draft_path(tmp_path).exists()  # draft consumed by the swap


def test_build_diagnosis_prompt_points_at_the_record_and_targets() -> None:
    prompt = build_diagnosis_prompt(
        program_md_path=Path("/repo/program.md"),
        learning_md_path=Path("/repo/experiments/learning.md"),
        experiment_json_path=Path("/repo/experiments/exp-1/experiment.json"),
        diagnosis_path=Path("/repo/experiments/exp-1/diagnosis.md"),
        draft_path=Path("/repo/experiments/learning.draft.md"),
        feedback_note="memo over budget",
    )
    assert "/repo/experiments/exp-1/experiment.json" in prompt
    assert "/repo/experiments/exp-1/diagnosis.md" in prompt
    assert "/repo/experiments/learning.draft.md" in prompt
    assert "memo over budget" in prompt


# --- RefreshBaseline / Conclude(baseline) -----------------------------------


def test_refresh_baseline_prewrites_loop_and_runs_all_configured(
    repo_root: Path, experiments_dir: Path
) -> None:
    fake = _FakeExp()
    ctx = _ctx(
        repo_root,
        experiments_dir,
        train={"a"},
        test={"b"},
        run_exp=fake,
        experiment_id="exp-base",
    )
    execute(RefreshBaseline(), _blank_world(), ctx)
    # prewritten as a pending baseline (decision still null until Conclude)
    loop = loop_mod.load_loop("exp-base", root=experiments_dir)
    assert loop.kind == "baseline" and loop.decision is None
    # ran the full configured panel in the primary repo, one exp call
    assert len(fake.calls) == 1
    exp_id, task_ids, worktree = fake.calls[0]
    assert exp_id == "exp-base" and task_ids == frozenset({"a", "b"})
    assert worktree == repo_root  # baseline runs unmodified HEAD in the primary
    # no baseline to derive from -> uniform-full on every task (§9 #7).
    assert fake.budgets[0] == {"a": 3, "b": 3}


def test_conclude_baseline_writes_a_keep_decision(
    repo_root: Path, experiments_dir: Path
) -> None:
    head = loop_mod.repo.get_head_commit(cwd=repo_root)
    loop = LoopResult(
        experiment_id="exp-base",
        kind="baseline",
        focus_name="baseline at HEAD",
        parent_baseline_experiment_id=None,
        decision=None,
    )
    world = _blank_world(
        pending=PendingRun(
            loop=loop,
            result=_result(
                "exp-base", commit=head, solved_by_task={"a": True, "b": True}
            ),
        )
    )
    ctx = _ctx(repo_root, experiments_dir, train={"a"}, test={"b"})
    execute(Conclude("exp-base"), world, ctx)
    concluded = loop_mod.load_loop("exp-base", root=experiments_dir)
    assert concluded.decision is not None and concluded.decision.kind == "keep"


# --- Conclude(candidate): the strict-order ref dance ------------------------


def test_conclude_candidate_keep_fast_forwards_head_and_drops_the_ref(
    repo_root: Path, experiments_dir: Path
) -> None:
    head_before = loop_mod.repo.get_head_commit(cwd=repo_root)
    commit = _make_candidate(repo_root, repo_root.parent / "worktrees", "exp-cand")
    # A pure frontier baseline (never solved either panel): a candidate train solve
    # is a frontier improvement (Fisher skipped, majority-solve bar) -> promotion
    # keep; the veto can't regress a 0-baseline -> combine keep.
    base = _result("exp-base", commit=head_before, solved_by_task={})
    # candidate solves the train task and ran the test panel (test_done)
    cand = _result("exp-cand", commit=commit, solved_by_task={"a": True, "b": False})
    cand_loop = LoopResult(
        experiment_id="exp-cand",
        kind="candidate",
        focus_name="mechanism-x",
        parent_baseline_experiment_id="exp-base",
        decision=None,
    )
    world = _blank_world(
        train={"a"},
        test={"b"},
        active_baseline=base,
        pending=PendingRun(loop=cand_loop, result=cand),
    )
    ctx = _ctx(repo_root, experiments_dir, train={"a"}, test={"b"})
    execute(Conclude("exp-cand"), world, ctx)
    # kept: HEAD fast-forwarded to the candidate commit, candidate ref dropped
    assert loop_mod.repo.get_head_commit(cwd=repo_root) == commit
    assert not loop_mod.repo.git_ref_exists(
        cwd=repo_root, ref=workspace.candidate_ref("exp-cand")
    )
    concluded = loop_mod.load_loop("exp-cand", root=experiments_dir)
    assert concluded.decision is not None and concluded.decision.kind == "keep"


def test_conclude_candidate_train_discard_sets_failed_ref_no_ff(
    repo_root: Path, experiments_dir: Path
) -> None:
    head_before = loop_mod.repo.get_head_commit(cwd=repo_root)
    commit = _make_candidate(repo_root, repo_root.parent / "worktrees", "exp-cand")
    base = _result("exp-base", commit=head_before, solved_by_task={"a": False})
    # candidate ran ONLY train (test never ran) and did not improve -> train discard
    cand = _result("exp-cand", commit=commit, solved_by_task={"a": False})
    cand_loop = LoopResult(
        experiment_id="exp-cand",
        kind="candidate",
        focus_name="mechanism-x",
        parent_baseline_experiment_id="exp-base",
        decision=None,
    )
    world = _blank_world(
        train={"a"},
        test={"b"},
        active_baseline=base,
        pending=PendingRun(loop=cand_loop, result=cand),
    )
    ctx = _ctx(repo_root, experiments_dir, train={"a"}, test={"b"})
    execute(Conclude("exp-cand"), world, ctx)
    # discarded: HEAD unmoved, commit preserved under the failed ref, candidate dropped
    assert loop_mod.repo.get_head_commit(cwd=repo_root) == head_before
    assert loop_mod.repo.git_ref_exists(
        cwd=repo_root, ref=workspace.failed_ref("exp-cand")
    )
    assert not loop_mod.repo.git_ref_exists(
        cwd=repo_root, ref=workspace.candidate_ref("exp-cand")
    )
    concluded = loop_mod.load_loop("exp-cand", root=experiments_dir)
    assert concluded.decision is not None and concluded.decision.kind == "discard"


def test_conclude_replays_cleanly_after_a_crash_dropped_the_candidate_ref(
    repo_root: Path, experiments_dir: Path
) -> None:
    # Crash window (plan.md:505-513): a keep that fast-forwarded HEAD and dropped the
    # candidate ref but died BEFORE writing the decision. Replay re-enters Conclude
    # with loop.decision still null and the candidate ref gone -- it must finish the
    # write idempotently, not blow up resolving the dropped ref.
    commit = _make_candidate(repo_root, repo_root.parent / "worktrees", "exp-cand")
    workspace.fast_forward_primary(repo_root, commit=commit)  # the keep's FF (step 1)
    workspace.drop_candidate_ref(repo_root, experiment_id="exp-cand")  # step 2
    # ... process dies here, before step 3 (write loop.json) -> decision still null.
    base = _result("exp-base", commit=commit, solved_by_task={})  # frontier -> keep
    cand = _result("exp-cand", commit=commit, solved_by_task={"a": True, "b": False})
    cand_loop = LoopResult(
        experiment_id="exp-cand",
        kind="candidate",
        focus_name="mechanism-x",
        parent_baseline_experiment_id="exp-base",
        decision=None,
    )
    world = _blank_world(
        train={"a"},
        test={"b"},
        active_baseline=base,
        pending=PendingRun(loop=cand_loop, result=cand),
    )
    ctx = _ctx(repo_root, experiments_dir, train={"a"}, test={"b"})
    execute(Conclude("exp-cand"), world, ctx)  # must not raise on the missing ref
    concluded = loop_mod.load_loop("exp-cand", root=experiments_dir)
    assert concluded.decision is not None and concluded.decision.kind == "keep"
    assert loop_mod.repo.get_head_commit(cwd=repo_root) == commit  # FF still holds


# --- ProposeAndLaunch / RunVeto ---------------------------------------------


def test_propose_and_launch_commits_refs_prewrites_and_runs_train(
    repo_root: Path, experiments_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _green_test_core(monkeypatch)
    head_before = loop_mod.repo.get_head_commit(cwd=repo_root)

    def edit_and_focus(worktree: Path) -> None:
        (worktree / "src" / "harness" / "core.py").write_text("VALUE = 7  # mech\n")
        (worktree / workspace.FOCUS_FILE).write_text("stuck-detect-arm\n")

    fake = _FakeExp()
    ctx = _ctx(
        repo_root,
        experiments_dir,
        train={"a"},
        test={"b"},
        backend=_ScriptedBackend(edit_and_focus),
        run_exp=fake,
    )
    base = _result(
        "exp-base", commit=head_before, solved_by_task={"a": False, "b": False}
    )
    world = _blank_world(train={"a"}, test={"b"}, active_baseline=base)
    execute(ProposeAndLaunch(), world, ctx)

    # a candidate commit was published as the candidate ref (HEAD untouched)
    assert loop_mod.repo.git_ref_exists(
        cwd=repo_root, ref=workspace.candidate_ref("exp-cand")
    )
    assert loop_mod.repo.get_head_commit(cwd=repo_root) == head_before
    # loop.json prewritten with the captured focus + the parent baseline, null decision
    loop = loop_mod.load_loop("exp-cand", root=experiments_dir)
    assert loop.kind == "candidate" and loop.decision is None
    assert loop.focus_name == "stuck-detect-arm"
    assert loop.parent_baseline_experiment_id == "exp-base"
    # exp ran the TRAIN panel only, at the candidate ref (not the primary)
    assert len(fake.calls) == 1
    exp_id, task_ids, worktree = fake.calls[0]
    assert exp_id == "exp-cand" and task_ids == frozenset({"a"})
    assert worktree != repo_root


def test_propose_and_launch_budgets_train_from_the_baseline(
    repo_root: Path, experiments_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The per-task train budget crosses the run_exp seam: a train task the baseline
    # solved on every trial starts at 1 confirming trial; an unsolved one at full
    # (budget_from_baseline, §9 #7). This is the consume side of EXP_TRIAL_BUDGET.
    _green_test_core(monkeypatch)

    def edit_and_focus(worktree: Path) -> None:
        (worktree / "src" / "harness" / "core.py").write_text("VALUE = 7  # mech\n")
        (worktree / workspace.FOCUS_FILE).write_text("mech\n")

    fake = _FakeExp()
    ctx = _ctx(
        repo_root,
        experiments_dir,
        train={"solved", "unsolved"},
        test={"b"},
        backend=_ScriptedBackend(edit_and_focus),
        run_exp=fake,
    )
    head = loop_mod.repo.get_head_commit(cwd=repo_root)
    base = _result(
        "exp-base",
        commit=head,
        solved_by_task={"solved": True, "unsolved": False},
    )
    world = _blank_world(train={"solved", "unsolved"}, test={"b"}, active_baseline=base)
    execute(ProposeAndLaunch(), world, ctx)
    assert fake.budgets[0] == {"solved": 1, "unsolved": 3}


def test_run_veto_runs_the_test_panel_at_the_candidate_ref(
    repo_root: Path, experiments_dir: Path
) -> None:
    _make_candidate(repo_root, repo_root.parent / "worktrees", "exp-cand")
    fake = _FakeExp()
    ctx = _ctx(repo_root, experiments_dir, train={"a"}, test={"b"}, run_exp=fake)
    world = _blank_world(train={"a"}, test={"b"})
    execute(RunVeto("exp-cand"), world, ctx)
    assert len(fake.calls) == 1
    exp_id, task_ids, worktree = fake.calls[0]
    assert exp_id == "exp-cand" and task_ids == frozenset({"b"})  # TEST panel only
    assert worktree != repo_root


def test_run_exp_serializes_the_trial_budget_to_the_env(
    repo_root: Path, experiments_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real _run_exp (not the fake) hands the per-task budget across the
    # subprocess boundary as the EXP_TRIAL_BUDGET JSON map, beside EXP_TASK_IDS /
    # EXP_EXPERIMENT_ID / EXP_EXPERIMENTS_DIR. cli._selected_trial_budget reads it.
    import json

    captured: dict[str, str] = {}

    def fake_tty(args, *, cwd, env):
        captured.update(env)
        # the orchestrator would write the record; stub it so the existence check passes
        write_experiment_result(
            _result("exp-x", commit="c0", solved_by_task={"a": True}),
            root=experiments_dir,
        )
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        )

    monkeypatch.setattr(loop_mod, "_run_with_live_tty_output", fake_tty)
    loop_mod._run_exp(
        worktree=repo_root,
        experiment_id="exp-x",
        task_ids=frozenset({"a", "b"}),
        experiments_dir=experiments_dir,
        trial_budget={"a": 1, "b": 3},
    )
    assert json.loads(captured["EXP_TRIAL_BUDGET"]) == {"a": 1, "b": 3}
    assert captured["EXP_TASK_IDS"] == "a,b"
    assert captured["EXP_EXPERIMENT_ID"] == "exp-x"


# --- Diagnose ---------------------------------------------------------------


def test_diagnose_writes_the_log_and_swaps_the_learning_memo(
    repo_root: Path, experiments_dir: Path
) -> None:
    (experiments_dir / "exp-cand").mkdir(parents=True)
    learning_memo_path(experiments_dir).write_text("stale memo\n")

    def write_memo(_worktree: Path) -> None:
        (experiments_dir / "exp-cand" / loop_mod.DIAGNOSIS_FILENAME).write_text(
            "raw cycle log\n"
        )
        learning_draft_path(experiments_dir).write_text("curated rewrite\n")

    ctx = _ctx(
        repo_root,
        experiments_dir,
        train={"a"},
        test={"b"},
        backend=_ScriptedBackend(write_memo),
    )
    execute(Diagnose("exp-cand"), _blank_world(), ctx)
    assert (experiments_dir / "exp-cand" / loop_mod.DIAGNOSIS_FILENAME).exists()
    assert read_learning_memo(experiments_dir) == "curated rewrite\n"  # swapped
    assert not learning_draft_path(experiments_dir).exists()  # draft consumed


def test_execute_on_halt_is_a_programming_error(
    repo_root: Path, experiments_dir: Path
) -> None:
    # decide() never hands Halt to execute(); run_auto returns it. Guard the seam.
    ctx = _ctx(repo_root, experiments_dir, train={"a"}, test={"b"})
    with pytest.raises(AssertionError, match="Halt"):
        execute(Halt("boom"), _blank_world(), ctx)


# --- run_auto driver --------------------------------------------------------


def test_run_auto_relaunches_after_a_prior_crashed_pending(
    repo_root: Path, experiments_dir: Path
) -> None:
    # A leftover crashed pending from a dead prior process is filtered (§11): the
    # driver does NOT halt on it -- it proceeds to a fresh RefreshBaseline. That
    # relaunch here also crashes, so `_run_exp` raises (as the real subprocess would
    # on a nonzero exit); the point is fake.calls == 1 with a fresh id, i.e. the loop
    # launched rather than wedging on the old corpse.
    _seed(
        experiments_dir,
        "exp-cand",
        kind="candidate",
        decision=None,
        parent=None,
        run_status="crashed",
    )
    fake = _FakeExp(run_status="crashed")
    ctx = _ctx(
        repo_root,
        experiments_dir,
        train={"a"},
        test={"b"},
        run_exp=fake,
        experiment_id="exp-base",
    )
    with pytest.raises(RuntimeError):
        run_auto(ctx)
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == "exp-base"  # a fresh baseline, not the old corpse


def test_run_auto_propagates_a_crash_during_a_run(
    repo_root: Path, experiments_dir: Path
) -> None:
    # Empty -> RefreshBaseline; the run crash-finalizes and the subprocess exits
    # nonzero, so `_run_exp` raises and run_auto propagates it (the crashing process
    # dies; the user reruns -- §11). It must not loop on the crashed record.
    fake = _FakeExp(run_status="crashed")
    ctx = _ctx(
        repo_root,
        experiments_dir,
        train={"a"},
        test={"b"},
        run_exp=fake,
        experiment_id="exp-base",
    )
    with pytest.raises(RuntimeError):
        run_auto(ctx)
    assert len(fake.calls) == 1  # one RefreshBaseline launch, then it raised


def _blank_world(
    *,
    train: Collection[str] = ("a",),
    test: Collection[str] = ("b",),
    active_baseline=None,
    pending=None,
    undiagnosed_candidate_id=None,
) -> World:
    return World(
        head_commit="0" * 40,
        primary_dirty=False,
        train_tasks=frozenset(train),
        test_tasks=frozenset(test),
        active_baseline=active_baseline,
        pending=pending,
        undiagnosed_candidate_id=undiagnosed_candidate_id,
    )
