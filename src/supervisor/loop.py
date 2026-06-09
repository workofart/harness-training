"""The outer loop's effects (plan.md §6): the read boundary and the write boundary.

``scan()`` reads ``experiments/`` + git into a ``World`` (the only input to the pure
``decide()``); ``execute()`` performs the one command ``decide()`` returns; ``run_auto``
drives ``while True: scan -> decide -> execute`` until ``Halt``. All control state is
derived here from the experiment dirs + git -- there is no ``state.json`` authority (§5).

Every auto run follows one lifecycle -- prewrite ``loop.json{decision:null}`` -> run
(1+ ``uv run exp`` calls) -> ``Conclude`` -- so no completed run is ever lost. ``scan()``
also enforces the §12 corruption hard-fails (panels non-empty + disjoint, <=1 pending,
a pending candidate's parent is the active baseline, recorded task presence) before any
command runs.

The write boundary is the six command executors (``RefreshBaseline``/
``ProposeAndLaunch``/``RunVeto``/``Conclude``/``Diagnose``/``Halt``) dispatched by
``execute()``; they own the only side effects -- the ``uv run exp`` subprocess seam
(§7), the candidate worktree/ref lifecycle (delegated to ``workspace``), and the
atomic-swap learning memo (§10). Their injectable dependencies are gathered in a
``LoopContext`` so the driver is unit-testable with a stub backend + a fake exp
runner.
"""

from __future__ import annotations

import errno
import json
import locale
import os
import pty
import selectors
import subprocess
import sys
import tty
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from src import repo
from src.experiment.record import ExperimentResult
from src.experiment.writer import write_json_atomic
from src.supervisor import agent, workspace
from src.supervisor.policy import (
    Command,
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
    budget_from_baseline,
    combine,
    decide,
    gate,
)

if TYPE_CHECKING:
    from src.config import HarnessConfig
    from src.supervisor.agent_backend import AgentBackend

LOOP_FILENAME = "loop.json"
DIAGNOSIS_FILENAME = "diagnosis.md"


class LoopCorruption(RuntimeError):
    """An impossible control state (§11/§12) -- a hard-fail for a human, distinct
    from the normal ``Halt`` transitions ``decide()`` returns. Raised by ``scan``;
    propagates out of ``run_auto`` rather than being silently recovered."""


# --- loop.json (auto-owned; one writer) -------------------------------------


def loop_path(experiment_id: str, *, root: Path) -> Path:
    return root.resolve() / experiment_id / LOOP_FILENAME


def load_loop(experiment_id: str, *, root: Path) -> LoopResult:
    return LoopResult.model_validate_json(
        loop_path(experiment_id, root=root).read_text()
    )


def write_loop(result: LoopResult, *, root: Path) -> None:
    # Atomic (temp + os.replace), so a crash never leaves a half-written decision;
    # `decision` is always written last by `Conclude` (§6).
    write_json_atomic(
        loop_path(result.experiment_id, root=root), result.model_dump(mode="json")
    )


# --- scan: experiments/ + git -> World --------------------------------------


@dataclass(frozen=True, slots=True)
class _AutoExperiment:
    loop: LoopResult
    result: ExperimentResult | None
    has_diagnosis: bool


def _load_auto_experiments(experiments_dir: Path) -> list[_AutoExperiment]:
    # An "auto experiment" is a dir with a loop.json (a plain `exp` one-off has
    # none, and scan ignores it). result is None => launch_incomplete (loop
    # prewritten, experiment.json never produced).
    if not experiments_dir.exists():
        return []
    autos: list[_AutoExperiment] = []
    for child in sorted(experiments_dir.iterdir()):
        if (
            not child.is_dir()
            or not loop_path(child.name, root=experiments_dir).exists()
        ):
            continue
        result = (
            ExperimentResult.load(child.name, root=experiments_dir)
            if ExperimentResult.path(child.name, root=experiments_dir).exists()
            else None
        )
        autos.append(
            _AutoExperiment(
                loop=load_loop(child.name, root=experiments_dir),
                result=result,
                has_diagnosis=(child / DIAGNOSIS_FILENAME).exists(),
            )
        )
    return autos


def _assert_panel_contract(*, train: frozenset[str], test: frozenset[str]) -> None:
    # §12: both panels non-empty and disjoint. An empty test panel would make the
    # veto vacuously "done" (decide()'s `test_tasks <= tasks` is trivially true),
    # silently skipping regression-veto; an overlap can't even be represented
    # (ExperimentResult.tasks is keyed by task_id).
    if not train or not test:
        raise LoopCorruption(
            "config train/test panels must both be non-empty (§12); "
            f"train={sorted(train)} test={sorted(test)}"
        )
    overlap = train & test
    if overlap:
        raise LoopCorruption(f"train/test panels overlap (§12): {sorted(overlap)}")


def _assert_pending_contract(
    pending: _AutoExperiment,
    active_baseline: ExperimentResult | None,
    *,
    train: frozenset[str],
    configured: frozenset[str],
) -> None:
    loop = pending.loop
    result = pending.result
    if loop.kind == "candidate":
        # §12: the sequential loop prewrites one candidate against the current
        # baseline; its parent must be the baseline gate() compares against.
        baseline_id = None if active_baseline is None else active_baseline.experiment_id
        if loop.parent_baseline_experiment_id != baseline_id:
            raise LoopCorruption(
                f"pending candidate {loop.experiment_id} parent "
                f"{loop.parent_baseline_experiment_id!r} != active baseline "
                f"{baseline_id!r} (§12)"
            )
        # A completed candidate must have recorded exactly train (awaiting/after a
        # train discard) or train|test (both panels ran) -- never a partial or
        # foreign set. A still-running run is checked by decide()'s Halt, not here.
        if result is not None and result.run_status == "completed":
            recorded = frozenset(result.tasks)
            if recorded not in (train, configured):
                raise LoopCorruption(
                    f"candidate {loop.experiment_id} recorded {sorted(recorded)}, "
                    f"expected train or train|test (§12)"
                )
    elif result is not None and result.run_status == "completed":
        # §12: an auto baseline ran exactly all configured tasks.
        recorded = frozenset(result.tasks)
        if recorded != configured:
            raise LoopCorruption(
                f"baseline {loop.experiment_id} recorded {sorted(recorded)}, "
                f"expected all configured tasks (§12)"
            )


def scan(*, experiments_dir: Path, repo_root: Path, config) -> World:
    """Build ``decide()``'s ``World`` from the experiment dirs + git, enforcing the
    §12 corruption hard-fails first. ``config`` is the live ``HarnessConfig`` (the
    primary is clean, so its working tree == HEAD)."""
    train = config.train_tasks
    test = config.test_tasks
    _assert_panel_contract(train=train, test=test)
    configured = train | test

    autos = _load_auto_experiments(experiments_dir)

    # Only a *completed* pending is "live" and routed on; a dead pending (crashed,
    # killed mid-run leaving run_status "running", or launched-but-never-recorded
    # with result is None) is filtered out here so a manual `uv run auto` after a
    # crash proceeds without interference (§11). A dead pending is never adopted
    # (its decision is null, so not a keep) and its artifacts stay on disk for
    # inspection. Only >1 *live* pending is corruption (two candidates the
    # sequential loop never concluded); stacked dead corpses are tolerated.
    live_pendings = [
        a
        for a in autos
        if a.loop.decision is None
        and a.result is not None
        and a.result.run_status == "completed"
    ]
    if len(live_pendings) > 1:
        raise LoopCorruption(
            "more than one live pending run (§12): "
            + ", ".join(a.loop.experiment_id for a in live_pendings)
        )

    keeps = [
        a
        for a in autos
        if a.loop.decision is not None
        and a.loop.decision.kind == "keep"
        and a.result is not None
        and a.result.finished_at is not None
    ]
    active_baseline = (
        max(keeps, key=lambda a: a.result.finished_at).result if keeps else None
    )

    head_commit = repo.get_head_commit(cwd=repo_root)
    # §12: the active baseline gate() compares against must have run exactly the
    # configured tasks -- else gate(test) scores against a 0-trial frontier and the
    # veto can never fire. Guarded by commit==HEAD ("subsumed by commit==HEAD",
    # plan.md:695): a stale baseline (commit moved => panels changed) ran the OLD
    # set and is legitimately mismatched -- decide() routes it to RefreshBaseline.
    if (
        active_baseline is not None
        and active_baseline.git_commit_hash == head_commit
        and frozenset(active_baseline.tasks) != configured
    ):
        raise LoopCorruption(
            f"active baseline {active_baseline.experiment_id} at HEAD recorded "
            f"{sorted(active_baseline.tasks)}, expected all configured tasks (§12)"
        )

    undiagnosed = next(
        (
            a.loop.experiment_id
            for a in autos
            if a.loop.kind == "candidate"
            and a.loop.decision is not None
            and not a.has_diagnosis
        ),
        None,
    )

    pending_run: PendingRun | None = None
    if live_pendings:
        pending = live_pendings[0]
        _assert_pending_contract(
            pending, active_baseline, train=train, configured=configured
        )
        pending_run = PendingRun(loop=pending.loop, result=pending.result)

    return World(
        head_commit=head_commit,
        primary_dirty=repo.is_dirty(cwd=repo_root),
        train_tasks=train,
        test_tasks=test,
        active_baseline=active_baseline,
        pending=pending_run,
        undiagnosed_candidate_id=undiagnosed,
    )


# --- the cumulative memo: raw log + atomic-swap curated view (§10) -----------

LEARNING_FILENAME = "learning.md"
LEARNING_DRAFT_FILENAME = "learning.draft.md"
# Line ceiling on the curated learning.md so the memo stays condensed: an
# over-budget draft is rejected and fed back to force a rewrite, not an append.
# Detail is never lost -- the per-cycle diagnosis.md raw log is the durable record.
LEARNING_MEMO_MAX_LINES = 150


def learning_memo_path(experiments_dir: Path) -> Path:
    return experiments_dir / LEARNING_FILENAME


def learning_draft_path(experiments_dir: Path) -> Path:
    return experiments_dir / LEARNING_DRAFT_FILENAME


def read_learning_memo(experiments_dir: Path) -> str:
    path = learning_memo_path(experiments_dir)
    return path.read_text() if path.exists() else ""


def swap_learning_draft(experiments_dir: Path) -> str | None:
    """Validate the agent's ``learning.draft.md`` (present, non-empty, within the
    line budget) and atomically swap it over ``learning.md`` (``os.replace``).
    Returns a rejection message (fed back to re-condense) if the draft is missing,
    empty, or over budget; ``None`` once swapped. The live ``learning.md`` is never
    half-written -- only replaced atomically or left untouched."""
    draft = learning_draft_path(experiments_dir)
    if not draft.exists():
        return f"write the full rewritten curated memo to {draft}"
    content = draft.read_text()
    if not content.strip():
        return "learning.draft.md is empty; emit the full curated memo"
    line_count = len(content.splitlines())
    if line_count > LEARNING_MEMO_MAX_LINES:
        return (
            f"learning.draft.md is {line_count} lines, over the "
            f"{LEARNING_MEMO_MAX_LINES}-line budget; rewrite it shorter (do not "
            "append). Keep durable, generic mechanism knowledge and the current "
            "frontier; drop per-cycle solve-count deltas and variance narration."
        )
    os.replace(draft, learning_memo_path(experiments_dir))
    return None


def build_diagnosis_prompt(
    *,
    program_md_path: Path,
    learning_md_path: Path,
    experiment_json_path: Path,
    diagnosis_path: Path,
    draft_path: Path,
    feedback_note: str | None = None,
) -> str:
    """The post-run diagnosis turn prompt (§10): read program.md + the run record +
    the current memo, then write the raw per-cycle log and a full rewritten memo
    draft (the loop validates + atomically swaps the draft over ``learning.md``)."""
    lines = [
        "Autonomous post-run diagnosis phase.",
        "Follow program.md.",
        "Read these authoritative files now:",
        f"- {program_md_path}",
        f"- {experiment_json_path}",
        f"- {learning_md_path}",
        "",
        f"Write the raw per-cycle diagnosis log to: {diagnosis_path}",
        f"Write the full rewritten curated memo to: {draft_path}",
        f"  (the supervisor validates it -- non-empty, <= {LEARNING_MEMO_MAX_LINES} "
        "lines -- and atomically swaps it over learning.md)",
    ]
    if feedback_note is not None:
        lines.extend(
            ["", "Supervisor feedback from the previous diagnosis turn:", feedback_note]
        )
    return "\n".join(lines)


# --- the `uv run exp` subprocess seam (§7) ----------------------------------

# (worktree, experiment_id, task_ids, experiments_dir, trial_budget) -> None;
# injectable so the loop is testable without a real container run.
ExpRunner = Callable[..., None]


def _run_with_live_tty_output(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """Run ``args`` under PTYs so the child's tty-buffered progress streams to our
    stdout/stderr live while also being captured for the post-mortem on failure."""
    encoding = locale.getencoding()
    stdout_master, stdout_slave = pty.openpty()
    stderr_master, stderr_slave = pty.openpty()
    tty.setraw(stdout_slave)
    tty.setraw(stderr_slave)
    process = subprocess.Popen(
        args,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=stdout_slave,
        stderr=stderr_slave,
        close_fds=True,
    )
    os.close(stdout_slave)
    os.close(stderr_slave)
    output_by_fd = {
        stdout_master: (sys.stdout, []),
        stderr_master: (sys.stderr, []),
    }
    selector = selectors.DefaultSelector()
    selector.register(stdout_master, selectors.EVENT_READ)
    selector.register(stderr_master, selectors.EVENT_READ)
    try:
        while selector.get_map():
            for key, _events in selector.select():
                try:
                    chunk = os.read(key.fd, 8192)
                except OSError as exc:
                    if exc.errno != errno.EIO:
                        raise
                    chunk = b""
                if not chunk:
                    selector.unregister(key.fd)
                    os.close(key.fd)
                    continue
                stream, chunks = output_by_fd[key.fd]
                chunks.append(chunk)
                stream.write(chunk.decode(encoding, errors="replace"))
                stream.flush()
        return subprocess.CompletedProcess(
            args=args,
            returncode=process.wait(),
            stdout=b"".join(output_by_fd[stdout_master][1]).decode(
                encoding, errors="replace"
            ),
            stderr=b"".join(output_by_fd[stderr_master][1]).decode(
                encoding, errors="replace"
            ),
        )
    finally:
        selector.close()
        for fd in (stdout_master, stderr_master):
            try:
                os.close(fd)
            except OSError:
                pass
        if process.poll() is None:
            process.kill()
            process.wait()


def _run_exp(
    *,
    worktree: Path,
    experiment_id: str,
    task_ids: frozenset[str],
    experiments_dir: Path,
    trial_budget: Mapping[str, int],
) -> None:
    """Launch ``uv run exp`` in ``worktree`` over ``task_ids``, appending into the
    one ``experiment_id`` dir under the absolute ``experiments_dir`` (§12 path
    anchoring -- ``EXP_EXPERIMENTS_DIR`` keeps a worktree run byte-equivalent to a
    primary run). ``trial_budget`` (the per-task ``budget_from_baseline`` count,
    §9) crosses the subprocess boundary as the ``EXP_TRIAL_BUDGET`` JSON map so a
    candidate train/veto run gets the deterministic-baseline single-trial shortcut;
    ``cli`` honors it like ``EXP_EXPERIMENTS_DIR``. Raises
    ``ChatGptCodexCredentialsExpiredError`` on the dead-creds halt (exit 42, needs
    ``codex login``), a ``RuntimeError`` on any other failure, and a
    ``RuntimeError`` if the run produced no ``experiment.json``."""
    from src.llm.codex import (
        CODEX_CREDENTIALS_EXPIRED_EXIT_CODE,
        CODEX_CREDENTIALS_EXPIRED_MESSAGE,
        ChatGptCodexCredentialsExpiredError,
    )

    env = {
        **os.environ,
        "EXP_EXPERIMENT_ID": experiment_id,
        "EXP_TASK_IDS": ",".join(sorted(task_ids)),
        "EXP_EXPERIMENTS_DIR": str(experiments_dir.resolve()),
        "EXP_TRIAL_BUDGET": json.dumps(dict(trial_budget)),
    }
    completed = _run_with_live_tty_output(["uv", "run", "exp"], cwd=worktree, env=env)
    if completed.returncode == CODEX_CREDENTIALS_EXPIRED_EXIT_CODE:
        # Dead credentials need a human; halt rather than finalizing a crash record
        # the loop would treat like a discard and re-launch into the same auth wall.
        raise ChatGptCodexCredentialsExpiredError(CODEX_CREDENTIALS_EXPIRED_MESSAGE)
    if completed.returncode != 0:
        raise RuntimeError(
            f"uv run exp failed (rc={completed.returncode}):\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    record_path = ExperimentResult.path(experiment_id, root=experiments_dir)
    if not record_path.exists():
        raise RuntimeError(f"exp produced no record: {record_path}")


def _new_experiment_id(experiments_dir: Path) -> str:
    base = f"exp-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    if not (experiments_dir / base).exists():
        return base
    index = 1
    while (experiments_dir / f"{base}-{index}").exists():
        index += 1
    return f"{base}-{index}"


# --- the write boundary: context + command executors (§6/§7) ----------------


@dataclass(frozen=True, slots=True)
class LoopContext:
    """The effectful dependencies the executors need, gathered so ``run_auto`` and
    each executor are unit-testable with a stub ``backend`` + a fake ``run_exp``.
    ``experiments_dir`` is the absolute ``<main_repo>/experiments`` (§12)."""

    repo_root: Path
    experiments_dir: Path
    worktree_root: Path
    config: HarnessConfig
    backend: AgentBackend
    program_md_path: Path
    run_exp: ExpRunner = _run_exp
    new_experiment_id: Callable[[Path], str] = _new_experiment_id


def _prewrite_loop(
    ctx: LoopContext,
    *,
    experiment_id: str,
    kind: str,
    focus_name: str,
    parent: str | None,
) -> None:
    # PREWRITE the loop.json with decision:null before the run, so a crash leaves a
    # pending run scan() can recover (Conclude is idempotent) -- never a lost run.
    write_loop(
        LoopResult(
            experiment_id=experiment_id,
            kind=kind,
            focus_name=focus_name,
            parent_baseline_experiment_id=parent,
            decision=None,
        ),
        root=ctx.experiments_dir,
    )


def _refresh_baseline(ctx: LoopContext) -> None:
    # No worktree/ref: run the unmodified HEAD in the (clean) primary over ALL
    # configured tasks; Conclude marks it keep. primary_dirty would have Halted.
    experiment_id = ctx.new_experiment_id(ctx.experiments_dir)
    configured = ctx.config.train_tasks | ctx.config.test_tasks
    _prewrite_loop(
        ctx,
        experiment_id=experiment_id,
        kind="baseline",
        focus_name="baseline at HEAD",
        parent=None,
    )
    ctx.run_exp(
        worktree=ctx.repo_root,
        experiment_id=experiment_id,
        task_ids=configured,
        experiments_dir=ctx.experiments_dir,
        # No baseline to derive from -> uniform-full on every configured task.
        trial_budget=budget_from_baseline(
            None, task_ids=configured, full=ctx.config.task_trials
        ),
    )


def _propose_and_launch(world: World, ctx: LoopContext) -> None:
    baseline = world.active_baseline
    assert baseline is not None  # rule 8 fires only when baseline_ok (commit==HEAD)
    experiment_id = ctx.new_experiment_id(ctx.experiments_dir)
    configured = world.train_tasks | world.test_tasks
    edit_worktree = ctx.worktree_root / f"{experiment_id}-edit"
    with workspace.sparse_worktree(ctx.repo_root, edit_worktree) as view:
        proposed = agent.propose_candidate(
            worktree_path=view,
            backend=ctx.backend,
            thread_id=None,
            prompt_builder=lambda note: agent.build_prelaunch_prompt(
                program_md_path=ctx.program_md_path,
                learning_md_path=learning_memo_path(ctx.experiments_dir),
                evidence_paths=(),
                feedback_note=note,
            ),
            validate=lambda wt: agent.validate_proposal(wt, task_ids=configured),
        )
        # §7 order: commit C, ref C, THEN drop the worktree -- C is never unreferenced.
        commit = workspace.commit_candidate(view, experiment_id=experiment_id)
        workspace.set_candidate_ref(
            ctx.repo_root, experiment_id=experiment_id, commit=commit
        )
    _prewrite_loop(
        ctx,
        experiment_id=experiment_id,
        kind="candidate",
        focus_name=proposed.focus_name,
        parent=baseline.experiment_id,
    )
    run_worktree = ctx.worktree_root / f"{experiment_id}-train"
    with workspace.full_worktree(
        ctx.repo_root, run_worktree, ref=workspace.candidate_ref(experiment_id)
    ) as run_view:
        ctx.run_exp(
            worktree=run_view,
            experiment_id=experiment_id,
            task_ids=world.train_tasks,
            experiments_dir=ctx.experiments_dir,
            # A train task the baseline solved deterministically starts at 1 trial
            # (confirm-on-fail expands it); everything else at full (§9 #7).
            trial_budget=budget_from_baseline(
                baseline, task_ids=world.train_tasks, full=ctx.config.task_trials
            ),
        )


def _run_veto(experiment_id: str, world: World, ctx: LoopContext) -> None:
    # train kept; run the TEST panel into the same experiment dir (the candidate ref
    # still exists -- Conclude drops it). exp appends to the existing experiment.json.
    run_worktree = ctx.worktree_root / f"{experiment_id}-veto"
    with workspace.full_worktree(
        ctx.repo_root, run_worktree, ref=workspace.candidate_ref(experiment_id)
    ) as run_view:
        ctx.run_exp(
            worktree=run_view,
            experiment_id=experiment_id,
            task_ids=world.test_tasks,
            experiments_dir=ctx.experiments_dir,
            # Same baseline-derived budget for the veto panel: a test task the
            # baseline solved deterministically starts at 1 trial (§9 #7).
            trial_budget=budget_from_baseline(
                world.active_baseline,
                task_ids=world.test_tasks,
                full=ctx.config.task_trials,
            ),
        )


def _conclude(world: World, ctx: LoopContext) -> None:
    pending = world.pending
    assert pending is not None and pending.result is not None
    loop = pending.loop
    if loop.kind == "baseline":
        decision = Decision(kind="keep", reason="baseline at HEAD", verdicts={})
        write_loop(
            loop.model_copy(update={"decision": decision}), root=ctx.experiments_dir
        )
        return
    baseline = world.active_baseline
    assert baseline is not None  # a pending candidate always has a parent (§12)
    result = pending.result
    train = gate(result, baseline, task_ids=world.train_tasks, purpose="promotion")
    test_done = world.test_tasks <= result.tasks.keys()
    test = (
        gate(result, baseline, task_ids=world.test_tasks, purpose="regression_veto")
        if test_done
        else None
    )
    decision = combine(train, test)
    # STRICT ORDER (§7), idempotent for crash-replay (plan.md:505-513): C is made
    # reachable from a ref (FF on keep / failed-ref on discard) BEFORE the candidate
    # ref drops, and the decision is persisted LAST. A crash anywhere re-enters here
    # via rule 4. The whole ref dance is guarded by the candidate ref still existing:
    # once it has been dropped the dance is already done (HEAD is at C for a keep, C
    # is at failed/<id> for a discard), so replay skips straight to (re)writing the
    # decision instead of erroring on `rev-parse` of the missing ref.
    candidate = workspace.candidate_ref(loop.experiment_id)
    if repo.git_ref_exists(cwd=ctx.repo_root, ref=candidate):
        commit = repo.resolve_ref(candidate, cwd=ctx.repo_root)
        if decision.kind == "keep":
            workspace.fast_forward_primary(ctx.repo_root, commit=commit)
        else:
            workspace.set_failed_ref(
                ctx.repo_root, experiment_id=loop.experiment_id, commit=commit
            )
        workspace.drop_candidate_ref(ctx.repo_root, experiment_id=loop.experiment_id)
    write_loop(loop.model_copy(update={"decision": decision}), root=ctx.experiments_dir)


def _diagnosis_rejection(experiments_dir: Path, experiment_id: str) -> str | None:
    diagnosis = experiments_dir / experiment_id / DIAGNOSIS_FILENAME
    if not diagnosis.exists() or not diagnosis.read_text().strip():
        return f"write the per-cycle raw diagnosis log to {diagnosis} (non-empty)"
    return swap_learning_draft(experiments_dir)


def _diagnose(experiment_id: str, ctx: LoopContext) -> None:
    # Cheap turn in a throwaway full worktree at HEAD (keeps the primary read-only):
    # the agent writes the raw diagnosis.md + a learning.md rewrite draft, then the
    # loop validates + atomically swaps the draft. "done" = diagnosis.md exists, so
    # a crash mid-cycle re-enters cleanly (rule 6 only fires while it is absent).
    record_path = ExperimentResult.path(experiment_id, root=ctx.experiments_dir)
    diagnosis_path = ctx.experiments_dir / experiment_id / DIAGNOSIS_FILENAME
    diagnose_worktree = ctx.worktree_root / f"{experiment_id}-diagnose"
    with workspace.full_worktree(ctx.repo_root, diagnose_worktree, ref="HEAD") as wt:
        agent.run_turns_until(
            backend=ctx.backend,
            repo_root=wt,
            thread_id=None,
            prompt_builder=lambda note: build_diagnosis_prompt(
                program_md_path=ctx.program_md_path,
                learning_md_path=learning_memo_path(ctx.experiments_dir),
                experiment_json_path=record_path,
                diagnosis_path=diagnosis_path,
                draft_path=learning_draft_path(ctx.experiments_dir),
                feedback_note=note,
            ),
            check=lambda: _diagnosis_rejection(ctx.experiments_dir, experiment_id),
            max_attempts=agent.DEFAULT_MAX_PROPOSAL_ATTEMPTS,
        )


def execute(command: Command, world: World, ctx: LoopContext) -> None:
    """Perform the single side effect ``decide()`` chose. ``Halt`` is terminal and
    is handled by ``run_auto`` (it never reaches here)."""
    match command:
        case RefreshBaseline():
            _refresh_baseline(ctx)
        case ProposeAndLaunch():
            _propose_and_launch(world, ctx)
        case RunVeto(experiment_id=experiment_id):
            _run_veto(experiment_id, world, ctx)
        case Conclude():
            _conclude(world, ctx)
        case Diagnose(experiment_id=experiment_id):
            _diagnose(experiment_id, ctx)
        case Halt(reason=reason):
            raise AssertionError(f"execute() reached Halt: {reason}")


def run_auto(ctx: LoopContext) -> Halt:
    """Drive ``scan -> decide -> execute`` until a ``Halt`` (needs a human, §6).
    Returns the terminal ``Halt`` so the caller can report its reason. Each tick
    re-derives ``World`` from disk + git, so a crash anywhere resumes cleanly on the
    next invocation. A ``LoopCorruption`` from ``scan`` propagates (it is not a
    normal transition); the autonomous loop otherwise only stops at ``Halt``."""
    while True:
        world = scan(
            experiments_dir=ctx.experiments_dir,
            repo_root=ctx.repo_root,
            config=ctx.config,
        )
        command = decide(world)
        if isinstance(command, Halt):
            return command
        execute(command, world, ctx)
