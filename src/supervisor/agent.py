"""The candidate proposer (plan.md §6/§7).

Drives the codex/claude agent turn inside the sparse edit worktree (cwd=worktree,
resuming its thread), then runs the validate gate -- ``policy.validate_candidate``
(editable-paths + task-id scan), a behavioral-change check, the ``focus_name``
sentinel, and the cheap ``test_core`` run -- re-prompting with the rejection note
on any failure, capped. On success it returns the thread id + the ``focus_name``
the agent wrote; the loop then commits, refs, prewrites ``loop.json``, and launches
``exp``. The validator and prompt builder are injected so the feedback loop is
unit-testable without real ``uv``/git.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.supervisor import workspace
from src.supervisor.agent_backend import (
    AgentBackend,
    MissingThreadRollout,
    TurnTimeout,
)
from src.supervisor.policy import validate_candidate

# Re-prompt budget for one proposal: a candidate that cannot clear the validate
# gate within this many turns is a stuck agent, not friction -- raise for a human
# rather than loop forever (the old loop was uncapped; §6 says capped).
DEFAULT_MAX_PROPOSAL_ATTEMPTS = 5


@dataclass(frozen=True, slots=True)
class ProposedCandidate:
    thread_id: str
    focus_name: str


class ProposalRejected(RuntimeError):
    """The agent did not produce a launchable candidate within the attempt cap."""


class AgentTurnsExhausted(RuntimeError):
    """An agent feedback loop hit its attempt cap without ``check`` accepting.
    Carries the last rejection note as its message."""


def run_turns_until(
    *,
    backend: AgentBackend,
    repo_root: Path,
    thread_id: str | None,
    prompt_builder: Callable[[str | None], str],
    check: Callable[[], str | None],
    max_attempts: int,
) -> str:
    """Resume agent turns until ``check()`` accepts (returns ``None``) or the cap
    is hit. The capped feedback primitive shared by the prelaunch proposal and the
    post-run diagnosis (§6/§10): both run an agent turn, inspect a side effect, and
    re-prompt with a rejection note on failure.

    ``check`` runs after each completed turn; its non-``None`` return is the note
    fed into the next prompt. A ``MissingThreadRollout`` drops the thread so the
    next turn starts fresh; a ``TurnTimeout`` becomes feedback on the same thread --
    neither counts as a satisfied turn. Returns the final thread id; raises
    ``AgentTurnsExhausted`` (last note as message) when the cap is exhausted."""
    feedback_note: str | None = None
    current_thread_id = thread_id
    for _ in range(max_attempts):
        try:
            turn = backend.run_turn(
                prompt=prompt_builder(feedback_note),
                repo_root=repo_root,
                thread_id=current_thread_id,
            )
        except MissingThreadRollout as exc:
            if current_thread_id is None:
                raise
            current_thread_id = None  # rollout gone -> next turn starts fresh
            feedback_note = str(exc)
            continue
        except TurnTimeout as exc:
            resume = exc.thread_id or current_thread_id
            if resume is None:
                raise
            current_thread_id = resume
            feedback_note = str(exc)
            continue
        current_thread_id = turn.thread_id
        rejection = check()
        if rejection is None:
            return current_thread_id
        feedback_note = rejection
    raise AgentTurnsExhausted(feedback_note or "no turns attempted")


def build_prelaunch_prompt(
    *,
    program_md_path: Path,
    learning_md_path: Path,
    evidence_paths: tuple[str, ...],
    feedback_note: str | None = None,
) -> str:
    """The prelaunch turn prompt. Points the agent at the authoritative files by
    absolute path (config is *not* listed -- it is hidden, §7); program.md tells
    it to write its mechanism label to the focus sentinel."""
    lines = [
        "Autonomous prelaunch phase.",
        "Follow program.md.",
        "Read these authoritative files now:",
        f"- {program_md_path}",
        f"- {learning_md_path}",
    ]
    if evidence_paths:
        lines.append("- evidence artifact paths:")
        lines.extend(f"- {path}" for path in evidence_paths)
    if feedback_note is not None:
        lines.extend(
            ["", "Supervisor feedback from the previous prelaunch turn:", feedback_note]
        )
    return "\n".join(lines)


def validate_proposal(worktree_path: Path, *, task_ids: frozenset[str]) -> str | None:
    """First rejection message for the candidate in ``worktree_path``, or ``None``
    if it is launchable. Cheap structural checks before the expensive
    ``test_core`` run: editable-paths + task-id scan, a behavioral change in
    ``core.py``, a non-empty ``focus_name``, then the contract test green."""
    diff = workspace.extract_candidate_diff(worktree_path)
    path_or_taskid_error = validate_candidate(diff, task_ids=task_ids)
    if path_or_taskid_error is not None:
        return path_or_taskid_error
    if "src/harness/core.py" not in diff.changed_paths:
        return (
            "no behavioral change in src/harness/core.py (program.md: it is the "
            "main behavior surface; express the mechanism there with a focused test)"
        )
    if workspace.read_focus_name(worktree_path) is None:
        return (
            f"write a one-line mechanism label to {workspace.FOCUS_FILE} in the "
            "worktree root (it captures focus_name; it is not committed)"
        )
    test_core = workspace.run_test_core(worktree_path)
    if not test_core.passed:
        return "tests/harness/test_core.py is red:\n" + test_core.output[-2000:]
    return None


def propose_candidate(
    *,
    worktree_path: Path,
    backend: AgentBackend,
    thread_id: str | None,
    prompt_builder: Callable[[str | None], str],
    validate: Callable[[Path], str | None],
    max_attempts: int = DEFAULT_MAX_PROPOSAL_ATTEMPTS,
) -> ProposedCandidate:
    """Run agent turns until the candidate clears ``validate`` or the cap is hit.

    A thin specialization of ``run_turns_until``: the check is ``validate`` over
    the worktree, and on success it reads back the ``focus_name`` the agent wrote.
    Raises ``ProposalRejected`` (wrapping the last rejection) if the cap is
    exhausted."""
    try:
        final_thread_id = run_turns_until(
            backend=backend,
            repo_root=worktree_path,
            thread_id=thread_id,
            prompt_builder=prompt_builder,
            check=lambda: validate(worktree_path),
            max_attempts=max_attempts,
        )
    except AgentTurnsExhausted as exc:
        raise ProposalRejected(
            f"candidate not launchable after {max_attempts} attempts; "
            f"last rejection: {exc}"
        ) from exc
    focus_name = workspace.read_focus_name(worktree_path)
    assert focus_name is not None  # validate guarantees a non-empty label
    return ProposedCandidate(thread_id=final_thread_id, focus_name=focus_name)
