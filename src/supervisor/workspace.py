"""Candidate isolation: a git ref + ephemeral throwaway worktrees (plan.md §7).

A candidate survives as ``refs/experiments/candidate/<id>`` -> commit ``C`` from
the moment it commits until ``Conclude``, so its code outlives any worktree. Each
operation runs in a *fresh* worktree that is removed on exit:

- the **sparse** edit worktree (a restricted ``VISIBLE_PATHS`` view at HEAD) where
  the agent edits ``core.py`` + its test -- omits the run machinery and config
  (which carries the literal task names), so the agent can't hardcode task ids it
  never sees, and ``EDITABLE ⊆ VISIBLE`` makes "can't edit what you can't see"
  structural;
- the **full** run worktree checked out at the candidate ref, where ``uv run exp``
  executes (the orchestrator needs all code).

No long-lived synced workspace, no primary hard-reset, no ``experiments/`` symlink
(``exp`` is handed ``--experiments-dir`` absolute, §12). The primary repo is
read-only except the single fast-forward on keep. This module owns only the git
worktree/ref lifecycle + the ``test_core`` gate; the pure rules it enforces live
in ``policy`` (``validate_candidate``/``VISIBLE_PATHS``/``EDITABLE_PATHS``).
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from src import repo
from src.supervisor.policy import CandidateDiff, EDITABLE_PATHS, VISIBLE_PATHS

# Paths scanned for added lines in the candidate diff (the editable surface). The
# task-id scan in policy.validate_candidate runs over these.
_DIFF_SCAN_PATHS: tuple[str, ...] = tuple(sorted(EDITABLE_PATHS))

# The agent writes its mechanism label here (one line) during the proposal turn;
# the loop reads it before committing (config is hidden, so focus_name can't live
# there anymore -- §6/§7). git-excluded in the edit worktree so it never enters
# the candidate diff (which would fail validate_candidate) or the commit.
FOCUS_FILE = ".candidate_focus"


def candidate_ref(experiment_id: str) -> str:
    return f"refs/experiments/candidate/{experiment_id}"


def failed_ref(experiment_id: str) -> str:
    return f"refs/experiments/failed/{experiment_id}"


@contextmanager
def sparse_worktree(
    repo_root: Path,
    worktree_path: Path,
    *,
    paths: frozenset[str] = VISIBLE_PATHS,
) -> Iterator[Path]:
    """A fresh detached worktree at HEAD with only ``paths`` checked out (the
    agent's sparse edit view). Removed on exit even if the body raises."""
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    repo.add_worktree(worktree_path, cwd=repo_root, ref="HEAD")
    try:
        repo.sparse_checkout_init_no_cone(cwd=worktree_path)
        repo.sparse_checkout_set(tuple(sorted(paths)), cwd=worktree_path)
        repo.sparse_checkout_reapply(cwd=worktree_path)
        # Keep the focus sentinel out of the diff/commit (see FOCUS_FILE).
        repo.ensure_info_exclude_entry(FOCUS_FILE, cwd=worktree_path)
        yield worktree_path
    finally:
        repo.remove_worktree(worktree_path, cwd=repo_root)


@contextmanager
def full_worktree(repo_root: Path, worktree_path: Path, *, ref: str) -> Iterator[Path]:
    """A fresh detached full worktree at ``ref`` (the candidate ref), where
    ``uv run exp`` runs. Removed on exit even if the body raises."""
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    repo.add_worktree(worktree_path, cwd=repo_root, ref=ref)
    try:
        yield worktree_path
    finally:
        repo.remove_worktree(worktree_path, cwd=repo_root)


def extract_candidate_diff(worktree_path: Path) -> CandidateDiff:
    """The candidate's change as pure data for ``policy.validate_candidate``:
    every changed path, plus the added lines under the editable surface."""
    changed_paths = repo.changed_paths(cwd=worktree_path)
    added_lines = repo.git_diff_added_lines_worktree(
        cwd=worktree_path, paths=_DIFF_SCAN_PATHS
    )
    return CandidateDiff(changed_paths=changed_paths, added_lines=tuple(added_lines))


@dataclass(frozen=True, slots=True)
class TestCoreResult:
    passed: bool
    output: str


def run_test_core(worktree_path: Path) -> TestCoreResult:
    """The cheap pre-experiment gate (§7 step 4): run the agent's own contract
    test inside its sparse view. A red result re-prompts before any expensive
    tracked run. Importable-by-construction: the view is the import closure of
    ``test_core.py``, so a green run proves the sparse checkout is complete."""
    completed = subprocess.run(
        ["uv", "run", "pytest", "tests/harness/test_core.py", "-q"],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    return TestCoreResult(
        passed=completed.returncode == 0,
        output=completed.stdout + completed.stderr,
    )


def read_focus_name(worktree_path: Path) -> str | None:
    """The mechanism label the agent wrote to ``FOCUS_FILE`` during its turn, or
    ``None`` if absent/empty (the loop re-prompts then). First non-empty line."""
    focus_path = worktree_path / FOCUS_FILE
    if not focus_path.exists():
        return None
    for line in focus_path.read_text().splitlines():
        if line.strip():
            return line.strip()
    return None


def commit_candidate(worktree_path: Path, *, experiment_id: str) -> str:
    """Commit the edit worktree's changes -> the candidate commit ``C``. Raises
    if the agent made no change (``repo.commit_all`` rejects an empty diff)."""
    return repo.commit_all(f"candidate {experiment_id}", cwd=worktree_path)


def set_candidate_ref(repo_root: Path, *, experiment_id: str, commit: str) -> None:
    repo.update_ref(candidate_ref(experiment_id), commit, cwd=repo_root)


def set_failed_ref(repo_root: Path, *, experiment_id: str, commit: str) -> None:
    """Preserve a discarded candidate's commit for diagnosis (§7)."""
    repo.update_ref(failed_ref(experiment_id), commit, cwd=repo_root)


def drop_candidate_ref(repo_root: Path, *, experiment_id: str) -> None:
    repo.delete_ref(candidate_ref(experiment_id), cwd=repo_root)


def fast_forward_primary(repo_root: Path, *, commit: str) -> None:
    """Fast-forward the primary HEAD to the kept candidate commit. Raises (-> the
    loop Halts) if HEAD diverged so ``commit`` can't fast-forward -- never a
    3-way merge onto the read-only primary (§6/§7)."""
    repo.merge_ff_only(commit, cwd=repo_root)
