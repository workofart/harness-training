"""Git policy for candidate capture, validation, and scratch worktrees."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from git import GitCommandError, Repo

if TYPE_CHECKING:
    from src.trainer.loss import Loss


class Parameter:
    """Storage for the harness: the main repo. .data is HEAD's sha; .grad holds the pending
    verdict; .applied is the sha the optimizer selected when it stepped."""

    def __init__(self, repo: Repo) -> None:
        self.repo = repo
        self.grad: Loss | None = None
        self.applied: str | None = None

    @property
    def data(self) -> str:
        return self.repo.head.commit.hexsha


class CandidateValidationError(RuntimeError):
    """A proposed candidate patch violates the training-cycle edit contract."""

    def __init__(
        self,
        message: str,
        *,
        cause: Literal["no_candidate", "invalid_candidate"],
    ) -> None:
        super().__init__(message)
        self.cause = cause


@dataclass(frozen=True)
class Candidate:
    """Captured candidate commit plus the baseline it is compared against."""

    commit: str
    base_commit: str


@contextlib.contextmanager
def scratch_worktree(
    repo: Repo,
    *,
    commit: str,
    root: Path,
    name: str,
    sparse: tuple[str, ...] = (),
) -> Iterator[Path]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}-{commit[:12]}"
    if path.exists():
        with contextlib.suppress(GitCommandError):
            repo.git.worktree("remove", "--force", str(path))
        repo.git.worktree("prune")
        if path.exists():
            shutil.rmtree(path)
    repo.git.worktree("add", "--no-checkout", "--detach", str(path), commit)
    try:
        # Sparse patterns land before the checkout populates the tree, so hidden
        # paths never materialize.
        worktree = Repo(path)
        if sparse:
            worktree.git.sparse_checkout("set", "--no-cone", *sparse)
        worktree.git.checkout()
        yield path
    finally:
        try:
            repo.git.worktree("remove", "--force", str(path))
        finally:
            repo.git.worktree("prune")


def capture_candidate(repo: Repo, *, base_commit: str) -> Candidate:
    """Stage + commit all proposer changes, pin the candidate ref, and return it.
    Raises if there is no staged change: an empty commit would mis-attribute a no-op
    as a candidate."""
    repo.git.reset("--soft", base_commit)
    repo.git.add("--all")
    staged = repo.git.diff("--cached", "--name-only")
    if not staged:
        raise CandidateValidationError(
            "candidate made no tracked change",
            cause="no_candidate",
        )
    repo.git.commit("--no-verify", "-m", "candidate")
    commit = repo.head.commit.hexsha
    repo.git.update_ref(f"refs/candidates/{commit}", commit)
    return Candidate(commit=commit, base_commit=base_commit)


def validate_candidate(
    repo: Repo,
    candidate: Candidate,
    *,
    surface: str,
    patch_paths: tuple[str, ...],
) -> None:
    parents = repo.commit(candidate.commit).parents
    if len(parents) != 1 or parents[0].hexsha != candidate.base_commit:
        raise CandidateValidationError(
            "candidate must be a direct child of its trainer baseline",
            cause="invalid_candidate",
        )
    changed_paths = tuple(
        path
        for path in repo.git.diff(
            "--name-only", candidate.base_commit, candidate.commit
        ).splitlines()
        if path
    )
    disallowed = tuple(path for path in changed_paths if path not in patch_paths)
    if disallowed:
        raise CandidateValidationError(
            "candidate changed files outside the allowed patch surface:\n"
            + "\n".join(disallowed),
            cause="invalid_candidate",
        )
    if surface not in changed_paths:
        raise CandidateValidationError(
            f"candidate must change {surface}; tests-only patches are not run",
            cause="invalid_candidate",
        )


SUITE_TIMEOUT_SEC = 900.0
_SUITE_TAIL_LINES = 30


def _run_suite(repo: Repo, commit: str, *, root: Path, name: str) -> str | None:
    """Trusted full-suite run of ``commit`` in a scratch worktree.

    Returns None when green, else a description of how it failed.
    """
    with scratch_worktree(repo, commit=commit, root=root, name=name) as path:
        try:
            # cwd+PYTHONPATH pin imports to the worktree; the editable .pth would test the invoker's tree.
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=path,
                env={**os.environ, "PYTHONPATH": str(path)},
                capture_output=True,
                text=True,
                timeout=SUITE_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            return f"test suite timed out after {SUITE_TIMEOUT_SEC:.0f}s"
    if proc.returncode == 0:
        return None
    tail = "\n".join(
        (proc.stdout + proc.stderr).strip().splitlines()[-_SUITE_TAIL_LINES:]
    )
    return "test suite failed:\n" + tail


def run_candidate_suite(repo: Repo, candidate: Candidate, *, root: Path) -> None:
    """Trusted full-suite run of the candidate tree; red or hung rejects it."""
    failure = _run_suite(repo, candidate.commit, root=root, name="check")
    if failure is None:
        return
    base_failure = _run_suite(repo, candidate.base_commit, root=root, name="base-check")
    if base_failure is not None:
        raise CandidateValidationError(
            f"baseline {base_failure}\n\n"
            f"This suite is already red at the base commit "
            f"{candidate.base_commit[:12]}, so the candidate did not cause it "
            "and no candidate can pass the gate. Fix the suite first.",
            cause="invalid_candidate",
        )
    raise CandidateValidationError(
        "candidate " + failure,
        cause="invalid_candidate",
    )
