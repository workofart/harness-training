from __future__ import annotations

import subprocess
from pathlib import Path


def _git_run(
    *args: str,
    cwd: Path | None = None,
    input: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=check,
        capture_output=True,
        text=True,
        input=input,
        cwd=cwd,
    )


def _git_stdout(*args: str, cwd: Path | None = None) -> str:
    completed = _git_run(*args, cwd=cwd)
    return completed.stdout


def git_path(path_name: str, *, cwd: Path | None = None) -> Path:
    resolved = Path(_git_stdout("rev-parse", "--git-path", path_name, cwd=cwd).strip())
    if resolved.is_absolute() or cwd is None:
        return resolved
    return cwd / resolved


def require_clean_worktree(*, cwd: Path | None = None) -> None:
    status_output = _git_stdout(
        "status",
        "--porcelain",
        "--untracked-files=all",
        cwd=cwd,
    ).strip()
    if status_output:
        raise RuntimeError(
            f"git worktree must be clean before starting experiment:\n{status_output}"
        )


def git_ref_exists(*, cwd: Path | None = None, ref: str) -> bool:
    completed = _git_run(
        "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", cwd=cwd, check=False
    )
    return completed.returncode == 0


def resolve_ref(ref: str, *, cwd: Path | None = None) -> str:
    return _git_stdout("rev-parse", ref, cwd=cwd).strip()


def is_dirty(*, cwd: Path | None = None) -> bool:
    """Whether the worktree has uncommitted changes (the predicate form of
    :func:`require_clean_worktree`). Respects ``.gitignore``, so the gitignored
    ``experiments/`` data dir never reads as dirty."""
    status = _git_stdout(
        "status", "--porcelain", "--untracked-files=all", cwd=cwd
    ).strip()
    return bool(status)


def _added_lines(diff_stdout: str) -> list[str]:
    """Extract the added (`+`) source lines from `git diff --unified=0` output,
    stripping the leading `+` and skipping the `+++` file header."""
    return [
        line[1:]
        for line in diff_stdout.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def git_diff_added_lines_worktree(
    *,
    cwd: Path | None = None,
    paths: tuple[str, ...],
) -> list[str]:
    """Added (`+`) lines in the uncommitted working-tree diff for `paths`:
    used to scan a candidate's not-yet-committed edits."""
    completed = _git_run("diff", "--unified=0", "--", *paths, cwd=cwd, check=False)
    if completed.returncode != 0:
        return []
    return _added_lines(completed.stdout)


def get_head_commit(*, cwd: Path | None = None) -> str:
    return resolve_ref("HEAD", cwd=cwd)


def changed_paths(*, cwd: Path | None = None) -> tuple[str, ...]:
    tracked_paths = _git_stdout(
        "diff",
        "--name-only",
        "--relative",
        "HEAD",
        cwd=cwd,
    ).splitlines()
    untracked_paths = _git_stdout(
        "ls-files",
        "--others",
        "--exclude-standard",
        cwd=cwd,
    ).splitlines()
    return tuple(
        sorted(
            {
                path.strip()
                for path in (*tracked_paths, *untracked_paths)
                if path.strip()
            }
        )
    )


def commit_all(message: str, *, cwd: Path | None = None) -> str:
    paths = changed_paths(cwd=cwd)
    if not paths:
        raise RuntimeError("candidate has no changes to commit")
    _git_run("add", "-A", "--", *paths, cwd=cwd)
    _git_run("commit", "-m", message, cwd=cwd)
    return get_head_commit(cwd=cwd)


def update_ref(ref_name: str, commit_hash: str, *, cwd: Path | None = None) -> None:
    _git_run("update-ref", ref_name, commit_hash, cwd=cwd)


def add_worktree(
    worktree_path: Path,
    *,
    cwd: Path | None = None,
    ref: str = "HEAD",
) -> None:
    _git_run("worktree", "add", "--force", "--detach", str(worktree_path), ref, cwd=cwd)


def remove_worktree(worktree_path: Path, *, cwd: Path | None = None) -> None:
    _git_run("worktree", "remove", "--force", str(worktree_path), cwd=cwd)


def merge_ff_only(commit_hash: str, *, cwd: Path | None = None) -> None:
    """Fast-forward the current branch to `commit_hash`, refusing a 3-way merge.

    The only HEAD-moving op the supervisor performs (on keep). `--ff-only` is the
    HEAD-drift guard: if HEAD diverged so `commit_hash` can't fast-forward onto
    it, git exits non-zero and this raises -- the caller Halts rather than merging
    onto the read-only primary (plan.md §6/§7).
    """
    _git_run("merge", "--ff-only", commit_hash, cwd=cwd)


def delete_ref(ref_name: str, *, cwd: Path | None = None) -> None:
    _git_run("update-ref", "-d", ref_name, cwd=cwd)


def sparse_checkout_init_no_cone(*, cwd: Path | None = None) -> None:
    _git_run("sparse-checkout", "init", "--no-cone", cwd=cwd)


def sparse_checkout_set(
    paths: tuple[str, ...],
    *,
    cwd: Path | None = None,
) -> None:
    _git_run(
        "sparse-checkout",
        "set",
        "--skip-checks",
        "--stdin",
        cwd=cwd,
        input="\n".join(paths) + "\n",
    )


def sparse_checkout_reapply(*, cwd: Path | None = None) -> None:
    _git_run("sparse-checkout", "reapply", cwd=cwd)


def ensure_info_exclude_entry(pattern: str, *, cwd: Path | None = None) -> None:
    exclude_path = git_path("info/exclude", cwd=cwd)
    existing = exclude_path.read_text() if exclude_path.exists() else ""
    lines = {line.strip() for line in existing.splitlines() if line.strip()}
    if pattern in lines:
        return
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    with exclude_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{pattern}\n")
