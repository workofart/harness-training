from __future__ import annotations

import subprocess
from pathlib import Path


def _git_run(
    *args: str,
    cwd: Path | None = None,
    input: str | None = None,
) -> subprocess.CompletedProcess[str]:
    kwargs = {
        "check": True,
        "capture_output": True,
        "text": True,
    }
    if input is not None:
        kwargs["input"] = input
    if cwd is not None:
        kwargs["cwd"] = cwd
    return subprocess.run(["git", *args], **kwargs)


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
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return completed.returncode == 0


def _added_lines(diff_stdout: str) -> list[str]:
    """Extract the added (`+`) source lines from `git diff --unified=0` output,
    stripping the leading `+` and skipping the `+++` file header."""
    return [
        line[1:]
        for line in diff_stdout.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def git_diff_added_lines_between(
    *,
    cwd: Path | None = None,
    base_ref: str,
    head_ref: str,
    paths: tuple[str, ...],
) -> list[str]:
    if not git_ref_exists(cwd=cwd, ref=base_ref):
        return []
    if not git_ref_exists(cwd=cwd, ref=head_ref):
        return []
    completed = subprocess.run(
        ["git", "diff", "--unified=0", base_ref, head_ref, "--", *paths],
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    if completed.returncode != 0:
        return []
    return _added_lines(completed.stdout)


def git_diff_added_lines_worktree(
    *,
    cwd: Path | None = None,
    paths: tuple[str, ...],
) -> list[str]:
    """Added (`+`) lines in the uncommitted working-tree diff for `paths`.

    The working-tree sibling of `git_diff_added_lines_between` (which diffs two
    commits): used to scan a candidate's not-yet-committed edits.
    """
    completed = subprocess.run(
        ["git", "diff", "--unified=0", "--", *paths],
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    if completed.returncode != 0:
        return []
    return _added_lines(completed.stdout)


def git_show_at_head(path: str, *, cwd: Path | None = None) -> str:
    """Return the contents of `path` as of HEAD (`git show HEAD:<path>`).

    Uses `check=False` so a missing/unreadable blob raises a RuntimeError that
    carries the captured stdout+stderr, rather than an opaque CalledProcessError.
    """
    completed = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"failed to read HEAD {path}:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed.stdout


def get_head_commit(*, cwd: Path | None = None) -> str:
    return _git_stdout("rev-parse", "HEAD", cwd=cwd).strip()


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


def hard_reset(commit_hash: str, *, cwd: Path | None = None) -> None:
    _git_run("reset", "--hard", commit_hash, cwd=cwd)


def update_ref(ref_name: str, commit_hash: str, *, cwd: Path | None = None) -> None:
    _git_run("update-ref", ref_name, commit_hash, cwd=cwd)


def add_worktree(
    worktree_path: Path,
    *,
    cwd: Path | None = None,
    ref: str = "HEAD",
    detach: bool = True,
) -> None:
    args = ["worktree", "add"]
    if detach:
        args.append("--detach")
    args.extend([str(worktree_path), ref])
    _git_run(*args, cwd=cwd)


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


def clean_untracked(*, cwd: Path | None = None, exclude: tuple[str, ...] = ()) -> None:
    args = ["clean", "-fd", *[f"--exclude={pattern}" for pattern in exclude]]
    _git_run(*args, cwd=cwd)
