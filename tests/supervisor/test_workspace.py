"""Tests for src/supervisor/workspace.py (plan.md §7).

Real-git integration over a minimal throwaway repo for the worktree/diff/commit/
ref/fast-forward lifecycle; a monkeypatched ``run_test_core`` contract test; and
one behavioral closure test that builds the *real* repo's sparse view at HEAD and
runs ``test_core`` -- the durable VISIBLE_PATHS guard (import-complete by
construction, verified by behavior not static analysis).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src import repo
from src.supervisor import workspace
from src.supervisor.policy import VISIBLE_PATHS


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", "-b", "main", cwd=root)
    _git("config", "user.email", "t@example.com", cwd=root)
    _git("config", "user.name", "tester", cwd=root)
    _git("config", "commit.gpgsign", "false", cwd=root)
    (root / "src" / "harness").mkdir(parents=True)
    (root / "tests" / "harness").mkdir(parents=True)
    (root / "src" / "env").mkdir(parents=True)
    (root / "src" / "harness" / "core.py").write_text("VALUE = 1\n")
    (root / "tests" / "harness" / "test_core.py").write_text(
        "def test_ok():\n    assert True\n"
    )
    (root / "src" / "env" / "harbor.py").write_text("HARBOR = 1\n")  # non-visible
    (root / "program.md").write_text("# program\n")  # visible, not editable
    _git("add", "-A", cwd=root)
    _git("commit", "-q", "-m", "init", cwd=root)
    return root


_EDIT_VIEW = frozenset(
    {"src/harness/core.py", "tests/harness/test_core.py", "program.md"}
)


@pytest.fixture
def primary(tmp_path: Path) -> Path:
    return _init_repo(tmp_path / "primary")


# --- sparse worktree lifecycle ----------------------------------------------


def test_sparse_worktree_checks_out_only_requested_paths(primary: Path) -> None:
    wt = primary.parent / "wt-sparse"
    with workspace.sparse_worktree(primary, wt, paths=_EDIT_VIEW) as view:
        assert (view / "src" / "harness" / "core.py").exists()
        assert (view / "tests" / "harness" / "test_core.py").exists()
        # A path outside the requested view is not materialized.
        assert not (view / "src" / "env" / "harbor.py").exists()
    # The throwaway worktree is gone on exit.
    assert not wt.exists()


def test_sparse_worktree_is_removed_even_on_error(primary: Path) -> None:
    wt = primary.parent / "wt-error"
    with pytest.raises(RuntimeError, match="boom"):
        with workspace.sparse_worktree(primary, wt, paths=_EDIT_VIEW):
            raise RuntimeError("boom")
    assert not wt.exists()


def test_extract_candidate_diff_reports_changes(primary: Path) -> None:
    wt = primary.parent / "wt-diff"
    with workspace.sparse_worktree(primary, wt, paths=_EDIT_VIEW) as view:
        (view / "src" / "harness" / "core.py").write_text("VALUE = 2  # tweak\n")
        (view / "program.md").write_text("# program\nedited\n")  # not editable
        diff = workspace.extract_candidate_diff(view)
    # Every change is reported (validate_candidate -- in policy -- decides which
    # are allowed); added_lines scan covers only the editable surface.
    assert "src/harness/core.py" in diff.changed_paths
    assert "program.md" in diff.changed_paths
    assert any("VALUE = 2" in line for line in diff.added_lines)
    assert not any("edited" in line for line in diff.added_lines)


def test_focus_file_is_readable_but_excluded_from_diff_and_commit(
    primary: Path,
) -> None:
    wt = primary.parent / "wt-focus"
    with workspace.sparse_worktree(primary, wt, paths=_EDIT_VIEW) as view:
        (view / "src" / "harness" / "core.py").write_text("VALUE = 8\n")
        (view / workspace.FOCUS_FILE).write_text("stuck-detect-arm\n")
        diff = workspace.extract_candidate_diff(view)
        # The sentinel is read back...
        assert workspace.read_focus_name(view) == "stuck-detect-arm"
        # ...but never appears as a changed path (so validate_candidate, which
        # rejects non-editable paths, would not trip on it).
        assert workspace.FOCUS_FILE not in diff.changed_paths
        commit = workspace.commit_candidate(view, experiment_id="exp-focus")
    # The committed tree carries the edit but not the sentinel.
    files = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", commit],
        cwd=primary,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert "src/harness/core.py" in files
    assert workspace.FOCUS_FILE not in files


def test_read_focus_name_is_none_when_absent(primary: Path) -> None:
    wt = primary.parent / "wt-nofocus"
    with workspace.sparse_worktree(primary, wt, paths=_EDIT_VIEW) as view:
        assert workspace.read_focus_name(view) is None


# --- commit + refs ----------------------------------------------------------


def _commit_candidate(primary: Path, *, experiment_id: str, body: str) -> str:
    wt = primary.parent / f"wt-{experiment_id}"
    with workspace.sparse_worktree(primary, wt, paths=_EDIT_VIEW) as view:
        (view / "src" / "harness" / "core.py").write_text(body)
        commit = workspace.commit_candidate(view, experiment_id=experiment_id)
    workspace.set_candidate_ref(primary, experiment_id=experiment_id, commit=commit)
    return commit


def test_commit_candidate_sets_a_reachable_candidate_ref(primary: Path) -> None:
    commit = _commit_candidate(primary, experiment_id="exp-1", body="VALUE = 9\n")
    ref = workspace.candidate_ref("exp-1")
    assert repo.git_ref_exists(cwd=primary, ref=ref)
    resolved = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=primary,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert resolved == commit


def test_commit_candidate_rejects_an_empty_diff(primary: Path) -> None:
    wt = primary.parent / "wt-empty"
    with workspace.sparse_worktree(primary, wt, paths=_EDIT_VIEW) as view:
        with pytest.raises(RuntimeError, match="no changes"):
            workspace.commit_candidate(view, experiment_id="exp-empty")


def test_full_worktree_checks_out_the_candidate_ref(primary: Path) -> None:
    _commit_candidate(primary, experiment_id="exp-2", body="VALUE = 42\n")
    run_wt = primary.parent / "wt-run"
    with workspace.full_worktree(
        primary, run_wt, ref=workspace.candidate_ref("exp-2")
    ) as view:
        # Full (not sparse): all tracked files present, with the candidate edit.
        assert (view / "src" / "env" / "harbor.py").exists()
        assert (view / "src" / "harness" / "core.py").read_text() == "VALUE = 42\n"
    assert not run_wt.exists()


# --- conclude: fast-forward / failed-ref / drop -----------------------------


def test_fast_forward_primary_advances_head_to_the_candidate(primary: Path) -> None:
    before = repo.get_head_commit(cwd=primary)
    commit = _commit_candidate(primary, experiment_id="exp-keep", body="VALUE = 7\n")
    assert commit != before
    workspace.fast_forward_primary(primary, commit=commit)
    assert repo.get_head_commit(cwd=primary) == commit


def test_fast_forward_primary_raises_when_head_diverged(primary: Path) -> None:
    # A candidate off the original HEAD...
    commit = _commit_candidate(primary, experiment_id="exp-drift", body="VALUE = 3\n")
    # ...then the primary HEAD moves elsewhere (a human commit), so the candidate
    # can no longer fast-forward: the merge is refused (-> the loop Halts), never
    # a 3-way merge onto the read-only primary.
    (primary / "program.md").write_text("# program\nhuman edit\n")
    _git("commit", "-q", "-am", "human", cwd=primary)
    with pytest.raises(subprocess.CalledProcessError):
        workspace.fast_forward_primary(primary, commit=commit)


def test_failed_ref_preserves_commit_then_candidate_ref_drops(primary: Path) -> None:
    commit = _commit_candidate(primary, experiment_id="exp-discard", body="VALUE = 5\n")
    workspace.set_failed_ref(primary, experiment_id="exp-discard", commit=commit)
    workspace.drop_candidate_ref(primary, experiment_id="exp-discard")
    # The candidate ref is gone but the commit survives for diagnosis.
    assert not repo.git_ref_exists(
        cwd=primary, ref=workspace.candidate_ref("exp-discard")
    )
    assert repo.git_ref_exists(cwd=primary, ref=workspace.failed_ref("exp-discard"))


# --- test_core gate ---------------------------------------------------------


def test_run_test_core_maps_return_code_and_captures_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_run(args, *, cwd, capture_output, text):
        calls["args"] = args
        calls["cwd"] = cwd
        return subprocess.CompletedProcess(
            args=args, returncode=1, stdout="out\n", stderr="err\n"
        )

    monkeypatch.setattr(workspace.subprocess, "run", fake_run)
    result = workspace.run_test_core(Path("/some/wt"))
    assert calls["args"] == ["uv", "run", "pytest", "tests/harness/test_core.py", "-q"]
    assert result.passed is False
    assert "out" in result.output and "err" in result.output


# --- the durable VISIBLE_PATHS closure guard (real repo) --------------------


def test_real_sparse_view_at_head_is_import_complete(tmp_path: Path) -> None:
    # §7: the sparse view is the import closure of tests/harness/test_core.py.
    # Build it at the real repo's HEAD and run test_core -- a green run proves the
    # checkout is complete (behavior, not static analysis). Also confirms config
    # (the literal task names) is omitted, so the agent can't see/hardcode them.
    repo_root = Path(__file__).resolve().parents[2]
    view_path = tmp_path / "sparse-head"
    with workspace.sparse_worktree(repo_root, view_path) as view:
        assert all((view / path).exists() for path in VISIBLE_PATHS)
        assert not (view / "config" / "harness_config.json").exists()
        result = workspace.run_test_core(view)
    assert result.passed, result.output[-2000:]
