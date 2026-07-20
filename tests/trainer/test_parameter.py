"""Tests for candidate git policy using throwaway repositories."""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from git import Repo
import pytest

from src.config import RunConfig
import src.trainer.parameter as parameter_module
from src.trainer.parameter import (
    Candidate,
    CandidateValidationError,
    capture_candidate,
    run_candidate_suite,
    scratch_worktree,
    validate_candidate,
)

# The shipped subject declaration: these tests exercise the real proposer
# visibility and patch surface against both synthetic repos and this checkout.
_TRAINING_TARGET = RunConfig.load(
    Path(__file__).resolve().parents[2] / "config/train_harness.yaml"
).training_target


RepoEnv = tuple[Path, Repo, str]


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], check=True, capture_output=True, text=True, cwd=cwd)


def _rev_parse(ref: str, *, cwd: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", ref], cwd=cwd, capture_output=True, text=True
    ).stdout.strip()


def _git_output(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _build_baseline_repo(root: Path) -> None:
    _git("init", "-q", cwd=root)
    _git("config", "user.email", "t@t.t", cwd=root)
    _git("config", "user.name", "t", cwd=root)
    (root / "file.txt").write_text("baseline\n")
    (root / ".gitignore").write_text("experiments/\n.venv\n")
    (root / "src" / "policy").mkdir(parents=True)
    (root / "src" / "trainer").mkdir(parents=True)
    (root / "tests" / "policy").mkdir(parents=True)
    (root / "tests" / "rollout").mkdir(parents=True)
    (root / "config").mkdir()
    (root / "scripts").mkdir()
    # Regular pkg like the real repo; namespace src loses imports to the editable install.
    (root / "src" / "__init__.py").write_text("")
    (root / "src" / "policy" / "core.py").write_text("VALUE = 1\n")
    for path in (
        "estimator.py",
        "loss.py",
        "optim.py",
        "parameter.py",
        "trainer.py",
    ):
        (root / "src" / "trainer" / path).write_text(f"# {path}\n")
    (root / "tests" / "policy" / "test_core_impl.py").write_text(
        "def test_ok():\n    assert True\n"
    )
    (root / "tests" / "policy" / "test_core_contracts.py").write_text(
        "def test_contract():\n    assert True\n"
    )
    (root / "tests" / "conftest.py").write_text("# conftest.py\n")
    (root / "tests" / "rollout" / "test_stub.py").write_text(
        "def test_stub():\n    assert True\n"
    )
    (root / "config" / "run_config.template.yaml").write_text("schema_version: 13\n")
    (root / "scripts" / "tool.py").write_text("# tool.py\n")
    (root / "scripts" / "evaluate.py").write_text("# evaluate.py\n")
    (root / "scripts" / "train.py").write_text("# train.py\n")
    (root / "AGENTS.md").write_text("agents\n")
    (root / "TECH_DESIGN.md").write_text("design\n")
    (root / "program.md").write_text("program\n")
    (root / "pyproject.toml").write_text("[project]\nname = 'test'\nversion = '0'\n")
    _git("add", "-A", cwd=root)
    _git("commit", "-q", "-m", "baseline", cwd=root)


_TEMPLATE_REPO: Path | None = None


def _template_repo() -> Path:
    """Build the baseline repo once per session; each test copies it (~5x cheaper
    than re-running git init+commit). The template is immutable — tests mutate only
    their own copy."""
    global _TEMPLATE_REPO
    if _TEMPLATE_REPO is None:
        template = Path(tempfile.mkdtemp(prefix="parameter-template-"))
        atexit.register(shutil.rmtree, template, ignore_errors=True)
        _build_baseline_repo(template)
        _TEMPLATE_REPO = template
    return _TEMPLATE_REPO


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    shutil.copytree(_template_repo(), root)
    return root


def _candidate_ref(candidate: Candidate) -> str:
    return f"refs/candidates/{candidate.commit}"


@pytest.fixture
def baseline_repo(tmp_path: Path) -> RepoEnv:
    """A fresh copy of the template repo plus its HEAD sha, collapsing the
    root/Repo/baseline preamble shared by most tests below."""
    root = _repo(tmp_path)
    repo = Repo(root)
    return root, repo, repo.head.commit.hexsha


def test_capture_candidate_advances_head_with_baseline_as_parent(
    baseline_repo: RepoEnv,
):
    root, repo, baseline = baseline_repo
    (root / "src" / "policy" / "core.py").write_text("VALUE = 2\n")
    candidate = capture_candidate(repo, base_commit=baseline)
    assert candidate.commit != baseline
    assert candidate.base_commit == baseline
    assert _rev_parse("HEAD", cwd=root) == candidate.commit
    assert _rev_parse("HEAD^", cwd=root) == baseline
    assert repo.git.status("--porcelain") == ""
    assert _git_output("show", "--name-only", "--format=", "HEAD", cwd=root) == (
        "src/policy/core.py"
    )
    assert repo.commit(_candidate_ref(candidate)).hexsha == candidate.commit


def test_capture_candidate_rebases_agent_commits_onto_trainer_baseline(
    baseline_repo: RepoEnv,
) -> None:
    root, repo, baseline = baseline_repo
    (root / "src" / "policy" / "core.py").write_text("VALUE = 2\n")
    repo.git.add("--all")
    repo.git.commit("-m", "agent commit")
    agent_commit = repo.head.commit.hexsha
    (root / "tests" / "policy" / "test_core_impl.py").write_text(
        "def test_ok():\n    assert 2 == 2\n"
    )

    candidate = capture_candidate(repo, base_commit=baseline)

    assert candidate.base_commit == baseline
    assert repo.commit(candidate.commit).parents[0].hexsha == baseline
    assert agent_commit != candidate.commit
    assert set(
        repo.git.diff("--name-only", baseline, candidate.commit).splitlines()
    ) == {
        "src/policy/core.py",
        "tests/policy/test_core_impl.py",
    }


def test_capture_candidate_raises_when_no_change(
    baseline_repo: RepoEnv,
):
    _root, repo, baseline = baseline_repo
    with pytest.raises(
        CandidateValidationError, match="candidate made no tracked change"
    ) as raised:
        capture_candidate(repo, base_commit=baseline)
    assert raised.value.cause == "no_candidate"


def test_validate_candidate_accepts_core_and_impl_test_changes(
    baseline_repo: RepoEnv,
) -> None:
    root, repo, baseline = baseline_repo
    (root / "src" / "policy" / "core.py").write_text("VALUE = 2\n")
    (root / "tests" / "policy" / "test_core_impl.py").write_text(
        "def test_ok():\n    assert 2 == 2\n"
    )

    candidate = capture_candidate(repo, base_commit=baseline)
    validate_candidate(
        repo,
        candidate,
        surface=_TRAINING_TARGET.surface,
        patch_paths=_TRAINING_TARGET.patch_paths,
    )


def test_validate_candidate_rejects_non_ancestral_candidate(
    baseline_repo: RepoEnv,
) -> None:
    root, repo, baseline = baseline_repo
    (root / "src" / "policy" / "core.py").write_text("VALUE = 2\n")
    repo.git.add("--all")
    repo.git.commit("-m", "agent commit")
    agent_commit = repo.head.commit.hexsha
    (root / "tests" / "policy" / "test_core_impl.py").write_text(
        "def test_ok():\n    assert 2 == 2\n"
    )
    candidate = capture_candidate(repo, base_commit=agent_commit)

    with pytest.raises(CandidateValidationError, match="direct child") as raised:
        validate_candidate(
            repo,
            Candidate(commit=candidate.commit, base_commit=baseline),
            surface=_TRAINING_TARGET.surface,
            patch_paths=_TRAINING_TARGET.patch_paths,
        )

    assert raised.value.cause == "invalid_candidate"


@pytest.mark.parametrize(
    "writes, match",
    [
        pytest.param(
            (("src/policy/core.py", "VALUE = 2\n"), ("README.md", "extra\n")),
            "README.md",
            id="rejects_disallowed_paths",
        ),
        pytest.param(
            (
                ("src/policy/core.py", "VALUE = 2\n"),
                (
                    "tests/policy/test_core_contracts.py",
                    "def test_contract():\n    assert False\n",
                ),
            ),
            "tests/policy/test_core_contracts.py",
            id="rejects_contract_test_edit",
        ),
        pytest.param(
            (
                (
                    "tests/policy/test_core_impl.py",
                    "def test_ok():\n    assert 2 == 2\n",
                ),
            ),
            "src/policy/core.py",
            id="requires_core_change",
        ),
    ],
)
def test_validate_candidate_rejects(
    baseline_repo: RepoEnv,
    writes: tuple[tuple[str, str], ...],
    match: str,
) -> None:
    root, repo, baseline = baseline_repo
    for rel, content in writes:
        (root / rel).write_text(content)

    candidate = capture_candidate(repo, base_commit=baseline)
    with pytest.raises(CandidateValidationError, match=match) as raised:
        validate_candidate(
            repo,
            candidate,
            surface=_TRAINING_TARGET.surface,
            patch_paths=_TRAINING_TARGET.patch_paths,
        )
    assert raised.value.cause == "invalid_candidate"
    # The candidate commit captures every mutated file, so the rejection is
    # about real captured content.
    for rel, content in writes:
        assert (
            _git_output("show", f"{candidate.commit}:{rel}", cwd=root)
            == content.strip()
        )


def test_scratch_worktree_is_detached_reachable_and_preserves_main_wip(
    baseline_repo: RepoEnv,
    tmp_path: Path,
) -> None:
    root, repo, baseline = baseline_repo
    worktrees = tmp_path / "scratch"
    expected = worktrees / f"propose-{baseline[:12]}"
    (root / "file.txt").write_text("developer WIP\n")
    (root / "untracked.txt").write_text("untracked WIP\n")
    main_status = repo.git.status("--porcelain")

    with scratch_worktree(
        repo, commit=baseline, root=worktrees, name="propose"
    ) as path:
        assert path == expected
        scratch = Repo(path)
        assert scratch.head.is_detached
        assert scratch.head.commit.hexsha == baseline
        assert repo.head.commit.hexsha == baseline
        assert repo.git.status("--porcelain") == main_status
        (path / "src" / "policy" / "core.py").write_text("VALUE = 2\n")
        scratch.git.add("--all")
        scratch.git.commit("-m", "scratch commit")
        scratch_commit = scratch.head.commit.hexsha
        assert repo.commit(scratch_commit).hexsha == scratch_commit

    assert not expected.exists()
    assert str(expected) not in repo.git.worktree("list", "--porcelain")
    assert repo.head.commit.hexsha == baseline
    assert repo.git.status("--porcelain") == main_status
    assert (root / "file.txt").read_text() == "developer WIP\n"


def test_sparse_worktree_materializes_only_proposer_surface(
    baseline_repo: RepoEnv,
    tmp_path: Path,
) -> None:
    root, repo, baseline = baseline_repo
    expected = tmp_path / "scratch" / f"propose-{baseline[:12]}"

    with scratch_worktree(
        repo,
        commit=baseline,
        root=tmp_path / "scratch",
        name="propose",
        sparse=_TRAINING_TARGET.proposer_visible,
    ) as path:
        scratch = Repo(path)
        assert scratch.head.is_detached
        assert scratch.head.commit.hexsha == baseline
        for hidden in (
            "src/trainer",
            "AGENTS.md",
            "TECH_DESIGN.md",
            "scripts",
        ):
            assert not (path / hidden).exists()
        for visible in (
            "src/policy",
            "tests/policy",
            "tests/rollout",
            "tests/conftest.py",
            "config/run_config.template.yaml",
            "program.md",
            "pyproject.toml",
            ".gitignore",
        ):
            assert (path / visible).exists()
        assert scratch.git.status("--porcelain") == ""

    assert not expected.exists()
    assert str(expected) not in repo.git.worktree("list", "--porcelain")


def test_capture_candidate_from_sparse_worktree_preserves_hidden_tree(
    baseline_repo: RepoEnv,
    tmp_path: Path,
) -> None:
    root, repo, baseline = baseline_repo

    with scratch_worktree(
        repo,
        commit=baseline,
        root=tmp_path / "scratch",
        name="propose",
        sparse=_TRAINING_TARGET.proposer_visible,
    ) as path:
        scratch = Repo(path)
        (path / ".venv").symlink_to(root / ".venv")
        assert scratch.git.status("--porcelain") == ""
        (path / "src" / "policy" / "core.py").write_text("VALUE = 2\n")
        candidate = capture_candidate(scratch, base_commit=baseline)

        assert (
            scratch.git.diff("--name-only", baseline, candidate.commit)
            == "src/policy/core.py"
        )
        assert repo.git.rev_parse(f"{baseline}:src/trainer") == repo.git.rev_parse(
            f"{candidate.commit}:src/trainer"
        )
        assert repo.git.rev_parse(f"{baseline}:scripts") == repo.git.rev_parse(
            f"{candidate.commit}:scripts"
        )
        assert scratch.git.status("--porcelain") == ""


@pytest.mark.xdist_group(name="shared_checkout")
def test_real_sparse_worktree_hides_gate_artifacts(tmp_path: Path) -> None:
    hidden = (
        "src/trainer/loss.py",
        "src/trainer/optim.py",
        "src/trainer/trainer.py",
        "src/trainer/estimator.py",
        "src/trainer/parameter.py",
        "scripts/evaluate.py",
        "scripts/train.py",
        "tests/trainer",
        "tests/test_evaluate.py",
        "TECH_DESIGN.md",
        "FUNCTIONALITY_MAP.md",
        "AGENTS.md",
    )
    root = Path(__file__).resolve().parents[2]
    repo = Repo(root)
    with scratch_worktree(
        repo,
        commit=repo.head.commit.hexsha,
        root=tmp_path / "scratch",
        name="propose",
        sparse=_TRAINING_TARGET.proposer_visible,
    ) as path:
        for artifact in hidden:
            assert not (path / artifact).exists()


@pytest.mark.xdist_group(name="shared_checkout")
def test_actual_sparse_propose_worktree_runs_full_suite(tmp_path: Path) -> None:
    # The proposer must be able to run and fix every test its patch can break.
    root = Path(__file__).resolve().parents[2]
    repo = Repo(root)
    with scratch_worktree(
        repo,
        commit=repo.head.commit.hexsha,
        root=tmp_path / "scratch",
        name="propose",
        sparse=_TRAINING_TARGET.proposer_visible,
    ) as path:
        (path / ".venv").symlink_to(root / ".venv")
        result = subprocess.run(
            [sys.executable, "-m", "pytest"],
            cwd=path,
            env={**os.environ, "PYTHONPATH": str(path)},
            capture_output=True,
            text=True,
        )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("sparse", [(), _TRAINING_TARGET.proposer_visible])
def test_scratch_worktree_same_path_can_be_reused(
    baseline_repo: RepoEnv,
    tmp_path: Path,
    sparse: tuple[str, ...],
) -> None:
    _root, repo, baseline = baseline_repo
    worktrees = tmp_path / "scratch"

    for _ in range(2):
        with scratch_worktree(
            repo, commit=baseline, root=worktrees, name="measure", sparse=sparse
        ) as path:
            assert path.exists()
            assert Repo(path).head.commit.hexsha == baseline
        assert not path.exists()


@pytest.mark.parametrize("sparse", [(), _TRAINING_TARGET.proposer_visible])
def test_scratch_worktree_tears_down_on_body_exception(
    baseline_repo: RepoEnv,
    tmp_path: Path,
    sparse: tuple[str, ...],
) -> None:
    _root, repo, baseline = baseline_repo
    worktrees = tmp_path / "scratch"
    expected = worktrees / f"measure-{baseline[:12]}"

    with pytest.raises(RuntimeError, match="boom"):
        with scratch_worktree(
            repo,
            commit=baseline,
            root=worktrees,
            name="measure",
            sparse=sparse,
        ):
            raise RuntimeError("boom")

    assert not expected.exists()
    assert str(expected) not in repo.git.worktree("list", "--porcelain")


@pytest.mark.parametrize("sparse", [(), _TRAINING_TARGET.proposer_visible])
def test_scratch_worktree_self_heals_registered_stale_path(
    baseline_repo: RepoEnv,
    tmp_path: Path,
    sparse: tuple[str, ...],
) -> None:
    _root, repo, baseline = baseline_repo
    worktrees = tmp_path / "scratch"
    stale = worktrees / f"measure-{baseline[:12]}"
    worktrees.mkdir()
    repo.git.worktree("add", "--detach", str(stale), baseline)
    (stale / "src" / "policy" / "core.py").write_text("stale dirty state\n")

    with scratch_worktree(
        repo, commit=baseline, root=worktrees, name="measure", sparse=sparse
    ) as path:
        assert path == stale
        assert (path / "src" / "policy" / "core.py").read_text() == "VALUE = 1\n"

    assert not stale.exists()


def test_candidate_ref_is_shared_with_main_and_survives_scratch_teardown(
    baseline_repo: RepoEnv,
    tmp_path: Path,
) -> None:
    root, repo, baseline = baseline_repo
    with scratch_worktree(
        repo, commit=baseline, root=tmp_path / "scratch", name="propose"
    ) as path:
        propose_repo = Repo(path)
        (path / "src" / "policy" / "core.py").write_text("VALUE = 2\n")
        candidate = capture_candidate(propose_repo, base_commit=baseline)
        assert repo.commit(_candidate_ref(candidate)).hexsha == candidate.commit

    assert _rev_parse("HEAD", cwd=root) == baseline
    assert repo.commit(_candidate_ref(candidate)).hexsha == candidate.commit


def test_invalid_candidate_ref_survives_scratch_teardown(
    baseline_repo: RepoEnv,
    tmp_path: Path,
) -> None:
    root, repo, baseline = baseline_repo
    with scratch_worktree(
        repo, commit=baseline, root=tmp_path / "scratch", name="propose"
    ) as path:
        propose_repo = Repo(path)
        (path / "src" / "policy" / "core.py").write_text("VALUE = 2\n")
        (path / "README.md").write_text("extra\n")
        candidate = capture_candidate(propose_repo, base_commit=baseline)
        with pytest.raises(CandidateValidationError, match="README.md"):
            validate_candidate(
                propose_repo,
                candidate,
                surface=_TRAINING_TARGET.surface,
                patch_paths=_TRAINING_TARGET.patch_paths,
            )

    assert _rev_parse("HEAD", cwd=root) == baseline
    assert repo.commit(_candidate_ref(candidate)).hexsha == candidate.commit


def test_run_candidate_suite_accepts_candidate_whose_own_tree_is_green(
    baseline_repo: RepoEnv,
    tmp_path: Path,
) -> None:
    root, repo, baseline = baseline_repo
    (root / "src" / "policy" / "core.py").write_text("VALUE = 2\n")
    # Green only against the candidate checkout, not the invoker's.
    (root / "tests" / "policy" / "test_core_impl.py").write_text(
        "from src.policy.core import VALUE\n\n\ndef test_ok():\n    assert VALUE == 2\n"
    )
    candidate = capture_candidate(repo, base_commit=baseline)

    run_candidate_suite(repo, candidate, root=tmp_path / "scratch")

    assert not (tmp_path / "scratch" / f"check-{candidate.commit[:12]}").exists()


def test_run_candidate_suite_rejects_red_tree_with_failure_tail(
    baseline_repo: RepoEnv,
    tmp_path: Path,
) -> None:
    root, repo, baseline = baseline_repo
    (root / "src" / "policy" / "core.py").write_text("VALUE = 2\n")
    (root / "tests" / "policy" / "test_core_impl.py").write_text(
        "def test_ok():\n    assert False, 'candidate broke the suite'\n"
    )
    candidate = capture_candidate(repo, base_commit=baseline)

    with pytest.raises(
        CandidateValidationError, match="candidate test suite failed"
    ) as raised:
        run_candidate_suite(repo, candidate, root=tmp_path / "scratch")

    assert raised.value.cause == "invalid_candidate"
    assert "candidate broke the suite" in str(raised.value)
    assert not (tmp_path / "scratch" / f"check-{candidate.commit[:12]}").exists()


def test_run_candidate_suite_blames_the_baseline_when_it_is_already_red(
    baseline_repo: RepoEnv,
    tmp_path: Path,
) -> None:
    # A shipped config referencing a deleted llm fragment left the suite red at
    # HEAD; every candidate was then rejected as `invalid_candidate` and two of
    # those tripped the epoch-skip breaker, 28 minutes in, blaming the proposal.
    root, repo, _baseline = baseline_repo
    (root / "tests" / "policy" / "test_core_impl.py").write_text(
        "def test_ok():\n    assert False, 'red before any candidate'\n"
    )
    repo.git.add("--all")
    repo.git.commit("-m", "red baseline")
    red_baseline = repo.head.commit.hexsha
    (root / "src" / "policy" / "core.py").write_text("VALUE = 2\n")
    (root / "tests" / "policy" / "test_core_impl.py").write_text(
        "def test_ok():\n    assert False, 'red only in candidate'\n"
    )
    candidate = capture_candidate(repo, base_commit=red_baseline)

    with pytest.raises(
        CandidateValidationError, match="baseline test suite failed"
    ) as raised:
        run_candidate_suite(repo, candidate, root=tmp_path / "scratch")

    assert raised.value.cause == "invalid_candidate"
    assert red_baseline[:12] in str(raised.value)
    assert "the candidate did not cause it" in str(raised.value)
    # The reported tail must be the baseline's own failure, not the candidate's.
    assert "red before any candidate" in str(raised.value)
    assert "red only in candidate" not in str(raised.value)


def test_run_candidate_suite_rejects_hung_suite(
    baseline_repo: RepoEnv,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, repo, baseline = baseline_repo
    (root / "src" / "policy" / "core.py").write_text("VALUE = 2\n")
    (root / "tests" / "policy" / "test_core_impl.py").write_text(
        "import time\n\n\ndef test_ok():\n    time.sleep(30)\n"
    )
    candidate = capture_candidate(repo, base_commit=baseline)
    monkeypatch.setattr(parameter_module, "SUITE_TIMEOUT_SEC", 1.0)

    with pytest.raises(CandidateValidationError, match="timed out") as raised:
        run_candidate_suite(repo, candidate, root=tmp_path / "scratch")

    assert raised.value.cause == "invalid_candidate"
    assert not (tmp_path / "scratch" / f"check-{candidate.commit[:12]}").exists()


def test_two_successive_captures_pin_two_distinct_refs(
    baseline_repo: RepoEnv,
) -> None:
    root, repo, baseline = baseline_repo
    (root / "src" / "policy" / "core.py").write_text("VALUE = 2\n")
    first = capture_candidate(repo, base_commit=baseline)
    (root / "src" / "policy" / "core.py").write_text("VALUE = 3\n")

    second = capture_candidate(repo, base_commit=first.commit)

    assert first.commit != second.commit
    assert repo.commit(_candidate_ref(first)).hexsha == first.commit
    assert repo.commit(_candidate_ref(second)).hexsha == second.commit
