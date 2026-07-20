from __future__ import annotations

import atexit
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from conftest import TEST_MEASUREMENT_IDENTITY
from src.config import TrainingTargetConfig
from src.llm.backend import AgentBackend, Emit, TurnResult
from src.measurement import PreflightError
from src.trainer.estimator import (
    AgenticEstimator,
    LEARNING_MEMO_MAX_LINES,
)
from src.rollout.records import ExperimentResult
from src.rollout.store import RunStore

_TARGET = TrainingTargetConfig(
    module="src.policy.core",
    extra_patch_paths=("tests/policy/test_core_impl.py",),
)


def _experiment_result(experiment_id: str = "exp-1") -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        git_commit_hash="c" * 40,
        measurement_identity=TEST_MEASUREMENT_IDENTITY,
        git_dirty=False,
        config_path="config/run.json",
        started_at=datetime.now(UTC),
        tasks={},
    )


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], check=True, capture_output=True, text=True, cwd=cwd)


def _rev_parse(ref: str, *, cwd: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", ref], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _build_baseline_repo(root: Path) -> None:
    _git("init", "-q", cwd=root)
    _git("config", "user.email", "t@t.t", cwd=root)
    _git("config", "user.name", "t", cwd=root)
    (root / ".gitignore").write_text("experiments/\n")
    (root / "src" / "policy").mkdir(parents=True)
    (root / "src" / "policy" / "core.py").write_text("VALUE = 1\n")
    _git("add", "-A", cwd=root)
    _git("commit", "-q", "-m", "baseline", cwd=root)


_TEMPLATE_REPO: Path | None = None


def _template_repo() -> Path:
    global _TEMPLATE_REPO
    if _TEMPLATE_REPO is None:
        template = Path(tempfile.mkdtemp(prefix="estimator-template-"))
        atexit.register(shutil.rmtree, template, ignore_errors=True)
        _build_baseline_repo(template)
        _TEMPLATE_REPO = template
    return _TEMPLATE_REPO


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    shutil.copytree(_template_repo(), root)
    return root


class FakeProposeBackend(AgentBackend):
    def __init__(
        self,
        *,
        write_patch: bool = True,
        write_disallowed_patch: bool = False,
    ) -> None:
        self.write_patch = write_patch
        self.write_disallowed_patch = write_disallowed_patch
        self.prompt: str | None = None
        self.repo_root: Path | None = None
        self.thread_id: str | None = None

    def run_turn(
        self,
        *,
        prompt: str,
        repo_root: Path,
        emit: Emit,
        thread_id: str | None = None,
    ) -> TurnResult:
        self.prompt = prompt
        self.repo_root = repo_root
        self.thread_id = thread_id
        emit("fake progress")
        if self.write_patch:
            (repo_root / "src" / "policy" / "core.py").write_text("VALUE = 2\n")
        if self.write_disallowed_patch:
            (repo_root / "README.md").write_text("extra\n")
        return TurnResult(
            thread_id="candidate-thread",
            progress_summary="6m04s · agent: cmd 24 / edit 1",
        )


class FakeDiagnosisBackend(AgentBackend):
    def __init__(self, drafts: list[str]) -> None:
        self._drafts = list(drafts)
        self.prompts: list[str] = []
        self.thread_ids: list[str | None] = []

    def run_turn(
        self,
        *,
        prompt: str,
        repo_root: Path,
        emit: Emit,
        thread_id: str | None = None,
    ) -> TurnResult:
        del repo_root, emit
        self.prompts.append(prompt)
        self.thread_ids.append(thread_id)
        draft_path = _draft_path_from_prompt(prompt)
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        draft_path.write_text(self._drafts.pop(0))
        return TurnResult(
            thread_id=f"diagnosis-{len(self.prompts)}",
            progress_summary="3m07s · agent: cmd 15 / msg 4",
        )


def test_propose_runs_one_agent_turn_and_leaves_patch_in_worktree(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _repo(tmp_path)
    backend = FakeProposeBackend()
    baseline = _rev_parse("HEAD", cwd=root)

    result = AgenticEstimator(backend=backend).propose(
        repo_root=root,
        tracker=RunStore(tmp_path / "store"),
        target=_TARGET,
    )

    assert result is None
    assert _rev_parse("HEAD", cwd=root) == baseline
    assert (root / "src" / "policy" / "core.py").read_text() == "VALUE = 2\n"
    assert backend.repo_root == root
    assert backend.thread_id is None
    assert capsys.readouterr().out == (
        "propose   · fake progress\npropose   · done · 6m04s · agent: cmd 24 / edit 1\n"
    )


def test_propose_stages_evidence_into_worktree_and_points_agent_there(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    store = RunStore(tmp_path / "store")
    store.root.mkdir()
    (store.root / "learning.md").write_text("prior memo\n")
    store.runs_index_path().write_text('{"experiment_id": "e0"}\n')
    backend = FakeProposeBackend()

    AgenticEstimator(backend=backend).propose(
        repo_root=root, tracker=store, target=_TARGET
    )

    # Point the confined agent at worktree-local evidence copies, never the shared store.
    assert backend.prompt is not None
    assert str(root / "program.md") in backend.prompt
    assert str(root / "experiments" / "learning.md") in backend.prompt
    assert str(root / "experiments" / "runs.jsonl") in backend.prompt
    # The config-resolved target reaches the prompt; program.md never names files.
    assert (
        "Editable files (machine-enforced): src/policy/core.py, "
        "tests/policy/test_core_impl.py. The candidate MUST change "
        "src/policy/core.py." in backend.prompt
    )
    assert "run the test command program.md's 'Patch' step names" in backend.prompt
    assert "Stop once the candidate patch is present" in backend.prompt
    assert (root / "experiments" / "learning.md").read_text() == "prior memo\n"
    assert (
        root / "experiments" / "runs.jsonl"
    ).read_text() == '{"experiment_id": "e0"}\n'


def test_propose_allows_no_change_for_trainer_to_reject(tmp_path: Path) -> None:
    root = _repo(tmp_path)

    result = AgenticEstimator(backend=FakeProposeBackend(write_patch=False)).propose(
        repo_root=root,
        tracker=RunStore(tmp_path / "store"),
        target=_TARGET,
    )

    assert result is None
    assert (
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        == ""
    )


def test_propose_leaves_disallowed_patch_for_trainer_to_reject(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    baseline = _rev_parse("HEAD", cwd=root)

    result = AgenticEstimator(
        backend=FakeProposeBackend(write_disallowed_patch=True)
    ).propose(
        repo_root=root,
        tracker=RunStore(tmp_path / "store"),
        target=_TARGET,
    )

    assert result is None
    assert _rev_parse("HEAD", cwd=root) == baseline
    assert (root / "README.md").read_text() == "extra\n"


def test_diagnose_publishes_valid_learning_draft(tmp_path: Path) -> None:
    backend = FakeDiagnosisBackend(["Current bottleneck\n"])
    store = RunStore(tmp_path / "store")

    result = AgenticEstimator(backend=backend).diagnose(
        _experiment_result(),
        repo_root=tmp_path,
        tracker=store,
        target=_TARGET,
    )

    assert result is None
    # The harness publishes the worktree-local draft across the store boundary.
    assert (store.root / "learning.md").read_text() == "Current bottleneck\n"
    assert backend.thread_ids == [None]


def test_diagnose_reprompts_rejected_draft(tmp_path: Path) -> None:
    backend = FakeDiagnosisBackend(["\n", "Research leads\n"])
    store = RunStore(tmp_path / "store")

    result = AgenticEstimator(backend=backend).diagnose(
        _experiment_result(),
        repo_root=tmp_path,
        tracker=store,
        target=_TARGET,
    )

    assert result is None
    assert (store.root / "learning.md").read_text() == "Research leads\n"
    assert "learning draft is empty" in backend.prompts[1]
    assert backend.thread_ids == [None, "diagnosis-1"]


def test_diagnose_raises_after_attempt_cap(tmp_path: Path) -> None:
    long_draft = "\n".join("line" for _ in range(LEARNING_MEMO_MAX_LINES + 1))

    with pytest.raises(RuntimeError, match="over the 150-line limit"):
        AgenticEstimator(
            backend=FakeDiagnosisBackend([long_draft, long_draft])
        ).diagnose(
            _experiment_result(),
            repo_root=tmp_path,
            tracker=RunStore(tmp_path / "store"),
            target=_TARGET,
        )


def test_diagnose_prompt_lists_program_and_staged_artifacts(tmp_path: Path) -> None:
    backend = FakeDiagnosisBackend(["Current bottleneck\n"])

    AgenticEstimator(backend=backend).diagnose(
        _experiment_result(),
        repo_root=tmp_path,
        tracker=RunStore(tmp_path / "store"),
        target=_TARGET,
    )

    [prompt] = backend.prompts
    assert str(tmp_path / "program.md") in prompt
    assert "The editable policy surface under training is src/policy/core.py." in prompt
    assert str(tmp_path / "experiments" / "exp-1" / "experiment.json") in prompt
    assert str(tmp_path / "experiments" / "runs.jsonl") in prompt
    assert str(tmp_path / "experiments" / "learning.md") in prompt
    assert str(tmp_path / "experiments" / "learning.draft.md") in prompt


def _draft_path_from_prompt(prompt: str) -> Path:
    marker = "Write the full rewritten learning memo to: "
    [line] = [line for line in prompt.splitlines() if line.startswith(marker)]
    return Path(line.removeprefix(marker))


def test_construction_hard_fails_when_the_backend_is_not_ready() -> None:
    class _UnreadyBackend(FakeProposeBackend):
        def _assert_ready(self) -> None:
            raise RuntimeError("claude turn failed (rc=1): not authenticated")

    with pytest.raises(PreflightError, match="not authenticated"):
        AgenticEstimator(backend=_UnreadyBackend())
