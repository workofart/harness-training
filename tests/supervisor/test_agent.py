"""Tests for src/supervisor/agent.py (plan.md §6/§7).

The feedback-loop logic is exercised with a stub backend + an injected validator
(no real uv/git): happy path, re-prompt-with-feedback, attempt-cap exhaustion,
MissingThreadRollout (thread drop), TurnTimeout. ``validate_proposal``'s
composition is tested over a real temp worktree with ``run_test_core`` stubbed.
"""

from __future__ import annotations

import subprocess
from collections import deque
from pathlib import Path

import pytest

from src.supervisor import agent, workspace
from src.supervisor.agent import (
    ProposalRejected,
    build_prelaunch_prompt,
    propose_candidate,
)
from src.supervisor.agent_backend import MissingThreadRollout, TurnResult, TurnTimeout


# --- feedback loop (stub backend + injected validate) -----------------------


class _StubBackend:
    """Records prompts + resumed thread ids. Writes the focus sentinel into the
    worktree on a non-raising turn (so the success path can read it). ``raises``
    scripts per-turn exceptions (None = a normal turn)."""

    def __init__(self, *, focus: str = "stuck-detect-arm", raises=None) -> None:
        self.prompts: list[str] = []
        self.resumed: list[str | None] = []
        self._focus = focus
        self._raises = deque(raises or [])

    def run_turn(self, *, prompt, repo_root, thread_id=None, timeout_sec=600.0):
        self.prompts.append(prompt)
        self.resumed.append(thread_id)
        if self._raises:
            exc = self._raises.popleft()
            if exc is not None:
                raise exc
        (Path(repo_root) / workspace.FOCUS_FILE).write_text(self._focus + "\n")
        return TurnResult(thread_id="thread-new")


def _prompt(note: str | None) -> str:
    return f"prompt|note={note}"


def test_propose_returns_focus_on_first_valid_turn(tmp_path: Path) -> None:
    backend = _StubBackend(focus="majority-early-stop")
    result = propose_candidate(
        worktree_path=tmp_path,
        backend=backend,
        thread_id="thread-0",
        prompt_builder=_prompt,
        validate=lambda _wt: None,
    )
    assert result == agent.ProposedCandidate(
        thread_id="thread-new", focus_name="majority-early-stop"
    )
    assert backend.resumed == ["thread-0"]  # resumed the prior thread


def test_propose_reprompts_with_feedback_then_succeeds(tmp_path: Path) -> None:
    backend = _StubBackend()
    verdicts = deque(["needs a core.py change", None])
    result = propose_candidate(
        worktree_path=tmp_path,
        backend=backend,
        thread_id=None,
        prompt_builder=_prompt,
        validate=lambda _wt: verdicts.popleft(),
    )
    assert result.focus_name == "stuck-detect-arm"
    assert len(backend.prompts) == 2
    assert "note=None" in backend.prompts[0]  # first turn has no feedback
    assert "needs a core.py change" in backend.prompts[1]  # rejection fed back


def test_propose_raises_after_the_attempt_cap(tmp_path: Path) -> None:
    backend = _StubBackend()
    with pytest.raises(ProposalRejected, match="still rejected"):
        propose_candidate(
            worktree_path=tmp_path,
            backend=backend,
            thread_id=None,
            prompt_builder=_prompt,
            validate=lambda _wt: "still rejected",
            max_attempts=3,
        )
    assert len(backend.prompts) == 3


def test_propose_drops_thread_on_missing_rollout(tmp_path: Path) -> None:
    backend = _StubBackend(raises=[MissingThreadRollout("thread-0"), None])
    result = propose_candidate(
        worktree_path=tmp_path,
        backend=backend,
        thread_id="thread-0",
        prompt_builder=_prompt,
        validate=lambda _wt: None,
    )
    # First turn raised (stale rollout); the retry starts fresh (thread_id None).
    assert backend.resumed == ["thread-0", None]
    assert result.thread_id == "thread-new"


def test_propose_feeds_back_on_turn_timeout(tmp_path: Path) -> None:
    backend = _StubBackend(raises=[TurnTimeout("thread-0", 10.0), None])
    result = propose_candidate(
        worktree_path=tmp_path,
        backend=backend,
        thread_id="thread-0",
        prompt_builder=_prompt,
        validate=lambda _wt: None,
    )
    # The timeout is fed back and the same thread is resumed (not dropped).
    assert backend.resumed == ["thread-0", "thread-0"]
    assert "timed out" in backend.prompts[1]
    assert result.thread_id == "thread-new"


# --- validate_proposal composition (real temp worktree) ---------------------


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(root: Path) -> Path:
    root.mkdir(parents=True)
    _git("init", "-q", "-b", "main", cwd=root)
    _git("config", "user.email", "t@example.com", cwd=root)
    _git("config", "user.name", "tester", cwd=root)
    _git("config", "commit.gpgsign", "false", cwd=root)
    (root / "src" / "harness").mkdir(parents=True)
    (root / "tests" / "harness").mkdir(parents=True)
    (root / "src" / "harness" / "core.py").write_text("VALUE = 1\n")
    (root / "tests" / "harness" / "test_core.py").write_text(
        "def test_ok():\n    assert True\n"
    )
    (root / "program.md").write_text("# program\n")
    _git("add", "-A", cwd=root)
    _git("commit", "-q", "-m", "init", cwd=root)
    return root


_VIEW = frozenset({"src/harness/core.py", "tests/harness/test_core.py", "program.md"})


@pytest.fixture
def green_test_core(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        workspace, "run_test_core", lambda _wt: workspace.TestCoreResult(True, "ok")
    )


def test_validate_proposal_accepts_a_clean_candidate(
    tmp_path: Path, green_test_core
) -> None:
    primary = _init_repo(tmp_path / "primary")
    with workspace.sparse_worktree(primary, tmp_path / "wt", paths=_VIEW) as view:
        (view / "src" / "harness" / "core.py").write_text("VALUE = 2  # mechanism\n")
        (view / workspace.FOCUS_FILE).write_text("mechanism-x\n")
        assert agent.validate_proposal(view, task_ids=frozenset({"task-a"})) is None


def test_validate_proposal_rejects_a_non_editable_path(tmp_path: Path) -> None:
    primary = _init_repo(tmp_path / "primary")
    with workspace.sparse_worktree(primary, tmp_path / "wt", paths=_VIEW) as view:
        (view / "program.md").write_text("# program\nhand edit\n")  # not editable
        (view / workspace.FOCUS_FILE).write_text("x\n")
        message = agent.validate_proposal(view, task_ids=frozenset())
        assert message is not None and "program.md" in message


def test_validate_proposal_requires_a_core_change(
    tmp_path: Path, green_test_core
) -> None:
    primary = _init_repo(tmp_path / "primary")
    with workspace.sparse_worktree(primary, tmp_path / "wt", paths=_VIEW) as view:
        # Only the test changed -- editable, but no behavioral harness change.
        (view / "tests" / "harness" / "test_core.py").write_text(
            "def test_ok():\n    assert 1 == 1\n"
        )
        (view / workspace.FOCUS_FILE).write_text("x\n")
        message = agent.validate_proposal(view, task_ids=frozenset())
        assert message is not None and "src/harness/core.py" in message


def test_validate_proposal_requires_a_focus_label(
    tmp_path: Path, green_test_core
) -> None:
    primary = _init_repo(tmp_path / "primary")
    with workspace.sparse_worktree(primary, tmp_path / "wt", paths=_VIEW) as view:
        (view / "src" / "harness" / "core.py").write_text("VALUE = 3\n")
        message = agent.validate_proposal(view, task_ids=frozenset())
        assert message is not None and workspace.FOCUS_FILE in message


def test_validate_proposal_rejects_a_red_test_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        workspace, "run_test_core", lambda _wt: workspace.TestCoreResult(False, "boom")
    )
    primary = _init_repo(tmp_path / "primary")
    with workspace.sparse_worktree(primary, tmp_path / "wt", paths=_VIEW) as view:
        (view / "src" / "harness" / "core.py").write_text("VALUE = 4\n")
        (view / workspace.FOCUS_FILE).write_text("x\n")
        message = agent.validate_proposal(view, task_ids=frozenset())
        assert message is not None and "test_core.py is red" in message


# --- prompt -----------------------------------------------------------------


def test_build_prelaunch_prompt_lists_paths_and_feedback() -> None:
    prompt = build_prelaunch_prompt(
        program_md_path=Path("/repo/program.md"),
        learning_md_path=Path("/repo/experiments/learning.md"),
        evidence_paths=("/repo/experiments/exp-1/tasks/t/agent/steps.jsonl",),
        feedback_note="fix the path violation",
    )
    assert "Follow program.md." in prompt
    assert "/repo/program.md" in prompt
    assert "/repo/experiments/learning.md" in prompt
    assert "steps.jsonl" in prompt
    assert "fix the path violation" in prompt
    # config is hidden -- never named in the prompt.
    assert "harness_config.json" not in prompt
