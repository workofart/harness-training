from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.control.agent_backend import (
    ClaudeBackend,
    CodexBackend,
    MissingThreadRollout,
    supervisor_root_for_repo,
)


class FakeStream:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)


def test_supervisor_root_for_repo_uses_sibling_dir_named_after_repo(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "research"
    repo_root.mkdir()

    assert supervisor_root_for_repo(repo_root) == tmp_path / "research_supervisor"


def test_run_codex_turn_sets_supervisor_codex_home_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import src.control.agent_backend as agent_backend_mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    codex_home = tmp_path / "research_supervisor" / "codex-home"
    codex_home.mkdir(parents=True)
    calls: dict[str, object] = {}

    class FakeProcess:
        def __init__(self, command, *, cwd, env) -> None:
            calls["command"] = command
            calls["cwd"] = cwd
            calls["env"] = env
            self.stdout = FakeStream(
                ['{"type":"thread.started","thread_id":"thread-1"}\n']
            )
            self.stderr = FakeStream([])
            self.returncode = 0

        def wait(self) -> int:
            return self.returncode

    def fake_popen(command, *, stdout, stderr, text, cwd, env):
        assert stdout == subprocess.PIPE
        assert stderr == subprocess.PIPE
        assert text is True
        return FakeProcess(command, cwd=cwd, env=env)

    monkeypatch.setattr(agent_backend_mod.subprocess, "Popen", fake_popen)

    backend = CodexBackend(binary="codex", codex_home=codex_home)
    result = backend.run_turn(
        prompt="hello",
        repo_root=repo_root,
    )

    assert result.thread_id == "thread-1"
    assert calls["cwd"] == repo_root
    assert calls["env"]["CODEX_HOME"] == str(codex_home.resolve())


def test_run_codex_turn_provisions_missing_supervisor_codex_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import src.control.agent_backend as agent_backend_mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    user_home = tmp_path / "home"
    user_codex_home = user_home / ".codex"
    user_codex_home.mkdir(parents=True)
    (user_codex_home / "auth.json").write_text("{}\n")
    (user_codex_home / "config.toml").write_text("model = 'gpt-5'\n")
    monkeypatch.setenv("HOME", str(user_home))
    codex_home = tmp_path / "repo_supervisor" / "codex-home"

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = FakeStream(
                ['{"type":"thread.started","thread_id":"thread-1"}\n']
            )
            self.stderr = FakeStream([])
            self.returncode = 0

        def wait(self) -> int:
            return self.returncode

    def fake_popen(command, *, stdout, stderr, text, cwd, env):
        assert codex_home.is_dir()
        assert (codex_home / "auth.json").resolve() == (user_codex_home / "auth.json")
        assert (codex_home / "config.toml").resolve() == (
            user_codex_home / "config.toml"
        )
        return FakeProcess()

    monkeypatch.setattr(agent_backend_mod.subprocess, "Popen", fake_popen)

    backend = CodexBackend(binary="codex", codex_home=codex_home)
    result = backend.run_turn(
        prompt="hello",
        repo_root=repo_root,
    )

    assert result.thread_id == "thread-1"


def test_run_codex_turn_relinks_stale_supervisor_codex_home_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import src.control.agent_backend as agent_backend_mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    user_home = tmp_path / "home"
    user_codex_home = user_home / ".codex"
    user_codex_home.mkdir(parents=True)
    (user_codex_home / "auth.json").write_text("{}\n")
    (user_codex_home / "config.toml").write_text("model = 'gpt-5'\n")
    monkeypatch.setenv("HOME", str(user_home))
    codex_home = tmp_path / "repo_supervisor" / "codex-home"
    codex_home.mkdir(parents=True)
    stale_home = tmp_path / "old_supervisor" / "codex-home"
    stale_home.mkdir(parents=True)
    stale_auth = stale_home / "auth.json"
    stale_auth.write_text("{}\n")
    broken_config = tmp_path / "deleted-config.toml"
    (codex_home / "auth.json").symlink_to(stale_auth)
    (codex_home / "config.toml").symlink_to(broken_config)

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = FakeStream(
                ['{"type":"thread.started","thread_id":"thread-1"}\n']
            )
            self.stderr = FakeStream([])
            self.returncode = 0

        def wait(self) -> int:
            return self.returncode

    def fake_popen(command, *, stdout, stderr, text, cwd, env):
        assert (codex_home / "auth.json").resolve() == (user_codex_home / "auth.json")
        assert (codex_home / "config.toml").resolve() == (
            user_codex_home / "config.toml"
        )
        return FakeProcess()

    monkeypatch.setattr(agent_backend_mod.subprocess, "Popen", fake_popen)

    backend = CodexBackend(binary="codex", codex_home=codex_home)
    result = backend.run_turn(
        prompt="hello",
        repo_root=repo_root,
    )

    assert result.thread_id == "thread-1"


def test_run_codex_turn_prints_agent_and_toolcall_logs_to_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import src.control.agent_backend as agent_backend_mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    codex_home = tmp_path / "research_supervisor" / "codex-home"
    codex_home.mkdir(parents=True)

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = FakeStream(
                [
                    '{"type":"thread.started","thread_id":"thread-1"}\n',
                    '{"type":"item.started","item":{"type":"command_execution","command":"echo hello"}}\n',
                    '{"type":"item.completed","item":{"type":"agent_message","text":"done now"}}\n',
                ]
            )
            self.stderr = FakeStream(["stderr line\n"])
            self.returncode = 0

        def wait(self) -> int:
            return self.returncode

    def fake_popen(command, *, stdout, stderr, text, cwd, env):
        assert stdout == subprocess.PIPE
        assert stderr == subprocess.PIPE
        assert text is True
        return FakeProcess()

    monkeypatch.setattr(agent_backend_mod.subprocess, "Popen", fake_popen)

    backend = CodexBackend(binary="codex", codex_home=codex_home)
    result = backend.run_turn(
        prompt="hello",
        repo_root=repo_root,
    )

    captured = capsys.readouterr()
    assert result.thread_id == "thread-1"
    assert "[codex]" in captured.out
    assert "[toolcall] cmd> echo hello" in captured.out
    assert "[codex] done now" in captured.out
    assert "[agent]" not in captured.out
    assert "done now" in captured.out
    assert "[codex stderr]" in captured.err
    assert "stderr line" in captured.err


def test_run_claude_turn_prints_agent_and_toolcall_logs_to_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import src.control.agent_backend as agent_backend_mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    settings_path = tmp_path / "claude-settings.json"

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = FakeStream(
                [
                    '{"type":"system","subtype":"init","session_id":"session-1"}\n',
                    '{"type":"assistant","message":{"content":[{"type":"text","text":"thinking now"}]}}\n',
                    '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Bash","input":{"command":"pwd"}}]}}\n',
                    '{"type":"result","subtype":"success","session_id":"session-1"}\n',
                ]
            )
            self.stderr = FakeStream([])
            self.returncode = 0

        def wait(self) -> int:
            return self.returncode

    def fake_popen(command, *, stdout, stderr, text, cwd, env):
        assert stdout == subprocess.PIPE
        assert stderr == subprocess.PIPE
        assert text is True
        return FakeProcess()

    monkeypatch.setattr(agent_backend_mod.subprocess, "Popen", fake_popen)

    backend = ClaudeBackend(binary="claude", settings_path=settings_path)
    result = backend.run_turn(
        prompt="hello",
        repo_root=repo_root,
    )

    captured = capsys.readouterr()
    assert result.thread_id == "session-1"
    assert "[claude] thinking now" in captured.out
    assert "[toolcall] cmd> pwd" in captured.out
    assert "[agent]" not in captured.out


def test_run_codex_turn_reports_missing_resume_thread(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import src.control.agent_backend as agent_backend_mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    codex_home = tmp_path / "research_supervisor" / "codex-home"
    codex_home.mkdir(parents=True)

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = FakeStream([])
            self.stderr = FakeStream(
                [
                    "Error: thread/resume: thread/resume failed: "
                    "no rollout found for thread id missing-thread (code -32600)\n"
                ]
            )
            self.returncode = 1

        def wait(self) -> int:
            return self.returncode

    def fake_popen(command, *, stdout, stderr, text, cwd, env):
        assert command[:4] == ["codex", "exec", "resume", "missing-thread"]
        assert stdout == subprocess.PIPE
        assert stderr == subprocess.PIPE
        assert text is True
        return FakeProcess()

    monkeypatch.setattr(agent_backend_mod.subprocess, "Popen", fake_popen)

    backend = CodexBackend(binary="codex", codex_home=codex_home)
    with pytest.raises(MissingThreadRollout, match="missing-thread"):
        backend.run_turn(
            prompt="hello",
            repo_root=repo_root,
            thread_id="missing-thread",
        )


def test_run_claude_turn_reports_missing_resume_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import src.control.agent_backend as agent_backend_mod

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    settings_path = tmp_path / "claude-settings.json"

    class FakeProcess:
        def __init__(self) -> None:
            self.stdout = FakeStream([])
            self.stderr = FakeStream(
                ["Error: No conversation found with session ID missing-session\n"]
            )
            self.returncode = 1

        def wait(self) -> int:
            return self.returncode

    def fake_popen(command, *, stdout, stderr, text, cwd, env):
        assert "--resume" in command
        assert command[command.index("--resume") + 1] == "missing-session"
        assert stdout == subprocess.PIPE
        assert stderr == subprocess.PIPE
        assert text is True
        return FakeProcess()

    monkeypatch.setattr(agent_backend_mod.subprocess, "Popen", fake_popen)

    backend = ClaudeBackend(binary="claude", settings_path=settings_path)
    with pytest.raises(MissingThreadRollout, match="missing-session"):
        backend.run_turn(
            prompt="hello",
            repo_root=repo_root,
            thread_id="missing-session",
        )
