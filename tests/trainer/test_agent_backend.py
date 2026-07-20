from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import src.llm.agent_backend as agent_module
from src.llm.agent_backend import ClaudeAgentBackend, CodexAgentBackend

_CLAUDE_INIT = '{"type":"system","subtype":"init","session_id":"s1"}\n'
_CLAUDE_SUCCESS = (
    '{"type":"result","subtype":"success","session_id":"s1",'
    '"usage":{"output_tokens":5}}\n'
)
_CLAUDE_ERROR = (
    '{"type":"result","subtype":"error","session_id":"s1","is_error":true}\n'
)
_CODEX_THREAD = '{"type":"thread.started","thread_id":"t1"}\n'
_CODEX_DONE = '{"type":"turn.completed"}\n'

# Collection cannot know the temp repo path; replace this sentinel at runtime.
_REPO_ROOT = object()
# Never written: the fake process creates no sessions/ dir.
_TRACES = Path("codex-traces-unused")


class _FakePopen:
    def __init__(self, stdout=(), stderr=(), returncode=0):
        self.stdout = iter(stdout)
        self.stderr = iter(stderr)
        self._final_returncode = returncode
        self.returncode = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def poll(self):
        return self.returncode

    def wait(self):
        self.returncode = self._final_returncode
        return self.returncode

    def kill(self):
        self.returncode = -9


def _json_line(payload) -> str:
    return f"{json.dumps(payload)}\n"


def _capture_process(
    monkeypatch,
    *,
    stdout=(),
    stderr=(),
    returncode=0,
) -> dict:
    captured: dict = {}

    def popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        return _FakePopen(stdout, stderr, returncode)

    monkeypatch.setattr(agent_module.subprocess, "Popen", popen)
    return captured


def _invoke(backend, repo_root: Path, *, thread_id=None, emit=None):
    return backend.run_turn(
        prompt="p",
        repo_root=repo_root,
        emit=emit or (lambda _message: None),
        thread_id=thread_id,
    )


@pytest.mark.parametrize(
    ("backend", "thread_id", "stdout", "expected"),
    [
        pytest.param(
            ClaudeAgentBackend(),
            None,
            (_CLAUDE_INIT, _CLAUDE_SUCCESS),
            [
                "claude",
                "-p",
                "--output-format",
                "stream-json",
                "--verbose",
                "--permission-mode",
                "bypassPermissions",
                "--effort",
                "xhigh",
                "p",
            ],
            id="claude-default",
        ),
        pytest.param(
            ClaudeAgentBackend(effort="low"),
            None,
            (_CLAUDE_INIT, _CLAUDE_SUCCESS),
            [
                "claude",
                "-p",
                "--output-format",
                "stream-json",
                "--verbose",
                "--permission-mode",
                "bypassPermissions",
                "--effort",
                "low",
                "p",
            ],
            id="claude-effort",
        ),
        pytest.param(
            ClaudeAgentBackend(model="claude-test-model"),
            None,
            (_CLAUDE_INIT, _CLAUDE_SUCCESS),
            [
                "claude",
                "-p",
                "--output-format",
                "stream-json",
                "--verbose",
                "--permission-mode",
                "bypassPermissions",
                "--model",
                "claude-test-model",
                "--effort",
                "xhigh",
                "p",
            ],
            id="claude-model",
        ),
        pytest.param(
            ClaudeAgentBackend(),
            "s1",
            (_CLAUDE_SUCCESS,),
            [
                "claude",
                "-p",
                "--output-format",
                "stream-json",
                "--verbose",
                "--permission-mode",
                "bypassPermissions",
                "--resume",
                "s1",
                "--effort",
                "xhigh",
                "p",
            ],
            id="claude-resume",
        ),
        pytest.param(
            CodexAgentBackend(trace_dir=_TRACES),
            None,
            (_CODEX_THREAD, _CODEX_DONE),
            [
                "codex",
                "exec",
                "-c",
                "project_doc_max_bytes=0",
                "--json",
                "-s",
                "workspace-write",
                "-C",
                _REPO_ROOT,
                "--skip-git-repo-check",
                "p",
            ],
            id="codex-default",
        ),
        pytest.param(
            CodexAgentBackend(
                trace_dir=_TRACES, model="codex-test-model", effort="high"
            ),
            None,
            (_CODEX_THREAD, _CODEX_DONE),
            [
                "codex",
                "exec",
                "-c",
                'model_reasoning_effort="high"',
                "-c",
                "project_doc_max_bytes=0",
                "-m",
                "codex-test-model",
                "--json",
                "-s",
                "workspace-write",
                "-C",
                _REPO_ROOT,
                "--skip-git-repo-check",
                "p",
            ],
            id="codex-model-and-effort",
        ),
        pytest.param(
            CodexAgentBackend(trace_dir=_TRACES, effort="high"),
            "t1",
            (_CODEX_DONE,),
            [
                "codex",
                "exec",
                "-c",
                'model_reasoning_effort="high"',
                "-c",
                "project_doc_max_bytes=0",
                "--json",
                "-s",
                "workspace-write",
                "-C",
                _REPO_ROOT,
                "--skip-git-repo-check",
                "resume",
                "t1",
                "p",
            ],
            id="codex-resume",
        ),
        pytest.param(
            CodexAgentBackend(trace_dir=_TRACES),
            "t1",
            (_CODEX_DONE,),
            [
                "codex",
                "exec",
                "-c",
                "project_doc_max_bytes=0",
                "--json",
                "-s",
                "workspace-write",
                "-C",
                _REPO_ROOT,
                "--skip-git-repo-check",
                "resume",
                "t1",
                "p",
            ],
            id="codex-resume-default",
        ),
    ],
)
def test_backend_commands_and_environment(
    monkeypatch, tmp_path: Path, backend, thread_id, stdout, expected
) -> None:
    monkeypatch.setenv("PROPOSER_ENV_PROBE", "inherited")
    captured = _capture_process(monkeypatch, stdout=stdout)
    repo_root = tmp_path / "repo"
    result = _invoke(backend, repo_root, thread_id=thread_id)
    expected = [str(repo_root) if part is _REPO_ROOT else part for part in expected]
    assert captured["command"] == expected
    assert captured["env"]["PROPOSER_ENV_PROBE"] == "inherited"
    if isinstance(backend, ClaudeAgentBackend):
        assert result.thread_id == "s1"
        assert captured["env"]["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    else:
        assert result.thread_id == "t1"
        home = Path(captured["env"]["CODEX_HOME"])
        assert home.name.startswith("codex-home-")
        assert not home.exists()


@pytest.mark.parametrize(
    ("backend", "stdout", "stderr", "returncode", "match"),
    [
        pytest.param(
            ClaudeAgentBackend(),
            (_CLAUDE_INIT,),
            ("No conversation found with session id gone\n",),
            1,
            r"(?s)claude turn failed \(rc=1\).*No conversation found with session id gone",
            id="claude-nonzero",
        ),
        pytest.param(
            CodexAgentBackend(trace_dir=_TRACES),
            (_CODEX_THREAD,),
            ("no rollout found for thread id gone (code -32600)\n",),
            1,
            r"(?s)codex turn failed \(rc=1\).*no rollout found for thread id gone",
            id="codex-nonzero",
        ),
        pytest.param(
            ClaudeAgentBackend(),
            (_CLAUDE_INIT, _CLAUDE_ERROR),
            (),
            0,
            "result event error",
            id="claude-result-error",
        ),
        pytest.param(
            CodexAgentBackend(trace_dir=_TRACES),
            (
                _CODEX_THREAD,
                _json_line(
                    {
                        "type": "turn.failed",
                        "error": {"message": "unexpected status 401 Unauthorized"},
                    }
                ),
            ),
            (),
            0,
            "unexpected status 401 Unauthorized",
            id="codex-result-error",
        ),
        pytest.param(
            ClaudeAgentBackend(),
            ('{"type":"assistant","message":{}}\n',),
            (),
            0,
            "did not report a session id",
            id="claude-missing-id",
        ),
        pytest.param(
            CodexAgentBackend(trace_dir=_TRACES),
            (_CODEX_DONE,),
            (),
            0,
            "did not report a thread id",
            id="codex-missing-id",
        ),
    ],
)
def test_backend_failure_contracts(
    monkeypatch,
    tmp_path: Path,
    backend,
    stdout,
    stderr,
    returncode,
    match,
) -> None:
    _capture_process(monkeypatch, stdout=stdout, stderr=stderr, returncode=returncode)
    with pytest.raises(RuntimeError, match=match):
        _invoke(backend, tmp_path)


@pytest.mark.parametrize(
    ("backend", "stdout"),
    [
        pytest.param(
            ClaudeAgentBackend(turn_timeout_sec=12.5),
            (_CLAUDE_INIT, _CLAUDE_SUCCESS),
            id="claude",
        ),
        pytest.param(
            CodexAgentBackend(trace_dir=_TRACES, turn_timeout_sec=12.5),
            (_CODEX_THREAD, _CODEX_DONE),
            id="codex",
        ),
    ],
)
def test_backend_uses_configured_timeout(monkeypatch, tmp_path, backend, stdout):
    captured = _capture_process(monkeypatch, stdout=stdout)

    class _Timer:
        def __init__(self, interval, function):
            captured["timeout"] = interval
            self.function = function

        def start(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setattr(agent_module.threading, "Timer", _Timer)
    _invoke(backend, tmp_path)
    assert captured["timeout"] == 12.5


def _claude_assistant(block: dict) -> str:
    return _json_line({"type": "assistant", "message": {"content": [block]}})


def test_claude_progress_counts_event_categories(monkeypatch, tmp_path: Path) -> None:
    emitted: list[str] = []
    stdout = [
        _CLAUDE_INIT,
        *[_claude_assistant({"type": "tool_use", "name": "Read"})] * 2,
        *[_claude_assistant({"type": "tool_use", "name": "Bash"})] * 3,
        *[_claude_assistant({"type": "text", "text": "message"})] * 14,
        _CLAUDE_SUCCESS,
    ]
    _capture_process(monkeypatch, stdout=stdout)
    result = _invoke(ClaudeAgentBackend(), tmp_path, emit=emitted.append)
    assert len(emitted) == 1
    assert re.fullmatch(r"running · \d+s · agent: read 2 · cmd 3 · msg 14", emitted[0])
    assert re.fullmatch(
        r"\d+s · agent: read 2 · cmd 3 · msg 14", result.progress_summary
    )


def _codex_item(event_type: str, item_type: str) -> str:
    return _json_line({"type": event_type, "item": {"type": item_type}})


def test_codex_progress_counts_event_categories(monkeypatch, tmp_path: Path) -> None:
    emitted: list[str] = []
    stdout = [
        _CODEX_THREAD,
        *[_codex_item("item.started", "command_execution")] * 3,
        *[_codex_item("item.completed", "file_change")] * 4,
        *[_codex_item("item.completed", "agent_message")] * 12,
        _codex_item("item.started", "command_execution"),
        *[_codex_item("item.completed", "file_change")] * 2,
        _CODEX_DONE,
    ]
    _capture_process(monkeypatch, stdout=stdout)
    result = _invoke(
        CodexAgentBackend(trace_dir=_TRACES), tmp_path, emit=emitted.append
    )
    assert len(emitted) == 1
    assert re.fullmatch(r"running · \d+s · agent: cmd 3 · edit 4 · msg 12", emitted[0])
    assert re.fullmatch(
        r"\d+s · agent: cmd 4 · edit 6 · msg 12", result.progress_summary
    )


def test_codex_home_is_ephemeral_and_archives_rollouts(
    monkeypatch, tmp_path: Path
) -> None:
    user_codex_home = tmp_path / "home" / ".codex"
    user_codex_home.mkdir(parents=True)
    (user_codex_home / "auth.json").write_text("{}\n")
    (user_codex_home / "config.toml").write_text("model = 'gpt-5'\n")
    monkeypatch.setenv("HOME", str(user_codex_home.parent))
    trace_dir = tmp_path / "traces"
    backend = CodexAgentBackend(trace_dir=trace_dir)
    homes: list[Path] = []
    rollout_rel = Path("2026/07/17/rollout-2026-07-17-t1.jsonl")

    def fake_codex(command, **kwargs):
        del command
        home = Path(kwargs["env"]["CODEX_HOME"])
        homes.append(home)
        assert (home / "auth.json").resolve() == user_codex_home / "auth.json"
        assert (home / "config.toml").resolve() == user_codex_home / "config.toml"
        rollout = home / "sessions" / rollout_rel
        rollout.parent.mkdir(parents=True, exist_ok=True)
        with rollout.open("a") as stream:
            stream.write('{"turn":1}\n')
        return _FakePopen((_CODEX_THREAD, _CODEX_DONE))

    monkeypatch.setattr(agent_module.subprocess, "Popen", fake_codex)
    _invoke(backend, tmp_path / "repo")
    assert not homes[0].exists()
    assert (trace_dir / rollout_rel).read_text() == '{"turn":1}\n'

    _invoke(backend, tmp_path / "repo", thread_id="t1")
    assert homes[1] != homes[0]
    assert not homes[1].exists()
    # Two lines: the fake appended to the archive-seeded rollout.
    assert (trace_dir / rollout_rel).read_text() == '{"turn":1}\n{"turn":1}\n'


def test_codex_rollouts_survive_failed_turns(monkeypatch, tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"

    def crashing_codex(command, **kwargs):
        del command
        rollout = Path(kwargs["env"]["CODEX_HOME"]) / "sessions" / "rollout-t1.jsonl"
        rollout.parent.mkdir(parents=True)
        rollout.write_text('{"turn":1}\n')
        return _FakePopen((_CODEX_THREAD,), returncode=1)

    monkeypatch.setattr(agent_module.subprocess, "Popen", crashing_codex)
    with pytest.raises(RuntimeError, match="codex turn failed"):
        _invoke(CodexAgentBackend(trace_dir=trace_dir), tmp_path / "repo")
    assert (trace_dir / "rollout-t1.jsonl").read_text() == '{"turn":1}\n'


@pytest.mark.parametrize(
    ("backend", "stdout", "model_args"),
    [
        pytest.param(
            ClaudeAgentBackend(model="claude-test-model"),
            (_CLAUDE_INIT, _CLAUDE_SUCCESS),
            ["--model", "claude-test-model"],
            id="claude",
        ),
        pytest.param(
            CodexAgentBackend(trace_dir=_TRACES, model="codex-test-model"),
            (_CODEX_THREAD, _CODEX_DONE),
            ["-m", "codex-test-model"],
            id="codex",
        ),
    ],
)
def test_readiness_probe_runs_a_bounded_turn_outside_the_repo(
    monkeypatch, tmp_path: Path, backend, stdout, model_args
) -> None:
    captured = _capture_process(monkeypatch, stdout=stdout)

    class _Timer:
        def __init__(self, interval, function):
            captured["timeout"] = interval
            self.function = function

        def start(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setattr(agent_module.threading, "Timer", _Timer)
    monkeypatch.chdir(tmp_path)

    backend._assert_ready()

    assert captured["timeout"] == agent_module._READY_TIMEOUT_SEC
    model_arg_index = captured["command"].index(model_args[0])
    assert captured["command"][model_arg_index : model_arg_index + 2] == model_args
    probe_root = Path(captured["cwd"])
    assert not probe_root.is_relative_to(tmp_path)
    assert not probe_root.exists()


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(ClaudeAgentBackend(), id="claude"),
        pytest.param(CodexAgentBackend(trace_dir=_TRACES), id="codex"),
    ],
)
def test_readiness_probe_surfaces_a_failing_cli(monkeypatch, backend) -> None:
    _capture_process(monkeypatch, stderr=("not authenticated\n",), returncode=1)
    with pytest.raises(RuntimeError, match="turn failed"):
        backend._assert_ready()


def test_readiness_probe_surfaces_a_missing_cli(monkeypatch) -> None:
    def missing(command, **kwargs):
        del command, kwargs
        raise FileNotFoundError(2, "No such file or directory", "claude")

    monkeypatch.setattr(agent_module.subprocess, "Popen", missing)
    with pytest.raises(FileNotFoundError):
        ClaudeAgentBackend()._assert_ready()
