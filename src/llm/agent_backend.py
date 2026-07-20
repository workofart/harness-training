"""Agent CLI backends: spawn one tool-using agent turn and stream its JSON event log.

The proposer drives a *real* agent -- not a single LLM call -- so it can read program.md
+ prior evidence with its own tools, edit `core.py`, run the contract test, and resume the
same thread to fix a rejected attempt. A throttled counter meter keeps the otherwise-silent
multi-minute turn legible.

Only two things the stream carries are load-bearing: the resumable thread/session id and
whether the turn failed. Everything else is throwaway console decoration, so each backend
only classifies events into shared meter categories.
"""

from __future__ import annotations

import collections
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.llm.backend import (
    AgentBackend,
    Emit,
    TurnResult,
)


_METER_EVERY_EVENTS = 20

# Readiness probe: a trivial bounded turn is the only check that proves auth.
_READY_TIMEOUT_SEC = 120.0
_READY_PROMPT = "Reply with OK. Do not use tools."


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _parse_json_line(raw_line: str) -> dict[str, Any] | None:
    line = raw_line.strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _error_message(payload: dict[str, Any]) -> str | None:
    """Both agent CLIs report a failed turn's message as either a nested `error.message`
    or a top-level `message`."""
    error = payload.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    return _as_str(payload.get("message"))


@dataclass(frozen=True, slots=True)
class _Event:
    """One parsed event: `category` feeds the meter; the rest are load-bearing signals."""

    category: str | None = None
    thread_id: str | None = None
    failed: bool = False
    failure_message: str | None = None


def _format_elapsed(seconds: float) -> str:
    total = int(seconds)
    if total >= 3600:
        hours, remainder = divmod(total, 3600)
        minutes = remainder // 60
        return f"{hours}h{minutes:02d}m"
    if total >= 60:
        minutes, seconds = divmod(total, 60)
        return f"{minutes}m{seconds:02d}s"
    return f"{total}s"


def _meter_suffix(counts: collections.Counter[str], started: float) -> str:
    counters = " · ".join(
        f"{name} {counts[name]}"
        for name in ("read", "cmd", "edit", "tool", "msg")
        if counts[name]
    )
    elapsed = _format_elapsed(time.monotonic() - started)
    return f"{elapsed} · agent: {counters}" if counters else elapsed


def _failure_detail(result_error: _Event | None, stderr_text: str) -> str:
    parts = []
    if result_error is not None:
        message = result_error.failure_message
        if message is not None and message.strip():
            parts.append(f"result event error: {' '.join(message.split())}")
        else:
            parts.append("result event error")
    if stderr_text.strip():
        parts.append(f"stderr:\n{stderr_text[-1000:]}")
    return " | ".join(parts) or "no diagnostic output"


def _run_agent_subprocess(
    *,
    backend_name: str,
    command: list[str],
    repo_root: Path,
    env: dict[str, str],
    emit: Emit,
    thread_id: str | None,
    turn_timeout_sec: float,
    parse: Callable[[dict[str, Any]], _Event],
    missing_id_label: str,
) -> TurnResult:
    started = time.monotonic()
    resolved_id = thread_id
    result_error: _Event | None = None
    stderr_chunks: list[str] = []
    timed_out = threading.Event()
    events = 0
    counts: collections.Counter[str] = collections.Counter()
    with subprocess.Popen(
        command,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    ) as proc:

        def _drain_stderr() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                stderr_chunks.append(line)

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        def _kill() -> None:
            timed_out.set()
            proc.kill()

        timer = threading.Timer(turn_timeout_sec, _kill)
        timer.start()
        try:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                payload = _parse_json_line(raw_line)
                if payload is None:
                    stripped = raw_line.rstrip("\n")
                    if stripped:
                        emit(stripped)
                    continue
                event = parse(payload)
                events += 1
                if event.category is not None:
                    counts[event.category] += 1
                if event.failed:
                    result_error = event
                if event.thread_id is not None and event.thread_id != resolved_id:
                    resolved_id = event.thread_id
                if events % _METER_EVERY_EVENTS == 0:
                    emit(f"running · {_meter_suffix(counts, started)}")
            proc.wait()
        finally:
            timer.cancel()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            stderr_thread.join(timeout=5)

        returncode = proc.returncode
        stderr_text = "".join(stderr_chunks)

    if timed_out.is_set():
        raise RuntimeError(f"agent turn timed out after {turn_timeout_sec:.0f}s")
    if returncode != 0 or result_error is not None:
        raise RuntimeError(
            f"{backend_name} turn failed (rc={returncode}): "
            f"{_failure_detail(result_error, stderr_text)}"
        )
    if resolved_id is None:
        raise RuntimeError(
            f"{backend_name} turn did not report a {missing_id_label}:\n"
            f"{stderr_text[-1000:]}"
        )
    return TurnResult(
        thread_id=resolved_id,
        progress_summary=_meter_suffix(counts, started),
    )


class CodexAgentBackend(AgentBackend):
    """Spawns `codex exec` per turn. Thread resume uses `exec resume <id>`.

    Each turn runs in a throwaway CODEX_HOME so the prompt is the only
    instruction channel. Session rollouts are archived to `trace_dir` on every
    exit path (crashes included) and restored from there on resume."""

    def __init__(
        self,
        *,
        trace_dir: Path,
        turn_timeout_sec: float = 3600.0,
        model: str | None = None,
        effort: str | None = None,
    ) -> None:
        self._trace_dir = trace_dir
        self._turn_timeout_sec = turn_timeout_sec
        self._model = model
        self._effort = effort

    def run_turn(
        self,
        *,
        prompt: str,
        repo_root: Path,
        emit: Emit,
        thread_id: str | None = None,
    ) -> TurnResult:
        command = ["codex", "exec"]
        if self._effort is not None:
            command.extend(["-c", f'model_reasoning_effort="{self._effort}"'])
        command.extend(["-c", "project_doc_max_bytes=0"])
        if self._model is not None:
            command.extend(["-m", self._model])
        # Sandbox writes to the worktree only; `resume` rejects trailing options.
        command.extend(["--json", "-s", "workspace-write", "-C", str(repo_root)])
        command.append("--skip-git-repo-check")
        if thread_id is not None:
            command.extend(["resume", thread_id])
        command.append(prompt)

        home = self._provision_home(thread_id)
        try:
            return _run_agent_subprocess(
                backend_name="codex",
                command=command,
                repo_root=repo_root,
                env={**os.environ, "CODEX_HOME": str(home)},
                emit=emit,
                thread_id=thread_id,
                turn_timeout_sec=self._turn_timeout_sec,
                parse=self._parse_event,
                missing_id_label="thread id",
            )
        finally:
            _copy_rollouts(home / "sessions", "*.jsonl", self._trace_dir)
            shutil.rmtree(home, ignore_errors=True)

    def _assert_ready(self) -> None:
        # Probe a bounded sibling: an unbounded turn would hang on a CLI that stalls.
        with tempfile.TemporaryDirectory(prefix="agent-ready-") as scratch:
            root = Path(scratch)
            CodexAgentBackend(
                trace_dir=root / "traces",
                turn_timeout_sec=_READY_TIMEOUT_SEC,
                model=self._model,
                effort="low",
            ).run_turn(prompt=_READY_PROMPT, repo_root=root, emit=lambda _message: None)

    def _parse_event(self, payload: dict[str, Any]) -> _Event:
        event_type = payload.get("type")
        match event_type:
            case "thread.started":
                thread_id = _as_str(payload.get("thread_id"))
                return _Event(thread_id=thread_id)
            case "turn.completed":
                return _Event()
            case "turn.failed":
                return _Event(failed=True, failure_message=_error_message(payload))
        item = payload.get("item")
        if isinstance(item, dict):
            match item.get("type"):
                case "command_execution" if event_type == "item.started":
                    return _Event(category="cmd")
                case "file_change" if event_type == "item.completed":
                    return _Event(category="edit")
                case "agent_message" if event_type == "item.completed":
                    return _Event(category="msg")
        return _Event()

    def _provision_home(self, thread_id: str | None) -> Path:
        home = Path(tempfile.mkdtemp(prefix="codex-home-"))
        user_codex_home = Path.home() / ".codex"
        for filename in ("auth.json", "config.toml"):
            source = user_codex_home / filename
            if source.exists():
                (home / filename).symlink_to(source)
        if thread_id is not None:
            _copy_rollouts(self._trace_dir, f"*{thread_id}*.jsonl", home / "sessions")
        return home


def _copy_rollouts(source: Path, pattern: str, target_root: Path) -> None:
    """Mirror matching rollouts, preserving the layout `exec resume` expects."""
    if not source.is_dir():
        return
    for rollout in source.rglob(pattern):
        target = target_root / rollout.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rollout, target)


class ClaudeAgentBackend(AgentBackend):
    """Spawns `claude -p` per turn; thread resume uses `--resume`."""

    def __init__(
        self,
        *,
        turn_timeout_sec: float = 3600.0,
        model: str | None = None,
        effort: str | None = None,
    ) -> None:
        self._turn_timeout_sec = turn_timeout_sec
        self._model = model
        self._effort = effort

    def run_turn(
        self,
        *,
        prompt: str,
        repo_root: Path,
        emit: Emit,
        thread_id: str | None = None,
    ) -> TurnResult:
        # The counterpart to Codex's workspace-write. acceptEdits covers edits
        # only, so under -p every command that executes code failed "requires
        # approval" with no human to ask: the proposer could not run the suite
        # program.md tells it to keep green, and shipped patches it had never
        # run. Explicit deny rules still apply under this mode.
        command = [
            "claude",
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            "bypassPermissions",
        ]
        if self._model is not None:
            command.extend(["--model", self._model])
        if thread_id is not None:
            command.extend(["--resume", thread_id])
        command.extend(["--effort", self._effort or "xhigh", prompt])

        return _run_agent_subprocess(
            backend_name="claude",
            command=command,
            repo_root=repo_root,
            # Disable shared auto-memory; experiment artifacts are durable memory.
            env={**os.environ, "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"},
            emit=emit,
            thread_id=thread_id,
            turn_timeout_sec=self._turn_timeout_sec,
            parse=self._parse_event,
            missing_id_label="session id",
        )

    def _assert_ready(self) -> None:
        # Probe a bounded sibling: an unbounded turn would hang on a CLI that stalls.
        with tempfile.TemporaryDirectory(prefix="agent-ready-") as scratch:
            ClaudeAgentBackend(
                turn_timeout_sec=_READY_TIMEOUT_SEC,
                model=self._model,
                effort="low",
            ).run_turn(
                prompt=_READY_PROMPT,
                repo_root=Path(scratch),
                emit=lambda _message: None,
            )

    def _parse_event(self, payload: dict[str, Any]) -> _Event:
        match payload.get("type"):
            case "system" if payload.get("subtype") == "init":
                session_id = _as_str(payload.get("session_id"))
                return _Event(thread_id=session_id)
            case "assistant":
                message = payload.get("message")
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, list):
                    return _Event()
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text = block.get("text")
                        if isinstance(text, str) and text.strip():
                            return _Event(category="msg")
                    elif block.get("type") == "tool_use":
                        name = _as_str(block.get("name")) or ""
                        category = {
                            "Bash": "cmd",
                            "Edit": "edit",
                            "Write": "edit",
                            "NotebookEdit": "edit",
                            "Read": "read",
                        }.get(name, "tool")
                        return _Event(category=category)
                return _Event()
            case "result":
                session_id = _as_str(payload.get("session_id"))
                if payload.get("subtype") == "success":
                    return _Event(thread_id=session_id)
                failed = payload.get("subtype") == "error" or bool(
                    payload.get("is_error")
                )
                return _Event(
                    thread_id=session_id,
                    failed=failed,
                    failure_message=_error_message(payload),
                )
        return _Event()
