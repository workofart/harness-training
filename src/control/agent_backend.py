from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any, Protocol

# ── shared constants ──────────────────────────────────────────────

DEFAULT_TURN_TIMEOUT_SEC = 600
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]


def supervisor_root_for_repo(repo_root: Path) -> Path:
    resolved = repo_root.resolve()
    return resolved.with_name(f"{resolved.name}_supervisor")


DEFAULT_SUPERVISOR_CODEX_HOME = (
    supervisor_root_for_repo(DEFAULT_REPO_ROOT) / "codex-home"
)


# ── shared types ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TurnResult:
    thread_id: str


class TurnTimeout(RuntimeError):
    def __init__(self, thread_id: str | None, timeout_sec: float):
        self.thread_id = thread_id
        super().__init__(f"agent turn timed out after {timeout_sec} seconds")


class MissingThreadRollout(RuntimeError):
    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        super().__init__(f"agent thread is unavailable: {thread_id}")


def _is_missing_resume_thread_error(stderr_text: str) -> bool:
    normalized = stderr_text.lower()
    return (
        "no rollout found for thread id" in normalized
        or "no conversation found with session id" in normalized
    )


# ── protocol ──────────────────────────────────────────────────────


class AgentBackend(Protocol):
    def run_turn(
        self,
        *,
        prompt: str,
        repo_root: Path,
        thread_id: str | None = None,
        timeout_sec: float = DEFAULT_TURN_TIMEOUT_SEC,
    ) -> TurnResult: ...


# ── terminal formatting helpers ───────────────────────────────────


def color_enabled() -> bool:
    force = os.environ.get("AUTO_LOG_COLOR")
    if force is not None:
        return force.lower() not in {"", "0", "false", "no"}
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def paint(text: str, code: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def role_label(role: str, *, enabled: bool) -> str:
    palette = {
        "supervisor": "1;35",
        "codex": "1;34",
        "claude": "1;36",
        "agent": "1;32",
        "toolcall": "1;33",
        "codex stderr": "1;31",
        "claude stderr": "1;31",
    }
    return paint(f"[{role}]", palette.get(role, "1"), enabled=enabled)


def _action_label(action: str, *, enabled: bool) -> str:
    palette = {
        "cmd>": "33",
        "cmd+": "32",
        "cmd!": "31",
        "edit>": "33",
        "edit+": "32",
        "read": "36",
        "tool": "33",
        "thread": "36",
        "turn+": "32",
        "tokens": "35",
        "done": "32",
        "error": "31",
    }
    return paint(action, palette.get(action, "0"), enabled=enabled)


def format_line(role: str, message: str, *, enabled: bool) -> str:
    return f"{role_label(role, enabled=enabled)} {message}"


def compact_path(path: str) -> str:
    try:
        rel = os.path.relpath(path, os.getcwd())
    except ValueError:
        rel = path
    if not rel.startswith(".."):
        return rel
    return path


def truncate(text: str, *, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def compact_paths(paths: list[str]) -> str:
    compacted = [compact_path(path) for path in paths]
    shown = compacted[:3]
    suffix = ""
    if len(compacted) > 3:
        suffix = f", +{len(compacted) - 3} more"
    return ", ".join(shown) + suffix


def print_terminal_lines(lines: list[str], *, use_stderr: bool) -> None:
    stream = sys.stderr if use_stderr else sys.stdout
    for line in lines:
        print(line, file=stream, flush=True)


def _format_agent_message(text: str, *, role: str, enabled: bool) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    lines = stripped.splitlines()
    rendered = [f"{role_label(role, enabled=enabled)} {lines[0].strip()}"]
    rendered.extend(f"  {line.rstrip()}" for line in lines[1:])
    return rendered


def _parse_json_line(raw_line: str) -> dict[str, Any] | None:
    line = raw_line.strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _run_streamed_process(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str] | None,
    role: str,
    timeout_sec: float,
    parse_event: Any,
    format_event: Any,
    extract_id: Any,
    initial_id: str | None,
) -> tuple[str | None, list[str], list[str]]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
    )
    enabled = color_enabled()
    resolved_id = initial_id
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stream_queue: Queue[tuple[str, str | None]] = Queue()

    def enqueue_stream(stream_name: str, stream: Any) -> None:
        if stream is None:
            stream_queue.put((stream_name, None))
            return
        for raw_line in stream:
            stream_queue.put((stream_name, raw_line))
        stream_queue.put((stream_name, None))

    stdout_thread = Thread(
        target=enqueue_stream,
        args=("stdout", process.stdout),
        daemon=True,
    )
    stderr_thread = Thread(
        target=enqueue_stream,
        args=("stderr", process.stderr),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    stderr_role = f"{role} stderr"
    stdout_done = False
    stderr_done = False
    deadline = time.monotonic() + timeout_sec
    while not (stdout_done and stderr_done):
        if time.monotonic() > deadline:
            process.kill()
            process.wait()
            raise TurnTimeout(resolved_id, timeout_sec)
        try:
            stream_name, raw_line = stream_queue.get(timeout=0.1)
        except Empty:
            continue
        if raw_line is None:
            if stream_name == "stdout":
                stdout_done = True
            else:
                stderr_done = True
            continue
        if stream_name == "stderr":
            stderr_chunks.append(raw_line)
            print_terminal_lines(
                [format_line(stderr_role, raw_line.rstrip("\n"), enabled=enabled)],
                use_stderr=True,
            )
            continue

        stdout_chunks.append(raw_line)
        payload = parse_event(raw_line)
        if payload is None:
            stripped = raw_line.rstrip("\n")
            if stripped:
                print_terminal_lines(
                    [format_line(role, stripped, enabled=enabled)],
                    use_stderr=False,
                )
            continue
        rendered = format_event(payload, enabled=enabled)
        if rendered is not None:
            print_terminal_lines(rendered, use_stderr=False)
        extracted = extract_id(payload)
        if extracted is not None:
            resolved_id = extracted

    process.wait()
    return resolved_id, stdout_chunks, stderr_chunks


# ── Codex backend ─────────────────────────────────────────────────


class CodexBackend:
    def __init__(
        self,
        *,
        binary: str = "codex",
        codex_home: Path = DEFAULT_SUPERVISOR_CODEX_HOME,
    ) -> None:
        self.binary = binary
        self.codex_home = codex_home

    def run_turn(
        self,
        *,
        prompt: str,
        repo_root: Path,
        thread_id: str | None = None,
        timeout_sec: float = DEFAULT_TURN_TIMEOUT_SEC,
    ) -> TurnResult:
        command = [self.binary, "exec"]
        if thread_id is not None:
            command.extend(["resume", thread_id])
        command.extend(
            [
                "--json",
                "--dangerously-bypass-approvals-and-sandbox",
                prompt,
            ]
        )
        codex_home = self._provision_codex_home()
        resolved_id, stdout_chunks, stderr_chunks = _run_streamed_process(
            command=command,
            cwd=repo_root,
            env={**os.environ, "CODEX_HOME": str(codex_home.resolve())},
            role="codex",
            timeout_sec=timeout_sec,
            parse_event=_parse_json_line,
            format_event=self._format_event,
            extract_id=self._extract_thread_id,
            initial_id=thread_id,
        )
        if not stdout_chunks and stderr_chunks:
            stderr_text = "".join(stderr_chunks)
            if thread_id is not None and _is_missing_resume_thread_error(stderr_text):
                raise MissingThreadRollout(thread_id)
            raise RuntimeError(f"codex turn failed:\nstderr:\n{stderr_text}")
        if resolved_id is None:
            raise RuntimeError("codex turn did not report a thread id")
        return TurnResult(thread_id=resolved_id)

    def _provision_codex_home(self) -> Path:
        self.codex_home.mkdir(parents=True, exist_ok=True)
        user_codex_home = Path.home() / ".codex"
        for filename in ("auth.json", "config.toml"):
            source = user_codex_home / filename
            if not source.exists():
                continue
            target = self.codex_home / filename
            if target.is_symlink():
                try:
                    points_to_source = target.resolve(strict=True) == source.resolve()
                except FileNotFoundError:
                    points_to_source = False
                if not points_to_source:
                    target.unlink()
            if not target.exists():
                target.symlink_to(source)
        return self.codex_home

    @staticmethod
    def _extract_thread_id(payload: dict[str, Any]) -> str | None:
        if payload.get("type") == "thread.started" and isinstance(
            payload.get("thread_id"), str
        ):
            return payload["thread_id"]
        return None

    @staticmethod
    def _context_window_remaining_percent(info: dict[str, Any]) -> int | None:
        model_context_window = info.get("model_context_window")
        if not isinstance(model_context_window, int) or model_context_window <= 0:
            return None
        last = info.get("last_token_usage")
        if not isinstance(last, dict):
            return None
        input_tokens = last.get("input_tokens")
        if not isinstance(input_tokens, int):
            return None
        remaining = max(model_context_window - input_tokens, 0)
        return (remaining * 100) // model_context_window

    @staticmethod
    def _compact_command(command: str) -> str:
        stripped = command.strip()
        try:
            argv = shlex.split(stripped)
        except ValueError:
            return truncate(stripped, limit=120)
        if (
            len(argv) == 3
            and argv[1] == "-lc"
            and os.path.basename(argv[0]) in {"sh", "bash", "zsh"}
        ):
            return truncate(argv[2], limit=120)
        return truncate(stripped, limit=120)

    @classmethod
    def _format_event(
        cls, payload: dict[str, Any], *, enabled: bool
    ) -> list[str] | None:
        event_type = payload.get("type")
        match event_type:
            case "thread.started":
                tid = payload.get("thread_id")
                message = _action_label("thread", enabled=enabled)
                if isinstance(tid, str) and tid:
                    message = f"{message} {tid}"
                return [format_line("codex", message, enabled=enabled)]
            case "turn.completed":
                return [
                    format_line(
                        "codex",
                        _action_label("turn+", enabled=enabled),
                        enabled=enabled,
                    )
                ]
            case "token_count":
                info = payload.get("info")
                if isinstance(info, dict):
                    last = info.get("last_token_usage")
                    if isinstance(last, dict):
                        parts: list[str] = []
                        aliases = {
                            "input_tokens": "in",
                            "cached_input_tokens": "cache",
                            "output_tokens": "out",
                            "reasoning_output_tokens": "reason",
                        }
                        for key, alias in aliases.items():
                            value = last.get(key)
                            if isinstance(value, int):
                                parts.append(f"{alias}={value}")
                        remaining = cls._context_window_remaining_percent(info)
                        if remaining is not None:
                            parts.append(f"left={remaining}%")
                        if parts:
                            return [
                                format_line(
                                    "codex",
                                    f"{_action_label('tokens', enabled=enabled)} {' '.join(parts)}",
                                    enabled=enabled,
                                )
                            ]
                return [
                    format_line(
                        "codex",
                        _action_label("tokens", enabled=enabled),
                        enabled=enabled,
                    )
                ]
            case _:
                pass
        item = payload.get("item")
        if not isinstance(item, dict):
            return None
        item_type = item.get("type")
        match (event_type, item_type):
            case ("item.completed", "agent_message"):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    return _format_agent_message(text, role="codex", enabled=enabled)
                return None
            case (_, "command_execution"):
                cmd = item.get("command")
                exit_code = item.get("exit_code")
                status = item.get("status")
                if event_type == "item.started":
                    action = "cmd>"
                elif status == "failed" or exit_code not in {None, 0}:
                    action = "cmd!"
                else:
                    action = "cmd+"
                line_parts = [_action_label(action, enabled=enabled)]
                if isinstance(cmd, str) and cmd:
                    line_parts.append(cls._compact_command(cmd))
                if isinstance(exit_code, int):
                    line_parts.append(f"rc={exit_code}")
                elif isinstance(status, str) and status not in {
                    "in_progress",
                    "completed",
                }:
                    line_parts.append(status)
                return [
                    f"  {format_line('toolcall', ' '.join(line_parts), enabled=enabled)}"
                ]
            case (_, "file_change"):
                action = "edit>" if event_type == "item.started" else "edit+"
                changes = item.get("changes")
                if isinstance(changes, list):
                    paths = [
                        change.get("path")
                        for change in changes
                        if isinstance(change, dict)
                        and isinstance(change.get("path"), str)
                    ]
                    if paths:
                        return [
                            f"  {format_line('toolcall', f'{_action_label(action, enabled=enabled)} {compact_paths(paths)}', enabled=enabled)}"
                        ]
                return [
                    f"  {format_line('toolcall', _action_label(action, enabled=enabled), enabled=enabled)}"
                ]
            case _:
                return None


# ── Claude backend ────────────────────────────────────────────────


class ClaudeBackend:
    def __init__(
        self,
        *,
        binary: str = "claude",
        settings_path: Path | None = None,
    ) -> None:
        self.binary = binary
        self.settings_path = settings_path

    def run_turn(
        self,
        *,
        prompt: str,
        repo_root: Path,
        thread_id: str | None = None,
        timeout_sec: float = DEFAULT_TURN_TIMEOUT_SEC,
    ) -> TurnResult:
        command = [
            self.binary,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]
        if self.settings_path is not None:
            command.extend(["--settings", str(self.settings_path.resolve())])
        if thread_id is not None:
            command.extend(["--resume", thread_id])
        command.extend(["--effort", "xhigh"])
        command.append(prompt)

        resolved_id, stdout_chunks, stderr_chunks = _run_streamed_process(
            command=command,
            cwd=repo_root,
            env=None,
            role="claude",
            timeout_sec=timeout_sec,
            parse_event=_parse_json_line,
            format_event=self._format_event,
            extract_id=self._extract_session_id,
            initial_id=thread_id,
        )
        if not stdout_chunks and stderr_chunks:
            stderr_text = "".join(stderr_chunks)
            if thread_id is not None and _is_missing_resume_thread_error(stderr_text):
                raise MissingThreadRollout(thread_id)
            raise RuntimeError(f"claude turn failed:\nstderr:\n{stderr_text}")
        if resolved_id is None:
            raise RuntimeError("claude turn did not report a session id")
        return TurnResult(thread_id=resolved_id)

    @staticmethod
    def _extract_session_id(payload: dict[str, Any]) -> str | None:
        if payload.get("type") == "system" and payload.get("subtype") == "init":
            session_id = payload.get("session_id")
            if isinstance(session_id, str):
                return session_id
        if payload.get("type") == "result":
            session_id = payload.get("session_id")
            if isinstance(session_id, str):
                return session_id
        return None

    @staticmethod
    def _tool_action(tool_name: str) -> tuple[str, str]:
        """Return (action_label, detail) for a Claude tool_use block."""
        if tool_name == "Bash":
            return "cmd>", ""
        if tool_name in ("Edit", "Write", "NotebookEdit"):
            return "edit>", ""
        if tool_name == "Read":
            return "read", ""
        return "tool", tool_name

    @staticmethod
    def _tool_detail(tool_name: str, tool_input: dict[str, Any]) -> str:
        if tool_name == "Bash":
            cmd = tool_input.get("command")
            if isinstance(cmd, str) and cmd.strip():
                return truncate(cmd.strip(), limit=120)
        if tool_name == "Read":
            fp = tool_input.get("file_path")
            if isinstance(fp, str):
                return compact_path(fp)
        if tool_name in ("Edit", "Write"):
            fp = tool_input.get("file_path")
            if isinstance(fp, str):
                return compact_path(fp)
        return ""

    @classmethod
    def _format_event(
        cls, payload: dict[str, Any], *, enabled: bool
    ) -> list[str] | None:
        event_type = payload.get("type")

        if event_type == "system" and payload.get("subtype") == "init":
            session_id = payload.get("session_id")
            message = _action_label("thread", enabled=enabled)
            if isinstance(session_id, str) and session_id:
                message = f"{message} {session_id}"
            return [format_line("claude", message, enabled=enabled)]

        if event_type == "assistant":
            msg = payload.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") == "text":
                            text = block.get("text")
                            if isinstance(text, str) and text.strip():
                                return _format_agent_message(
                                    text, role="claude", enabled=enabled
                                )
                        if block.get("type") == "tool_use":
                            tool_name = block.get("name", "")
                            tool_input = block.get("input", {})
                            action, _fallback = cls._tool_action(tool_name)
                            detail = cls._tool_detail(
                                tool_name,
                                tool_input if isinstance(tool_input, dict) else {},
                            )
                            label = _action_label(action, enabled=enabled)
                            if detail:
                                label = f"{label} {detail}"
                            elif _fallback:
                                label = f"{label} {_fallback}"
                            return [
                                f"  {format_line('toolcall', label, enabled=enabled)}"
                            ]
            return None

        if event_type == "result":
            subtype = payload.get("subtype")
            usage = payload.get("usage")
            if subtype == "success":
                parts: list[str] = []
                if isinstance(usage, dict):
                    for key, alias in (
                        ("input_tokens", "in"),
                        ("cache_read_input_tokens", "cache"),
                        ("output_tokens", "out"),
                    ):
                        value = usage.get(key)
                        if isinstance(value, int):
                            parts.append(f"{alias}={value}")
                label = _action_label("done", enabled=enabled)
                if parts:
                    label = f"{label} {' '.join(parts)}"
                return [format_line("claude", label, enabled=enabled)]
            if subtype == "error" or payload.get("is_error"):
                return [
                    format_line(
                        "claude",
                        _action_label("error", enabled=enabled),
                        enabled=enabled,
                    )
                ]
            return None

        return None


# ── factory ───────────────────────────────────────────────────────


def create_backend(agent_type: str) -> AgentBackend:
    if agent_type == "codex":
        return CodexBackend()
    if agent_type == "claude":
        return ClaudeBackend()
    raise ValueError(f"unknown agent type: {agent_type!r}")
