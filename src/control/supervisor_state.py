"""Supervisor finite-state-machine state and event log.

The supervisor persists which phase a repo is mid-flight in
(:class:`SupervisorState`, one ``state.json`` per repo) and appends an
``events.jsonl`` audit trail of every phase transition. Both are keyed by a
stable per-repo fingerprint. Extracted from supervisor.py so the durable
state/observability layer stays separate from the orchestration loop that
drives it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from src.control.agent_backend import (
    color_enabled,
    compact_paths,
    format_line,
    print_terminal_lines,
    supervisor_root_for_repo,
    truncate,
)
from src.experiment.record import write_json_atomic


SUPERVISOR_STATE_FILENAME = "state.json"
SUPERVISOR_EVENTS_FILENAME = "events.jsonl"
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUPERVISOR_ROOT = supervisor_root_for_repo(DEFAULT_REPO_ROOT)
Phase = Literal["prelaunch", "launch", "postrun"]


@dataclass(frozen=True, slots=True)
class SupervisorState:
    phase: Phase
    thread_id: str | None
    updated_at: str
    postrun_original_payload: dict[str, object] | None = None
    postrun_original_learning: str | None = None
    postrun_completed_experiment_id: str | None = None
    launch_experiment_id: str | None = None
    launch_baseline_commit: str | None = None

    @classmethod
    def path(
        cls,
        *,
        repo_root: Path,
        root: Path = DEFAULT_SUPERVISOR_ROOT,
    ) -> Path:
        return root / repo_fingerprint(repo_root) / SUPERVISOR_STATE_FILENAME

    @classmethod
    def maybe_load(
        cls,
        *,
        repo_root: Path,
        root: Path = DEFAULT_SUPERVISOR_ROOT,
    ) -> "SupervisorState | None":
        path = cls.path(repo_root=repo_root, root=root)
        if not path.exists():
            return None
        return cls(**json.loads(path.read_text()))

    def save(
        self,
        *,
        repo_root: Path,
        root: Path = DEFAULT_SUPERVISOR_ROOT,
    ) -> None:
        write_json_atomic(
            self.path(repo_root=repo_root, root=root),
            asdict(self),
        )

    @classmethod
    def clear(
        cls,
        *,
        repo_root: Path,
        root: Path = DEFAULT_SUPERVISOR_ROOT,
    ) -> None:
        path = cls.path(repo_root=repo_root, root=root)
        if path.exists():
            path.unlink()


def repo_fingerprint(repo_root: Path) -> str:
    resolved = repo_root.resolve()
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    return f"{resolved.name}-{digest}"


def supervisor_events_path(
    *,
    repo_root: Path,
    root: Path = DEFAULT_SUPERVISOR_ROOT,
) -> Path:
    return root / repo_fingerprint(repo_root) / SUPERVISOR_EVENTS_FILENAME


def append_supervisor_event(
    *,
    repo_root: Path,
    event: str,
    root: Path = DEFAULT_SUPERVISOR_ROOT,
    **fields: object,
) -> None:
    path = supervisor_events_path(repo_root=repo_root, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "fields": fields,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    print_terminal_lines(
        [_format_supervisor_event(event=event, fields=fields)],
        use_stderr=False,
    )


def _format_supervisor_event(*, event: str, fields: dict[str, object]) -> str:
    enabled = color_enabled()
    details: list[str] = []
    for key in (
        "phase",
        "experiment_id",
        "status",
        "thread_id",
        "baseline_experiment_id",
        "experiment_json_path",
    ):
        value = fields.get(key)
        if isinstance(value, str) and value:
            details.append(f"{key}={value}")
    error = fields.get("error")
    if isinstance(error, str) and error:
        details.append(f"error={truncate(error, limit=160)}")
    note = fields.get("note")
    if isinstance(note, str) and note:
        details.append(f"note={truncate(note, limit=160)}")
    session_count = fields.get("session_count")
    if isinstance(session_count, int):
        details.append(f"session_count={session_count}")
    changed_paths_value = fields.get("changed_paths")
    if isinstance(changed_paths_value, list):
        rendered_paths = [path for path in changed_paths_value if isinstance(path, str)]
        if rendered_paths:
            details.append(f"paths={compact_paths(rendered_paths)}")
    message = event if not details else f"{event} {' '.join(details)}"
    return format_line("supervisor", message, enabled=enabled)
