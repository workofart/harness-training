"""Shared test scaffolding.

pytest puts this file's directory (``tests/``) on ``sys.path``, so test modules
import these helpers with ``from conftest import _StubLlm, ...``. Keep only
genuinely cross-cutting fakes here -- stubs used by a single test module belong
next to that module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.adapters.llm_base import BaseLlm, LlmCompletion, LlmToolCall
from src.harness.contracts import RawState, TaskResult


class _StubLlm(BaseLlm):
    """Returns a pre-set sequence of LlmCompletions; records every call."""

    def __init__(self, completions: list[LlmCompletion]) -> None:
        self._completions = list(completions)
        self.calls: list[list[dict[str, Any]]] = []
        self.closed = False

    async def complete(self, *, messages, tools=None, reasoning_effort=None):
        del tools, reasoning_effort
        self.calls.append([dict(m) for m in messages])
        return self._completions.pop(0)

    async def close(self) -> None:
        self.closed = True


class _StubEnv:
    """Minimal HarnessEnv stub: returns canned states, records calls."""

    def __init__(
        self,
        *,
        reset_state: RawState | None = None,
        exec_states: list[RawState] | None = None,
        verify_state: RawState | None = None,
        trial_dir: str = "/tmp/trial",
        verifier_stdout_path: str | None = None,
    ) -> None:
        self._reset_state = reset_state or RawState(
            instruction="do the thing", working_dir="/work"
        )
        self._exec_states = list(exec_states or [])
        self._verify_state = verify_state or RawState(
            done=True, passed=True, reward=1.0
        )
        self.trial_dir: str | None = trial_dir
        self.verifier_stdout_path: str | None = verifier_stdout_path
        self.exec_calls: list[dict[str, Any]] = []
        self.verify_calls = 0
        self.closed = False

    async def reset(self) -> RawState:
        return self._reset_state

    async def exec(self, *, command, cwd=None, timeout_sec=None, workload="heavy"):
        self.exec_calls.append(
            {
                "command": command,
                "cwd": cwd,
                "timeout_sec": timeout_sec,
                "workload": workload,
            }
        )
        if self._exec_states:
            return self._exec_states.pop(0)
        return RawState(return_code=0)

    async def verify(self) -> RawState:
        self.verify_calls += 1
        return self._verify_state

    async def close(self) -> None:
        self.closed = True


def _tool_call(name: str, **args: Any) -> LlmToolCall:
    return LlmToolCall(name=name, arguments=json.dumps(args))


def _completion(*calls: LlmToolCall, content: str | None = None) -> LlmCompletion:
    return LlmCompletion(tool_calls=tuple(calls), content=content)


def _task_result(
    *,
    task_name: str,
    reward: float | None,
    solved: bool | None = None,
    error: str | None = None,
) -> TaskResult:
    """A finished TaskResult; `solved` defaults to whether a non-error trial
    earned positive reward."""
    if solved is None:
        solved = error is None and reward is not None and reward > 0.0
    return TaskResult(
        task_name=task_name,
        reward=reward,
        steps_used=1,
        error=error,
        trial_dir=None,
        verifier_stdout_path=None,
        started_at="2026-04-10T00:00:00+00:00",
        finished_at="2026-04-10T00:00:01+00:00",
        solved=solved,
    )


def _write_task_artifacts(root: Path, task_name: str) -> dict[str, str]:
    """Create the canonical per-task artifact files under ``root/task_name`` and
    return their paths (as a real run's recorder would leave them on disk)."""
    task_dir = root / task_name
    agent_dir = task_dir / "agent"
    agent_dir.mkdir(parents=True)
    steps_path = agent_dir / "steps.jsonl"
    metrics_path = agent_dir / "metrics.json"
    exec_log_path = agent_dir / "exec.log"
    verifier_path = task_dir / "verifier.txt"
    for path in (steps_path, metrics_path, exec_log_path, verifier_path):
        path.write_text("{}\n")
    return {
        "trial_dir": str(task_dir),
        "trace_path": str(steps_path),
        "metrics_path": str(metrics_path),
        "verifier_stdout_path": str(verifier_path),
        "exec_log_path": str(exec_log_path),
    }
