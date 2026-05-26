"""
Harness policy core.

Public critical path from `uv run exp`:
- `src.cli.main_exp()` loads runtime config and constructs
  `src.experiment.runner.ExperimentRunner`.
- `ExperimentRunner.run()` creates the experiment record, resolves task dirs,
  and runs the train panel.
- `src.experiment.trial.run_task()` owns reset, timeout, artifact paths,
  tracing/metrics, cleanup, and the returned `TaskResult`.
- `run_task_loop()` owns the post-reset policy/environment loop.

This module defines the loop-local design:
- 8-action vocabulary and model-facing tool specs
- JSON tool-call parse, validation, and typed action construction
- action execution as one shell command or verifier call through `HarnessEnv`
- prompt replay from `(Action, RawState)` trajectory into chat messages
- bounded observation rendering for stdout/stderr
- `act()`: one model completion attempt plus configurable repair retries
- `run_task_loop()`: execute emitted actions in order until env `done` or
  `max_steps`, updating optional trace/metrics recorders and progress state

Boundary contracts:
- Environment adapters implement `src.harness.contracts.HarnessEnv`.
- LLM adapters implement `src.adapters.llm_base.BaseLlm`.
- Trial results cross back to the runner as `src.harness.contracts.TaskResult`.
"""

from __future__ import annotations

import json
import shlex
from abc import ABC
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Literal, TypeAlias

from src.adapters.llm_base import BaseLlm
from src.harness.contracts import HarnessEnv, RawState


# ============================================================================
# Constants
# ============================================================================

MISSING_TOOL_CALL_REPAIR_PROMPT = (
    "Your previous response omitted the required tool call. "
    "Return one or more tool calls and no natural language."
)
DEFAULT_READ_WINDOW_LINES = 200
DEFAULT_RESULT_CHAR_LIMIT = 6000


# ============================================================================
# Action vocabulary
# ============================================================================

ActionName: TypeAlias = Literal[
    "list_dir",
    "find_files",
    "search_text",
    "read_file",
    "write_file",
    "edit_file",
    "run",
    "verify",
]

Trajectory: TypeAlias = tuple[tuple["Action", "RawState"], ...]


class Action(ABC):
    NAME: ClassVar[ActionName]


@dataclass(frozen=True, slots=True)
class ListDirAction(Action):
    NAME: ClassVar[Literal["list_dir"]] = "list_dir"
    path: str | None = None


@dataclass(frozen=True, slots=True)
class FindFilesAction(Action):
    NAME: ClassVar[Literal["find_files"]] = "find_files"
    pattern: str
    root: str | None = None


@dataclass(frozen=True, slots=True)
class SearchTextAction(Action):
    NAME: ClassVar[Literal["search_text"]] = "search_text"
    query: str
    root: str | None = None


@dataclass(frozen=True, slots=True)
class ReadFileAction(Action):
    NAME: ClassVar[Literal["read_file"]] = "read_file"
    path: str
    start_line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True, slots=True)
class WriteFileAction(Action):
    NAME: ClassVar[Literal["write_file"]] = "write_file"
    path: str
    content: str


@dataclass(frozen=True, slots=True)
class EditFileAction(Action):
    NAME: ClassVar[Literal["edit_file"]] = "edit_file"
    path: str
    old_text: str
    new_text: str


@dataclass(frozen=True, slots=True)
class RunAction(Action):
    NAME: ClassVar[Literal["run"]] = "run"
    command: str
    cwd: str | None = None
    timeout_sec: int | None = None


@dataclass(frozen=True, slots=True)
class VerifyAction(Action):
    NAME: ClassVar[Literal["verify"]] = "verify"


@dataclass(frozen=True, slots=True)
class TaskLoopResult:
    reward: float
    solved: bool
    steps_used: int
    final_passed: bool | None


@dataclass(slots=True)
class TaskLoopProgress:
    reward: float = 0.0
    steps_used: int = 0
    final_passed: bool | None = None


# ============================================================================
# Action specs and tool specs
# ============================================================================


@dataclass(frozen=True, slots=True)
class ActionSpec:
    name: ActionName
    description: str
    required_keys: tuple[str, ...]
    optional_keys: tuple[str, ...] = ()


ACTION_SPECS: dict[ActionName, ActionSpec] = {
    "list_dir": ActionSpec(
        name="list_dir",
        description="List directory contents.",
        required_keys=(),
        optional_keys=("path",),
    ),
    "find_files": ActionSpec(
        name="find_files",
        description="Find files by filename pattern.",
        required_keys=("pattern",),
        optional_keys=("root",),
    ),
    "search_text": ActionSpec(
        name="search_text",
        description="Search file contents for text.",
        required_keys=("query",),
        optional_keys=("root",),
    ),
    "read_file": ActionSpec(
        name="read_file",
        description="Read a file, optionally by line range.",
        required_keys=("path",),
        optional_keys=("start_line", "end_line"),
    ),
    "write_file": ActionSpec(
        name="write_file",
        description="Write the full contents of a file.",
        required_keys=("path", "content"),
    ),
    "edit_file": ActionSpec(
        name="edit_file",
        description="Replace one exact text span in a file.",
        required_keys=("path", "old_text", "new_text"),
    ),
    "run": ActionSpec(
        name="run",
        description="Run one shell command.",
        required_keys=("command",),
        optional_keys=("cwd", "timeout_sec"),
    ),
    "verify": ActionSpec(
        name="verify",
        description="Ask the environment for the authoritative task judgment.",
        required_keys=(),
    ),
}


INTEGER_FIELDS: frozenset[str] = frozenset({"start_line", "end_line", "timeout_sec"})


def build_tool_specs() -> list[dict[str, Any]]:
    """Render ACTION_SPECS as model-facing tool specs."""
    tools: list[dict[str, Any]] = []
    for name in sorted(ACTION_SPECS):
        spec = ACTION_SPECS[name]
        properties: dict[str, Any] = {}
        for key in (*spec.required_keys, *spec.optional_keys):
            required = key in spec.required_keys
            base = "integer" if key in INTEGER_FIELDS else "string"
            properties[key] = {"type": base if required else [base, "null"]}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": list(spec.required_keys),
                        "additionalProperties": False,
                    },
                },
            }
        )
    return tools


# ============================================================================
# Argument parsing and validation
# ============================================================================


def parse_action_args(args_text: str) -> Any:
    raw = args_text.strip()
    if raw == "":
        return {}
    return json.loads(raw)


def validate_action_args(action_name: ActionName, args: Any) -> dict[str, Any]:
    if not isinstance(args, dict):
        raise ValueError(f"{action_name}: arguments must decode to an object")
    spec = ACTION_SPECS[action_name]
    allowed = set(spec.required_keys) | set(spec.optional_keys)
    missing = [k for k in spec.required_keys if k not in args]
    if missing:
        raise ValueError(f"{action_name}: missing required keys {missing}")
    unknown = sorted(set(args) - allowed)
    if unknown:
        raise ValueError(f"{action_name}: unknown keys {unknown}")
    for key in INTEGER_FIELDS & args.keys():
        value = args[key]
        if value is not None and not isinstance(value, int):
            raise ValueError(f"{key}: expected integer or null")
    return dict(args)


def build_action(action_name: ActionName, args: dict[str, Any]) -> Action:
    match action_name:
        case "list_dir":
            return ListDirAction(path=_optional_str(args.get("path")))
        case "find_files":
            return FindFilesAction(
                pattern=_required_str(args, "pattern"),
                root=_optional_str(args.get("root")),
            )
        case "search_text":
            return SearchTextAction(
                query=_required_str(args, "query"),
                root=_optional_str(args.get("root")),
            )
        case "read_file":
            return ReadFileAction(
                path=_required_str(args, "path"),
                start_line=_optional_int(args.get("start_line")),
                end_line=_optional_int(args.get("end_line")),
            )
        case "write_file":
            return WriteFileAction(
                path=_required_str(args, "path"),
                content=_required_str(args, "content"),
            )
        case "edit_file":
            return EditFileAction(
                path=_required_str(args, "path"),
                old_text=_required_str(args, "old_text"),
                new_text=_required_str(args, "new_text"),
            )
        case "run":
            return RunAction(
                command=_required_str(args, "command"),
                cwd=_optional_str(args.get("cwd")),
                timeout_sec=_optional_int(args.get("timeout_sec")),
            )
        case "verify":
            return VerifyAction()
    raise ValueError(f"unknown action name: {action_name!r}")


def _required_str(args: dict[str, Any], key: str) -> str:
    value = args[key]
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{key}: expected non-empty string")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("expected string or null")
    return value


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError("expected integer or null")
    return value


# ============================================================================
# Environment execution
# ============================================================================


async def execute_action(env: HarnessEnv, action: Action) -> RawState:
    match action:
        case ListDirAction(path=path):
            return await env.exec(command=f"ls -1Ap {shlex.quote(path or '.')}")
        case FindFilesAction(pattern=pattern, root=root):
            return await env.exec(
                command=f"find {shlex.quote(root or '.')} -name {shlex.quote(pattern)}"
            )
        case SearchTextAction(query=query, root=root):
            return await env.exec(
                command=f"grep -rn {shlex.quote(query)} {shlex.quote(root or '.')}"
            )
        case ReadFileAction(path=path, start_line=start_line, end_line=end_line):
            start = start_line or 1
            end = (
                end_line
                if end_line is not None
                else start + DEFAULT_READ_WINDOW_LINES - 1
            )
            return await env.exec(
                command=f"sed -n '{start},{end}p' {shlex.quote(path)}"
            )
        case WriteFileAction(path=path, content=content):
            return await env.exec(
                command=f"printf %s {shlex.quote(content)} > {shlex.quote(path)}"
            )
        case EditFileAction(path=path, old_text=old_text, new_text=new_text):
            script = (
                "import sys, pathlib; "
                "p = pathlib.Path(sys.argv[1]); "
                "p.write_text(p.read_text().replace(sys.argv[2], sys.argv[3], 1))"
            )
            return await env.exec(
                command=(
                    f"python3 -c {shlex.quote(script)} "
                    f"{shlex.quote(path)} {shlex.quote(old_text)} {shlex.quote(new_text)}"
                )
            )
        case RunAction(command=command, cwd=cwd, timeout_sec=timeout_sec):
            return await env.exec(command=command, cwd=cwd, timeout_sec=timeout_sec)
        case VerifyAction():
            return await env.verify()
    raise TypeError(f"Unsupported action: {type(action).__name__}")


# ============================================================================
# Prompt building
# ============================================================================


def build_system_prompt() -> str:
    return "\n".join(
        [
            "You are controlling a coding environment.",
            "Return one or more tool calls. They execute in order with no intermediate observation, then the resulting state appears in the next turn.",
            "Batch calls only when later calls do not depend on earlier results; otherwise emit one call and wait for the observation.",
            "Use list_dir, find_files, search_text, and read_file before broad edits.",
            "Call verify as soon as you believe the task is complete to confirm success and end the trial.",
        ]
    )


def build_initial_user_prompt(*, instruction: str, working_dir: str | None) -> str:
    return "\n\n".join(
        [
            "Task instruction:",
            instruction,
            f"working_dir: {working_dir or '(unknown)'}",
        ]
    )


def render_tool_result(raw_state: RawState, *, char_limit: int) -> str:
    """Render one observation as the body of a `role:"tool"` message."""
    lines: list[str] = []
    if raw_state.return_code is not None:
        lines.append(f"rc={raw_state.return_code}")
    if raw_state.stdout:
        lines.append("stdout:")
        lines.append(_clip(raw_state.stdout.rstrip("\n"), limit=char_limit))
    if raw_state.stderr:
        lines.append("stderr:")
        lines.append(_clip(raw_state.stderr.rstrip("\n"), limit=char_limit))
    return "\n".join(lines) if lines else "(no output)"


def _clip(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 15] + "\n...[truncated]"


def _replay_tool_call(action: Action, *, step_index: int) -> dict[str, Any]:
    """Synthesize the assistant `tool_call` envelope for a stored Action."""
    match action:
        case ListDirAction(path=path):
            args: dict[str, Any] = {"path": path}
        case FindFilesAction(pattern=pattern, root=root):
            args = {"pattern": pattern, "root": root}
        case SearchTextAction(query=query, root=root):
            args = {"query": query, "root": root}
        case ReadFileAction(path=path, start_line=start_line, end_line=end_line):
            args = {"path": path, "start_line": start_line, "end_line": end_line}
        case WriteFileAction(path=path, content=content):
            args = {"path": path, "content": content}
        case EditFileAction(path=path, old_text=old_text, new_text=new_text):
            args = {"path": path, "old_text": old_text, "new_text": new_text}
        case RunAction(command=command, cwd=cwd, timeout_sec=timeout_sec):
            args = {"command": command, "cwd": cwd, "timeout_sec": timeout_sec}
        case VerifyAction():
            args = {}
        case _:
            raise TypeError(f"unsupported action type {type(action).__name__}")
    args = {k: v for k, v in args.items() if v is not None}
    return {
        "id": f"call_{step_index:04d}",
        "type": "function",
        "function": {
            "name": action.NAME,
            "arguments": json.dumps(args, sort_keys=True, separators=(",", ":")),
        },
    }


def summarize_action(action: Action) -> dict[str, Any]:
    return {key: value for key, value in asdict(action).items() if value is not None}


def build_messages(
    *,
    instruction: str,
    working_dir: str | None,
    trajectory: Trajectory,
    char_limit: int = DEFAULT_RESULT_CHAR_LIMIT,
) -> list[dict[str, Any]]:
    """Render the trial as a multi-turn chat.

    Shape: system, user, then (assistant tool_calls, tool result) per step.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "user",
            "content": build_initial_user_prompt(
                instruction=instruction,
                working_dir=working_dir,
            ),
        },
    ]
    for step_index, (action, raw_state) in enumerate(trajectory, start=1):
        tool_call = _replay_tool_call(action, step_index=step_index)
        messages.append(
            {"role": "assistant", "content": None, "tool_calls": [tool_call]}
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": render_tool_result(raw_state, char_limit=char_limit),
            }
        )
    return messages


# ============================================================================
# Agent loop
# ============================================================================


async def act(
    *,
    llm: BaseLlm,
    instruction: str,
    working_dir: str | None,
    trajectory: Trajectory,
    max_output_retries: int,
    recorder: Any = None,
) -> tuple[Action, ...]:
    """One LLM round-trip. Returns the typed Actions parsed from its tool calls.

    Retries on malformed output (no tool call, bad JSON, validation error) up
    to `max_output_retries` times by appending a single corrective message.
    """
    tools = build_tool_specs()
    base_messages = build_messages(
        instruction=instruction,
        working_dir=working_dir,
        trajectory=trajectory,
    )
    repair_messages: list[dict[str, Any]] = []
    for attempt_index in range(max_output_retries + 1):
        completion = await llm.complete(
            messages=[*base_messages, *repair_messages],
            tools=tools,
        )
        if recorder is not None:
            recorder.completion_received(
                attempt_index=attempt_index,
                request_messages=[*base_messages, *repair_messages],
                request_tools=tools,
                completion=completion,
            )
        if not completion.tool_calls:
            if recorder is not None:
                recorder.action_parse_failed(
                    error="MissingToolCall",
                    detail="model response omitted required tool call",
                )
            repair_messages = [
                {"role": "assistant", "content": completion.content or ""},
                {"role": "user", "content": MISSING_TOOL_CALL_REPAIR_PROMPT},
            ]
            continue
        try:
            actions: list[Action] = []
            for call in completion.tool_calls:
                action_name = _validate_action_name(call.name)
                args = validate_action_args(
                    action_name, parse_action_args(call.arguments)
                )
                actions.append(build_action(action_name, args))
            return tuple(actions)
        except Exception as exc:
            if recorder is not None:
                recorder.action_parse_failed(
                    error=type(exc).__name__,
                    detail=str(exc),
                )
            repair_messages = [
                {
                    "role": "user",
                    "content": (
                        f"Your previous tool call was invalid: {exc}. "
                        "Emit a valid JSON tool call this turn."
                    ),
                }
            ]
    raise RuntimeError("failed to parse a valid action call")


def _validate_action_name(value: str) -> ActionName:
    if value not in ACTION_SPECS:
        raise ValueError(f"unknown action name {value!r}")
    return value  # type: ignore[return-value]


async def run_task_loop(
    *,
    task_name: str,
    llm: BaseLlm,
    env: HarnessEnv,
    reset_state: RawState,
    max_steps: int,
    max_output_retries: int = 2,
    recorder: Any = None,
    progress: TaskLoopProgress | None = None,
) -> TaskLoopResult:
    """Run the agent loop after environment reset.

    Lifecycle concerns (reset, timeout, artifacts, cleanup, timestamps) live
    outside the harness; this core loop only decides and executes actions.
    """
    del task_name
    trajectory: Trajectory = ()
    steps_used = 0
    reward: float = 0.0
    final_passed: bool | None = None
    done = False
    if progress is not None:
        progress.reward = reward
        progress.steps_used = steps_used
        progress.final_passed = final_passed
    while steps_used < max_steps and not done:
        step_index = steps_used + 1
        step_recorder = None if recorder is None else recorder.for_step(step_index)
        actions = await act(
            llm=llm,
            instruction=reset_state.instruction,
            working_dir=reset_state.working_dir,
            trajectory=trajectory,
            max_output_retries=max_output_retries,
            recorder=step_recorder,
        )
        for action in actions:
            if done or steps_used >= max_steps:
                break
            step_index = steps_used + 1
            step_recorder = None if recorder is None else recorder.for_step(step_index)
            action_summary = summarize_action(action)
            if step_recorder is not None:
                step_recorder.action_chosen(
                    action_name=action.NAME,
                    action_summary=action_summary,
                )
            raw_state = await execute_action(env, action)
            if step_recorder is not None:
                step_recorder.env_step_completed(
                    action_name=action.NAME,
                    action_summary=action_summary,
                    raw_state=raw_state,
                )
            trajectory = (*trajectory, (action, raw_state))
            steps_used += 1
            done = raw_state.done
            if raw_state.reward is not None:
                reward = raw_state.reward
            if raw_state.done and raw_state.passed is not None:
                final_passed = raw_state.passed
            if progress is not None:
                progress.reward = reward
                progress.steps_used = steps_used
                progress.final_passed = final_passed
    solved = final_passed is True
    return TaskLoopResult(
        reward=reward,
        solved=solved,
        steps_used=steps_used,
        final_passed=final_passed,
    )
