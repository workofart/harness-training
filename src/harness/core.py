"""
Harness policy core.

Public critical path from `uv run exp`:
- `src.cli.main_exp()` loads runtime config and runs the experiment
  orchestrator over the task set.
- `src.experiment.orchestrator` schedules trials (two-level concurrency,
  admission, majority early-stop) and aggregates them into an `ExperimentResult`.
- `src.experiment.executor.run_trial()` owns one trial: reset, the two timeouts
  (env-setup vs agent), the verify-ceiling env wrapper, classification, cleanup,
  and the returned `TrialResult`.
- `run_task_loop()` owns the post-reset policy/environment loop.

This module defines the loop-local design:
- 8-action vocabulary and model-facing tool specs
- JSON tool-call parse, validation, and typed action construction
- action execution as one shell command or verifier call through `HarnessEnv`
- prompt replay from `(Action, RawState)` trajectory into chat messages
- bounded observation rendering for stdout/stderr
- `act()`: one model completion attempt plus configurable repair retries
- `run_task_loop()`: execute emitted actions in order until a verify action
  (the terminal judgment) or `max_steps`, updating the optional trace/metrics
  recorder and the caller-owned `TaskLoopState`

Boundary contracts:
- Environment adapters implement `src.contracts.HarnessEnv`.
- LLM adapters implement `src.llm.base.BaseLlm`.
- Trial results cross back to the orchestrator as
  `src.experiment.record.TrialResult`.
"""

from __future__ import annotations

import json
import shlex
from abc import ABC
from dataclasses import MISSING, asdict, dataclass, fields
from typing import Any, ClassVar, Literal, TypeAlias, get_args, get_type_hints

from src.llm.base import BaseLlm
from src.contracts import EnvExecWorkload, HarnessEnv, RawState
from src.trace import NOOP_RECORDER, Recorder


# ============================================================================
# Constants
# ============================================================================

MISSING_TOOL_CALL_REPAIR_PROMPT = (
    "Your previous response omitted the required tool call. "
    "Return one or more tool calls and no natural language."
)
DEFAULT_READ_WINDOW_LINES = 200
DEFAULT_RESULT_CHAR_LIMIT = 6000
SHORT_RUN_LIGHT_TIMEOUT_SEC = 30


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
JsonScalarType: TypeAlias = Literal["integer", "string"]

Trajectory: TypeAlias = tuple[tuple["Action", "RawState"], ...]


@dataclass(frozen=True, slots=True)
class Action(ABC):
    """Base for the 8 typed actions.

    Declared a frozen+slots dataclass — matching every subclass — so the base
    itself is a dataclass type. That makes the "every Action is a dataclass"
    invariant true at the type level, so `fields()`/`asdict()` over the action
    classes type-check. `frozen` must agree with the subclasses (Python forbids
    mixing frozen and non-frozen across a dataclass hierarchy).
    """

    NAME: ClassVar[ActionName]
    DESCRIPTION: ClassVar[str]


@dataclass(frozen=True, slots=True)
class ListDirAction(Action):
    NAME: ClassVar[Literal["list_dir"]] = "list_dir"
    DESCRIPTION: ClassVar[str] = "List directory contents."
    path: str | None = None


@dataclass(frozen=True, slots=True)
class FindFilesAction(Action):
    NAME: ClassVar[Literal["find_files"]] = "find_files"
    DESCRIPTION: ClassVar[str] = "Find files by filename pattern."
    pattern: str
    root: str | None = None


@dataclass(frozen=True, slots=True)
class SearchTextAction(Action):
    NAME: ClassVar[Literal["search_text"]] = "search_text"
    DESCRIPTION: ClassVar[str] = "Search file contents for text."
    query: str
    root: str | None = None


@dataclass(frozen=True, slots=True)
class ReadFileAction(Action):
    NAME: ClassVar[Literal["read_file"]] = "read_file"
    DESCRIPTION: ClassVar[str] = "Read a file, optionally by line range."
    path: str
    start_line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True, slots=True)
class WriteFileAction(Action):
    NAME: ClassVar[Literal["write_file"]] = "write_file"
    DESCRIPTION: ClassVar[str] = "Write the full contents of a file."
    path: str
    content: str


@dataclass(frozen=True, slots=True)
class EditFileAction(Action):
    NAME: ClassVar[Literal["edit_file"]] = "edit_file"
    DESCRIPTION: ClassVar[str] = "Replace one exact text span in a file."
    path: str
    old_text: str
    new_text: str


@dataclass(frozen=True, slots=True)
class RunAction(Action):
    NAME: ClassVar[Literal["run"]] = "run"
    DESCRIPTION: ClassVar[str] = "Run one shell command."
    command: str
    cwd: str | None = None
    timeout_sec: int | None = None


@dataclass(frozen=True, slots=True)
class VerifyAction(Action):
    NAME: ClassVar[Literal["verify"]] = "verify"
    DESCRIPTION: ClassVar[str] = (
        "Ask the environment for the authoritative task judgment."
    )


@dataclass(slots=True)
class TaskLoopState:
    """Mutable per-trial loop outcome, owned by the caller of `run_task_loop`.

    The loop updates this in place after every action, so the caller can read
    the last observed outcome on both the normal-return path and the path where
    an outer `asyncio.timeout` cancels the loop mid-flight (a returned value
    would be lost there). `solved` is derived from `final_passed`, so the loop
    never sets it independently.
    """

    reward: float = 0.0
    steps_used: int = 0
    final_passed: bool | None = None

    @property
    def solved(self) -> bool:
        return self.final_passed is True


# ============================================================================
# Action specs and tool specs
# ============================================================================


# The 8 action dataclasses above are the single source of truth for action
# structure, descriptions, required/optional keys, and JSON scalar types.
ACTION_CLASSES: tuple[type[Action], ...] = (
    ListDirAction,
    FindFilesAction,
    SearchTextAction,
    ReadFileAction,
    WriteFileAction,
    EditFileAction,
    RunAction,
    VerifyAction,
)
ACTION_BY_NAME: dict[ActionName, type[Action]] = {
    cls.NAME: cls for cls in ACTION_CLASSES
}


@dataclass(frozen=True, slots=True)
class ActionSpec:
    name: ActionName
    description: str
    required_keys: tuple[str, ...]
    optional_keys: tuple[str, ...]
    json_type_by_key: dict[str, JsonScalarType]


def _json_scalar_type(annotation: Any) -> JsonScalarType:
    args = get_args(annotation)
    if annotation is int or (int in args and type(None) in args):
        return "integer"
    return "string"


def _action_spec(cls: type[Action]) -> ActionSpec:
    action_fields = fields(cls)
    type_hints = get_type_hints(cls)
    required = tuple(
        f.name
        for f in action_fields
        if f.default is MISSING and f.default_factory is MISSING
    )
    optional = tuple(
        f.name
        for f in action_fields
        if f.default is not MISSING or f.default_factory is not MISSING
    )
    return ActionSpec(
        name=cls.NAME,
        description=cls.DESCRIPTION,
        required_keys=required,
        optional_keys=optional,
        json_type_by_key={
            f.name: _json_scalar_type(type_hints[f.name]) for f in action_fields
        },
    )


ACTION_SPECS: dict[ActionName, ActionSpec] = {
    cls.NAME: _action_spec(cls) for cls in ACTION_CLASSES
}


def build_tool_specs() -> list[dict[str, Any]]:
    """Render ACTION_SPECS as model-facing tool specs."""
    tools: list[dict[str, Any]] = []
    for name in sorted(ACTION_SPECS):
        spec = ACTION_SPECS[name]
        properties: dict[str, Any] = {}
        for key in (*spec.required_keys, *spec.optional_keys):
            required = key in spec.required_keys
            base = spec.json_type_by_key[key]
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
    required = set(spec.required_keys)
    allowed = required | set(spec.optional_keys)
    missing = [k for k in spec.required_keys if k not in args]
    if missing:
        raise ValueError(f"{action_name}: missing required keys {missing}")
    unknown = sorted(set(args) - allowed)
    if unknown:
        raise ValueError(f"{action_name}: unknown keys {unknown}")
    for key, value_type in spec.json_type_by_key.items():
        if value_type != "integer" or key not in args:
            continue
        value = args[key]
        if value is not None and not isinstance(value, int):
            raise ValueError(f"{key}: expected integer or null")
    # String fields: a required one must be a non-empty string; an optional one
    # may be a string or null. Validated here rather than in build_action so a
    # bad type surfaces as a ValueError inside act()'s repair loop, and
    # build_action can construct the dataclass from already-typed args.
    for key in (*spec.required_keys, *spec.optional_keys):
        if spec.json_type_by_key[key] == "integer" or key not in args:
            continue
        value = args[key]
        if key in required:
            if not isinstance(value, str) or value == "":
                raise ValueError(f"{key}: expected non-empty string")
        elif value is not None and not isinstance(value, str):
            raise ValueError("expected string or null")
    return dict(args)


def build_action(action_name: ActionName, args: dict[str, Any]) -> Action:
    """Construct the typed Action from already-validated args.

    `args` must be the output of `validate_action_args`, which guarantees the
    keys and value types line up with the dataclass fields — so the keyword
    splat into the action class is safe.
    """
    return ACTION_BY_NAME[action_name](**args)


# ============================================================================
# Environment execution
# ============================================================================


async def execute_action(env: HarnessEnv, action: Action) -> RawState:
    match action:
        case ListDirAction(path=path):
            return await env.exec(
                command=f"ls -1Ap {shlex.quote(path or '.')}", workload="light"
            )
        case FindFilesAction(pattern=pattern, root=root):
            return await env.exec(
                command=f"find {shlex.quote(root or '.')} -name {shlex.quote(pattern)}",
                workload="light",
            )
        case SearchTextAction(query=query, root=root):
            return await env.exec(
                command=f"grep -rn {shlex.quote(query)} {shlex.quote(root or '.')}",
                workload="light",
            )
        case ReadFileAction(path=path, start_line=start_line, end_line=end_line):
            start = start_line or 1
            end = (
                end_line
                if end_line is not None
                else start + DEFAULT_READ_WINDOW_LINES - 1
            )
            return await env.exec(
                command=f"sed -n '{start},{end}p' {shlex.quote(path)}",
                workload="light",
            )
        case WriteFileAction(path=path, content=content):
            return await env.exec(
                command=f"printf %s {shlex.quote(content)} > {shlex.quote(path)}",
                workload="light",
            )
        case EditFileAction(path=path, old_text=old_text, new_text=new_text):
            # A non-matching old_text must fail loudly: rewriting the file
            # unchanged with rc=0 reads as a landed edit to the model.
            script = (
                "import sys, pathlib; "
                "p = pathlib.Path(sys.argv[1]); "
                "text = p.read_text(); "
                "sys.exit('old_text not found in ' + sys.argv[1]) "
                "if sys.argv[2] not in text "
                "else p.write_text(text.replace(sys.argv[2], sys.argv[3], 1))"
            )
            return await env.exec(
                command=(
                    f"python3 -c {shlex.quote(script)} "
                    f"{shlex.quote(path)} {shlex.quote(old_text)} {shlex.quote(new_text)}"
                ),
                workload="light",
            )
        case RunAction(command=command, cwd=cwd, timeout_sec=timeout_sec):
            return await env.exec(
                command=command,
                cwd=cwd,
                timeout_sec=timeout_sec,
                workload=_run_workload(timeout_sec),
            )
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
            "Each observation reports time_remaining, your remaining wall-clock budget in seconds (absent only when unbounded); pace your work and verify before it runs out.",
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
    if raw_state.time_remaining_sec is not None:
        # Surface the trial's remaining wall budget so the agent can pace itself
        # and verify before the task timeout cuts it off. `None` means unbounded
        # (no budget configured), so the line is simply omitted.
        lines.append(f"time_remaining: {int(raw_state.time_remaining_sec)}s")
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
    args = summarize_action(action)
    return {
        "id": f"call_{step_index:04d}",
        "type": "function",
        "function": {
            "name": action.NAME,
            # sort_keys keeps the emitted JSON byte-stable (alphabetical), which
            # prompt-cache reuse depends on; do not drop it.
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


class NoValidActionError(RuntimeError):
    """The model failed to emit a parseable tool call within the output-retry
    budget.

    A distinct type (not a bare RuntimeError) so the trial layer can classify it
    as an agent failure (`no_valid_action`) rather than an infra `crash`: the
    model produced nothing usable -- often an empty/refused completion -- which
    is the agent's outcome, not a broken environment.
    """


async def act(
    *,
    llm: BaseLlm,
    instruction: str,
    working_dir: str | None,
    trajectory: Trajectory,
    max_output_retries: int,
    recorder: Recorder = NOOP_RECORDER,
    step_index: int | None = None,
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
        recorder.completion_received(
            step_index=step_index,
            attempt_index=attempt_index,
            request_messages=[*base_messages, *repair_messages],
            request_tools=tools,
            completion=completion,
        )
        if not completion.tool_calls:
            recorder.action_parse_failed(
                step_index=step_index,
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
            recorder.action_parse_failed(
                step_index=step_index,
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
    raise NoValidActionError("failed to parse a valid action call")


def _validate_action_name(value: str) -> ActionName:
    if value not in ACTION_SPECS:
        raise ValueError(f"unknown action name {value!r}")
    return value  # type: ignore[return-value]


def _run_workload(timeout_sec: int | None) -> EnvExecWorkload:
    if timeout_sec is not None and timeout_sec <= SHORT_RUN_LIGHT_TIMEOUT_SEC:
        return "light"
    return "heavy"


async def run_task_loop(
    *,
    llm: BaseLlm,
    env: HarnessEnv,
    reset_state: RawState,
    max_steps: int,
    max_output_retries: int = 2,
    recorder: Recorder = NOOP_RECORDER,
    state: TaskLoopState,
) -> None:
    """Run the agent loop after environment reset, recording into `state`.

    `state` is caller-owned and updated in place after every action, so the
    caller can read the last observed outcome even when an outer
    `asyncio.timeout` cancels this coroutine mid-flight. Lifecycle concerns
    (reset, timeout, artifacts, cleanup, timestamps) live outside the harness;
    this core loop only decides and executes actions. A verify action is the
    terminal judgment, so it ends the loop.
    """
    trajectory: Trajectory = ()
    done = False

    async def run_loop_action(action: Action) -> None:
        nonlocal done, trajectory

        step_index = state.steps_used + 1
        recorder.action_chosen(
            step_index=step_index,
            action_name=action.NAME,
            action_summary=summarize_action(action),
        )
        raw_state = await execute_action(env, action)
        recorder.env_step_completed(
            step_index=step_index,
            action_name=action.NAME,
            raw_state=raw_state,
        )
        trajectory = (*trajectory, (action, raw_state))
        state.steps_used += 1
        if raw_state.reward is not None:
            state.reward = raw_state.reward
        if isinstance(action, VerifyAction):
            done = True
            if raw_state.passed is not None:
                state.final_passed = raw_state.passed

    while state.steps_used < max_steps and not done:
        actions = await act(
            llm=llm,
            instruction=reset_state.instruction,
            working_dir=reset_state.working_dir,
            trajectory=trajectory,
            max_output_retries=max_output_retries,
            recorder=recorder,
            step_index=state.steps_used + 1,
        )
        for action in actions:
            if done or state.steps_used >= max_steps:
                break
            await run_loop_action(action)
