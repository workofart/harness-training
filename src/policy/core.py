"""The editable harness: everything the model sees and how its replies act.

Data path per step: prompt surface (§1) plus the trajectory are assembled and
fitted (§4) into one completion request; the reply is parsed and repaired (§5),
guarded (§3), rendered into environment commands (§2), executed by the frozen
loop, and observed back into the trajectory (§3).

Where to change what:
  §1 PROMPT SURFACE     -- every model-visible string (system, initial, repair)
  §2 ACTION SPACE       -- tool schemas, descriptions, and shell rendering
  §3 AGENT              -- step policy: retries, thinking, rules/guards seams
  §4 ASSEMBLY & WINDOW  -- request rendering, clipping, context fitting
  §5 PARSING & REPAIR   -- completion -> validated actions; repair turns

Substrate code stays mechanism-free; rollout ownership lives in
`src.rollout.episode` and env transitions in `src.env.base`.
"""

from __future__ import annotations

import base64
import inspect
import json
import shlex
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.env.base import (
    RawEnvOutput,
    RunAction,
    StepResult,
)
from src.llm.backend import (
    Completion,
    CompletionBackend,
    CompletionRequest,
    ContextWindowExceededError,
    ProviderRejectedToolCallError,
)
from src.llm.token_counter import (
    TRUNCATION_MARKER,
    HfTokenCounter,
    resolve_token_counter,
)
from src.policy.base import (
    NOOP_AGENT_CALLBACK,
    AgentCallback,
    NoValidActionError,
    RepeatedLengthCutoffError as RepeatedLengthCutoffError,
    SUBMIT_ACTION_NAME,
)

# §1 Prompt surface


def build_system_prompt() -> str:
    return "\n".join(
        [
            "You are controlling a coding environment.",
            "Use run to execute shell commands in the environment.",
            "Use read to view file contents.",
            "run output is scratch: editing a file inside a `python` snippet or a here-doc that only prints does NOT persist your fix.",
            f"Each run command is terminated after its timeout_sec (default {COMMAND_TIMEOUT_SEC}s) and returns the output captured so far.",
            "Observation text may mask volatile tokens as <TIME>, <PID>, <BINARY_STDOUT>, and similar placeholders; treat them as framework redactions of unstable runtime data.",
            "To change a file on disk, call write (create or overwrite a whole file) or replace (one exact old_text edit in an existing file); the on-disk state at submit is the only thing graded.",
            "Return one or more tool calls. They execute in order with no intermediate observation, then the resulting observation appears in the next turn.",
            "Call submit when the solution is ready for the authoritative task judgment; submit ends the trial.",
        ]
    )


def initial_user_prompt(instruction: str, working_dir: str | None) -> str:
    return "\n\n".join(
        [
            "Task instruction:",
            instruction,
            f"working_dir: {working_dir or '(unknown)'}",
        ]
    )


MISSING_TOOL_CALL_REPAIR_PROMPT = (
    "Your previous response omitted the required tool call. "
    "Return one or more tool calls and no natural language."
)

INVALID_TOOL_CALL_REPAIR_PROMPT = (
    "Your previous tool call was invalid: {error}\n"
    "Emit a corrected JSON tool call this turn."
)

CONTEXT_OMITTED_NOTE_TEMPLATE = (
    "[{dropped} earlier step(s) omitted to fit the context window; "
    "continue from the steps below.]"
)

# §2 Action space and tools

COMMAND_TIMEOUT_SEC = 300
ActionName: TypeAlias = Literal["run", "submit", "replace", "read", "write"]


class ToolArgs(BaseModel):
    # Coerce unambiguous model near-misses like '"120"'; keep unknown fields forbidden.
    model_config = ConfigDict(extra="forbid", frozen=True)


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: ActionName
    args: ToolArgs


@dataclass(frozen=True, slots=True)
class Tool:
    """Model-facing spec plus optional renderer into an environment `RunAction`.

    None means the rollout loop owns the action, e.g. submit.
    """

    description: str
    args_model: type[ToolArgs]
    to_run: Callable[[Any], RunAction] | None = None


class RunArgs(ToolArgs):
    command: str = Field(min_length=1)
    cwd: str | None = None
    timeout_sec: int = Field(
        default=COMMAND_TIMEOUT_SEC,
        ge=1,
        description=(
            "Seconds this command may run before it is terminated and its output "
            f"so far is returned to you. Omit to use the {COMMAND_TIMEOUT_SEC}s "
            "default."
        ),
    )


FILE_TOOL_TIMEOUT_SEC = 30


class ReadArgs(ToolArgs):
    path: str = Field(
        min_length=1,
        description="File to read; relative paths resolve against the task working directory.",
    )
    offset: int = Field(
        default=1,
        ge=1,
        description="1-based line number to start reading from.",
    )
    limit: int = Field(
        default=250,
        ge=1,
        le=2000,
        description="Maximum number of lines to return.",
    )


def build_read_command(args: ReadArgs) -> str:
    qpath = shlex.quote(args.path)
    end = args.offset + args.limit - 1
    return (
        f"sed -n '{args.offset},{end}p' < {qpath} && "
        f"printf '(read: lines {args.offset}-{end}; file has %s lines total)\\n' "
        f'"$(($(wc -l < {qpath})))"'
    )


class WriteArgs(ToolArgs):
    path: str = Field(
        min_length=1,
        description=(
            "File to create or overwrite; relative paths resolve against the task "
            "working directory. Parent directories are created."
        ),
    )
    content: str = Field(
        description="Full file content, written to disk exactly as given."
    )


def build_write_command(args: WriteArgs) -> str:
    qpath = shlex.quote(args.path)
    blob = base64.b64encode(args.content.encode("utf-8")).decode("ascii")
    return (
        f'mkdir -p -- "$(dirname -- {qpath})" && '
        f"printf '%s' '{blob}' | base64 -d > {qpath} && "
        f"printf 'write: wrote %s bytes to %s\\n' \"$(($(wc -c < {qpath})))\" {qpath}"
    )


class ReplaceArgs(ToolArgs):
    path: str = Field(
        min_length=1,
        description="File to edit; relative paths resolve against the task working directory.",
    )
    old_text: str = Field(
        min_length=1,
        description=(
            "Exact existing text to replace, whitespace and indentation included; "
            "must occur exactly once in the file."
        ),
    )
    new_text: str = Field(description="Replacement text written in old_text's place.")


def _replace_once(path, old_text, new_text):
    import os
    import sys

    if not os.path.exists(path):
        raise SystemExit("replace: file not found: " + path)
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    count = text.count(old_text)
    if count == 0:
        raise SystemExit("replace: old_text not found in " + path)
    if count > 1:
        raise SystemExit(
            "replace: old_text matches %d locations in %s; extend old_text with "
            "surrounding lines to make it unique" % (count, path)
        )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text.replace(old_text, new_text, 1))
    sys.stdout.write("replace: updated " + path + "\n")


_REPLACE_SRC = inspect.getsource(_replace_once)


def build_replace_command(args: ReplaceArgs) -> str:
    payload = tuple(
        base64.b64encode(text.encode("utf-8")).decode("ascii")
        for text in (args.path, args.old_text, args.new_text)
    )
    script = _REPLACE_SRC + (
        "\nimport base64\n"
        "_replace_once(*[base64.b64decode(a).decode('utf-8') for a in (%r, %r, %r)])\n"
        % payload
    )
    blob = base64.b64encode(script.encode("utf-8")).decode("ascii")
    inner = "import base64;exec(base64.b64decode('" + blob + "').decode())"
    return (
        'PYBIN=python3; command -v "$PYBIN" >/dev/null 2>&1 || PYBIN=python; '
        '"$PYBIN" -c "' + inner + '"'
    )


class SubmitArgs(ToolArgs):
    pass


TOOLS: dict[ActionName, Tool] = {
    "run": Tool(
        description="Run one shell command.",
        args_model=RunArgs,
        to_run=lambda args: RunAction(
            command=args.command,
            cwd=args.cwd,
            timeout_sec=args.timeout_sec,
        ),
    ),
    "replace": Tool(
        description=(
            "Persist a code change to disk by replacing text in one file. "
            "`old_text` must match the file exactly (whitespace and indentation "
            "included) and occur exactly once; the call fails loudly when it is "
            "missing or ambiguous -- extend old_text with surrounding lines to "
            "disambiguate. To edit several places, return several replace calls."
        ),
        args_model=ReplaceArgs,
        to_run=lambda args: RunAction(
            command=build_replace_command(args),
            cwd=None,
            timeout_sec=FILE_TOOL_TIMEOUT_SEC,
        ),
    ),
    "read": Tool(
        description=(
            "Read a window of one file as plain text. Defaults to the first 250 "
            "lines; page through longer files with offset/limit."
        ),
        args_model=ReadArgs,
        to_run=lambda args: RunAction(
            command=build_read_command(args),
            cwd=None,
            timeout_sec=FILE_TOOL_TIMEOUT_SEC,
        ),
    ),
    "write": Tool(
        description=(
            "Create or overwrite one file on disk with the given content (parent "
            "directories are created). For a small targeted edit to an existing "
            "file, prefer replace."
        ),
        args_model=WriteArgs,
        to_run=lambda args: RunAction(
            command=build_write_command(args),
            cwd=None,
            timeout_sec=FILE_TOOL_TIMEOUT_SEC,
        ),
    ),
    SUBMIT_ACTION_NAME: Tool(
        description="Submit the current solution for grading.",
        args_model=SubmitArgs,
    ),
}


def build_tool_specs() -> list[dict[str, Any]]:
    """Build model-facing tool specs."""
    specs: list[dict[str, Any]] = []
    for name in sorted(TOOLS):
        tool = TOOLS[name]
        parameters = tool.args_model.model_json_schema()
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": parameters,
                },
            }
        )
    return specs


def build_env_action(action: Action) -> RunAction:
    """Render a non-submit action as an environment run."""
    to_run = TOOLS[action.name].to_run
    if to_run is None:
        raise ValueError(f"unsupported editable action: {action.name!r}")
    return to_run(action.args)


# §3 Agent


@dataclass(frozen=True, slots=True)
class ReminderRule:
    """Step-start intervention rule.

    `build` returns injected messages, or [] when it does not fire.
    `_reminder_messages` traces every evaluation.
    """

    name: str
    build: Callable[["LlmAgent", int], list[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class ActionGuard:
    """Parsed-action transformation rule.

    `apply` returns the batch, possibly rewritten. `_guard_actions` traces every
    evaluation.
    """

    name: str
    apply: Callable[["LlmAgent", tuple["Action", ...]], tuple["Action", ...]]


@dataclass(slots=True)
class LlmAgent:
    # Repeated whole-budget thinking runaways disable thinking for later steps.
    LENGTH_CUTOFF_THINKING_OFF_THRESHOLD: ClassVar[int] = 2

    llm: CompletionBackend
    max_context_length: int
    max_completion_tokens: int
    max_output_retries: int = 2
    events: AgentCallback = NOOP_AGENT_CALLBACK
    thinking_toggleable: bool = False
    # If set, measure/clip with model tokens so token-dense content stays bounded.
    token_counter: HfTokenCounter | None = None
    _initial_env_output: RawEnvOutput = field(init=False)
    _trajectory: Trajectory = field(init=False, default=())
    # One build per rollout; the window counts the same specs the request sends.
    _tools: list[dict[str, Any]] = field(init=False)
    _request_builder: _RequestBuilder = field(init=False)
    # Per-rollout; calibration must not leak across rollouts.
    _context_window: _ContextWindow = field(init=False)
    # Prevent one thinking runaway from recurring later in the rollout.
    _thinking_disabled: bool = field(init=False, default=False)
    _length_cutoff_missing_tool_calls: int = field(init=False, default=0)

    def reset(self, raw_env_output: RawEnvOutput) -> None:
        self._initial_env_output = raw_env_output
        self._trajectory = ()
        self._thinking_disabled = False
        self._length_cutoff_missing_tool_calls = 0
        self._tools = build_tool_specs()
        self._request_builder = _RequestBuilder(
            token_counter=self.token_counter,
        )
        self._context_window = _ContextWindow(
            max_context_length=self.max_context_length,
            max_completion_tokens=self.max_completion_tokens,
            tools=self._tools,
            token_counter=self.token_counter,
        )

    def observe(self, action: Action, step_result: StepResult) -> None:
        if not (step_result.terminated or step_result.truncated):
            self._trajectory = (*self._trajectory, (action, step_result.raw_env_output))
            step_index = len(self._trajectory)
            raw = step_result.raw_env_output
            for field_name in ("stdout", "stderr"):
                text = getattr(raw, field_name)
                if text and self._request_builder.would_clip(text.rstrip("\n")):
                    self.events(
                        "observation_clipped", step_index=step_index, field=field_name
                    )

    def _reminder_messages(self, step_index: int) -> list[dict[str, Any]]:
        """Evaluate all step-start rules, returning the first firing messages.

        Later rules still run so fired/not-fired events are traced.
        """
        chosen: list[dict[str, Any]] = []
        for rule in REMINDER_RULES:
            messages = rule.build(self, step_index)
            self.events(
                "policy_rule",
                step_index=step_index,
                rule=rule.name,
                fired=bool(messages),
            )
            if messages and not chosen:
                chosen = messages
        return chosen

    async def act(self) -> tuple[Action, ...]:
        """Complete and parse one action batch."""
        step_index = len(self._trajectory) + 1
        # The active correction must remain the final turn.
        turn_messages = self._reminder_messages(step_index)
        last_error = "unknown parse error"
        enable_thinking: bool | None = False if self._thinking_disabled else None
        attempt_index = 0
        offered = frozenset(spec["function"]["name"] for spec in self._tools)
        for attempt_index in range(self.max_output_retries + 1):
            try:
                completion = await self._complete_fitted(
                    turn_messages,
                    step_index=step_index,
                    attempt_index=attempt_index,
                    enable_thinking=enable_thinking,
                )
            except ProviderRejectedToolCallError as exc:
                last_error = str(exc)
                self.events(
                    "action_parse_failed",
                    step_index=step_index,
                    error=type(exc).__name__,
                    detail=str(exc),
                )
                turn_messages = [
                    {
                        "role": "user",
                        "content": INVALID_TOOL_CALL_REPAIR_PROMPT.format(error=exc),
                    }
                ]
                continue
            try:
                actions = ActionParser.actions(completion, offered=offered)
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)
                self.events(
                    "action_parse_failed",
                    step_index=step_index,
                    error=type(exc).__name__,
                    detail=str(exc),
                )
                if completion.finish_reason == "length":
                    # A thinking-off retry will decode identically on deterministic
                    # endpoints, so stop this step instead of spending another attempt.
                    if enable_thinking is False:
                        break
                    if isinstance(exc, MissingToolCall):
                        self._length_cutoff_missing_tool_calls += 1
                        if (
                            self._length_cutoff_missing_tool_calls
                            >= self.LENGTH_CUTOFF_THINKING_OFF_THRESHOLD
                            and self.thinking_toggleable
                        ):
                            self._thinking_disabled = True
                    elif not self.thinking_toggleable:
                        break
                    if self.thinking_toggleable:
                        # One runaway only affects this step; repeated missing calls
                        # arm the rollout-wide breaker above.
                        enable_thinking = False
                turn_messages = ActionParser.repair_messages(completion, exc)
            else:
                return self._guard_actions(actions, step_index=step_index)
        self._trajectory = (
            *self._trajectory,
            FailedStep(attempts=attempt_index + 1, error=last_error),
        )
        raise NoValidActionError("failed to parse a valid action call")

    def _guard_actions(
        self, actions: tuple[Action, ...], *, step_index: int
    ) -> tuple[Action, ...]:
        """Apply action guards left to right."""
        for guard in ACTION_GUARDS:
            guarded = guard.apply(self, actions)
            self.events(
                "policy_rule",
                step_index=step_index,
                rule=guard.name,
                fired=guarded != actions,
            )
            actions = guarded
        return actions

    async def _complete_fitted(
        self,
        turn_messages: list[dict[str, Any]],
        *,
        step_index: int,
        attempt_index: int,
        enable_thinking: bool | None,
    ) -> Completion:
        """Complete once, re-trimming on context overflow."""
        # Assembly is calibration-independent; only the fit tightens on retrim.
        head, step_groups, floor = self._request_builder.assemble(
            instruction=self._initial_env_output.instruction,
            working_dir=self._initial_env_output.working_dir,
            trajectory=self._trajectory,
        )
        last_error: ContextWindowExceededError | None = None
        for _ in range(self._context_window.RETRIM_BUDGET + 1):
            base_messages, dropped = self._context_window.fit(
                head, step_groups, floor, turn_messages=turn_messages
            )
            if dropped > 0:
                self.events(
                    "context_groups_dropped",
                    step_index=step_index,
                    dropped=dropped,
                )
            request = [*base_messages, *turn_messages]
            try:
                completion = await self.llm.complete(
                    CompletionRequest(
                        messages=request,
                        tools=self._tools,
                        enable_thinking=enable_thinking,
                    )
                )
            except ContextWindowExceededError as exc:
                last_error = exc
                if not self._context_window.shrink_for_overflow(exc, request):
                    raise
                self.events(
                    "context_window_retrimmed",
                    step_index=step_index,
                    attempt_index=attempt_index,
                    limit=exc.limit,
                    requested=exc.requested,
                    calibration=self._context_window.calibration,
                )
                continue
            self._context_window.calibrate_from_usage(completion, request)
            return completion
        assert last_error is not None
        raise last_error


def build_policy(
    llm: CompletionBackend,
    events: AgentCallback,
    *,
    max_context_length: int,
    max_completion_tokens: int,
    thinking_toggleable: bool,
    tokenizer_name: str | None,
    model_name: str,
) -> LlmAgent:
    """The one place the measurement path obtains a concrete harness policy.

    The frozen runner resolves this export by name on the configured
    training-target module, so construction wiring is candidate-tunable but
    the signature is the contract: these scalars are everything the harness may know
    about the run.
    """
    return LlmAgent(
        llm=llm,
        max_context_length=max_context_length,
        max_completion_tokens=max_completion_tokens,
        thinking_toggleable=thinking_toggleable,
        token_counter=resolve_token_counter(tokenizer_name, model_name),
        events=events,
    )


# The agent cannot see the committed 110-step cap. This threshold is above every
# currently-solved submit, so it only nudges trials otherwise likely to exhaust it.
LATE_RUN_SUBMIT_REMINDER_STEP = 100


def _late_run_submit_reminder(
    _agent: "LlmAgent", step_index: int
) -> list[dict[str, Any]]:
    if step_index < LATE_RUN_SUBMIT_REMINDER_STEP:
        return []
    return [
        {
            "role": "user",
            "content": (
                "You are in the final steps of this run, which ends automatically very "
                "soon. Only the files on disk at the moment you call submit are graded; "
                "running more commands or re-checking a fix you already made does not "
                "improve that grade. If your fix is already written to disk, call submit "
                "now. Otherwise make your single most important remaining edit and then "
                "submit -- a run that ends without a submit scores nothing."
            ),
        }
    ]


NO_EDIT_REMINDER_STEP = 70


def _no_edit_reminder(agent: "LlmAgent", step_index: int) -> list[dict[str, Any]]:
    if not (NO_EDIT_REMINDER_STEP <= step_index < LATE_RUN_SUBMIT_REMINDER_STEP):
        return []
    if _trajectory_has_persisted_edit(agent._trajectory):
        return []
    return [
        {
            "role": "user",
            "content": (
                "You have taken many steps without writing any change to a file on disk. "
                "Only the on-disk state when you call submit is graded -- exploring, "
                "reproducing, or running commands does not by itself change any file, and "
                "a run that ends with no edit on disk scores nothing. If you have already "
                "identified the fix, apply it now with write or replace. If you have not, "
                "stop broadening the investigation and narrow in on the exact file and "
                "lines to change, leaving yourself enough remaining steps to verify the "
                "edit before this run ends."
            ),
        }
    ]


REMINDER_RULES: tuple[ReminderRule, ...] = (
    ReminderRule(name="late_run_submit", build=_late_run_submit_reminder),
    ReminderRule(name="no_edit_progress", build=_no_edit_reminder),
)

# Last-resort compulsion inside the 110-step horizon. It only preserves an
# existing on-disk edit; trials with nothing to grade remain untouched.
FORCED_FINALIZE_STEP = 109
MULTIFILE_SUBMIT_REVIEW_STEP = 60
MULTIFILE_SUBMIT_REVIEW_COMMAND = (
    "printf '%s\\n' 'pre-submit review: first submit for a multi-file patch.'; "
    "printf '%s\\n' 'Keep the production diff minimal; do not rely on test-only "
    "changes, broad shared-code edits, or unrelated cleanup to make the grade pass.'; "
    "git diff --stat --; "
    "git diff --check --"
)
GIT_STATE_GUARD_COMMAND = (
    "printf '%s\\n' 'git state mutation skipped: an edit is already on disk.'; "
    "printf '%s\\n' 'Keep the working tree patch in place; inspect it with git diff "
    "or read, run tests against the edited tree, then submit when ready.'; "
    "git diff --stat --; "
    "git diff --check --"
)
GIT_STATE_MUTATION_MARKERS = (
    "git stash",
    "git checkout --",
    "git restore",
    "git reset --hard",
    "git clean",
)


def _edit_paths_in_actions(actions: tuple["Action", ...]) -> frozenset[str]:
    paths: set[str] = set()
    for action in actions:
        if action.name == "write":
            paths.add(cast(WriteArgs, action.args).path)
        elif action.name == "replace":
            paths.add(cast(ReplaceArgs, action.args).path)
    return frozenset(paths)


def _persisted_edit_paths(trajectory: Trajectory) -> frozenset[str]:
    paths: set[str] = set()
    for step in trajectory:
        if isinstance(step, FailedStep):
            continue
        action, _ = step
        paths.update(_edit_paths_in_actions((action,)))
    return frozenset(paths)


def _trajectory_has_persisted_edit(trajectory: Trajectory) -> bool:
    return bool(_persisted_edit_paths(trajectory))


def _trajectory_has_multifile_submit_review(trajectory: Trajectory) -> bool:
    for step in trajectory:
        if isinstance(step, FailedStep):
            continue
        action, _ = step
        if (
            action.name == "run"
            and cast(RunArgs, action.args).command == MULTIFILE_SUBMIT_REVIEW_COMMAND
        ):
            return True
    return False


def _is_git_state_mutation(action: "Action") -> bool:
    if action.name != "run":
        return False
    command = cast(RunArgs, action.args).command
    return any(marker in command for marker in GIT_STATE_MUTATION_MARKERS)


def _git_state_guard(
    agent: "LlmAgent", actions: tuple["Action", ...]
) -> tuple["Action", ...]:
    if not _trajectory_has_persisted_edit(agent._trajectory):
        return actions
    if not any(_is_git_state_mutation(action) for action in actions):
        return actions
    guard = Action(
        name="run",
        args=RunArgs(command=GIT_STATE_GUARD_COMMAND, timeout_sec=30),
    )
    return tuple(
        guard if _is_git_state_mutation(action) else action for action in actions
    )


def _multifile_submit_review_guard(
    agent: "LlmAgent", actions: tuple["Action", ...]
) -> tuple["Action", ...]:
    step_index = len(agent._trajectory) + 1
    if not (MULTIFILE_SUBMIT_REVIEW_STEP <= step_index < LATE_RUN_SUBMIT_REMINDER_STEP):
        return actions
    if not any(action.name == SUBMIT_ACTION_NAME for action in actions):
        return actions
    if _trajectory_has_multifile_submit_review(agent._trajectory):
        return actions
    edit_paths = _persisted_edit_paths(agent._trajectory) | _edit_paths_in_actions(
        actions
    )
    if len(edit_paths) < 2:
        return actions
    review = Action(
        name="run",
        args=RunArgs(command=MULTIFILE_SUBMIT_REVIEW_COMMAND, timeout_sec=30),
    )
    return tuple(
        review if action.name == SUBMIT_ACTION_NAME else action for action in actions
    )


def _forced_finalize_guard(
    agent: "LlmAgent", actions: tuple["Action", ...]
) -> tuple["Action", ...]:
    step_index = len(agent._trajectory) + 1
    if step_index < FORCED_FINALIZE_STEP:
        return actions
    if any(action.name == SUBMIT_ACTION_NAME for action in actions):
        return actions
    if not _trajectory_has_persisted_edit(agent._trajectory):
        return actions
    return (Action(name=SUBMIT_ACTION_NAME, args=SubmitArgs()),)


ACTION_GUARDS: tuple[ActionGuard, ...] = (
    ActionGuard(name="git_state_guard", apply=_git_state_guard),
    ActionGuard(name="multifile_submit_review", apply=_multifile_submit_review_guard),
    ActionGuard(name="forced_finalize", apply=_forced_finalize_guard),
)


# §4 Request assembly and context window


@dataclass(frozen=True, slots=True)
class FailedStep:
    attempts: int
    error: str


TrajectoryStep: TypeAlias = tuple["Action", "RawEnvOutput"] | FailedStep
Trajectory: TypeAlias = tuple[TrajectoryStep, ...]


@dataclass(frozen=True, slots=True)
class _RequestBuilder:
    """Renders a rollout's trajectory into chat messages.

    Pure and stateless: owns observation formatting and per-field content
    clipping (prompt text lives in §1). Emits `(head, step_groups, floor)`; the
    retention `floor` is a rendering fact, so this class owns it while
    `_ContextWindow` stays unaware of rendering and owns only token budgeting.
    """

    RESULT_CHAR_LIMIT: ClassVar[int] = 6000 * 4
    # Tokenizer clipping caps observations in server units; tuned against binary-dense
    # (~1 char/token) payloads in the untrimmable newest step.
    RESULT_TOKEN_LIMIT: ClassVar[int] = 6000
    # Replayed tool-call args are model-visible and recoverable, so use a tighter cap.
    ARG_TOKEN_LIMIT: ClassVar[int] = 1000
    ARG_CHAR_LIMIT: ClassVar[int] = ARG_TOKEN_LIMIT * 4

    token_counter: HfTokenCounter | None = None

    def tool_result(self, raw_env_output: RawEnvOutput) -> str:
        lines: list[str] = []
        if raw_env_output.exit_code is not None:
            lines.append(f"rc={raw_env_output.exit_code}")
        if raw_env_output.stdout:
            lines.append("stdout:")
            lines.append(self._clip(raw_env_output.stdout.rstrip("\n")))
        if raw_env_output.stderr:
            lines.append("stderr:")
            lines.append(self._clip(raw_env_output.stderr.rstrip("\n")))
        return "\n".join(lines) if lines else "(no output)"

    def failed_step_result(self, failed_step: FailedStep) -> str:
        return "\n".join(
            [
                (
                    "Previous step produced no valid action after "
                    f"{failed_step.attempts} attempt(s)."
                ),
                "Last validation error:",
                self._clip(failed_step.error),
            ]
        )

    def assemble(
        self,
        *,
        instruction: str,
        working_dir: str | None,
        trajectory: Trajectory,
    ) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]], int]:
        """Render the fixed head, one message group per step, and the retention floor.

        A step group is the atomic unit `_ContextWindow` keeps or drops together.
        `floor` = newest real (executed) step; never dropped past, else a lone
        trailing FailedStep note blinds an already-failing agent. No real step
        yet: the last group, so the request is never empty.

        Replay is canonically sequential: every executed action becomes its own
        single-call assistant/tool pair, so a multi-call batch is deliberately
        replayed as consecutive turns, not the original batched assistant
        message. This rendering feeds the completion cache key, so changing its
        shape forks replay identity from the first affected request.
        """
        head: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": build_system_prompt(),
            },
            {
                "role": "user",
                "content": initial_user_prompt(instruction, working_dir),
            },
        ]
        step_groups: list[list[dict[str, Any]]] = []
        newest_real_step = -1
        for step_index, step in enumerate(trajectory, start=1):
            if isinstance(step, FailedStep):
                step_groups.append(
                    [
                        {
                            "role": "user",
                            "content": self.failed_step_result(step),
                        }
                    ]
                )
                continue
            action, raw_env_output = step
            tool_call = self._replay_tool_call(action, step_index=step_index)
            newest_real_step = len(step_groups)
            step_groups.append(
                [
                    {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": self.tool_result(raw_env_output),
                    },
                ]
            )
        floor = newest_real_step if newest_real_step >= 0 else len(step_groups) - 1
        return head, step_groups, floor

    def _clip(self, text: str) -> str:
        return self._bounded(text, self.RESULT_TOKEN_LIMIT, self.RESULT_CHAR_LIMIT)

    def would_clip(self, text: str) -> bool:
        if self.token_counter is not None:
            return self.token_counter.count(text) > self.RESULT_TOKEN_LIMIT
        return len(text) > self.RESULT_CHAR_LIMIT

    def _bounded(self, text: str, token_limit: int, char_limit: int) -> str:
        if self.token_counter is not None:
            return self.token_counter.truncate(text, token_limit)
        if len(text) <= char_limit:
            return text
        return text[: char_limit - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER

    def _replay_tool_call(self, action: Action, *, step_index: int) -> dict[str, Any]:
        # Clip string values before JSON serialization so replayed calls stay valid JSON.
        args = {
            key: (
                self._bounded(value, self.ARG_TOKEN_LIMIT, self.ARG_CHAR_LIMIT)
                if isinstance(value, str)
                else value
            )
            for key, value in action.args.model_dump(exclude_none=True).items()
        }
        return {
            "id": f"call_{step_index:04d}",
            "type": "function",
            "function": {
                "name": action.name,
                "arguments": json.dumps(args, sort_keys=True, separators=(",", ":")),
            },
        }


@dataclass(slots=True)
class _ContextWindow:
    """Per-rollout context-window state and request-fitting policy.

    Consumes the builder's `(head, step_groups, floor)` and knows nothing about
    how they were rendered; owns token estimation, group-dropping to fit the reply
    budget, and calibration from server usage. `calibration` mutates per rollout,
    so LlmAgent rebuilds this on reset.
    """

    CHARS_PER_TOKEN: ClassVar[int] = 4
    # Tokenizer mode adds fitted chat-template overhead from recorded request usage.
    TEMPLATE_PER_MESSAGE_TOKENS: ClassVar[int] = 16
    TEMPLATE_FIXED_TOKENS: ClassVar[int] = 128
    # Safety bound on fitted density after pathological server/tokenizer mismatch.
    MAX_CALIBRATION: ClassVar[float] = 4.0
    # Margin over observed overflow ratio so the next retrim lands under the cap.
    CALIBRATION_MARGIN: ClassVar[float] = 1.05
    # Blind geometric backoff when a provider reports overflow without input tokens.
    BLIND_RETRIM_STEP: ClassVar[float] = 1.3
    # Bounded retrim attempts keep act() wall time bounded.
    RETRIM_BUDGET: ClassVar[int] = 3

    max_context_length: int
    max_completion_tokens: int
    # The exact specs the request sends; counting anything else would misfit.
    tools: list[dict[str, Any]]
    calibration: float = 1.0
    token_counter: HfTokenCounter | None = None

    def estimate_message_tokens(self, message: dict[str, Any]) -> int:
        texts = [message.get("content") or ""]
        for call in message.get("tool_calls") or ():
            function = call.get("function", {})
            texts.append(function.get("arguments") or "")
            texts.append(function.get("name") or "")
        if self.token_counter is not None:
            return (
                sum(self.token_counter.count(t) for t in texts)
                + self.TEMPLATE_PER_MESSAGE_TOKENS
            )
        return sum(len(t) for t in texts) // self.CHARS_PER_TOKEN + 8

    def estimate_request_tokens(self, messages: list[dict[str, Any]]) -> int:
        tools_json = json.dumps(self.tools)
        if self.token_counter is not None:
            tool_tokens = self.token_counter.count(tools_json)
            tool_tokens += self.TEMPLATE_FIXED_TOKENS
        else:
            tool_tokens = len(tools_json) // self.CHARS_PER_TOKEN
        return sum(self.estimate_message_tokens(m) for m in messages) + tool_tokens

    def fit(
        self,
        head: list[dict[str, Any]],
        step_groups: list[list[dict[str, Any]]],
        floor: int,
        turn_messages: Sequence[dict[str, Any]] = (),
    ) -> tuple[list[dict[str, Any]], int]:
        """Return `(messages, dropped)`: `head` plus the newest step groups that
        fit the reply budget, and how many leading groups were dropped.

        `floor` is the renderer's retention boundary; never dropped past.
        `turn_messages` (reminder/repair turns) are appended to the request after
        the fitted base, so their tokens are reserved from the step budget here;
        an unbudgeted repair turn echoing a large failed completion is exactly
        what overflowed the server cap unrecoverably before.
        """
        budget = (self.max_context_length - self.max_completion_tokens) / max(
            self.calibration, 1.0
        )
        head_tokens = self.estimate_request_tokens(head) + sum(
            self.estimate_message_tokens(message) for message in turn_messages
        )
        group_tokens = [
            sum(self.estimate_message_tokens(message) for message in group)
            for group in step_groups
        ]
        step_tokens = sum(group_tokens)
        start = 0
        while step_tokens > budget - head_tokens and start < floor:
            step_tokens -= group_tokens[start]
            start += 1
        kept = [message for group in step_groups[start:] for message in group]
        if start == 0:
            return head + kept, 0
        note = {
            "role": "user",
            "content": CONTEXT_OMITTED_NOTE_TEMPLATE.format(dropped=start),
        }
        return [*head, note, *kept], start

    def calibrate_from_usage(
        self, completion: Completion, request: list[dict[str, Any]]
    ) -> None:
        real = completion.usage.prompt_tokens
        if real is None:
            return
        density = real / self.estimate_request_tokens(request)
        self.calibration = min(self.MAX_CALIBRATION, max(1.0, density))

    def shrink_for_overflow(
        self,
        exc: ContextWindowExceededError,
        request: list[dict[str, Any]],
    ) -> bool:
        if exc.input_tokens is not None:
            estimate = self.estimate_request_tokens(request)
            target = (exc.input_tokens / estimate) * self.CALIBRATION_MARGIN
        else:
            target = self.calibration * self.BLIND_RETRIM_STEP
        new_calibration = min(self.MAX_CALIBRATION, max(self.calibration, target))
        if new_calibration <= self.calibration:
            return False
        self.calibration = new_calibration
        return True


# §5 Parsing and repair


class MissingToolCall(ValueError):
    """Completion had no tool call, often due to truncation."""


class ActionParser:
    """Parse completions into actions and render the matching retry turn on failure."""

    @staticmethod
    def parse_args(args_text: str) -> Any:
        raw = args_text.strip()
        if raw == "":
            return {}
        return json.loads(raw)

    @staticmethod
    def validate_args(action_name: ActionName, args: Any) -> ToolArgs:
        if not isinstance(args, dict):
            raise ValueError(f"{action_name}: arguments must decode to an object")
        return TOOLS[action_name].args_model.model_validate(args)

    @classmethod
    def action(cls, name: str, arguments: str) -> Action:
        # `actions()` already rejected names outside `offered` == TOOLS' keys.
        action_name = cast(ActionName, name)
        args = cls.validate_args(action_name, cls.parse_args(arguments))
        return Action(name=action_name, args=args)

    @classmethod
    def actions(
        cls, completion: Completion, *, offered: frozenset[str]
    ) -> tuple[Action, ...]:
        if not completion.tool_calls:
            raise MissingToolCall("model response omitted required tool call")
        for call in completion.tool_calls:
            if call.name not in offered:
                raise ValueError(
                    f"tool {call.name!r} was not offered on this request; "
                    f"offered tools: {', '.join(sorted(offered))}"
                )
        return tuple(
            cls.action(call.name, call.arguments) for call in completion.tool_calls
        )

    @classmethod
    def repair_messages(
        cls, completion: Completion, exc: Exception
    ) -> list[dict[str, Any]]:
        if isinstance(exc, MissingToolCall):
            return [
                {"role": "assistant", "content": completion.content or ""},
                {"role": "user", "content": MISSING_TOOL_CALL_REPAIR_PROMPT},
            ]
        return [
            {
                "role": "user",
                "content": INVALID_TOOL_CALL_REPAIR_PROMPT.format(error=exc),
            }
        ]
