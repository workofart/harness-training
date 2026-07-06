"""Agent-editable implementation tests for src.policy.core."""

from __future__ import annotations

import asyncio
import base64
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from conftest import _completion, _StubLlm, _tool_call
from openai import APIError

from src.env.base import (
    RawEnvOutput,
    RunAction,
    scrub_raw_env_output,
)
from src.env.base import StepResult
from src.policy import core
from src.policy.core import (
    FILE_TOOL_TIMEOUT_SEC,
    TOOLS,
    Action,
    ActionGuard,
    ActionParser,
    COMMAND_TIMEOUT_SEC,
    FailedStep,
    LlmAgent,
    NoValidActionError,
    ReadArgs,
    ReminderRule,
    ReplaceArgs,
    RepeatedLengthCutoffError,
    RunArgs,
    SubmitArgs,
    Trajectory,
    WriteArgs,
    _ContextWindow,
    _RequestBuilder,
    build_env_action,
    build_read_command,
    build_replace_command,
    build_tool_specs,
    build_write_command,
)
from src.llm.backend import (
    Completion,
    CompletionBackend,
    CompletionRequest,
    ContextWindowExceededError,
    ToolCall,
    Usage,
)
from src.llm.token_counter import HfTokenCounter

_SUBMIT_COMPLETION = _completion(_tool_call("submit"))
_SUBMIT_ACTIONS = (Action(name="submit", args=SubmitArgs()),)

_TRUNCATED_ARGS_LENGTH_CUTOFF = Completion(
    tool_calls=(ToolCall(name="submit", arguments='{"trunca'),),
    finish_reason="length",
)


def test_build_tool_specs_cover_action_space_and_embed_pydantic_schemas() -> None:
    by_name = {s["function"]["name"]: s for s in build_tool_specs()}

    assert (
        set(by_name)
        == set(TOOLS)
        == {
            "run",
            "submit",
            "replace",
            "read",
            "write",
        }
    )
    assert by_name["run"]["function"]["parameters"] == RunArgs.model_json_schema()
    assert by_name["submit"]["function"]["parameters"] == SubmitArgs.model_json_schema()
    assert (
        by_name["replace"]["function"]["parameters"] == ReplaceArgs.model_json_schema()
    )
    assert by_name["read"]["function"]["parameters"] == ReadArgs.model_json_schema()
    assert by_name["write"]["function"]["parameters"] == WriteArgs.model_json_schema()
    assert by_name["run"]["function"]["parameters"]["required"] == ["command"]
    assert by_name["run"]["function"]["parameters"]["additionalProperties"] is False
    assert RunArgs(command="pytest").timeout_sec == COMMAND_TIMEOUT_SEC
    assert RunArgs(command="pytest", timeout_sec=45).timeout_sec == 45
    run_params = by_name["run"]["function"]["parameters"]
    assert run_params["properties"]["timeout_sec"]["default"] == COMMAND_TIMEOUT_SEC
    assert "timeout_sec" not in run_params["required"]


def test_build_env_action_passes_through_run_fields() -> None:
    action = Action(
        name="run", args=RunArgs(command="pytest", cwd="/repo", timeout_sec=31)
    )

    assert build_env_action(action) == RunAction(
        command="pytest", cwd="/repo", timeout_sec=31
    )


def test_build_env_action_renders_file_tools_as_bounded_runs() -> None:
    replace = ReplaceArgs(path="pkg/mod.py", old_text="old", new_text="new")
    read = ReadArgs(path="pkg/mod.py", offset=10, limit=20)
    write = WriteArgs(path="pkg/new.py", content="content\n")

    assert build_env_action(Action(name="replace", args=replace)) == RunAction(
        command=build_replace_command(replace),
        timeout_sec=FILE_TOOL_TIMEOUT_SEC,
    )
    assert build_env_action(Action(name="read", args=read)) == RunAction(
        command=build_read_command(read),
        timeout_sec=FILE_TOOL_TIMEOUT_SEC,
    )
    assert build_env_action(Action(name="write", args=write)) == RunAction(
        command=build_write_command(write),
        timeout_sec=FILE_TOOL_TIMEOUT_SEC,
    )


def test_replace_and_write_commands_persist_exact_content(tmp_path: Path) -> None:
    target = tmp_path / "pkg" / "mod.py"
    target.parent.mkdir()
    target.write_text("before\n", encoding="utf-8")

    replace = subprocess.run(
        build_replace_command(
            ReplaceArgs(path="pkg/mod.py", old_text="before\n", new_text="after\n")
        ),
        shell=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    write = subprocess.run(
        build_write_command(WriteArgs(path="pkg/new.py", content="a'$`\\b\n")),
        shell=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert replace.returncode == 0, replace.stderr
    assert write.returncode == 0, write.stderr
    assert target.read_text(encoding="utf-8") == "after\n"
    assert (tmp_path / "pkg" / "new.py").read_text(encoding="utf-8") == "a'$`\\b\n"


def test_read_command_returns_requested_window_and_total(tmp_path: Path) -> None:
    target = tmp_path / "quoted file.py"
    target.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

    read = subprocess.run(
        build_read_command(ReadArgs(path=target.name, offset=2, limit=2)),
        shell=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert read.returncode == 0, read.stderr
    assert read.stdout == "two\nthree\n(read: lines 2-3; file has 4 lines total)\n"


@pytest.mark.parametrize(
    ("old_text", "error"),
    [("absent\n", "not found"), ("same\n", "2 locations")],
)
def test_replace_failure_is_atomic(tmp_path: Path, old_text: str, error: str) -> None:
    target = tmp_path / "mod.py"
    original = "same\nmiddle\nsame\n"
    target.write_text(original, encoding="utf-8")

    replace = subprocess.run(
        build_replace_command(
            ReplaceArgs(path=target.name, old_text=old_text, new_text="changed\n")
        ),
        shell=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert replace.returncode != 0
    assert error in replace.stderr
    assert target.read_text(encoding="utf-8") == original


def test_build_env_action_rejects_submit():
    with pytest.raises(ValueError, match="unsupported editable action"):
        build_env_action(Action(name="submit", args=SubmitArgs()))


def test_request_builder_tool_result_renders_fields_and_truncates() -> None:
    rendered = _RequestBuilder().tool_result(
        RawEnvOutput(
            exit_code=0,
            stdout="out" + "x" * _RequestBuilder.RESULT_CHAR_LIMIT,
            stderr="err",
        )
    )

    assert "rc=0" in rendered
    assert "stdout:\nout" in rendered
    assert "...[truncated]" in rendered
    assert "stderr:\nerr" in rendered


def test_request_builder_tool_result_returns_no_output_when_state_is_empty() -> None:
    assert _RequestBuilder().tool_result(RawEnvOutput()) == "(no output)"


# Regression: django-14017 object addresses forked identical trajectories; scrub once at the env boundary.


def test_scrub_raw_env_output_scrubs_stderr_too():
    scrubbed = scrub_raw_env_output(
        RawEnvOutput(exit_code=1, stderr="built at 15:11:50 done")
    )
    assert "15:11:50" not in scrubbed.stderr
    assert "built at" in scrubbed.stderr and "done" in scrubbed.stderr


def test_request_builder_byte_stable_when_scrubbed_output_differs_only_by_address() -> (
    None
):
    def traj(addr):
        obs = scrub_raw_env_output(
            RawEnvOutput(exit_code=0, stdout=f"<Exists object at {addr}>")
        )
        return (
            (Action(name="run", args=RunArgs(command="python -c 'print(e & q)'")), obs),
        )

    builder = _RequestBuilder()
    m1 = builder.assemble(
        instruction="do",
        working_dir="/w",
        trajectory=traj("0x51dd30"),
    )
    m2 = builder.assemble(
        instruction="do",
        working_dir="/w",
        trajectory=traj("0x544d30"),
    )
    assert m1 == m2


# Regression: circuit-fibsqrt replayed 10-25KB heredocs on every request.


def _single_action_messages(builder, action):
    head, step_groups, _floor = builder.assemble(
        instruction="do",
        working_dir="/w",
        trajectory=((action, RawEnvOutput()),),
    )
    return head + [m for group in step_groups for m in group]


def _single_step_messages(builder, command):
    return _single_action_messages(
        builder, Action(name="run", args=RunArgs(command=command))
    )


def _replayed_args(messages):
    [assistant] = [m for m in messages if m.get("tool_calls")]
    [call] = assistant["tool_calls"]
    return json.loads(call["function"]["arguments"])


def test_request_builder_replay_clips_oversized_args_in_tokens():
    heredoc = "cat > gen.py << 'EOF'\n" + "print('gates')\n" * 2000 + "EOF"
    builder = _RequestBuilder(token_counter=_QWEN_LIKE_COUNTER)

    args = _replayed_args(_single_step_messages(builder, heredoc))

    assert args["command"].startswith("cat > gen.py")
    assert args["command"].endswith("...[truncated]")
    assert _QWEN_LIKE_COUNTER.count(args["command"]) <= builder.ARG_TOKEN_LIMIT


def test_request_builder_replay_clips_oversized_args_in_chars_without_tokenizer():
    heredoc = "cat > gen.py << 'EOF'\n" + "x" * (2 * _RequestBuilder.ARG_CHAR_LIMIT)

    args = _replayed_args(_single_step_messages(_RequestBuilder(), heredoc))

    assert args["command"].endswith("...[truncated]")
    assert len(args["command"]) <= _RequestBuilder.ARG_CHAR_LIMIT


def test_request_builder_replay_preserves_small_args_verbatim():
    args = _replayed_args(_single_step_messages(_RequestBuilder(), "pytest -x"))

    # timeout_sec carries its default (the single source of truth for the budget),
    # so the replayed call shows the command's effective timeout verbatim.
    assert args == {"command": "pytest -x", "timeout_sec": COMMAND_TIMEOUT_SEC}


def test_initial_user_prompt_projects_instruction_and_working_directory() -> None:
    known = core.initial_user_prompt(instruction="do thing", working_dir="/work")
    unknown = core.initial_user_prompt(instruction="do thing", working_dir=None)

    assert "do thing" in known
    assert "/work" in known
    assert "do thing" in unknown
    assert "(unknown)" in unknown


def test_system_prompt_documents_contract_and_redactions() -> None:
    prompt = core.build_system_prompt()

    assert str(COMMAND_TIMEOUT_SEC) in prompt
    assert "timeout_sec" in prompt
    assert "submit" in prompt
    assert "graded" in prompt
    assert "volatile tokens" in prompt
    assert "<TIME>" in prompt
    assert "<PID>" in prompt
    assert "<BINARY_STDOUT>" in prompt


class _RecordingEvents:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, event: str, /, **fields: Any) -> None:
        self.events.append((event, fields))


def _rule_events(events: _RecordingEvents) -> list[tuple[str, bool]]:
    return [
        (fields["rule"], fields["fired"])
        for name, fields in events.events
        if name == "policy_rule"
    ]


def _agent(
    llm: CompletionBackend,
    *,
    events: Any = core.NOOP_AGENT_CALLBACK,
    max_context_length: int = 200_000,
    max_completion_tokens: int = 8192,
    max_output_retries: int = 2,
    thinking_toggleable: bool = False,
    token_counter: HfTokenCounter | None = None,
) -> LlmAgent:
    agent = LlmAgent(
        llm=llm,
        events=events,
        max_context_length=max_context_length,
        max_completion_tokens=max_completion_tokens,
        max_output_retries=max_output_retries,
        thinking_toggleable=thinking_toggleable,
        token_counter=token_counter,
    )
    agent.reset(RawEnvOutput(instruction="do", working_dir="/w"))
    return agent


def _recording_agent(
    completion: Completion,
) -> tuple[LlmAgent, _StubLlm, _RecordingEvents]:
    llm = _StubLlm([completion])
    events = _RecordingEvents()
    return _agent(llm, events=cast(Any, events)), llm, events


def _continuing_step(raw_env_output: RawEnvOutput) -> StepResult:
    return StepResult(
        raw_env_output=raw_env_output,
        reward=0.0,
        terminated=False,
        truncated=False,
    )


def _long_trajectory(
    n: int, char_count: int = _RequestBuilder.RESULT_CHAR_LIMIT
) -> Trajectory:
    output = "y" * char_count
    return tuple(
        (
            Action(name="run", args=RunArgs(command=f"cmd{i}")),
            RawEnvOutput(stdout=output),
        )
        for i in range(n)
    )


def _fitted(window, trajectory):
    # Wire the two decoupled halves the way LlmAgent does: render, then fit.
    head, step_groups, floor = _RequestBuilder().assemble(
        instruction="do", working_dir="/w", trajectory=trajectory
    )
    messages, _dropped = window.fit(head, step_groups, floor)
    return messages


def test_context_window_windows_oldest_steps_when_context_capped():
    # Preserve reply headroom by dropping oldest pairs while keeping newest context.
    trajectory = _long_trajectory(20)

    untrimmed = _fitted(
        _ContextWindow(
            max_context_length=10_000_000,
            max_completion_tokens=8192,
            tools=build_tool_specs(),
        ),
        trajectory,
    )
    trimmed = _fitted(
        _ContextWindow(
            max_context_length=10000,
            max_completion_tokens=6000,
            tools=build_tool_specs(),
        ),
        trajectory,
    )

    assert trimmed[0]["role"] == "system"
    assert trimmed[1]["role"] == "user"
    assert len(trimmed) < len(untrimmed)
    tool_msgs = [m for m in trimmed if m["role"] == "tool"]
    assert 1 <= len(tool_msgs) < 20
    assert any(
        "omitted to fit the context window" in (m.get("content") or "") for m in trimmed
    )
    blob = json.dumps(trimmed)
    assert "cmd19" in blob
    assert "cmd0" not in blob
    total = sum(_estimate(m) for m in trimmed)
    assert total <= 10000


def test_context_window_uses_configured_completion_reserve_when_trimming():
    trajectory = _long_trajectory(20)

    smaller_reply = _fitted(
        _ContextWindow(
            max_context_length=30000,
            max_completion_tokens=4096,
            tools=build_tool_specs(),
        ),
        trajectory,
    )
    larger_reply = _fitted(
        _ContextWindow(
            max_context_length=30000,
            max_completion_tokens=12288,
            tools=build_tool_specs(),
        ),
        trajectory,
    )

    smaller_reply_steps = sum(1 for m in smaller_reply if m["role"] == "tool")
    larger_reply_steps = sum(1 for m in larger_reply if m["role"] == "tool")
    assert larger_reply_steps < smaller_reply_steps
    assert sum(_estimate(m) for m in larger_reply) + 12288 <= 30000


def test_context_window_keeps_newest_real_step_past_trailing_failed_step():
    # A trailing FailedStep renders alone; retain the newest real observation too.
    trajectory = (
        *_long_trajectory(20),
        FailedStep(attempts=3, error="missing tool call"),
    )

    trimmed = _fitted(
        _ContextWindow(
            max_context_length=10000,
            max_completion_tokens=6000,
            tools=build_tool_specs(),
        ),
        trajectory,
    )

    blob = json.dumps(trimmed)
    assert "no valid action" in blob
    assert any(m["role"] == "tool" for m in trimmed)
    assert "cmd19" in blob
    assert "cmd0" not in blob


@pytest.mark.parametrize(
    ("stdout", "stderr", "expected"),
    [
        pytest.param(
            "a" * (_RequestBuilder.RESULT_CHAR_LIMIT + 1),
            "b" * (_RequestBuilder.RESULT_CHAR_LIMIT + 1),
            [
                ("observation_clipped", 1, "stdout"),
                ("observation_clipped", 1, "stderr"),
            ],
            id="oversized-stdout-and-stderr",
        ),
        pytest.param("abc", "de", [], id="small-output"),
    ],
)
def test_llm_agent_observation_clipping_events(
    stdout: str, stderr: str, expected: list[tuple[str, int, str]]
) -> None:
    events = _RecordingEvents()
    agent = _agent(_StubLlm([]), events=cast(Any, events))

    agent.observe(
        Action(name="run", args=RunArgs(command="pwd")),
        _continuing_step(RawEnvOutput(stdout=stdout, stderr=stderr)),
    )

    assert [
        (name, fields["step_index"], fields["field"]) for name, fields in events.events
    ] == expected


_TOKEN_ESTIMATOR = _ContextWindow(
    max_context_length=1, max_completion_tokens=1, tools=build_tool_specs()
)


def _estimate(message):
    return _TOKEN_ESTIMATOR.estimate_message_tokens(message)


class _DenseServerLlm(CompletionBackend):
    """Reject requests using a denser token estimate than the harness."""

    def __init__(
        self,
        completions: Completion | list[Completion],
        *,
        limit: int,
        density: float,
        reply_tokens: int,
    ) -> None:
        self._completions = (
            [completions] if isinstance(completions, Completion) else list(completions)
        )
        self._limit = limit
        self._density = density
        self._reply_tokens = reply_tokens
        self.request_estimates: list[int] = []

    async def _complete(self, request):
        estimate = _TOKEN_ESTIMATOR.estimate_request_tokens(request.messages)
        self.request_estimates.append(estimate)
        real_input = int(estimate * self._density)
        if real_input + self._reply_tokens > self._limit:
            raise ContextWindowExceededError(
                "Requested token count exceeds the model's maximum context length",
                limit=self._limit,
                requested=real_input + self._reply_tokens,
                input_tokens=real_input,
            )
        return self._completions.pop(0)


def test_act_recovers_from_context_window_error_by_retrimming():
    # If chars/4 underestimates server tokens, retry with tighter trimming.
    events = _RecordingEvents()
    llm = _DenseServerLlm(
        _SUBMIT_COMPLETION, limit=30000, density=1.5, reply_tokens=4096
    )
    agent = _agent(
        llm,
        events=cast(Any, events),
        max_context_length=30000,
        max_completion_tokens=4096,
    )
    agent._trajectory = _long_trajectory(20)

    actions = asyncio.run(agent.act())

    assert actions == _SUBMIT_ACTIONS
    assert len(llm.request_estimates) >= 2
    assert llm.request_estimates[-1] < llm.request_estimates[0]
    assert agent._context_window.calibration > 1.0
    retrims = [
        fields for name, fields in events.events if name == "context_window_retrimmed"
    ]
    assert len(retrims) == len(llm.request_estimates) - 1
    first = retrims[0]
    assert set(first) == {
        "step_index",
        "attempt_index",
        "limit",
        "requested",
        "calibration",
    }
    assert first["step_index"] == 21
    assert first["attempt_index"] == 0
    assert first["limit"] == 30000
    assert first["requested"] > 30000
    assert first["calibration"] > 1.0


def test_act_reserves_fit_budget_for_repair_turn_messages():
    # Regression: exp-20260711-115438 overflowed on an unbudgeted repair echo.
    events = _RecordingEvents()
    llm = _DenseServerLlm(
        [
            _completion(content="z" * 8000),  # no tool calls -> repair turn echoes this
            _SUBMIT_COMPLETION,
        ],
        limit=10000,
        density=0.9,
        reply_tokens=6000,
    )
    agent = _agent(
        llm,
        events=cast(Any, events),
        max_context_length=10000,
        max_completion_tokens=6000,
    )
    agent._trajectory = _long_trajectory(20, char_count=1200)

    actions = asyncio.run(agent.act())

    assert actions == _SUBMIT_ACTIONS
    # Repair headroom must not rely on recovery from a server 400.
    assert not [
        fields for name, fields in events.events if name == "context_window_retrimmed"
    ]


@pytest.mark.parametrize(
    ("trajectory_length", "max_context_length", "max_completion_tokens", "drops"),
    [
        pytest.param(20, 10000, 6000, True, id="drops-old-steps"),
        pytest.param(1, 200_000, 8192, False, id="keeps-all-steps"),
    ],
)
def test_llm_agent_context_groups_dropped_events(
    trajectory_length: int,
    max_context_length: int,
    max_completion_tokens: int,
    drops: bool,
) -> None:
    events = _RecordingEvents()
    llm = _StubLlm([_SUBMIT_COMPLETION])
    agent = _agent(
        llm,
        events=cast(Any, events),
        max_context_length=max_context_length,
        max_completion_tokens=max_completion_tokens,
    )
    agent._trajectory = _long_trajectory(trajectory_length)

    asyncio.run(agent.act())

    dropped = [
        fields for name, fields in events.events if name == "context_groups_dropped"
    ]
    if drops:
        assert len(dropped) == 1
        assert dropped[0]["step_index"] == 21
        assert dropped[0]["dropped"] > 0
    else:
        assert dropped == []


def _recorded_binary_observation() -> str:
    """Recorded NUL-dense ELF output from exp-20260703-165350-417763."""
    fixture = (
        Path(__file__).parent / "fixtures" / "dense_binary_observation.b64"
    ).read_text()
    return base64.b64decode(fixture).decode("utf-8")


class _QwenLikeTokenizer:
    """Treat printable ASCII as 4 chars/token and binary as 1 char/token."""

    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        pieces = list(re.finditer(r"[ -~\n\t]{1,4}|.", text, re.DOTALL))
        return SimpleNamespace(
            ids=[1] * len(pieces), offsets=[m.span() for m in pieces]
        )


# One unit of measure shared by the fake server and the agent under test.
_QWEN_LIKE_COUNTER = HfTokenCounter(_QwenLikeTokenizer())


class _TokenizingServerLlm(CompletionBackend):
    """Enforce a nonlinear, content-sensitive token limit."""

    def __init__(
        self, completion: Completion, *, limit: int, reply_tokens: int
    ) -> None:
        self._completion = completion
        self._limit = limit
        self._reply_tokens = reply_tokens
        self.real_inputs: list[int] = []

    def _real_input_tokens(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for message in messages:
            total += _QWEN_LIKE_COUNTER.count(message.get("content") or "")
            for call in message.get("tool_calls") or ():
                function = call.get("function", {})
                total += _QWEN_LIKE_COUNTER.count(function.get("arguments") or "")
                total += _QWEN_LIKE_COUNTER.count(function.get("name") or "")
            total += 16  # chat-template wrapper per message (measured ~15.9)
        return total

    async def _complete(self, request):
        real_input = self._real_input_tokens(request.messages)
        self.real_inputs.append(real_input)
        if real_input + self._reply_tokens > self._limit:
            raise ContextWindowExceededError(
                "Requested token count exceeds the model's maximum context "
                f"length of {self._limit} tokens. You requested a total of "
                f"{real_input + self._reply_tokens} tokens: {real_input} tokens "
                f"from the input messages and {self._reply_tokens} tokens for "
                "the completion.",
                limit=self._limit,
                requested=real_input + self._reply_tokens,
                input_tokens=real_input,
            )
        return Completion(
            tool_calls=self._completion.tool_calls,
            content=self._completion.content,
            usage=Usage(prompt_tokens=real_input),
        )


def _trajectory_with_binary_tail(n_normal: int) -> Trajectory:
    """Append the recorded binary observation to ordinary steps."""
    normal = "output line of ordinary shell text\n" * 80
    steps = [
        (
            Action(name="run", args=RunArgs(command=f"cmd{i}")),
            RawEnvOutput(stdout=normal),
        )
        for i in range(n_normal)
    ]
    steps.append(
        (
            Action(name="run", args=RunArgs(command="readelf -x .data interp")),
            RawEnvOutput(stdout=_recorded_binary_observation()),
        )
    )
    return tuple(steps)


def test_act_survives_token_dense_binary_observation():
    # Regression: one NUL-heavy observation overflowed the untrimmable newest step.
    llm = _TokenizingServerLlm(_SUBMIT_COMPLETION, limit=32768, reply_tokens=8192)
    agent = _agent(
        llm,
        max_context_length=32768,
        max_completion_tokens=8192,
        token_counter=_QWEN_LIKE_COUNTER,
    )
    agent._trajectory = _trajectory_with_binary_tail(18)

    actions = asyncio.run(agent.act())

    assert actions == _SUBMIT_ACTIONS
    assert len(llm.real_inputs) == 1
    assert llm.real_inputs[-1] + 8192 <= 32768


def test_act_calibration_tightens_after_high_usage_feedback():
    # Large server-reported prompt usage calibrates an underestimated local counter.
    big_usage = Completion(
        tool_calls=(ToolCall(name="submit", arguments="{}"),),
        usage=Usage(prompt_tokens=999_999),
    )
    llm = _StubLlm([big_usage])
    agent = _agent(llm, max_context_length=40000, max_completion_tokens=4096)
    agent._trajectory = _long_trajectory(5)

    asyncio.run(agent.act())

    assert agent._context_window.calibration > 1.0


def _string_contents(messages: list[dict[str, Any]]) -> list[str]:
    """Return string-valued content from a message list (skip tool_call envelopes)."""
    return [m["content"] for m in messages if isinstance(m.get("content"), str)]


def test_action_parser_parse_args_accepted_variants() -> None:
    parser = ActionParser()

    assert parser.parse_args("") == {}
    assert parser.parse_args("   ") == {}
    assert parser.parse_args('{"command": "pwd", "timeout_sec": 5}') == {
        "command": "pwd",
        "timeout_sec": 5,
    }


def test_action_parser_rejects_double_encoded_arguments() -> None:
    double_encoded = json.dumps(json.dumps({"command": "pwd"}))
    with pytest.raises(ValueError, match="must decode to an object"):
        ActionParser.action("run", double_encoded)


def test_action_parser_parse_args_raises_on_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        ActionParser().parse_args("not json")


@pytest.mark.parametrize(
    ("args", "error"),
    [
        pytest.param("string", "must decode to an object", id="non-object"),
        pytest.param({}, "Field required", id="missing-command"),
        pytest.param(
            {"command": "pwd", "timeout_sec": "soon"},
            "Input should be a valid integer",
            id="non-numeric-timeout",
        ),
        pytest.param({"command": ""}, "at least 1 character", id="empty-command"),
        pytest.param(
            {"command": "pwd", "cwd": 5},
            "Input should be a valid string",
            id="non-string-cwd",
        ),
    ],
)
def test_action_parser_validate_args_rejected_variants(
    args: object, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        ActionParser().validate_args("run", args)


def test_action_parser_validate_args_accepted_variants() -> None:
    parser = ActionParser()

    assert parser.validate_args(
        "run", {"command": "pwd", "timeout_sec": "120"}
    ) == RunArgs(command="pwd", timeout_sec=120)
    assert parser.validate_args("run", {"command": "pwd", "cwd": ""}) == RunArgs(
        command="pwd", cwd=""
    )


def _unoffered_read_completion() -> Completion:
    arguments = (
        '{"limit": "15", "offset": "218", '
        '"path": "/testbed/django/db/models/deletion.py"}'
    )
    return Completion(
        tool_calls=(ToolCall(name="read", arguments=arguments),),
        finish_reason="tool_calls",
    )


def test_action_parser_actions_rejects_unoffered_tool():
    # Provenance: archived_experiments/v2-exp-swe-0709/exp-20260709-232839-805029,
    # django__django-11087 step 80.
    with pytest.raises(ValueError, match="not offered"):
        ActionParser.actions(
            _unoffered_read_completion(), offered=frozenset({"run", "submit"})
        )


def test_action_parser_actions_accepts_offered_tool():
    completion = Completion(
        tool_calls=(ToolCall(name="submit", arguments="{}"),),
        finish_reason="tool_calls",
    )

    actions = ActionParser.actions(completion, offered=frozenset({"submit"}))

    assert actions == _SUBMIT_ACTIONS


def test_repair_messages_for_unoffered_tool_names_the_offered_set():
    exc = ValueError(
        "tool 'read' was not offered on this request; offered tools: run, submit"
    )

    messages = ActionParser.repair_messages(_unoffered_read_completion(), exc)

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "offered tools: run, submit" in messages[0]["content"]


def _policy_scalars(**overrides: Any) -> dict[str, Any]:
    return {
        "max_context_length": 12345,
        "max_completion_tokens": 8192,
        "thinking_toggleable": False,
        "tokenizer_name": None,
        "model_name": "gpt-test",
    } | overrides


def test_build_policy_wires_config_fields():
    llm = _StubLlm([])
    events = core.NOOP_AGENT_CALLBACK

    agent = core.build_policy(
        llm,
        events,
        **_policy_scalars(max_completion_tokens=678, thinking_toggleable=True),
    )

    assert type(agent) is LlmAgent
    assert agent.llm is llm
    assert agent.events is events
    assert agent.max_output_retries == 2
    assert agent.max_context_length == 12345
    assert agent.max_completion_tokens == 678
    assert agent.thinking_toggleable is True
    assert agent.token_counter is None


def test_llm_agent_materializes_default_command_timeout_when_omitted() -> None:
    llm = _StubLlm(
        [
            _completion(_tool_call("run", command="make")),
            _completion(_tool_call("run", command="pytest", timeout_sec=45)),
        ]
    )
    agent = core.build_policy(
        llm,
        core.NOOP_AGENT_CALLBACK,
        **_policy_scalars(),
    )
    agent.reset(RawEnvOutput(instruction="do", working_dir="/w"))

    defaulted = asyncio.run(agent.act())
    explicit = asyncio.run(agent.act())

    assert defaulted == (
        Action(
            name="run", args=RunArgs(command="make", timeout_sec=COMMAND_TIMEOUT_SEC)
        ),
    )
    assert explicit == (
        Action(name="run", args=RunArgs(command="pytest", timeout_sec=45)),
    )


def test_command_timeout_default_is_visible_in_prompt_and_tool_schema() -> None:
    prompt = core.build_system_prompt()
    specs = {spec["function"]["name"]: spec for spec in build_tool_specs()}
    timeout_schema = specs["run"]["function"]["parameters"]["properties"]["timeout_sec"]

    assert f"default {COMMAND_TIMEOUT_SEC}s" in prompt
    assert timeout_schema["default"] == COMMAND_TIMEOUT_SEC
    assert f"{COMMAND_TIMEOUT_SEC}s default" in timeout_schema["description"]


def test_context_window_accounts_for_the_request_tools() -> None:
    agent = core.build_policy(
        _StubLlm([]),
        core.NOOP_AGENT_CALLBACK,
        **_policy_scalars(),
    )
    agent.reset(RawEnvOutput(instruction="do", working_dir="/w"))

    window = agent._context_window
    assert window.tools == build_tool_specs()
    expected_tool_tokens = len(json.dumps(window.tools)) // window.CHARS_PER_TOKEN
    assert window.estimate_request_tokens([]) == expected_tool_tokens


def test_llm_agent_retries_on_missing_tool_call_and_appends_repair_prompt():
    llm = _StubLlm(
        [
            _completion(content="thinking..."),
            _SUBMIT_COMPLETION,
        ]
    )
    actions = asyncio.run(_agent(llm).act())
    assert actions == _SUBMIT_ACTIONS
    assert len(llm.calls) == 2
    second_contents = _string_contents(llm.calls[1])
    assert any(core.MISSING_TOOL_CALL_REPAIR_PROMPT in c for c in second_contents)
    assert llm.thinking_overrides == [None, None]


def test_llm_agent_drops_thinking_for_rest_of_rollout_after_truncated_output():
    llm = _StubLlm(
        [
            Completion(finish_reason="length"),
            _SUBMIT_COMPLETION,
            _SUBMIT_COMPLETION,
            _SUBMIT_COMPLETION,
        ]
    )
    agent = _agent(llm, thinking_toggleable=True)

    actions = asyncio.run(agent.act())

    assert actions == _SUBMIT_ACTIONS
    assert llm.thinking_overrides == [None, False]

    asyncio.run(agent.act())
    assert llm.thinking_overrides == [None, False, False]

    agent.reset(RawEnvOutput(instruction="do", working_dir="/w"))
    asyncio.run(agent.act())
    assert llm.thinking_overrides == [None, False, False, None]


def test_llm_agent_aborts_after_repeated_length_cutoffs():
    llm = _StubLlm(
        [
            Completion(content="runaway reasoning", finish_reason="length"),
            _completion(_tool_call("run", command="pwd")),
            Completion(content="runaway again", finish_reason="length"),
            _SUBMIT_COMPLETION,
        ]
    )
    agent = _agent(llm)

    assert asyncio.run(agent.act()) == (
        Action(name="run", args=RunArgs(command="pwd")),
    )
    with pytest.raises(RepeatedLengthCutoffError, match="length"):
        asyncio.run(agent.act())

    assert len(llm.calls) == 3


def test_llm_agent_keeps_output_budget_and_drops_thinking_on_truncated_args():
    llm = _StubLlm([_TRUNCATED_ARGS_LENGTH_CUTOFF, _SUBMIT_COMPLETION])
    agent = _agent(llm, thinking_toggleable=True)

    actions = asyncio.run(agent.act())

    assert actions == _SUBMIT_ACTIONS
    assert llm.thinking_overrides == [None, False]


def test_llm_agent_breaks_early_when_truncated_tool_call_recurs_thinking_off():
    # A repeated cutoff with thinking already off cannot improve deterministically.
    llm = _StubLlm([_TRUNCATED_ARGS_LENGTH_CUTOFF] * 3)
    agent = _agent(llm, thinking_toggleable=True)

    with pytest.raises(NoValidActionError):
        asyncio.run(agent.act())

    assert len(llm.calls) == 2
    assert llm.thinking_overrides == [None, False]


def test_llm_agent_non_toggleable_length_cutoff_stops_without_thinking_override():
    llm = _StubLlm([_TRUNCATED_ARGS_LENGTH_CUTOFF, _SUBMIT_COMPLETION])
    agent = _agent(llm)

    with pytest.raises(NoValidActionError):
        asyncio.run(agent.act())

    assert len(llm.calls) == 1
    assert llm.thinking_overrides == [None]


def test_llm_agent_retries_on_invalid_json_and_appends_correction():
    llm = _StubLlm(
        [
            _completion(ToolCall(name="submit", arguments="not json")),
            _SUBMIT_COMPLETION,
        ]
    )
    actions = asyncio.run(_agent(llm).act())
    assert actions == _SUBMIT_ACTIONS
    second_contents = _string_contents(llm.calls[1])
    assert any("invalid" in c.lower() for c in second_contents)


def test_llm_agent_repairs_provider_rejected_malformed_tool_args():
    class RejectMalformedToolArgsOnce(CompletionBackend):
        def __init__(self) -> None:
            self.calls: list[CompletionRequest] = []

        async def _complete(self, request: CompletionRequest) -> Completion:
            self.calls.append(request)
            if len(self.calls) == 1:
                raise APIError(
                    "Upstream error from Groq: Failed to parse tool call arguments as JSON",
                    request=httpx.Request("POST", "https://openrouter.ai/api/v1"),
                    body=None,
                )
            return _SUBMIT_COMPLETION

    llm = RejectMalformedToolArgsOnce()
    actions = asyncio.run(_agent(llm).act())

    assert actions == _SUBMIT_ACTIONS
    assert len(llm.calls) == 2
    second_contents = _string_contents(llm.calls[1].messages)
    assert any(
        "Failed to parse tool call arguments as JSON" in content
        for content in second_contents
    )


def test_llm_agent_raises_after_exhausting_retries():
    # Exhausted repairs are typed so the driver can score the turn.
    llm = _StubLlm([_completion(content="no calls")] * 3)
    with pytest.raises(NoValidActionError, match="failed to parse"):
        asyncio.run(_agent(llm).act())


def test_llm_agent_carries_exhausted_parse_error_into_next_step_context():
    llm = _StubLlm(
        [
            *[
                _completion(_tool_call("run", command="pwd", timeout_sec="soon"))
                for _ in range(3)
            ],
            _SUBMIT_COMPLETION,
        ]
    )
    agent = _agent(llm)

    with pytest.raises(NoValidActionError, match="failed to parse"):
        asyncio.run(agent.act())

    assert asyncio.run(agent.act()) == _SUBMIT_ACTIONS
    next_contents = _string_contents(llm.calls[-1])
    assert any(
        "Previous step produced no valid action" in content
        and "RunArgs" in content
        and "timeout_sec" in content
        and "Input should be a valid integer" in content
        for content in next_contents
    )


def test_llm_agent_returns_all_calls_in_multi_tool_response():
    llm = _StubLlm(
        [
            _completion(
                _tool_call("run", command="pwd"),
                _tool_call("run", command="ls"),
            )
        ]
    )
    actions = asyncio.run(_agent(llm).act())
    assert actions == (
        Action(name="run", args=RunArgs(command="pwd")),
        Action(name="run", args=RunArgs(command="ls")),
    )


def test_llm_agent_observe_replays_action_raw_env_output_history():
    llm = _StubLlm(
        [
            _completion(_tool_call("run", command="pwd")),
            _SUBMIT_COMPLETION,
        ]
    )
    agent = _agent(llm)
    [action] = asyncio.run(agent.act())
    agent.observe(
        action,
        _continuing_step(RawEnvOutput(exit_code=0, stdout="/work\n")),
    )

    assert asyncio.run(agent.act()) == _SUBMIT_ACTIONS
    second_messages = llm.calls[1]
    assert second_messages[2]["tool_calls"][0]["function"]["name"] == "run"
    assert second_messages[3]["content"] == "rc=0\nstdout:\n/work"


def test_reminder_rules_trace_all_evaluations_and_inject_first_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def silent(_agent: LlmAgent, _step_index: int) -> list[dict[str, Any]]:
        return []

    def hit(body: str):
        def build(_agent: LlmAgent, _step_index: int) -> list[dict[str, Any]]:
            return [{"role": "user", "content": body}]

        return build

    monkeypatch.setattr(
        core,
        "REMINDER_RULES",
        (
            ReminderRule(name="silent", build=silent),
            ReminderRule(name="first", build=hit("first nudge")),
            ReminderRule(name="second", build=hit("second nudge")),
        ),
    )
    agent, llm, events = _recording_agent(_SUBMIT_COMPLETION)

    asyncio.run(agent.act())

    assert _rule_events(events) == [
        ("silent", False),
        ("first", True),
        ("second", True),
    ]
    contents = [message.get("content") for message in llm.calls[-1]]
    assert "first nudge" in contents
    assert "second nudge" not in contents


def test_action_guards_trace_and_apply_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "REMINDER_RULES", ())

    def passthrough(
        _agent: LlmAgent, actions: tuple[Action, ...]
    ) -> tuple[Action, ...]:
        return actions

    def rewrite(_agent: LlmAgent, actions: tuple[Action, ...]) -> tuple[Action, ...]:
        return tuple(
            Action(name="run", args=RunArgs(command=f"{action.args.command} --checked"))
            if action.name == "run"
            else action
            for action in actions
        )

    monkeypatch.setattr(
        core,
        "ACTION_GUARDS",
        (
            ActionGuard(name="passthrough", apply=passthrough),
            ActionGuard(name="rewrite", apply=rewrite),
        ),
    )
    agent, _, events = _recording_agent(_completion(_tool_call("run", command="pwd")))

    actions = asyncio.run(agent.act())

    assert _rule_events(events) == [("passthrough", False), ("rewrite", True)]
    assert actions == (Action(name="run", args=RunArgs(command="pwd --checked")),)


def test_action_guard_failure_is_not_repaired_as_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(_agent: LlmAgent, _actions: tuple[Action, ...]) -> tuple[Action, ...]:
        raise RuntimeError("guard bug")

    monkeypatch.setattr(
        core,
        "ACTION_GUARDS",
        (ActionGuard(name="broken", apply=broken),),
    )
    llm = _StubLlm([_SUBMIT_COMPLETION, _SUBMIT_COMPLETION])

    with pytest.raises(RuntimeError, match="guard bug"):
        asyncio.run(_agent(llm).act())

    assert len(llm.calls) == 1


def test_policy_event_failure_propagates_without_model_output_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "REMINDER_RULES", ())

    def broken_events(_event: str, **_fields: Any) -> None:
        raise RuntimeError("event callback bug")

    llm = _StubLlm([_completion(content="no tool"), _SUBMIT_COMPLETION])
    agent = _agent(llm, events=broken_events, max_output_retries=1)

    with pytest.raises(RuntimeError, match="event callback bug"):
        asyncio.run(agent.act())

    assert len(llm.calls) == 1


def _agent_at_step(
    llm: CompletionBackend,
    step_index: int,
    events: Any = core.NOOP_AGENT_CALLBACK,
) -> LlmAgent:
    agent = _agent(llm, events=events)
    filler = (
        Action(name="run", args=RunArgs(command="true")),
        RawEnvOutput(exit_code=0),
    )
    agent._trajectory = (filler,) * (step_index - 1)
    return agent


def test_shipped_late_submit_reminder_is_silent_before_final_band() -> None:
    step_index = core.LATE_RUN_SUBMIT_REMINDER_STEP - 1
    events = _RecordingEvents()
    llm = _StubLlm([_SUBMIT_COMPLETION])
    agent = _agent_at_step(llm, step_index, cast(Any, events))

    asyncio.run(agent.act())

    assert llm.calls[-1][-1]["role"] == "tool"
    assert ("late_run_submit", False) in _rule_events(events)


def test_shipped_late_submit_reminder_fires_at_final_band() -> None:
    step_index = core.LATE_RUN_SUBMIT_REMINDER_STEP
    events = _RecordingEvents()
    llm = _StubLlm([_SUBMIT_COMPLETION])
    agent = _agent_at_step(llm, step_index, cast(Any, events))

    asyncio.run(agent.act())

    assert llm.calls[-1][-1]["role"] == "user"
    assert "submit" in llm.calls[-1][-1]["content"].lower()
    assert ("late_run_submit", True) in _rule_events(events)


def test_late_edited_trajectory_preserves_model_action() -> None:
    assert [rule.name for rule in core.REMINDER_RULES] == ["late_run_submit"]
    assert core.ACTION_GUARDS == ()
    llm = _StubLlm([_completion(_tool_call("run", command="ls"))])
    agent = _agent(llm)
    agent._trajectory = tuple(
        (
            Action(name="run", args=RunArgs(command=f"cmd{i}")),
            RawEnvOutput(exit_code=0),
        )
        for i in range(200)
    )

    actions = asyncio.run(agent.act())

    assert actions == (Action(name="run", args=RunArgs(command="ls")),)
