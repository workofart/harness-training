"""Tests for src/harness/core.py."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from src.adapters.llm_base import LlmToolCall
from src.harness.contracts import RawState
from src.harness.core import (
    ACTION_SPECS,
    DEFAULT_READ_WINDOW_LINES,
    MISSING_TOOL_CALL_REPAIR_PROMPT,
    EditFileAction,
    FindFilesAction,
    ListDirAction,
    NoValidActionError,
    ReadFileAction,
    RunAction,
    SearchTextAction,
    TaskLoopState,
    VerifyAction,
    WriteFileAction,
    act,
    build_action,
    build_initial_user_prompt,
    build_messages,
    build_system_prompt,
    build_tool_specs,
    execute_action,
    parse_action_args,
    render_tool_result,
    run_task_loop,
    validate_action_args,
)

from conftest import _StubLlm, _StubEnv, _tool_call, _completion


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


def _string_contents(messages: list[dict[str, Any]]) -> list[str]:
    """Return string-valued content from a message list (skip tool_call envelopes)."""
    return [m["content"] for m in messages if isinstance(m.get("content"), str)]


# ----------------------------------------------------------------------------
# parse_action_args
# ----------------------------------------------------------------------------


def test_parse_action_args_empty_string_returns_empty_dict():
    assert parse_action_args("") == {}
    assert parse_action_args("   ") == {}


def test_parse_action_args_decodes_json_object():
    assert parse_action_args('{"path": "/x", "limit": 5}') == {
        "path": "/x",
        "limit": 5,
    }


def test_parse_action_args_raises_on_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        parse_action_args("not json")


# ----------------------------------------------------------------------------
# validate_action_args
# ----------------------------------------------------------------------------


def test_validate_action_args_rejects_non_dict():
    with pytest.raises(ValueError, match="must decode to an object"):
        validate_action_args("read_file", "string")


def test_validate_action_args_rejects_missing_required_key():
    with pytest.raises(ValueError, match="missing required keys"):
        validate_action_args("read_file", {})


def test_validate_action_args_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown keys"):
        validate_action_args("read_file", {"path": "/x", "extra": "boom"})


def test_validate_action_args_rejects_non_integer_for_integer_field():
    with pytest.raises(ValueError, match="start_line: expected integer or null"):
        validate_action_args("read_file", {"path": "/x", "start_line": "1"})


def test_validate_action_args_accepts_null_for_optional_integer():
    out = validate_action_args("read_file", {"path": "/x", "start_line": None})
    assert out == {"path": "/x", "start_line": None}


def test_validate_action_args_rejects_empty_required_string():
    # A required string field must be non-empty (validation owns this so the
    # error is a ValueError inside act()'s repair loop, not a later TypeError).
    with pytest.raises(ValueError, match="content: expected non-empty string"):
        validate_action_args("write_file", {"path": "/x", "content": ""})


def test_validate_action_args_rejects_non_string_for_optional_string_field():
    with pytest.raises(ValueError, match="expected string or null"):
        validate_action_args("find_files", {"pattern": "*.py", "root": 5})


def test_validate_action_args_accepts_empty_string_for_optional_string_field():
    # Asymmetry preserved from the original _required_str/_optional_str split:
    # an optional string may be empty even though a required one may not.
    out = validate_action_args("find_files", {"pattern": "*.py", "root": ""})
    assert out == {"pattern": "*.py", "root": ""}


# ----------------------------------------------------------------------------
# build_action
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, args, action_cls",
    [
        ("list_dir", {"path": "/x"}, ListDirAction),
        ("find_files", {"pattern": "*.py", "root": "/r"}, FindFilesAction),
        ("search_text", {"query": "foo", "root": "/r"}, SearchTextAction),
        (
            "read_file",
            {"path": "/x", "start_line": 1, "end_line": 10},
            ReadFileAction,
        ),
        ("write_file", {"path": "/x", "content": "hi"}, WriteFileAction),
        (
            "edit_file",
            {"path": "/x", "old_text": "a", "new_text": "b"},
            EditFileAction,
        ),
        ("run", {"command": "echo hi"}, RunAction),
        ("verify", {}, VerifyAction),
    ],
)
def test_build_action_round_trips(name, args, action_cls):
    action = build_action(name, validate_action_args(name, args))
    assert isinstance(action, action_cls)
    assert action.NAME == name


# ----------------------------------------------------------------------------
# build_tool_specs
# ----------------------------------------------------------------------------


def test_build_tool_specs_covers_all_action_names():
    specs = build_tool_specs()
    assert {s["function"]["name"] for s in specs} == set(ACTION_SPECS)


def test_build_tool_specs_marks_integer_fields_with_integer_type():
    properties_by_tool = {
        spec["function"]["name"]: spec["function"]["parameters"]["properties"]
        for spec in build_tool_specs()
    }

    assert "limit" not in properties_by_tool["list_dir"]
    assert "limit" not in properties_by_tool["find_files"]
    assert "limit" not in properties_by_tool["search_text"]
    assert properties_by_tool["read_file"]["path"]["type"] == "string"
    assert properties_by_tool["read_file"]["start_line"] == {
        "type": ["integer", "null"],
    }
    assert properties_by_tool["read_file"]["end_line"] == {
        "type": ["integer", "null"],
    }
    assert properties_by_tool["run"]["timeout_sec"] == {
        "type": ["integer", "null"],
    }


def test_build_tool_specs_allows_null_for_optional_string_fields():
    properties_by_tool = {
        spec["function"]["name"]: spec["function"]["parameters"]["properties"]
        for spec in build_tool_specs()
    }

    assert properties_by_tool["list_dir"]["path"]["type"] == ["string", "null"]
    assert properties_by_tool["find_files"]["root"]["type"] == ["string", "null"]
    assert properties_by_tool["search_text"]["root"]["type"] == ["string", "null"]
    assert properties_by_tool["run"]["cwd"]["type"] == ["string", "null"]


def test_build_tool_specs_lists_required_keys():
    by_name = {s["function"]["name"]: s for s in build_tool_specs()}
    assert by_name["find_files"]["function"]["parameters"]["required"] == ["pattern"]
    assert by_name["list_dir"]["function"]["parameters"]["required"] == []


# ----------------------------------------------------------------------------
# execute_action: each action compiles to the expected shell command
# ----------------------------------------------------------------------------


def test_execute_action_list_dir_compiles_ls_command():
    env = _StubEnv()
    asyncio.run(execute_action(env, ListDirAction(path="/work")))
    assert env.exec_calls[0]["command"] == "ls -1Ap /work"
    assert env.exec_calls[0]["workload"] == "light"


def test_execute_action_list_dir_defaults_to_dot():
    env = _StubEnv()
    asyncio.run(execute_action(env, ListDirAction()))
    assert env.exec_calls[0]["command"] == "ls -1Ap ."
    assert env.exec_calls[0]["workload"] == "light"


def test_execute_action_find_files_compiles_find_command():
    env = _StubEnv()
    asyncio.run(execute_action(env, FindFilesAction(pattern="*.py", root="src")))
    assert env.exec_calls[0]["command"] == "find src -name '*.py'"
    assert env.exec_calls[0]["workload"] == "light"


def test_execute_action_search_text_compiles_grep_command():
    env = _StubEnv()
    asyncio.run(execute_action(env, SearchTextAction(query="hello world", root=".")))
    assert env.exec_calls[0]["command"] == "grep -rn 'hello world' ."
    assert env.exec_calls[0]["workload"] == "light"


def test_execute_action_read_file_uses_sed_with_explicit_line_range():
    env = _StubEnv()
    asyncio.run(
        execute_action(env, ReadFileAction(path="/x", start_line=5, end_line=20))
    )
    assert env.exec_calls[0]["command"] == "sed -n '5,20p' /x"
    assert env.exec_calls[0]["workload"] == "light"


def test_execute_action_read_file_uses_default_window_when_no_end_line():
    stdout = "".join(f"{line}\n" for line in range(1, 202))
    env = _StubEnv(exec_states=[RawState(return_code=0, stdout=stdout, passed=True)])
    result = asyncio.run(execute_action(env, ReadFileAction(path="/x", start_line=1)))
    expected_end = DEFAULT_READ_WINDOW_LINES
    assert env.exec_calls[0]["command"] == f"sed -n '1,{expected_end}p' /x"
    assert env.exec_calls[0]["workload"] == "light"
    assert result.stdout == stdout


def test_execute_action_write_file_uses_printf():
    env = _StubEnv()
    asyncio.run(execute_action(env, WriteFileAction(path="/x", content="hi there")))
    assert env.exec_calls[0]["command"] == "printf %s 'hi there' > /x"
    assert env.exec_calls[0]["workload"] == "light"


def test_execute_action_edit_file_uses_python_script():
    env = _StubEnv()
    asyncio.run(
        execute_action(env, EditFileAction(path="/x", old_text="foo", new_text="BAR"))
    )
    cmd = env.exec_calls[0]["command"]
    assert cmd.startswith("python3 -c ")
    assert "p.read_text().replace" in cmd
    assert cmd.endswith(" /x foo BAR")
    assert env.exec_calls[0]["workload"] == "light"


def test_execute_action_run_passes_through_command_cwd_timeout():
    env = _StubEnv()
    asyncio.run(
        execute_action(env, RunAction(command="pytest", cwd="/repo", timeout_sec=31))
    )
    assert env.exec_calls[0] == {
        "command": "pytest",
        "cwd": "/repo",
        "timeout_sec": 31,
        "workload": "heavy",
    }


def test_execute_action_short_timeout_run_uses_light_workload():
    env = _StubEnv()
    asyncio.run(
        execute_action(env, RunAction(command="python probe.py", timeout_sec=30))
    )
    assert env.exec_calls[0] == {
        "command": "python probe.py",
        "cwd": None,
        "timeout_sec": 30,
        "workload": "light",
    }


def test_execute_action_verify_calls_env_verify():
    env = _StubEnv()
    asyncio.run(execute_action(env, VerifyAction()))
    assert env.verify_calls == 1
    assert env.exec_calls == []


# ----------------------------------------------------------------------------
# render_tool_result
# ----------------------------------------------------------------------------


def test_render_tool_result_includes_rc_stdout_stderr():
    rendered = render_tool_result(
        RawState(return_code=0, stdout="out", stderr="err"), char_limit=100
    )
    assert "rc=0" in rendered
    assert "stdout:\nout" in rendered
    assert "stderr:\nerr" in rendered


def test_render_tool_result_returns_no_output_when_state_is_empty():
    assert render_tool_result(RawState(), char_limit=100) == "(no output)"


def test_render_tool_result_truncates_long_output():
    long = "x" * 1000
    rendered = render_tool_result(RawState(return_code=0, stdout=long), char_limit=100)
    assert "...[truncated]" in rendered


# ----------------------------------------------------------------------------
# build_system_prompt + build_initial_user_prompt
# ----------------------------------------------------------------------------


def test_build_system_prompt_mentions_verify():
    assert "verify" in build_system_prompt()


def test_build_initial_user_prompt_includes_instruction_and_working_dir():
    prompt = build_initial_user_prompt(instruction="do thing", working_dir="/work")
    assert "do thing" in prompt
    assert "/work" in prompt


def test_build_initial_user_prompt_handles_none_working_dir():
    prompt = build_initial_user_prompt(instruction="do thing", working_dir=None)
    assert "(unknown)" in prompt


# ----------------------------------------------------------------------------
# build_messages
# ----------------------------------------------------------------------------


def test_build_messages_starts_with_system_then_user():
    msgs = build_messages(instruction="do thing", working_dir="/work", trajectory=())
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "do thing" in msgs[1]["content"]


def test_build_messages_renders_each_step_as_assistant_then_tool():
    trajectory = (
        (ListDirAction(path="/x"), RawState(return_code=0, stdout="a\nb\n")),
        (VerifyAction(), RawState(done=True, passed=True, reward=1.0)),
    )
    msgs = build_messages(instruction="do", working_dir="/w", trajectory=trajectory)
    # system + user + (assistant + tool) per step
    assert len(msgs) == 2 + 2 * 2
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "list_dir"
    assert msgs[3]["role"] == "tool"
    assert msgs[3]["tool_call_id"] == msgs[2]["tool_calls"][0]["id"]
    assert msgs[4]["tool_calls"][0]["function"]["name"] == "verify"


def test_build_messages_replays_full_write_content_for_cache_stability():
    # write_file must replay the full content (no redaction) so the chat
    # prefix stays byte-stable for prompt cache hits.
    long = "x" * 10_000
    trajectory = ((WriteFileAction(path="/x", content=long), RawState(return_code=0)),)
    msgs = build_messages(instruction="do", working_dir="/w", trajectory=trajectory)
    args = json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"])
    assert args["content"] == long


def test_build_messages_emits_alphabetically_sorted_tool_arguments():
    # The replayed `arguments` JSON must stay byte-stable and alphabetically
    # keyed (sort_keys), independent of dataclass field order, so the prompt
    # prefix is cache-stable.
    trajectory = (
        (ReadFileAction(path="/x", start_line=1, end_line=10), RawState(return_code=0)),
    )
    msgs = build_messages(instruction="do", working_dir="/w", trajectory=trajectory)
    assert (
        msgs[2]["tool_calls"][0]["function"]["arguments"]
        == '{"end_line":10,"path":"/x","start_line":1}'
    )


def test_build_messages_prefix_is_byte_stable_when_step_appended():
    t1 = ((VerifyAction(), RawState(done=True, passed=True)),)
    t2 = (
        *t1,
        (ListDirAction(path="/x"), RawState(return_code=0, stdout="hi")),
    )
    m1 = build_messages(instruction="do", working_dir="/w", trajectory=t1)
    m2 = build_messages(instruction="do", working_dir="/w", trajectory=t2)
    assert m2[: len(m1)] == m1


# ----------------------------------------------------------------------------
# act
# ----------------------------------------------------------------------------


def test_act_returns_typed_actions_from_tool_calls():
    llm = _StubLlm([_completion(_tool_call("verify"))])
    actions = asyncio.run(
        act(
            llm=llm,
            instruction="do",
            working_dir="/w",
            trajectory=(),
            max_output_retries=2,
        )
    )
    assert actions == (VerifyAction(),)


def test_act_retries_on_missing_tool_call_and_appends_repair_prompt():
    llm = _StubLlm(
        [
            _completion(content="thinking..."),  # no tool_calls
            _completion(_tool_call("verify")),
        ]
    )
    actions = asyncio.run(
        act(
            llm=llm,
            instruction="do",
            working_dir="/w",
            trajectory=(),
            max_output_retries=2,
        )
    )
    assert actions == (VerifyAction(),)
    assert len(llm.calls) == 2
    second_contents = _string_contents(llm.calls[1])
    assert any(MISSING_TOOL_CALL_REPAIR_PROMPT in c for c in second_contents)


def test_act_retries_on_invalid_json_and_appends_correction():
    llm = _StubLlm(
        [
            _completion(LlmToolCall(name="verify", arguments="not json")),
            _completion(_tool_call("verify")),
        ]
    )
    actions = asyncio.run(
        act(
            llm=llm,
            instruction="do",
            working_dir="/w",
            trajectory=(),
            max_output_retries=2,
        )
    )
    assert actions == (VerifyAction(),)
    second_contents = _string_contents(llm.calls[1])
    assert any("invalid" in c.lower() for c in second_contents)


def test_act_raises_after_exhausting_retries():
    # Exhausting the output-retry budget raises the typed NoValidActionError (a
    # RuntimeError subclass) so the trial layer classifies it as an agent
    # failure (no_valid_action) rather than an infra crash.
    llm = _StubLlm([_completion(content="no calls")] * 3)
    with pytest.raises(NoValidActionError, match="failed to parse"):
        asyncio.run(
            act(
                llm=llm,
                instruction="do",
                working_dir="/w",
                trajectory=(),
                max_output_retries=2,
            )
        )


def test_act_returns_all_calls_in_multi_tool_response():
    llm = _StubLlm(
        [
            _completion(
                _tool_call("list_dir", path="/a"),
                _tool_call("list_dir", path="/b"),
            )
        ]
    )
    actions = asyncio.run(
        act(
            llm=llm,
            instruction="do",
            working_dir="/w",
            trajectory=(),
            max_output_retries=2,
        )
    )
    assert actions == (ListDirAction(path="/a"), ListDirAction(path="/b"))


# ----------------------------------------------------------------------------
# run_task_loop
# ----------------------------------------------------------------------------


def test_run_task_loop_solved_when_verify_returns_passed_true():
    llm = _StubLlm([_completion(_tool_call("verify"))])
    env = _StubEnv(verify_state=RawState(done=True, passed=True, reward=1.0))
    state = TaskLoopState()
    asyncio.run(
        run_task_loop(
            llm=llm,
            env=env,
            reset_state=asyncio.run(env.reset()),
            max_steps=5,
            state=state,
        )
    )
    assert state.solved is True
    assert state.reward == 1.0
    assert state.steps_used == 1
    assert env.verify_calls == 1


def test_run_task_loop_unsolved_when_verify_returns_passed_false():
    llm = _StubLlm([_completion(_tool_call("verify"))])
    env = _StubEnv(verify_state=RawState(done=True, passed=False, reward=0.0))
    state = TaskLoopState()
    asyncio.run(
        run_task_loop(
            llm=llm,
            env=env,
            reset_state=asyncio.run(env.reset()),
            max_steps=5,
            state=state,
        )
    )
    assert state.solved is False
    assert state.reward == 0.0


def test_run_task_loop_unsolved_when_max_steps_reached_without_verify():
    # Seed has no forced final verify -- step exhaustion just returns unsolved.
    llm = _StubLlm([_completion(_tool_call("list_dir", path="/x")) for _ in range(3)])
    env = _StubEnv()
    state = TaskLoopState()
    asyncio.run(
        run_task_loop(
            llm=llm,
            env=env,
            reset_state=asyncio.run(env.reset()),
            max_steps=3,
            state=state,
        )
    )
    assert state.solved is False
    assert state.steps_used == 3
    assert env.verify_calls == 0


def test_run_task_loop_stops_batch_when_action_returns_done_true():
    # act() emits two calls in one turn; the first marks the trial done,
    # so the seed's inner break must skip executing the second.
    llm = _StubLlm(
        [
            _completion(
                _tool_call("verify"),
                _tool_call("list_dir", path="/x"),
            )
        ]
    )
    env = _StubEnv(verify_state=RawState(done=True, passed=True, reward=1.0))
    state = TaskLoopState()
    asyncio.run(
        run_task_loop(
            llm=llm,
            env=env,
            reset_state=asyncio.run(env.reset()),
            max_steps=5,
            state=state,
        )
    )
    assert state.solved is True
    assert env.verify_calls == 1
    assert env.exec_calls == []
