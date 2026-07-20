"""Tests for the cross-run environment step-cache plugin.

No Docker: a scripted fake env stands in for the real one. Each test isolates
the shared cache store onto a tmp SQLite file, matching the completion-cache
test pattern.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import itertools
import json
import logging
from functools import wraps

import pytest

from src.plugins.caching import store as cc
import src.plugins.replay.step_cache as replay_module
import src.plugins.replay.contract as contract
from src.env.base import (
    RawEnvOutput,
    RunAction,
    VerifyOutcome,
    VerifyVerdict,
    VerifyAction,
)
from src.env.base import StepResult, execute_env_action
from src.plugins.replay.step_cache import ReplayCache, make_replay_cache
from src.rollout.execution import ExecutionDriftError

from conftest import _result


def _async_test(test):
    """Run one async scenario while preserving pytest's fixture signature."""

    @wraps(test)
    def run(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))

    return run


class _ScriptedEnvBase:
    """Returns scripted results in call order; 'timeout' raises TimeoutError."""

    def __init__(self, results) -> None:
        self._results = list(results)
        self.step_calls: list[RunAction | VerifyAction] = []
        self.closed = False

    async def reset(self) -> RawEnvOutput:
        return RawEnvOutput(instruction="do the task")

    async def _result_for(self, action: RunAction | VerifyAction) -> StepResult:
        self.step_calls.append(action)
        result = self._results[len(self.step_calls) - 1]
        if result == "timeout":
            raise TimeoutError("command timed out")
        return result

    async def execute(self, action: RunAction) -> RawEnvOutput:
        return (await self._result_for(action)).raw_env_output

    async def verify(self) -> VerifyOutcome:
        result = await self._result_for(VerifyAction())
        if result.verdict is None:
            raise AssertionError("scripted verify result must include a verdict")
        return VerifyOutcome(
            verdict=result.verdict,
            output=result.raw_env_output,
            reward=result.reward,
            info=result.info,
            metrics=result.metrics,
        )

    async def close(self) -> None:
        self.closed = True


class _ScriptedEnv(_ScriptedEnvBase):
    def __init__(self, results) -> None:
        super().__init__(results)
        self.verify_artifacts: list[StepResult] = []

    def write_verify_artifacts(self, result: StepResult) -> None:
        self.verify_artifacts.append(result)


class _NoArtifactScriptedEnv(_ScriptedEnvBase):
    """TaskEnv fake with no VerifyArtifactWriter capability."""


class _DelayedScriptedEnv(_ScriptedEnv):
    def __init__(self, results, *, delay_sec: float) -> None:
        super().__init__(results)
        self._delay_sec = delay_sec

    async def _result_for(self, action: RunAction | VerifyAction) -> StepResult:
        await asyncio.sleep(self._delay_sec)
        return await super()._result_for(action)


class _ScopedScriptedEnv(_ScriptedEnv):
    def __init__(self, results) -> None:
        super().__init__(results)
        self.pinned_snapshots: list[str] = []

    def pin_network_snapshot(self, token: str) -> None:
        self.pinned_snapshots.append(token)


_A1 = RunAction(command="echo one", timeout_sec=240.0)
_A2 = RunAction(command="echo two", timeout_sec=240.0)
_A3 = RunAction(command="echo three", timeout_sec=240.0)
_A4 = RunAction(command="echo four", timeout_sec=240.0)
_BG = RunAction(command="python -m http.server 8000 &", timeout_sec=240.0)
_COMMAND_TIMEOUT_EXIT_CODE = 124


def _chain_key(namespace: str, *actions) -> str:
    digest = hashlib.sha256(
        f'{{"epoch":0,"namespace":"{namespace}"}}'.encode()
    ).hexdigest()
    for action in actions:
        canonical = replay_module._canonical_action(action)
        digest = hashlib.sha256(f"{digest}|{canonical}".encode()).hexdigest()
    return f"env:{digest}"


async def _step_verify(cache):
    cached = await cache.prepare_submit()
    if cached is not None:
        return cached
    return await cache.step_verify()


async def _record(namespace: str, actions, results) -> _ScriptedEnv:
    inner = _ScriptedEnv(results)
    cache = ReplayCache(namespace=namespace, epoch=0, env=inner)
    for action in actions:
        if isinstance(action, VerifyAction):
            await _step_verify(cache)
        else:
            await cache.step_run(action)
    return inner


@_async_test
async def test_record_then_replay_never_touches_inner(store_env):
    r1, r2 = _result("one"), _result("two")
    await _record("ns", [_A1, _A2], [r1, r2])
    inner = _ScriptedEnv([])  # Any inner call fails.
    cache = ReplayCache(namespace="ns", epoch=0, env=inner)
    await cache.provision()
    assert await cache.step_run(_A1) == r1
    assert await cache.step_run(_A2) == r2
    assert inner.step_calls == []


@_async_test
async def test_fork_materializes_prefix_then_runs_live(store_env):
    await _record("ns", [_A1], [_result("one")])
    inner = _ScriptedEnv([_result("one"), _result("three")])
    cache = ReplayCache(namespace="ns", epoch=0, env=inner)
    assert (await cache.step_run(_A1)).raw_env_output.stdout == "one"
    assert inner.step_calls == []
    result = await cache.step_run(_A3)
    assert result.raw_env_output.stdout == "three"
    assert inner.step_calls == [_A1, _A3]


@pytest.mark.parametrize(
    ("content_id", "has_cache", "scopes"),
    [
        (
            "swe:version:task-1",
            True,
            [
                contract._network_snapshot_token(
                    namespace=f"{contract._CACHE_SCHEMA}:swe:version:task-1",
                    epoch=0,
                )
            ],
        ),
        (None, False, []),
    ],
    ids=("scoped", "live-only"),
)
@_async_test
async def test_make_replay_cache_applies_available_scope(
    store_env, content_id, has_cache, scopes
):
    inner = _ScopedScriptedEnv([])
    cache = await make_replay_cache(content_id=content_id, env=inner)
    assert (cache is not None) is has_cache
    assert inner.pinned_snapshots == scopes


@_async_test
async def test_replayed_prefix_keeps_only_hash_and_cache_key(store_env):
    stdout = "x" * 256
    await _record("ns", [_A1], [_result(stdout)])
    inner = _ScriptedEnv([_result(stdout)])
    cache = ReplayCache(namespace="ns", epoch=0, env=inner)

    assert (await cache.step_run(_A1)).raw_env_output.stdout == stdout
    [replayed] = cache._replayed
    assert replayed.action == _A1
    assert replayed.key == _chain_key("ns", _A1)
    assert replayed.recorded_hash == replay_module.scrubbed_hash(_result(stdout))
    assert not hasattr(replayed, "recorded")


@_async_test
async def test_corrupt_replay_row_hard_fails(store_env):
    await _record(
        "ns", [_A1, _A2, _A3], [_result("one"), _result("two"), _result("three")]
    )
    # put() is INSERT OR IGNORE, so corrupt the recorded row directly.
    conn = cc.store()._conn()
    conn.execute(
        "UPDATE cache SET value=? WHERE key=?",
        ("{corrupt", _chain_key("ns", _A1, _A2)),
    )
    conn.commit()
    inner = _ScriptedEnv([_result("one"), _result("two"), _result("three-live")])
    cache = ReplayCache(namespace="ns", epoch=0, env=inner)
    await cache.step_run(_A1)
    assert inner.step_calls == []
    with pytest.raises(json.JSONDecodeError):
        await cache.step_run(_A2)
    assert inner.step_calls == []


@_async_test
async def test_blocked_key_makes_materialized_suffix_live_and_unrecordable(store_env):
    await _record(
        "ns", [_A1, _A2, _A3], [_result("one"), _result("two"), _result("three")]
    )
    await cc.put(f"{_chain_key('ns', _A1, _A2)}:live-only", '{"live_only":true}')
    inner = _ScriptedEnv(
        [_result("one"), _result("two-live"), _result("three-live"), _result("four")]
    )
    cache = ReplayCache(namespace="ns", epoch=0, env=inner)

    await cache.step_run(_A1)
    assert inner.step_calls == []
    # Blocked key: materialization re-runs the a1 prefix live, then a2.
    await cache.step_run(_A2)
    assert inner.step_calls == [_A1, _A2]
    assert (await cache.step_run(_A3)).raw_env_output.stdout == "three-live"
    assert (await cache.step_run(_A4)).raw_env_output.stdout == "four"
    assert inner.step_calls == [_A1, _A2, _A3, _A4]
    assert await cc.get(_chain_key("ns", _A1, _A2, _A3, _A4)) is None


@_async_test
async def test_blocked_initial_key_runs_live_from_first_step(store_env):
    await _record("ns", [_A1], [_result("one")])
    await cc.put(f"{_chain_key('ns', _A1)}:live-only", '{"live_only":true}')
    inner = _ScriptedEnv([_result("one-live"), _result("two-live")])
    cache = ReplayCache(namespace="ns", epoch=0, env=inner)
    # A blocked first key leaves no prefix to replay: live from step 1, nothing recorded.
    await cache.step_run(_A1)
    await cache.step_run(_A2)
    assert inner.step_calls == [_A1, _A2]
    assert await cc.get(_chain_key("ns", _A1, _A2)) is None


@_async_test
async def test_background_step_is_live_only_never_recorded(store_env):
    await _record(
        "ns",
        [_A1, _BG, _A2],
        [_result("one"), _result("server started"), _result("ok")],
    )
    assert await cc.get(_chain_key("ns", _A1)) is not None
    assert await cc.get(_chain_key("ns", _A1, _BG)) is None
    assert await cc.get(_chain_key("ns", _A1, _BG, _A2)) is None

    inner = _ScriptedEnv([_result("one"), _result("server live"), _result("two live")])
    cache = ReplayCache(namespace="ns", epoch=0, env=inner)
    assert (await cache.step_run(_A1)).raw_env_output.stdout == "one"
    assert inner.step_calls == []
    assert (await cache.step_run(_BG)).raw_env_output.stdout == "server live"
    assert inner.step_calls == [
        _A1,
        _BG,
    ]
    assert (await cache.step_run(_A2)).raw_env_output.stdout == "two live"
    assert inner.step_calls == [_A1, _BG, _A2]


_LIVE_PROCESS_CASES = {
    "heredoc-content": (
        [
            """cat > encoder.py << 'PYEOF'
def bit(x):
    return (x >> 1) & 1
PYEOF
python -m py_compile encoder.py
""",
            """cat <<A <<B
x & 1
A
y & 1
B
echo done
""",
            """cat <<< 'value & 1'
echo done
""",
        ],
        False,
    ),
    "quoted-and-redirection": (
        ['python -c "print(1 & 1)"', "apt-get update 2>&1 && echo done"],
        False,
    ),
    "background-control": (
        [
            "python -m http.server 8000 &",
            "nohup python server.py >/tmp/server.log 2>&1 &",
            "cat <<EOF &\nx & 1\nEOF",
            "printf '<<EOF'\npython server.py &",
        ],
        True,
    ),
}


@pytest.mark.parametrize("case", _LIVE_PROCESS_CASES.values(), ids=_LIVE_PROCESS_CASES)
def test_live_process_detection(case):
    commands, expected = case
    assert all(
        replay_module._acquires_live_process_state(
            RunAction(command=command, timeout_sec=300)
        )
        is expected
        for command in commands
    )


@_async_test
async def test_catchup_drift_marks_step_live_only_instead_of_bumping_epoch(store_env):
    # The contaminated rollout still fails; repair only the drifted chain step so the next attempt runs it live without invalidating the namespace.
    await _record("ns", [_A1, _A2], [_result("one"), _result("two")])
    inner = _ScriptedEnv([_result("DRIFTED"), _result("three")])
    cache = ReplayCache(namespace="ns", epoch=0, env=inner)
    await cache.step_run(_A1)
    with pytest.raises(ExecutionDriftError, match="drift") as raised:
        await cache.step_run(_A3)
    error = raised.value
    assert (
        error.diagnostic["namespace"],
        error.diagnostic["epoch"],
        error.action_index,
        error.diagnostic["action"],
        error.diagnostic["recorded"]["raw_env_output"]["stdout"],
        error.diagnostic["live"]["raw_env_output"]["stdout"],
    ) == ("ns", 0, 1, {"kind": "run", **dataclasses.asdict(_A1)}, "one", "DRIFTED")

    healed_inner = _ScriptedEnv([_result("new-one"), _result("new-two")])
    healed = ReplayCache(namespace="ns", epoch=0, env=healed_inner)
    assert (await healed.step_run(_A1)).raw_env_output.stdout == "new-one"
    assert (await healed.step_run(_A2)).raw_env_output.stdout == "new-two"
    assert healed_inner.step_calls == [_A1, _A2]


_DRIFT_SIGNAL_CASES = {
    "known-header-name-case": (
        _result("X-Request-Id: abc\nContent-Type: text/plain\n"),
        _result("x-request-id: abc\ncontent-type: text/plain\n"),
        False,
    ),
    "unknown-header-name-case": (
        _result("X-New-Cache-Status: HIT\n"),
        _result("x-new-cache-status: HIT\n"),
        False,
    ),
    "stderr-line-order": (
        _result(
            "done",
            raw_env_output=RawEnvOutput(
                exit_code=0,
                stdout="done",
                stderr="worker one warning\nworker two warning\nworker one warning",
            ),
        ),
        _result(
            "done",
            raw_env_output=RawEnvOutput(
                exit_code=0,
                stdout="done",
                stderr="worker one warning\nworker one warning\nworker two warning",
            ),
        ),
        False,
    ),
    "numeric-stdout": (_result("answer: 41\n"), _result("answer: 42\n"), True),
    "header-value": (
        _result("X-Request-Id: abc\n"),
        _result("x-request-id: def\n"),
        True,
    ),
}


@pytest.mark.parametrize("case", _DRIFT_SIGNAL_CASES.values(), ids=_DRIFT_SIGNAL_CASES)
@_async_test
async def test_catchup_drift_audit_preserves_only_semantic_signal(store_env, case):
    recorded, live, drifts = case
    await _record("ns", [_A1], [recorded])
    inner = _ScriptedEnv([live, _result("three")])
    cache = ReplayCache(namespace="ns", epoch=0, env=inner)
    assert await cache.step_run(_A1) == recorded

    if drifts:
        with pytest.raises(ExecutionDriftError):
            await cache.step_run(_A3)
        assert inner.step_calls == [_A1]
    else:
        assert (await cache.step_run(_A3)).raw_env_output.stdout == "three"
        assert inner.step_calls == [_A1, _A3]


@_async_test
async def test_inner_timeout_error_is_not_cached(store_env):
    with pytest.raises(TimeoutError):
        await ReplayCache(
            namespace="ns", epoch=0, env=_ScriptedEnv(["timeout"])
        ).step_run(_A1)

    inner = _ScriptedEnv([_result("would-succeed")])
    result = await ReplayCache(namespace="ns", epoch=0, env=inner).step_run(_A1)
    assert result.raw_env_output.stdout == "would-succeed"
    assert inner.step_calls == [_A1]


@_async_test
async def test_command_timeout_exit_result_is_live_only_never_recorded(store_env):
    timed_out = _result(
        "partial",
        raw_env_output=RawEnvOutput(
            exit_code=_COMMAND_TIMEOUT_EXIT_CODE,
            stdout="partial",
            stderr="[command timed out after 240.0s]",
        ),
    )
    await _record("ns", [_A1, _A2], [timed_out, _result("two")])
    assert await cc.get(_chain_key("ns", _A1)) is None
    assert await cc.get(_chain_key("ns", _A1, _A2)) is None

    inner = _ScriptedEnv([_result("live timeout"), _result("two live")])
    cache = ReplayCache(namespace="ns", epoch=0, env=inner)
    assert (await cache.step_run(_A1)).raw_env_output.stdout == "live timeout"
    assert (await cache.step_run(_A2)).raw_env_output.stdout == "two live"
    assert inner.step_calls == [_A1, _A2]


@_async_test
async def test_host_stall_below_deadline_slack_still_caches_success(
    store_env, monkeypatch
):
    action = RunAction(command="fast-success", timeout_sec=100.0)
    stall_inner = _ScriptedEnv([_result("cached despite host stall")])
    cache = ReplayCache(namespace="ns", epoch=0, env=stall_inner)
    times = itertools.chain([0.0, 85.0], itertools.repeat(100.0))
    monkeypatch.setattr(replay_module.time, "monotonic", lambda: next(times))
    await cache.step_run(action)

    inner = _ScriptedEnv([_result("fresh")])
    result = await ReplayCache(namespace="ns", epoch=0, env=inner).step_run(action)
    assert result.raw_env_output.stdout == "cached despite host stall"
    assert inner.step_calls == []


@_async_test
async def test_near_deadline_step_is_live_only_with_bounded_log_and_suffix(
    store_env, caplog
):
    hidden_tail = "UNBOUNDED_TAIL_SHOULD_NOT_APPEAR"
    slow = RunAction(command=f"printf '{'x' * 500}{hidden_tail}'", timeout_sec=0.001)
    inner = _DelayedScriptedEnv(
        [_result("barely finished"), _result("two")], delay_sec=0.002
    )
    cache = ReplayCache(namespace="ns", epoch=0, env=inner)
    caplog.set_level(logging.WARNING, logger=replay_module.__name__)

    assert (await cache.step_run(slow)).raw_env_output.stdout == "barely finished"
    await cache.step_run(_A2)
    assert await cc.get(_chain_key("ns", slow)) is None
    assert await cc.get(_chain_key("ns", slow, _A2)) is None
    assert "env step cache skipped recording" in caplog.text
    assert "near_deadline" in caplog.text
    assert "RunAction(command=" in caplog.text
    assert "..." in caplog.text
    assert hidden_tail not in caplog.text

    replay_inner = _ScriptedEnv([_result("fresh"), _result("two fresh")])
    replay = ReplayCache(namespace="ns", epoch=0, env=replay_inner)
    assert (await replay.step_run(slow)).raw_env_output.stdout == "fresh"
    await replay.step_run(_A2)
    assert replay_inner.step_calls == [slow, _A2]


@_async_test
async def test_legacy_timeout_sentinel_hard_fails(store_env):
    await cc.put(_chain_key("ns", _A1), '{"timeout":true}')
    inner = _ScriptedEnv([_result("finished"), _result("two")])
    cache = ReplayCache(namespace="ns", epoch=0, env=inner)
    with pytest.raises(KeyError):
        await cache.step_run(_A1)
    assert inner.step_calls == []
    assert await cc.get(f"{_chain_key('ns', _A1)}:live-only") is None


@_async_test
async def test_verify_replay_without_artifact_writer_does_not_reconstruct(store_env):
    result = _result(
        "passed",
        reward=1.0,
        terminated=True,
        verdict=VerifyVerdict(completed=True, passed=True, error=None),
    )
    await _record("ns", [VerifyAction()], [result])
    inner = _NoArtifactScriptedEnv([])
    replayed = await _step_verify(ReplayCache(namespace="ns", epoch=0, env=inner))
    assert replayed.reward == 1.0
    assert inner.step_calls == []
    assert not hasattr(inner, "write_verify_artifacts")


@_async_test
async def test_prepare_submit_miss_materializes_before_live_verify(store_env):
    prefix = _result("one")
    verified = _result(
        "passed",
        reward=1.0,
        terminated=True,
        verdict=VerifyVerdict(completed=True, passed=True, error=None),
    )
    await _record("ns", [_A1], [prefix])
    inner = _ScriptedEnv([prefix, verified])
    cache = ReplayCache(namespace="ns", epoch=0, env=inner)

    assert await cache.step_run(_A1) == prefix
    assert await cache.prepare_submit() is None
    assert inner.step_calls == [_A1]
    assert await cache.step_verify() == verified
    assert inner.step_calls == [_A1, VerifyAction()]

    fresh_inner = _ScriptedEnv([])
    fresh = ReplayCache(namespace="ns", epoch=0, env=fresh_inner)
    assert await fresh.step_run(_A1) == prefix
    assert await fresh.prepare_submit() == verified
    assert fresh_inner.step_calls == []
    assert fresh_inner.verify_artifacts == [verified]

    await _record("drift-ns", [_A1], [prefix])
    drifted_inner = _ScriptedEnv([_result("DRIFTED")])
    drifted = ReplayCache(namespace="drift-ns", epoch=0, env=drifted_inner)
    assert await drifted.step_run(_A1) == prefix
    with pytest.raises(ExecutionDriftError, match="drift"):
        await drifted.prepare_submit()
    assert drifted_inner.step_calls == [_A1]


@_async_test
async def test_step_verify_requires_prepare_submit(store_env):
    result = _result(
        "passed",
        reward=1.0,
        terminated=True,
        verdict=VerifyVerdict(completed=True, passed=True, error=None),
    )
    inner = _ScriptedEnv([])
    cache = ReplayCache(namespace="ns", epoch=0, env=inner)
    with pytest.raises(RuntimeError, match="prepare_submit"):
        await cache.step_verify()

    await _record("recorded-verify", [VerifyAction()], [result])
    prepared = ReplayCache(namespace="recorded-verify", epoch=0, env=inner)
    assert await prepared.prepare_submit() == result
    with pytest.raises(RuntimeError, match="go live"):
        await prepared.step_verify()
    assert inner.step_calls == []


@_async_test
async def test_prepare_submit_already_live_has_no_catchup_awaits(store_env):
    inner = _ScriptedEnv([_result("one")])
    cache = ReplayCache(namespace="ns", epoch=0, env=inner)
    assert await cache.step_run(_A1) == _result("one")
    assert inner.step_calls == [_A1]
    async with asyncio.timeout_at(asyncio.get_running_loop().time() - 1.0):
        assert await cache.prepare_submit() is None
    assert inner.step_calls == [_A1]


@_async_test
async def test_non_json_info_hard_fails(store_env):
    opaque = _result(
        "one",
        reward=1.0,
        terminated=True,
        info={"handle": object()},
        verdict=VerifyVerdict(completed=True, passed=True, error=None),
    )
    with pytest.raises(TypeError):
        await _record("ns", [VerifyAction()], [opaque])
    assert await cc.get(_chain_key("ns", VerifyAction())) is None


@_async_test
async def test_fresh_cache_restores_recordable_chain_after_skipped_recording(store_env):
    one, two = _result("one"), _result("two")
    inner = _ScriptedEnv([one, _result("server started"), _result("three")])
    first = ReplayCache(namespace="ns", epoch=0, env=inner)
    assert await first.step_run(_A1) == one
    await first.step_run(_BG)
    await first.step_run(_A3)
    assert await cc.get(_chain_key("ns", _A1)) == replay_module.serialize_step_result(
        one
    )
    assert await cc.get(_chain_key("ns", _A1, _BG)) is None
    assert await cc.get(_chain_key("ns", _A1, _BG, _A3)) is None

    recordable_inner = _ScriptedEnv([one, two])
    next_attempt = ReplayCache(namespace="ns", epoch=0, env=recordable_inner)
    assert await next_attempt.step_run(_A1) == one
    assert await next_attempt.step_run(_A2) == two
    assert recordable_inner.step_calls == [_A1, _A2]
    assert await cc.get(
        _chain_key("ns", _A1, _A2)
    ) == replay_module.serialize_step_result(two)

    replay_inner = _ScriptedEnv([])
    replay = ReplayCache(namespace="ns", epoch=0, env=replay_inner)
    assert await replay.step_run(_A1) == one
    assert await replay.step_run(_A2) == two
    assert replay_inner.step_calls == []


def test_result_codec_roundtrips_and_matches_live_verify_bytes():
    result = _result(
        "out",
        reward=1.0,
        terminated=True,
        info={"passed": True, "rewards": {"reward": 1.0}},
        metrics={"fail_to_pass_passed": 2},
        verdict=VerifyVerdict(completed=True, passed=True, error=None),
    )
    payload = replay_module.serialize_step_result(result)
    assert payload is not None
    assert replay_module._deserialize(payload) == result
    assert json.loads(payload)["info"] == {"passed": True, "rewards": {"reward": 1.0}}
    assert json.loads(payload)["metrics"] == {"fail_to_pass_passed": 2}
    assert json.loads(payload)["verdict"] == {
        "completed": True,
        "passed": True,
        "error": None,
    }
    inner = _ScriptedEnv([result])
    live = asyncio.run(execute_env_action(inner, VerifyAction()))
    assert replay_module.serialize_step_result(live) == payload


def test_scrubbed_hash_uses_canonical_json_order():
    first = _result("same", info={"a": 1, "b": 2})
    second = _result("same", info={"b": 2, "a": 1})

    assert replay_module.scrubbed_hash(first) == replay_module.scrubbed_hash(second)


def test_cache_schema_fields_are_reviewed():
    expected = {
        RunAction: {"command", "cwd", "timeout_sec"},
        VerifyAction: set(),
        StepResult: {
            "raw_env_output",
            "reward",
            "terminated",
            "truncated",
            "info",
            "metrics",
            "verdict",
        },
    }
    assert {
        record_type: {field.name for field in dataclasses.fields(record_type)}
        for record_type in expected
    } == expected
