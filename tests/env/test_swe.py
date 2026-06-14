"""Unit tests for the pure scoring logic of the SWE-bench env adapter.

`_score_report` turns a swebench eval report into (reward, resolved). The docker
round-trip is exercised end-to-end by the headroom spike; here we pin the
arithmetic + the resolution rule (every F2P passes AND every P2P still passes,
with at least one F2P actually present) against synthetic reports.
"""

import asyncio
from types import SimpleNamespace

import pytest
from swebench.harness.constants import (
    END_TEST_OUTPUT,
    FAIL_TO_PASS,
    MAP_REPO_VERSION_TO_SPECS,
    PASS_TO_PASS,
    START_TEST_OUTPUT,
    TESTS_ERROR,
)
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

from src.env import swe as swe_module
from src.env.swe import (
    DOCKER_INFRA_RETRY_BUDGET,
    SweEnv,
    VerifierCorruptError,
    _as_list,
    _grade_log,
    _score_report,
)


def _report(*, f2p_ok, f2p_bad, p2p_ok, p2p_bad) -> dict:
    return {
        FAIL_TO_PASS: {"success": f2p_ok, "failure": f2p_bad},
        PASS_TO_PASS: {"success": p2p_ok, "failure": p2p_bad},
    }


def test_full_pass_is_reward_one_and_resolved():
    report = _report(f2p_ok=["a"], f2p_bad=[], p2p_ok=["b", "c"], p2p_bad=[])
    assert _score_report(report) == (1.0, True)


def test_f2p_failing_is_p2p_floor_and_not_resolved():
    # 1 F2P fails, 5 P2P pass -> 5/6 floor, not resolved (the discriminating bit).
    report = _report(f2p_ok=[], f2p_bad=["a"], p2p_ok=list("bcdef"), p2p_bad=[])
    reward, resolved = _score_report(report)
    assert reward == 5 / 6
    assert resolved is False


def test_p2p_regression_breaks_resolution():
    # F2P all pass but the fix broke a P2P -> not resolved even though F2P is green.
    report = _report(f2p_ok=["a"], f2p_bad=[], p2p_ok=["b"], p2p_bad=["c"])
    reward, resolved = _score_report(report)
    assert reward == 2 / 3
    assert resolved is False


def test_no_f2p_present_is_not_resolved():
    # A repo with no F2P successes recorded must not count as resolved.
    report = _report(f2p_ok=[], f2p_bad=[], p2p_ok=["a", "b"], p2p_bad=[])
    reward, resolved = _score_report(report)
    assert reward == 1.0  # everything present passed...
    assert resolved is False  # ...but no F2P was demonstrated -> not a fix


def test_empty_report_is_zero_reward():
    assert _score_report(_report(f2p_ok=[], f2p_bad=[], p2p_ok=[], p2p_bad=[])) == (
        0.0,
        False,
    )


def test_as_list_passthrough_and_json():
    assert _as_list(["x", "y"]) == ["x", "y"]
    assert _as_list('["x", "y"]') == ["x", "y"]


# --- _grade_log: corruption must fail fast, not score as a rejection ---------
#
# A verify() run that swebench cannot grade (held-out test patch failed to apply,
# repo reset failed, test runner errored/timed out, or the output markers are
# absent) is infra noise. Scoring it 0.0 would record a `verified_rejected` and
# contaminate the graded gate; raising makes `run_trial` set `error` and exclude
# the trial. These pin that boundary against `get_logs_eval`'s real `ok` flag.


def _stub_spec():
    # get_logs_eval only needs repo+version to pick a parser; the corruption
    # branches return ({}, False) before the parser is ever invoked, so any valid
    # repo present in both maps works.
    repo = next(r for r in MAP_REPO_VERSION_TO_SPECS if r in MAP_REPO_TO_PARSER)
    version = next(iter(MAP_REPO_VERSION_TO_SPECS[repo]))
    return SimpleNamespace(repo=repo, version=version, instance_id="stub__repo-1")


def _grade(output, tmp_path):
    return _grade_log(
        spec=_stub_spec(),
        output=output,
        f2p=["test_fix"],
        p2p=["test_keep"],
        artifacts_dir=str(tmp_path),
    )


def test_missing_markers_raise_not_scored(tmp_path):
    # eval.sh died before emitting the markers (setup crash / OOM / timeout kill).
    output = "+ conda activate testbed\nImportError: cannot import name x\n"
    with pytest.raises(VerifierCorruptError):
        _grade(output, tmp_path)


def test_bad_code_sentinel_raises(tmp_path):
    # Markers present but a swebench corruption sentinel is too (apply/reset/
    # error/timeout class) -> get_logs_eval returns ok=False first.
    output = f"{START_TEST_OUTPUT}\n{TESTS_ERROR}\n{END_TEST_OUTPUT}\n"
    with pytest.raises(VerifierCorruptError):
        _grade(output, tmp_path)


def test_markers_present_no_tests_is_scored_zero_not_raised(tmp_path):
    # Markers present, no sentinel, parser finds nothing: swebench returns
    # ok=True. That is a gradable all-failing zero (a real verified_rejection),
    # NOT infra -- we must score it, not over-raise.
    output = f"{START_TEST_OUTPUT}\n(no recognizable test output)\n{END_TEST_OUTPUT}\n"
    reward, resolved = _grade(output, tmp_path)
    assert reward == 0.0
    assert resolved is False


# --- transient docker-daemon retry ------------------------------------------
#
# A momentary OrbStack/Docker socket reset ("error during connect ... EOF")
# wiped 107 container starts in a single 3.6s window and was recorded as 107
# crashes, silently dropping ~4 repos' worth of tasks from a characterization
# (a crash slot is never re-run -- record.py). `reset()` and the verify
# eval-script exec issue raw docker CLI calls; these pin that a docker-level
# failure (non-zero WITH output on docker's own stderr) is now retried, while a
# verdict from the inner command (non-zero, clean docker stderr) is NOT.


def _no_sleep(monkeypatch) -> None:
    async def _sleep(_seconds):
        return None

    monkeypatch.setattr(swe_module.asyncio, "sleep", _sleep)


def _fake_run_sequence(results):
    """A drop-in for `swe._run_once` that returns queued (rc, out, err) tuples
    and records how many times it was called (last tuple repeats once
    exhausted)."""
    calls = {"n": 0}

    async def _run_once(*_args, **_kwargs):
        i = calls["n"]
        calls["n"] += 1
        return results[min(i, len(results) - 1)]

    return _run_once, calls


def test_run_retries_a_docker_blip(monkeypatch):
    # First attempt = docker-level failure (non-zero WITH output on docker's own
    # stderr), second = success. retry=True must ride out the blip.
    _no_sleep(monkeypatch)
    blip = (1, "", 'error during connect: Get ".../_ping": EOF')
    ok = (0, "started", "")
    run_once, calls = _fake_run_sequence([blip, ok])
    monkeypatch.setattr(swe_module, "_run_once", run_once)

    rc, out, _err = asyncio.run(swe_module._run("docker", "run", retry=True))
    assert (rc, out) == (0, "started")
    assert calls["n"] == 2


def test_run_does_not_retry_a_verdict(monkeypatch):
    # A non-zero from the inner command (a failing test run) carries its output
    # on stdout and leaves docker's stderr clean -- it is a verdict, returned
    # as-is even under retry=True, never re-run.
    _no_sleep(monkeypatch)
    fail = (1, "3 failed, 5 passed", "")
    run_once, calls = _fake_run_sequence([fail, (0, "should-not-reach", "")])
    monkeypatch.setattr(swe_module, "_run_once", run_once)

    rc, out, _err = asyncio.run(swe_module._run("docker", "exec", retry=True))
    assert (rc, out) == (1, "3 failed, 5 passed")
    assert calls["n"] == 1


def test_run_without_retry_runs_once(monkeypatch):
    # Default retry=False (the exec/close/cp path): a docker-level failure is NOT
    # retried -- it returns as-is, so an agent command keeps raw semantics.
    _no_sleep(monkeypatch)
    blip = (1, "", "Cannot connect to the Docker daemon")
    run_once, calls = _fake_run_sequence([blip, (0, "unreached", "")])
    monkeypatch.setattr(swe_module, "_run_once", run_once)

    rc, _out, _err = asyncio.run(swe_module._run("docker", "exec"))
    assert rc == 1
    assert calls["n"] == 1


def test_run_gives_up_after_budget(monkeypatch):
    # A daemon that stays unreachable exhausts the budget and re-raises, so the
    # trial still records a crash (a real outage must fail, not hang forever).
    _no_sleep(monkeypatch)
    blip = (1, "", "Cannot connect to the Docker daemon")
    run_once, calls = _fake_run_sequence([blip])
    monkeypatch.setattr(swe_module, "_run_once", run_once)

    with pytest.raises(RuntimeError):
        asyncio.run(swe_module._run("docker", "run", retry=True))
    assert calls["n"] == DOCKER_INFRA_RETRY_BUDGET + 1


def test_reset_rides_out_a_daemon_blip(monkeypatch):
    # End-to-end at the reset() seam: a transient `docker run` failure must not
    # crash the trial -- reset() retries and returns a usable RawState.
    _no_sleep(monkeypatch)
    env = object.__new__(SweEnv)
    env._row = {"problem_statement": "fix it"}
    env._image = "sweb.eval.x86_64.demo:latest"
    env._container = "swe_env_demo"
    env._heavy_semaphore = None
    env._started = False

    blip = (1, "", 'error during connect: Get ".../_ping": EOF')
    ok = (0, "deadbeef", "")
    run_once, calls = _fake_run_sequence([blip, ok])
    monkeypatch.setattr(swe_module, "_run_once", run_once)

    state = asyncio.run(env.reset())
    assert state.instruction == "fix it"
    assert env._started is True
    assert calls["n"] == 2
