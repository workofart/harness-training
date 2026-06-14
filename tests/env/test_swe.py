"""Unit tests for the pure scoring logic of the SWE-bench env adapter.

`_score_report` turns a swebench eval report into (reward, resolved). The docker
round-trip is exercised end-to-end by the headroom spike; here we pin the
arithmetic + the resolution rule (every F2P passes AND every P2P still passes,
with at least one F2P actually present) against synthetic reports.
"""

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

from src.env.swe import VerifierCorruptError, _as_list, _grade_log, _score_report


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
