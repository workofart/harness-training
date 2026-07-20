from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.env import swebench_verify as swe_verify
from src.plugins.caching import store as cache
from src.plugins.replay import verify_cache


OFFICIAL_ARTIFACTS = swe_verify._OfficialArtifacts


def _spec(instance_id: str = "inst"):
    return SimpleNamespace(instance_id=instance_id)


def _result(*, stdout: str = "ok\n", passed: bool = True):
    # f2p/p2p match what SweBenchVerifyResult.from_report derives on restore, so a
    # store->serve round trip compares equal.
    return swe_verify.SweBenchVerifyResult(
        completed=True,
        passed=passed,
        stdout=stdout,
        report={"inst": {"resolved": passed}},
        fail_to_pass_passed=0,
        pass_to_pass_failed=0,
    )


def _grader(result, calls):
    async def grader(*, spec, patch, rollout_artifact_dir):
        calls.append((spec.instance_id, patch))
        return result

    return grader


def test_cache_wrapper_stores_then_serves_and_materializes(store_env, tmp_path):
    calls: list = []
    result = _result(stdout="official verifier output\n")
    cached = verify_cache.cache_wrapper(_grader(result, calls))

    first = asyncio.run(
        cached(spec=_spec(), patch="diffA", rollout_artifact_dir=tmp_path)
    )
    assert first == result
    assert calls == [("inst", "diffA")]

    # Second call: cache hit -> grader is not re-run, artifacts re-materialize.
    second = asyncio.run(
        cached(spec=_spec(), patch="diffA", rollout_artifact_dir=tmp_path)
    )
    assert second == result
    assert calls == [("inst", "diffA")]
    root = OFFICIAL_ARTIFACTS.for_rollout(
        rollout_artifact_dir=tmp_path, instance_id="inst"
    ).root
    assert (root / OFFICIAL_ARTIFACTS.PATCH).read_text() == "diffA"
    assert (
        root / OFFICIAL_ARTIFACTS.TEST_OUTPUT
    ).read_text() == "official verifier output\n"
    assert json.loads((root / OFFICIAL_ARTIFACTS.REPORT).read_text()) == result.report


def test_cache_wrapper_keyed_on_instance_and_diff(store_env, tmp_path):
    calls: list = []
    cached = verify_cache.cache_wrapper(_grader(_result(), calls))
    asyncio.run(
        cached(spec=_spec("inst"), patch="diffA", rollout_artifact_dir=tmp_path)
    )
    asyncio.run(
        cached(spec=_spec("inst"), patch="diffB", rollout_artifact_dir=tmp_path)
    )
    asyncio.run(cached(spec=_spec("x"), patch="diffA", rollout_artifact_dir=tmp_path))
    # Distinct (instance, diff) each miss -> grader runs three times.
    assert calls == [("inst", "diffA"), ("inst", "diffB"), ("x", "diffA")]


def test_cache_wrapper_does_not_store_incomplete_grade(store_env, tmp_path):
    calls: list = []
    incomplete = swe_verify.SweBenchVerifyResult(completed=False, error="boom")
    cached = verify_cache.cache_wrapper(_grader(incomplete, calls))
    asyncio.run(cached(spec=_spec(), patch="d", rollout_artifact_dir=tmp_path))
    asyncio.run(cached(spec=_spec(), patch="d", rollout_artifact_dir=tmp_path))
    # Never stored, so the second call grades live again.
    assert len(calls) == 2


def test_cache_wrapper_treats_corrupt_payload_as_miss(store_env, tmp_path):
    calls: list = []
    result = _result()
    cached = verify_cache.cache_wrapper(_grader(result, calls))
    asyncio.run(
        cache.put(
            swe_verify.verify_cache_key("inst", "d"),
            json.dumps(
                {
                    "completed": True,
                    "stdout": "x",
                    "report": {"inst": {"resolved": "no"}},
                }
            ),
        )
    )
    out = asyncio.run(cached(spec=_spec(), patch="d", rollout_artifact_dir=tmp_path))
    assert out == result
    assert len(calls) == 1  # corrupt hit ignored -> graded live


def test_restore_write_failure_propagates(store_env, tmp_path, monkeypatch):
    calls: list = []
    cached = verify_cache.cache_wrapper(_grader(_result(), calls))
    asyncio.run(cached(spec=_spec(), patch="d", rollout_artifact_dir=tmp_path))
    paths = OFFICIAL_ARTIFACTS.for_rollout(
        rollout_artifact_dir=tmp_path, instance_id="inst"
    )
    original_write_text = Path.write_text

    def fail_patch_write(path: Path, data: str, *args, **kwargs):
        if path == paths.patch:
            raise OSError("disk full")
        return original_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_patch_write)
    with pytest.raises(OSError, match="disk full"):
        asyncio.run(cached(spec=_spec(), patch="d", rollout_artifact_dir=tmp_path))


def test_verify_cache_key_includes_package_version_and_schema(monkeypatch):
    original = swe_verify.verify_cache_key("inst", "diff")
    monkeypatch.setattr(swe_verify, "_SWEBENCH_VERSION", "different")
    assert swe_verify.verify_cache_key("inst", "diff") != original
    monkeypatch.setattr(swe_verify, "_VERIFY_CACHE_SCHEMA", "different-schema")
    assert swe_verify.verify_cache_key("inst", "diff") != original


def test_cache_wrapper_inert_when_cache_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "_DISABLED", True)
    calls: list = []
    result = _result()
    cached = verify_cache.cache_wrapper(_grader(result, calls))
    # Disabled store: get misses, put is fail-open, so the grader always runs.
    assert (
        asyncio.run(cached(spec=_spec(), patch="d", rollout_artifact_dir=tmp_path))
        == result
    )
    assert (
        asyncio.run(cached(spec=_spec(), patch="d", rollout_artifact_dir=tmp_path))
        == result
    )
    assert len(calls) == 2
