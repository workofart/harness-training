"""Official SWE-bench verification."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, ClassVar

import docker

from src.env.base import FrameworkEnvError

# Mirror report keys locally to avoid swebench's eager datasets/pandas import.
KEY_INSTANCE_ID = "instance_id"
KEY_MODEL = "model_name_or_path"
KEY_PREDICTION = "model_patch"
_LOG_REPORT = "report.json"
_LOG_TEST_OUTPUT = "test_output.txt"

OFFICIAL_EVAL_TIMEOUT_SEC = 1800
_SWEBENCH_LOG_DIR_LOCK = asyncio.Lock()


@dataclass(frozen=True, slots=True)
class _OfficialArtifacts:
    ROOT_DIR: ClassVar[Path] = Path("swebench_official")
    RUN_ID: ClassVar[str] = "verify"
    RUN_DIR: ClassVar[Path] = Path(RUN_ID)
    MODEL_NAME: ClassVar[str] = "framework_swe_hybrid"
    INSTANCE_PARENT_DIR: ClassVar[Path] = RUN_DIR / MODEL_NAME
    PATCH: ClassVar[Path] = Path("patch.diff")
    REPORT: ClassVar[Path] = Path(_LOG_REPORT)
    TEST_OUTPUT: ClassVar[Path] = Path(_LOG_TEST_OUTPUT)

    instance_id: str
    root: Path
    run_dir: Path
    instance_dir: Path
    patch: Path
    report: Path
    test_output: Path

    @classmethod
    def for_rollout(
        cls, *, rollout_artifact_dir: Path, instance_id: str
    ) -> "_OfficialArtifacts":
        root = rollout_artifact_dir / cls.ROOT_DIR
        return cls(
            instance_id=instance_id,
            root=root,
            run_dir=root / cls.RUN_DIR,
            instance_dir=root / cls.INSTANCE_PARENT_DIR / instance_id,
            patch=root / cls.PATCH,
            report=root / cls.REPORT,
            test_output=root / cls.TEST_OUTPUT,
        )


class VerifierCorruptError(FrameworkEnvError):
    """A SWE-bench grading run could not turn into an official report."""


@dataclass(frozen=True, slots=True)
class SweBenchVerifyResult:
    """Terminal SWE-bench grade for one submit.

    `completed` is the grading lifecycle flag: completed=True carries `passed`
    plus the official `report`; completed=False carries `error`.
    """

    completed: bool
    stdout: str = ""
    passed: bool | None = None
    report: Mapping[str, Any] | None = None
    error: str | None = None
    fail_to_pass_passed: int | None = None
    pass_to_pass_failed: int | None = None

    @property
    def reward(self) -> float:
        return 1.0 if (self.completed and self.passed) else 0.0

    @classmethod
    def from_report(
        cls, report: Mapping[str, Any], stdout: str, instance_id: str
    ) -> SweBenchVerifyResult:
        instance_report = report.get(instance_id)
        if not isinstance(instance_report, Mapping):
            raise VerifierCorruptError(
                f"official SWE-bench resolved field missing for {instance_id}"
            )
        resolved = instance_report.get("resolved")
        if not isinstance(resolved, bool):
            raise VerifierCorruptError(
                f"official SWE-bench resolved field is not boolean for {instance_id}"
            )
        tests_status = instance_report.get("tests_status", {})
        return cls(
            completed=True,
            passed=resolved,
            stdout=stdout,
            report=report,
            fail_to_pass_passed=len(
                tests_status.get("FAIL_TO_PASS", {}).get("success", [])
            ),
            pass_to_pass_failed=len(
                tests_status.get("PASS_TO_PASS", {}).get("failure", [])
            ),
        )


# Grade identity for the verify cache; bump the schema when the payload format
# (cache_payload/restore_cached) changes.
_VERIFY_CACHE_SCHEMA = "swe-verify-v2"
_SWEBENCH_VERSION = version("swebench")


def verify_cache_key(instance_id: str, diff: str) -> str:
    digest = hashlib.sha256(
        f"{_VERIFY_CACHE_SCHEMA}\0{_SWEBENCH_VERSION}\0{instance_id}\0{diff}".encode()
    ).hexdigest()
    return f"v:{digest}"


def restore_cached(
    raw: str, *, rollout_artifact_dir: Path, instance_id: str, diff: str
) -> SweBenchVerifyResult | None:
    """Rebuild a grade from a cached payload; None if unusable (caller retries live)."""
    paths = _OfficialArtifacts.for_rollout(
        rollout_artifact_dir=rollout_artifact_dir, instance_id=instance_id
    )
    try:
        cached = json.loads(raw)
        if cached["completed"] is not True:
            raise ValueError("cached verification is not completed")
        stdout = cached["stdout"]
        report = cached["report"]
        if not isinstance(stdout, str) or not isinstance(report, Mapping):
            raise TypeError("cached verification payload has invalid types")
        result = SweBenchVerifyResult.from_report(report, stdout, instance_id)
    except (
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        VerifierCorruptError,
    ):
        return None
    if paths.root.exists():
        shutil.rmtree(paths.root)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.patch.write_text(diff)
    paths.test_output.write_text(result.stdout)
    paths.report.write_text(json.dumps(result.report, indent=4))
    return result


def cache_payload(result: SweBenchVerifyResult) -> str:
    """Serialize a completed grade for the verify cache."""
    return json.dumps(
        {
            "completed": True,
            "stdout": result.stdout,
            "report": result.report,
        },
        separators=(",", ":"),
    )


async def verify(
    *,
    spec: Any,
    patch: str,
    rollout_artifact_dir: Path,
) -> SweBenchVerifyResult:
    paths = _OfficialArtifacts.for_rollout(
        rollout_artifact_dir=rollout_artifact_dir,
        instance_id=spec.instance_id,
    )
    docker_client = docker.from_env()
    try:
        official_result = await _run_official_eval(
            spec=spec,
            patch=patch,
            paths=paths,
            docker_client=docker_client,
        )
    finally:
        docker_client.close()

    completed = official_result["completed"]
    if completed and not paths.instance_dir.exists():
        raise VerifierCorruptError(
            f"official SWE-bench artifact dir missing for {paths.instance_id}"
        )
    if paths.instance_dir.exists():
        for source in paths.instance_dir.iterdir():
            shutil.move(source, paths.root / source.name)
        shutil.rmtree(paths.run_dir, ignore_errors=True)

    if completed:
        if not paths.report.exists():
            raise VerifierCorruptError(
                f"official SWE-bench report missing for {paths.instance_id}"
            )
        if not paths.test_output.exists():
            raise VerifierCorruptError(
                f"official SWE-bench test output missing for {paths.instance_id}"
            )
        report = json.loads(paths.report.read_text())
        result = SweBenchVerifyResult.from_report(
            report,
            paths.test_output.read_text(),
            paths.instance_id,
        )
        return result

    return SweBenchVerifyResult(
        completed=False,
        error=(
            "official SWE-bench evaluation did not complete for "
            f"{spec.instance_id}: {official_result}"
        ),
    )


async def _run_official_eval(
    *,
    spec: Any,
    patch: str,
    paths: _OfficialArtifacts,
    docker_client: Any,
) -> Mapping[str, Any]:
    # Defer swebench/Hugging Face imports until grading.
    from swebench.harness import run_evaluation

    shutil.rmtree(paths.root, ignore_errors=True)
    async with _SWEBENCH_LOG_DIR_LOCK:
        with _swebench_run_log_root(paths.root):
            raw_result = await asyncio.to_thread(
                run_evaluation.run_instance,
                test_spec=spec,
                pred={
                    KEY_INSTANCE_ID: spec.instance_id,
                    KEY_MODEL: _OfficialArtifacts.MODEL_NAME,
                    KEY_PREDICTION: patch,
                },
                rm_image=False,
                force_rebuild=False,
                client=docker_client,
                run_id=_OfficialArtifacts.RUN_ID,
                timeout=OFFICIAL_EVAL_TIMEOUT_SEC,
                rewrite_reports=False,
            )
    if not isinstance(raw_result, Mapping):
        raise VerifierCorruptError(
            f"official SWE-bench result is not a mapping for {spec.instance_id}"
        )
    completed = raw_result.get("completed")
    if not isinstance(completed, bool):
        raise VerifierCorruptError(
            f"official SWE-bench completed flag missing for {spec.instance_id}"
        )
    return raw_result


@contextlib.contextmanager
def _swebench_run_log_root(root: Path):
    from swebench.harness import run_evaluation

    previous = run_evaluation.RUN_EVALUATION_LOG_DIR
    run_evaluation.RUN_EVALUATION_LOG_DIR = root
    try:
        yield
    finally:
        run_evaluation.RUN_EVALUATION_LOG_DIR = previous
