"""Unit tests for the SWE-bench env adapter boundaries."""

import asyncio
import hashlib
import json
import re
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from docker.errors import DockerException

from conftest import install_fake_swebench
from src import determinism
from src.config import EnvironmentConfig
import src.env.swe as swe_module
import src.env.docker_shell as docker_shell_module
from src.env.docker_shell import DockerShellSession
from src.env.base import (
    MODEL_PATCH_INFO_KEY,
    RunAction,
    VerifyAction,
    VerifyVerdict,
    execute_env_action,
    scrub_step_result,
)
from src.env.swe import _MODEL_PATCH_COMMAND, SweEnv
from src.env.base import StepResult
from src.env import swebench_verify as swe_verify

# Mirror report keys locally without importing swebench; the schema test guards
# version drift.
FAIL_TO_PASS = "FAIL_TO_PASS"
PASS_TO_PASS = "PASS_TO_PASS"

OFFICIAL_ARTIFACTS = swe_verify._OfficialArtifacts
# Only tests name the per-instance runner log; production moves it generically.
RUN_INSTANCE_LOG = "run_instance.log"


def _unset_run_instance(**_kwargs):
    raise AssertionError("verify-path test must set run_instance on the swebench fake")


# Fake lazy-import target; verify tests replace run_instance and production swaps
# the log root.
_swebench_run_evaluation = SimpleNamespace(
    run_instance=_unset_run_instance,
    RUN_EVALUATION_LOG_DIR=Path("logs/run_evaluation"),
)


@pytest.fixture(autouse=True)
def _stub_swebench(monkeypatch):
    _swebench_run_evaluation.run_instance = _unset_run_instance
    _swebench_run_evaluation.RUN_EVALUATION_LOG_DIR = Path("logs/run_evaluation")
    install_fake_swebench(monkeypatch, run_evaluation=_swebench_run_evaluation)


# The episode scrubs every transition before the policy sees it; mirror that
# here so these tests observe exactly what a rollout would.
def _step_run(env: SweEnv, command: str) -> StepResult:
    action = RunAction(command=command)
    result = asyncio.run(execute_env_action(env, action))
    return scrub_step_result(result, command=action.command)


def _step_submit(env: SweEnv) -> StepResult:
    result = asyncio.run(execute_env_action(env, VerifyAction()))
    return scrub_step_result(result, command=None)


def test_swe_env_exposes_official_grader_timeout(tmp_path) -> None:
    env = SweEnv(task=_task(), artifacts_dir=tmp_path)
    assert env.verify_timeout_sec == 1800.0


def _report(*, f2p_ok, f2p_bad, p2p_ok, p2p_bad) -> dict:
    return {
        FAIL_TO_PASS: {"success": f2p_ok, "failure": f2p_bad},
        PASS_TO_PASS: {"success": p2p_ok, "failure": p2p_bad},
    }


def test_model_patch_command_excludes_binary_keeps_source(tmp_path):
    def build_base_repo(repo: Path) -> None:
        (repo / "src").mkdir(parents=True)
        (repo / "src" / "mod.py").write_text("def fix():\n    return 1\n")
        (repo / "old binary.bin").write_bytes(b"\0" + b"a" * 2048)
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo,
            check=True,
        )
        subprocess.run(
            ["git", "add", "src/mod.py", "old binary.bin"], cwd=repo, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "base"],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    repo = tmp_path / "testbed"
    repo.mkdir()
    build_base_repo(repo)

    (repo / "src" / "mod.py").write_text("def fix():\n    return 42\n")
    (repo / "src" / "newmod.py").write_text("def new_fix():\n    return 42\n")
    (repo / "old binary.bin").rename(repo / "new binary [artifact].bin")
    (repo / "new binary [artifact].bin").write_bytes(b"\0" + b"a" * 2047 + b"b")
    build_dir = repo / "test issue" / "_build" / ".doctrees"
    build_dir.mkdir(parents=True)
    (build_dir / "environment.pickle").write_bytes(bytes(range(256)) * 8)

    result = subprocess.run(
        ["bash", "-lc", _MODEL_PATCH_COMMAND],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    patch = result.stdout

    assert "GIT binary patch" not in patch
    assert "Binary files" not in patch
    assert "environment.pickle" not in patch
    assert "binary [artifact]" not in patch
    assert "return 42" in patch
    assert "newmod.py" in patch

    apply_repo = tmp_path / "apply-testbed"
    apply_repo.mkdir()
    build_base_repo(apply_repo)
    apply_result = subprocess.run(
        ["git", "apply"],
        cwd=apply_repo,
        input=patch,
        capture_output=True,
        text=True,
    )
    assert apply_result.returncode == 0, apply_result.stderr
    assert "return 42" in (apply_repo / "src" / "mod.py").read_text()
    assert (apply_repo / "src" / "newmod.py").exists()


def test_swe_taskset_loads_requested_rows_and_builds_specs(monkeypatch, tmp_path):
    rows = [
        {"instance_id": "task-a", "problem_statement": "fix a"},
        {"instance_id": "task-b", "problem_statement": "fix b"},
    ]

    def fake_load_dataset(name, revision, split):
        assert name == "princeton-nlp/SWE-bench_Verified"
        assert revision == "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
        assert split == "test"
        return rows

    def fake_make_test_spec(row, namespace):
        assert namespace == "swebench"
        return SimpleNamespace(
            instance_id=row["instance_id"],
            instance_image_key=f"image-{row['instance_id']}",
        )

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )
    monkeypatch.setattr(
        sys.modules["swebench.harness.test_spec.test_spec"],
        "make_test_spec",
        fake_make_test_spec,
    )

    taskset = asyncio.run(
        swe_module.load_tasks(
            task_ids=["task-b"],
            environment=EnvironmentConfig(kind="swe", task_names=["task-b"]),
        )
    )
    env = taskset.task("task-b").make_env(tmp_path / "task-b")

    assert env._task.instruction == "fix b"
    assert env._task.spec.instance_id == "task-b"
    assert env._task.spec.instance_image_key == "image-task-b"
    # SWE-bench defines no per-task agent budget; the harness default applies.
    assert env._task.agent_timeout_sec is None
    expected = hashlib.sha256(
        f"{determinism.PINS_FINGERPRINT}\0{_MODEL_PATCH_COMMAND}".encode()
    ).hexdigest()[:12]
    assert re.fullmatch(r"[0-9a-f]{12}", taskset.tasks["task-b"].replay_id or "")
    assert taskset.tasks["task-b"].replay_id == expected
    with pytest.raises(KeyError):
        taskset.tasks["task-a"]


def _no_sleep(monkeypatch) -> None:
    async def _sleep(_seconds):
        return None

    monkeypatch.setattr(docker_shell_module.asyncio, "sleep", _sleep)


class _FakeContainer:
    id = "container-id"

    def __init__(
        self,
        *,
        exec_result=None,
        exec_errors=None,
        block: threading.Event | None = None,
    ):
        self.exec_result = exec_result or SimpleNamespace(
            exit_code=0,
            output=(b"", b""),
        )
        self.exec_errors = list(exec_errors or [])
        self.block = block
        self.exec_run_calls = []
        self.remove_calls = []

    def exec_run(self, *args, **kwargs):
        self.exec_run_calls.append((args, kwargs))
        if self.block is not None:
            self.block.wait(timeout=10)
        if self.exec_errors:
            raise self.exec_errors.pop(0)
        return self.exec_result

    def remove(self, *args, **kwargs):
        self.remove_calls.append((args, kwargs))
        if self.block is not None:
            self.block.set()


class _FakeContainers:
    def __init__(self, *, container=None, run_errors=None):
        self.container = container or _FakeContainer()
        self.run_errors = list(run_errors or [])
        self.run_calls = []

    def run(self, *args, **kwargs):
        self.run_calls.append((args, kwargs))
        if self.run_errors:
            raise self.run_errors.pop(0)
        return self.container

    def get(self, _name):
        return self.container


class _FakeClient:
    def __init__(self, *, containers=None):
        self.containers = containers or _FakeContainers()
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def _task() -> SimpleNamespace:
    return SimpleNamespace(
        instruction="fix demo",
        spec=SimpleNamespace(
            instance_id="demo__task-1",
            instance_image_key="sweb.eval.x86_64.demo:latest",
        ),
    )


def _docker_session(monkeypatch, *, container=None, run_errors=None):
    containers = _FakeContainers(container=container, run_errors=run_errors)
    client = _FakeClient(containers=containers)
    monkeypatch.setattr(docker_shell_module.docker, "from_env", lambda: client)
    return DockerShellSession(image="sweb.eval.x86_64.demo:latest"), containers, client


def test_swe_env_close_removes_container_but_keeps_shared_client(monkeypatch, tmp_path):
    container = _FakeContainer()
    docker_client = _FakeClient(containers=_FakeContainers(container=container))
    monkeypatch.setattr(docker_shell_module.docker, "from_env", lambda: docker_client)
    env = SweEnv(task=_task(), artifacts_dir=tmp_path)
    env._solve_env._started = True

    asyncio.run(env.close())

    assert container.remove_calls == [((), {"force": True, "v": True})]
    assert docker_client.close_calls == 0


def test_docker_shell_sessions_share_one_client_across_instances(monkeypatch):
    # Regression: one client per session retains ~12 daemon sockets each, so
    # 32 concurrent rollouts exceed macOS's default 256-fd limit (EMFILE).
    created = []

    def fake_from_env():
        client = _FakeClient()
        created.append(client)
        return client

    monkeypatch.setattr(docker_shell_module.docker, "from_env", fake_from_env)

    first = DockerShellSession(image="img-a")
    second = DockerShellSession(image="img-b")

    assert len(created) == 1
    assert first._docker_client is second._docker_client

    first._started = True
    asyncio.run(first.close())
    asyncio.run(second.close())

    assert created[0].close_calls == 0


def test_swe_env_run_lazily_starts_container_and_defaults_to_task_workdir(
    monkeypatch, tmp_path
):
    container = _FakeContainer(
        exec_result=SimpleNamespace(exit_code=0, output=(b"ok\n", b""))
    )
    containers = _FakeContainers(container=container)
    monkeypatch.setattr(
        docker_shell_module.docker,
        "from_env",
        lambda: _FakeClient(containers=containers),
    )
    env = SweEnv(task=_task(), artifacts_dir=tmp_path)

    result = _step_run(env, "pwd")

    assert result.raw_env_output.stdout == "ok\n"
    assert result.terminated is False
    assert len(containers.run_calls) == 1
    (cmd,), kwargs = container.exec_run_calls[-1]
    assert cmd[-3:] == ["bash", "-lc", "pwd"]
    assert kwargs["workdir"] == "/testbed"


def test_solve_env_start_retries_a_daemon_blip(monkeypatch):
    _no_sleep(monkeypatch)
    solve_env, containers, _client = _docker_session(
        monkeypatch, run_errors=[DockerException("error during connect: EOF")]
    )

    asyncio.run(solve_env.start())

    assert solve_env._started is True
    assert len(containers.run_calls) == 2
    args, kwargs = containers.run_calls[-1]
    assert args == ("sweb.eval.x86_64.demo:latest", ["sleep", "infinity"])
    assert kwargs["detach"] is True
    assert kwargs["platform"] == "linux/amd64"
    assert kwargs["network_mode"] == "none"
    assert kwargs["name"] == solve_env.container_name


def test_solve_env_start_gives_up_after_retry_budget(monkeypatch):
    _no_sleep(monkeypatch)
    solve_env, containers, _client = _docker_session(
        monkeypatch,
        run_errors=[
            DockerException("Cannot connect to the Docker daemon")
            for _ in range(docker_shell_module.DOCKER_INFRA_RETRY_BUDGET + 1)
        ],
    )

    with pytest.raises(DockerException, match="Cannot connect"):
        asyncio.run(solve_env.start())

    assert (
        len(containers.run_calls) == docker_shell_module.DOCKER_INFRA_RETRY_BUDGET + 1
    )
    assert solve_env._started is False


def test_solve_env_start_keeps_container_when_mtime_reset_times_out(monkeypatch):
    original_wait_for = docker_shell_module.asyncio.wait_for

    async def _time_out_mtime_reset(awaitable, timeout=None):
        if timeout == 30:
            awaitable.close()
            raise TimeoutError
        return await original_wait_for(awaitable, timeout=timeout)

    container = _FakeContainer()
    monkeypatch.setattr(docker_shell_module.asyncio, "wait_for", _time_out_mtime_reset)
    solve_env, _containers, _client = _docker_session(monkeypatch, container=container)

    asyncio.run(solve_env.start())

    assert solve_env._started is True
    assert container.remove_calls == []


def test_solve_env_exec_returns_demuxed_verdict(monkeypatch):
    container = _FakeContainer(
        exec_result=SimpleNamespace(
            exit_code=1,
            output=(b"3 failed, 5 passed", b"warn\n"),
        )
    )
    solve_env, _containers, _client = _docker_session(monkeypatch, container=container)

    result = asyncio.run(
        solve_env.run(
            command="pytest",
            cwd="/testbed",
            timeout=None,
        )
    )

    assert result.exit_code == 1
    assert result.stdout == "3 failed, 5 passed"
    assert result.stderr == "warn\n"
    (pin_cmd,), _pin_kwargs = container.exec_run_calls[0]
    assert pin_cmd == [
        "bash",
        "-lc",
        docker_shell_module.determinism.MTIME_RESET_COMMAND,
    ]
    (cmd,), kwargs = container.exec_run_calls[-1]
    assert cmd == ["bash", "-lc", "pytest"]
    assert kwargs["workdir"] == "/testbed"
    assert kwargs["demux"] is True


def _write_official_artifacts(
    root: Path,
    *,
    run_id: str,
    instance_id: str,
    model_name: str,
    patch: str,
    resolved: bool = True,
) -> None:
    log_dir = root / run_id / model_name / instance_id
    log_dir.mkdir(parents=True)
    report = {
        instance_id: {
            "patch_exists": True,
            "patch_successfully_applied": True,
            "resolved": resolved,
            "tests_status": _report(
                f2p_ok=["test_fix"] if resolved else [],
                f2p_bad=[] if resolved else ["test_fix"],
                p2p_ok=["test_keep"],
                p2p_bad=[],
            ),
        }
    }
    (log_dir / OFFICIAL_ARTIFACTS.REPORT).write_text(json.dumps(report))
    (log_dir / OFFICIAL_ARTIFACTS.TEST_OUTPUT).write_text("official verifier output\n")
    (log_dir / RUN_INSTANCE_LOG).write_text("official runner log\n")
    (log_dir / OFFICIAL_ARTIFACTS.PATCH).write_text(patch)


def _official_dir(artifacts_dir: Path) -> Path:
    return artifacts_dir / OFFICIAL_ARTIFACTS.ROOT_DIR


def _nested_official_dir(artifacts_dir: Path) -> Path:
    return (
        _official_dir(artifacts_dir)
        / OFFICIAL_ARTIFACTS.INSTANCE_PARENT_DIR
        / "demo__task-1"
    )


def _started_env(
    monkeypatch: pytest.MonkeyPatch,
    artifacts_dir: Path,
    *,
    patch: str,
) -> tuple[SweEnv, _FakeContainer, _FakeClient]:
    # grader=None: grades live, no cache I/O. Cache wrapping is tested separately.
    container = _FakeContainer(
        exec_result=SimpleNamespace(exit_code=0, output=(patch.encode(), b""))
    )
    shell_client = _FakeClient(containers=_FakeContainers(container=container))
    verify_client = _FakeClient()
    docker_clients = iter([shell_client, verify_client])
    monkeypatch.setattr(
        docker_shell_module.docker, "from_env", lambda: next(docker_clients)
    )
    env = SweEnv(task=_task(), artifacts_dir=artifacts_dir)
    env._solve_env._started = True
    return env, container, verify_client


def _verify_result(*, stdout: str, passed: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        reward=0.0,
        stdout=stdout,
        completed=True,
        passed=passed,
        error=None,
        fail_to_pass_passed=0,
        pass_to_pass_failed=0,
    )


def _official_log_dir(kwargs: dict) -> Path:
    return (
        _swebench_run_evaluation.RUN_EVALUATION_LOG_DIR
        / kwargs["run_id"]
        / kwargs["pred"][swe_verify.KEY_MODEL]
        / kwargs["test_spec"].instance_id
    )


def _grading_run_instance(*, resolved: bool = True, pre_assert=None):
    """A fake swebench run_instance that writes a resolved/unresolved official
    artifact tree and records its call kwargs. Returns (fake, calls)."""
    calls = []

    def fake_run_instance(**kwargs):
        calls.append(kwargs)
        if pre_assert is not None:
            pre_assert()
        _write_official_artifacts(
            _swebench_run_evaluation.RUN_EVALUATION_LOG_DIR,
            run_id=kwargs["run_id"],
            instance_id=kwargs["test_spec"].instance_id,
            model_name=kwargs["pred"][swe_verify.KEY_MODEL],
            patch=kwargs["pred"][swe_verify.KEY_PREDICTION],
            resolved=resolved,
        )
        return {"completed": True, "resolved": resolved}

    return fake_run_instance, calls


def _assert_graded_artifacts(
    official_dir: Path, *, patch: str, resolved: bool = True
) -> None:
    assert (official_dir / OFFICIAL_ARTIFACTS.REPORT).exists()
    assert (official_dir / OFFICIAL_ARTIFACTS.PATCH).read_text() == patch
    assert (
        official_dir / OFFICIAL_ARTIFACTS.TEST_OUTPUT
    ).read_text() == "official verifier output\n"
    assert not (official_dir / "diagnostics.json").exists()
    assert not (official_dir.parent / "model.patch").exists()
    assert (
        json.loads((official_dir / OFFICIAL_ARTIFACTS.REPORT).read_text())[
            "demo__task-1"
        ]["resolved"]
        is resolved
    )


def test_patch_extraction_failure_grades_empty_patch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env, _, _ = _started_env(monkeypatch, tmp_path, patch="")

    async def failed_patch(*, command, cwd, timeout, lossless=False):
        del command, cwd, timeout, lossless
        return swe_module.RawEnvOutput(exit_code=128, stderr="bad git state")

    async def verify(*, spec, patch, rollout_artifact_dir):
        del spec, rollout_artifact_dir
        assert patch == ""
        return _verify_result(stdout="rejected")

    monkeypatch.setattr(env._solve_env, "run", failed_patch)
    monkeypatch.setattr(swe_module.swebench_verify, "verify", verify)

    outcome = asyncio.run(env.verify())

    assert outcome.verdict == VerifyVerdict(completed=True, passed=False, error=None)
    assert outcome.info[MODEL_PATCH_INFO_KEY] == ""


def test_oversized_patch_reaches_grader_intact(monkeypatch, tmp_path):
    patch = "diff --git a/demo.py b/demo.py\n" + "x" * (
        docker_shell_module.MAX_CAPTURED_STREAM_BYTES + 1
    )
    env, _, _ = _started_env(monkeypatch, tmp_path, patch=patch)

    async def verify(*, spec, patch, rollout_artifact_dir):
        del spec, rollout_artifact_dir
        assert patch == expected_patch
        return _verify_result(stdout="graded")

    expected_patch = patch
    monkeypatch.setattr(swe_module.swebench_verify, "verify", verify)

    outcome = asyncio.run(env.verify())

    assert outcome.info[MODEL_PATCH_INFO_KEY] == expected_patch


def test_submit_scores_official_report(monkeypatch, tmp_path):
    patch = "diff --git a/demo.py b/demo.py\n--- a/demo.py\n+++ b/demo.py\n"
    env, container, verify_client = _started_env(monkeypatch, tmp_path, patch=patch)
    previous_log_root = _swebench_run_evaluation.RUN_EVALUATION_LOG_DIR
    official_dir = _official_dir(tmp_path)
    nested_dir = _nested_official_dir(tmp_path)
    official_dir.mkdir(parents=True)
    (official_dir / OFFICIAL_ARTIFACTS.REPORT).write_text("stale report\n")

    def assert_stale_report_cleared():
        assert not (official_dir / OFFICIAL_ARTIFACTS.REPORT).exists()

    fake_run_instance, calls = _grading_run_instance(
        pre_assert=assert_stale_report_cleared
    )
    monkeypatch.setattr(_swebench_run_evaluation, "run_instance", fake_run_instance)

    result = _step_submit(env)

    assert result.terminated is True
    assert result.truncated is False
    assert result.reward == 1.0
    assert result.info[MODEL_PATCH_INFO_KEY] == patch
    assert result.info["instance_id"] == "demo__task-1"
    assert result.metrics == {"fail_to_pass_passed": 1, "pass_to_pass_failed": 0}
    assert result.verdict == VerifyVerdict(completed=True, passed=True, error=None)
    assert result.raw_env_output.stdout == "official verifier output\n"
    assert calls[0]["pred"][swe_verify.KEY_PREDICTION] == patch
    assert calls[0]["client"] is verify_client
    assert calls[0]["run_id"] == OFFICIAL_ARTIFACTS.RUN_ID
    assert calls[0]["rm_image"] is False
    assert calls[0]["force_rebuild"] is False
    assert container.remove_calls == [((), {"force": True, "v": True})]
    assert env._solve_env._started is False
    (patch_cmd,), patch_kwargs = container.exec_run_calls[-1]
    assert patch_cmd[-3:] == ["bash", "-lc", _MODEL_PATCH_COMMAND]
    assert patch_kwargs["workdir"] == "/testbed"
    _assert_graded_artifacts(official_dir, patch=patch)
    assert (official_dir / RUN_INSTANCE_LOG).read_text() == "official runner log\n"
    assert not nested_dir.exists()
    assert _swebench_run_evaluation.RUN_EVALUATION_LOG_DIR == previous_log_root


def test_submit_uses_official_resolved_for_reward(monkeypatch, tmp_path):
    patch = "diff --git a/demo.py b/demo.py\n--- a/demo.py\n+++ b/demo.py\n"
    env, _container, _docker_client = _started_env(monkeypatch, tmp_path, patch=patch)

    fake_run_instance, _calls = _grading_run_instance(resolved=False)
    monkeypatch.setattr(_swebench_run_evaluation, "run_instance", fake_run_instance)

    result = _step_submit(env)

    assert result.metrics == {"fail_to_pass_passed": 0, "pass_to_pass_failed": 0}
    assert result.verdict == VerifyVerdict(completed=True, passed=False, error=None)
    assert result.reward == 0.0
    assert (
        json.loads((_official_dir(tmp_path) / OFFICIAL_ARTIFACTS.REPORT).read_text())[
            "demo__task-1"
        ]["resolved"]
        is False
    )
    assert not (_official_dir(tmp_path) / "diagnostics.json").exists()


def test_eager_env_grades_live_without_cache_io(monkeypatch, tmp_path):
    patch = "diff --git a/demo.py b/demo.py\n--- a/demo.py\n+++ b/demo.py\n"
    env, _container, _docker_client = _started_env(monkeypatch, tmp_path, patch=patch)
    fake_run_instance, calls = _grading_run_instance()
    monkeypatch.setattr(_swebench_run_evaluation, "run_instance", fake_run_instance)
    from src.plugins.caching import store as cache

    cache_get = AsyncMock()
    cache_put = AsyncMock()
    monkeypatch.setattr(cache, "get", cache_get)
    monkeypatch.setattr(cache, "put", cache_put)

    result = _step_submit(env)

    assert result.reward == 1.0
    assert len(calls) == 1
    cache_get.assert_not_awaited()
    cache_put.assert_not_awaited()


def test_verify_applies_injected_wrapper(monkeypatch, tmp_path):
    env, _, _ = _started_env(monkeypatch, tmp_path, patch="mypatch")
    calls: list = []

    async def fake_grader(*, spec, patch, rollout_artifact_dir):
        calls.append((spec.instance_id, patch))
        return _verify_result(stdout="wrapped")

    env._verify_wrapper = lambda grader: fake_grader
    outcome = asyncio.run(env.verify())

    assert calls == [("demo__task-1", "mypatch")]
    assert outcome.output.stdout == "wrapped"


@pytest.mark.parametrize(
    ("official_result", "artifact", "match"),
    [
        pytest.param(
            {"completed": True, "resolved": False},
            None,
            "artifact dir missing",
            id="missing-artifact-directory",
        ),
        pytest.param(
            {"completed": "yes", "resolved": False},
            None,
            "completed flag",
            id="non-boolean-completed",
        ),
        pytest.param(
            {"completed": True, "resolved": False},
            (False, False),
            "test output missing",
            id="missing-test-output",
        ),
        pytest.param(
            {"completed": True, "resolved": False},
            ("no", True),
            "resolved field",
            id="non-boolean-resolved",
        ),
    ],
)
def test_submit_rejects_corrupt_official_results(
    monkeypatch, tmp_path, official_result, artifact, match
):
    env, _container, _docker_client = _started_env(
        monkeypatch,
        tmp_path,
        patch="diff --git a/x b/x\n",
    )

    def fake_run_instance(**kwargs):
        if artifact is not None:
            resolved, include_output = artifact
            log_dir = _official_log_dir(kwargs)
            log_dir.mkdir(parents=True)
            (log_dir / OFFICIAL_ARTIFACTS.REPORT).write_text(
                json.dumps({kwargs["test_spec"].instance_id: {"resolved": resolved}})
            )
            if include_output:
                (log_dir / OFFICIAL_ARTIFACTS.TEST_OUTPUT).write_text("output\n")
        return official_result

    monkeypatch.setattr(_swebench_run_evaluation, "run_instance", fake_run_instance)
    with pytest.raises(swe_verify.VerifierCorruptError, match=match):
        _step_submit(env)


def test_submit_empty_patch_uses_official_incomplete_result(monkeypatch, tmp_path):
    env, _container, _docker_client = _started_env(monkeypatch, tmp_path, patch="")

    def fake_run_instance(**kwargs):
        log_dir = _official_log_dir(kwargs)
        log_dir.mkdir(parents=True)
        (log_dir / OFFICIAL_ARTIFACTS.PATCH).write_text(
            kwargs["pred"][swe_verify.KEY_PREDICTION]
        )
        (log_dir / RUN_INSTANCE_LOG).write_text("empty patch failed to apply\n")
        return {"completed": False, "resolved": False}

    monkeypatch.setattr(_swebench_run_evaluation, "run_instance", fake_run_instance)
    result = _step_submit(env)

    assert result.reward == 0.0
    assert result.verdict is not None
    assert result.verdict.completed is False
    assert result.verdict.error is not None
    assert (_official_dir(tmp_path) / OFFICIAL_ARTIFACTS.PATCH).read_text() == ""
    assert (_official_dir(tmp_path) / RUN_INSTANCE_LOG).exists()
    assert not (_official_dir(tmp_path) / OFFICIAL_ARTIFACTS.REPORT).exists()
    assert not (tmp_path / "model.patch").exists()
    assert not _nested_official_dir(tmp_path).exists()
