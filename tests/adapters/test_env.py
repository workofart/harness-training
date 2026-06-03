from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from harbor.environments.base import ExecResult
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import TaskOS, VerifierEnvironmentMode
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths

from src.adapters.env import Harbor, HarborConfig, TaskDirectoryResolver


def _write_minimal_task(
    task_dir: Path,
    *,
    cpus: int | None = 1,
    docker_image: str | None = "example/task:latest",
    verifier_env: dict[str, str] | None = None,
    verifier_environment_mode: str | None = None,
    verifier_docker_image: str | None = None,
) -> None:
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "solution").mkdir()
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("solve it\n")
    task_toml = [
        'version = "1.0"',
        "",
        "[environment]",
    ]
    if docker_image is not None:
        task_toml.append(f'docker_image = "{docker_image}"')
    if cpus is not None:
        task_toml.append(f"cpus = {cpus}")
    if verifier_environment_mode is not None or verifier_docker_image is not None:
        task_toml.extend(["", "[verifier]"])
    if verifier_environment_mode is not None:
        task_toml.append(f'environment_mode = "{verifier_environment_mode}"')
    if verifier_env:
        task_toml.extend(["", "[verifier.env]"])
        task_toml.extend(f'{key} = "{value}"' for key, value in verifier_env.items())
    if verifier_docker_image is not None:
        task_toml.extend(
            [
                "",
                "[verifier.environment]",
                f'docker_image = "{verifier_docker_image}"',
            ]
        )
    (task_dir / "task.toml").write_text("\n".join(task_toml) + "\n")
    (task_dir / "environment" / "Dockerfile").write_text("FROM python:3.13-slim\n")
    (task_dir / "solution" / "solve.sh").write_text("#!/bin/bash\n")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\n")


def test_task_directory_resolver_prefers_local_task_override(tmp_path: Path) -> None:
    overrides_dir = tmp_path / "overrides"
    override_task_dir = overrides_dir / "task-a"
    _write_minimal_task(override_task_dir)
    task_dirs = asyncio.run(
        TaskDirectoryResolver(
            HarborConfig(
                experiments_dir=tmp_path / "experiments",
                task_overrides_dir=overrides_dir,
            )
        ).resolve(["task-a"])
    )

    assert task_dirs == {"task-a": override_task_dir.resolve()}


def test_task_directory_resolver_falls_back_to_registry_download_when_override_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from harbor.registry.client import factory as registry_factory
    from harbor.tasks import client as tasks_client

    downloaded_task_dir = tmp_path / "downloaded-task"

    class FakeTaskId:
        def __init__(self, name: str) -> None:
            self.name = name

        def get_name(self) -> str:
            return self.name

    source_task_id = FakeTaskId("task-a")

    class FakeRegistryClient:
        async def get_dataset_metadata(self, dataset_name):
            assert dataset_name == "terminal-bench"
            return SimpleNamespace(task_ids=[source_task_id])

        async def download_dataset(self, dataset_name):
            raise AssertionError(f"unexpected full dataset download: {dataset_name}")

    class FakeTaskClient:
        async def download_tasks(self, task_ids):
            assert task_ids == [source_task_id]
            return SimpleNamespace(paths=[downloaded_task_dir])

    monkeypatch.setattr(
        registry_factory.RegistryClientFactory,
        "create",
        staticmethod(lambda: FakeRegistryClient()),
    )
    monkeypatch.setattr(tasks_client, "TaskClient", FakeTaskClient)

    task_dirs = asyncio.run(
        TaskDirectoryResolver(
            HarborConfig(
                experiments_dir=tmp_path / "experiments",
                task_overrides_dir=tmp_path / "overrides",
            )
        ).resolve(["task-a"])
    )

    assert task_dirs == {"task-a": downloaded_task_dir}


def test_task_directory_resolver_uses_dataset_version_for_registry_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from harbor.registry.client import factory as registry_factory
    from harbor.tasks import client as tasks_client

    downloaded_task_dir = tmp_path / "downloaded-task"

    class FakeTaskId:
        def __init__(self, name: str) -> None:
            self.name = name

        def get_name(self) -> str:
            return self.name

    class FakeRegistryClient:
        async def get_dataset_metadata(self, name):
            assert name == "terminal-bench@v1"
            return SimpleNamespace(task_ids=[FakeTaskId("task-a")])

    class FakeTaskClient:
        async def download_tasks(self, task_ids):
            assert [task_id.get_name() for task_id in task_ids] == ["task-a"]
            return SimpleNamespace(paths=[downloaded_task_dir])

    monkeypatch.setattr(
        registry_factory.RegistryClientFactory,
        "create",
        staticmethod(lambda: FakeRegistryClient()),
    )
    monkeypatch.setattr(tasks_client, "TaskClient", FakeTaskClient)

    task_dirs = asyncio.run(
        TaskDirectoryResolver(
            HarborConfig(
                experiments_dir=tmp_path / "experiments",
                task_overrides_dir=tmp_path / "overrides",
                dataset_version="v1",
            )
        ).resolve(["task-a"])
    )

    assert task_dirs == {"task-a": downloaded_task_dir}


def test_task_directory_resolver_rejects_invalid_local_task_override(
    tmp_path: Path,
) -> None:
    invalid_override_dir = tmp_path / "overrides" / "task-a"
    invalid_override_dir.mkdir(parents=True)
    (invalid_override_dir / "instruction.md").write_text("incomplete\n")

    with pytest.raises(RuntimeError, match="local task override is invalid"):
        asyncio.run(
            TaskDirectoryResolver(
                HarborConfig(
                    experiments_dir=tmp_path / "experiments",
                    task_overrides_dir=tmp_path / "overrides",
                )
            ).resolve(["task-a"])
        )


def _stub_harbor_with_inner_exec(
    tmp_path: Path, *, inner_exec_side_effect: Exception
) -> Harbor:
    """Build a Harbor whose session.environment.exec raises a chosen exception.

    Bypasses real Harbor.reset (which requires docker) by injecting a
    mock session directly. Sufficient for testing the exec error-handling
    contract in isolation.
    """
    config = HarborConfig(experiments_dir=tmp_path / "experiments")
    harbor = Harbor(config, task_name="task-a", task_dir=tmp_path / "task")
    fake_session = MagicMock()
    fake_session.environment.exec = AsyncMock(side_effect=inner_exec_side_effect)
    agent_dir = tmp_path / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    fake_session.trial_paths.agent_dir = agent_dir
    harbor._session = fake_session
    return harbor


def test_harbor_exec_converts_timeout_runtime_error_to_failed_raw_state(
    tmp_path: Path,
) -> None:
    # Harbor's docker/gke backends raise RuntimeError("Command timed out
    # after N seconds") on per-command timeout. Harbor.exec must absorb
    # that into a failed RawState(rc=124) so the harness loop sees a
    # recoverable timeout observation instead of a trial-fatal exception.
    harbor = _stub_harbor_with_inner_exec(
        tmp_path,
        inner_exec_side_effect=RuntimeError("Command timed out after 15 seconds"),
    )

    result = asyncio.run(harbor.exec(command="apt update", timeout_sec=15))

    assert result.return_code == 124
    assert result.passed is False
    assert result.stderr is not None
    assert "timed out after 15 seconds" in result.stderr


def test_harbor_exec_converts_value_error_to_failed_raw_state(tmp_path: Path) -> None:
    # An agent action whose content cannot be marshalled into a container
    # command -- e.g. a write_file with an embedded NUL, which the subprocess
    # layer rejects with ValueError("embedded null byte") -- must become a
    # recoverable failed observation, not a trial-fatal crash.
    harbor = _stub_harbor_with_inner_exec(
        tmp_path,
        inner_exec_side_effect=ValueError("embedded null byte"),
    )

    result = asyncio.run(harbor.exec(command="printf %s 'x\x00y' > f"))

    assert result.passed is False
    assert result.return_code != 0
    assert result.stderr is not None
    assert "embedded null byte" in result.stderr


def test_harbor_exec_propagates_non_timeout_runtime_error(tmp_path: Path) -> None:
    # Only timeout-shaped RuntimeError is absorbed. Genuine
    # container/transport failures must surface as trial-fatal so the
    # supervisor distinguishes recoverable from unrecoverable.
    harbor = _stub_harbor_with_inner_exec(
        tmp_path,
        inner_exec_side_effect=RuntimeError("docker socket disconnected"),
    )

    with pytest.raises(RuntimeError, match="docker socket disconnected"):
        asyncio.run(harbor.exec(command="ls"))


class RecordingEnvironment:
    is_mounted = True
    os = TaskOS.LINUX
    capabilities = SimpleNamespace(mounted=True)

    def __init__(self, reward_path: Path | None = None) -> None:
        self.reward_path = reward_path
        self.exec_calls: list[dict[str, object]] = []

    async def upload_file(self, source_path: Path | str, target_path: str):
        del source_path, target_path

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        del source_dir, target_dir

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        self.exec_calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "timeout_sec": timeout_sec,
                "user": user,
            }
        )
        if self.reward_path is not None and "/tests/test.sh" in command:
            self.reward_path.write_text("1.0")
        return ExecResult(return_code=0, stdout="", stderr="")


def _attach_recording_harbor(
    tmp_path: Path,
    *,
    cpus: int | None,
    verifier_env: dict[str, str] | None = None,
) -> tuple[Harbor, RecordingEnvironment]:
    task_dir = tmp_path / "task"
    _write_minimal_task(
        task_dir,
        cpus=cpus,
        verifier_env=verifier_env,
        verifier_environment_mode="shared",
    )
    trial_paths = TrialPaths(trial_dir=tmp_path / "trial")
    trial_paths.mkdir()
    task = Task(task_dir=task_dir)
    environment = RecordingEnvironment(reward_path=trial_paths.reward_text_path)
    harbor = Harbor(
        HarborConfig(experiments_dir=tmp_path / "experiments"),
        task_name="task-a",
        task_dir=task_dir,
    )
    harbor._attach_session(
        task=task,
        trial_paths=trial_paths,
        harbor_environment=environment,
    )
    return harbor, environment


def test_harbor_exec_applies_declared_cpu_resource_caps(tmp_path: Path) -> None:
    harbor, environment = _attach_recording_harbor(tmp_path, cpus=2)

    asyncio.run(harbor.exec(command="python -c pass"))

    env = environment.exec_calls[-1]["env"]
    assert env["OPENBLAS_NUM_THREADS"] == "2"
    assert env["OMP_NUM_THREADS"] == "2"
    assert env["RAYON_NUM_THREADS"] == "2"
    assert env["CMAKE_BUILD_PARALLEL_LEVEL"] == "2"
    assert env["MAKEFLAGS"] == "-j2"
    assert env["TOKENIZERS_PARALLELISM"] == "false"


def test_harbor_exec_does_not_invent_cpu_caps_when_task_cpu_is_unspecified(
    tmp_path: Path,
) -> None:
    harbor, environment = _attach_recording_harbor(tmp_path, cpus=None)

    asyncio.run(harbor.exec(command="python -c pass"))

    env = environment.exec_calls[-1]["env"]
    assert env == {"TOKENIZERS_PARALLELISM": "false"}


def test_harbor_verifier_applies_caps_while_preserving_verifier_env(
    tmp_path: Path,
) -> None:
    harbor, environment = _attach_recording_harbor(
        tmp_path,
        cpus=3,
        verifier_env={"OMP_NUM_THREADS": "7", "CUSTOM_ENV": "set"},
    )

    asyncio.run(harbor.session.verifier.verify())

    chmod_env = environment.exec_calls[0]["env"]
    verifier_env = environment.exec_calls[1]["env"]
    assert environment.exec_calls[0]["user"] == "root"
    assert chmod_env["OMP_NUM_THREADS"] == "3"
    assert verifier_env["OMP_NUM_THREADS"] == "7"
    assert verifier_env["OPENBLAS_NUM_THREADS"] == "3"
    assert verifier_env["CUSTOM_ENV"] == "set"


def test_harbor_close_uses_raw_environment_for_lifecycle_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harbor, environment = _attach_recording_harbor(tmp_path, cpus=2)
    stopped = []

    async def fake_stop_environment(target):
        stopped.append(target)

    monkeypatch.setattr(harbor, "_stop_environment", fake_stop_environment)

    asyncio.run(harbor.close())

    assert stopped == [environment]


def _stub_harbor_with_recording_exec(
    tmp_path: Path,
    *,
    order: list[str],
    exec_semaphore: asyncio.Semaphore | None,
) -> Harbor:
    """Build a Harbor whose inner exec records enter/exit order around an await
    boundary, so concurrent `harbor.exec` calls reveal whether they overlap."""

    async def fake_exec(*, command, cwd=None, env=None, timeout_sec=None):
        del cwd, env, timeout_sec
        order.append(f"enter:{command}")
        await asyncio.sleep(0.01)
        order.append(f"exit:{command}")
        return ExecResult(return_code=0, stdout="", stderr="")

    config = HarborConfig(experiments_dir=tmp_path / "experiments")
    harbor = Harbor(
        config,
        task_name="task-a",
        task_dir=tmp_path / "task",
        exec_semaphore=exec_semaphore,
    )
    fake_session = MagicMock()
    fake_session.environment.exec = fake_exec
    agent_dir = tmp_path / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    fake_session.trial_paths.agent_dir = agent_dir
    harbor._session = fake_session
    return harbor


def test_harbor_exec_serializes_under_shared_semaphore(tmp_path: Path) -> None:
    # The run-scoped semaphore bounds heavyweight harness actions. A
    # size-1 gate must forbid two default Harbor.exec calls from overlapping
    # inside the container — the signal-preserving guarantee that lets
    # trial-admission (outer max_trial_concurrency) run ahead of host-CPU
    # capacity (inner max_heavy_action_concurrency) without oversubscribing cores.
    order: list[str] = []
    harbor = _stub_harbor_with_recording_exec(
        tmp_path, order=order, exec_semaphore=asyncio.Semaphore(1)
    )

    async def go():
        await asyncio.gather(harbor.exec(command="a"), harbor.exec(command="b"))

    asyncio.run(go())

    # Each command fully exits before the next enters — no interleaving.
    assert order in (
        ["enter:a", "exit:a", "enter:b", "exit:b"],
        ["enter:b", "exit:b", "enter:a", "exit:a"],
    )


def test_harbor_exec_overlaps_without_semaphore(tmp_path: Path) -> None:
    # Negative control: with no gate (exec_semaphore=None, the single-trial /
    # test default), two concurrent execs overlap — proving the serialization
    # above is caused by the semaphore, not by something incidental in exec.
    order: list[str] = []
    harbor = _stub_harbor_with_recording_exec(
        tmp_path, order=order, exec_semaphore=None
    )

    async def go():
        await asyncio.gather(harbor.exec(command="a"), harbor.exec(command="b"))

    asyncio.run(go())

    # Both enter before either exits.
    assert order[:2] == ["enter:a", "enter:b"]


def test_harbor_light_exec_bypasses_shared_heavy_semaphore(tmp_path: Path) -> None:
    order: list[str] = []
    harbor = _stub_harbor_with_recording_exec(
        tmp_path, order=order, exec_semaphore=asyncio.Semaphore(1)
    )

    async def go():
        heavy = asyncio.create_task(harbor.exec(command="heavy"))
        await asyncio.sleep(0)
        light = asyncio.create_task(harbor.exec(command="light", workload="light"))
        await asyncio.gather(heavy, light)

    asyncio.run(go())

    assert order[:2] == ["enter:heavy", "enter:light"]


def test_harbor_reset_serializes_startup_under_shared_semaphore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from harbor.environments.factory import EnvironmentFactory

    order: list[str] = []

    class FakeEnvironment:
        def __init__(self, name: str) -> None:
            self.name = name

        async def start(self, *, force_build: bool) -> None:
            del force_build
            order.append(f"enter:{self.name}")
            await asyncio.sleep(0.01)
            order.append(f"exit:{self.name}")

        async def exec(
            self, *, command, cwd=None, env=None, timeout_sec=None, user=None
        ):
            del cwd, env, timeout_sec, user
            assert command == "pwd"
            return ExecResult(return_code=0, stdout="/app\n", stderr="")

        async def stop(self, *, delete: bool):
            del delete

    def fake_create_environment_from_config(**kwargs):
        return FakeEnvironment(kwargs["environment_name"])

    monkeypatch.setattr(
        EnvironmentFactory,
        "create_environment_from_config",
        staticmethod(fake_create_environment_from_config),
    )
    task_a = tmp_path / "task-a"
    task_b = tmp_path / "task-b"
    _write_minimal_task(task_a)
    _write_minimal_task(task_b)
    config = HarborConfig(experiments_dir=tmp_path / "experiments")
    exec_semaphore = asyncio.Semaphore(1)
    harbor_a = Harbor(
        config,
        task_name="task-a",
        task_dir=task_a,
        exec_semaphore=exec_semaphore,
    )
    harbor_b = Harbor(
        config,
        task_name="task-b",
        task_dir=task_b,
        exec_semaphore=exec_semaphore,
    )

    async def go():
        await asyncio.gather(harbor_a.reset(), harbor_b.reset())

    asyncio.run(go())

    assert order in (
        ["enter:task-a", "exit:task-a", "enter:task-b", "exit:task-b"],
        ["enter:task-b", "exit:task-b", "enter:task-a", "exit:task-a"],
    )


def test_harbor_reset_passes_trial_log_mounts_to_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from harbor.environments.factory import EnvironmentFactory

    captured_mounts = []

    class FakeEnvironment:
        async def start(self, *, force_build: bool) -> None:
            del force_build

        async def exec(
            self, *, command, cwd=None, env=None, timeout_sec=None, user=None
        ):
            del cwd, env, timeout_sec, user
            assert command == "pwd"
            return ExecResult(return_code=0, stdout="/app\n", stderr="")

        async def stop(self, *, delete: bool):
            del delete

    def fake_create_environment_from_config(**kwargs):
        captured_mounts.extend(kwargs["mounts"])
        return FakeEnvironment()

    monkeypatch.setattr(
        EnvironmentFactory,
        "create_environment_from_config",
        staticmethod(fake_create_environment_from_config),
    )
    task_dir = tmp_path / "task-a"
    _write_minimal_task(task_dir)
    harbor = Harbor(
        HarborConfig(experiments_dir=tmp_path / "experiments"),
        task_name="task-a",
        task_dir=task_dir,
    )

    asyncio.run(harbor.reset())
    asyncio.run(harbor.close())

    mounts_by_target = {mount["target"]: mount for mount in captured_mounts}
    assert mounts_by_target["/logs/agent"]["source"].endswith("/agent")
    assert mounts_by_target["/logs/verifier"]["source"].endswith("/verifier")
    assert mounts_by_target["/logs/artifacts"]["source"].endswith("/artifacts")


class LifecycleRecordingEnvironment:
    os = TaskOS.LINUX
    capabilities = SimpleNamespace(mounted=True)

    def __init__(self, *, trial_paths: TrialPaths, label: str) -> None:
        self.trial_paths = trial_paths
        self.label = label
        self.default_user = None
        self.start_calls: list[bool] = []
        self.stop_calls: list[bool] = []
        self.exec_calls: list[dict[str, object]] = []
        self.empty_dir_calls: list[dict[str, object]] = []
        self.upload_dir_calls: list[tuple[Path | str, str]] = []
        self.download_dir_calls: list[tuple[str, Path | str]] = []
        self.download_file_calls: list[tuple[str, Path | str]] = []
        self.is_dir_values: dict[str, bool] = {}

    @contextlib.contextmanager
    def with_default_user(self, user):
        previous = self.default_user
        self.default_user = user
        try:
            yield
        finally:
            self.default_user = previous

    async def start(self, *, force_build: bool) -> None:
        self.start_calls.append(force_build)

    async def stop(self, *, delete: bool):
        self.stop_calls.append(delete)

    async def upload_file(self, source_path: Path | str, target_path: str):
        del source_path, target_path

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        self.upload_dir_calls.append((source_dir, target_dir))

    async def download_dir(self, source_dir: str, target_dir: Path | str):
        self.download_dir_calls.append((source_dir, target_dir))
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        (Path(target_dir) / "workspace-marker.txt").write_text("downloaded\n")

    async def download_file(self, source_path: str, target_path: Path | str):
        self.download_file_calls.append((source_path, target_path))
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        Path(target_path).write_text("downloaded\n")

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        del user
        return self.is_dir_values.get(path, True)

    async def empty_dirs(self, dirs, *, chmod: bool = True):
        self.empty_dir_calls.append({"dirs": dirs, "chmod": chmod})
        return ExecResult(return_code=0, stdout="", stderr="")

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        effective_user = user if user is not None else self.default_user
        self.exec_calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "timeout_sec": timeout_sec,
                "user": effective_user,
            }
        )
        if command == "pwd":
            return ExecResult(return_code=0, stdout="/app\n", stderr="")
        if "/tests/test.sh" in command:
            self.trial_paths.reward_text_path.write_text("1.0")
            self.trial_paths.test_stdout_path.write_text(
                f"{self.label} verifier stdout\n"
            )
        return ExecResult(return_code=0, stdout="", stderr="")


def _patch_lifecycle_environment_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[LifecycleRecordingEnvironment], list[dict[str, object]]]:
    from harbor.environments.factory import EnvironmentFactory

    created_envs: list[LifecycleRecordingEnvironment] = []
    create_calls: list[dict[str, object]] = []

    def fake_create_environment_from_config(**kwargs):
        label = "agent" if not created_envs else "verifier"
        env = LifecycleRecordingEnvironment(
            trial_paths=kwargs["trial_paths"],
            label=label,
        )
        created_envs.append(env)
        create_calls.append(kwargs)
        return env

    monkeypatch.setattr(
        EnvironmentFactory,
        "create_environment_from_config",
        staticmethod(fake_create_environment_from_config),
    )
    return created_envs, create_calls


def test_harbor_verify_runs_separate_verifier_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created_envs, create_calls = _patch_lifecycle_environment_factory(monkeypatch)
    task_dir = tmp_path / "task-a"
    _write_minimal_task(
        task_dir,
        verifier_docker_image="example/verifier:latest",
    )
    harbor = Harbor(
        HarborConfig(experiments_dir=tmp_path / "experiments"),
        task_name="task-a",
        task_dir=task_dir,
    )

    async def fail_docker_cli(
        args,
        *,
        failure_context="Docker command failed",
    ):
        del failure_context
        raise AssertionError(f"prebuilt verifier image should not build: {args}")

    monkeypatch.setattr(harbor, "_run_docker_cli", fail_docker_cli)

    asyncio.run(harbor.reset())
    result = asyncio.run(harbor.verify())

    assert result.passed is True
    assert result.reward == 1.0
    assert result.stdout == "verifier verifier stdout\n"
    assert len(created_envs) == 2
    agent_env, verifier_env = created_envs
    agent_call, verifier_call = create_calls
    assert agent_call["environment_dir"] == task_dir / "environment"
    assert verifier_call["environment_dir"] == task_dir / "tests"
    assert agent_call["task_env_config"].docker_image == "example/task:latest"
    assert verifier_call["task_env_config"].docker_image == "example/verifier:latest"
    assert verifier_call["session_id"] != agent_call["session_id"]
    assert "__verifier" in verifier_call["session_id"]
    assert verifier_call["config"].extra_docker_compose == []
    verifier_mounts_by_target = {
        mount["target"]: mount for mount in verifier_call["mounts"]
    }
    assert set(verifier_mounts_by_target) == {"/logs/verifier"}
    assert verifier_mounts_by_target["/logs/verifier"]["source"].endswith("/verifier")
    assert not any(
        source == task_dir / "tests" for source, _ in verifier_env.upload_dir_calls
    )
    assert verifier_env.start_calls == [False]
    assert verifier_env.stop_calls == [True]
    assert agent_env.stop_calls == []

    asyncio.run(harbor.close())
    assert agent_env.stop_calls == [True]


def test_harbor_verify_caches_dockerfile_verifier_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created_envs, create_calls = _patch_lifecycle_environment_factory(monkeypatch)
    task_dir = tmp_path / "task-a"
    _write_minimal_task(
        task_dir,
        docker_image=None,
        verifier_environment_mode="separate",
    )
    (task_dir / "tests" / "Dockerfile").write_text(
        "FROM python:3.13-slim\nCOPY test.sh /tests/test.sh\n"
    )
    harbor = Harbor(
        HarborConfig(experiments_dir=tmp_path / "experiments"),
        task_name="task-a",
        task_dir=task_dir,
    )
    existing_images: set[str] = set()
    docker_calls: list[list[str]] = []

    async def fake_docker_cli(
        args: list[str],
        *,
        failure_context: str = "Docker command failed",
    ) -> ExecResult:
        del failure_context
        docker_calls.append(args)
        if args[:3] == ["image", "inspect", "--format"]:
            image_name = args[-1]
            if image_name not in existing_images:
                raise RuntimeError(f"No such image: {image_name}")
            return ExecResult(return_code=0, stdout="sha256:cached\n", stderr=None)
        if args[:2] == ["build", "--tag"]:
            existing_images.add(args[2])
            return ExecResult(return_code=0, stdout="", stderr=None)
        raise AssertionError(f"unexpected docker call: {args}")

    monkeypatch.setattr(harbor, "_run_docker_cli", fake_docker_cli)

    asyncio.run(harbor.reset())
    first = asyncio.run(harbor.verify())
    second = asyncio.run(harbor.verify())

    assert first.passed is True
    assert second.passed is True
    verifier_calls = create_calls[1:]
    verifier_images = [call["task_env_config"].docker_image for call in verifier_calls]
    assert len(verifier_images) == 2
    assert verifier_images[0] == verifier_images[1]
    assert verifier_images[0].startswith("hb-verifier-cache")
    assert verifier_calls[0]["environment_dir"] == task_dir / "tests"
    assert verifier_calls[1]["environment_dir"] == task_dir / "tests"
    build_calls = [args for args in docker_calls if args[:2] == ["build", "--tag"]]
    inspect_calls = [
        args for args in docker_calls if args[:3] == ["image", "inspect", "--format"]
    ]
    assert len(build_calls) == 1
    assert len(inspect_calls) == 2
    assert build_calls[0][2] == verifier_images[0]
    assert build_calls[0][-1] == str((task_dir / "tests").resolve())

    asyncio.run(harbor.close())
    assert created_envs[0].stop_calls == [True]


def test_harbor_auto_converts_shared_tasks_to_official_separate_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created_envs, create_calls = _patch_lifecycle_environment_factory(monkeypatch)
    task_dir = tmp_path / "task-a"
    _write_minimal_task(task_dir)
    harbor = Harbor(
        HarborConfig(experiments_dir=tmp_path / "experiments"),
        task_name="task-a",
        task_dir=task_dir,
    )
    existing_images: set[str] = set()
    docker_calls: list[list[str]] = []

    async def fake_docker_cli(
        args: list[str],
        *,
        failure_context: str = "Docker command failed",
    ) -> ExecResult:
        del failure_context
        docker_calls.append(args)
        if args[:3] == ["image", "inspect", "--format"]:
            image_name = args[-1]
            if image_name not in existing_images:
                raise RuntimeError(f"No such image: {image_name}")
            return ExecResult(return_code=0, stdout="sha256:cached\n", stderr=None)
        if args[:2] == ["build", "--tag"]:
            existing_images.add(args[2])
            return ExecResult(return_code=0, stdout="", stderr=None)
        raise AssertionError(f"unexpected docker call: {args}")

    monkeypatch.setattr(harbor, "_run_docker_cli", fake_docker_cli)

    asyncio.run(harbor.reset())
    result = asyncio.run(harbor.verify())

    assert result.passed is True
    agent_env, verifier_env = created_envs
    verifier_call = create_calls[1]
    verifier_config = verifier_call["task_env_config"]
    verifier_context = verifier_call["environment_dir"]
    assert harbor.session.task.config.verifier.environment_mode == (
        VerifierEnvironmentMode.SEPARATE
    )
    assert verifier_config.docker_image.startswith("hb-verifier-cache")
    assert verifier_context != task_dir / "tests"
    dockerfile = (verifier_context / "Dockerfile").read_text()
    assert "FROM example/task:latest" in dockerfile
    assert "WORKDIR /app" in dockerfile
    assert "COPY . /tests/" in dockerfile
    assert (verifier_context / "test.sh").exists()
    assert agent_env.download_dir_calls == [
        ("/app", harbor.session.trial_paths.artifacts_dir / "app")
    ]
    assert any(target == "/app" for _, target in verifier_env.upload_dir_calls)
    assert any(args[:2] == ["build", "--tag"] for args in docker_calls)

    asyncio.run(harbor.close())


def _stub_harbor_for_bootstrap(tmp_path: Path, *, exec_side_effect) -> Harbor:
    """Build a Harbor whose session.environment.exec follows a scripted
    sequence (`exec_side_effect`), with a real on-disk bootstrap log dir so we
    can assert the artifacts `_bootstrap_environment` leaves behind."""
    config = HarborConfig(
        experiments_dir=tmp_path / "experiments",
        bootstrap_commands=("apt-get update",),
        bootstrap_timeout_sec=600,
    )
    harbor = Harbor(config, task_name="task-a", task_dir=tmp_path / "task")
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir(parents=True)
    fake_session = MagicMock()
    fake_session.environment.upload_file = AsyncMock()
    fake_session.environment.exec = AsyncMock(side_effect=exec_side_effect)
    fake_session.trial_paths.trial_dir = trial_dir
    harbor._session = fake_session
    return harbor


def test_bootstrap_retries_command_timeout_then_succeeds(
    tmp_path: Path, monkeypatch
) -> None:
    # A transient bootstrap timeout (Harbor raises RuntimeError("Command timed
    # out after N seconds")) is retried within the trial; a later success
    # resolves it without surfacing a crash.
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    harbor = _stub_harbor_for_bootstrap(
        tmp_path,
        exec_side_effect=[
            RuntimeError("Command timed out after 600 seconds"),
            ExecResult(return_code=0, stdout="done", stderr=""),
        ],
    )

    asyncio.run(harbor._bootstrap_environment())

    assert harbor.session.environment.exec.await_count == 2
    bootstrap_dir = tmp_path / "trial" / "bootstrap"
    assert (bootstrap_dir / "return-code.txt").read_text() == "0"
    assert not (bootstrap_dir / "status.txt").exists()


def test_bootstrap_timeout_exhausts_budget_and_marks_status(
    tmp_path: Path, monkeypatch
) -> None:
    # A persistent bootstrap timeout exhausts the within-trial budget and
    # re-raises; the only debuggable artifact (Harbor discards partial output on
    # timeout) is a status.txt marker recording the exhausted timeout.
    from src.adapters.infra_retry import INFRA_RETRY_BUDGET

    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    harbor = _stub_harbor_for_bootstrap(
        tmp_path,
        exec_side_effect=RuntimeError("Command timed out after 600 seconds"),
    )

    with pytest.raises(RuntimeError, match="timed out"):
        asyncio.run(harbor._bootstrap_environment())

    assert harbor.session.environment.exec.await_count == INFRA_RETRY_BUDGET + 1
    status = (tmp_path / "trial" / "bootstrap" / "status.txt").read_text()
    assert "timed out" in status
    assert "retries exhausted" in status


def _stub_docker_environment(*, session_id: str) -> DockerEnvironment:
    env = object.__new__(DockerEnvironment)
    env.session_id = session_id
    env._run_docker_compose_command = AsyncMock()
    return env


def test_stop_docker_environment_uses_image_preserving_compose_down(
    tmp_path: Path,
) -> None:
    harbor = Harbor(
        HarborConfig(experiments_dir=tmp_path / "experiments"),
        task_name="task-a",
        task_dir=tmp_path / "task",
    )
    env = _stub_docker_environment(session_id="Task.Name__Run.ID")

    asyncio.run(harbor._stop_environment(env))

    env._run_docker_compose_command.assert_awaited_once_with(
        ["down", "--volumes", "--remove-orphans"]
    )


def test_stop_docker_environment_fallback_cleans_only_same_compose_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    harbor = Harbor(
        HarborConfig(experiments_dir=tmp_path / "experiments"),
        task_name="task-a",
        task_dir=tmp_path / "task",
    )
    env = _stub_docker_environment(session_id="Task.Name__Run.ID")
    env._run_docker_compose_command = AsyncMock(
        side_effect=RuntimeError("compose network cleanup failed")
    )
    calls: list[list[str]] = []
    expected_label = "label=com.docker.compose.project=task-name__run-id"

    async def fake_docker_cli(
        args: list[str],
        *,
        failure_context: str = "Docker command failed",
    ) -> ExecResult:
        del failure_context
        calls.append(args)
        if args == [
            "container",
            "ls",
            "--all",
            "--quiet",
            "--filter",
            expected_label,
        ]:
            return ExecResult(stdout="container-1\ncontainer-2\n", return_code=0)
        if args == ["volume", "ls", "--quiet", "--filter", expected_label]:
            return ExecResult(stdout="volume-1\n", return_code=0)
        if args == ["network", "ls", "--quiet", "--filter", expected_label]:
            return ExecResult(stdout="network-1\n", return_code=0)
        return ExecResult(stdout="", return_code=0)

    monkeypatch.setattr(harbor, "_run_docker_cli", fake_docker_cli)

    asyncio.run(harbor._stop_environment(env))

    assert "recovered via label fallback" in caplog.text
    assert "compose network cleanup failed" in caplog.text
    assert calls == [
        [
            "container",
            "ls",
            "--all",
            "--quiet",
            "--filter",
            expected_label,
        ],
        [
            "container",
            "rm",
            "--force",
            "--volumes",
            "container-1",
            "container-2",
        ],
        ["volume", "ls", "--quiet", "--filter", expected_label],
        ["volume", "rm", "--force", "volume-1"],
        ["network", "ls", "--quiet", "--filter", expected_label],
        ["network", "rm", "network-1"],
    ]
    assert all("image" not in call and "prune" not in call for call in calls)


def test_stop_docker_environment_raises_when_compose_and_fallback_cleanup_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harbor = Harbor(
        HarborConfig(experiments_dir=tmp_path / "experiments"),
        task_name="task-a",
        task_dir=tmp_path / "task",
    )
    env = _stub_docker_environment(session_id="task-a__run-id")
    env._run_docker_compose_command = AsyncMock(
        side_effect=RuntimeError("compose down failed")
    )

    async def fake_docker_cli(
        args: list[str],
        *,
        failure_context: str = "Docker command failed",
    ) -> ExecResult:
        del failure_context
        raise RuntimeError("docker API unavailable")

    monkeypatch.setattr(harbor, "_run_docker_cli", fake_docker_cli)

    with pytest.raises(RuntimeError, match="Docker cleanup failed"):
        asyncio.run(harbor._stop_environment(env))


def test_run_docker_cli_reports_combined_output_on_failure(tmp_path: Path) -> None:
    harbor = Harbor(
        HarborConfig(experiments_dir=tmp_path / "experiments"),
        task_name="task-a",
        task_dir=tmp_path / "task",
    )

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(harbor._run_docker_cli(["definitely-not-a-real-docker-command"]))

    message = str(exc_info.value)
    assert message.startswith("Docker command failed.")
    assert "Docker cleanup command failed." not in message
    assert "Output:" in message
    assert "Stderr: None" not in message


def test_bootstrap_script_recovers_stale_apt_lock_before_commands(
    tmp_path: Path, monkeypatch
) -> None:
    # A bootstrap attempt orphaned by an exec timeout leaves its in-container
    # apt-get holding /var/lib/apt/lists/lock (Harbor's timeout kills the host
    # docker client, not the in-container process), so a retried bootstrap fails
    # with "Could not get lock ... (apt-get)" -> exit 100. The generated script
    # must clear that stale lock *before* the user commands so the retry can
    # proceed, and cap apt's network wait so an unreachable mirror fails fast
    # instead of hanging until the bootstrap timeout.
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    harbor = _stub_harbor_for_bootstrap(
        tmp_path,
        exec_side_effect=[ExecResult(return_code=0, stdout="", stderr="")],
    )

    asyncio.run(harbor._bootstrap_environment())

    script = (tmp_path / "trial" / "bootstrap" / "bootstrap.sh").read_text()
    assert "/var/lib/apt/lists/lock" in script
    assert "pkill -9 -x apt " in script
    assert script.index("rm -f") < script.index("apt-get update")
    assert "Acquire::http::Timeout" in script


def test_bootstrap_script_routes_apt_through_host_cache_when_reachable(
    tmp_path: Path, monkeypatch
) -> None:
    # The generated bootstrap must probe the optional host-side apt cache and,
    # only when the port is reachable, point apt at it -- so repeated installs
    # hit a local cache instead of the public mirror (the dominant cause of
    # bootstrap stalls under concurrent load). The probe is curl-free
    # (bash /dev/tcp) and the proxy config is written before the user commands.
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    harbor = _stub_harbor_for_bootstrap(
        tmp_path,
        exec_side_effect=[ExecResult(return_code=0, stdout="", stderr="")],
    )

    asyncio.run(harbor._bootstrap_environment())

    script = (tmp_path / "trial" / "bootstrap" / "bootstrap.sh").read_text()
    assert "/dev/tcp/host.docker.internal/3142" in script
    assert 'Acquire::http::Proxy "http://host.docker.internal:3142"' in script
    # Reachability-gated: only set inside the success branch of the probe.
    assert script.index("/dev/tcp/host.docker.internal/3142") < script.index(
        "Acquire::http::Proxy"
    )
    # Configured before the user apt commands run.
    assert script.index("Acquire::http::Proxy") < script.index("apt-get update")


def test_bootstrap_script_routes_pip_uv_through_host_cache_when_reachable(
    tmp_path: Path, monkeypatch
) -> None:
    # The generated bootstrap must also probe the optional host-side PyPI cache
    # (proxpi) and, only when reachable, point uv/pip at it -- so verifier
    # `uv run` installs (torch + CUDA, ~2GB) hit a local cache instead of PyPI on
    # every trial. Same curl-free /dev/tcp probe; the index config is written
    # into the container (/etc/uv/uv.toml, /etc/pip.conf) before user commands so
    # the later in-container verify picks it up.
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    harbor = _stub_harbor_for_bootstrap(
        tmp_path,
        exec_side_effect=[ExecResult(return_code=0, stdout="", stderr="")],
    )

    asyncio.run(harbor._bootstrap_environment())

    script = (tmp_path / "trial" / "bootstrap" / "bootstrap.sh").read_text()
    assert "/dev/tcp/host.docker.internal/3141" in script
    assert "/etc/uv/uv.toml" in script
    assert "/etc/pip.conf" in script
    assert "http://host.docker.internal:3141/index/" in script
    # Reachability-gated: index config only inside the success branch of the probe.
    assert script.index("/dev/tcp/host.docker.internal/3141") < script.index(
        "/etc/uv/uv.toml"
    )
    # Configured before the user commands run.
    assert script.index("/etc/uv/uv.toml") < script.index("apt-get update")
