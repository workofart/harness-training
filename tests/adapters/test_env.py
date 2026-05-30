from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from harbor.environments.base import ExecResult
from harbor.environments.docker.docker import DockerEnvironment

from src.adapters.env import Harbor, HarborConfig, TaskDirectoryResolver


def _write_minimal_task(task_dir: Path) -> None:
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "solution").mkdir()
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("solve it\n")
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n\n[environment]\ndocker_image = "example/task:latest"\n'
    )
    (task_dir / "environment" / "Dockerfile").write_text("FROM python:3.13-slim\n")
    (task_dir / "solution" / "solve.sh").write_text("#!/bin/bash\n")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\n")


def test_task_directory_resolver_prefers_local_task_override(tmp_path: Path) -> None:
    overrides_dir = tmp_path / "overrides"
    override_task_dir = overrides_dir / "task-a"
    _write_minimal_task(override_task_dir)
    task_dirs = TaskDirectoryResolver(
        HarborConfig(
            experiments_dir=tmp_path / "experiments",
            task_overrides_dir=overrides_dir,
        )
    ).resolve(["task-a"])

    assert task_dirs == {"task-a": override_task_dir.resolve()}


def test_task_directory_resolver_falls_back_to_registry_download_when_override_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from harbor.registry.client import factory as registry_factory
    from harbor.tasks import client as tasks_client

    downloaded_task_dir = tmp_path / "downloaded-task"
    source_task_id = object()

    class FakeTask:
        name = "task-a"

        def to_source_task_id(self):
            return source_task_id

    class FakeRegistryClient:
        def get_dataset_spec(self, dataset_name, dataset_version):
            assert dataset_name == "terminal-bench"
            assert dataset_version is None
            return type("Dataset", (), {"tasks": [FakeTask()]})()

    class FakeTaskClient:
        def download_tasks(self, task_ids):
            assert task_ids == [source_task_id]
            return [downloaded_task_dir]

    monkeypatch.setattr(
        registry_factory.RegistryClientFactory,
        "create",
        staticmethod(lambda: FakeRegistryClient()),
    )
    monkeypatch.setattr(tasks_client, "TaskClient", FakeTaskClient)

    task_dirs = TaskDirectoryResolver(
        HarborConfig(
            experiments_dir=tmp_path / "experiments",
            task_overrides_dir=tmp_path / "overrides",
        )
    ).resolve(["task-a"])

    assert task_dirs == {"task-a": downloaded_task_dir}


def test_task_directory_resolver_rejects_invalid_local_task_override(
    tmp_path: Path,
) -> None:
    invalid_override_dir = tmp_path / "overrides" / "task-a"
    invalid_override_dir.mkdir(parents=True)
    (invalid_override_dir / "instruction.md").write_text("incomplete\n")

    with pytest.raises(RuntimeError, match="local task override is invalid"):
        TaskDirectoryResolver(
            HarborConfig(
                experiments_dir=tmp_path / "experiments",
                task_overrides_dir=tmp_path / "overrides",
            )
        ).resolve(["task-a"])


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

    async def fake_docker_cli(args: list[str]) -> ExecResult:
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

    async def fake_docker_cli(args: list[str]) -> ExecResult:
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
    assert "Output:" in message
    assert "Stderr: None" not in message
