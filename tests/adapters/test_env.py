from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from harbor.environments.base import ExecResult
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths

from src.adapters.env import Harbor, HarborConfig, TaskDirectoryResolver


def _write_minimal_task(
    task_dir: Path,
    *,
    cpus: int = 1,
    verifier_env: dict[str, str] | None = None,
) -> None:
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "solution").mkdir()
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("solve it\n")
    task_toml = [
        'version = "1.0"',
        "",
        "[environment]",
        'docker_image = "example/task:latest"',
        f"cpus = {cpus}",
    ]
    if verifier_env:
        task_toml.extend(["", "[verifier.env]"])
        task_toml.extend(f'{key} = "{value}"' for key, value in verifier_env.items())
    (task_dir / "task.toml").write_text("\n".join(task_toml) + "\n")
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
    ) -> ExecResult:
        self.exec_calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "timeout_sec": timeout_sec,
            }
        )
        if self.reward_path is not None and command.startswith("/tests/test.sh"):
            self.reward_path.write_text("1.0")
        return ExecResult(return_code=0, stdout="", stderr="")


def _attach_recording_harbor(
    tmp_path: Path,
    *,
    cpus: int,
    verifier_env: dict[str, str] | None = None,
) -> tuple[Harbor, RecordingEnvironment]:
    task_dir = tmp_path / "task"
    _write_minimal_task(task_dir, cpus=cpus, verifier_env=verifier_env)
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
    # The run-scoped exec_semaphore bounds heavyweight container commands. A
    # size-1 gate must forbid two default Harbor.exec calls from overlapping
    # inside the container — the signal-preserving guarantee that lets
    # trial-admission (outer max_trial_concurrency) run ahead of host-CPU
    # capacity (inner max_env_concurrency) without oversubscribing cores.
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

        async def exec(self, *, command, cwd=None, env=None, timeout_sec=None):
            del cwd, env, timeout_sec
            assert command == "pwd"
            return ExecResult(return_code=0, stdout="/app\n", stderr="")

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
