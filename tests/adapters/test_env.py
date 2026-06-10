from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from harbor.environments.base import ExecResult
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import TaskOS
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths

from src.env.docker import DockerCleanup
from src.env.harbor import (
    Harbor,
    HarborConfig,
    TaskDirectoryResolver,
    _ResourceCappedEnvironment,
    _strip_ipv6_no_proxy,
)


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


def test_verifier_context_cache_dir_is_stable_when_trial_config_scopes_experiments(
    tmp_path: Path,
) -> None:
    config = HarborConfig(experiments_dir=tmp_path / "experiments")
    trial_config = config.model_copy(
        update={"experiments_dir": config.experiments_dir / "exp-id" / "tasks"}
    )

    assert (
        config.verifier_context_cache_dir
        == tmp_path / "experiments" / "_verifier_contexts"
    )
    assert trial_config.experiments_dir == config.experiments_dir / "exp-id" / "tasks"
    assert trial_config.verifier_context_cache_dir == config.verifier_context_cache_dir


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
    assert result.action_passed is False
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

    assert result.action_passed is False
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

    asyncio.run(harbor.verify())

    chmod_env = environment.exec_calls[0]["env"]
    verifier_env = environment.exec_calls[1]["env"]
    assert environment.exec_calls[0]["user"] == "root"
    assert chmod_env["OMP_NUM_THREADS"] == "3"
    assert verifier_env["OMP_NUM_THREADS"] == "7"
    assert verifier_env["OPENBLAS_NUM_THREADS"] == "3"
    assert verifier_env["CUSTOM_ENV"] == "set"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # OrbStack's real injection: keep hostnames / IPv4 / IPv4-CIDR / domain
        # suffixes, drop the IPv6 address and IPv6 CIDR that crash httpx.
        (
            "localhost,127.0.0.1,::1,10.0.0.0/8,fd07:b51a:cc66:f0::/64,*.orb.internal",
            "localhost,127.0.0.1,10.0.0.0/8,*.orb.internal",
        ),
        # host:port (single colon) survives; a bare IPv6 (>=2 colons) is dropped.
        ("cache:3142,fe80::1", "cache:3142"),
        # No IPv6 entry -> returned unchanged.
        ("localhost,127.0.0.1,.example.com", "localhost,127.0.0.1,.example.com"),
    ],
)
def test_strip_ipv6_no_proxy(value: str, expected: str) -> None:
    assert _strip_ipv6_no_proxy(value) == expected


class _NoProxyProbeEnvironment:
    os = TaskOS.LINUX
    capabilities = SimpleNamespace(mounted=True)

    def __init__(self, *, no_proxy: str, no_proxy_upper: str = "") -> None:
        self._values = (no_proxy, no_proxy_upper)
        self.exec_calls: list[dict[str, object]] = []

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        del cwd, timeout_sec, user
        self.exec_calls.append({"command": command, "env": env})
        if "printf" in command and "no_proxy" in command:
            # Emulate the container echoing its injected values, in probe order.
            return ExecResult(
                return_code=0,
                stdout=f"{self._values[0]}\n{self._values[1]}\n",
                stderr="",
            )
        return ExecResult(return_code=0, stdout="", stderr="")


def _sanitize_harbor(tmp_path: Path) -> Harbor:
    return Harbor(
        HarborConfig(experiments_dir=tmp_path / "experiments"),
        task_name="task-a",
        task_dir=tmp_path / "task",
    )


def test_sanitize_no_proxy_pins_cleaned_override_and_it_flows_into_execs(
    tmp_path: Path,
) -> None:
    harbor = _sanitize_harbor(tmp_path)
    raw = _NoProxyProbeEnvironment(no_proxy="localhost,::1,10.0.0.0/8")
    capped = _ResourceCappedEnvironment(raw, {"TOKENIZERS_PARALLELISM": "false"})

    asyncio.run(harbor._sanitize_no_proxy(capped))

    # The lowercase value carried IPv6 -> a cleaned override is pinned; the empty
    # uppercase value is left untouched (no spurious key).
    assert capped.resource_env["no_proxy"] == "localhost,10.0.0.0/8"
    assert "NO_PROXY" not in capped.resource_env

    # The override now rides on every later exec, overriding the container's env.
    asyncio.run(capped.exec(command="python -c pass"))
    assert raw.exec_calls[-1]["env"]["no_proxy"] == "localhost,10.0.0.0/8"


def test_sanitize_no_proxy_is_noop_without_ipv6(tmp_path: Path) -> None:
    harbor = _sanitize_harbor(tmp_path)
    raw = _NoProxyProbeEnvironment(no_proxy="localhost,127.0.0.1")
    capped = _ResourceCappedEnvironment(raw, {})

    asyncio.run(harbor._sanitize_no_proxy(capped))

    assert "no_proxy" not in capped.resource_env
    assert "NO_PROXY" not in capped.resource_env


def test_harbor_close_uses_raw_environment_for_lifecycle_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harbor, environment = _attach_recording_harbor(tmp_path, cpus=2)
    stopped = []

    class _FakeDockerCleanup:
        async def stop_environment(self, target):
            stopped.append(target)

    monkeypatch.setattr(harbor, "_docker_cleanup", _FakeDockerCleanup)

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


def test_harbor_exec_concurrency_never_exceeds_heavy_cap(tmp_path: Path) -> None:
    # #1-caps (heavy): a size-N gate admits up to N concurrent heavy execs and
    # no more. The size-1 serialize test above proves mutual exclusion; this
    # proves the cap is the *limit* -- with cap=2 and 5 concurrent execs, peak
    # in-flight reaches 2 (the gate does admit up to the cap) and never 3.
    inflight = 0
    peak = 0

    async def fake_exec(*, command, cwd=None, env=None, timeout_sec=None):
        del command, cwd, env, timeout_sec
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        await asyncio.sleep(0.01)
        inflight -= 1
        return ExecResult(return_code=0, stdout="", stderr="")

    harbor = Harbor(
        HarborConfig(experiments_dir=tmp_path / "experiments"),
        task_name="task-a",
        task_dir=tmp_path / "task",
        exec_semaphore=asyncio.Semaphore(2),
    )
    fake_session = MagicMock()
    fake_session.environment.exec = fake_exec
    agent_dir = tmp_path / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    fake_session.trial_paths.agent_dir = agent_dir
    harbor._session = fake_session

    async def go():
        await asyncio.gather(*(harbor.exec(command=f"c{i}") for i in range(5)))

    asyncio.run(go())

    assert peak == 2


def test_harbor_verify_serializes_under_shared_semaphore(tmp_path: Path) -> None:
    # #1-enforcer: verify() is heavyweight container work (a separate build+test
    # env that can re-pull multi-GB toolchains) and must acquire the same gate as
    # exec/reset, else a panel of concurrent trials runs N verifiers at once and
    # oversubscribes cores. A graded verify and a heavy exec under a size-1 gate
    # must not overlap.
    order: list[str] = []

    async def fake_exec(*, command, cwd=None, env=None, timeout_sec=None):
        del cwd, env, timeout_sec
        order.append(f"enter:{command}")
        await asyncio.sleep(0.01)
        order.append(f"exit:{command}")
        return ExecResult(return_code=0, stdout="", stderr="")

    class _RecordingVerifier:
        async def verify(self):
            order.append("enter:verify")
            await asyncio.sleep(0.01)
            order.append("exit:verify")
            return SimpleNamespace(rewards={"reward": 1.0})

    harbor = Harbor(
        HarborConfig(experiments_dir=tmp_path / "experiments"),
        task_name="task-a",
        task_dir=tmp_path / "task",
        exec_semaphore=asyncio.Semaphore(1),
    )
    agent_dir = tmp_path / "trial" / "agent"
    agent_dir.mkdir(parents=True)
    harbor._session = SimpleNamespace(
        environment=SimpleNamespace(exec=fake_exec),
        verifier_session=_RecordingVerifier(),
        trial_paths=SimpleNamespace(
            agent_dir=agent_dir,
            test_stdout_path=tmp_path / "trial" / "stdout.txt",
            test_stderr_path=tmp_path / "trial" / "stderr.txt",
        ),
    )

    async def go():
        await asyncio.gather(harbor.verify(), harbor.exec(command="heavy"))

    asyncio.run(go())

    # verify acquires the same gate as exec -> the two never interleave.
    assert order in (
        ["enter:verify", "exit:verify", "enter:heavy", "exit:heavy"],
        ["enter:heavy", "exit:heavy", "enter:verify", "exit:verify"],
    )


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


def _patch_docker_image_cache(
    harbor: Harbor,
    monkeypatch: pytest.MonkeyPatch,
) -> list[list[str]]:
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
    return docker_calls


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
    _skip_stale_docker_cleanup(monkeypatch, harbor)

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


def test_long_separate_verifier_session_ids_remain_cleanup_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _created_envs, create_calls = _patch_lifecycle_environment_factory(monkeypatch)
    task_name = f"task-{'a' * 40}"
    task_dir = tmp_path / task_name
    _write_minimal_task(
        task_dir,
        verifier_docker_image="example/verifier:latest",
    )
    harbor = Harbor(
        HarborConfig(experiments_dir=tmp_path / "experiments"),
        task_name=task_name,
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
    _skip_stale_docker_cleanup(monkeypatch, harbor)

    asyncio.run(harbor.reset())
    asyncio.run(harbor.verify())

    verifier_session_id = create_calls[1]["session_id"]
    run_id = harbor.session.trial_paths.trial_dir.name
    assert len(verifier_session_id) <= 63
    assert verifier_session_id.endswith(f"__{run_id}__verifier")
    cleanup = harbor._docker_cleanup()
    assert cleanup.is_stale_cleanup_candidate_project(verifier_session_id)

    prefix, digest, suffix = verifier_session_id.split("__", 2)
    other_digest = "ffffffff" if digest != "ffffffff" else "00000000"
    assert not cleanup.is_stale_cleanup_candidate_project(
        f"{prefix}__{other_digest}__{suffix}"
    )

    asyncio.run(harbor.close())


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
    _skip_stale_docker_cleanup(monkeypatch, harbor)

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


def test_harbor_separate_verifier_applies_network_preamble_as_root_before_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created_envs, _create_calls = _patch_lifecycle_environment_factory(monkeypatch)
    task_dir = tmp_path / "task-a"
    _write_minimal_task(task_dir)
    harbor = Harbor(
        HarborConfig(experiments_dir=tmp_path / "experiments"),
        task_name="task-a",
        task_dir=task_dir,
    )

    _patch_docker_image_cache(harbor, monkeypatch)
    _skip_stale_docker_cleanup(monkeypatch, harbor)

    asyncio.run(harbor.reset())
    result = asyncio.run(harbor.verify())

    assert result.passed is True
    # The separate verifier container must receive the same apt/pip cache and
    # retry/timeout config the agent gets, applied as root before the immutable
    # test.sh runs its own toolchain install -- otherwise a flaky mirror stalls
    # the whole grade.
    verifier_env = created_envs[1]
    commands = [str(call["command"]) for call in verifier_env.exec_calls]
    preamble_idx = next(
        i
        for i, command in enumerate(commands)
        if "host.docker.internal:3142" in command and "Acquire::http::Proxy" in command
    )
    test_idx = next(
        i for i, command in enumerate(commands) if "/tests/test.sh" in command
    )
    assert preamble_idx < test_idx
    assert verifier_env.exec_calls[preamble_idx]["user"] == "root"

    asyncio.run(harbor.close())


def test_harbor_auto_separates_docker_task_when_task_does_not_request_mode(
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

    docker_calls = _patch_docker_image_cache(harbor, monkeypatch)
    _skip_stale_docker_cleanup(monkeypatch, harbor)

    asyncio.run(harbor.reset())
    result = asyncio.run(harbor.verify())

    assert result.passed is True
    assert result.stdout == "verifier verifier stdout\n"
    assert len(created_envs) == 2
    assert len(create_calls) == 2
    agent_env = created_envs[0]
    agent_call = create_calls[0]
    verifier_call = create_calls[1]
    assert agent_call["environment_dir"] == task_dir / "environment"
    assert verifier_call["environment_dir"] != task_dir / "tests"
    assert verifier_call["task_env_config"].docker_image.startswith("hb-verifier-cache")
    assert [args[:2] for args in docker_calls].count(["build", "--tag"]) == 1
    assert not any(
        source == task_dir / "tests" for source, _ in agent_env.upload_dir_calls
    )

    asyncio.run(harbor.close())


def test_harbor_auto_separate_verifier_generates_context_for_plain_tests(
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

    docker_calls = _patch_docker_image_cache(harbor, monkeypatch)
    _skip_stale_docker_cleanup(monkeypatch, harbor)

    asyncio.run(harbor.reset())
    asyncio.run(harbor.verify())
    asyncio.run(harbor.verify())

    verifier_context = create_calls[1]["environment_dir"]
    reused_verifier_context = create_calls[2]["environment_dir"]
    assert verifier_context != task_dir / "tests"
    assert reused_verifier_context == verifier_context
    assert verifier_context.parent.parent == harbor.config.verifier_context_cache_dir
    assert (verifier_context / "test.sh").read_text() == "#!/bin/bash\n"
    assert (verifier_context / "Dockerfile").read_text() == (
        "FROM example/task:latest\n"
        "WORKDIR /app\n"
        "COPY . /tests/\n"
        "RUN chmod +x /tests/test.sh && mkdir -p /logs/verifier /logs/artifacts\n"
    )
    agent_env = created_envs[0]
    verifier_envs = created_envs[1:]
    assert ("/app", harbor.session.trial_paths.artifacts_dir / "app") in (
        agent_env.download_dir_calls
    )
    assert all(
        (
            harbor.session.trial_paths.artifacts_dir / "app",
            "/app",
        )
        in verifier_env.upload_dir_calls
        for verifier_env in verifier_envs
    )
    assert not (harbor.session.trial_paths.artifacts_dir / "app").exists()
    assert (harbor.session.trial_paths.artifacts_dir / "manifest.json").exists()
    build_calls = [args for args in docker_calls if args[:2] == ["build", "--tag"]]
    assert len(build_calls) == 1
    assert build_calls[0][-1] == str(verifier_context.resolve())

    asyncio.run(harbor.close())


def test_harbor_keeps_configured_service_task_verifier_shared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created_envs, create_calls = _patch_lifecycle_environment_factory(monkeypatch)
    task_dir = tmp_path / "pypi-server"
    _write_minimal_task(task_dir)
    harbor = Harbor(
        HarborConfig(
            experiments_dir=tmp_path / "experiments",
            shared_verifier_task_names=("pypi-server",),
        ),
        task_name="pypi-server",
        task_dir=task_dir,
    )

    async def fail_docker_cli(
        args: list[str],
        *,
        failure_context: str = "Docker command failed",
    ) -> ExecResult:
        del failure_context
        raise AssertionError(f"shared verifier should not build verifier image: {args}")

    monkeypatch.setattr(harbor, "_run_docker_cli", fail_docker_cli)
    _skip_stale_docker_cleanup(monkeypatch, harbor)

    asyncio.run(harbor.reset())
    result = asyncio.run(harbor.verify())

    assert result.passed is True
    assert result.stdout == "agent verifier stdout\n"
    assert len(created_envs) == 1
    assert len(create_calls) == 1
    agent_env = created_envs[0]
    agent_call = create_calls[0]
    assert agent_call["environment_dir"] == task_dir / "environment"
    assert any(source == task_dir / "tests" for source, _ in agent_env.upload_dir_calls)
    assert agent_env.download_dir_calls == []

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
    from src.retry import INFRA_RETRY_BUDGET

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


def _docker_cleanup(
    *,
    tmp_path: Path,
    task_name: str,
    run_docker_cli,
) -> DockerCleanup:
    return DockerCleanup(
        task_name=task_name,
        environment_config=HarborConfig(
            experiments_dir=tmp_path / "experiments"
        ).environment,
        run_docker_cli=run_docker_cli,
        logger=logging.getLogger("tests.adapters.test_env.docker_cleanup"),
    )


def _skip_stale_docker_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    harbor: Harbor,
):
    cleanup = harbor._docker_cleanup()
    cleanup.cleanup_stale_docker_compose_projects = AsyncMock()  # type: ignore[method-assign]
    monkeypatch.setattr(harbor, "_docker_cleanup", lambda: cleanup)
    return cleanup.cleanup_stale_docker_compose_projects


def test_stop_docker_environment_uses_image_preserving_compose_down(
    tmp_path: Path,
) -> None:
    async def fail_docker_cli(
        args: list[str],
        *,
        failure_context: str = "Docker command failed",
    ) -> ExecResult:
        del args, failure_context
        raise AssertionError("docker CLI should not run when compose down succeeds")

    cleanup = _docker_cleanup(
        tmp_path=tmp_path,
        task_name="Task.A",
        run_docker_cli=fail_docker_cli,
    )
    env = _stub_docker_environment(session_id="Task.Name__Run.ID")

    asyncio.run(cleanup.stop_environment(env))

    env._run_docker_compose_command.assert_awaited_once_with(
        ["down", "--volumes", "--remove-orphans"]
    )


def test_stop_docker_environment_fallback_cleans_only_same_compose_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
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

    cleanup = _docker_cleanup(
        tmp_path=tmp_path,
        task_name="task-a",
        run_docker_cli=fake_docker_cli,
    )

    asyncio.run(cleanup.stop_environment(env))

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

    cleanup = _docker_cleanup(
        tmp_path=tmp_path,
        task_name="task-a",
        run_docker_cli=fake_docker_cli,
    )

    with pytest.raises(RuntimeError, match="Docker cleanup failed"):
        asyncio.run(cleanup.stop_environment(env))


def test_cleanup_stale_docker_projects_selects_stopped_agent_and_verifier_projects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed_projects: list[str] = []

    async def fake_docker_cli(
        args: list[str],
        *,
        failure_context: str = "Docker command failed",
    ) -> ExecResult:
        del failure_context
        assert args == [
            "container",
            "ls",
            "--all",
            "--filter",
            "label=com.docker.compose.project",
            "--format",
            "{{json .}}",
        ]
        rows = [
            {
                "ID": "agent-stopped",
                "State": "exited",
                "Labels": "com.docker.compose.project=task-a__20260603-120000-abcdef12",
            },
            {
                "ID": "verifier-stopped",
                "State": "dead",
                "Labels": "com.docker.compose.project=task-a__20260603-120000-abcdef12__verifier",
            },
            {
                "ID": "agent-running",
                "State": "running",
                "Labels": "com.docker.compose.project=task-a__20260603-120001-abcdef12",
            },
            {
                "ID": "agent-created",
                "State": "created",
                "Labels": "com.docker.compose.project=task-a__20260603-120002-abcdef12",
            },
            {
                "ID": "verifier-removing",
                "State": "removing",
                "Labels": "com.docker.compose.project=task-a__20260603-120003-abcdef12__verifier",
            },
            {
                "ID": "other-compose",
                "State": "exited",
                "Labels": "com.docker.compose.project=postgres",
            },
            {
                "ID": "other-task-stopped",
                "State": "exited",
                "Labels": "com.docker.compose.project=task-b__20260603-120000-abcdef12",
            },
            {
                "ID": "substring-task-stopped",
                "State": "exited",
                "Labels": "com.docker.compose.project=task-a__b__20260603-120000-abcdef12",
            },
        ]
        return ExecResult(
            return_code=0,
            stdout="\n".join(json.dumps(row) for row in rows),
            stderr=None,
        )

    async def fake_cleanup(project_name: str) -> None:
        removed_projects.append(project_name)

    cleanup = _docker_cleanup(
        tmp_path=tmp_path,
        task_name="Task.A",
        run_docker_cli=fake_docker_cli,
    )
    monkeypatch.setattr(cleanup, "cleanup_docker_project_by_label", fake_cleanup)

    asyncio.run(cleanup.cleanup_stale_docker_compose_projects())

    assert removed_projects == [
        "task-a__20260603-120000-abcdef12",
        "task-a__20260603-120000-abcdef12__verifier",
    ]


def test_cleanup_stale_docker_projects_skips_malformed_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removed_projects: list[str] = []

    async def fake_docker_cli(
        args: list[str],
        *,
        failure_context: str = "Docker command failed",
    ) -> ExecResult:
        del args, failure_context
        rows = [
            "{not-json",
            json.dumps({"ID": "missing-labels", "State": "exited"}),
            json.dumps(
                {
                    "ID": "missing-state",
                    "Labels": "com.docker.compose.project=task-a__20260603-120000-abcdef12",
                }
            ),
            json.dumps(
                {
                    "ID": "valid",
                    "State": "exited",
                    "Labels": "com.docker.compose.project=task-a__20260603-120001-abcdef12",
                }
            ),
        ]
        return ExecResult(return_code=0, stdout="\n".join(rows), stderr=None)

    async def fake_cleanup(project_name: str) -> None:
        removed_projects.append(project_name)

    cleanup = _docker_cleanup(
        tmp_path=tmp_path,
        task_name="task-a",
        run_docker_cli=fake_docker_cli,
    )
    monkeypatch.setattr(cleanup, "cleanup_docker_project_by_label", fake_cleanup)

    asyncio.run(cleanup.cleanup_stale_docker_compose_projects())

    assert removed_projects == ["task-a__20260603-120001-abcdef12"]


def test_cleanup_stale_docker_projects_ignores_idempotent_docker_cleanup_races(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stale_project = "task-a__20260603-120000-abcdef12"
    calls: list[list[str]] = []

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
            "--filter",
            "label=com.docker.compose.project",
            "--format",
            "{{json .}}",
        ]:
            return ExecResult(
                return_code=0,
                stdout=json.dumps(
                    {
                        "ID": "container-1",
                        "State": "exited",
                        "Labels": f"com.docker.compose.project={stale_project}",
                    }
                ),
                stderr=None,
            )
        if args == [
            "container",
            "ls",
            "--all",
            "--quiet",
            "--filter",
            f"label=com.docker.compose.project={stale_project}",
        ]:
            return ExecResult(return_code=0, stdout="container-1\n", stderr=None)
        if args == [
            "container",
            "rm",
            "--force",
            "--volumes",
            "container-1",
        ]:
            raise RuntimeError(
                "Docker cleanup command failed. Command: docker container rm "
                "--force --volumes container-1. Return code: 1. Output: "
                "Error response from daemon: removal of container container-1 "
                "is already in progress\n."
            )
        if args == [
            "volume",
            "ls",
            "--quiet",
            "--filter",
            f"label=com.docker.compose.project={stale_project}",
        ]:
            return ExecResult(return_code=0, stdout="", stderr=None)
        if args == [
            "network",
            "ls",
            "--quiet",
            "--filter",
            f"label=com.docker.compose.project={stale_project}",
        ]:
            return ExecResult(return_code=0, stdout="network-1\n", stderr=None)
        if args == ["network", "rm", "network-1"]:
            raise RuntimeError(
                "Docker cleanup command failed. Command: docker network rm "
                "network-1. Return code: 1. Output: Error response from "
                "daemon: network network-1 not found\nexit status 1\n."
            )
        return ExecResult(return_code=0, stdout="", stderr=None)

    cleanup = _docker_cleanup(
        tmp_path=tmp_path,
        task_name="task-a",
        run_docker_cli=fake_docker_cli,
    )

    asyncio.run(cleanup.cleanup_stale_docker_compose_projects())

    assert "Failed to clean stale Docker compose project" not in caplog.text
    assert calls == [
        [
            "container",
            "ls",
            "--all",
            "--filter",
            "label=com.docker.compose.project",
            "--format",
            "{{json .}}",
        ],
        [
            "container",
            "ls",
            "--all",
            "--quiet",
            "--filter",
            f"label=com.docker.compose.project={stale_project}",
        ],
        [
            "container",
            "rm",
            "--force",
            "--volumes",
            "container-1",
        ],
        [
            "volume",
            "ls",
            "--quiet",
            "--filter",
            f"label=com.docker.compose.project={stale_project}",
        ],
        [
            "network",
            "ls",
            "--quiet",
            "--filter",
            f"label=com.docker.compose.project={stale_project}",
        ],
        ["network", "rm", "network-1"],
    ]


def test_cleanup_stale_docker_projects_serializes_same_task_cleanup(
    tmp_path: Path,
) -> None:
    stale_project = "task-a__20260603-120000-abcdef12"
    active_stale_listings = 0
    max_active_stale_listings = 0

    async def fake_docker_cli(
        args: list[str],
        *,
        failure_context: str = "Docker command failed",
    ) -> ExecResult:
        nonlocal active_stale_listings, max_active_stale_listings
        del failure_context
        if args == [
            "container",
            "ls",
            "--all",
            "--filter",
            "label=com.docker.compose.project",
            "--format",
            "{{json .}}",
        ]:
            active_stale_listings += 1
            max_active_stale_listings = max(
                max_active_stale_listings,
                active_stale_listings,
            )
            await asyncio.sleep(0.01)
            active_stale_listings -= 1
            return ExecResult(
                return_code=0,
                stdout=json.dumps(
                    {
                        "ID": "container-1",
                        "State": "exited",
                        "Labels": f"com.docker.compose.project={stale_project}",
                    }
                ),
                stderr=None,
            )
        return ExecResult(return_code=0, stdout="", stderr=None)

    cleanup_a = _docker_cleanup(
        tmp_path=tmp_path,
        task_name="task-a",
        run_docker_cli=fake_docker_cli,
    )
    cleanup_b = _docker_cleanup(
        tmp_path=tmp_path,
        task_name="task-a",
        run_docker_cli=fake_docker_cli,
    )

    async def go() -> None:
        await asyncio.gather(
            cleanup_a.cleanup_stale_docker_compose_projects(),
            cleanup_b.cleanup_stale_docker_compose_projects(),
        )

    asyncio.run(go())

    assert max_active_stale_listings == 1


def test_cleanup_stale_docker_projects_waits_for_same_task_environment_teardown(
    tmp_path: Path,
) -> None:
    stale_project = "task-a__20260603-120000-abcdef12"
    teardown_active = False
    stale_listing_overlapped_teardown = False

    env = _stub_docker_environment(session_id=stale_project)

    async def fake_compose_down(args: list[str]) -> None:
        nonlocal teardown_active
        assert args == ["down", "--volumes", "--remove-orphans"]
        teardown_active = True
        await asyncio.sleep(0.01)
        teardown_active = False

    env._run_docker_compose_command = AsyncMock(side_effect=fake_compose_down)

    async def fake_docker_cli(
        args: list[str],
        *,
        failure_context: str = "Docker command failed",
    ) -> ExecResult:
        nonlocal stale_listing_overlapped_teardown
        del failure_context
        if args == [
            "container",
            "ls",
            "--all",
            "--filter",
            "label=com.docker.compose.project",
            "--format",
            "{{json .}}",
        ]:
            stale_listing_overlapped_teardown = (
                stale_listing_overlapped_teardown or teardown_active
            )
            return ExecResult(
                return_code=0,
                stdout=json.dumps(
                    {
                        "ID": "container-1",
                        "State": "exited",
                        "Labels": f"com.docker.compose.project={stale_project}",
                    }
                ),
                stderr=None,
            )
        return ExecResult(return_code=0, stdout="", stderr=None)

    cleanup = _docker_cleanup(
        tmp_path=tmp_path,
        task_name="task-a",
        run_docker_cli=fake_docker_cli,
    )

    async def go() -> None:
        stop = asyncio.create_task(cleanup.stop_environment(env))
        await asyncio.sleep(0)
        await cleanup.cleanup_stale_docker_compose_projects()
        await stop

    asyncio.run(go())

    assert not stale_listing_overlapped_teardown


def test_harbor_reset_runs_stale_docker_project_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_lifecycle_environment_factory(monkeypatch)
    task_dir = tmp_path / "task-a"
    _write_minimal_task(task_dir)
    harbor = Harbor(
        HarborConfig(experiments_dir=tmp_path / "experiments"),
        task_name="task-a",
        task_dir=task_dir,
    )
    cleanup = _skip_stale_docker_cleanup(monkeypatch, harbor)

    asyncio.run(harbor.reset())

    cleanup.assert_awaited_once_with()


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
