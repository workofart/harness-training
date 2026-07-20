"""Deterministic Docker shell used by env adapters."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import io
import logging
import os
import shlex
import tarfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import docker
from docker.errors import DockerException, ImageNotFound, NotFound
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_incrementing,
)

from src.env.base import (
    COMMAND_TIMEOUT_EXIT_CODE,
    DEFAULT_SETUP_TIMEOUT_SEC,
    RawEnvOutput,
    RunAction,
    UnscorableInfraError,
)
from src import determinism

logger = logging.getLogger(__name__)
DOCKER_INFRA_RETRY_BUDGET = 5

# Owner PID scopes exit sweeps to this process's solve containers.
SESSION_OWNER_PID_LABEL = "harness_experiment.owner_pid"


@dataclass(frozen=True, slots=True)
class ExecResult:
    exit_code: int | None
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class ManagedDockerNetwork:
    subnet: str
    ipv4_address: str


# Budgets enforced inside the container so slow commands die cleanly: exec
# returns, the container survives, and no docker exec thread strands.
COMMAND_TIMEOUT_KILL_GRACE_SEC = 10
"""Ensures TERM-ignoring commands are eventually killed."""
COMMAND_TIMEOUT_BACKSTOP_SEC = 60
"""Allows force-removal if Docker or the in-container timeout wedges."""


# Bounds trace serialization, with >10x headroom over the ~24KB prompt-visible prefix.
MAX_CAPTURED_STREAM_BYTES = 256 * 1024
BINARY_STDOUT_PLACEHOLDER = "<BINARY_STDOUT>"
_STAGED_COMMAND_ENV = "FRAMEWORK_EXEC_COMMAND"
_STAGED_COMMAND_WRAPPER = (
    f'exec env -u {_STAGED_COMMAND_ENV} bash -lc "${_STAGED_COMMAND_ENV}"'
)
_COMMAND_ARGV_STAGING_BYTE_THRESHOLD = 4096


def _bounded_decode(data: bytes | None) -> str:
    raw = data or b""
    text = raw[:MAX_CAPTURED_STREAM_BYTES].decode("utf-8", "replace")
    if len(raw) > MAX_CAPTURED_STREAM_BYTES:
        text += f"\n...[env output truncated: {len(raw)} bytes total]"
    return text


def _decode_stdout(data: bytes | None) -> str:
    raw = data or b""
    if b"\x00" in raw:
        return BINARY_STDOUT_PLACEHOLDER
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return BINARY_STDOUT_PLACEHOLDER
    return _bounded_decode(raw)


class DockerShellSession:
    """Interactive shell in a long-lived Docker container."""

    # One process-wide client avoids EMFILE from per-session daemon sockets.
    _shared_client: ClassVar[docker.DockerClient | None] = None

    @classmethod
    def _shared_docker_client(cls) -> docker.DockerClient:
        # Sessions are only created on the event loop thread, so no lock.
        if cls._shared_client is None:
            cls._shared_client = docker.from_env()
        return cls._shared_client

    @classmethod
    def sweep_owner_resources(cls, owner_pid: int) -> None:
        """Last-resort teardown of the containers and networks owned by ``owner_pid``.

        Per-rollout ``close()`` covers normal completion and programmatic
        cancellation, but Ctrl-C tears down the event loop before those awaits
        finish, stranding in-flight containers. Registered with ``atexit``
        below, this runs synchronously outside the loop, so nothing can cancel
        it. The measurement parent also calls it with the reaped worker's PID:
        the only fallback that survives a child SIGKILL. PID scoping keeps it
        off concurrent runs' containers and the persistent caches.
        """
        client = cls._shared_client
        temporary = client is None
        if temporary:
            if owner_pid == os.getpid():
                # No client means this process never created docker resources.
                return
            try:
                # Bounded so a wedged docker cannot hang the parent.
                client = docker.from_env(timeout=30)
            except DockerException:
                return
        label = f"{SESSION_OWNER_PID_LABEL}={owner_pid}"
        try:
            try:
                containers = client.containers.list(all=True, filters={"label": label})
            except DockerException:
                return
            removed = 0
            for container in containers:
                with contextlib.suppress(DockerException):
                    container.remove(force=True, v=True)
                    removed += 1
            with contextlib.suppress(DockerException):
                for network in client.networks.list(filters={"label": label}):
                    with contextlib.suppress(DockerException, NotFound):
                        network.remove()
            if removed:
                print(f"cleaned up {removed} leftover solve container(s)")
        finally:
            if temporary:
                with contextlib.suppress(DockerException):
                    client.close()

    def __init__(
        self,
        *,
        image: str,
        managed_network: ManagedDockerNetwork | None = None,
        extra_run_environment: Mapping[str, str] | None = None,
        read_only_binds: Mapping[str, str] | None = None,
        name_prefix: str = "swe_env",
        cpu_limit: int | None = None,
        memory_limit_mb: int | None = None,
        setup_timeout_sec: float = DEFAULT_SETUP_TIMEOUT_SEC,
        setup_command: str | None = None,
    ) -> None:
        self.container_name = f"{name_prefix}_{uuid.uuid4().hex[:12]}"
        self._image = image
        self._managed_network = managed_network
        self._managed_network_obj = None
        self._run_environment = {
            **determinism.SOLVE_EXEC_ENV,
            **(extra_run_environment or {}),
        }
        # Avoid per-rollout fetch/build for injected files such as fakerandom.
        self._read_only_binds = dict(read_only_binds or {})
        self._cpu_limit = cpu_limit
        self._memory_limit_mb = memory_limit_mb
        self._setup_timeout_sec = setup_timeout_sec
        self._setup_command = setup_command
        self._docker_client = self._shared_docker_client()
        self._best_effort_warned_commands: set[str] = set()
        self._started = False

    def set_setup_command(self, setup_command: str | None) -> None:
        if self._started and setup_command != self._setup_command:
            raise RuntimeError("cannot change Docker setup command after start")
        self._setup_command = setup_command

    async def start(self) -> None:
        if self._started:
            return

        # A daemon blip fails before the container exists, so retried `docker run`
        # safely reuses the same --name.
        async def _run_container():
            return await asyncio.wait_for(
                asyncio.to_thread(self._run_container_sync),
                timeout=self._setup_timeout_sec,
            )

        retrying = AsyncRetrying(
            retry=retry_if_exception_type(DockerException),
            stop=stop_after_attempt(DOCKER_INFRA_RETRY_BUDGET + 1),
            wait=wait_incrementing(start=1, increment=1),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            sleep=asyncio.sleep,
            reraise=True,
        )
        try:
            container = await retrying(_run_container)
        except Exception:
            await self._remove_managed_network()
            raise
        self._started = True
        # Setup runs before mtime reset so its written files are normalized too.
        if self._setup_command is not None:
            try:
                await self._exec_required(container, self._setup_command)
            except Exception:
                with contextlib.suppress(Exception):
                    await self._force_remove_container(container)
                await self._remove_managed_network()
                self._started = False
                raise
        await self._exec_best_effort(container, determinism.GIT_HOOKS_INIT_COMMAND)
        await self._exec_best_effort(container, determinism.GDB_INIT_COMMAND)
        # Fixed-epoch mtimes so `ls -l` cannot leak the run's wall-clock start time.
        await self._exec_best_effort(container, determinism.MTIME_RESET_COMMAND)

    async def _exec_best_effort(self, container, command: str) -> None:
        try:
            result = await self._exec_control_command(container, command)
        except Exception as exc:
            detail = repr(exc)
        else:
            if result.exit_code == 0:
                return
            _stdout, stderr = result.output
            detail = (
                f"exit_code={result.exit_code} "
                f"stderr={_bounded_decode(stderr).strip()[:500]!r}"
            )
        if command in self._best_effort_warned_commands:
            return
        self._best_effort_warned_commands.add(command)
        # A skipped determinism pin otherwise appears later as unexplained drift.
        logger.warning(
            "best-effort determinism pin failed: container=%s command=%r detail=%s",
            self.container_name,
            command[:80],
            detail,
        )

    async def _exec_required(self, container, command: str) -> None:
        result = await self._exec_control_command(
            container, command, timeout=self._setup_timeout_sec
        )
        if result.exit_code == 0:
            return
        stdout, stderr = result.output
        detail = "\n".join(
            part
            for part in (
                _bounded_decode(stderr).strip(),
                _bounded_decode(stdout).strip(),
            )
            if part
        )
        raise UnscorableInfraError(
            f"required setup command failed with exit code {result.exit_code}: {detail}"
        )

    async def _exec_control_command(
        self, container, command: str, *, timeout: float = 30.0
    ):
        return await asyncio.wait_for(
            asyncio.to_thread(
                container.exec_run,
                ["bash", "-lc", command],
                stdout=True,
                stderr=True,
                workdir="/",
                demux=True,
                environment=determinism.SOLVE_EXEC_ENV,
            ),
            timeout=timeout,
        )

    def _run_container_sync(self):
        run_kwargs = {
            "detach": True,
            "platform": "linux/amd64",
            "name": self.container_name,
            # Fixed hostname: default $HOSTNAME is the per-run container id.
            "hostname": determinism.CONTAINER_HOSTNAME,
            "labels": {SESSION_OWNER_PID_LABEL: str(os.getpid())},
        }
        if self._read_only_binds:
            run_kwargs["volumes"] = {
                host: {"bind": container, "mode": "ro"}
                for host, container in self._read_only_binds.items()
            }
        if self._cpu_limit is not None:
            run_kwargs["nano_cpus"] = self._cpu_limit * 1_000_000_000
        if self._memory_limit_mb is not None:
            run_kwargs["mem_limit"] = self._memory_limit_mb * 1024 * 1024
        if self._managed_network is None:
            run_kwargs["network_mode"] = "none"
        else:
            network = self._managed_network
            network_name = f"{self.container_name}_net"
            self._create_managed_network(network_name, network)
            run_kwargs["network"] = network_name
            # Keyed by network name, unwrapped: `containers.run` wraps it itself and
            # drops any config it cannot find the network in (silently un-pinning
            # the IP). Only `api.create_container` takes a NetworkingConfig.
            run_kwargs["networking_config"] = {
                network_name: self._docker_client.api.create_endpoint_config(
                    ipv4_address=network.ipv4_address
                )
            }
        argv = ["sleep", "infinity"]
        try:
            return self._docker_client.containers.run(self._image, argv, **run_kwargs)
        except ImageNotFound:
            self._docker_client.images.pull(self._image)
            return self._docker_client.containers.run(self._image, argv, **run_kwargs)

    def _create_managed_network(self, name: str, network: ManagedDockerNetwork) -> None:
        # A retried `docker run` must not try to recreate an existing network.
        if self._managed_network_obj is not None:
            return
        ipam_pool = docker.types.IPAMPool(subnet=network.subnet)
        ipam = docker.types.IPAMConfig(pool_configs=[ipam_pool])
        self._managed_network_obj = self._docker_client.networks.create(
            name,
            driver="bridge",
            ipam=ipam,
            labels={SESSION_OWNER_PID_LABEL: str(os.getpid())},
        )

    async def _get_container(self):
        return await asyncio.to_thread(
            self._docker_client.containers.get, self.container_name
        )

    async def _force_remove_container(self, container) -> None:
        await asyncio.wait_for(
            asyncio.to_thread(container.remove, force=True, v=True),
            timeout=120,
        )

    async def run(
        self,
        *,
        command: str,
        cwd: str,
        timeout: float | None,
        lossless: bool = False,
    ) -> ExecResult:
        """Run one command in the container.

        ``lossless`` returns output verbatim for control-plane callers such as
        patch capture; the default bounds it and scrubs binary stdout, since that
        output reaches the model and the trace.
        """
        container = await self._get_container()
        # Prior commands mutate top-level dir mtimes (/tmp especially); re-pin at
        # the boundary, before model-visible output can inspect them.
        await self._exec_best_effort(container, determinism.MTIME_RESET_COMMAND)
        command_env = self._run_environment
        shell_command = command
        if (
            "\n" in command
            or len(command.encode()) > _COMMAND_ARGV_STAGING_BYTE_THRESHOLD
        ):
            command_env = {
                **self._run_environment,
                _STAGED_COMMAND_ENV: command,
            }
            shell_command = _STAGED_COMMAND_WRAPPER
        if timeout is None:
            # Verify runs directly; its timeout applies one layer up.
            argv = ["bash", "-lc", shell_command]
            backstop: float | None = None
        else:
            argv = [
                "timeout",
                "-k",
                str(COMMAND_TIMEOUT_KILL_GRACE_SEC),
                str(int(timeout)),
                "bash",
                "-lc",
                shell_command,
            ]
            backstop = (
                int(timeout)
                + COMMAND_TIMEOUT_KILL_GRACE_SEC
                + COMMAND_TIMEOUT_BACKSTOP_SEC
            )
        exec_run = asyncio.to_thread(
            container.exec_run,
            argv,
            stdout=True,
            stderr=True,
            workdir=cwd,
            demux=True,
            # Pin process-level entropy sources so tool output is reproducible.
            environment=command_env,
        )
        try:
            result = await asyncio.wait_for(exec_run, backstop)
        except TimeoutError:
            await asyncio.to_thread(container.remove, force=True, v=True)
            self._started = False
            raise
        stdout, stderr = result.output
        if lossless:
            return ExecResult(
                exit_code=result.exit_code,
                stdout=(stdout or b"").decode("utf-8", "replace"),
                stderr=(stderr or b"").decode("utf-8", "replace"),
            )
        return ExecResult(
            exit_code=result.exit_code,
            stdout=_decode_stdout(stdout),
            stderr=_bounded_decode(stderr),
        )

    async def upload_dir(self, *, source_dir: Path, target_dir: str) -> None:
        container = await self._get_container()
        await asyncio.to_thread(
            container.put_archive, target_dir, self._tar_directory_contents(source_dir)
        )

    @staticmethod
    def _tar_directory_contents(source_dir: Path) -> bytes:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            for path in sorted(source_dir.rglob("*")):
                tar.add(
                    path,
                    arcname=path.relative_to(source_dir).as_posix(),
                    recursive=False,
                    filter=DockerShellSession._normalize_tarinfo,
                )
        buffer.seek(0)
        return buffer.read()

    @staticmethod
    def _normalize_tarinfo(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = 0
        return info

    async def close(self) -> None:
        # Only the container is released; the client is shared process-wide.
        try:
            if self._started:
                with contextlib.suppress(NotFound):
                    container = await self._get_container()
                    await self._force_remove_container(container)
                self._started = False
        finally:
            await self._remove_managed_network()

    async def _remove_managed_network(self) -> None:
        network = self._managed_network_obj
        self._managed_network_obj = None
        if network is not None:
            with contextlib.suppress(DockerException, NotFound):
                await asyncio.wait_for(asyncio.to_thread(network.remove), timeout=120)


async def run_step(
    session: DockerShellSession, action: RunAction, *, default_cwd: str
) -> RawEnvOutput:
    """Run one agent shell action on a started session.

    A self-inflicted timeout is non-terminal: the agent gets the partial output
    plus the budget it hit, so one slow command cannot lose the rollout."""
    exec_result = await session.run(
        command=action.command,
        cwd=action.cwd or default_cwd,
        timeout=action.timeout_sec,
    )
    stderr = exec_result.stderr
    if (
        action.timeout_sec is not None
        and exec_result.exit_code == COMMAND_TIMEOUT_EXIT_CODE
    ):
        note = (
            f"[command timed out after {action.timeout_sec}s and was terminated; the "
            "output above is partial -- narrow the command or pass a larger "
            "timeout_sec]"
        )
        stderr = f"{stderr}\n{note}" if stderr else note
    return RawEnvOutput(
        exit_code=exec_result.exit_code,
        stdout=exec_result.stdout,
        stderr=stderr,
    )


async def read_file(session: DockerShellSession, path: str) -> str | None:
    """Read a container file's text; None when it cannot be read (e.g. missing)."""
    exec_result = await session.run(
        command=f"cat {shlex.quote(path)}", cwd="/", timeout=30
    )
    if exec_result.exit_code != 0:
        return None
    return exec_result.stdout


atexit.register(lambda: DockerShellSession.sweep_owner_resources(os.getpid()))
