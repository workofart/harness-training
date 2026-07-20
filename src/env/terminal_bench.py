"""Terminal-Bench 2.1 Docker env."""

from __future__ import annotations

import asyncio
import fcntl
import functools
import hashlib
import io
import json
import os
import tarfile
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib
from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError

from src.env.base import (
    DockerTaskEnv,
    RawEnvOutput,
    StepResult,
    TaskSet,
    VerifyOutcome,
    VerifyVerdict,
    VerifyWrapper,
)
from src.env.docker_shell import DockerShellSession, ManagedDockerNetwork, read_file
from src.env import netcache
from src import determinism
from src.config import EnvironmentConfig

_TERMINAL_BENCH_REPO_URL = "https://github.com/harbor-framework/terminal-bench-2.git"
# Pin tasks and verifier tests to avoid branch drift; bump with parity checks.
_TERMINAL_BENCH_REPO_REF = "2fd12b88aafdd04a52c298e3940bcb189f9766d6"
_TERMINAL_BENCH_CACHE_DIR = (
    Path.home() / ".cache" / "harness-experiment" / "terminal-bench-2-1"
)
_TASK_WORKDIR = "/app"
_WORKDIR_SETUP_COMMAND = f"mkdir -p {_TASK_WORKDIR}"
_TESTS_DIR = "/tests"
_VERIFIER_DIR = "/logs/verifier"
_AGENT_LOG_DIR = "/logs/agent"
_ARTIFACTS_DIR = "/logs/artifacts"
_TEST_STDOUT_PATH = f"{_VERIFIER_DIR}/test-stdout.txt"
_REWARD_TEXT_PATH = f"{_VERIFIER_DIR}/reward.txt"
_VERIFIER_ARTIFACT_DIR = "terminal_bench_verifier"
_VERIFIER_COMMAND = (
    f"rm -f {_REWARD_TEXT_PATH} {_TEST_STDOUT_PATH} && "
    f"chmod +x {_TESTS_DIR}/test.sh && "
    f"{_TESTS_DIR}/test.sh > {_TEST_STDOUT_PATH} 2>&1"
)
_FAKERANDOM_TASK_NAMES = frozenset({"distribution-search"})
# Pinned GPL-2.0 shim, fetched not vendored: see assets/faketime/README.md.
_FAKERANDOM_LIB_HOST_PATH = (
    Path(__file__).resolve().parent / "assets" / "faketime" / "libfaketimeMT.so.1"
)
_FAKERANDOM_LIB_SHA256 = (
    "c68193ec89877804e36f9f9c09279936ba94f869b0dea86684080ca0b72daf9f"
)
# libfaketime_0.9.10-2.1_amd64.deb, behind an opaque by-hash URL.
_FAKERANDOM_DEB_URL = (
    "https://snapshot.debian.org/file/781fba01c508a6a61e6f22a44ef43a9bd4f3d419"
)
_FAKERANDOM_DEB_SHA256 = (
    "b09481e7690680966005330c3f907bba4b5eefc35e1faaea4783cc55655d1150"
)
_FAKERANDOM_DEB_MEMBER = "./usr/lib/x86_64-linux-gnu/faketime/libfaketimeMT.so.1"
_FAKERANDOM_LIB_CONTAINER_PATH = "/opt/framework/libfaketimeMT.so.1"
_FAKERANDOM_RUN_ENV = {
    "LD_PRELOAD": _FAKERANDOM_LIB_CONTAINER_PATH,
    "FAKERANDOM_SEED": "0x12345678DEADBEEF",
}
# These tasks have grade-defining observations that depend on live host
# measurements, so replay would freeze one timing sample as deterministic truth.
_LIVE_ONLY_TASK_NAMES = frozenset({"portfolio-optimization"})
_RESOURCE_ENFORCEMENT_VERSION = 1


def _ar_member(archive: bytes, member_name: str) -> bytes:
    if archive[:8] != b"!<arch>\n":
        raise RuntimeError(f"not an ar archive: {_FAKERANDOM_DEB_URL}")
    offset = 8
    while offset < len(archive):
        header = archive[offset : offset + 60]
        name = header[:16].decode().strip()
        size = int(header[48:58])
        offset += 60
        if name == member_name:
            return archive[offset : offset + size]
        offset += size + (size & 1)
    raise RuntimeError(f"{member_name} not found in {_FAKERANDOM_DEB_URL}")


def _fakerandom_lib_path() -> Path:
    path = _FAKERANDOM_LIB_HOST_PATH
    if not path.is_file():
        with urllib.request.urlopen(_FAKERANDOM_DEB_URL, timeout=60) as response:
            deb = response.read()
        deb_digest = hashlib.sha256(deb).hexdigest()
        if deb_digest != _FAKERANDOM_DEB_SHA256:
            raise RuntimeError(
                f"{_FAKERANDOM_DEB_URL} sha256 {deb_digest} != {_FAKERANDOM_DEB_SHA256}"
            )
        with tarfile.open(
            fileobj=io.BytesIO(_ar_member(deb, "data.tar.xz")), mode="r:xz"
        ) as tar:
            extracted = tar.extractfile(_FAKERANDOM_DEB_MEMBER)
            if extracted is None:
                raise RuntimeError(
                    f"{_FAKERANDOM_DEB_MEMBER} not found in {_FAKERANDOM_DEB_URL}"
                )
            lib = extracted.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        tmp_path.write_bytes(lib)
        os.replace(tmp_path, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != _FAKERANDOM_LIB_SHA256:
        raise RuntimeError(
            f"{path} sha256 {digest} != {_FAKERANDOM_LIB_SHA256}; delete it to refetch"
        )
    return path


@functools.cache
def _env_fingerprint() -> str:
    determinism_payload = json.dumps(
        {
            "source_pins": determinism.PINS_FINGERPRINT,
            "terminal_bench_fakerandom_run_env": _FAKERANDOM_RUN_ENV,
            # _fakerandom_lib_path() validates the on-disk lib against this pin,
            # so the fingerprint can name the pin directly.
            "terminal_bench_fakerandom_lib_sha256": _FAKERANDOM_LIB_SHA256,
            "terminal_bench_fakerandom_task_names": sorted(_FAKERANDOM_TASK_NAMES),
            "terminal_bench_live_only_task_names": sorted(_LIVE_ONLY_TASK_NAMES),
            "terminal_bench_resource_enforcement_version": (
                _RESOURCE_ENFORCEMENT_VERSION
            ),
            "terminal_bench_workdir_setup_command": _WORKDIR_SETUP_COMMAND,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    code = hashlib.sha256(
        (
            f"{_VERIFIER_COMMAND}\x00"
            f"{netcache.CACHE_PROXY_SETUP_SCRIPT.read_text()}\x00"
            f"{determinism_payload}"
        ).encode()
    ).hexdigest()[:12]
    return f"{_TERMINAL_BENCH_REPO_REF}:{code}"


@dataclass(frozen=True, slots=True)
class TerminalBenchTask:
    task_name: str
    instruction: str
    task_dir: Path
    docker_image: str
    agent_timeout_sec: float
    build_timeout_sec: float
    verify_timeout_sec: float
    cpus: int
    memory_mb: int
    replay_id: str | None
    # Full dataset index, not selected panel, fixes each task's subnet identity.
    network: ManagedDockerNetwork


async def load_tasks(
    *,
    task_ids: Sequence[str],
    environment: EnvironmentConfig,
    verify_wrapper: VerifyWrapper | None = None,
) -> TaskSet[TerminalBenchTask]:
    del verify_wrapper
    root = await asyncio.to_thread(_sync_git_dataset)
    index = _load_task_index(root)
    missing = [task_id for task_id in task_ids if task_id not in index]
    if missing:
        available = sorted(index)
        raise KeyError(
            "unknown Terminal-Bench task names: "
            f"{missing}; available examples: {available[:10]}"
        )
    tasks = {}
    for task_id in task_ids:
        task_dir, network = index[task_id]
        tasks[task_id] = _load_task(task_dir, network=network)

    return TaskSet(
        kind=environment.kind,
        tasks=tasks,
        env_factory=lambda task, rollout_dir: TerminalBenchEnv(
            task=task,
            artifacts_dir=rollout_dir,
            host_netcache=environment.host_netcache,
        ),
    )


class TerminalBenchEnv(DockerTaskEnv[TerminalBenchTask]):
    """One Terminal-Bench 2.1 task behind the TaskEnv protocol."""

    _task_workdir = _TASK_WORKDIR

    def __init__(
        self,
        *,
        task: TerminalBenchTask,
        artifacts_dir: Path,
        host_netcache: bool,
    ) -> None:
        self._netcache = netcache.ContainerNetworkCache(enabled=host_netcache)
        self.setup_timeout_sec = task.build_timeout_sec
        super().__init__(
            task=task,
            artifacts_dir=artifacts_dir,
            verify_timeout_sec=task.verify_timeout_sec,
        )

    def _build_solve_env(self, task: TerminalBenchTask) -> DockerShellSession:
        pin_urandom = task.task_dir.name in _FAKERANDOM_TASK_NAMES
        return DockerShellSession(
            image=task.docker_image,
            managed_network=task.network,
            extra_run_environment=(_FAKERANDOM_RUN_ENV if pin_urandom else None),
            read_only_binds=(
                {str(_fakerandom_lib_path()): _FAKERANDOM_LIB_CONTAINER_PATH}
                if pin_urandom
                else None
            ),
            name_prefix="terminal_bench_env",
            cpu_limit=task.cpus,
            memory_limit_mb=task.memory_mb,
            setup_timeout_sec=task.build_timeout_sec,
            setup_command=self._netcache.setup_command(
                base_command=_WORKDIR_SETUP_COMMAND
            ),
        )

    async def verify(self) -> VerifyOutcome:
        await self.provision()
        await self._install_tests()
        exec_result = await self._solve_env.run(
            command=_VERIFIER_COMMAND,
            cwd=_TASK_WORKDIR,
            timeout=self.verify_timeout_sec,
        )
        test_stdout = await read_file(self._solve_env, _TEST_STDOUT_PATH) or ""
        reward = 0.0
        reward_text: str | None = None
        info = {
            "task_name": self._task.task_name,
            "verifier_exit_code": exec_result.exit_code,
        }
        try:
            reward, reward_text = await self._read_reward()
        except (RuntimeError, ValueError) as exc:
            verdict = VerifyVerdict(completed=False, passed=None, error=str(exc))
        else:
            verdict = VerifyVerdict(completed=True, passed=reward >= 1.0, error=None)
        self._write_verifier_artifacts(
            test_stdout=test_stdout,
            info=info,
            verdict=verdict,
            reward_text=reward_text,
        )
        return VerifyOutcome(
            output=RawEnvOutput(
                exit_code=exec_result.exit_code,
                stdout=test_stdout,
                stderr=exec_result.stderr,
            ),
            reward=reward,
            info=info,
            verdict=verdict,
        )

    async def _install_tests(self) -> None:
        await self._solve_env.run(
            command=(
                f"mkdir -p {_TESTS_DIR} {_VERIFIER_DIR} "
                f"{_AGENT_LOG_DIR} {_ARTIFACTS_DIR}"
            ),
            cwd="/",
            timeout=30,
        )
        await self._solve_env.upload_dir(
            source_dir=self._task.task_dir / "tests",
            target_dir=_TESTS_DIR,
        )

    async def _read_reward(self) -> tuple[float, str]:
        # Re-read once to separate Docker stdout loss from invalid verifier output.
        for _ in range(2):
            stdout = await read_file(self._solve_env, _REWARD_TEXT_PATH)
            if stdout is None:
                raise RuntimeError(f"verifier did not write {_REWARD_TEXT_PATH}")
            if stdout:
                reward = stdout.strip()
                if reward not in ("0", "1"):
                    raise ValueError("reward.txt must contain exactly '0' or '1'")
                return float(reward), stdout
        raise RuntimeError("verifier reward output was not readable")

    def write_verify_artifacts(self, result: StepResult) -> None:
        """Reconstruct verifier artifacts from a replayed verify verdict.

        Semantically equivalent to the live writes in ``verify``; the reward
        text is regenerated from the parsed result.
        """
        self._write_verifier_artifacts(
            test_stdout=result.raw_env_output.stdout,
            info=result.info,
            verdict=result.verdict,
            reward_text=(
                f"{int(result.reward)}\n" if result.verdict.completed else None
            ),
        )

    def _write_verifier_artifacts(
        self,
        *,
        test_stdout: str,
        info: Mapping[str, Any],
        verdict: VerifyVerdict,
        reward_text: str | None,
    ) -> None:
        verifier_dir = self._artifacts_dir / _VERIFIER_ARTIFACT_DIR
        verifier_dir.mkdir(parents=True, exist_ok=True)
        (verifier_dir / "test-stdout.txt").write_text(test_stdout)
        artifact = dict(info)
        artifact["completed"] = verdict.completed
        if verdict.completed:
            artifact["passed"] = verdict.passed
        else:
            artifact["error"] = verdict.error
        (verifier_dir / "result.json").write_text(
            json.dumps(artifact, sort_keys=True, indent=2)
        )
        if reward_text is not None:
            (verifier_dir / "reward.txt").write_text(reward_text)

    def pin_network_snapshot(self, token: str) -> None:
        if self._netcache.pin(token):
            self._solve_env.set_setup_command(
                self._netcache.setup_command(base_command=_WORKDIR_SETUP_COMMAND)
            )

    async def provision(self) -> None:
        await self._netcache.ensure_host_caches()
        await self._solve_env.start()


def _managed_network_for_index(index: int) -> ManagedDockerNetwork:
    second_octet = 240 + index // 256
    third_octet = index % 256
    return ManagedDockerNetwork(
        subnet=f"10.{second_octet}.{third_octet}.0/24",
        ipv4_address=f"10.{second_octet}.{third_octet}.2",
    )


def _sync_git_dataset() -> Path:
    # GitHub permits fetching the pinned SHA directly.
    target = _TERMINAL_BENCH_CACHE_DIR.expanduser().resolve()
    if target.exists() and not (target / ".git").exists():
        raise FileExistsError(f"Terminal-Bench cache dir is not a git repo: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Overlapping experiment workers sync the same checkout; serialize host-wide.
    lock_fd = os.open(f"{target}.lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            if target.exists():
                repo = Repo(target)
            else:
                repo = Repo.clone_from(_TERMINAL_BENCH_REPO_URL, target, depth=1)
            repo.remote("origin").fetch(_TERMINAL_BENCH_REPO_REF, depth=1)
            repo.git.checkout("--detach", "FETCH_HEAD")
        except (GitCommandError, InvalidGitRepositoryError) as exc:
            stderr = getattr(exc, "stderr", "")
            detail = stderr.strip() if isinstance(stderr, str) else ""
            raise RuntimeError(
                f"Terminal-Bench git sync failed: {detail or exc}"
            ) from exc
    finally:
        os.close(lock_fd)
    return target


def _load_task_index(
    root: Path,
) -> dict[str, tuple[Path, ManagedDockerNetwork]]:
    index: dict[str, tuple[Path, ManagedDockerNetwork]] = {}
    for network_index, task_toml in enumerate(sorted(root.glob("*/task.toml"))):
        index[task_toml.parent.name] = (
            task_toml.parent,
            _managed_network_for_index(network_index),
        )
    if not index:
        raise FileNotFoundError(f"no Terminal-Bench task.toml files found under {root}")
    return index


def _load_task(task_dir: Path, *, network: ManagedDockerNetwork) -> TerminalBenchTask:
    config = tomllib.loads((task_dir / "task.toml").read_text())
    task_meta = config["task"]
    env_config = config["environment"]
    agent_config = config["agent"]
    verifier_config = config["verifier"]
    task_name = str(task_meta.get("name") or f"terminal-bench/{task_dir.name}")

    return TerminalBenchTask(
        task_name=task_name,
        instruction=(task_dir / "instruction.md").read_text(),
        task_dir=task_dir,
        docker_image=str(env_config["docker_image"]),
        agent_timeout_sec=float(agent_config["timeout_sec"]),
        build_timeout_sec=float(env_config["build_timeout_sec"]),
        verify_timeout_sec=float(verifier_config["timeout_sec"]),
        cpus=env_config["cpus"],
        memory_mb=env_config["memory_mb"],
        replay_id=(
            None if task_dir.name in _LIVE_ONLY_TASK_NAMES else _env_fingerprint()
        ),
        network=network,
    )
