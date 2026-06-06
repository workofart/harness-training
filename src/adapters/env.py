"""Harbor environment adapter."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import shutil
import tomllib
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig as TaskEnvironmentConfig
from harbor.models.task.task import Task
from harbor.models.task.verifier_mode import resolve_effective_verifier_env_config
from harbor.models.trial.config import EnvironmentConfig, ServiceVolumeConfig
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.verifier import Verifier
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Heavy harbor imports deferred to first use: `harbor.environments.factory` pulls
# in all environment backends (daytona, e2b, gke, modal, runloop -- ~2.6s cold);
# `harbor.registry.client.factory` pulls in supabase (~0.35s). Moving them
# behind first call to reset() / download path keeps `uv run exp` cold-start
# fast and shaves cost from invocations that never reach those paths.

from src.adapters.infra_retry import INFRA_RETRY_BUDGET, retry_transient
from src.adapters.harbor_docker import (
    DockerCleanup,
    _MAX_VERIFIER_ENV_SESSION_ID_LEN,
    _compact_verifier_task_session_prefix,
    _safe_verifier_session_text,
)
from src.harness.contracts import EnvExecWorkload, RawState

DEFAULT_HARBOR_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "harbor_config.toml"
)
DEFAULT_TASK_OVERRIDES_DIR = Path(__file__).resolve().parents[2] / "task_overrides"
VERIFIER_CONTEXT_DOCKERFILE_TEMPLATE_PATH = (
    Path(__file__).resolve().parent / "verifier_context.Dockerfile"
)
_VERIFIER_IMAGE_BUILD_LOCKS: dict[str, asyncio.Lock] = {}
_VERIFIER_CONTEXT_LOCKS: dict[str, asyncio.Lock] = {}

logger = logging.getLogger(__name__)

# Prepended to every generated bootstrap script. Every line is best-effort
# ("|| true") so a minimal image (no procps, no curl) never hard-fails; the
# cache probes use bash /dev/tcp since there is no curl. Three concerns:
#
#   1. Locks: an exec timeout kills the docker client, not the in-container
#      apt-get, so its orphaned lock makes the retry fail `Could not get lock`
#      (exit 100). Clear stale locks; cap apt timeouts so a dead mirror fails
#      fast instead of hanging until the bootstrap timeout.
#   2. apt cache (optional): apt-cacher-ng on host.docker.internal:3142
#      (setup-apt-cache.sh) -> proxy repeat `python3` installs locally, else the
#      direct mirror.
#   3. PyPI cache (optional): verifiers `uv run` re-pull torch+CUDA (~2GB) every
#      verify; proxpi on :3141 (setup-pypi-cache.sh) -> point uv/pip there, else
#      direct PyPI. apt's proxy can't cache PyPI (it's HTTPS), so it's separate.
_BOOTSTRAP_PREAMBLE = """\
set -eu
pkill -9 -x apt 2>/dev/null || true
pkill -9 -x apt-get 2>/dev/null || true
pkill -9 -x dpkg 2>/dev/null || true
rm -f /var/lib/apt/lists/lock 2>/dev/null || true
rm -f /var/lib/dpkg/lock /var/lib/dpkg/lock-frontend 2>/dev/null || true
rm -f /var/cache/apt/archives/lock 2>/dev/null || true
# Some published task images ship a hermetic shim at /usr/bin/apt-get whose
# `update` subcommand is a no-op (exit 0, fetches nothing), keeping the real
# binary at apt-get.real. That silently breaks any setup/verify step which
# installs a package the image did not pre-bake (e.g. git-multibranch's verify
# installs `expect`). Restore the real apt so it works normally through our
# apt-cacher-ng proxy. No-op on images without the shim.
[ -e /usr/bin/apt-get.real ] && cp -f /usr/bin/apt-get.real /usr/bin/apt-get 2>/dev/null || true
mkdir -p /etc/apt/apt.conf.d 2>/dev/null || true
cat > /etc/apt/apt.conf.d/99-harness-bootstrap 2>/dev/null <<'EOF' || true
Acquire::Retries "3";
Acquire::http::Timeout "30";
Acquire::https::Timeout "30";
DPkg::Lock::Timeout "60";
EOF
if timeout 2 bash -c ': < /dev/tcp/host.docker.internal/3142' 2>/dev/null; then
printf 'Acquire::http::Proxy "http://host.docker.internal:3142";\\n' > /etc/apt/apt.conf.d/00-harness-apt-cache 2>/dev/null || true
fi
if timeout 2 bash -c ': < /dev/tcp/host.docker.internal/3141' 2>/dev/null; then
mkdir -p /etc/uv 2>/dev/null || true
cat > /etc/uv/uv.toml 2>/dev/null <<'EOF' || true
[[index]]
url = "http://host.docker.internal:3141/index/"
default = true
EOF
cat > /etc/pip.conf 2>/dev/null <<'EOF' || true
[global]
index-url = http://host.docker.internal:3141/index/
trusted-host = host.docker.internal
EOF
fi
"""


def _is_command_timeout(exc: Exception) -> bool:
    # Harbor's docker/gke backends signal a per-command timeout only as a
    # RuntimeError whose message contains "timed out"; there is no typed
    # timeout exception to match on.
    return isinstance(exc, RuntimeError) and "timed out" in str(exc).lower()


# High trial concurrency exposes host CPU contention from libraries/build tools
# that fan out inside each container. Cap those defaults at the Harbor boundary
# from the declared task CPU budget so agent and verifier exec share one policy.
_CPU_THREAD_ENV_KEYS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "CARGO_BUILD_JOBS",
    "GOMAXPROCS",
    "CMAKE_BUILD_PARALLEL_LEVEL",
)


def _cpu_resource_env(cpus: int | None) -> dict[str, str]:
    if cpus is None:
        return {"TOKENIZERS_PARALLELISM": "false"}
    cap = str(cpus)
    return {
        **{key: cap for key in _CPU_THREAD_ENV_KEYS},
        "MAKEFLAGS": f"-j{cap}",
        "TOKENIZERS_PARALLELISM": "false",
    }


# OrbStack -- and some other Docker setups -- inject a `no_proxy`/`NO_PROXY` into
# every container listing OrbStack's internal IPv6 ULA range (e.g. `::1` and
# `fd07:...::/64`). httpx cannot parse a bare IPv6 entry in no_proxy: it splits
# each entry on ':' to peel off a port, then reads the address bytes as the port
# and raises `httpx.InvalidURL: Invalid port: '...'` at Client construction. Any
# task that builds an httpx client then crashes before its first request -- most
# commonly huggingface_hub v1.x (pulled in by `datasets`), so every HF download
# dies and silently falls back to the offline cache, surfacing as a misleading
# "cache not found". The injection is intentional and OrbStack rewrites
# ~/.docker/config.json on every engine restart, so a host-side edit does not
# stick; stripping the IPv6 entries from no_proxy in each exec's env instead is
# restart-proof and a no-op on hosts that inject no IPv6.
#   httpx fix (open):     https://github.com/encode/httpx/pull/3741
#                         (encode/httpx Issues are disabled; the PR closes the
#                          original report #3221)
#   same bug in requests: https://github.com/psf/requests/issues/6313
#   OrbStack injection:   https://github.com/orbstack/orbstack/issues/2449
_NO_PROXY_ENV_KEYS = ("no_proxy", "NO_PROXY")


def _strip_ipv6_no_proxy(value: str) -> str:
    # A no_proxy entry is a hostname, domain suffix, IPv4 address, IPv4 CIDR, or
    # host:port -- none carry more than one ':'. An entry with two or more colons
    # is therefore an IPv6 address or CIDR, which is exactly what trips httpx.
    # Drop those and keep everything else in its original order.
    return ",".join(token for token in value.split(",") if token.count(":") < 2)


def _directory_content_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        relative_path = path.relative_to(directory).as_posix().encode()
        digest.update(relative_path)
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"L")
            digest.update(str(path.readlink()).encode())
        elif path.is_file():
            digest.update(b"F")
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
        elif path.is_dir():
            digest.update(b"D")
    return digest.hexdigest()


def _string_hash(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _trial_log_mounts(
    trial_paths: TrialPaths,
    task: Task,
) -> list[ServiceVolumeConfig]:
    env_paths = EnvironmentPaths.for_os(task.config.environment.os)
    return [
        {
            "type": "bind",
            "source": str(trial_paths.agent_dir.resolve()),
            "target": str(env_paths.agent_dir),
        },
        {
            "type": "bind",
            "source": str(trial_paths.verifier_dir.resolve()),
            "target": str(env_paths.verifier_dir),
        },
        {
            "type": "bind",
            "source": str(trial_paths.artifacts_dir.resolve()),
            "target": str(env_paths.artifacts_dir),
        },
    ]


def _verifier_log_mounts(
    trial_paths: TrialPaths,
    verifier_env_config: TaskEnvironmentConfig,
) -> list[ServiceVolumeConfig]:
    env_paths = EnvironmentPaths.for_os(verifier_env_config.os)
    return [
        {
            "type": "bind",
            "source": str(trial_paths.verifier_dir.resolve()),
            "target": str(env_paths.verifier_dir),
        },
    ]


@dataclass
class _ResourceCappedEnvironment:
    environment: BaseEnvironment
    resource_env: dict[str, str]

    def __getattr__(self, name: str):
        return getattr(self.environment, name)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        merged_env = dict(self.resource_env)
        if env is not None:
            merged_env.update(env)
        return await self.environment.exec(
            command=command,
            cwd=cwd,
            env=merged_env,
            timeout_sec=timeout_sec,
            user=user,
        )


@dataclass
class SeparateVerifierSession:
    env_config: TaskEnvironmentConfig
    build_context: Path | None


VerifierSession = Verifier | SeparateVerifierSession


@dataclass
class HarborSession:
    task: Task
    trial_paths: TrialPaths
    environment: _ResourceCappedEnvironment
    raw_environment: BaseEnvironment
    verifier_session: VerifierSession
    working_dir: str | None = None


class HarborConfig(BaseModel):
    """Shared Harbor settings.

    Task discovery fields are consumed by ``TaskDirectoryResolver``;
    ``Harbor`` receives an already-resolved task directory for each runtime
    session.
    """

    model_config = ConfigDict(extra="forbid")

    experiments_dir: Path = Path("experiments")
    verifier_contexts_dir: Path | None = None
    dataset_name: str = "terminal-bench"
    dataset_version: str | None = None
    task_overrides_dir: Path | None = DEFAULT_TASK_OVERRIDES_DIR
    bootstrap_commands: tuple[str, ...] = ()
    bootstrap_timeout_sec: int = 600
    shared_verifier_task_names: tuple[str, ...] = ()
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)

    @model_validator(mode="after")
    def resolve_paths(self) -> "HarborConfig":
        self.experiments_dir = self.experiments_dir.resolve()
        if self.verifier_contexts_dir is None:
            self.verifier_contexts_dir = self.experiments_dir / "_verifier_contexts"
        self.verifier_contexts_dir = self.verifier_contexts_dir.resolve()
        if self.task_overrides_dir is not None:
            self.task_overrides_dir = self.task_overrides_dir.resolve()
        return self

    @property
    def verifier_context_cache_dir(self) -> Path:
        if self.verifier_contexts_dir is None:
            raise RuntimeError("verifier_contexts_dir was not resolved")
        return self.verifier_contexts_dir

    @classmethod
    def from_toml(
        cls,
        path: str | Path = DEFAULT_HARBOR_CONFIG_PATH,
    ) -> "HarborConfig":
        return cls.model_validate(tomllib.loads(Path(path).read_text()))


class Harbor:
    def __init__(
        self,
        config: HarborConfig,
        *,
        task_name: str,
        task_dir: Path,
        exec_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self.config = config
        self.task_name = task_name
        self._task_dir = task_dir.resolve()
        self._session: HarborSession | None = None
        self._trial_dir: Path | None = None
        self._verifier_stdout_path: Path | None = None
        # Optional run-scoped gate, shared across the panel's Harbor instances,
        # bounding heavyweight container work (reset/startup, run, verify).
        # Cheap harness-generated file/list/search/edit commands use workload
        # "light" and bypass this gate, so they do not queue behind compiles or
        # long verifiers. None => ungated (single-trial use, tests).
        self._exec_semaphore = exec_semaphore

    @property
    def session(self) -> HarborSession:
        session = self._session
        if session is None:
            raise RuntimeError("Call `reset()` before using Harbor session methods.")
        return session

    @property
    def trial_dir(self) -> str | None:
        if self._trial_dir is None:
            return None
        return str(self._trial_dir)

    @property
    def verifier_stdout_path(self) -> str | None:
        if self._verifier_stdout_path is None:
            return None
        return str(self._verifier_stdout_path)

    def _agent_log_path(self) -> Path:
        return self.session.trial_paths.agent_dir / "exec.log"

    def _bootstrap_log_dir(self) -> Path:
        return self.session.trial_paths.trial_dir / "bootstrap"

    def _append_agent_log(
        self,
        *,
        command: str,
        result: ExecResult,
        cwd: str | None,
    ) -> None:
        log_lines = [f"$ {command}\n"]
        if cwd is not None:
            log_lines.append(f"cwd={cwd}\n")
        log_lines.append(f"return_code={result.return_code}\n")
        # Skip stdout/stderr bodies for clean exits — successful commands
        # dominate by count, and their raw output is rarely needed for
        # post-hoc diagnosis. The action_chosen+env_step_completed pair in
        # steps.jsonl preserves the structural record.
        if result.return_code != 0:
            log_lines.extend(
                [
                    "[stdout]\n",
                    f"{result.stdout or ''}\n",
                    "[stderr]\n",
                    f"{result.stderr or ''}\n",
                ]
            )
        self._agent_log_path().open("a").write("".join(log_lines))

    def _attach_session(
        self,
        *,
        task: Task,
        trial_paths: TrialPaths,
        harbor_environment: BaseEnvironment,
    ) -> None:
        capped_environment = _ResourceCappedEnvironment(
            harbor_environment,
            _cpu_resource_env(task.config.environment.cpus),
        )
        verifier_env_config = resolve_effective_verifier_env_config(
            task.config,
            step_cfg=None,
        )
        verifier_build_context: Path | None = task.paths.tests_dir
        # Default to fresh verifier environments; keep stateful service tasks
        # shared so their agent-started localhost services remain testable.
        if (
            self.task_name not in self.config.shared_verifier_task_names
            and task.config.verifier.environment_mode is None
            and task.config.verifier.environment is None
        ):
            verifier_env_config = task.config.environment.model_copy(
                deep=True,
                update={"docker_image": None},
            )
            verifier_build_context = None
        elif (
            self.task_name not in self.config.shared_verifier_task_names
            and task.config.verifier.environment_mode is None
        ):
            verifier_env_config = (
                verifier_env_config or task.config.environment.model_copy(deep=True)
            )
        verifier_session: VerifierSession
        if verifier_env_config is None:
            verifier_session = Verifier(
                task=task,
                trial_paths=trial_paths,
                environment=capped_environment,
            )
        else:
            verifier_session = SeparateVerifierSession(
                env_config=verifier_env_config,
                build_context=verifier_build_context,
            )
        self._session = HarborSession(
            task=task,
            trial_paths=trial_paths,
            environment=capped_environment,
            raw_environment=harbor_environment,
            verifier_session=verifier_session,
        )

    async def _sanitize_no_proxy(
        self,
        environment: _ResourceCappedEnvironment,
    ) -> None:
        """Strip httpx-breaking IPv6 entries from the container's injected
        `no_proxy`/`NO_PROXY` (see `_strip_ipv6_no_proxy`).

        Reads the values the container was started with and, for any that carry
        an IPv6 entry, pins a cleaned override into ``resource_env`` so every
        later exec (agent actions and the in-container verifier) inherits it.
        Best-effort: a probe failure leaves the env untouched -- the bug is
        benign for tasks that never construct an httpx client.
        """
        probe = "; ".join(f'printf "%s\\n" "${{{key}-}}"' for key in _NO_PROXY_ENV_KEYS)
        try:
            result = await environment.exec(command=probe, timeout_sec=10)
        except Exception as exc:  # noqa: BLE001 -- best-effort, never fail a trial
            logger.debug("no_proxy probe skipped for %s: %s", self.task_name, exc)
            return
        if result.return_code != 0 or result.stdout is None:
            return
        for key, raw in zip(_NO_PROXY_ENV_KEYS, result.stdout.splitlines()):
            if not raw:
                continue
            cleaned = _strip_ipv6_no_proxy(raw)
            if cleaned != raw:
                environment.resource_env[key] = cleaned

    async def _bootstrap_environment(self) -> None:
        commands = self.config.bootstrap_commands
        if not commands:
            return

        bootstrap_dir = self._bootstrap_log_dir()
        bootstrap_dir.mkdir(parents=True, exist_ok=True)
        script_path = bootstrap_dir / "bootstrap.sh"
        script_path.write_text(_BOOTSTRAP_PREAMBLE + "\n".join(commands) + "\n")
        remote_script_path = "/tmp/harbor-bootstrap.sh"
        await self.session.environment.upload_file(
            source_path=script_path,
            target_path=remote_script_path,
        )

        timeout_sec = self.config.bootstrap_timeout_sec

        async def run_bootstrap() -> ExecResult:
            return await self.session.environment.exec(
                command=f"bash {remote_script_path}",
                env={"DEBIAN_FRONTEND": "noninteractive"},
                timeout_sec=timeout_sec,
            )

        def _log_retry(retry: int, exc: Exception) -> None:
            logger.warning(
                "bootstrap timed out for task %s; retrying (%d/%d): %s",
                self.task_name,
                retry,
                INFRA_RETRY_BUDGET,
                exc,
            )

        try:
            result = await retry_transient(
                run_bootstrap,
                is_transient=_is_command_timeout,
                on_retry=_log_retry,
            )
        except RuntimeError as exc:
            # Harbor discards the command's partial output when it times out, so
            # the only debuggable artifact we can leave for a timed-out bootstrap
            # is a marker recording that it timed out after the budget.
            if _is_command_timeout(exc):
                (bootstrap_dir / "status.txt").write_text(
                    f"timed out after {timeout_sec}s (retries exhausted): {exc}\n"
                )
            raise

        (bootstrap_dir / "return-code.txt").write_text(str(result.return_code))
        if result.stdout is not None:
            (bootstrap_dir / "stdout.txt").write_text(result.stdout)
        if result.stderr is not None:
            (bootstrap_dir / "stderr.txt").write_text(result.stderr)
        if result.return_code != 0:
            raise RuntimeError(
                f"Harbor bootstrap failed with exit code {result.return_code}. "
                f"See logs in {bootstrap_dir}"
            )

    async def reset(self) -> RawState:
        from harbor.environments.factory import EnvironmentFactory

        if self._session is not None:
            await self.close()

        await self._docker_cleanup().cleanup_stale_docker_compose_projects()

        run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
        session_id = f"{self.task_name}__{run_id}"
        trial_paths = TrialPaths(
            trial_dir=self.config.experiments_dir / self.task_name / run_id
        )
        trial_paths.mkdir()
        self._trial_dir = trial_paths.trial_dir
        self._verifier_stdout_path = trial_paths.test_stdout_path
        task = Task(task_dir=self._task_dir)
        harbor_environment = EnvironmentFactory.create_environment_from_config(
            config=self.config.environment,
            environment_dir=task.paths.environment_dir,
            environment_name=task.name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task.config.environment,
            mounts=_trial_log_mounts(trial_paths, task),
        )
        try:
            async with self._env_gate():
                await harbor_environment.start(
                    force_build=self.config.environment.force_build
                )
                self._attach_session(
                    task=task,
                    trial_paths=trial_paths,
                    harbor_environment=harbor_environment,
                )
                await self._sanitize_no_proxy(self.session.environment)
                await self._bootstrap_environment()
                working_dir = await self._detect_working_dir()
                self.session.working_dir = working_dir
            return RawState(
                instruction=task.instruction,
                working_dir=working_dir,
            )
        except Exception:
            await self._docker_cleanup().stop_environment(harbor_environment)
            raise

    def _env_gate(self, workload: EnvExecWorkload = "heavy"):
        """Per-call gate around heavyweight container CPU work."""
        if workload == "light" or self._exec_semaphore is None:
            return contextlib.nullcontext()
        return self._exec_semaphore

    async def exec(
        self,
        *,
        command: str,
        cwd: str | None = None,
        timeout_sec: int | None = None,
        workload: EnvExecWorkload = "heavy",
    ) -> RawState:
        async with self._env_gate(workload):
            try:
                result = await self.session.environment.exec(
                    command=command,
                    cwd=cwd,
                    timeout_sec=timeout_sec,
                )
            except RuntimeError as exc:
                # Harbor's docker/gke backends raise RuntimeError on per-command
                # timeout instead of returning a failed ExecResult. Convert into
                # a failed observation so the agent sees the timeout and can pick
                # a longer `timeout_sec` or a different approach. Other
                # RuntimeErrors (e.g. container crash) still surface as
                # trial-fatal.
                if not _is_command_timeout(exc):
                    raise
                return RawState(
                    return_code=124,
                    stdout=None,
                    stderr=str(exc),
                    passed=False,
                )
            except ValueError as exc:
                # The agent can emit action arguments that cannot be marshalled
                # into a container command -- most concretely a write_file/run
                # whose content carries an embedded NUL, which Python's
                # subprocess layer rejects with ValueError("embedded null byte").
                # That is the agent's own input, not an infra failure, so surface
                # it as a failed observation it can react to next step instead of
                # a trial-fatal crash.
                return RawState(
                    return_code=1,
                    stdout=None,
                    stderr=f"invalid command (cannot execute): {exc}",
                    passed=False,
                )
            self._append_agent_log(command=command, result=result, cwd=cwd)
            return RawState(
                return_code=result.return_code,
                stdout=result.stdout,
                stderr=result.stderr,
                passed=result.return_code == 0,
            )

    def _separate_verifier_session_id(self) -> str:
        run_id = self.session.trial_paths.trial_dir.name
        raw = f"{self.task_name}__{run_id}__verifier"
        safe = _safe_verifier_session_text(raw)
        if len(safe) <= _MAX_VERIFIER_ENV_SESSION_ID_LEN:
            return safe

        compact_prefix = _compact_verifier_task_session_prefix(self.task_name)
        if compact_prefix is None:
            return safe
        return f"{compact_prefix}{run_id}__verifier"

    def _verifier_cache_image_name(self, build_context: Path) -> str:
        from harbor.environments.docker.docker import _sanitize_docker_image_name

        repository = _sanitize_docker_image_name(
            f"hb-verifier-cache__{self.session.task.name}"
        )
        digest = _directory_content_hash(build_context)[:16]
        return f"{repository}:{digest}"

    def _should_cache_verifier_image(
        self,
        verifier_env_config: TaskEnvironmentConfig,
        build_context: Path,
    ) -> bool:
        return (
            self.config.environment.type == EnvironmentType.DOCKER
            and self.config.environment.import_path is None
            and verifier_env_config.docker_image is None
            and (build_context / "Dockerfile").is_file()
            and not (build_context / "docker-compose.yaml").exists()
        )

    async def _verifier_build_context(
        self,
        verifier_session: SeparateVerifierSession,
    ) -> Path:
        if verifier_session.build_context is not None:
            return verifier_session.build_context
        return await self._ensure_generated_verifier_context()

    def _generated_verifier_context_dir(self) -> Path:
        base_image = self.session.task.config.environment.docker_image
        working_dir = self.session.working_dir
        if base_image is None or working_dir is None:
            raise RuntimeError("Generated separate verifier context is not available")

        digest = _string_hash(
            "auto-separated-verifier-v1",
            base_image,
            working_dir,
            _directory_content_hash(self.session.task.paths.tests_dir),
        )[:16]
        return self.config.verifier_context_cache_dir / self.session.task.name / digest

    async def _ensure_generated_verifier_context(self) -> Path:
        base_image = self.session.task.config.environment.docker_image
        working_dir = self.session.working_dir
        if base_image is None or working_dir is None:
            raise RuntimeError("Generated separate verifier context is not available")

        context_dir = self._generated_verifier_context_dir()
        lock = _VERIFIER_CONTEXT_LOCKS.setdefault(str(context_dir), asyncio.Lock())
        async with lock:
            if (context_dir / "Dockerfile").exists():
                return context_dir

            context_dir.parent.mkdir(parents=True, exist_ok=True)
            if context_dir.exists():
                shutil.rmtree(context_dir)
            shutil.copytree(self.session.task.paths.tests_dir, context_dir)
            (context_dir / "Dockerfile").write_text(
                VERIFIER_CONTEXT_DOCKERFILE_TEMPLATE_PATH.read_text().format(
                    base_image=base_image,
                    working_dir=working_dir,
                )
            )
        return context_dir

    async def _docker_image_exists(self, image_name: str) -> bool:
        try:
            await self._run_docker_cli(
                ["image", "inspect", "--format", "{{.Id}}", image_name],
                failure_context="Docker image inspect failed",
            )
        except RuntimeError:
            return False
        return True

    async def _ensure_cached_verifier_image(
        self,
        *,
        image_name: str,
        build_context: Path,
    ) -> None:
        lock = _VERIFIER_IMAGE_BUILD_LOCKS.setdefault(image_name, asyncio.Lock())
        async with lock:
            if await self._docker_image_exists(image_name):
                return
            await self._run_docker_cli(
                ["build", "--tag", image_name, str(build_context.resolve())],
                failure_context="Docker image build failed",
            )

    async def _verifier_env_config_with_cached_image(
        self,
        verifier_env_config: TaskEnvironmentConfig,
        build_context: Path,
    ) -> TaskEnvironmentConfig:
        if not self._should_cache_verifier_image(verifier_env_config, build_context):
            return verifier_env_config

        image_name = self._verifier_cache_image_name(build_context)
        await self._ensure_cached_verifier_image(
            image_name=image_name,
            build_context=build_context,
        )
        return verifier_env_config.model_copy(update={"docker_image": image_name})

    def _artifact_handler(self):
        from harbor.trial.artifact_handler import ArtifactHandler

        return ArtifactHandler(
            artifacts=self.session.task.config.artifacts,
            logger=logger,
        )

    def _separate_verifier_extra_artifacts(
        self,
        verifier_session: SeparateVerifierSession,
    ) -> tuple[str, ...]:
        if verifier_session.build_context is not None:
            return ()
        working_dir = self.session.working_dir
        assert working_dir is not None
        return (working_dir,)

    async def _collect_separate_verifier_artifacts(
        self,
        verifier_session: SeparateVerifierSession,
    ) -> None:
        agent_env_paths = EnvironmentPaths.for_os(
            self.session.task.config.environment.os
        )
        await self._artifact_handler().download_artifacts(
            self.session.environment,
            self.session.trial_paths.artifacts_dir,
            source_artifacts_dir=agent_env_paths.artifacts_dir,
            artifacts=self._separate_verifier_extra_artifacts(verifier_session),
        )

    async def _upload_separate_verifier_artifacts(
        self,
        verifier_session: SeparateVerifierSession,
        verifier_environment: _ResourceCappedEnvironment,
    ) -> None:
        agent_env_paths = EnvironmentPaths.for_os(
            self.session.task.config.environment.os
        )
        verifier_env_paths = EnvironmentPaths.for_os(verifier_environment.os)
        await self._artifact_handler().upload_artifacts(
            verifier_environment,
            artifacts_dir=self.session.trial_paths.artifacts_dir,
            source_artifacts_dir=agent_env_paths.artifacts_dir,
            target_artifacts_dir=verifier_env_paths.artifacts_dir,
            artifacts=self._separate_verifier_extra_artifacts(verifier_session),
        )

    async def _apply_verifier_network_preamble(
        self,
        verifier_environment: _ResourceCappedEnvironment,
    ) -> None:
        """Give the separate verifier container the same apt/pip cache + retry
        and timeout policy the agent container already gets via
        `_BOOTSTRAP_PREAMBLE`.

        The task's verifier `test.sh` is immutable and runs its own toolchain
        install (`apt-get`, `uv`, `pip`). Without this it goes direct to live
        mirrors with no timeout, so a flaky mirror can stall the entire trial
        budget and the grade never runs. The preamble only writes proxy and
        retry/timeout config -- nothing task- or run-derived -- so verifier
        isolation is unchanged; it is also a no-op when the caches are
        unreachable (the proxy lines are guarded by a reachability probe).
        Best-effort: a failure here must never fail the grade, so the verifier
        falls back to the previous direct-fetch behaviour.
        """
        try:
            # Harbor runs every Linux exec under `bash -c`, so the preamble's
            # heredocs and `/dev/tcp` reachability probe work as-is.
            await verifier_environment.exec(
                command=_BOOTSTRAP_PREAMBLE,
                user="root",
                timeout_sec=30,
            )
        except Exception as exc:  # noqa: BLE001 -- best-effort network hardening
            logger.debug(
                "verifier network preamble skipped for %s: %s",
                self.session.task.name,
                exc,
            )

    @contextlib.asynccontextmanager
    async def _separate_verifier_environment(
        self,
        verifier_session: SeparateVerifierSession,
    ) -> AsyncGenerator[_ResourceCappedEnvironment, None]:
        from harbor.environments.factory import EnvironmentFactory

        verifier_env_config = verifier_session.env_config
        build_context = await self._verifier_build_context(verifier_session)
        verifier_env_config = await self._verifier_env_config_with_cached_image(
            verifier_env_config,
            build_context,
        )
        raw_environment = EnvironmentFactory.create_environment_from_config(
            config=self.config.environment.model_copy(
                update={"extra_docker_compose": []}
            ),
            environment_dir=build_context,
            environment_name=self.session.task.name,
            session_id=self._separate_verifier_session_id(),
            trial_paths=self.session.trial_paths,
            task_env_config=verifier_env_config,
            mounts=_verifier_log_mounts(
                self.session.trial_paths,
                verifier_env_config,
            ),
        )
        try:
            await raw_environment.start(force_build=False)
            verifier_environment = _ResourceCappedEnvironment(
                raw_environment,
                _cpu_resource_env(verifier_env_config.cpus),
            )
            await self._sanitize_no_proxy(verifier_environment)
            yield verifier_environment
        finally:
            try:
                await asyncio.shield(
                    self._docker_cleanup().stop_environment(raw_environment)
                )
            except Exception as exc:
                logger.debug("Failed to stop separate verifier environment: %s", exc)

    async def _verify_in_separate_environment(
        self,
        verifier_session: SeparateVerifierSession,
    ) -> VerifierResult:
        await self._collect_separate_verifier_artifacts(verifier_session)
        async with self._separate_verifier_environment(
            verifier_session
        ) as verifier_environment:
            with verifier_environment.with_default_user(
                self.session.task.config.verifier.user
            ):
                env_paths = EnvironmentPaths.for_os(verifier_environment.os)
                await verifier_environment.empty_dirs(
                    [env_paths.verifier_dir],
                    chmod=True,
                )
                await self._upload_separate_verifier_artifacts(
                    verifier_session,
                    verifier_environment,
                )
                await self._apply_verifier_network_preamble(verifier_environment)
                verifier = Verifier(
                    task=self.session.task,
                    trial_paths=self.session.trial_paths,
                    environment=verifier_environment,
                    skip_tests_upload=True,
                )
                return await verifier.verify()

    async def verify(self) -> RawState:
        async with self._env_gate():
            verifier_session = self.session.verifier_session
            if isinstance(verifier_session, SeparateVerifierSession):
                verifier_result = await self._verify_in_separate_environment(
                    verifier_session
                )
            else:
                verifier_result = await verifier_session.verify()
            rewards = verifier_result.rewards
            raw_reward = 0.0 if rewards is None else rewards.get("reward", 0.0)
            reward = 0.0 if raw_reward is None else float(raw_reward)
            stdout_path = self.session.trial_paths.test_stdout_path
            stderr_path = self.session.trial_paths.test_stderr_path
            return RawState(
                reward=reward,
                done=True,
                passed=reward > 0.0,
                stdout=stdout_path.read_text() if stdout_path.exists() else None,
                stderr=stderr_path.read_text() if stderr_path.exists() else None,
            )

    async def _detect_working_dir(self) -> str:
        result = await self.session.environment.exec(command="pwd")
        if result.return_code != 0 or result.stdout is None:
            raise RuntimeError("failed to detect environment working directory")
        return result.stdout.strip()

    async def _run_docker_cli(
        self,
        args: list[str],
        *,
        failure_context: str = "Docker command failed",
    ) -> ExecResult:
        process = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout_bytes, _ = await process.communicate()
        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else None
        result = ExecResult(
            stdout=stdout,
            stderr=None,
            return_code=process.returncode or 0,
        )
        if result.return_code != 0:
            raise RuntimeError(
                f"{failure_context}. "
                f"Command: docker {' '.join(args)}. "
                f"Return code: {result.return_code}. "
                f"Output: {result.stdout}."
            )
        return result

    def _docker_cleanup(self) -> DockerCleanup:
        return DockerCleanup(
            task_name=self.task_name,
            environment_config=self.config.environment,
            run_docker_cli=lambda args, **kwargs: self._run_docker_cli(
                args,
                **kwargs,
            ),
            logger=logger,
        )

    async def close(self) -> None:
        if self._session is None:
            self._trial_dir = None
            self._verifier_stdout_path = None
            return

        try:
            await self._docker_cleanup().stop_environment(self._session.raw_environment)
        finally:
            self._session = None
            self._trial_dir = None
            self._verifier_stdout_path = None


class TaskDirectoryResolver:
    """Owns task-name to task-directory resolution before Harbor starts."""

    def __init__(self, config: HarborConfig) -> None:
        self.config = config

    async def resolve(self, task_names: list[str]) -> dict[str, Path]:
        resolved: dict[str, Path] = {}
        needs_registry: list[str] = []
        for task_name in dict.fromkeys(task_names):
            override = self._resolve_task_override(task_name)
            if override is not None:
                resolved[task_name] = override
            else:
                needs_registry.append(task_name)

        if not needs_registry:
            return resolved

        from harbor.registry.client.factory import RegistryClientFactory
        from harbor.tasks.client import TaskClient

        registry_client = RegistryClientFactory.create()
        dataset_ref = self.config.dataset_name
        if self.config.dataset_version is not None:
            dataset_ref = f"{dataset_ref}@{self.config.dataset_version}"
        metadata = await registry_client.get_dataset_metadata(dataset_ref)
        task_ids = list(metadata.task_ids)
        pending_ids = []
        for task_name in needs_registry:
            matches = [
                task_id for task_id in task_ids if task_id.get_name() == task_name
            ]
            if not matches:
                raise ValueError(
                    f"Task `{task_name}` was not found in dataset "
                    f"`{self.config.dataset_name}`."
                )
            if len(matches) != 1:
                raise ValueError(
                    f"Task `{task_name}` matched multiple registry entries in "
                    f"`{self.config.dataset_name}`."
                )
            pending_ids.append((task_name, matches[0]))

        task_client = TaskClient()
        downloaded = await task_client.download_tasks(
            [task_id for _, task_id in pending_ids]
        )
        for (task_name, _), path in zip(pending_ids, downloaded.paths, strict=True):
            resolved[task_name] = path
        return resolved

    def _resolve_task_override(self, task_name: str) -> Path | None:
        overrides_root = self.config.task_overrides_dir
        if overrides_root is None:
            return None
        override_dir = overrides_root / task_name
        if not override_dir.exists():
            return None

        if not Task.is_valid_dir(override_dir):
            raise RuntimeError(f"local task override is invalid: {override_dir}")
        return override_dir
