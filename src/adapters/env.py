"""Harbor environment adapter."""

from __future__ import annotations

import asyncio
import logging
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.task.paths import TaskPaths
from harbor.models.task.task import Task
from harbor.models.trial.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from harbor.verifier.verifier import Verifier
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Heavy harbor imports deferred to first use: `harbor.environments.factory` pulls
# in all environment backends (daytona, e2b, gke, modal, runloop -- ~2.6s cold);
# `harbor.registry.client.factory` pulls in supabase (~0.35s). Moving them
# behind first call to reset() / download path keeps `uv run exp` cold-start
# fast and shaves cost from invocations that never reach those paths.

from src.harness.contracts import RawState

DEFAULT_HARBOR_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "harbor_config.toml"
)
DEFAULT_TASK_OVERRIDES_DIR = Path(__file__).resolve().parents[2] / "task_overrides"

logger = logging.getLogger(__name__)


@dataclass
class HarborSession:
    task: Task
    trial_paths: TrialPaths
    environment: BaseEnvironment
    verifier: Verifier


class HarborConfig(BaseModel):
    """Shared Harbor settings.

    Task discovery fields are consumed by ``TaskDirectoryResolver``;
    ``Harbor`` receives an already-resolved task directory for each runtime
    session.
    """

    model_config = ConfigDict(extra="forbid")

    experiments_dir: Path = Path("experiments")
    dataset_name: str = "terminal-bench"
    dataset_version: str | None = None
    task_overrides_dir: Path | None = DEFAULT_TASK_OVERRIDES_DIR
    bootstrap_commands: tuple[str, ...] = ()
    bootstrap_timeout_sec: int = 600
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)

    @model_validator(mode="after")
    def resolve_paths(self) -> "HarborConfig":
        self.experiments_dir = self.experiments_dir.resolve()
        if self.task_overrides_dir is not None:
            self.task_overrides_dir = self.task_overrides_dir.resolve()
        return self

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
    ) -> None:
        self.config = config
        self.task_name = task_name
        self._task_dir = task_dir.resolve()
        self._session: HarborSession | None = None
        self._trial_dir: Path | None = None
        self._verifier_stdout_path: Path | None = None

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
        self._session = HarborSession(
            task=task,
            trial_paths=trial_paths,
            environment=harbor_environment,
            verifier=Verifier(
                task=task,
                trial_paths=trial_paths,
                environment=harbor_environment,
            ),
        )

    async def _bootstrap_environment(self) -> None:
        commands = self.config.bootstrap_commands
        if not commands:
            return

        bootstrap_dir = self._bootstrap_log_dir()
        bootstrap_dir.mkdir(parents=True, exist_ok=True)
        script_path = bootstrap_dir / "bootstrap.sh"
        script_path.write_text("set -eu\n" + "\n".join(commands) + "\n")
        remote_script_path = "/tmp/harbor-bootstrap.sh"
        await self.session.environment.upload_file(
            source_path=script_path,
            target_path=remote_script_path,
        )
        result = await self.session.environment.exec(
            command=f"bash {remote_script_path}",
            env={"DEBIAN_FRONTEND": "noninteractive"},
            timeout_sec=self.config.bootstrap_timeout_sec,
        )
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
        )
        try:
            await harbor_environment.start(
                force_build=self.config.environment.force_build
            )
            self._attach_session(
                task=task,
                trial_paths=trial_paths,
                harbor_environment=harbor_environment,
            )
            await self._bootstrap_environment()
            working_dir = await self._detect_working_dir()
            return RawState(
                instruction=task.instruction,
                working_dir=working_dir,
            )
        except Exception:
            await self._stop_environment(harbor_environment)
            raise

    async def exec(
        self,
        *,
        command: str,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> RawState:
        try:
            result = await self.session.environment.exec(
                command=command,
                cwd=cwd,
                timeout_sec=timeout_sec,
            )
        except RuntimeError as exc:
            # Harbor's docker/gke backends raise RuntimeError on per-command
            # timeout instead of returning a failed ExecResult. Convert into a
            # failed observation so the agent sees the timeout and can pick a
            # longer `timeout_sec` or a different approach. Match narrowly on
            # "timed out" so other RuntimeErrors (e.g. container crash) still
            # surface as trial-fatal.
            if "timed out" not in str(exc).lower():
                raise
            return RawState(
                return_code=124,
                stdout=None,
                stderr=str(exc),
                passed=False,
            )
        self._append_agent_log(command=command, result=result, cwd=cwd)
        return RawState(
            return_code=result.return_code,
            stdout=result.stdout,
            stderr=result.stderr,
            passed=result.return_code == 0,
        )

    async def verify(self) -> RawState:
        verifier_result = await self.session.verifier.verify()
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

    async def _run_docker_cli(self, args: list[str]) -> ExecResult:
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
                "Docker cleanup command failed. "
                f"Command: docker {' '.join(args)}. "
                f"Return code: {result.return_code}. "
                f"Output: {result.stdout}."
            )
        return result

    @staticmethod
    def _docker_compose_project_name(env: BaseEnvironment) -> str:
        # Must stay in sync with Harbor's private DockerEnvironment
        # `docker compose -p` derivation; drift makes label cleanup a no-op.
        return env.session_id.lower().replace(".", "-")

    async def _docker_ids_by_project_label(
        self,
        *,
        resource: str,
        project_name: str,
    ) -> list[str]:
        label = f"label=com.docker.compose.project={project_name}"
        args = [resource, "ls", "--quiet", "--filter", label]
        if resource == "container":
            args.insert(2, "--all")
        result = await self._run_docker_cli(args)
        return [line for line in (result.stdout or "").splitlines() if line]

    async def _cleanup_docker_project_by_label(self, project_name: str) -> None:
        container_ids = await self._docker_ids_by_project_label(
            resource="container",
            project_name=project_name,
        )
        if container_ids:
            await self._run_docker_cli(
                ["container", "rm", "--force", "--volumes", *container_ids]
            )

        volume_ids = await self._docker_ids_by_project_label(
            resource="volume",
            project_name=project_name,
        )
        if volume_ids:
            await self._run_docker_cli(["volume", "rm", "--force", *volume_ids])

        network_ids = await self._docker_ids_by_project_label(
            resource="network",
            project_name=project_name,
        )
        if network_ids:
            await self._run_docker_cli(["network", "rm", *network_ids])

    async def _stop_environment(self, env: BaseEnvironment) -> None:
        from harbor.environments.docker.docker import DockerEnvironment

        if self.config.environment.delete and isinstance(env, DockerEnvironment):
            try:
                await env._run_docker_compose_command(
                    ["down", "--volumes", "--remove-orphans"]
                )
            except RuntimeError as compose_error:
                project_name = self._docker_compose_project_name(env)
                try:
                    await self._cleanup_docker_project_by_label(project_name)
                    logger.warning(
                        "Docker compose down failed for project %r; "
                        "recovered via label fallback: %s",
                        project_name,
                        compose_error,
                    )
                except RuntimeError as fallback_error:
                    raise RuntimeError(
                        "Docker cleanup failed after compose down and "
                        f"project-label fallback for project {project_name!r}. "
                        f"Compose error: {compose_error}. "
                        f"Fallback error: {fallback_error}."
                    ) from compose_error
        else:
            await env.stop(delete=self.config.environment.delete)

    async def close(self) -> None:
        if self._session is None:
            self._trial_dir = None
            self._verifier_stdout_path = None
            return

        try:
            await self._stop_environment(self._session.environment)
        finally:
            self._session = None
            self._trial_dir = None
            self._verifier_stdout_path = None


class TaskDirectoryResolver:
    """Owns task-name to task-directory resolution before Harbor starts."""

    def __init__(self, config: HarborConfig) -> None:
        self.config = config

    def resolve(self, task_names: list[str]) -> dict[str, Path]:
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
        dataset = registry_client.get_dataset_spec(
            self.config.dataset_name,
            self.config.dataset_version,
        )
        task_client = TaskClient()
        pending_ids = []
        for task_name in needs_registry:
            matches = [task for task in dataset.tasks if task.name == task_name]
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
            pending_ids.append((task_name, matches[0].to_source_task_id()))

        downloaded = task_client.download_tasks(
            [source_id for _, source_id in pending_ids]
        )
        for (task_name, _), path in zip(pending_ids, downloaded, strict=True):
            resolved[task_name] = path
        return resolved

    def _resolve_task_override(self, task_name: str) -> Path | None:
        overrides_root = self.config.task_overrides_dir
        if overrides_root is None:
            return None
        override_dir = overrides_root / task_name
        if not override_dir.exists():
            return None

        if not TaskPaths(override_dir).is_valid():
            raise RuntimeError(f"local task override is invalid: {override_dir}")
        return override_dir
