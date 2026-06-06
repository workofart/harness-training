"""Docker cleanup helpers for the Harbor adapter."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from hashlib import sha1

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import EnvironmentConfig

_MAX_VERIFIER_ENV_SESSION_ID_LEN = 63
_HARBOR_RUN_ID_EXAMPLE = "20260603-120000-abcdef12"
_HARBOR_COMPOSE_PROJECT_SUFFIX_RE = re.compile(
    r"^\d{8}-\d{6}-[0-9a-f]{8}(?:__verifier)?$"
)
_TERMINAL_DOCKER_STATES = frozenset({"dead", "exited"})

_DockerCli = Callable[..., Awaitable[ExecResult]]


def _safe_verifier_session_text(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-._" else "_" for char in value)


def _compact_verifier_task_session_prefix(task_name: str) -> str | None:
    safe_task = _safe_verifier_session_text(task_name)
    if (
        len(f"{safe_task}__{_HARBOR_RUN_ID_EXAMPLE}__verifier")
        <= _MAX_VERIFIER_ENV_SESSION_ID_LEN
    ):
        return None

    digest = sha1(safe_task.encode()).hexdigest()[:8]
    suffix_len = len(f"__{digest}__{_HARBOR_RUN_ID_EXAMPLE}__verifier")
    task_prefix = safe_task[: _MAX_VERIFIER_ENV_SESSION_ID_LEN - suffix_len].rstrip(
        "-._"
    )
    return f"{task_prefix or digest}__{digest}__"


class DockerCleanup:
    def __init__(
        self,
        *,
        task_name: str,
        environment_config: EnvironmentConfig,
        run_docker_cli: _DockerCli,
        logger: logging.Logger,
    ) -> None:
        self.task_name = task_name
        self.environment_config = environment_config
        self._run_docker_cli = run_docker_cli
        self._logger = logger

    @staticmethod
    def docker_compose_project_name(env: BaseEnvironment) -> str:
        from harbor.environments.docker.docker import (
            _sanitize_docker_compose_project_name,
        )

        # Must stay in sync with Harbor's private DockerEnvironment
        # `docker compose -p` derivation; drift makes label cleanup a no-op.
        return _sanitize_docker_compose_project_name(env.session_id)

    def _docker_compose_task_project_prefix(self) -> str:
        from harbor.environments.docker.docker import (
            _sanitize_docker_compose_project_name,
        )

        return f"{_sanitize_docker_compose_project_name(self.task_name)}__"

    def _docker_compose_cleanup_project_prefixes(self) -> tuple[str, ...]:
        from harbor.environments.docker.docker import (
            _sanitize_docker_compose_project_name,
        )

        prefixes = [self._docker_compose_task_project_prefix()]
        compact_verifier_prefix = _compact_verifier_task_session_prefix(self.task_name)
        if compact_verifier_prefix is not None:
            prefixes.append(
                _sanitize_docker_compose_project_name(compact_verifier_prefix)
            )
        return tuple(prefixes)

    async def docker_ids_by_project_label(
        self,
        *,
        resource: str,
        project_name: str,
    ) -> list[str]:
        label = f"label=com.docker.compose.project={project_name}"
        args = [resource, "ls", "--quiet", "--filter", label]
        if resource == "container":
            args.insert(2, "--all")
        result = await self._run_docker_cli(
            args,
            failure_context="Docker cleanup command failed",
        )
        return [line for line in (result.stdout or "").splitlines() if line]

    async def cleanup_docker_project_by_label(self, project_name: str) -> None:
        container_ids = await self.docker_ids_by_project_label(
            resource="container",
            project_name=project_name,
        )
        if container_ids:
            await self._run_docker_cli(
                ["container", "rm", "--force", "--volumes", *container_ids],
                failure_context="Docker cleanup command failed",
            )

        volume_ids = await self.docker_ids_by_project_label(
            resource="volume",
            project_name=project_name,
        )
        if volume_ids:
            await self._run_docker_cli(
                ["volume", "rm", "--force", *volume_ids],
                failure_context="Docker cleanup command failed",
            )

        network_ids = await self.docker_ids_by_project_label(
            resource="network",
            project_name=project_name,
        )
        if network_ids:
            await self._run_docker_cli(
                ["network", "rm", *network_ids],
                failure_context="Docker cleanup command failed",
            )

    @staticmethod
    def docker_compose_project_label(labels: str) -> str:
        for label in labels.split(","):
            if label.startswith("com.docker.compose.project="):
                return label.split("=", 1)[1]
        raise ValueError("Docker compose project label is missing")

    def is_stale_cleanup_candidate_project(self, project_name: str) -> bool:
        for prefix in self._docker_compose_cleanup_project_prefixes():
            if project_name.startswith(prefix):
                suffix = project_name[len(prefix) :]
                return _HARBOR_COMPOSE_PROJECT_SUFFIX_RE.fullmatch(suffix) is not None
        return False

    async def cleanup_stale_docker_compose_projects(self) -> None:
        if (
            not self.environment_config.delete
            or self.environment_config.type != EnvironmentType.DOCKER
        ):
            return

        result = await self._run_docker_cli(
            [
                "container",
                "ls",
                "--all",
                "--filter",
                "label=com.docker.compose.project",
                "--format",
                "{{json .}}",
            ],
            failure_context="Docker stale container listing failed",
        )
        project_states: dict[str, set[str]] = {}
        for line in (result.stdout or "").splitlines():
            try:
                row = json.loads(line)
                project_name = self.docker_compose_project_label(row["Labels"])
                state = row["State"]
            except (json.JSONDecodeError, KeyError, ValueError):
                self._logger.debug(
                    "Skipping malformed Docker compose container row: %r", line
                )
                continue
            if self.is_stale_cleanup_candidate_project(project_name):
                project_states.setdefault(project_name, set()).add(state)

        for project_name, states in sorted(project_states.items()):
            if not states <= _TERMINAL_DOCKER_STATES:
                continue
            try:
                await self.cleanup_docker_project_by_label(project_name)
            except RuntimeError as exc:
                self._logger.warning(
                    "Failed to clean stale Docker compose project %r: %s",
                    project_name,
                    exc,
                )

    async def stop_environment(self, env: BaseEnvironment) -> None:
        from harbor.environments.docker.docker import DockerEnvironment

        if self.environment_config.delete and isinstance(env, DockerEnvironment):
            try:
                await env._run_docker_compose_command(
                    ["down", "--volumes", "--remove-orphans"]
                )
            except RuntimeError as compose_error:
                project_name = self.docker_compose_project_name(env)
                try:
                    await self.cleanup_docker_project_by_label(project_name)
                    self._logger.warning(
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
            await env.stop(delete=self.environment_config.delete)
