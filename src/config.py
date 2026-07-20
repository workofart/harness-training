from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_EXTENDS_KEY = "$extends"
_INCLUDE_KEY = "$include"


def load_config_payload(path: str | Path) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Effective config payload plus every file it was read from."""
    sources: list[Path] = []
    path = Path(path).expanduser().resolve()
    config_dir = path.parent
    payload = _load_config_mapping(path, sources)
    extends = payload.get(_EXTENDS_KEY, [])
    if (
        not isinstance(extends, list)
        or len(extends) > 1
        or not all(isinstance(item, str) for item in extends)
    ):
        raise ValueError(
            f"{_EXTENDS_KEY} must contain at most one relative path: {path}"
        )

    merged: dict[str, Any] = {}
    if extends:
        parent_path = _relative_config_path(config_dir, extends[0])
        merged = _load_config_mapping(parent_path, sources)
    current = {key: value for key, value in payload.items() if key != _EXTENDS_KEY}
    payload = _merge_config(
        merged, _resolve_includes(current, config_dir=config_dir, sources=sources)
    )
    return payload, tuple(sources)


def _load_config_mapping(path: Path, sources: list[Path]) -> dict[str, Any]:
    sources.append(path)
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"config file must contain a YAML mapping: {path}")
    return payload


def _resolve_includes(
    payload: dict[str, Any],
    *,
    config_dir: Path,
    sources: list[Path],
) -> dict[str, Any]:
    resolved = {}
    for key, value in payload.items():
        if isinstance(value, dict) and _INCLUDE_KEY in value:
            if set(value) != {_INCLUDE_KEY}:
                raise ValueError(f"{_INCLUDE_KEY} cannot have sibling fields")
            include = value[_INCLUDE_KEY]
            if not isinstance(include, str):
                raise ValueError(f"{_INCLUDE_KEY} must be a relative path")
            include_path = _relative_config_path(config_dir, include)
            value = _load_config_mapping(include_path, sources)
        resolved[key] = value
    return resolved


def _relative_config_path(config_dir: Path, path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        raise ValueError(f"config include path must be relative: {path}")
    resolved = (config_dir / candidate).resolve()
    if not resolved.is_relative_to(config_dir):
        raise ValueError(
            f"config include path must stay within config directory: {path}"
        )
    return resolved


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


class LlmProviderConfig(BaseModel):
    """One OpenAI-compatible chat-completions endpoint plus the model to call on it.

    Deliberately flat, litellm-style: any provider speaking the OpenAI protocol
    (local SGLang/vLLM, OpenRouter, ...) is just ``base_url`` + ``api_key_env``;
    model-specific request shaping is plain optional fields."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai_compatible"] = Field(
        default="openai_compatible",
        description="Completion transport.",
    )
    model_name: str = Field(min_length=1, description="Model id to call.")
    base_url: str = Field(
        description="OpenAI-compatible /v1 base URL, e.g. https://api.openai.com/v1.",
    )
    api_key_env: str = Field(
        min_length=1,
        description="Environment variable holding the API key; never the key itself.",
    )
    max_context_length: int = Field(
        gt=0,
        description="Prompt context budget the framework enforces for this model.",
    )
    tokenizer_name: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "HuggingFace Hub model id whose tokenizer measures requests and clips "
            "observations in real model tokens. Unset: model_name when it is "
            "Hub-id-shaped, else chars/4."
        ),
    )
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    seed: int | None = None
    max_tokens: int = Field(
        gt=0,
        description=(
            "Cap on generated tokens. A thinking model needs headroom for its "
            "reasoning plus the tool call, or the response truncates."
        ),
    )
    enable_thinking: bool | None = Field(
        default=None,
        description=(
            "Qwen-style thinking control (chat_template_kwargs.enable_thinking). "
            "Unset omits chat_template_kwargs entirely, for servers without it."
        ),
    )
    thinking_budget_tokens: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Early abort for thinking runaways: this many reasoning tokens with no "
            "answer started ends the stream as finish_reason='length', firing "
            "thinking-off recovery without decoding to max_tokens. In the cache "
            "key, since an aborted completion differs from a full one. Requires "
            "enable_thinking: true and a resolvable tokenizer."
        ),
    )
    reasoning_effort: Literal["low", "medium", "high", "xhigh"] | None = Field(
        default=None,
        description=(
            "Provider reasoning effort; None omits the field from the request, "
            "so serialized config and request body agree."
        ),
    )
    provider_only: list[str] | None = Field(
        default=None,
        min_length=1,
        description="OpenRouter provider slugs allowed to serve the request.",
    )

    @property
    def is_deterministic(self) -> bool:
        return self.provider == "openai_compatible" and self.seed is not None

    @property
    def provider_revision(self) -> str:
        revision = f"{self.provider}:{self.base_url}:{self.model_name}"
        override = os.environ.get("FRAMEWORK_LLM_CACHE_REV")
        return revision if override is None else f"{revision}:{override}"

    @model_validator(mode="after")
    def _validate_provider_contract(self) -> "LlmProviderConfig":
        if self.thinking_budget_tokens is not None and self.enable_thinking is not True:
            raise ValueError("thinking_budget_tokens requires enable_thinking: true")
        return self


class EnvironmentConfig(BaseModel):
    """Environment backend plus the task panel it runs.

    kind decides how task_names resolve, so the two live in one block: a
    task-panel fragment carries its own kind and cannot be paired with the
    wrong backend."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["swe", "terminal_bench"] = Field(
        description=(
            "How task_names resolve: 'swe' as SWE-bench-Verified instance ids, "
            "'terminal_bench' as Terminal-Bench 2.1 task names."
        )
    )
    task_names: list[str] = Field(min_length=1)
    host_netcache: bool = Field(
        default=True,
        description=(
            "Freeze container network observations on the host (Terminal-Bench "
            "determinism substrate; 'swe' ignores it)."
        ),
    )

    @model_validator(mode="after")
    def _validate_task_panel(self) -> "EnvironmentConfig":
        if len(self.task_names) != len(set(self.task_names)):
            raise ValueError("task_names must be unique")
        if self.kind == "swe" and not self.host_netcache:
            # swe never reads it: a non-default value forks identity, not execution.
            raise ValueError(
                "host_netcache applies only to terminal_bench; 'swe' must leave "
                "it default (true)"
            )
        return self


class PluginsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_cache: bool = True
    execution: Literal["eager", "replay"] = "eager"


class TrainingTargetConfig(BaseModel):
    """The trained subject: the policy module the measured loop drives and the
    files a candidate patch may touch. ``module`` is the single source -- the
    editable surface path derives from it, so the file the trainer validates
    and the module the rollout imports cannot disagree."""

    model_config = ConfigDict(extra="forbid")

    module: str = Field(
        min_length=1,
        description=(
            "Import path of the editable policy module, e.g. 'src.policy.core'; "
            "must export build_policy and build_env_action (src/policy/base.py). "
            "Imported in the measurement worker, so it binds to the measured worktree."
        ),
    )
    extra_patch_paths: tuple[str, ...] = Field(
        default=(),
        description=(
            "Repo-relative paths a candidate may change besides the surface, "
            "e.g. its contract test."
        ),
    )
    proposer_visible: tuple[str, ...] = Field(
        default=(),
        description=(
            "gitignore-style non-cone sparse patterns for the proposer's checkout; "
            "order matters. Visibility only -- writes are enforced by "
            "validate_candidate."
        ),
    )

    @property
    def surface(self) -> str:
        """The one file every candidate must change."""
        return self.module.replace(".", "/") + ".py"

    @property
    def patch_paths(self) -> tuple[str, ...]:
        return (self.surface, *self.extra_patch_paths)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config_path: str | None = Field(default=None, exclude=True)
    schema_version: Literal[13] = Field(
        description="Strict schema version; older payloads are rejected, not migrated."
    )
    training_target: TrainingTargetConfig = Field(
        description="The trained policy module and candidate patch surface.",
    )
    environment: EnvironmentConfig = Field(
        description="Environment backend plus the task panel to run on it.",
    )
    max_steps: int = Field(
        default=50,
        gt=0,
        description="Maximum policy/environment steps allowed per task episode.",
    )
    agent_timeout_multiplier: float = Field(
        default=1.0,
        gt=0,
        allow_inf_nan=False,
        description="Multiplier applied to each task's agent wall-clock timeout.",
    )
    max_rollout_concurrency: int = Field(
        default=10,
        gt=0,
        description=(
            "Task rollouts in flight at once; each holds a live container, so "
            "memory and the LLM backend bound it."
        ),
    )
    plugins: PluginsConfig = PluginsConfig()
    llm_provider_config: LlmProviderConfig = Field(
        description="LLM provider settings used by the harness policy."
    )

    @model_validator(mode="after")
    def _validate_plugin_contract(self) -> "RunConfig":
        if self.plugins.execution == "replay":
            # Replay fails closed instead of silently falling back to eager execution.
            if not self.llm_provider_config.is_deterministic:
                raise ValueError(
                    'plugins.execution "replay" requires a deterministic '
                    "llm_provider_config (openai_compatible with seed)"
                )
            if (
                self.environment.kind == "terminal_bench"
                and not self.environment.host_netcache
            ):
                raise ValueError(
                    'plugins.execution "replay" requires environment.host_netcache '
                    "for terminal_bench"
                )
        return self

    def measurement_identity_payload(self) -> dict[str, Any]:
        """Measured-config projection for the identity digest: excludes
        trainer-only fields (extra_patch_paths, proposer_visible) that steer
        candidate production, not the measured rollout.
        """
        payload = self.model_dump(mode="json")
        for trainer_only in ("extra_patch_paths", "proposer_visible"):
            payload["training_target"].pop(trainer_only, None)
        return payload

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        payload, _ = load_config_payload(path)
        return cls.model_validate(payload).model_copy(
            # cwd-independent: the worker reopens this path from measure_root.
            update={"config_path": str(Path(path).expanduser().resolve())}
        )

    def with_task_panel(self, task_names: Sequence[str]) -> "RunConfig":
        """Same measurement definition on a different task panel."""
        environment = EnvironmentConfig.model_validate(
            {**self.environment.model_dump(), "task_names": list(task_names)}
        )
        return self.model_copy(update={"environment": environment})
