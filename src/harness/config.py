from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_HARNESS_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "harness_config.json"
)
ReasoningEffort = Literal["none", "low", "medium", "high"]
ServiceTier = Literal["auto", "default", "flex"]


class OpenRouterProviderRouting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: tuple[str, ...]
    allow_fallbacks: bool
    ignore: tuple[str, ...] = ()


class OpenRouterProviderKwargs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: OpenRouterProviderRouting | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    require_parameters: bool | None = None


class OpenRouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["openrouter"] = "openrouter"
    model_name: str = Field(description="OpenRouter model identifier to call.")
    temperature: float = Field(
        default=0.0,
        description="Sampling temperature passed to the LLM provider.",
    )
    max_output_tokens: int = Field(
        default=32768,
        description="Maximum completion tokens requested from the provider.",
    )
    reasoning_effort: ReasoningEffort = Field(
        default="medium",
        description="Provider-specific reasoning effort for model completions.",
    )
    timeout_seconds: float = Field(
        default=60.0,
        description="Per-request timeout for one LLM completion attempt.",
    )
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for the OpenRouter-compatible API endpoint.",
    )
    top_p: float | None = Field(
        default=None,
        description="Optional nucleus sampling parameter forwarded to the provider.",
    )
    seed: int | None = Field(
        default=None,
        description="Optional deterministic sampling seed.",
    )
    service_tier: ServiceTier | None = Field(
        default=None,
        description="Provider service tier requested for completions.",
    )
    provider_kwargs: OpenRouterProviderKwargs = Field(
        default_factory=lambda: OpenRouterProviderKwargs(),
        description="Additional provider-specific request arguments.",
    )


class ChatGptCodexConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["chatgpt_codex"] = "chatgpt_codex"
    model_name: str = Field(description="Codex backend model identifier to call.")
    max_context_length: int = Field(
        gt=0,
        description="Prompt context budget used by the harness for this model.",
    )
    reasoning_effort: ReasoningEffort = Field(
        default="medium",
        description="Reasoning effort sent to the Codex backend; 'none' omits it.",
    )
    timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Per-request timeout for one LLM completion attempt.",
    )
    base_url: str = Field(
        default="https://chatgpt.com/backend-api",
        description="ChatGPT backend API base URL.",
    )
    auth_file: str | None = Field(
        default=None,
        description="Optional path to Codex auth.json; defaults to CODEX_HOME/auth.json.",
    )
    text_verbosity: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Responses text verbosity.",
    )
    service_tier: ServiceTier | None = Field(
        default=None,
        description="Optional Codex backend service tier.",
    )
    prompt_cache_key: str | None = Field(
        default=None,
        description="Optional prompt cache key forwarded to the Codex backend.",
    )


LlmProviderConfig = OpenRouterConfig | ChatGptCodexConfig


class HarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(
        min_length=1,
        description="Stable identifier written into experiment records.",
    )
    focus_name: str = Field(
        default="",
        description="Short label for the mechanism family being studied.",
    )
    train_task_names: list[str] = Field(
        min_length=1,
        description="Task ids used for the train panel.",
    )
    max_steps: int = Field(
        default=50,
        gt=0,
        description="Maximum policy/environment steps allowed per task episode.",
    )
    max_concurrency: int = Field(
        default=10,
        gt=0,
        description="Maximum number of task episodes to run concurrently.",
    )
    task_timeout_sec: float = Field(
        default=600.0,
        gt=0,
        description="Wall-clock timeout for one task episode.",
    )
    max_output_retries: int = Field(
        default=2,
        ge=0,
        description="Maximum retries after invalid model output formatting.",
    )
    task_trials: int = Field(
        default=1,
        ge=1,
        description="Number of independent trials per task per panel run. Selection uses majority across trials.",
    )
    llm_provider_config: LlmProviderConfig = Field(
        description="LLM provider settings used by the harness policy."
    )
