from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_HARNESS_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "harness_config.json"
)
ReasoningEffort = Literal["none", "low", "medium", "high"]
ServiceTier = Literal["auto", "default", "flex"]
PanelPurpose = Literal["promotion", "regression_veto"]
PanelRunStatus = Literal["keep", "discard", "crash"]


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


class PanelRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    when: Literal["always"] | None = None
    after_panel: str | None = None
    when_status: PanelRunStatus | None = None

    @model_validator(mode="after")
    def exactly_one_run_mode(self) -> Self:
        is_always = self.when == "always"
        is_gated = self.after_panel is not None or self.when_status is not None
        if is_always == is_gated:
            raise ValueError(
                "panel run config must be either {'when': 'always'} or "
                "{'after_panel': ..., 'when_status': ...}"
            )
        if is_gated and (self.after_panel is None or self.when_status is None):
            raise ValueError(
                "gated panel run config requires after_panel and when_status"
            )
        return self


class PanelBaselineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required: bool = False


class PanelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    purpose: PanelPurpose
    task_names: list[str] = Field(min_length=1)
    task_timeout_sec: float = Field(default=600.0, gt=0)
    run: PanelRunConfig = Field(default_factory=lambda: PanelRunConfig(when="always"))
    baseline: PanelBaselineConfig = Field(default_factory=PanelBaselineConfig)


class ExcludedTaskGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_names: list[str] = Field(default_factory=list)
    reason: str = ""


class HarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = Field(
        description="Strict config schema version. v1 train/test fields are not accepted."
    )
    experiment_id: str = Field(
        min_length=1,
        description="Stable identifier written into experiment records.",
    )
    focus_name: str = Field(
        default="",
        description="Short label for the mechanism family being studied.",
    )
    panels: list[PanelConfig] = Field(
        min_length=1,
        description="Configured task panels. Runtime policy is compiled from purpose/run fields.",
    )
    excluded_task_groups: dict[str, ExcludedTaskGroup] = Field(
        default_factory=dict,
        description="Named task groups kept out of runnable panels.",
    )
    max_steps: int = Field(
        default=50,
        gt=0,
        description="Maximum policy/environment steps allowed per task episode.",
    )
    max_trial_concurrency: int = Field(
        default=10,
        gt=0,
        description=(
            "Maximum number of task trials in flight at once (each holds one "
            "live container). Bounded by memory and the LLM backend. Raise above "
            "max_heavy_action_concurrency to overlap trials idling on the LLM."
        ),
    )
    max_heavy_action_concurrency: int = Field(
        default=10,
        gt=0,
        description=(
            "Maximum number of trials executing heavyweight harness actions "
            "(reset/startup, run, verify) at once. Cheap harness-generated "
            "file/list/search/edit commands bypass this gate so they do not "
            "queue behind compiles or long verifiers. Bounds host-CPU "
            "contention independently of `max_trial_concurrency`, so trials "
            "idling on the LLM can overlap without oversubscribing cores and "
            "slowing real builds. No effect when >= max_trial_concurrency."
        ),
    )
    env_setup_timeout_sec: float = Field(
        default=600.0,
        gt=0,
        description=(
            "Wall-clock timeout for environment reset (docker start + "
            "bootstrap), budgeted separately from task_timeout_sec so a slow or "
            "hung bootstrap fails fast as a crash without consuming the agent's "
            "step budget or being misreported as a task timeout."
        ),
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

    @model_validator(mode="after")
    def panel_contract_is_valid(self) -> Self:
        panel_ids = [panel.id for panel in self.panels]
        duplicate_ids = sorted(
            panel_id for panel_id in set(panel_ids) if panel_ids.count(panel_id) > 1
        )
        if duplicate_ids:
            raise ValueError("panel ids must be unique: " + ", ".join(duplicate_ids))

        promotion_panels = [
            panel for panel in self.panels if panel.purpose == "promotion"
        ]
        if len(promotion_panels) != 1:
            raise ValueError("exactly one promotion panel is required")
        regression_veto_panels = [
            panel for panel in self.panels if panel.purpose == "regression_veto"
        ]
        if len(regression_veto_panels) > 1:
            raise ValueError("at most one regression_veto panel is supported")

        panel_index_by_id = {panel.id: index for index, panel in enumerate(self.panels)}
        promotion_panel = promotion_panels[0]
        promotion_index = panel_index_by_id[promotion_panel.id]
        for panel in promotion_panels:
            if panel.run.when != "always" or panel.baseline.required:
                raise ValueError(
                    "promotion panel must run always and must not require a baseline"
                )
        for panel in regression_veto_panels:
            if (
                panel.run.after_panel != promotion_panel.id
                or panel_index_by_id[panel.id] <= promotion_index
            ):
                raise ValueError(
                    "regression_veto panel must run after the promotion panel "
                    f"({promotion_panel.id}) and appear later in configured panels"
                )
            if panel.run.when_status != "keep" or not panel.baseline.required:
                raise ValueError(
                    "regression_veto panel must require baseline and run after "
                    "an existing panel reaches keep"
                )

        task_groups: list[tuple[str, set[str]]] = [
            (f"panels.{panel.id}.task_names", set(panel.task_names))
            for panel in self.panels
        ]
        task_groups.extend(
            (f"excluded_task_groups.{name}.task_names", set(group.task_names))
            for name, group in self.excluded_task_groups.items()
        )
        for index, (left_name, left_tasks) in enumerate(task_groups):
            for right_name, right_tasks in task_groups[index + 1 :]:
                overlap = sorted(left_tasks & right_tasks)
                if overlap:
                    raise ValueError(
                        f"{left_name} and {right_name} must be disjoint: "
                        + ", ".join(overlap)
                    )
        return self

    @property
    def promotion_panel(self) -> PanelConfig:
        return next(panel for panel in self.panels if panel.purpose == "promotion")

    @property
    def regression_veto_panel(self) -> PanelConfig | None:
        return next(
            (panel for panel in self.panels if panel.purpose == "regression_veto"),
            None,
        )
