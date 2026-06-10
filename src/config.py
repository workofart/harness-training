from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_HARNESS_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "harness_config.json"
)
ReasoningEffort = Literal["none", "low", "medium", "high"]
OpenRouterServiceTier = Literal["auto", "default", "flex"]
ChatGptCodexServiceTier = Literal["auto", "default", "flex", "priority", "standard"]
PanelPurpose = Literal["promotion", "regression_veto"]


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
    service_tier: OpenRouterServiceTier | None = Field(
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
    service_tier: ChatGptCodexServiceTier | None = Field(
        default=None,
        description="Optional Codex backend service tier.",
    )
    prompt_cache_key: str | None = Field(
        default=None,
        description="Optional prompt cache key forwarded to the Codex backend.",
    )


LlmProviderConfig = OpenRouterConfig | ChatGptCodexConfig


class PanelConfig(BaseModel):
    """A task panel: membership and per-trial wall budget only. The panel's role
    comes from ``purpose``; the sequencing (promotion runs first, the veto runs
    only after a train keep) is hardcoded in ``supervisor/policy.decide()``, not
    configured here."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    purpose: PanelPurpose
    task_names: list[str] = Field(min_length=1)
    task_timeout_sec: float = Field(default=600.0, gt=0)


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

    @property
    def train_tasks(self) -> frozenset[str]:
        # The promotion panel's task set -- what `auto` runs as the train panel
        # and `gate(.., purpose="promotion")` scores. Derived from panels, not a
        # separate persisted field, so the new `supervisor` consumes the redesign
        # train/test vocabulary (§5/§12) while the config schema is unchanged and
        # the old stack keeps reading `panels[]`. Always non-empty (promotion
        # `task_names` has min_length 1).
        return frozenset(self.promotion_panel.task_names)

    @property
    def test_tasks(self) -> frozenset[str]:
        # The regression-veto panel's task set; empty when no veto panel is
        # configured. `scan()` asserts both panels non-empty + disjoint when it
        # builds `World` (§12); panel disjointness itself is already enforced by
        # `panel_contract_is_valid`.
        veto = self.regression_veto_panel
        return frozenset() if veto is None else frozenset(veto.task_names)
