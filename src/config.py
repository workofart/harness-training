from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.llm.base import ReasoningEffort

DEFAULT_HARNESS_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "harness_config.json"
)
OpenRouterServiceTier = Literal["auto", "default", "flex"]
ChatGptCodexServiceTier = Literal["auto", "default", "flex", "priority", "standard"]


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
        default_factory=OpenRouterProviderKwargs,
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


class TaskPanel(BaseModel):
    """One task set + its per-trial wall budget. ``train`` is the promotion
    panel (the gate scores it for improvement); ``test`` is the regression-veto
    panel (it can only block). The promotion-then-veto sequencing is hardcoded
    in ``supervisor/policy.decide()``, not configured here."""

    model_config = ConfigDict(extra="forbid")

    task_names: list[str] = Field(min_length=1)
    task_timeout_sec: float = Field(default=600.0, gt=0)
    # Wall ceiling for one graded verify on this panel's tasks (the executor's
    # verify-ceiling wrapper). Sits next to task_timeout_sec so each panel can
    # trade grader patience for wall time: a hung grader burns the full ceiling
    # as a terminal failing verdict either way.
    verify_timeout_sec: float = Field(default=900.0, gt=0)


class HarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[3] = Field(
        description="Strict config schema version. v2 panels[] payloads are not accepted."
    )
    train: TaskPanel = Field(
        description="The promotion panel: what auto trains and gates on."
    )
    test: TaskPanel | None = Field(
        default=None,
        description="Optional regression-veto panel (held-out tasks; block-only).",
    )
    env_backend: Literal["harbor", "swe"] = Field(
        default="harbor",
        description=(
            "Which HarnessEnv backs each trial. 'harbor' (default): Terminal "
            "Bench task directories. 'swe': SWE-bench-Verified instances, where "
            "task_names are instance ids resolved to dataset rows. Selects the "
            "trial_runner the cli builds; every other field applies unchanged."
        ),
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
    def task_groups_are_disjoint(self) -> Self:
        # A task sits in exactly one panel: train or test.
        task_groups: list[tuple[str, set[str]]] = [
            ("train.task_names", set(self.train.task_names))
        ]
        if self.test is not None:
            task_groups.append(("test.task_names", set(self.test.task_names)))
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
    def train_tasks(self) -> frozenset[str]:
        # What `auto` runs as the train panel and `gate(.., "promotion")` scores.
        # Always non-empty (`task_names` has min_length 1).
        return frozenset(self.train.task_names)

    @property
    def test_tasks(self) -> frozenset[str]:
        # Empty when no test panel is configured. `scan()` asserts both panels
        # non-empty + disjoint when it builds `World` (§12).
        return frozenset() if self.test is None else frozenset(self.test.task_names)
