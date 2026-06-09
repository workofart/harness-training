"""Foundation contracts: harness boundary types + per-trial telemetry.

`RawState` and `HarnessEnv` define the environment-side input contract: what the
environment must hand the harness on reset/exec/verify, and the lifecycle
methods the harness can call. `TaskMetrics` is the mutable per-trial telemetry
recorded by `trace.py` and persisted to `metrics.json`; `FailureMode` is its
terminal-state bucket. `is_majority_solved`/`is_majority_decided` are the
majority-vote predicates shared by the orchestrator's early-stop and the gate. The
trial *output* contract (one trial's result) lives in `experiment.record`
(`TrialResult`), a layer up — `contracts` owns only the boundary + telemetry.

Keeping these in one foundation module lets `src/harness/core.py` and
`src/experiment/*` share a single source of truth without depending on each
other (this is the merge of the former `harness/contracts.py` and the
foundation half of the former `metrics.py`; the gate statistics now live in
`supervisor/policy.py`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from src.llm.base import LlmUsage

EnvExecWorkload = Literal["light", "heavy"]

# Single source of truth for the per-trial terminal-state bucket persisted as
# `metrics.json.failure_mode` (mirrored in program.md "Post-Run Diagnosis").
# `crash` is an infra failure that exhausted within-trial retries and is
# excluded from evidence; `interrupted` is a trial stopped from the outside (a
# Ctrl-C or a supervisor restart) rather than failing on its own, also excluded
# from evidence; `None` means the trial reached no terminal outcome.
FailureMode = Literal[
    "solved",
    "never_verified",
    "verified_rejected",
    "hit_step_cap",
    "hit_timeout",
    "no_valid_action",
    "interrupted",
    "crash",
]


@dataclass(frozen=True, slots=True)
class RawState:
    """Explicit environment-to-harness contract.

    The environment owns these fields. The harness may derive higher-level state
    from them, but should not reinterpret missing structure from hidden adapter
    code.

    Contract notes:
    - `reset()` returns `instruction` and `working_dir`.
    - `exec()` returns one command result.
    - `verify()` returns the authoritative terminal judgment for the task.
    """

    instruction: str = ""
    working_dir: str | None = None
    reward: float | None = None
    done: bool = False
    passed: bool | None = None
    return_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None


class HarnessEnv(Protocol):
    """Stable adapter boundary for the harness loop.

    Implementations expose transport and task lifecycle only. Harness action
    semantics stay in `src/harness/core.py`.
    """

    @property
    def trial_dir(self) -> str | None: ...

    @property
    def verifier_stdout_path(self) -> str | None: ...

    async def reset(self) -> RawState:
        """Start a task session and return the initial reset state."""

    async def exec(
        self,
        *,
        command: str,
        cwd: str | None = None,
        timeout_sec: int | None = None,
        workload: EnvExecWorkload = "heavy",
    ) -> RawState:
        """Run one shell command inside the task environment."""

    async def verify(self) -> RawState:
        """Return the authoritative terminal task judgment."""

    async def close(self) -> None:
        """Release all task resources."""


class TaskMetrics(BaseModel):
    """Per-trial counters and terminal-state summary.

    Mutated in place over a trial by the recorders in ``trace.py`` (which own
    a single instance), then frozen into the persisted ``metrics.json`` and
    carried on ``TaskResult.metrics``. Not hashed anywhere, so it is a plain
    mutable model rather than a frozen target fronted by a builder.
    """

    model_config = ConfigDict(extra="forbid")

    steps_total: int = 0
    run_count: int = 0
    verify_count: int = 0
    action_parse_failure_count: int = 0
    token_input_total: int = 0
    token_output_total: int = 0
    token_reasoning_total: int = 0
    token_cached_input_total: int = 0
    rule_fires: dict[str, int] = Field(default_factory=dict)
    custom_counters: dict[str, int] = Field(default_factory=dict)
    # Trial-final summary fields. Populated by the runner once `final_passed`
    # is known so the supervisor can answer "where did it die" without
    # walking steps.jsonl. `failure_mode` is one of: "solved",
    # "never_verified", "verified_rejected", "hit_step_cap", "hit_timeout",
    # "no_valid_action", "interrupted" (stopped from the outside), or "crash"
    # (an infra failure) -- the last three excluded from evidence -- or None
    # when the trial did not produce a terminal outcome.
    final_action_passed: bool | None = None
    verifier_passed: bool | None = None
    failure_mode: FailureMode | None = None

    def record_action(self, step_index: int, action_name: str) -> None:
        self.steps_total = step_index
        self.final_action_passed = None
        if action_name == "run":
            self.run_count += 1
        if action_name == "verify":
            self.verify_count += 1

    def record_action_parse_failure(self) -> None:
        self.action_parse_failure_count += 1

    def record_completion_usage(self, usage: LlmUsage) -> None:
        if usage.prompt_tokens is not None:
            self.token_input_total += usage.prompt_tokens
        if usage.completion_tokens is not None:
            self.token_output_total += usage.completion_tokens
        if usage.reasoning_tokens is not None:
            self.token_reasoning_total += usage.reasoning_tokens
        if usage.cached_input_tokens is not None:
            self.token_cached_input_total += usage.cached_input_tokens

    def record_step_passed(self, raw_state: RawState) -> None:
        """Attribute the env outcome to the most recent agent-chosen action.
        Called only from `env_step_completed` (in-loop)."""
        if isinstance(raw_state.passed, bool):
            self.final_action_passed = raw_state.passed

    def record_rule_fire(self, rule_name: str) -> None:
        self.rule_fires[rule_name] = self.rule_fires.get(rule_name, 0) + 1

    def set_trial_outcome(
        self,
        *,
        verifier_passed: bool | None,
        failure_mode: FailureMode | None,
    ) -> None:
        self.verifier_passed = verifier_passed
        self.failure_mode = failure_mode

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2) + "\n")


def is_majority_solved(*, solved: int, total: int) -> bool:
    """True when `solved` reaches the ceil(total/2) majority threshold."""
    if total <= 0:
        return False
    threshold = (total + 1) // 2
    return solved >= threshold


def is_majority_decided(*, solved: int, finished: int, expected_total: int) -> bool:
    """True when the final majority outcome over `expected_total` trials is
    already locked in by the `solved`/`finished` counts seen so far. Once
    True, remaining trials cannot flip the majority verdict.
    """
    if finished >= expected_total:
        return True
    remaining = expected_total - finished
    final_threshold = (expected_total + 1) // 2
    can_be_true = (solved + remaining) >= final_threshold
    can_be_false = solved < final_threshold
    return not (can_be_true and can_be_false)
