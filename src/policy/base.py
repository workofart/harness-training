"""The policy contract: what ``run_task_loop`` drives, plus the signals it exchanges.

``build_policy``'s keyword signature is fixed by the runner. This module imports
nothing from the concrete policy, yet the policy imports the failure signals and
:class:`AgentCallback` from here (the loop catches the signals and feeds the
callback, so both are part of the loop↔policy contract, not policy internals).
"""

from __future__ import annotations

from typing import Any, Final, Literal, Protocol, runtime_checkable

from src.env.base import RawEnvOutput, StepResult


SUBMIT_ACTION_NAME: Final[Literal["submit"]] = "submit"
"""Frozen episode-routing key; the editable harness must expose this tool name."""


@runtime_checkable
class Policy(Protocol):
    """One rollout's agent, as ``run_task_loop`` drives it.

    Actions stay opaque here: the concrete ``Action`` type lives in the editable
    policy, and the episode loop receives them as ``Any``, rendering each through
    the harness's ``build_env_action`` (the env-render seam), so the frozen framework
    never depends on the harness's action shape.
    """

    def reset(self, raw_env_output: RawEnvOutput) -> None: ...

    async def act(self) -> tuple[Any, ...]: ...

    def observe(self, action: Any, step_result: StepResult) -> None: ...


class NoValidActionError(RuntimeError):
    """No parseable tool call within retry budget; classified as agent failure."""


class RepeatedLengthCutoffError(RuntimeError):
    """The model repeatedly used the whole output budget before a tool call."""


PolicyEventName = Literal[
    "context_window_retrimmed",
    "action_parse_failed",
    "observation_clipped",
    "context_groups_dropped",
    "policy_rule",
]
"""Closed diagnosis-only vocabulary; candidate mechanisms use policy_rule, so new names change the frozen contract."""


class AgentCallback(Protocol):
    """Diagnosis-only event channel the editable agent emits into the rollout trace.

    Nothing the gate reads flows through here; grade-bearing measurement (rendered
    requests, completions, first-try validity) is captured on the frozen side of the
    LLM boundary instead.
    """

    def __call__(self, event: PolicyEventName, /, **fields: Any) -> None: ...


def _noop_agent_callback(_event: PolicyEventName, /, **_: Any) -> None: ...


NOOP_AGENT_CALLBACK: AgentCallback = _noop_agent_callback
