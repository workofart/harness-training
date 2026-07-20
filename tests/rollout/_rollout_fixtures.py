"""Shared rollout fixtures for the episode/sampler/telemetry test modules.

Fake backend/env, config builders, and trace helpers used by both the
sampler (run_experiment) and episode (RolloutRunner) test modules. Not a
conftest: imported by name so it does not shadow the root conftest that
other experiment tests import ``make_llm_provider_config`` from.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from conftest import make_llm_provider_config
from src.config import (
    RunConfig,
    EnvironmentConfig,
)
from src.env.base import (
    MODEL_PATCH_INFO_KEY,
    RawEnvOutput,
    RunAction,
    VerifyOutcome,
    VerifyVerdict,
)
from src.env.base import StepResult
from src.rollout.telemetry import RolloutTelemetry
from src.llm.backend import (
    Completion,
    CompletionBackend,
    CompletionRequest,
)


_MODEL_PATCH = "diff --git a/demo.py b/demo.py\n--- a/demo.py\n+++ b/demo.py\n"


class _FakeLlm(CompletionBackend):
    def __init__(
        self, completions: list[Completion], *, completion_delay_sec: float = 0.0
    ) -> None:
        self._completions = list(completions)
        self._completion_delay_sec = completion_delay_sec
        self.closed = False

    async def _complete(self, request: CompletionRequest) -> Completion:
        del request
        if self._completion_delay_sec:
            await asyncio.sleep(self._completion_delay_sec)
        return self._completions.pop(0)

    async def close(self) -> None:
        self.closed = True


class _FakeEnv:
    def __init__(
        self,
        *,
        verify_result: StepResult | None = None,
        reset_delay_sec: float = 0.0,
        provision_delay_sec: float = 0.0,
        step_delay_sec: float = 0.0,
        verify_delay_sec: float = 0.0,
        verify_timeout_sec: float = 1800.0,
        setup_timeout_sec: float = 600.0,
    ) -> None:
        self.verify_timeout_sec = verify_timeout_sec
        self.setup_timeout_sec = setup_timeout_sec
        self._verify_result = verify_result or StepResult(
            raw_env_output=RawEnvOutput(stdout="verified\n"),
            reward=1.0,
            terminated=True,
            truncated=False,
            info={MODEL_PATCH_INFO_KEY: _MODEL_PATCH},
            verdict=VerifyVerdict(completed=True, passed=True, error=None),
        )
        self._reset_delay_sec = reset_delay_sec
        self._provision_delay_sec = provision_delay_sec
        self._step_delay_sec = step_delay_sec
        self._verify_delay_sec = verify_delay_sec
        self.reset_calls = 0
        self.provision_calls = 0
        self.exec_calls: list[str] = []
        self.verify_calls = 0
        self.closed = False

    async def reset(self) -> RawEnvOutput:
        self.reset_calls += 1
        if self._reset_delay_sec:
            await asyncio.sleep(self._reset_delay_sec)
        return RawEnvOutput(instruction="do the task", working_dir="/work")

    async def provision(self) -> None:
        self.provision_calls += 1
        if self._provision_delay_sec:
            await asyncio.sleep(self._provision_delay_sec)

    async def execute(self, action: RunAction) -> RawEnvOutput:
        self.exec_calls.append(action.command)
        if self._step_delay_sec:
            await asyncio.sleep(self._step_delay_sec)
        return RawEnvOutput(exit_code=0, stdout="ran\n")

    async def verify(self) -> VerifyOutcome:
        self.verify_calls += 1
        if self._verify_delay_sec:
            await asyncio.sleep(self._verify_delay_sec)
        return VerifyOutcome(
            verdict=self._verify_result.verdict,
            output=self._verify_result.raw_env_output,
            reward=self._verify_result.reward,
            info=self._verify_result.info,
            metrics=self._verify_result.metrics,
        )

    async def close(self) -> None:
        self.closed = True


def _rollout_config(**overrides: Any) -> RunConfig:
    values = {
        "config_path": "config/run_config.json",
        "max_steps": 3,
        "max_context_length": 16384,
        "max_completion_tokens": 8192,
    }
    values.update(overrides)
    return RunConfig(
        config_path=values["config_path"],
        schema_version=13,
        training_target={"module": "src.policy.core"},
        environment=EnvironmentConfig(
            kind="swe",
            task_names=["task-a"],
        ),
        max_steps=values["max_steps"],
        max_rollout_concurrency=1,
        llm_provider_config=make_llm_provider_config(
            max_context_length=values["max_context_length"],
            max_tokens=values["max_completion_tokens"],
        ),
    )


def _telemetry(tmp_path: Path) -> RolloutTelemetry:
    return RolloutTelemetry(
        rollout_dir=tmp_path,
        trace_path=tmp_path / "agent" / "steps.jsonl",
    )
