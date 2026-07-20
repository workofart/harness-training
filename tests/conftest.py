"""Shared test scaffolding.

pytest puts this file's directory (``tests/``) on ``sys.path``, so test modules
import these helpers with ``from conftest import _StubLlm, ...``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from src.config import LlmProviderConfig
from src.env.base import (
    RawEnvOutput,
    RunAction,
    VerifyOutcome,
    VerifyVerdict,
    VerifyAction,
)
from src.env.base import StepResult
from src.llm.backend import (
    CompletionBackend,
    Completion,
    CompletionRequest,
    ToolCall,
)
from src.rollout.records import MeasurementIdentity


TEST_MEASUREMENT_IDENTITY = MeasurementIdentity(
    effective_config_digest="effective-config",
    provider_revision="provider-revision",
    replay_regime_digest="replay-regime",
)


def install_fake_swebench(
    monkeypatch, *, run_evaluation: Any = None, make_test_spec: Any = None
) -> ModuleType:
    """Stub the ``swebench`` package tree in ``sys.modules`` so the env/verify code's
    lazy ``from swebench... import ...`` statements resolve to fakes instead of
    importing the real swebench -- which eagerly pulls in HuggingFace ``datasets``
    (~0.7s) that no unit test needs. Returns the fake ``run_evaluation`` module so
    verify-path tests can set ``run_instance`` / read ``RUN_EVALUATION_LOG_DIR``."""
    if run_evaluation is None:
        run_evaluation = ModuleType("swebench.harness.run_evaluation")
        run_evaluation.RUN_EVALUATION_LOG_DIR = Path("logs/run_evaluation")
    test_spec_leaf = ModuleType("swebench.harness.test_spec.test_spec")
    test_spec_leaf.make_test_spec = make_test_spec
    test_spec_pkg = ModuleType("swebench.harness.test_spec")
    test_spec_pkg.test_spec = test_spec_leaf
    harness = ModuleType("swebench.harness")
    harness.run_evaluation = run_evaluation
    harness.test_spec = test_spec_pkg
    swebench = ModuleType("swebench")
    swebench.harness = harness
    for name, module in {
        "swebench": swebench,
        "swebench.harness": harness,
        "swebench.harness.run_evaluation": run_evaluation,
        "swebench.harness.test_spec": test_spec_pkg,
        "swebench.harness.test_spec.test_spec": test_spec_leaf,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return run_evaluation


def make_llm_provider_config(
    *,
    max_context_length: int = 1024,
    max_tokens: int = 8192,
    seed: int | None = 1,
) -> LlmProviderConfig:
    """Scaffolding LLM config for tests that need a valid RunConfig but never
    assert on the LLM payload itself."""
    return LlmProviderConfig(
        model_name="gpt-test",
        base_url="http://127.0.0.1:18000/v1",
        api_key_env="OPENAI_API_KEY",
        max_context_length=max_context_length,
        max_tokens=max_tokens,
        seed=seed,
    )


@pytest.fixture(autouse=True)
def _isolate_shared_docker_client(monkeypatch):
    """Reset the process-wide Docker client singleton so each test's
    ``docker.from_env`` monkeypatch takes effect instead of a client cached by
    an earlier test."""
    from src.env.docker_shell import DockerShellSession

    monkeypatch.setattr(DockerShellSession, "_shared_client", None)


@pytest.fixture(autouse=True)
def _isolate_task_locks(tmp_path, monkeypatch):
    """Point the cross-process task-lock dir at a throwaway path so tests never
    contend on (or litter) the real per-host lock files."""
    import src.concurrency as concurrency

    monkeypatch.setattr(concurrency, "_TASK_LOCK_DIR", tmp_path / "task-locks")


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Point the shared LLM/verify cache at a throwaway db for every test, so nothing
    writes to the real ./cache and no cached state leaks across tests. The store is
    created lazily, so tests that never touch the cache pay nothing."""
    from src.plugins.caching import store as cache

    monkeypatch.setattr(cache, "DB_PATH", tmp_path / "llm_cache.db")
    monkeypatch.setattr(cache, "_STORE", None)
    yield
    if cache._STORE is not None:
        cache._STORE.close()


class _StubLlm(CompletionBackend):
    """Returns a pre-set sequence of completions; records every call."""

    def __init__(self, completions: list[Completion]) -> None:
        self._completions = list(completions)
        self.calls: list[list[dict[str, Any]]] = []
        self.thinking_overrides: list[bool | None] = []
        self.closed = False

    async def _complete(self, request: CompletionRequest) -> Completion:
        self.calls.append([dict(m) for m in request.messages])
        self.thinking_overrides.append(request.enable_thinking)
        return self._completions.pop(0)

    async def close(self) -> None:
        self.closed = True


class _StubEnv:
    """Minimal TaskEnv stub: returns canned states, records calls."""

    setup_timeout_sec = 600.0
    verify_timeout_sec = 1800.0

    def __init__(
        self,
        *,
        reset_env_output: RawEnvOutput | None = None,
        step_results: list[StepResult] | None = None,
        verify_result: StepResult | None = None,
    ) -> None:
        self._reset_env_output = reset_env_output or RawEnvOutput(
            instruction="do the thing", working_dir="/work"
        )
        self._step_results = list(step_results or [])
        self._verify_result = verify_result or StepResult(
            raw_env_output=RawEnvOutput(stdout="verified\n"),
            reward=1.0,
            terminated=True,
            truncated=False,
            verdict=VerifyVerdict(completed=True, passed=True, error=None),
        )
        self.step_calls: list[RunAction | VerifyAction] = []
        self.exec_calls: list[dict[str, Any]] = []
        self.verify_calls = 0
        self.closed = False

    async def reset(self) -> RawEnvOutput:
        return self._reset_env_output

    async def provision(self) -> None:
        pass

    async def execute(self, action: RunAction) -> RawEnvOutput:
        self.step_calls.append(action)
        self.exec_calls.append(
            {
                "command": action.command,
                "cwd": action.cwd,
                "timeout_sec": action.timeout_sec,
            }
        )
        if self._step_results:
            return self._step_results.pop(0).raw_env_output
        return RawEnvOutput(exit_code=0)

    async def verify(self) -> VerifyOutcome:
        self.step_calls.append(VerifyAction())
        self.verify_calls += 1
        return VerifyOutcome(
            verdict=self._verify_result.verdict,
            output=self._verify_result.raw_env_output,
            reward=self._verify_result.reward,
            info=self._verify_result.info,
            metrics=self._verify_result.metrics,
        )

    async def close(self) -> None:
        self.closed = True


def _tool_call(name: str, **args: Any) -> ToolCall:
    return ToolCall(name=name, arguments=json.dumps(args))


def _completion(*calls: ToolCall, content: str | None = None) -> Completion:
    return Completion(tool_calls=tuple(calls), content=content)


def _result(stdout: str, **overrides: Any) -> StepResult:
    fields = {
        "raw_env_output": RawEnvOutput(exit_code=0, stdout=stdout),
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "info": {},
    }
    fields.update(overrides)
    return StepResult(**fields)


def _write_task_artifacts(root: Path, task_name: str) -> dict[str, str]:
    """Create the canonical per-task artifact files under ``root/task_name`` and
    return their paths (as a real run's recorder would leave them on disk)."""
    task_dir = root / task_name
    agent_dir = task_dir / "agent"
    agent_dir.mkdir(parents=True)
    steps_path = agent_dir / "steps.jsonl"
    exec_log_path = agent_dir / "exec.log"
    for path in (steps_path, exec_log_path):
        path.write_text("{}\n")
    return {
        "rollout_dir": str(task_dir),
        "trace_path": str(steps_path),
        "exec_log_path": str(exec_log_path),
    }
