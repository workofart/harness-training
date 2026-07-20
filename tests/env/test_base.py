from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import EnvironmentConfig
from src.env import swe as swe_module
from src.env.base import (
    DockerTaskEnv,
    RawEnvOutput,
    RunAction,
    VerifyOutcome,
    VerifyVerdict,
    benchmark,
)
from src.env.docker_shell import ExecResult
import src.env.terminal_bench as terminal_bench_module


@dataclass(frozen=True, slots=True)
class _Task:
    instruction: str
    agent_timeout_sec: float | None = None
    version: str | None = None


class _Session:
    def __init__(self) -> None:
        self.start_calls = 0
        self.close_calls = 0
        self.runs: list[dict] = []

    async def start(self) -> None:
        self.start_calls += 1

    async def run(
        self,
        *,
        command: str,
        cwd: str,
        timeout: float | None,
    ) -> ExecResult:
        self.runs.append({"command": command, "cwd": cwd, "timeout": timeout})
        return ExecResult(exit_code=0, stdout="ran\n", stderr="")

    async def close(self) -> None:
        self.close_calls += 1


class _Env(DockerTaskEnv[_Task]):
    _task_workdir = "/work"

    def _build_solve_env(self, _task: _Task) -> _Session:
        return _Session()

    async def verify(self) -> VerifyOutcome:
        return VerifyOutcome(
            output=RawEnvOutput(stdout="verified\n"),
            reward=1.0,
            verdict=VerifyVerdict(completed=True, passed=True, error=None),
            info={},
        )


def test_docker_task_env_requires_verify_timeout_at_construction(
    tmp_path: Path,
) -> None:
    # Forgetting the grader budget must fail at construction, not as an
    # AttributeError at submit time deep in a rollout.
    with pytest.raises(TypeError):
        _Env(task=_Task(instruction="solve it"), artifacts_dir=tmp_path)


def test_docker_task_env_owns_common_lifecycle(tmp_path: Path) -> None:
    env = _Env(
        task=_Task(instruction="solve it"),
        artifacts_dir=tmp_path / "rollout",
        verify_timeout_sec=1800.0,
    )
    assert env.verify_timeout_sec == 1800.0

    raw = asyncio.run(env.reset())
    assert raw.instruction == "solve it"
    assert raw.working_dir == "/work"
    assert env._solve_env.start_calls == 0

    run_result = asyncio.run(env.execute(RunAction(command="pwd")))
    assert run_result.stdout == "ran\n"
    assert env._solve_env.start_calls == 1
    assert env._solve_env.runs == [{"command": "pwd", "cwd": "/work", "timeout": None}]

    asyncio.run(env.close())
    assert env._solve_env.close_calls == 1


@pytest.mark.parametrize(
    ("kind", "env_cls", "task_id"),
    [
        ("swe", swe_module.SweEnv, "task-a"),
        ("terminal_bench", terminal_bench_module.TerminalBenchEnv, "regex-log"),
    ],
)
def test_benchmark_registry_load_tasks_dispatches_backend(
    monkeypatch, tmp_path: Path, kind, env_cls, task_id
):
    calls: list[tuple[str, ...]] = []

    async def load(*, task_ids, environment, verify_wrapper=None):
        assert environment == EnvironmentConfig(kind=kind, task_names=[task_id])
        assert verify_wrapper is None
        calls.append(tuple(task_ids))
        return SimpleNamespace(
            task=lambda task_id: SimpleNamespace(
                make_env=lambda rollout_dir: SimpleNamespace(
                    task_id=task_id,
                    rollout_dir=rollout_dir,
                ),
            ),
        )

    module = swe_module if env_cls is swe_module.SweEnv else terminal_bench_module
    monkeypatch.setattr(module, "load_tasks", load)

    taskset = asyncio.run(
        benchmark(kind).load_tasks(
            task_ids=[task_id],
            environment=EnvironmentConfig(kind=kind, task_names=[task_id]),
        )
    )
    env = taskset.task(task_id).make_env(tmp_path / "rollout")

    assert calls == [(task_id,)]
    assert env.task_id == task_id
    assert env.rollout_dir == tmp_path / "rollout"


def test_verify_verdict_rejects_invalid_completion_states() -> None:
    cases = [
        (
            {"completed": True, "passed": None, "error": None},
            "completed verdict",
        ),
        (
            {"completed": False, "passed": False, "error": None},
            "incomplete verdict",
        ),
    ]

    for fields, message in cases:
        with pytest.raises(ValueError, match=message):
            VerifyVerdict(**fields)


@pytest.mark.parametrize(
    ("kind", "expected_names"),
    [
        pytest.param(
            "swe",
            ["f2p_progress", "invalid_first_attempts", "steps_used"],
            id="swe-prepends-f2p",
        ),
        pytest.param(
            "terminal_bench",
            ["invalid_first_attempts", "steps_used"],
            id="terminal_bench-generic-pair",
        ),
    ],
)
def test_benchmark_secondary_metrics(kind, expected_names) -> None:
    metrics = benchmark(kind).secondary_metrics
    assert [metric.name for metric in metrics] == expected_names
