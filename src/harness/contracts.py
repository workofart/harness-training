"""Harness boundary types in both directions.

`RawState` and `HarnessEnv` define the environment-side input contract: what
the environment must hand the harness on reset/exec/verify, and the lifecycle
methods the harness can call. `TaskResult` defines the experiment trial output
contract: the value `src.experiment.trial.run_task()` returns and the runner
persists.

Keeping both contracts here lets `src/harness/core.py` and
`src/experiment/runner.py` share a single source of truth without either
depending on the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.metrics import TaskMetrics

EnvExecWorkload = Literal["light", "heavy"]


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


class TaskResult(BaseModel):
    """Experiment trial output contract: one trial's full result.

    Returned by `src.experiment.trial.run_task()`, aggregated by the runner
    into `TaskTrials`, persisted to `experiment.json`, and reloaded via
    `model_validate` for diagnosis.
    Every constructor passes concrete values for `solved` and `steps_used` —
    crash and timeout paths resolve to `solved=False` with a recorded step
    count — so neither is optional. (`reward` stays optional: a trial that
    never reached the verifier has no reward.)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_name: str
    reward: float | None = None
    solved: bool = False
    error: str | None = None
    steps_used: int = 0
    trial_dir: str | None = None
    trace_path: str | None = None
    metrics_path: str | None = None
    verifier_stdout_path: str | None = None
    metrics: TaskMetrics = Field(default_factory=TaskMetrics)
    started_at: str | None = None
    finished_at: str | None = None
