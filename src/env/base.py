"""Environment contract between the framework loop and concrete task envs.

Envs must guarantee observations are a pure function of task content plus
ordered action history, modulo ``scrub_nondeterminism``. This is the framework
measurement premise: same policy, task, and model imply the same trajectory.
Both the LLM completion cache and env-step replay cache consume that premise.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    from src.env.docker_shell import DockerShellSession
    from src.rollout.metrics import SecondaryRewardMetric

from src.determinism import scrub_nondeterminism

DEFAULT_SETUP_TIMEOUT_SEC = 600.0
# Exit status `timeout` returns when it must stop a command (its default).
# Shared contract: the shell session reports it and the replay step cache
# refuses to record such steps.
COMMAND_TIMEOUT_EXIT_CODE = 124
# Wall budget for tasks whose definition declares no agent budget (SWE-bench
# defines none); tasks that do declare one always win.
DEFAULT_AGENT_TIMEOUT_SEC = 4500.0
# Outer watchdog beyond the env-owned grader deadline; only catches a grader
# that failed to enforce its own bound. Failure detector, not a policy knob.
VERIFY_BACKSTOP_SLACK_SEC = 600.0
MODEL_PATCH_INFO_KEY = "model_patch"


@dataclass(frozen=True, slots=True)
class VerifyVerdict:
    completed: bool
    passed: bool | None
    error: str | None

    def __post_init__(self) -> None:
        if self.completed:
            if self.passed is None or self.error is not None:
                raise ValueError("completed verdict requires passed and no error")
            return
        if self.error is None or self.passed is not None:
            raise ValueError("incomplete verdict requires error and no passed")


@dataclass(frozen=True, slots=True)
class RawEnvOutput:
    """Policy-visible environment output after reset or an action."""

    instruction: str = ""
    working_dir: str | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True, slots=True)
class RunAction:
    command: str
    cwd: str | None = None
    timeout_sec: float | None = None


@dataclass(frozen=True, slots=True)
class VerifyAction:
    pass


@dataclass(frozen=True, slots=True)
class VerifyOutcome:
    verdict: VerifyVerdict
    output: RawEnvOutput
    reward: float
    info: dict[str, Any]
    metrics: dict[str, int | float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StepResult:
    """RL-style transition result for one environment action."""

    raw_env_output: RawEnvOutput
    reward: float
    terminated: bool
    truncated: bool
    # JSON-serializable only: replay recording round-trips it through json.dumps.
    info: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, int | float] = field(default_factory=dict)
    verdict: VerifyVerdict | None = None


class UnscorableInfraError(RuntimeError):
    """The env could not produce a scorable rollout because required infra failed."""


class FrameworkEnvError(RuntimeError):
    """The frozen env adapter or persisted verifier data violated its contract."""


class StepExecutionError(Exception):
    """An env call failed crossing the TaskEnv boundary; framework failures stay raw."""


def scrub_raw_env_output(
    raw_env_output: RawEnvOutput, *, command: str | None = None
) -> RawEnvOutput:
    """Return a copy with volatile stdout/stderr tokens scrubbed.

    ``command`` is the run command that produced the output, when known; it
    gates command-conditioned scrub rules.
    """
    return replace(
        raw_env_output,
        stdout=scrub_nondeterminism(raw_env_output.stdout, command=command),
        stderr=scrub_nondeterminism(raw_env_output.stderr, command=command),
    )


def scrub_step_result(result: StepResult, *, command: str | None = None) -> StepResult:
    """The one scrub applied wherever recorded and live step results are
    compared; record, materialize, and audit must stay in lockstep."""
    return replace(
        result,
        raw_env_output=scrub_raw_env_output(result.raw_env_output, command=command),
    )


@runtime_checkable
class TaskEnv(Protocol):
    setup_timeout_sec: float
    verify_timeout_sec: float

    async def reset(self) -> RawEnvOutput: ...
    async def provision(self) -> None:
        """Bring up the env's execution substrate (idempotent).

        Callers that will execute live actions provision explicitly under
        ``setup_timeout_sec``. A rollout that goes live only mid-flight (cache
        miss, catch-up) provisions lazily inside that live work instead.
        """
        ...

    async def execute(self, action: RunAction) -> RawEnvOutput: ...
    async def verify(self) -> VerifyOutcome: ...
    async def close(self) -> None: ...


async def execute_env_action(
    env: TaskEnv, action: RunAction | VerifyAction
) -> StepResult:
    """Single action->transition mapping; replay recordings and drift hashes are keyed on this exact shape.

    The one exception ladder at the env boundary: typed contract and budget
    failures pass through raw; anything else becomes StepExecutionError
    (env-plane), so failures in caller machinery stay framework-fatal.
    """
    try:
        if isinstance(action, VerifyAction):
            outcome = await env.verify()
            return StepResult(
                raw_env_output=outcome.output,
                reward=outcome.reward,
                terminated=True,
                truncated=False,
                info=outcome.info,
                metrics=outcome.metrics,
                verdict=outcome.verdict,
            )
        return StepResult(
            raw_env_output=await env.execute(action),
            reward=0.0,
            terminated=False,
            truncated=False,
        )
    except (FrameworkEnvError, TimeoutError, UnscorableInfraError):
        raise
    except Exception as exc:
        raise StepExecutionError from exc


@runtime_checkable
class NetworkSnapshotEnv(Protocol):
    def pin_network_snapshot(self, token: str) -> None: ...


@runtime_checkable
class VerifyArtifactWriter(Protocol):
    def write_verify_artifacts(self, result: StepResult) -> None: ...


class TaskDefinition(Protocol):
    """One loaded task record: the instruction plus framework-facing metadata."""

    @property
    def instruction(self) -> str: ...

    @property
    def agent_timeout_sec(self) -> float | None:
        """Agent wall budget; None means the framework default applies."""
        ...

    @property
    def replay_id(self) -> str | None:
        """The key under which replay is sound: non-empty str = content
        fingerprint segment; None means not stably identified and therefore
        live-only.

        Fingerprints cover every observation-affecting env input and compose
        with ``determinism.PINS_FINGERPRINT``. Invalidation levers: step-cache
        schema salt (``_CACHE_SCHEMA``, manual key/serialization/audit change),
        env content fingerprint (automatic behavior change), and epoch counter
        (operator-forced).
        """
        ...


TaskT = TypeVar("TaskT", bound=TaskDefinition)


@dataclass(frozen=True, slots=True)
class TaskHandle:
    """One task's rollout-facing facts, bound to its env factory."""

    task_id: str
    content_id: str | None
    agent_timeout_sec: float
    make_env: Callable[[Path], TaskEnv]


@dataclass(frozen=True, slots=True)
class TaskSet(Generic[TaskT]):
    """The loaded task records plus the env factory for one benchmark."""

    kind: str
    tasks: Mapping[str, TaskT]
    env_factory: Callable[[TaskT, Path], TaskEnv]

    def content_id(self, task_id: str) -> str | None:
        """What the task's content is; None means live-only."""
        replay_id = self.tasks[task_id].replay_id
        return None if replay_id is None else f"{self.kind}:{replay_id}:{task_id}"

    def task(self, task_id: str, *, timeout_multiplier: float = 1.0) -> TaskHandle:
        """Resolve one task's handle; the single choke point between task
        definitions and the frozen loop for the scaled agent wall budget."""
        task = self.tasks[task_id]
        timeout = task.agent_timeout_sec
        if timeout is None:
            timeout = DEFAULT_AGENT_TIMEOUT_SEC
        elif (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError(
                f"{task_id}: agent_timeout_sec must be positive and finite when defined"
            )
        return TaskHandle(
            task_id=task_id,
            content_id=self.content_id(task_id),
            agent_timeout_sec=float(timeout) * timeout_multiplier,
            make_env=lambda rollout_dir: self.env_factory(task, rollout_dir),
        )


# Wraps a grader with memoization (None grades live); Any-typed to keep the core
# protocol benchmark-agnostic.
VerifyWrapper = Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class Benchmark:
    load_tasks: Callable[..., Awaitable[TaskSet]]
    secondary_metrics: tuple["SecondaryRewardMetric", ...]


def benchmark(kind: str) -> Benchmark:
    """The benchmark bundle for one config kind; the registry's single lazy point.

    Imports happen here so an unused benchmark's heavy deps are never pulled;
    the returned row holds direct references.
    """
    if kind == "swe":
        from src.env import swe

        return Benchmark(
            load_tasks=swe.load_tasks,
            secondary_metrics=swe.SECONDARY_METRICS,
        )
    if kind == "terminal_bench":
        from src.env import terminal_bench
        from src.rollout.metrics import GENERIC_SECONDARY_METRICS

        return Benchmark(
            load_tasks=terminal_bench.load_tasks,
            secondary_metrics=GENERIC_SECONDARY_METRICS,
        )
    raise KeyError(kind)


class DockerTaskEnv(Generic[TaskT]):
    """Common lifecycle for one Docker-backed benchmark task."""

    _task_workdir: ClassVar[str]
    setup_timeout_sec = DEFAULT_SETUP_TIMEOUT_SEC

    def __init__(
        self,
        *,
        task: TaskT,
        artifacts_dir: Path,
        verify_timeout_sec: float,
    ) -> None:
        self._task = task
        self._artifacts_dir = artifacts_dir
        self.verify_timeout_sec = verify_timeout_sec
        self._solve_env = self._build_solve_env(task)

    def _build_solve_env(self, task: TaskT) -> "DockerShellSession":
        raise NotImplementedError(
            f"{type(self).__name__}._build_solve_env is not implemented"
        )

    async def reset(self) -> RawEnvOutput:
        return RawEnvOutput(
            instruction=self._task.instruction,
            working_dir=self._task_workdir,
        )

    async def provision(self) -> None:
        await self._solve_env.start()

    async def execute(self, action: RunAction) -> RawEnvOutput:
        from src.env.docker_shell import run_step

        await self.provision()
        return await run_step(self._solve_env, action, default_cwd=self._task_workdir)

    async def verify(self) -> VerifyOutcome:
        raise NotImplementedError(f"{type(self).__name__}.verify is not implemented")

    async def close(self) -> None:
        await self._solve_env.close()
