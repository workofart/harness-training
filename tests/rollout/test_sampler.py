"""Experiment orchestration tests: run_experiment fan-out, infra-retry, run refs."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from git import Repo
import pytest
from src.plugins.caching import store as cc
from conftest import make_llm_provider_config, _completion, _tool_call
from src.config import (
    RunConfig,
    EnvironmentConfig,
)
from src.env.base import (
    Benchmark,
    DEFAULT_AGENT_TIMEOUT_SEC,
    TaskEnv,
    RawEnvOutput,
    RunAction,
    TaskSet,
    VerifyVerdict,
)
import src.plugins.replay.step_cache as step_cache
import src.plugins.replay as replay_plugin
import src.plugins.replay.contract as contract
import src.rollout.execution as execution
from src.env.base import StepResult
from src.rollout.records import ExperimentResult, RolloutResult
from src.rollout.store import RunStore
from src.rollout.telemetry import RolloutTelemetry
from src.llm.backend import (
    Completion,
    CompletionBackend,
    Usage,
)
import src.rollout.sampler as sampler
from src.rollout.episode import RolloutRunner
from src.rollout.sampler import run_experiment
from src.rollout.telemetry import (
    LIVE_LLM_CALLS_KEY,
    MEDIAN_LIVE_LLM_LATENCY_SEC_KEY,
    P25_LIVE_OUTPUT_TOKENS_PER_SEC_KEY,
    SUM_LIVE_LLM_LATENCY_SEC_KEY,
)

from _rollout_fixtures import (
    _FakeEnv,
    _FakeLlm,
    _rollout_config,
)


_CONFIG_PATH = "config/run_config.json"


@pytest.fixture(autouse=True)
def _cpu_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.concurrency.psutil.cpu_percent", lambda interval: 0.0)
    monkeypatch.setattr("src.concurrency.ADMISSION_INTERVAL_SEC", 0)


class _Observer:
    def __init__(self, logs: list[str] | None = None) -> None:
        self.logs = [] if logs is None else logs
        self.tasks: list[tuple[str, str]] = []

    def log(self, line: str) -> None:
        self.logs.append(line)

    def experiment_started(self, experiment_id: str) -> None:
        del experiment_id

    def task_finished(self, task_id: str, failure_mode: str) -> None:
        self.tasks.append((task_id, failure_mode))

    def experiment_finished(self, result: ExperimentResult) -> None:
        del result


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], check=True, capture_output=True, text=True, cwd=cwd)


def _patch_head_commit(
    monkeypatch: pytest.MonkeyPatch, commit_hash: str, *, dirty: bool = False
) -> None:
    monkeypatch.setattr(
        sampler,
        "Repo",
        lambda path: SimpleNamespace(
            head=SimpleNamespace(commit=SimpleNamespace(hexsha=commit_hash)),
            is_dirty=lambda: dirty,
        ),
    )


def _patch_experiment_metadata(
    monkeypatch: pytest.MonkeyPatch,
    experiment_id: str,
    *,
    commit_hash: str = "abc123",
    dirty: bool = False,
) -> list[tuple[str, str]]:
    """Patch head commit, experiment-id minting, and run-ref saving.

    Returns the list that captures ``_save_run_ref`` calls as
    ``(experiment_id, commit_hash)`` so callers can assert the ref was written.
    Env/backend/rollout patching stays with the caller.
    """
    _patch_head_commit(monkeypatch, commit_hash, dirty=dirty)
    monkeypatch.setattr(sampler, "_new_experiment_id", lambda: experiment_id)
    ref_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        sampler,
        "_save_run_ref",
        lambda repo, experiment_id, commit_hash: ref_calls.append(
            (experiment_id, commit_hash)
        ),
        raising=False,
    )
    return ref_calls


def _patch_env_create(
    monkeypatch: pytest.MonkeyPatch,
    build_env,
    *,
    agent_timeout_sec: float | None = None,
) -> None:
    class _Tasks(dict):
        def __missing__(self, task_id: str) -> SimpleNamespace:
            return SimpleNamespace(
                task_id=task_id,
                agent_timeout_sec=agent_timeout_sec,
                replay_id="version",
            )

    async def load_tasks(*, task_ids, environment, verify_wrapper=None):
        del task_ids, environment, verify_wrapper
        return TaskSet(
            kind="swe",
            tasks=_Tasks(),
            env_factory=lambda task, rollout_dir: build_env(task.task_id, rollout_dir),
        )

    def resolved(kind):
        del kind
        return Benchmark(load_tasks=load_tasks, secondary_metrics=())

    monkeypatch.setattr(sampler, "benchmark", resolved)
    monkeypatch.setattr(replay_plugin, "benchmark", resolved)


def _patch_completion_factory(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[], CompletionBackend],
) -> None:
    def make_backend(config, *, cache: bool = False):
        del config
        assert cache is True
        return factory()

    monkeypatch.setattr(sampler, "make_backend", make_backend)


def _isolate_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    if cc._STORE is not None:
        cc._STORE.close()
    monkeypatch.setattr(cc, "DB_PATH", tmp_path / "cache" / "llm_cache.db")
    monkeypatch.setattr(cc, "_DISABLED", False)
    monkeypatch.setattr(cc, "_STORE", None)


def _run_config() -> RunConfig:
    return RunConfig(
        config_path=_CONFIG_PATH,
        schema_version=13,
        training_target={"module": "src.policy.core"},
        environment=EnvironmentConfig(
            kind="swe",
            task_names=["train-a", "test-a"],
        ),
        max_steps=7,
        max_rollout_concurrency=4,
        llm_provider_config=make_llm_provider_config(),
    )


def _run_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    experiment_id: str | None = None,
    config: RunConfig | None = None,
    build_env=None,
    backend_factory=None,
    observer=None,
    agent_timeout_sec: float | None = None,
) -> tuple[ExperimentResult, RunStore]:
    _patch_head_commit(monkeypatch, "abc123")
    if experiment_id is not None:
        monkeypatch.setattr(sampler, "_new_experiment_id", lambda: experiment_id)
    monkeypatch.setattr(
        sampler,
        "_save_run_ref",
        lambda repo, run_id, commit_hash: None,
    )
    tracker = RunStore(tmp_path)
    _patch_env_create(
        monkeypatch,
        build_env or (lambda task_id, rollout_dir: _FakeEnv()),
        agent_timeout_sec=agent_timeout_sec,
    )
    _patch_completion_factory(
        monkeypatch,
        backend_factory or (lambda: _FakeLlm([_completion(_tool_call("submit"))])),
    )
    result = asyncio.run(
        run_experiment(
            run_config=config or _run_config(),
            tracker=tracker,
            observer=observer or _Observer(),
        )
    )
    return result, tracker


def _task_handle(agent_timeout_sec: float = 900.0):
    return TaskSet(
        kind="swe",
        tasks={
            "task-a": SimpleNamespace(
                task_id="task-a",
                agent_timeout_sec=agent_timeout_sec,
                replay_id="version",
            )
        },
        env_factory=lambda task, rollout_dir: _FakeEnv(),
    ).task("task-a")


def _rollout_result(
    runner: RolloutRunner, failure_mode: str, metrics: dict
) -> RolloutResult:
    return RolloutResult(
        task_id=runner.task_id,
        failure_mode=failure_mode,
        error=None,
        metrics=metrics,
        rollout_dir=str(runner.telemetry.rollout_dir),
        trace_path=str(runner.telemetry.trace_path),
        started_at=None,
        finished_at=None,
    )


def test_run_ref_keeps_discarded_commit_reachable_after_reset(tmp_path: Path) -> None:
    _git("init", "-q", cwd=tmp_path)
    _git("config", "user.email", "t@t.t", cwd=tmp_path)
    _git("config", "user.name", "t", cwd=tmp_path)
    (tmp_path / "file.txt").write_text("baseline\n")
    _git("add", "-A", cwd=tmp_path)
    _git("commit", "-q", "-m", "baseline", cwd=tmp_path)
    repo = Repo(tmp_path)
    baseline = repo.head.commit.hexsha
    (tmp_path / "file.txt").write_text("candidate\n")
    repo.git.add("--", "file.txt")
    repo.git.commit("-m", "candidate")
    candidate = repo.head.commit.hexsha

    ref = "refs/experiments/runs/exp-1"
    sampler._save_run_ref(repo, "exp-1", candidate)
    repo.head.reset(baseline, index=True, working_tree=True)

    assert repo.head.commit.hexsha == baseline
    assert repo.commit(ref).hexsha == candidate
    assert repo.git.show(f"{ref}:file.txt") == "candidate"
    # The run ref must land in its own namespace, not as a branch: `create_head`
    # prepended `refs/heads/`, leaving one pseudo-branch per run in `git branch`
    # and pushing them with the checkout. `repo.commit(ref)` resolves either way,
    # so assert the stored refname itself.
    assert repo.git.for_each_ref("--format=%(refname)", ref) == ref
    assert "refs/experiments" not in repo.git.for_each_ref(
        "--format=%(refname)", "refs/heads/"
    )


@pytest.mark.parametrize("dirty", [False, True])
def test_run_experiment_runs_configured_tasks_once_and_persists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dirty: bool,
) -> None:
    ref_calls = _patch_experiment_metadata(monkeypatch, "exp-1", dirty=dirty)
    tracker = RunStore(tmp_path)
    built_envs: list[tuple[str, Path]] = []
    logs: list[str] = []

    def build_env(
        task_id: str,
        rollout_artifact_dir: Path,
    ) -> TaskEnv:
        built_envs.append((task_id, rollout_artifact_dir))
        return _FakeEnv()

    _patch_env_create(monkeypatch, build_env)
    _patch_completion_factory(
        monkeypatch,
        lambda: _FakeLlm([_completion(_tool_call("submit"))]),
    )
    observer = _Observer(logs)
    result = asyncio.run(
        run_experiment(
            run_config=_run_config(),
            tracker=tracker,
            observer=observer,
        )
    )

    assert result.experiment_id == "exp-1"
    assert result.git_commit_hash == "abc123"
    assert result.git_dirty is dirty
    assert result.config_path == _CONFIG_PATH
    assert result.finished_at is not None
    assert result.crash_reason is None
    assert set(result.tasks) == {"train-a", "test-a"}
    assert all(rollout is not None for rollout in result.tasks.values())
    assert all(rollout.failure_mode == "solved" for rollout in result.tasks.values())
    assert built_envs == [
        ("train-a", tmp_path / "exp-1" / "tasks" / "train-a"),
        ("test-a", tmp_path / "exp-1" / "tasks" / "test-a"),
    ]
    assert logs == []
    assert observer.tasks == [("train-a", "solved"), ("test-a", "solved")]
    assert ref_calls == [("exp-1", "abc123")]
    persisted = tracker.load_experiment("exp-1")
    assert persisted.git_dirty is dirty
    assert persisted.config_path == _CONFIG_PATH
    assert persisted.config == _run_config().model_dump(mode="json")
    assert tracker.load_experiment("exp-1") == result


def test_run_experiment_holds_cross_process_task_lock_per_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overlapping experiment processes must not roll out the same task at
    once (fixed per-task docker subnets); every rollout runs inside the
    host-wide per-task lock, keyed by env kind and task id."""
    from contextlib import asynccontextmanager

    _patch_experiment_metadata(monkeypatch, "exp-1")
    locked: list[str] = []
    lock_held_during_rollout: list[bool] = []
    holding: set[str] = set()

    @asynccontextmanager
    async def fake_lock(key: str, *, on_wait=None):
        del on_wait
        locked.append(key)
        holding.add(key)
        try:
            yield
        finally:
            holding.remove(key)

    monkeypatch.setattr(sampler, "cross_process_task_lock", fake_lock)

    def build_env(task_id: str, rollout_dir: Path) -> TaskEnv:
        del rollout_dir
        lock_held_during_rollout.append(f"swe:{task_id}" in holding)
        return _FakeEnv()

    _patch_env_create(monkeypatch, build_env)
    _patch_completion_factory(
        monkeypatch, lambda: _FakeLlm([_completion(_tool_call("submit"))])
    )

    asyncio.run(
        run_experiment(
            run_config=_run_config(),
            tracker=RunStore(tmp_path),
            observer=_Observer(),
        )
    )

    assert sorted(locked) == ["swe:test-a", "swe:train-a"]
    assert lock_held_during_rollout == [True, True]
    assert holding == set()


def test_run_experiment_skips_env_wrap_when_step_cache_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def make_replay_cache_spy(**kwargs):
        del kwargs
        raise AssertionError(
            "make_replay_cache must not be called when env_step_cache is off"
        )

    monkeypatch.setattr(step_cache, "make_replay_cache", make_replay_cache_spy)
    result, _tracker = _run_case(monkeypatch, tmp_path)

    assert all(rollout is not None for rollout in result.tasks.values())
    assert all(rollout.failure_mode == "solved" for rollout in result.tasks.values())


def test_run_experiment_runs_train_and_test_together_when_slots_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_experiment_metadata(monkeypatch, "exp-ordered")
    tracker = RunStore(tmp_path)
    built_envs: list[str] = []

    async def run() -> ExperimentResult:
        train_started = asyncio.Event()
        test_started = asyncio.Event()
        release_train = asyncio.Event()

        class _BlockingTrainEnv(_FakeEnv):
            async def reset(self) -> RawEnvOutput:
                train_started.set()
                await release_train.wait()
                return await super().reset()

        class _ObservedTestEnv(_FakeEnv):
            async def reset(self) -> RawEnvOutput:
                test_started.set()
                return await super().reset()

        def build_env(
            task_id: str,
            rollout_artifact_dir: Path,
        ) -> TaskEnv:
            built_envs.append(task_id)
            if task_id == "train-a":
                return _BlockingTrainEnv()
            return _ObservedTestEnv()

        _patch_env_create(monkeypatch, build_env)
        _patch_completion_factory(
            monkeypatch,
            lambda: _FakeLlm([_completion(_tool_call("submit"))]),
        )
        task = asyncio.create_task(
            run_experiment(
                run_config=_run_config(),
                tracker=tracker,
                observer=_Observer(),
            )
        )
        await asyncio.wait_for(train_started.wait(), timeout=1.0)
        await asyncio.wait_for(test_started.wait(), timeout=1.0)
        assert set(built_envs) == {"train-a", "test-a"}
        persisted_tasks = tracker.load_experiment("exp-ordered").tasks
        assert all(rollout is None for rollout in persisted_tasks.values())
        release_train.set()
        return await task

    result = asyncio.run(run())

    assert set(result.tasks) == {"train-a", "test-a"}


def test_run_experiment_continues_remaining_tasks_after_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_experiment_metadata(monkeypatch, "exp-regression")
    tracker = RunStore(tmp_path)
    active_started = asyncio.Event()
    pending_started = asyncio.Event()
    built_envs: list[str] = []
    event_order: list[str] = []

    class _RegressingEnv(_FakeEnv):
        async def reset(self) -> RawEnvOutput:
            event_order.append("start:regressed")
            await asyncio.wait_for(active_started.wait(), timeout=1.0)
            try:
                await asyncio.wait_for(pending_started.wait(), timeout=0.01)
            except TimeoutError:
                pass
            assert not pending_started.is_set()
            return await super().reset()

        async def verify(self):
            result = await super().verify()
            event_order.append("finish:regressed")
            return result

    class _ActiveEnv(_FakeEnv):
        async def reset(self) -> RawEnvOutput:
            event_order.append("start:active")
            active_started.set()
            await asyncio.wait_for(pending_started.wait(), timeout=1.0)
            return await super().reset()

    class _PendingEnv(_FakeEnv):
        async def reset(self) -> RawEnvOutput:
            event_order.append("start:pending")
            pending_started.set()
            return await super().reset()

    def build_env(
        task_id: str,
        rollout_artifact_dir: Path,
    ) -> TaskEnv:
        del rollout_artifact_dir
        built_envs.append(task_id)
        if task_id == "regressed":
            return _RegressingEnv(
                verify_result=StepResult(
                    raw_env_output=RawEnvOutput(stdout="not solved\n"),
                    reward=0.0,
                    terminated=True,
                    truncated=False,
                    verdict=VerifyVerdict(completed=True, passed=False, error=None),
                ),
            )
        if task_id == "active":
            return _ActiveEnv()
        return _PendingEnv()

    run_config = _run_config().model_copy(
        update={
            "environment": EnvironmentConfig(
                kind="swe",
                task_names=["regressed", "active", "pending"],
            ),
            "max_rollout_concurrency": 2,
        }
    )

    _patch_env_create(monkeypatch, build_env)
    _patch_completion_factory(
        monkeypatch,
        lambda: _FakeLlm([_completion(_tool_call("submit"))]),
    )
    result = asyncio.run(
        run_experiment(
            run_config=run_config,
            tracker=tracker,
            observer=_Observer(),
        )
    )

    rollouts = result.tasks
    assert rollouts["regressed"].failure_mode == "verified_rejected"
    assert rollouts["active"].failure_mode == "solved"
    assert rollouts["pending"].failure_mode == "solved"
    assert built_envs == ["regressed", "active", "pending"]
    assert set(event_order[:2]) == {"start:regressed", "start:active"}
    assert event_order.index("finish:regressed") < event_order.index("start:pending")
    assert result.finished_at is not None
    assert result.crash_reason is None
    assert {rollout.failure_mode for rollout in result.tasks.values()} == {
        "verified_rejected",
        "solved",
    }
    assert tracker.load_experiment("exp-regression") == result


class _DriftEnv(_FakeEnv):
    def __init__(self, stdout: str, *, error: bool = False) -> None:
        super().__init__()
        self.stdout = stdout
        self.error = error

    async def execute(self, action: RunAction) -> RawEnvOutput:
        if self.error:
            raise RuntimeError("retry infra boom")
        self.exec_calls.append(action.command)
        return RawEnvOutput(exit_code=0, stdout=self.stdout)


@pytest.mark.parametrize("retry_crashes", [False, True], ids=["heals", "retry-crashes"])
def test_execution_drift_retry_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retry_crashes: bool,
) -> None:
    _isolate_cache(monkeypatch, tmp_path)
    namespace = contract.namespace_for("swe:version:task-a")
    action = RunAction(command="echo old", timeout_sec=240)

    async def seed_old_cache() -> None:
        assert namespace is not None
        await step_cache.ReplayCache(
            namespace=namespace, epoch=0, env=_DriftEnv("old\n")
        ).step_run(action)

    asyncio.run(seed_old_cache())
    built_envs: list[str] = []

    def build_env(task_id: str, rollout_artifact_dir: Path) -> TaskEnv:
        del rollout_artifact_dir
        built_envs.append(task_id)
        return _DriftEnv("new\n", error=retry_crashes and len(built_envs) == 2)

    backend_builds: list[int] = []

    def build_backend() -> CompletionBackend:
        backend_builds.append(len(backend_builds) + 1)
        return _FakeLlm(
            [
                _completion(_tool_call("run", command="echo old", timeout_sec=240)),
                _completion(_tool_call("submit")),
            ]
        )

    real_make_replay_cache = step_cache.make_replay_cache
    replay_cache_builds: list[str] = []

    async def spy_make_replay_cache(*, content_id, env):
        replay_cache_builds.append(content_id)
        return await real_make_replay_cache(content_id=content_id, env=env)

    monkeypatch.setattr(step_cache, "make_replay_cache", spy_make_replay_cache)
    config = _rollout_config()
    config = config.model_copy(
        update={"plugins": config.plugins.model_copy(update={"execution": "replay"})}
    )
    result, _tracker = _run_case(
        monkeypatch,
        tmp_path,
        experiment_id="exp-drift",
        config=config,
        build_env=build_env,
        backend_factory=build_backend,
    )
    rollout = result.tasks["task-a"]
    assert built_envs == ["task-a", "task-a"]
    assert backend_builds == [1, 2]
    assert replay_cache_builds == ["swe:version:task-a"]
    [retry] = rollout.infra_retries
    assert retry["kind"] == "execution_drift"
    # Plugin vocabulary (namespace/epoch) lives in the artifact, not the record.
    assert "epoch" not in retry
    if retry_crashes:
        assert (rollout.failure_mode, rollout.failure_origin, rollout.error) == (
            "unscorable_infra",
            None,
            "RuntimeError: retry infra boom",
        )
    else:
        assert rollout.failure_mode == "solved"
        artifact = json.loads(Path(retry["artifact_path"]).read_text())
        assert artifact["epoch"] == 0
        assert artifact["action"] == {
            "kind": "run",
            "command": "echo old",
            "cwd": None,
            "timeout_sec": 240,
        }
        assert artifact["recorded"]["raw_env_output"]["stdout"] == "old\n"
        assert artifact["live"]["raw_env_output"]["stdout"] == "new\n"


def test_drift_retry_keeps_final_policy_crash_scorable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def run(self: RolloutRunner) -> RolloutResult:
        nonlocal attempts
        attempts += 1
        self.telemetry.trace_path.touch()
        if attempts == 1:
            raise execution.ExecutionDriftError(
                action_index=1,
                diagnostic={"namespace": "env:test", "epoch": 0},
            )
        return RolloutResult(
            task_id=self.task_id,
            failure_mode="crash",
            failure_origin="policy",
            error="RuntimeError: candidate bug",
            metrics={},
            rollout_dir=str(self.telemetry.rollout_dir),
            trace_path=str(self.telemetry.trace_path),
            started_at=None,
            finished_at=None,
        )

    async def make_replay_cache(**kwargs):
        return execution.LiveStepExecutor(kwargs["env"])

    monkeypatch.setattr(RolloutRunner, "run", run)
    monkeypatch.setattr(step_cache, "make_replay_cache", make_replay_cache)
    _patch_completion_factory(monkeypatch, lambda: _FakeLlm([]))
    task = _task_handle()
    config = _rollout_config()
    config = config.model_copy(
        update={"plugins": config.plugins.model_copy(update={"execution": "replay"})}
    )

    result = asyncio.run(
        sampler._run_rollout_with_infra_retries(
            task=task,
            rollout_dir=tmp_path,
            trace_path=tmp_path / "trace.jsonl",
            run_config=config,
            execution=execution.resolve_execution(config),
            log=lambda message: None,
        )
    )

    assert attempts == 2
    assert result.failure_mode == "crash"
    assert result.failure_origin == "policy"


def test_task_agent_timeout_is_passed_without_config_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[float] = []

    async def run(self: RolloutRunner) -> RolloutResult:
        assert self.agent_timeout_sec is not None
        captured.append(self.agent_timeout_sec)
        return _rollout_result(self, "solved", {})

    monkeypatch.setattr(RolloutRunner, "run", run)
    monkeypatch.setattr(sampler, "make_backend", lambda config, cache: _FakeLlm([]))

    result = asyncio.run(
        sampler._run_rollout_with_infra_retries(
            task=_task_handle(),
            rollout_dir=tmp_path,
            trace_path=tmp_path / "trace.jsonl",
            run_config=_rollout_config(),
            execution=execution.EagerExecution(),
            log=lambda message: None,
        )
    )

    assert result.failure_mode == "solved"
    assert captured == [900.0]


def _timeout_taskset() -> TaskSet:
    return TaskSet(
        kind="swe",
        tasks={
            "defined": SimpleNamespace(agent_timeout_sec=900.0, replay_id=None),
            "silent": SimpleNamespace(agent_timeout_sec=None, replay_id=None),
        },
        env_factory=lambda task, rollout_dir: _FakeEnv(),
    )


def test_task_handle_defaults_timeout_when_task_defines_none() -> None:
    """A task definition without an agent budget falls back to the harness
    default at the single validation choke point; defined budgets always win."""
    taskset = _timeout_taskset()

    assert taskset.task("defined").agent_timeout_sec == 900.0
    assert taskset.task("silent").agent_timeout_sec == DEFAULT_AGENT_TIMEOUT_SEC


def test_task_handle_applies_multiplier_to_defined_and_default() -> None:
    taskset = _timeout_taskset()

    assert taskset.task("defined", timeout_multiplier=4.0).agent_timeout_sec == 3600.0
    assert (
        taskset.task("silent", timeout_multiplier=4.0).agent_timeout_sec
        == 4.0 * DEFAULT_AGENT_TIMEOUT_SEC
    )


def test_run_experiment_passes_scaled_task_timeout_to_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[float] = []

    async def run(self: RolloutRunner) -> RolloutResult:
        captured.append(self.agent_timeout_sec)
        return _rollout_result(self, "solved", {})

    monkeypatch.setattr(RolloutRunner, "run", run)
    config = _run_config().model_copy(update={"agent_timeout_multiplier": 4.0})

    _run_case(
        monkeypatch,
        tmp_path,
        config=config,
        agent_timeout_sec=900.0,
    )

    assert captured == [3600.0, 3600.0]


def test_task_load_failure_marks_experiment_crashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task-set load failures happen after the experiment record exists, so the
    crash is captured on it instead of dying recordless."""
    _patch_head_commit(monkeypatch, "abc123")
    monkeypatch.setattr(sampler, "_new_experiment_id", lambda: "exp-load-crash")
    monkeypatch.setattr(
        sampler, "_save_run_ref", lambda repo, run_id, commit_hash: None
    )

    async def load_tasks(*, task_ids, environment, verify_wrapper=None):
        del task_ids, environment, verify_wrapper
        raise RuntimeError("dataset sync failed")

    monkeypatch.setattr(
        sampler,
        "benchmark",
        lambda kind: Benchmark(load_tasks=load_tasks, secondary_metrics=()),
    )
    tracker = RunStore(tmp_path)

    with pytest.raises(RuntimeError, match="dataset sync failed"):
        asyncio.run(
            run_experiment(
                run_config=_run_config(),
                tracker=tracker,
                observer=_Observer(),
            )
        )

    crashed = tracker.load_experiment("exp-load-crash")
    assert crashed.crash_reason is not None
    assert "dataset sync failed" in crashed.crash_reason


def test_invalid_taskset_agent_timeout_fails_before_env_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_calls: list[str] = []

    def build_env(task_id: str, rollout_dir: Path) -> _FakeEnv:
        env_calls.append(task_id)
        return _FakeEnv()

    with pytest.raises(ValueError, match="agent_timeout_sec"):
        _run_case(
            monkeypatch,
            tmp_path,
            build_env=build_env,
            agent_timeout_sec=0.0,
        )

    assert env_calls == []


@pytest.mark.parametrize(
    ("output_tokens", "deterministic", "expected_calls", "expected_mode"),
    [
        pytest.param(45, True, 2, "solved", id="slow-deterministic-retries"),
        pytest.param(45, False, 1, "hit_timeout", id="nondeterministic-no-retry"),
        pytest.param(6000, True, 1, "hit_timeout", id="healthy-decode-no-retry"),
    ],
)
def test_slow_timeout_retry_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_tokens: int,
    deterministic: bool,
    expected_calls: int,
    expected_mode: str,
) -> None:
    run_calls = 0

    async def run(self: RolloutRunner) -> RolloutResult:
        nonlocal run_calls
        run_calls += 1
        if expected_mode == "solved" and run_calls == 2:
            return _rollout_result(self, "solved", {})
        metrics = _record_completion_series(
            self.telemetry,
            [(150.0, output_tokens)] * 4,
        )
        return _rollout_result(self, "hit_timeout", metrics)

    monkeypatch.setattr(RolloutRunner, "run", run)
    config = _rollout_config()
    if not deterministic:
        config = config.model_copy(
            update={"llm_provider_config": make_llm_provider_config(seed=None)}
        )
    result, _tracker = _run_case(
        monkeypatch,
        tmp_path,
        experiment_id="exp-slow-policy",
        config=config,
        backend_factory=lambda: _FakeLlm([]),
        agent_timeout_sec=500.0,
    )
    rollout = result.tasks["task-a"]
    assert (run_calls, rollout.failure_mode) == (expected_calls, expected_mode)
    if expected_calls == 1:
        assert rollout.infra_retries == []
    else:
        [retry] = rollout.infra_retries
        assert {
            key: retry[key]
            for key in (
                "kind",
                "live_llm_calls",
                "sum_live_llm_latency_sec",
                "median_live_llm_latency_sec",
                "p25_live_output_tokens_per_sec",
            )
        } == {
            "kind": "slow_llm_timeout",
            "live_llm_calls": 4,
            "sum_live_llm_latency_sec": 600.0,
            "median_live_llm_latency_sec": 150.0,
            "p25_live_output_tokens_per_sec": 0.3,
        }
        assert Path(retry["archived_trace_path"]).exists()


def test_run_experiment_marks_crashed_and_reraises_panel_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_experiment_metadata(monkeypatch, "exp-crash")
    tracker = RunStore(tmp_path)
    unhandled_contexts: list[dict[str, Any]] = []

    def build_env(
        task_id: str,
        rollout_artifact_dir: Path,
    ) -> TaskEnv:
        del task_id, rollout_artifact_dir
        raise RuntimeError("factory boom")

    _patch_env_create(monkeypatch, build_env)
    _patch_completion_factory(
        monkeypatch,
        lambda: _FakeLlm([_completion(_tool_call("submit"))]),
    )

    async def run() -> None:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(
            lambda _loop, context: unhandled_contexts.append(context)
        )
        with pytest.raises(RuntimeError, match="factory boom"):
            await run_experiment(
                run_config=_run_config(),
                tracker=tracker,
                observer=_Observer(),
            )

    asyncio.run(run())

    assert unhandled_contexts == []
    result = tracker.load_experiment("exp-crash")
    assert result.crash_reason == "RuntimeError: factory boom"
    assert result.finished_at is not None
    assert result.tasks["train-a"] is None


def _record_completion_series(
    telemetry: RolloutTelemetry, rows: list[tuple[float, int]]
) -> dict[str, int | float]:
    for latency_sec, completion_tokens in rows:
        telemetry.event(
            "completion_received",
            request_messages=[],
            request_tools=[],
            completion=dataclasses.asdict(
                Completion(
                    usage=Usage(completion_tokens=completion_tokens),
                )
            ),
            llm_latency_sec=latency_sec,
        )
    latencies = [latency_sec for latency_sec, _ in rows]
    rates = [completion_tokens / latency_sec for latency_sec, completion_tokens in rows]
    return {
        LIVE_LLM_CALLS_KEY: len(latencies),
        SUM_LIVE_LLM_LATENCY_SEC_KEY: sum(latencies),
        MEDIAN_LIVE_LLM_LATENCY_SEC_KEY: sorted(latencies)[len(latencies) // 2],
        P25_LIVE_OUTPUT_TOKENS_PER_SEC_KEY: rates[0],
    }
