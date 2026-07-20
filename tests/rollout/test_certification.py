"""Tests for certification, shared replay identity, and chain artifacts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import TEST_MEASUREMENT_IDENTITY

from src.plugins.caching import store as cc
import src.plugins.replay.audit as audit_module
import src.plugins.replay.contract as contract
from src.env.base import (
    RawEnvOutput,
    RunAction,
    StepExecutionError,
    TaskSet,
    VerifyOutcome,
    VerifyVerdict,
    VerifyAction,
    StepResult,
)
from src.config import RunConfig
from src.plugins.replay import ReplayExecution
from src.plugins.replay.audit import _certify_tasks, exclude_nondeterministic_tasks
from src.rollout.certification import (
    ChainStep,
    action_payload,
    scrubbed_hash,
    resolve_measurement_identity,
    write_chain,
)
from src.rollout.execution import EagerExecution, resolve_execution
from src.rollout.records import ExperimentResult, RolloutResult, TaskCertification
from src.rollout.store import RunStore

from _rollout_fixtures import _rollout_config

from conftest import _result


def test_cache_namespace_content_id_contract():
    assert contract.namespace_for(None) is None
    assert contract.namespace_for("kind:rev:task-a") == (
        f"{contract._CACHE_SCHEMA}:kind:rev:task-a"
    )


def _regime_digest(*lines: str) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def test_measurement_identity_live_regime_uses_effective_model() -> None:
    base = _rollout_config()
    config = base.model_copy(
        update={
            "environment": base.environment.model_copy(
                update={"task_names": ["task-b", "task-a"]}
            ),
        }
    )

    identity = asyncio.run(resolve_measurement_identity(config, EagerExecution()))

    canonical = config.measurement_identity_payload()
    canonical["environment"]["task_names"] = ["task-a", "task-b"]
    expected_config = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    expected_regime = _regime_digest("task-a:live", "task-b:live")
    assert identity.effective_config_digest == expected_config
    assert identity.provider_revision == (
        "openai_compatible:http://127.0.0.1:18000/v1:gpt-test"
    )
    assert identity.replay_regime_digest == expected_regime


def test_measurement_identity_ignores_panel_order_and_trainer_only_fields() -> None:
    base = _rollout_config()

    def identity(
        task_names: list[str],
        *,
        proposer_visible: tuple[str, ...] = ("/src/", "/tests/"),
        extra_patch_paths: tuple[str, ...] = (),
    ):
        config = base.model_copy(
            update={
                "environment": base.environment.model_copy(
                    update={"task_names": task_names}
                ),
                "training_target": base.training_target.model_copy(
                    update={
                        "proposer_visible": proposer_visible,
                        "extra_patch_paths": extra_patch_paths,
                    }
                ),
            }
        )
        return asyncio.run(resolve_measurement_identity(config, EagerExecution()))

    original = identity(["task-b", "task-a"])

    assert identity(["task-a", "task-b"]).digest == original.digest
    assert identity(["task-a", "task-c"]).effective_config_digest != (
        original.effective_config_digest
    )
    # Trainer-only fields steer candidate production, not the measured rollout;
    # neither forks identity (even proposer_visible order, otherwise significant).
    assert (
        identity(
            ["task-b", "task-a"], proposer_visible=("/tests/", "/src/")
        ).effective_config_digest
        == original.effective_config_digest
    )
    assert (
        identity(
            ["task-b", "task-a"],
            extra_patch_paths=("tests/policy/test_core_impl.py",),
        ).effective_config_digest
        == original.effective_config_digest
    )


def test_measurement_identity_collapses_config_spelling() -> None:
    base = _rollout_config()
    provider = base.model_dump(mode="json")["llm_provider_config"]
    minimal = {
        "schema_version": 13,
        "training_target": {"module": "src.policy.core"},
        "environment": {"kind": "swe", "task_names": ["task-a"]},
        "llm_provider_config": provider,
    }
    explicit = {
        **minimal,
        "training_target": {"module": "src.policy.core", "extra_patch_paths": []},
        "max_steps": 50,
    }

    def digest(config: RunConfig) -> str:
        return asyncio.run(
            resolve_measurement_identity(config, EagerExecution())
        ).effective_config_digest

    # Defaults spelled out vs omitted, and the load path, never fork identity.
    assert digest(RunConfig.model_validate(minimal)) == digest(
        RunConfig.model_validate(explicit)
    )
    assert digest(base.model_copy(update={"config_path": "elsewhere.yaml"})) == digest(
        base
    )


def test_measurement_identity_resolves_per_task_epochs(monkeypatch) -> None:
    taskset = TaskSet(
        kind="swe",
        tasks={
            "task-a": SimpleNamespace(replay_id="rev"),
            "task-b": SimpleNamespace(replay_id=None),
        },
        env_factory=lambda task, rollout_dir: None,
    )

    async def load_tasks(*, task_ids, environment, verify_wrapper=None):
        assert task_ids == ("task-b", "task-a")
        assert verify_wrapper is not None
        return taskset

    async def get_counter(key: str) -> int:
        assert key.endswith("step-result-v3-metrics:swe:rev:task-a")
        return 4

    import src.plugins.replay as replay_plugin

    monkeypatch.setattr(
        replay_plugin,
        "benchmark",
        lambda kind: SimpleNamespace(load_tasks=load_tasks),
    )
    monkeypatch.setattr(cc, "get_counter", get_counter)
    config = _rollout_config()
    config = config.model_copy(
        update={
            "plugins": config.plugins.model_copy(update={"execution": "replay"}),
            "environment": config.environment.model_copy(
                update={"task_names": ["task-b", "task-a"]}
            ),
        }
    )

    identity = asyncio.run(
        resolve_measurement_identity(config, ReplayExecution(config))
    )

    namespace = contract.namespace_for("swe:rev:task-a")
    expected = _regime_digest(f"{namespace}:4", "task-b:live")
    assert identity.replay_regime_digest == expected


class _CertEnv:
    setup_timeout_sec = 600.0
    verify_timeout_sec = 900.0

    def __init__(
        self,
        outputs: list[RawEnvOutput | BaseException],
        *,
        verify: VerifyOutcome | None = None,
        verify_delay_sec: float = 0.0,
    ) -> None:
        self._verify_delay_sec = verify_delay_sec
        self.outputs = list(outputs)
        self.verify_outcome = verify or VerifyOutcome(
            verdict=VerifyVerdict(completed=True, passed=True, error=None),
            output=RawEnvOutput(stdout="verified\n"),
            reward=1.0,
            info={},
        )
        self.exec_calls: list[str] = []
        self.reset_calls = 0
        self.provision_calls = 0
        self.closed = False

    async def reset(self) -> RawEnvOutput:
        self.reset_calls += 1
        return RawEnvOutput(instruction="task")

    async def provision(self) -> None:
        self.provision_calls += 1

    async def execute(self, action: RunAction) -> RawEnvOutput:
        self.exec_calls.append(action.command)
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return output

    async def verify(self) -> VerifyOutcome:
        if self._verify_delay_sec:
            await asyncio.sleep(self._verify_delay_sec)
        return self.verify_outcome

    async def close(self) -> None:
        self.closed = True


class _ScopedCertEnv(_CertEnv):
    """_CertEnv that records replay-scope wiring and when it happened."""

    def __init__(
        self, outputs: list[RawEnvOutput | BaseException], *, verify=None
    ) -> None:
        super().__init__(outputs, verify=verify)
        # reset_calls proves snapshot pinning precedes reset(), matching live replay setup.
        self.pinned_snapshots: list[tuple[str, int]] = []

    def pin_network_snapshot(self, token: str) -> None:
        self.pinned_snapshots.append((token, self.reset_calls))


class _SlowCertEnv(_CertEnv):
    async def execute(self, action: RunAction) -> RawEnvOutput:
        await asyncio.sleep(1)
        return await super().execute(action)


def _baseline(task_ids: tuple[str, ...] = ("task-a",)) -> ExperimentResult:
    return ExperimentResult(
        experiment_id="baseline-1",
        git_commit_hash="baseline",
        measurement_identity=TEST_MEASUREMENT_IDENTITY,
        git_dirty=False,
        config_path="config/run.json",
        started_at="2026-07-10T00:00:00Z",
        finished_at="2026-07-10T00:01:00Z",
        tasks={
            task_id: RolloutResult(
                task_id=task_id,
                failure_mode="verified_rejected",
                error=None,
                metrics={},
                rollout_dir=None,
                trace_path=None,
                started_at=None,
                finished_at=None,
            )
            for task_id in task_ids
        },
    )


def _certified_baseline(
    certification: dict[str, tuple[str, str]],
) -> ExperimentResult:
    baseline = _baseline(tuple(certification))
    return baseline.model_copy(
        update={
            "determinism_certification": {
                task_id: TaskCertification(chain_digest=digest, verdict=verdict)
                for task_id, (digest, verdict) in certification.items()
            }
        }
    )


def _unexpected_certify(**kwargs):
    del kwargs
    raise AssertionError("recertified")


def _run_row(command: str, output: RawEnvOutput, *, timed_out: bool = False) -> dict:
    action = RunAction(command=command)
    result = StepResult(
        raw_env_output=output,
        reward=0.0,
        terminated=False,
        truncated=False,
    )
    return {
        "action": action_payload(action),
        "audit_hash": scrubbed_hash(result),
        "timed_out": timed_out,
    }


def _chain_path(tracker: RunStore) -> Path:
    return tracker.task_dir("baseline-1", "task-a") / "infra/determinism_chain.jsonl"


def _write_chain(tracker: RunStore, rows: list[dict]) -> None:
    path = _chain_path(tracker)
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fork_artifact(tracker: RunStore) -> dict:
    path = tracker.task_dir("baseline-1", "task-a") / "infra/determinism_fork.json"
    return json.loads(path.read_text())


def _forked_artifact(tmp_path: Path, env: _CertEnv, rows: list[dict]) -> dict:
    excluded, tracker = _run_check(tmp_path, env, rows)
    assert excluded == {"task-a": "forked"}
    return _fork_artifact(tracker)


def test_write_chain_preserves_compact_canonical_artifact_bytes(tmp_path: Path) -> None:
    run_result = _result("ran\n")
    verify_result = _result(
        "verified\n",
        reward=1.0,
        terminated=True,
        verdict=VerifyVerdict(completed=True, passed=True, error=None),
    )
    path = tmp_path / "infra/determinism_chain.jsonl"

    write_chain(
        path,
        [
            ChainStep(
                env_action=RunAction(command="pwd", cwd="/work", timeout_sec=3.0),
                step_result=run_result,
                timed_out=False,
            ),
            ChainStep(VerifyAction(), verify_result, timed_out=False),
        ],
    )

    assert path.read_text() == (
        '{"action":{"command":"pwd","cwd":"/work","kind":"run",'
        f'"timeout_sec":3.0}},"audit_hash":"{scrubbed_hash(run_result)}",'
        '"timed_out":false}\n'
        '{"action":{"kind":"verify"},"timed_out":false,'
        '"verdict":{"passed":true,"reward":1.0}}\n'
    )


def test_write_chain_persists_verify_timeout_bit(tmp_path: Path) -> None:
    # A verify (submit) action can time out; episode records the synthetic
    # truncated placeholder with timed_out=True. The bit must survive the
    # write->read round trip so the audit classifies the chain no_chain
    # instead of re-executing an unreplayable trace as a spurious fork.
    timed_out_verify = StepResult(
        raw_env_output=RawEnvOutput(stderr="timed out"),
        reward=0.0,
        terminated=False,
        truncated=True,
    )

    write_chain(
        tmp_path / "infra/determinism_chain.jsonl",
        [ChainStep(VerifyAction(), timed_out_verify, timed_out=True)],
    )

    _rows, _digest, has_timeout = audit_module._load_chain(tmp_path)
    assert has_timeout is True


def _run_check(
    tmp_path: Path,
    env: _CertEnv,
    rows: list[dict] | None,
    *,
    run_config=None,
    agent_timeout_sec: float | None = 1.0,
    agent_timeout_multiplier: float = 1.0,
    raw_chain: str | None = None,
    replay_id: str | None = "",
    log: Callable[[str], None] | None = None,
):
    log = log or (lambda _line: None)
    run_config = run_config or _rollout_config()
    tracker = RunStore(tmp_path)
    if rows is not None:
        _write_chain(tracker, rows)
    if raw_chain is not None:
        path = _chain_path(tracker)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw_chain)

    def make_env(task, rollout_dir: Path) -> _CertEnv:
        assert rollout_dir == (
            tracker.task_dir("baseline-1", "task-a") / "determinism_check"
        )
        return env

    taskset = TaskSet(
        kind="swe",
        tasks={
            "task-a": SimpleNamespace(
                agent_timeout_sec=agent_timeout_sec, replay_id=replay_id
            ),
        },
        env_factory=make_env,
    )

    async def load_tasks(*, task_ids, environment, verify_wrapper=None):
        assert task_ids == ("task-a",)
        assert environment == run_config.environment
        assert verify_wrapper is not None
        return taskset

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(cc, "DB_PATH", tmp_path / "cache" / "llm_cache.db")
        monkeypatch.setattr(cc, "_DISABLED", False)
        monkeypatch.setattr(cc, "_STORE", None)
        monkeypatch.setattr(
            audit_module,
            "benchmark",
            lambda kind: SimpleNamespace(load_tasks=load_tasks),
        )
        certification = asyncio.run(
            _certify_tasks(
                run_config=run_config.model_copy(
                    update={"agent_timeout_multiplier": agent_timeout_multiplier}
                ),
                tracker=tracker,
                baseline=_baseline(),
                task_ids=("task-a",),
                log=log,
            )
        )
    excluded = {
        task_id: item.verdict
        for task_id, item in certification.items()
        if item.verdict != "deterministic"
    }
    return excluded, tracker


def test_resolve_execution_composes_replay_and_degrades_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    base = _rollout_config()
    replay = base.model_copy(
        update={"plugins": base.plugins.model_copy(update={"execution": "replay"})}
    )
    monkeypatch.setattr(cc, "DB_PATH", tmp_path / "cache" / "llm_cache.db")
    monkeypatch.setattr(cc, "_DISABLED", False)
    monkeypatch.setattr(cc, "_STORE", None)

    assert isinstance(resolve_execution(base), EagerExecution)
    assert isinstance(resolve_execution(replay), ReplayExecution)

    # Cache unavailability may degrade replay to eager, but it must warn.
    monkeypatch.setattr(cc, "_DISABLED", True)
    with caplog.at_level(logging.WARNING, logger="src.rollout.execution"):
        assert isinstance(resolve_execution(replay), EagerExecution)
    assert any("degraded to eager" in record.message for record in caplog.records)


_CHECK_SCOPE_CASES = {
    "scoped-before-reset": (
        "version",
        [
            (
                contract._network_snapshot_token(
                    namespace=contract.namespace_for("swe:version:task-a"),
                    epoch=0,
                ),
                0,
            )
        ],
    ),
    "live-only": (None, []),
}


@pytest.mark.parametrize("case", _CHECK_SCOPE_CASES.values(), ids=_CHECK_SCOPE_CASES)
def test_check_matching_chain_applies_available_scope(tmp_path: Path, case) -> None:
    replay_id, scopes = case
    output = RawEnvOutput(stdout="same\n", exit_code=0)
    env = _ScopedCertEnv([output])
    config = _rollout_config()
    config = config.model_copy(
        update={"plugins": config.plugins.model_copy(update={"execution": "replay"})}
    )
    excluded, _ = _run_check(
        tmp_path,
        env,
        [_run_row("pwd", output)],
        run_config=config,
        replay_id=replay_id,
    )
    assert excluded == {}
    assert env.pinned_snapshots == scopes
    assert env.reset_calls == 1
    assert env.closed is True


def test_check_hash_mismatch_writes_fork_artifact(tmp_path: Path) -> None:
    recorded = RawEnvOutput(stdout="recorded\n", exit_code=0)
    live = RawEnvOutput(stdout="live\n", stderr="err\n", exit_code=2)
    logs: list[str] = []

    excluded, tracker = _run_check(
        tmp_path,
        _CertEnv([live]),
        [_run_row("pwd", recorded)],
        replay_id="version",
        log=logs.append,
    )
    assert excluded == {"task-a": "forked"}
    artifact = _fork_artifact(tracker)
    namespace = contract.namespace_for("swe:version:task-a")
    assert artifact["replay_namespace"] == namespace
    assert logs == [
        f"fork hint: task-a — stale recording? bump: uv run python -m "
        f"src.plugins.replay.bump_epoch '{namespace}'; forks persisting after "
        "re-record are real nondeterminism (fix the scrub, not the epoch)"
    ]
    assert artifact["task"] == "task-a"
    assert artifact["action_index"] == 1
    assert artifact["action"] == {
        "kind": "run",
        "command": "pwd",
        "cwd": None,
        "timeout_sec": None,
    }
    assert artifact["recorded_audit_hash"] == _run_row("pwd", recorded)["audit_hash"]
    assert artifact["live"]["raw_env_output"] == {
        "instruction": "",
        "working_dir": None,
        "exit_code": 2,
        "stdout": "live\n",
        "stderr": "err\n",
    }


def test_check_verify_verdict_mismatch_is_forked(tmp_path: Path) -> None:
    verify = VerifyOutcome(
        verdict=VerifyVerdict(completed=True, passed=False, error=None),
        output=RawEnvOutput(),
        reward=0.0,
        info={},
    )
    rows = [{"action": {"kind": "verify"}, "verdict": {"passed": True, "reward": 1.0}}]

    artifact = _forked_artifact(tmp_path, _CertEnv([], verify=verify), rows)
    assert artifact["verdict_mismatch"] == {
        "recorded": {"passed": True, "reward": 1.0},
        "live": {"passed": False, "reward": 0.0},
    }


def test_check_verify_row_runs_on_verify_budget_not_agent_budget(
    tmp_path: Path,
) -> None:
    """The audit mirrors the live loop: grading never spends the agent budget."""
    rows = [{"action": {"kind": "verify"}, "verdict": {"passed": True, "reward": 1.0}}]
    env = _CertEnv([], verify_delay_sec=0.05)

    excluded, _ = _run_check(tmp_path, env, rows, agent_timeout_sec=0.01)

    assert excluded == {}


def test_check_run_rows_use_multiplied_agent_budget(tmp_path: Path) -> None:
    class _SlowCertEnv(_CertEnv):
        async def execute(self, action: RunAction) -> RawEnvOutput:
            await asyncio.sleep(0.05)
            return await super().execute(action)

    output = RawEnvOutput(stdout="same\n", exit_code=0)
    env = _SlowCertEnv([output])

    excluded, _ = _run_check(
        tmp_path,
        env,
        [_run_row("pwd", output)],
        agent_timeout_sec=0.01,
        agent_timeout_multiplier=10.0,
    )

    assert excluded == {}


def test_check_missing_chain_is_no_chain(tmp_path: Path) -> None:
    excluded, _ = _run_check(tmp_path, _CertEnv([]), None)

    assert excluded == {"task-a": "no_chain"}


def test_check_corrupt_chain_raises(tmp_path: Path) -> None:
    with pytest.raises(json.JSONDecodeError):
        _run_check(tmp_path, _CertEnv([]), None, raw_chain="not json\n")


def test_check_recorded_timeout_stops_before_untrusted_suffix(tmp_path: Path) -> None:
    first = RawEnvOutput(stdout="first\n")
    env = _CertEnv([first])
    rows = [
        _run_row("first", first),
        _run_row("timeout", RawEnvOutput(stderr="timed out"), timed_out=True),
        _run_row("untrusted", RawEnvOutput(stdout="never")),
    ]

    excluded, _ = _run_check(tmp_path, env, rows)

    assert excluded == {"task-a": "no_chain"}
    assert env.exec_calls == []


_CHECK_FAILURE_CASES = {
    # The env boundary translates the raw failure; the audit lets it fly.
    "env-error": (
        lambda: _CertEnv([RuntimeError("container died")]),
        1.0,
        StepExecutionError,
        None,
    ),
    "deadline": (
        lambda: _SlowCertEnv([RawEnvOutput(stdout="same")]),
        0.01,
        TimeoutError,
        None,
    ),
}


@pytest.mark.parametrize(
    "case", _CHECK_FAILURE_CASES.values(), ids=_CHECK_FAILURE_CASES
)
def test_check_infra_failure_crashes_loud(tmp_path: Path, case) -> None:
    # Audit failures propagate without retries.
    make_env, timeout, error_type, match = case
    env = make_env()
    with pytest.raises(error_type, match=match):
        _run_check(
            tmp_path,
            env,
            [_run_row("pwd", RawEnvOutput(stdout="same"))],
            agent_timeout_sec=timeout,
        )
    assert env.reset_calls == 1


def test_check_cancellation_propagates(tmp_path: Path) -> None:
    with pytest.raises(asyncio.CancelledError):
        _run_check(
            tmp_path,
            _CertEnv([asyncio.CancelledError()]),
            [_run_row("pwd", RawEnvOutput(stdout="same"))],
        )


def test_check_taskset_load_failure_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def load_tasks(*, task_ids, environment, verify_wrapper=None):
        del task_ids, environment, verify_wrapper
        raise RuntimeError("taskset unavailable")

    monkeypatch.setattr(
        "src.plugins.replay.audit.benchmark",
        lambda kind: SimpleNamespace(load_tasks=load_tasks),
    )

    with pytest.raises(RuntimeError, match="taskset unavailable"):
        asyncio.run(
            exclude_nondeterministic_tasks(
                run_config=_rollout_config(),
                tracker=RunStore(tmp_path),
                baseline=_baseline(),
                log=lambda line: None,
            )
        )


def test_exclude_nondeterministic_tasks_runs_persists_reports_and_returns_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = RunStore(tmp_path)
    baseline = _baseline(("task-a", "task-b"))
    logs: list[str] = []

    async def check(**kwargs):
        assert kwargs["baseline"] is baseline
        return {
            "task-a": TaskCertification(chain_digest="a", verdict="deterministic"),
            "task-b": TaskCertification(chain_digest="b", verdict="forked"),
        }

    monkeypatch.setattr("src.plugins.replay.audit._certify_tasks", check)

    checked, panel = asyncio.run(
        exclude_nondeterministic_tasks(
            run_config=_rollout_config(),
            tracker=tracker,
            baseline=baseline,
            log=logs.append,
        )
    )

    assert panel == ("task-a",)
    assert tracker.load_experiment("baseline-1") == checked
    assert logs == [
        "determinism: excluded 1/2 nondeterministic tasks",
        "excluded: task-b (forked) "
        f"{tracker.task_dir('baseline-1', 'task-b')}/infra/determinism_fork.json",
    ]


def test_exclude_nondeterministic_tasks_cached_reports_without_rechecking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = RunStore(tmp_path)
    baseline = _certified_baseline(
        {"task-a": ("a", "deterministic"), "task-b": ("b", "no_chain")}
    )
    monkeypatch.setattr(audit_module, "_certify_tasks", _unexpected_certify)
    logs: list[str] = []

    returned, panel = asyncio.run(
        exclude_nondeterministic_tasks(
            run_config=_rollout_config(),
            tracker=tracker,
            baseline=baseline,
            log=logs.append,
            inherited=_baseline(),
        )
    )

    assert returned is baseline
    assert panel == ("task-a",)
    assert logs == [
        "determinism: cached (baseline-1, excluded 1/2 nondeterministic tasks)",
        "excluded: task-b (no_chain)",
    ]


def test_exclude_nondeterministic_tasks_inherits_skips_recert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = RunStore(tmp_path)
    baseline = _baseline()
    rows = [_run_row("pwd", RawEnvOutput(stdout="same\n"))]
    _write_chain(tracker, rows)
    inherited = _certified_baseline(
        {"task-a": (audit_module._chain_digest(rows), "deterministic")}
    )
    monkeypatch.setattr(audit_module, "_certify_tasks", _unexpected_certify)
    logs: list[str] = []

    returned, panel = asyncio.run(
        exclude_nondeterministic_tasks(
            run_config=_rollout_config(),
            tracker=tracker,
            baseline=baseline,
            log=logs.append,
            inherited=inherited,
        )
    )

    assert panel == ("task-a",)
    assert tracker.load_experiment("baseline-1").determinism_certification == (
        inherited.determinism_certification
    )
    assert logs == [
        "determinism: inherited 1/1 task certificates "
        "(excluded 0/1 nondeterministic tasks)",
    ]


@pytest.mark.parametrize("changed_fact", ["chain", "identity"])
def test_certification_inheritance_reaudits_changed_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_fact: str,
) -> None:
    tracker = RunStore(tmp_path)
    recorded_rows = [_run_row("old", RawEnvOutput(stdout="same\n"))]
    current_rows = (
        [_run_row("new", RawEnvOutput(stdout="same\n"))]
        if changed_fact == "chain"
        else recorded_rows
    )
    _write_chain(tracker, current_rows)
    inherited = _certified_baseline(
        {"task-a": (audit_module._chain_digest(recorded_rows), "deterministic")}
    )
    baseline = _baseline()
    if changed_fact == "identity":
        inherited = inherited.model_copy(
            update={
                "measurement_identity": inherited.measurement_identity.model_copy(
                    update={"provider_revision": "changed"}
                )
            }
        )
    audited: list[tuple[str, ...]] = []
    expected = TaskCertification(
        chain_digest=audit_module._chain_digest(current_rows),
        verdict="deterministic",
    )

    async def certify(**kwargs):
        audited.append(tuple(kwargs["task_ids"]))
        return {"task-a": expected}

    monkeypatch.setattr(audit_module, "_certify_tasks", certify)

    returned, panel = asyncio.run(
        exclude_nondeterministic_tasks(
            run_config=_rollout_config(),
            tracker=tracker,
            baseline=baseline,
            log=lambda line: None,
            inherited=inherited,
        )
    )

    assert audited == [("task-a",)]
    assert panel == ("task-a",)
    assert returned.determinism_certification == {"task-a": expected}


def test_exclude_nondeterministic_tasks_rejects_empty_panel(tmp_path: Path) -> None:
    baseline = _certified_baseline({"task-a": ("a", "forked")})

    with pytest.raises(RuntimeError, match="determinism check excluded every task"):
        asyncio.run(
            exclude_nondeterministic_tasks(
                run_config=_rollout_config(),
                tracker=RunStore(tmp_path),
                baseline=baseline,
                log=lambda line: None,
            )
        )


def test_audit_concurrency_capped_at_cores_minus_one():
    # The audit ceiling stays a positive sub-core cap so a contended host cannot
    # stall trivial control commands into TimeoutError false forks.
    cap = audit_module._AUDIT_MAX_CONCURRENCY
    cores = os.cpu_count() or 2
    assert cap == max(1, cores - 1)
    assert 1 <= cap < cores or cap == 1
