"""Phase-0 refactor net: pin the on-disk schema of every persisted artifact.

The upcoming restructure renames modules and classes (e.g. ``models.py`` ->
``rollout.py``, ``RolloutResult`` -> ``RolloutResult``). Pydantic serializes *field
names*, not class names, so a pure rename must leave ``experiment.json`` and
``runs.jsonl`` byte-for-byte identical. These tests make that guarantee
enforceable: if a rename, cleanup, or refactor changes which fields land on disk
-- or how they serialize -- one of the assertions below fails loudly instead of
silently forking the artifact contract the diagnosis agent, gate, and tracker
all read back.

The two persisted surfaces:
- ``experiment.json`` <- ``ExperimentResult`` and its nested models.
- ``runs.jsonl``      <- one ``RunIndexRow`` per line.

Both are written with a plain ``model_dump(mode="json")`` (tracker.py), so every
declared field is always emitted and the field set below is the literal on-disk
key set.
"""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import BaseModel, ValidationError

from src.rollout.records import (
    DecisionOutcome,
    ExperimentResult,
    FailureMode,
    ResultDecision,
    RunIndexRow,
    RunKind,
    MeasurementIdentity,
    SecondaryRewardComparison,
    SecondaryRewardOutcome,
    TaskCertification,
    RolloutResult,
)

# Field changes alter the on-disk contract and require an explicit clean-slate migration.
EXPECTED_FIELDS: dict[type[BaseModel], frozenset[str]] = {
    ExperimentResult: frozenset(
        {
            "experiment_id",
            "git_commit_hash",
            "measurement_identity",
            "git_dirty",
            "config_path",
            "kind",
            "parent_commit_hash",
            "baseline_experiment_id",
            "config",
            "started_at",
            "finished_at",
            "crash_reason",
            "tasks",
            "decision",
            "determinism_certification",
        }
    ),
    RolloutResult: frozenset(
        {
            "task_id",
            "failure_mode",
            "failure_origin",
            "error",
            "metrics",
            "rollout_dir",
            "trace_path",
            "infra_retries",
            "started_at",
            "finished_at",
        }
    ),
    TaskCertification: frozenset({"chain_digest", "verdict"}),
    ResultDecision: frozenset(
        {
            "outcome",
            "reason",
            "candidate_solved",
            "baseline_solved",
            "new_solves",
            "regressions",
            "secondary_rewards",
            "invalid_infra_tasks",
            "criterion",
        }
    ),
    SecondaryRewardComparison: frozenset(
        {"name", "baseline_value", "candidate_value", "outcome"}
    ),
    RunIndexRow: frozenset(
        {
            "experiment_id",
            "kind",
            "git_commit_hash",
            "measurement_identity_digest",
            "parent_commit_hash",
            "baseline_experiment_id",
            "started_at",
            "finished_at",
            "crash_reason",
            "solved",
            "verdict",
            "reason",
        }
    ),
}


def test_rollout_crash_requires_failure_origin() -> None:
    payload = {
        "task_id": "task-a",
        "failure_mode": "crash",
        "error": "boom",
        "metrics": {},
        "rollout_dir": None,
        "trace_path": None,
        "started_at": None,
        "finished_at": None,
    }

    with pytest.raises(ValidationError, match="failure_origin"):
        RolloutResult(**payload)
    with pytest.raises(ValidationError, match="failure_origin"):
        RolloutResult(
            **{
                **payload,
                "failure_mode": "verified_rejected",
                "failure_origin": "policy",
            }
        )

    assert RolloutResult(**{**payload, "failure_origin": "env"}).failure_origin == "env"


# Literal membership is part of the on-disk contract, like field names.
EXPECTED_LITERALS: dict[str, frozenset[str]] = {
    "RunKind": frozenset({"baseline", "candidate", "eval"}),
    "FailureMode": frozenset(
        {
            "solved",
            "verified_rejected",
            "hit_step_cap",
            "hit_timeout",
            "verify_timeout",
            "no_valid_action",
            "crash",
            "unscorable_infra",
        }
    ),
    "DecisionOutcome": frozenset({"promoted", "rejected", "invalid_infra"}),
    "SecondaryRewardOutcome": frozenset(
        {"candidate_better", "baseline_better", "tied", "unavailable"}
    ),
}

_LITERAL_ALIASES = {
    "RunKind": RunKind,
    "FailureMode": FailureMode,
    "DecisionOutcome": DecisionOutcome,
    "SecondaryRewardOutcome": SecondaryRewardOutcome,
}

# Populate every field to pin the full shape; Pydantic emits zero-microsecond UTC timestamps with Z.
GOLDEN_EXPERIMENT_JSON: dict = {
    "experiment_id": "exp-20260101-000000-000000",
    "git_commit_hash": "0" * 40,
    "measurement_identity": {
        "effective_config_digest": "effective",
        "provider_revision": "provider",
        "replay_regime_digest": "replay",
    },
    "git_dirty": False,
    "config_path": "config/run_config.json",
    "kind": "candidate",
    "parent_commit_hash": "1" * 40,
    "baseline_experiment_id": "exp-20260101-000000-000001",
    "config": {"schema_version": 6, "environment": {"kind": "swe"}},
    "started_at": "2026-01-01T00:00:00Z",
    "finished_at": "2026-01-01T00:10:00Z",
    "crash_reason": None,
    "tasks": {
        "django__django-10973": {
            "task_id": "django__django-10973",
            "failure_mode": "solved",
            "failure_origin": None,
            "error": None,
            "metrics": {
                "reward": 1.0,
                "steps_used": 21,
                "first_attempt_valid": 20,
                "first_attempt_total": 20,
                "fail_to_pass_passed": 5,
                "pass_to_pass_failed": 0,
            },
            "rollout_dir": "/x/tasks/django__django-10973",
            "trace_path": "/x/tasks/django__django-10973/agent/steps.jsonl",
            "infra_retries": [],
            "started_at": "2026-01-01T00:00:01Z",
            "finished_at": "2026-01-01T00:00:06Z",
        }
    },
    "decision": {
        "outcome": "promoted",
        "reason": "strict_improvement_without_regression",
        "candidate_solved": ["django__django-10973"],
        "baseline_solved": [],
        "new_solves": ["django__django-10973"],
        "regressions": [],
        "secondary_rewards": [
            {
                "name": "steps_used",
                "baseline_value": 25,
                "candidate_value": 21,
                "outcome": "candidate_better",
            }
        ],
        "invalid_infra_tasks": [],
        "criterion": "strict_pareto",
    },
    "determinism_certification": {
        "other-task": {"chain_digest": "chain", "verdict": "forked"}
    },
}

GOLDEN_RUNS_JSONL_ROW: dict = {
    "experiment_id": "exp-20260101-000000-000000",
    "kind": "candidate",
    "git_commit_hash": "0" * 40,
    "measurement_identity_digest": "identity",
    "parent_commit_hash": "1" * 40,
    "baseline_experiment_id": "exp-20260101-000000-000001",
    "started_at": "2026-01-01T00:00:00Z",
    "finished_at": "2026-01-01T00:10:00Z",
    "crash_reason": None,
    "solved": 1,
    "verdict": "promoted",
    "reason": "strict_improvement_without_regression",
}


def test_persisted_model_field_sets_are_pinned() -> None:
    """Every persisted model declares exactly its pinned field set -- an added,
    removed, or renamed field breaks this before it can silently change an
    artifact on disk."""
    for model, expected in EXPECTED_FIELDS.items():
        assert set(model.model_fields) == expected, model.__name__


def test_persisted_literal_domains_are_pinned() -> None:
    """Persisted string domains (failure modes, verdicts, kinds) are frozen --
    a member added or dropped is an on-disk contract change."""
    for name, alias in _LITERAL_ALIASES.items():
        assert set(get_args(alias)) == EXPECTED_LITERALS[name], name


def test_measurement_identity_is_frozen_strict_and_canonically_hashed() -> None:
    identity = MeasurementIdentity(
        effective_config_digest="effective",
        provider_revision="provider",
        replay_regime_digest="replay",
    )

    assert identity.digest == (
        "6c9301e4699384609c10ccb8a7b5ee5a90bf652070b29f36de1b927d05020478"
    )
    with pytest.raises(ValidationError, match="frozen"):
        identity.provider_revision = "changed"
    with pytest.raises(ValidationError, match="extra"):
        MeasurementIdentity.model_validate(
            identity.model_dump() | {"extra": "forbidden"}
        )


def test_task_certification_is_frozen_and_strictly_typed() -> None:
    certification = TaskCertification(chain_digest="chain", verdict="deterministic")

    with pytest.raises(ValidationError, match="frozen"):
        certification.verdict = "forked"
    with pytest.raises(ValidationError, match="extra"):
        TaskCertification.model_validate(
            certification.model_dump() | {"extra": "forbidden"}
        )
    with pytest.raises(ValidationError, match="verdict"):
        TaskCertification.model_validate(
            {"chain_digest": "chain", "verdict": "unknown"}
        )


def test_experiment_json_round_trips_byte_stable() -> None:
    """A golden experiment.json validates and re-serializes to itself. Catches
    serialization drift (datetime format, a field rename that changes output,
    an accidental exclude=/alias) that the field-set pin alone would miss."""
    reserialized = ExperimentResult.model_validate(GOLDEN_EXPERIMENT_JSON).model_dump(
        mode="json"
    )
    assert reserialized == GOLDEN_EXPERIMENT_JSON


def test_runs_jsonl_row_round_trips_byte_stable() -> None:
    """A golden runs.jsonl row validates and re-serializes to itself."""
    reserialized = RunIndexRow.model_validate(GOLDEN_RUNS_JSONL_ROW).model_dump(
        mode="json"
    )
    assert reserialized == GOLDEN_RUNS_JSONL_ROW


def test_sparse_legacy_experiment_artifact_is_rejected() -> None:
    with pytest.raises(ValidationError, match="git_dirty"):
        ExperimentResult.model_validate(
            {
                "experiment_id": "legacy",
                "git_commit_hash": "candidate",
                "config_path": "configs/swe.yaml",
                "started_at": "2026-06-29T00:23:21+00:00",
                "tasks": {},
            }
        )


def _rollout(task_id: str) -> RolloutResult:
    return RolloutResult.model_validate(
        GOLDEN_EXPERIMENT_JSON["tasks"]["django__django-10973"] | {"task_id": task_id}
    )


def _experiment(**updates: object) -> ExperimentResult:
    fields = {
        "experiment_id": "exp-1",
        "git_commit_hash": "abc123",
        "measurement_identity": {
            "effective_config_digest": "effective",
            "provider_revision": "provider",
            "replay_regime_digest": "replay",
        },
        "git_dirty": False,
        "config_path": "config/run.json",
        "started_at": "2026-07-11T00:00:00Z",
        "tasks": {"task-a": _rollout("task-a")},
    }
    fields.update(updates)
    return ExperimentResult.model_validate(fields)


def test_finished_populated_experiment_is_valid() -> None:
    result = _experiment(finished_at="2026-07-11T00:01:00Z")

    assert result.tasks["task-a"] == _rollout("task-a")


def test_finished_experiment_rejects_none_task() -> None:
    with pytest.raises(
        ValidationError, match="finished experiments require task results"
    ):
        _experiment(finished_at="2026-07-11T00:01:00Z", tasks={"task-a": None})


def test_finished_experiment_rejects_mismatched_nested_task_id() -> None:
    with pytest.raises(ValidationError, match="task result id must match its map key"):
        _experiment(
            finished_at="2026-07-11T00:01:00Z",
            tasks={"task-a": _rollout("task-b")},
        )


def test_running_experiment_accepts_none_task() -> None:
    assert _experiment(tasks={"task-a": None}).tasks == {"task-a": None}


def test_crashed_experiment_accepts_none_task() -> None:
    result = _experiment(
        finished_at="2026-07-11T00:01:00Z",
        crash_reason="worker exited",
        tasks={"task-a": None},
    )

    assert result.tasks == {"task-a": None}


def test_record_defaults_match_production_lifecycle() -> None:
    assert {
        name
        for name, field in ExperimentResult.model_fields.items()
        if field.is_required()
    } == {
        "experiment_id",
        "git_commit_hash",
        "measurement_identity",
        "git_dirty",
        "config_path",
        "started_at",
        "tasks",
    }
    assert {
        name
        for name, field in RolloutResult.model_fields.items()
        if field.is_required()
    } == {
        "task_id",
        "failure_mode",
        "error",
        "metrics",
        "rollout_dir",
        "trace_path",
        "started_at",
        "finished_at",
    }
    assert all(field.is_required() for field in RunIndexRow.model_fields.values())


def test_run_index_verdict_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError, match="verdict"):
        RunIndexRow.model_validate(GOLDEN_RUNS_JSONL_ROW | {"verdict": "unknown"})
