"""Thin rollout ledger plus the run-level records that persist it.

``RolloutResult`` records identity, outcome, generic numeric ``metrics``, artifact
pointers, and timestamps. Publishers write rollout signals as metric keys; the gate
reads them through ``SecondaryRewardMetric`` classes, while diagnosis agents inspect
the JSON ledger directly. The run/experiment naming is not yet unified -- see the
``TODO(naming)`` on ``ExperimentResult`` below.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


RunKind = Literal["baseline", "candidate", "eval"]
FailureMode = Literal[
    "solved",
    "verified_rejected",
    "hit_step_cap",
    "hit_timeout",
    "verify_timeout",
    "no_valid_action",
    "crash",
    "unscorable_infra",
]
FailureOrigin = Literal["policy", "env"]
# Measurement-integrity tiers; the second contains the first by construction:
# a rollout with no cost signal cannot have a trustworthy verdict either.
# Unscorable: infra broke the rollout — no verdict, no secondary cost.
UNSCORABLE_FAILURE_MODES = frozenset({"crash", "unscorable_infra"})
# Verdict-untrusted: grading never finished; the agent loop did run, so
# secondary cost still counts.
VERDICT_UNTRUSTED_FAILURE_MODES = UNSCORABLE_FAILURE_MODES | {"verify_timeout"}
DecisionOutcome = Literal["promoted", "rejected", "invalid_infra"]
SecondaryRewardOutcome = Literal[
    "candidate_better",
    "baseline_better",
    "tied",
    "unavailable",
]


class RolloutResult(BaseModel):
    """One rollout outcome plus generic metrics and raw investigation artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    failure_mode: FailureMode
    failure_origin: FailureOrigin | None = None
    error: str | None
    metrics: dict[str, int | float]
    rollout_dir: str | None
    trace_path: str | None
    infra_retries: list[dict[str, Any]] = Field(default_factory=list)
    started_at: datetime | None
    finished_at: datetime | None

    @model_validator(mode="after")
    def _check_outcome_invariants(self) -> Self:
        if (self.failure_mode == "crash") != (self.failure_origin is not None):
            raise ValueError("failure_origin must be present exactly for crash results")
        if self.error is not None and self.failure_mode == "solved":
            raise ValueError("errored rollouts cannot be solved")
        return self


class SecondaryRewardComparison(BaseModel):
    """One compact baseline-vs-candidate secondary reward comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    baseline_value: int | float | None = None
    candidate_value: int | float | None = None
    outcome: SecondaryRewardOutcome


class ResultDecision(BaseModel):
    """The promotion verdict, persisted so it is never re-derived from gate source.

    Current promotion treats crashed candidate tasks as unsolved, so the regression gate
    handles crashes on baseline-solved tasks.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: DecisionOutcome
    reason: str
    candidate_solved: list[str] = Field(default_factory=list)
    baseline_solved: list[str] = Field(default_factory=list)
    new_solves: list[str] = Field(default_factory=list)
    regressions: list[str] = Field(default_factory=list)
    secondary_rewards: list[SecondaryRewardComparison] = Field(default_factory=list)
    invalid_infra_tasks: list[str] = Field(default_factory=list)
    criterion: str | None = None


class MeasurementIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    effective_config_digest: str
    provider_revision: str
    replay_regime_digest: str

    @property
    def digest(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class TaskCertification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chain_digest: str
    verdict: Literal["deterministic", "forked", "no_chain"]


# TODO(naming): Unifying Experiment/Run requires migrating persisted IDs, CLI, and paths.
class ExperimentResult(BaseModel):
    """Canonical run record: rollout outcomes keyed by task.

    Consolidated run-level artifact: identity, an inline config snapshot, rollout
    ledgers, and the promotion ``decision`` all live here, so this one file answers a
    run's triage questions without source archaeology or a second hop.
    """

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    git_commit_hash: str
    measurement_identity: MeasurementIdentity
    git_dirty: bool
    config_path: str | None
    kind: RunKind | None = None
    parent_commit_hash: str | None = None
    baseline_experiment_id: str | None = None
    config: dict[str, Any] | None = None
    started_at: datetime
    finished_at: datetime | None = None
    crash_reason: str | None = None
    tasks: dict[str, RolloutResult | None]
    decision: ResultDecision | None = None
    # None = not yet certified; {} = certified, nothing excluded.
    determinism_certification: dict[str, TaskCertification] | None = None

    @model_validator(mode="after")
    def _check_crash_timestamp_consistency(self) -> Self:
        if self.crash_reason is not None and self.finished_at is None:
            raise ValueError("crashed experiments must have finished_at")
        if self.finished_at is not None and self.crash_reason is None:
            for task_id, rollout in self.tasks.items():
                if rollout is None:
                    raise ValueError("finished experiments require task results")
                if rollout.task_id != task_id:
                    raise ValueError("task result id must match its map key")
        return self


class RunIndexRow(BaseModel):
    """One finalized run in the cross-run index.

    Enriched with ``solved``/``verdict``/``reason`` so the whole cross-run triage is one
    hop over ``runs.jsonl`` -- no per-run ``experiment.json`` fan-out, no separate index.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str
    kind: RunKind
    git_commit_hash: str
    measurement_identity_digest: str
    parent_commit_hash: str | None
    baseline_experiment_id: str | None
    started_at: datetime
    finished_at: datetime
    crash_reason: str | None
    solved: int
    verdict: DecisionOutcome | None
    reason: str | None


def solved_task_ids(experiment: ExperimentResult) -> frozenset[str]:
    return frozenset(
        task_id
        for task_id, rollout in experiment.tasks.items()
        if rollout is not None and rollout.failure_mode == "solved"
    )


def error_text(exc: BaseException) -> str:
    detail = str(exc)
    if detail:
        return f"{type(exc).__name__}: {detail}"
    return type(exc).__name__
