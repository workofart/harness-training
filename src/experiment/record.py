"""Experiment persistence models (dumb).

Bottom of the experiment layer (depends only on ``contracts``). The hierarchy is
**trial -> task -> experiment**: ``TrialResult`` is one trial's outcome,
``TaskResult`` aggregates a task's trials, ``ExperimentResult`` is the whole run
keyed by task. Every roll-up (``solved_count``, ``majority_solved``,
``is_finished``) is a ``@property`` derivation -- never stored. Owns no lifecycle transitions, no gate logic, no decisions, and no write
I/O: the ``writer`` persists ``experiment.json``; ``scan``/``policy`` derive all
control state. Telemetry (``TaskMetrics``) is referenced by ``metrics_path``,
never embedded, so ``experiment.json`` carries run-level triage on its own and
deep telemetry is one file-read away.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.contracts import FailureMode, is_majority_solved

EXPERIMENT_FILENAME = "experiment.json"

# Mechanical "did the run finish?" -- deliberately distinct from the gate's
# keep/discard decision, which the auto layer owns in loop.json. The orchestrator
# sets this; it reflects the latest orchestrator call, so a candidate may be
# `completed` after train, `running` again during the veto, then `completed`.
# What actually ran is read from task presence, not from this field.
RunStatus = Literal["running", "completed", "crashed"]


class TrialResult(BaseModel):
    """One trial's outcome (formerly ``contracts.TaskResult``).

    Outcome fields are first-class here: ``solved`` is the gate's single source
    of truth, ``failure_mode`` the categorical *why* (always set -- every
    recorded trial is classified), ``verifier_passed`` the grader judgment
    (``None`` when the trial never reached the verifier), ``error`` the infra
    crash/interrupt marker (set => the trial is excluded from scoring).
    ``run_id`` is the trial's key (the artifact dir is
    ``tasks/<task_id>/<run_id>/``). Telemetry stays in ``metrics.json``, reached
    via ``metrics_path`` -- never embedded; raw reward lives at
    ``verifier/reward.txt`` and the step count in ``TaskMetrics.steps_total``,
    so neither is duplicated here (plan.md §13).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    solved: bool
    failure_mode: FailureMode
    verifier_passed: bool | None = None
    error: str | None = None
    trial_dir: str | None = None
    trace_path: str | None = None
    metrics_path: str | None = None
    verifier_stdout_path: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    @model_validator(mode="after")
    def _check_outcome_invariants(self) -> "TrialResult":
        # §5: a trial is solved iff its terminal bucket is "solved", and a trial
        # that recorded an infra error is never scored as solved.
        if self.solved != (self.failure_mode == "solved"):
            raise ValueError(
                f"solved={self.solved} contradicts failure_mode={self.failure_mode!r}"
            )
        if self.error is not None and self.solved:
            raise ValueError("a trial with error set cannot be solved")
        return self


class TaskResult(BaseModel):
    """One task's trials plus the per-task derivations the scheduler and gate read.

    ``trials`` are recorded completed slots (valid or crash). The budget counts
    every slot so it terminates and a crash slot is never re-run; solve scoring
    excludes crash trials (``error is not None``).
    """

    model_config = ConfigDict(extra="forbid")

    expected_trial_count: int
    trials: list[TrialResult] = Field(default_factory=list)

    @property
    def valid_trials(self) -> list[TrialResult]:
        # Scorable trials only. `error is None` is the contract: a trial with
        # `error` set is an infra crash/interrupt, recorded for diagnosis but
        # excluded from solve counts, the majority verdict, and the gate pool.
        return [trial for trial in self.trials if trial.error is None]

    @property
    def solved_count(self) -> int:
        return sum(1 for trial in self.valid_trials if trial.solved)

    @property
    def majority_solved(self) -> bool | None:
        valid = self.valid_trials
        if not valid:
            return None
        return is_majority_solved(solved=self.solved_count, total=len(valid))

    @property
    def is_deterministic_solved(self) -> bool:
        # True iff every valid trial passed. Lets a candidate budget a single
        # trial against a baseline that reliably solves the task; confirm-on-fail
        # expands back to the full budget if that single trial fails.
        valid = self.valid_trials
        return bool(valid) and all(trial.solved for trial in valid)

    @property
    def is_finished(self) -> bool:
        return len(self.trials) >= self.expected_trial_count

    def append(self, trial: TrialResult) -> None:
        self.trials.append(trial)

    @classmethod
    def empty(cls, *, expected_trial_count: int) -> "TaskResult":
        return cls(expected_trial_count=expected_trial_count, trials=[])


class ExperimentResult(BaseModel):
    """The whole run -- mode-agnostic, the only file ``uv run exp`` writes.

    No panels, focus, parent, decision, or evidence: those are auto-layer facts
    that live in ``loop.json``. train/test membership is config (asserted
    non-empty + disjoint at load, §12), not record structure.
    """

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    git_commit_hash: str
    run_status: RunStatus
    started_at: str
    finished_at: str | None = None
    tasks: dict[str, TaskResult] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_status_timestamp_consistency(self) -> "ExperimentResult":
        # A run is `running` exactly while it has no finish timestamp; `completed`
        # and `crashed` are terminal and must carry `finished_at` -- the ordering
        # and recovery key (§5). Rejects an impossible terminal-but-unfinished
        # (or running-but-finished) record at construction.
        if (self.run_status == "running") != (self.finished_at is None):
            raise ValueError(
                f"run_status={self.run_status!r} is inconsistent with "
                f"finished_at={self.finished_at!r} (running iff not finished)"
            )
        return self

    @classmethod
    def path(cls, experiment_id: str, *, root: Path) -> Path:
        return root.resolve() / experiment_id / EXPERIMENT_FILENAME

    @classmethod
    def load(cls, experiment_id: str, *, root: Path) -> "ExperimentResult":
        return cls.model_validate_json(cls.path(experiment_id, root=root).read_text())
