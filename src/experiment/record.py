"""Experiment persistence model.

The on-disk experiment record (``experiment.json``) and run state
(``state.json``): the dataclasses the runner writes and reloads, plus the small
helpers that build per-task evidence and terminal/crash trial results. Owns no
orchestration and no gate logic — the bottom of the experiment-package layering
(record <- gate <- runner).
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from src.harness.contracts import TaskResult
from src.metrics import BaselineComparison, FailureMode, TaskMetrics, is_majority_solved


EXPERIMENT_FILENAME = "experiment.json"
ExperimentStatus = Literal["keep", "discard", "crash"]
PanelLifecycle = Literal["pending", "active", "finished", "skipped"]
PanelPurpose = Literal["promotion", "regression_veto"]


class ExperimentAbandoned(RuntimeError):
    """A run was stopped by the outer supervisor loop (process restart) rather
    than failing on its own. Trials filled to conclude such a record classify as
    `interrupted` -- like a Ctrl-C -- not `crash` (see ``terminal_task_result``).
    """


def failed_experiment_git_ref(experiment_id: str) -> str:
    return f"refs/experiments/failed/{experiment_id}"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


class TaskTrials(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: str
    expected_trial_count: int
    trials: list[TaskResult]

    @property
    def trial_count(self) -> int:
        return len(self.trials)

    @property
    def finished_trials(self) -> list[TaskResult]:
        return [trial for trial in self.trials if trial.finished_at is not None]

    @property
    def valid_trials(self) -> list[TaskResult]:
        # Trials that produced task evidence. `error is None` is the contract:
        # a trial with `error is not None` is an infra `crash`, recorded for
        # diagnosis but never scored. Solve counts, the majority verdict, and
        # the promotion gate all read valid trials only.
        return [trial for trial in self.finished_trials if trial.error is None]

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
        # True iff every valid trial passed. Used by candidates to budget a
        # single trial against baselines that show a task as reliably solved;
        # confirm-on-fail expands back to task_trials if that single candidate
        # trial fails.
        valid = self.valid_trials
        if not valid:
            return False
        return all(trial.solved for trial in valid)

    @property
    def representative(self) -> TaskResult | None:
        # Prefer a valid trial matching the majority outcome for evidence. When
        # a task produced no valid trials, surface the most recent trial (a
        # crash) so its error is visible for diagnosis.
        valid = self.valid_trials
        if not valid:
            return self.trials[-1] if self.trials else None
        majority = self.majority_solved
        for trial in reversed(valid):
            if trial.solved is majority:
                return trial
        return valid[-1]

    @property
    def is_finished(self) -> bool:
        # Counts every completed slot, valid or crash, so the per-task budget
        # terminates and a crash slot is never re-run. Solve scoring excludes
        # crash trials; termination must not, or an all-crash task would spawn
        # slots forever.
        return len(self.finished_trials) >= self.expected_trial_count

    def append(self, trial: TaskResult) -> None:
        self.trials.append(trial)

    @classmethod
    def empty(cls, *, task_name: str, expected_trial_count: int) -> "TaskTrials":
        return cls(
            task_name=task_name,
            expected_trial_count=expected_trial_count,
            trials=[],
        )


class CandidateChangeEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    commit: str
    parent_baseline_experiment_id: str | None = None
    parent_baseline_commit: str | None = None


class TaskOutcomeEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    baseline_solved: bool | None
    candidate_solved: bool | None
    outcome: Literal[
        "new_solve",
        "regression",
        "unchanged_solved",
        "unchanged_unsolved",
        "uncompared",
    ]
    trial_dir: str | None = None
    agent_steps_path: str | None = None
    agent_exec_log_path: str | None = None
    metrics_path: str | None = None
    verifier_stdout_path: str | None = None
    error: str | None = None

    @classmethod
    def from_trials(
        cls,
        *,
        task_id: str,
        candidate_trials: "TaskTrials",
        baseline_trials: TaskTrials | None,
        verdict: BaselineComparison | None,
    ) -> "TaskOutcomeEvidence":
        def existing_artifact_path(path: str | None) -> str | None:
            if path is None:
                return None
            if not Path(path).exists():
                return None
            return path

        def agent_exec_log_path(task_result: TaskResult) -> str | None:
            if task_result.trial_dir is None:
                return None
            return str(Path(task_result.trial_dir) / "agent" / "exec.log")

        def outcome_label() -> Literal[
            "new_solve",
            "regression",
            "unchanged_solved",
            "unchanged_unsolved",
            "uncompared",
        ]:
            if verdict is None or verdict.kind == "uncompared":
                return "uncompared"
            if verdict.kind == "regression":
                return "regression"
            if verdict.kind == "improvement":
                return "new_solve"
            if candidate_solved is True and baseline_solved is True:
                return "unchanged_solved"
            return "unchanged_unsolved"

        baseline_solved = (
            None if baseline_trials is None else baseline_trials.majority_solved
        )
        candidate_solved = candidate_trials.majority_solved
        representative = candidate_trials.representative
        return cls(
            task_id=task_id,
            baseline_solved=baseline_solved,
            candidate_solved=candidate_solved,
            outcome=outcome_label(),
            trial_dir=existing_artifact_path(
                None if representative is None else representative.trial_dir
            ),
            agent_steps_path=existing_artifact_path(
                None if representative is None else representative.trace_path
            ),
            agent_exec_log_path=existing_artifact_path(
                None if representative is None else agent_exec_log_path(representative)
            ),
            metrics_path=existing_artifact_path(
                None if representative is None else representative.metrics_path
            ),
            verifier_stdout_path=existing_artifact_path(
                None if representative is None else representative.verifier_stdout_path
            ),
            error=None if representative is None else representative.error,
        )


class ExperimentEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_change: CandidateChangeEvidence
    panel_outcomes: dict[str, list[TaskOutcomeEvidence]]

    @classmethod
    def empty(cls, *, record: "ExperimentRecord") -> "ExperimentEvidence":
        return cls(
            candidate_change=CandidateChangeEvidence(
                commit=record.git_commit_hash,
                parent_baseline_experiment_id=record.parent_baseline_experiment_id,
            ),
            panel_outcomes={panel_id: [] for panel_id in record.panel_order},
        )


class PanelEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ExperimentStatus
    decision_reason: str
    verdicts: dict[str, BaselineComparison]


class PanelRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    panel_id: str
    purpose: PanelPurpose
    lifecycle: PanelLifecycle
    task_ids: list[str]
    task_results: dict[str, TaskTrials]
    started_at: str | None = None
    finished_at: str | None = None
    skip_reason: str = ""
    evaluation: PanelEvaluation | None = None

    @classmethod
    def initialize(
        cls,
        *,
        panel_id: str,
        purpose: PanelPurpose,
        task_ids: Sequence[str],
        expected_trial_count: int,
        lifecycle: PanelLifecycle,
    ) -> "PanelRecord":
        canonical_task_ids = sorted(task_ids)
        return cls(
            panel_id=panel_id,
            purpose=purpose,
            lifecycle=lifecycle,
            task_ids=canonical_task_ids,
            task_results={
                task_id: TaskTrials.empty(
                    task_name=task_id,
                    expected_trial_count=expected_trial_count,
                )
                for task_id in canonical_task_ids
            },
        )

    @model_validator(mode="after")
    def task_results_match_task_ids(self) -> "PanelRecord":
        if len(set(self.task_ids)) != len(self.task_ids) or set(self.task_ids) != set(
            self.task_results
        ):
            raise ValueError("panel task_results must cover exactly task_ids")
        return self

    @property
    def solved_count(self) -> int:
        return sum(
            1 for trials in self.task_results.values() if trials.majority_solved is True
        )


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2]
    experiment_id: str
    parent_baseline_experiment_id: str | None
    git_commit_hash: str
    focus_name: str
    status: ExperimentStatus | None
    decision_reason: str
    error: str
    started_at: str
    finished_at: str | None
    panel_order: list[str]
    panels: dict[str, PanelRecord]
    evidence: ExperimentEvidence | None

    @model_validator(mode="after")
    def panel_order_matches_panels(self) -> "ExperimentRecord":
        if len(set(self.panel_order)) != len(self.panel_order) or set(
            self.panel_order
        ) != set(self.panels):
            raise ValueError("panel_order must cover exactly panels")
        for panel_id in self.panel_order:
            if self.panels[panel_id].panel_id != panel_id:
                raise ValueError("panel key must match panel_id")
        return self

    @classmethod
    def path(cls, experiment_id: str, *, root: Path) -> Path:
        return root.resolve() / experiment_id / EXPERIMENT_FILENAME

    @classmethod
    def load(
        cls,
        experiment_id: str,
        *,
        root: Path,
    ) -> "ExperimentRecord":
        payload = json.loads(cls.path(experiment_id, root=root).read_text())
        return cls.model_validate(payload)

    @classmethod
    def initialize(
        cls,
        *,
        experiment_id: str,
        git_commit_hash: str,
        parent_baseline_experiment_id: str | None,
        panels: Sequence[PanelRecord],
        focus_name: str = "",
        started_at: str,
    ) -> "ExperimentRecord":
        return cls(
            schema_version=2,
            experiment_id=experiment_id,
            parent_baseline_experiment_id=parent_baseline_experiment_id,
            git_commit_hash=git_commit_hash,
            focus_name=focus_name,
            status=None,
            decision_reason="",
            error="",
            started_at=started_at,
            finished_at=None,
            panel_order=[panel.panel_id for panel in panels],
            panels={panel.panel_id: panel for panel in panels},
            evidence=None,
        )

    def write(self, *, root: Path) -> None:
        if self.evidence is None:
            self.evidence = ExperimentEvidence.empty(record=self)
        payload = self.model_dump(mode="json")
        write_json_atomic(self.path(self.experiment_id, root=root), payload)

    def record_task_result(self, task_result: TaskResult) -> None:
        panel_results = self._panel_results(task_result.task_name)
        panel_results[task_result.task_name].append(task_result)

    def finalize(
        self,
        *,
        status: ExperimentStatus,
        error: str | None = None,
        decision_reason: str | None = None,
    ) -> None:
        if any(not trials.is_finished for trials in self._all_task_results()):
            raise RuntimeError(
                "terminal experiment records require finished task results"
            )
        self.status = status
        self.decision_reason = "" if decision_reason is None else decision_reason
        self.error = "" if error is None else error
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def refresh_evidence(
        self,
        *,
        baseline: ExperimentRecord | None,
        verdicts: Mapping[str, BaselineComparison] | None = None,
    ) -> None:
        self.evidence = build_experiment_evidence(
            candidate=self,
            baseline=baseline,
            verdicts=verdicts,
        )

    def finalize_crash(
        self,
        *,
        exc: BaseException,
        baseline: ExperimentRecord | None,
        root: Path,
    ) -> None:
        """Finalize this record as a crash and persist it.

        Shared by every crash path (baseline and candidate). Completes any
        unfinished trial slots with the failure, marks the record ``crash``,
        and refreshes evidence with no gate verdicts so per-task labels stay
        "uncompared" (the gate produced nothing). Deliberately does NOT touch
        run state, git refs, or the promotion gate: those differ between the
        baseline and candidate lifecycles and stay at the call site.
        """
        self._complete_unfinished_task_results(exc=exc)
        self.finalize(status="crash", error=str(exc))
        self.refresh_evidence(baseline=baseline)
        self.write(root=root)

    def is_concluded(self) -> bool:
        return (
            self.status in {"keep", "discard", "crash"} and self.finished_at is not None
        )

    def _panel_results(self, task_id: str) -> dict[str, TaskTrials]:
        for panel_id in self.panel_order:
            task_results = self.panels[panel_id].task_results
            if task_id in task_results:
                return task_results
        raise KeyError(
            f"task {task_id!r} is not part of experiment {self.experiment_id}"
        )

    def _task_trials(self, task_id: str) -> TaskTrials:
        return self._panel_results(task_id)[task_id]

    def _all_task_results(self) -> list[TaskTrials]:
        return [
            trials
            for panel_id in self.panel_order
            for trials in self.panels[panel_id].task_results.values()
        ]

    def _complete_unfinished_task_results(self, *, exc: BaseException) -> None:
        for panel_id in self.panel_order:
            if self.panels[panel_id].lifecycle == "skipped":
                continue
            for task_id in self.panels[panel_id].task_results:
                trials = self._task_trials(task_id)
                while not trials.is_finished:
                    self.record_task_result(
                        terminal_task_result(task_id=task_id, exc=exc)
                    )


def build_experiment_evidence(
    *,
    candidate: ExperimentRecord,
    baseline: ExperimentRecord | None,
    verdicts: Mapping[str, BaselineComparison] | None = None,
) -> ExperimentEvidence:
    """Assemble per-task evidence for the persisted record.

    ``verdicts`` is the gate's per-task verdict dict (typically built by
    :func:`build_gate_verdicts` at promotion time). When ``verdicts`` is
    ``None`` -- the case for crash and supervisor cleanup paths -- every
    task is labelled "uncompared" because there is genuinely no gate decision
    yet. Once the gate runs, ``_run_experiment`` refreshes evidence
    with the populated dict so the persisted record carries verdict-driven
    labels.
    """
    verdicts_map: Mapping[str, BaselineComparison] = verdicts or {}
    panel_outcomes: dict[str, list[TaskOutcomeEvidence]] = {}
    for panel_id in candidate.panel_order:
        candidate_panel = candidate.panels[panel_id]
        baseline_results = (
            {}
            if baseline is None or panel_id not in baseline.panels
            else baseline.panels[panel_id].task_results
        )
        panel_outcomes[panel_id] = [
            TaskOutcomeEvidence.from_trials(
                task_id=task_id,
                candidate_trials=task_trials,
                baseline_trials=baseline_results.get(task_id),
                verdict=verdicts_map.get(task_id),
            )
            for task_id, task_trials in candidate_panel.task_results.items()
            if task_trials.expected_trial_count > 0 or task_trials.trials
        ]
    return ExperimentEvidence(
        candidate_change=CandidateChangeEvidence(
            commit=candidate.git_commit_hash,
            parent_baseline_experiment_id=candidate.parent_baseline_experiment_id,
            parent_baseline_commit=None
            if baseline is None
            else baseline.git_commit_hash,
        ),
        panel_outcomes=panel_outcomes,
    )


class ExperimentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    active_baseline_experiment_id: str | None
    current_experiment_id: str | None = None
    updated_at: str | None = None

    @classmethod
    def path(cls, *, root: Path) -> Path:
        return root.resolve() / "state.json"

    @classmethod
    def load(cls, *, root: Path) -> "ExperimentState":
        path = cls.path(root=root)
        if not path.exists():
            return cls(
                active_baseline_experiment_id=None,
                updated_at=None,
            )
        return cls.model_validate_json(path.read_text())

    def save(self, *, root: Path) -> None:
        write_json_atomic(self.path(root=root), self.model_dump(mode="json"))


def raise_if_no_valid_evidence(record: ExperimentRecord) -> None:
    # Each active panel must yield at least one valid trial. An
    # isolated per-trial `crash` (`error is not None`) is tolerated: it is
    # excluded from the gate and the pool. But a run where *every* trial
    # crashed in an active panel produced no task evidence — the gate would compare against
    # nothing, and a baseline would be installed empty so that every later
    # candidate compares against an all-zero pool. That is an experiment-level
    # failure, so it crashes (and a baseline is therefore not promoted).
    for panel_name in record.panel_order:
        panel = record.panels[panel_name]
        if panel.lifecycle == "skipped":
            continue
        panel_trials = [
            trials
            for trials in panel.task_results.values()
            if trials.expected_trial_count > 0 or trials.trials
        ]
        if panel_trials and not any(trials.valid_trials for trials in panel_trials):
            raise RuntimeError(
                f"experiment produced no valid trials in {panel_name} panel (every trial crashed)"
            )


def terminal_task_result(*, task_id: str, exc: BaseException) -> TaskResult:
    finished_at = datetime.now(timezone.utc).isoformat()
    # An interrupt (Ctrl-C, KeyboardInterrupt, or a supervisor-restart abandon)
    # means the trial was stopped from the outside, not that it failed on its
    # own -- bucket it as `interrupted`, distinct from a genuine infra `crash`.
    # Both keep `error` set, so both stay excluded from the gate's valid trials.
    failure_mode: FailureMode
    if isinstance(exc, ExperimentAbandoned):
        error = str(exc) or type(exc).__name__
        failure_mode = "interrupted"
    elif isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
        error = "canceled"
        failure_mode = "interrupted"
    else:
        error = str(exc) or type(exc).__name__
        failure_mode = "crash"
    return TaskResult(
        task_name=task_id,
        reward=0.0,
        solved=False,
        error=error,
        steps_used=0,
        metrics=TaskMetrics(failure_mode=failure_mode),
        started_at=finished_at,
        finished_at=finished_at,
    )
