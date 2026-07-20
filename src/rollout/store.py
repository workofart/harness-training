"""Central experiment artifact recorder and path owner."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from git import Repo

from src.rollout.records import (
    ExperimentResult,
    ResultDecision,
    RunIndexRow,
    RunKind,
    RolloutResult,
    solved_task_ids,
)


def invoker_repo_root() -> Path:
    return Path(Repo(Path.cwd(), search_parent_directories=True).working_tree_dir)


EXPERIMENT_FILENAME = "experiment.json"
LEARNING_FILENAME = "learning.md"
RUNS_INDEX_FILENAME = "runs.jsonl"
RUN_LOG_FILENAME = "run.log"
TASKS_DIRNAME = "tasks"
AGENT_DIRNAME = "agent"
STEPS_FILENAME = "steps.jsonl"


class RunObserver(Protocol):
    def log(self, line: str) -> None: ...

    def experiment_started(self, experiment_id: str) -> None: ...

    def task_finished(self, task_id: str, failure_mode: str) -> None: ...

    def measurement_heartbeat(self) -> None: ...


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename so a concurrent reader never sees a partial file.
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        handle.write(text)
    os.replace(tmp_path, path)


class RunStore:
    """Owns the filesystem contract for run artifacts.

    Models define the data shape; this class defines where that data lives and
    how it is written.
    """

    def __init__(self, root_dir: Path):
        self.root = root_dir

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def experiment_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / EXPERIMENT_FILENAME

    def task_dir(self, run_id: str, task_id: str) -> Path:
        return self.run_dir(run_id) / TASKS_DIRNAME / task_id

    def trace_path(self, run_id: str, task_id: str) -> Path:
        return self.task_dir(run_id, task_id) / AGENT_DIRNAME / STEPS_FILENAME

    def run_log_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / RUN_LOG_FILENAME

    def runs_index_path(self) -> Path:
        return self.root / RUNS_INDEX_FILENAME

    def learning_path(self) -> Path:
        return self.root / LEARNING_FILENAME

    def publish_learning(self, text: str) -> None:
        _atomic_write(self.learning_path(), text)

    def save_experiment(self, result: ExperimentResult) -> None:
        _atomic_write(
            self.experiment_path(result.experiment_id),
            json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        )

    def load_experiment(self, run_id: str) -> ExperimentResult:
        return ExperimentResult.model_validate_json(
            self.experiment_path(run_id).read_text()
        )

    def mark_crashed(self, run_id: str, *, reason: str) -> ExperimentResult:
        """Finalize a persisted experiment as crashed.

        For a run whose process died before it could finalize itself -- e.g. the parent
        watchdog killed a wedged child. ``crash_reason`` is the source fact; the
        display state "crashed" is derived from it.
        """
        result = self.load_experiment(run_id)
        if result.crash_reason is not None:
            return result

        crashed = result.model_copy(
            update={
                "finished_at": datetime.now(UTC),
                "crash_reason": reason,
            }
        )
        self.save_experiment(crashed)
        return crashed

    def log_rollout(self, run_id: str, rollout_result: RolloutResult) -> None:
        result = self.load_experiment(run_id)
        result.tasks[rollout_result.task_id] = rollout_result
        self.save_experiment(result)

    def latest_completed_baseline(
        self,
        baseline_commit_hash: str,
        measurement_identity_digest: str,
    ) -> ExperimentResult | None:
        for row in reversed(self.read_index()):
            if (
                row.kind == "baseline"
                and row.crash_reason is None
                and row.git_commit_hash == baseline_commit_hash
                and row.measurement_identity_digest == measurement_identity_digest
            ):
                return self.load_experiment(row.experiment_id)
        return None

    def record_finalized_run(
        self,
        result: ExperimentResult,
        *,
        kind: RunKind,
        parent_commit_hash: str | None = None,
        baseline_experiment_id: str | None = None,
        decision: ResultDecision | None = None,
    ) -> ExperimentResult:
        """Persist loop-owned metadata and append one index row for a finalized run.

        The sampler finalizes a run's intrinsic facts (rollouts and inline config)
        and saves them; the loop owns the relational facts and the gate verdict. Keeping
        the experiment update and index append together prevents one finalized run from
        being recorded through two separate public APIs.
        """
        updated = result.model_copy(
            update={
                "kind": kind,
                "parent_commit_hash": parent_commit_hash,
                "baseline_experiment_id": baseline_experiment_id,
                "decision": decision,
            }
        )
        row = RunIndexRow(
            experiment_id=updated.experiment_id,
            kind=updated.kind,
            git_commit_hash=updated.git_commit_hash,
            measurement_identity_digest=updated.measurement_identity.digest,
            parent_commit_hash=updated.parent_commit_hash,
            baseline_experiment_id=updated.baseline_experiment_id,
            started_at=updated.started_at,
            finished_at=updated.finished_at,
            crash_reason=updated.crash_reason,
            solved=len(solved_task_ids(updated)),
            verdict=None if updated.decision is None else updated.decision.outcome,
            reason=None if updated.decision is None else updated.decision.reason,
        )
        self.save_experiment(updated)
        path = self.runs_index_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n")
        return updated

    def read_index(self) -> list[RunIndexRow]:
        path = self.runs_index_path()
        if not path.exists():
            return []
        return [
            RunIndexRow.model_validate_json(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
