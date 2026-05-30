from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Literal, Sequence

from collections.abc import Callable, Mapping

from src.control import repo as control_repo
from src.experiment.trial import run_task
from src.harness.contracts import TaskResult
from src.metrics import (
    PROMOTION_P_VALUE_ALPHA,
    BaselineComparison,
    compare_candidate_against_baseline,
    is_majority_decided,
    is_majority_solved,
)

if TYPE_CHECKING:
    from src.adapters.env import HarborConfig
    from src.harness.config import HarnessConfig, LlmProviderConfig


EXPERIMENT_FILENAME = "experiment.json"
ExperimentStatus = Literal["keep", "discard", "crash"]


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


@dataclass
class TaskTrials:
    task_name: str
    expected_trial_count: int = 1
    trials: list[TaskResult] = field(default_factory=list)

    @property
    def trial_count(self) -> int:
        return len(self.trials)

    @property
    def finished_trials(self) -> list[TaskResult]:
        return [trial for trial in self.trials if trial.finished_at is not None]

    @property
    def solved_count(self) -> int:
        return sum(1 for trial in self.finished_trials if trial.solved is True)

    @property
    def majority_solved(self) -> bool | None:
        finished = self.finished_trials
        if not finished:
            return None
        return is_majority_solved(solved=self.solved_count, total=len(finished))

    @property
    def is_deterministic_solved(self) -> bool:
        # True iff every observed trial passed. Used by candidates to budget
        # a single trial against baselines that show a task as reliably
        # solved; confirm-on-fail expands back to task_trials if that single
        # candidate trial fails.
        finished = self.finished_trials
        if not finished:
            return False
        return all(trial.solved is True for trial in finished)

    @property
    def representative(self) -> TaskResult | None:
        finished = self.finished_trials
        if not finished:
            return self.trials[-1] if self.trials else None
        majority = self.majority_solved
        if majority is None:
            return finished[-1]
        for trial in reversed(finished):
            if trial.solved is majority:
                return trial
        return finished[-1]

    @property
    def is_finished(self) -> bool:
        return len(self.finished_trials) >= self.expected_trial_count

    def append(self, trial: TaskResult) -> None:
        self.trials.append(trial)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskTrials":
        trials_payload = payload["trials"]
        trials = [TaskResult.from_dict(item) for item in trials_payload]
        return cls(
            task_name=str(payload["task_name"]),
            expected_trial_count=int(payload["expected_trial_count"]),
            trials=trials,
        )

    @classmethod
    def empty(cls, *, task_name: str, expected_trial_count: int) -> "TaskTrials":
        return cls(
            task_name=task_name,
            expected_trial_count=expected_trial_count,
            trials=[],
        )


@dataclass(frozen=True, slots=True)
class CandidateChangeEvidence:
    commit: str
    parent_baseline_experiment_id: str | None = None
    parent_baseline_commit: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateChangeEvidence":
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class TaskOutcomeEvidence:
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
    def from_dict(cls, payload: dict[str, Any]) -> "TaskOutcomeEvidence":
        return cls(**payload)

    @classmethod
    def from_trials(
        cls,
        *,
        task_id: str,
        candidate_trials: "TaskTrials",
        baseline_trials: "TaskTrials" | None,
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


@dataclass(frozen=True, slots=True)
class ExperimentEvidence:
    candidate_change: CandidateChangeEvidence
    task_outcomes: list[TaskOutcomeEvidence] = field(default_factory=list)

    @classmethod
    def empty(cls, *, record: "ExperimentRecord") -> "ExperimentEvidence":
        return cls(
            candidate_change=CandidateChangeEvidence(
                commit=record.git_commit_hash,
                parent_baseline_experiment_id=record.parent_baseline_experiment_id,
            )
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentEvidence":
        return cls(
            candidate_change=CandidateChangeEvidence.from_dict(
                payload["candidate_change"]
            ),
            task_outcomes=[
                TaskOutcomeEvidence.from_dict(task_payload)
                for task_payload in payload["task_outcomes"]
            ],
        )


@dataclass
class ExperimentRecord:
    experiment_id: str
    parent_baseline_experiment_id: str | None
    git_commit_hash: str
    focus_name: str
    train_task_ids: list[str]
    status: ExperimentStatus | None
    train_solved_count: int | None
    decision_reason: str
    error: str
    started_at: str
    finished_at: str | None
    train_task_results: dict[str, TaskTrials]
    evidence: ExperimentEvidence | None = None

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
        payload["train_task_ids"] = sorted(payload["train_task_ids"])
        payload["train_task_results"] = {
            str(task_id): TaskTrials.from_dict(task_trials_payload)
            for task_id, task_trials_payload in payload["train_task_results"].items()
        }
        evidence_payload = payload.pop("evidence")
        record = cls(**payload)
        record.evidence = ExperimentEvidence.from_dict(evidence_payload)
        return record

    @classmethod
    def initialize(
        cls,
        *,
        experiment_id: str,
        git_commit_hash: str,
        parent_baseline_experiment_id: str | None,
        train_task_ids: list[str],
        focus_name: str = "",
        started_at: str,
        expected_trial_count: int = 1,
    ) -> "ExperimentRecord":
        canonical_train_task_ids = sorted(train_task_ids)
        return cls(
            experiment_id=experiment_id,
            parent_baseline_experiment_id=parent_baseline_experiment_id,
            git_commit_hash=git_commit_hash,
            focus_name=focus_name,
            train_task_ids=canonical_train_task_ids,
            status=None,
            train_solved_count=0,
            decision_reason="",
            error="",
            started_at=started_at,
            finished_at=None,
            train_task_results={
                task_id: TaskTrials.empty(
                    task_name=task_id,
                    expected_trial_count=expected_trial_count,
                )
                for task_id in canonical_train_task_ids
            },
            evidence=None,
        )

    def write(self, *, root: Path) -> None:
        if self.evidence is None:
            self.evidence = ExperimentEvidence.empty(record=self)
        payload = asdict(self)
        write_json_atomic(self.path(self.experiment_id, root=root), payload)

    def record_task_result(self, task_result: TaskResult) -> None:
        panel_results = self._panel_results(task_result.task_name)
        panel_results[task_result.task_name].append(task_result)
        self.train_solved_count = sum(
            1
            for trials in self.train_task_results.values()
            if trials.majority_solved is True
        )

    def finalize(
        self,
        *,
        status: ExperimentStatus,
        error: str | None = None,
        decision_reason: str | None = None,
    ) -> None:
        if any(not trials.is_finished for trials in self.train_task_results.values()):
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
        baseline: "ExperimentRecord" | None,
        verdicts: Mapping[str, BaselineComparison] | None = None,
    ) -> None:
        self.evidence = build_experiment_evidence(
            candidate=self,
            baseline=baseline,
            verdicts=verdicts,
        )

    def is_concluded(self) -> bool:
        return (
            self.status in {"keep", "discard", "crash"} and self.finished_at is not None
        )

    def _panel_results(self, task_id: str) -> dict[str, TaskTrials]:
        if task_id in self.train_task_results:
            return self.train_task_results
        raise KeyError(
            f"task {task_id!r} is not part of experiment {self.experiment_id}"
        )

    def _task_trials(self, task_id: str) -> TaskTrials:
        return self._panel_results(task_id)[task_id]

    def _complete_unfinished_task_results(self, *, exc: BaseException) -> None:
        for task_id in self.train_task_results:
            trials = self._task_trials(task_id)
            while not trials.is_finished:
                self.record_task_result(_terminal_task_result(task_id=task_id, exc=exc))


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
    baseline_train_results = {} if baseline is None else baseline.train_task_results
    verdicts_map: Mapping[str, BaselineComparison] = verdicts or {}
    return ExperimentEvidence(
        candidate_change=CandidateChangeEvidence(
            commit=candidate.git_commit_hash,
            parent_baseline_experiment_id=candidate.parent_baseline_experiment_id,
            parent_baseline_commit=None
            if baseline is None
            else baseline.git_commit_hash,
        ),
        task_outcomes=[
            TaskOutcomeEvidence.from_trials(
                task_id=task_id,
                candidate_trials=task_trials,
                baseline_trials=baseline_train_results.get(task_id),
                verdict=verdicts_map.get(task_id),
            )
            for task_id, task_trials in candidate.train_task_results.items()
        ],
    )


def build_gate_verdicts(
    *,
    candidate: ExperimentRecord,
    pool: Mapping[str, tuple[int, int]],
) -> dict[str, BaselineComparison]:
    """Single source of truth for per-task verdicts.

    For each task in the candidate's train panel, compute the
    :class:`BaselineComparison` against the pooled-control samples. The gate
    and persisted evidence both consume this dict so there is exactly one
    verdict per task per candidate.

    Tasks absent from ``pool`` get treated as no-baseline frontier
    (baseline_total == 0) inside
    :func:`compare_candidate_against_baseline`.
    """
    verdicts: dict[str, BaselineComparison] = {}
    for task_id, candidate_trials in candidate.train_task_results.items():
        baseline_solved, baseline_total = pool.get(task_id, (0, 0))
        verdicts[task_id] = compare_candidate_against_baseline(
            candidate_solved=candidate_trials.solved_count,
            candidate_total=candidate_trials.trial_count,
            baseline_solved=baseline_solved,
            baseline_total=baseline_total,
            alpha=PROMOTION_P_VALUE_ALPHA,
        )
    return verdicts


def decide_from_verdicts(
    *,
    candidate: ExperimentRecord,
    verdicts: Mapping[str, BaselineComparison],
) -> tuple[ExperimentStatus, str]:
    """Resolve verdicts into a promotion decision.

    Iterates the candidate's panel in ``train_task_ids`` order so the
    decision reason deterministically names the first triggering task.
    Regressions take priority over improvements.
    """
    panel_order = tuple(candidate.train_task_ids)
    for task_id in panel_order:
        verdict = verdicts.get(task_id)
        if verdict is not None and verdict.kind == "regression":
            return "discard", f"train task {task_id} regressed"
    for task_id in panel_order:
        verdict = verdicts.get(task_id)
        if verdict is not None and verdict.kind == "improvement":
            return "keep", f"train task {task_id} improved"
    return "discard", "no train task improvement reached significance"


# ----------------------------------------------------------------------------
# Pool construction for the promotion gate.
# ----------------------------------------------------------------------------

RULE_DIFF_PATHS: tuple[str, ...] = ("src/harness/core.py",)

# Mechanism candidates surface a rule name in the diff three ways:
#   1. Dataclass declarations like ArgumentRule(name="foo", ...)
#   2. Constant assignments like FOO_RULE_NAME = "foo" that get passed to
#      record_rule_fire later in the file (or already-defined elsewhere)
#   3. Direct record_rule_fire("foo") calls with a string literal
RULE_NAME_DIFF_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r'name=["\']([A-Za-z_][A-Za-z0-9_]*)["\']'),
    re.compile(r'_RULE_NAME\s*[:=][\s(]*["\']([A-Za-z_][A-Za-z0-9_]*)["\']'),
    re.compile(r'record_rule_fire\(\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']\s*\)'),
)


def rule_names_from_added_lines(added_lines: list[str]) -> set[str]:
    # Joined-text scan so the _RULE_NAME constant pattern can cross the
    # `FOO_RULE_NAME = (` line into the indented string literal that
    # follows. Per-line scanning misses multi-line constant definitions
    # common in PEP-8 wrapped declarations.
    text = "\n".join(added_lines)
    names: set[str] = set()
    for pattern in RULE_NAME_DIFF_PATTERNS:
        for match in pattern.finditer(text):
            names.add(match.group(1))
    return names


def candidate_new_rule_names(
    *,
    workspace_root: Path,
    record: ExperimentRecord,
) -> tuple[str, ...]:
    """Return rule names added by ``record`` relative to its parent baseline.

    Scoped to RULE_DIFF_PATHS (src/harness/core.py only) so
    test fixtures using ``name="..."`` literals don't pollute the result.
    Returns () when the parent baseline commit is unknown — for pooled-control
    filtering we prefer silence over a guess.
    """
    parent_ref = (
        record.evidence.candidate_change.parent_baseline_commit
        if record.evidence is not None
        else None
    )
    if parent_ref is None:
        return ()
    added_lines = control_repo.git_diff_added_lines_between(
        cwd=workspace_root,
        base_ref=parent_ref,
        head_ref=record.git_commit_hash,
        paths=RULE_DIFF_PATHS,
    )
    return tuple(sorted(rule_names_from_added_lines(added_lines)))


# Decision reasons that mark a record as baseline bookkeeping rather than a
# real candidate. These records must not contribute to pooled-control samples
# via recent-candidate history because they don't represent a proposed mechanism.
BASELINE_DECISION_REASONS: frozenset[str] = frozenset(
    {
        "baseline seed",
        "baseline rerun",
    }
)

# Default lookback for pooled-control aggregation.
POOLED_CONTROL_WINDOW = 20


def load_recent_candidate_records(
    *,
    experiments_root: Path,
    window: int = POOLED_CONTROL_WINDOW,
) -> list[ExperimentRecord]:
    """Most-recently-finished concluded candidate records, newest first.

    Filters: only records with a parent baseline, only concluded. Caller is responsible for
    further filtering (e.g., excluding the candidate currently being
    evaluated).
    """
    records: list[ExperimentRecord] = []
    if not experiments_root.exists():
        return records
    for child in sorted(experiments_root.iterdir()):
        if not child.is_dir():
            continue
        if not ExperimentRecord.path(child.name, root=experiments_root).exists():
            continue
        try:
            record = ExperimentRecord.load(child.name, root=experiments_root)
        except Exception:
            continue
        if record.is_concluded() and record.parent_baseline_experiment_id is not None:
            records.append(record)
    records.sort(key=lambda r: r.finished_at or r.started_at, reverse=True)
    return records[:window]


def build_pooled_control_samples(
    *,
    active_baseline: "ExperimentRecord",
    recent_candidates: Sequence["ExperimentRecord"],
    candidate_new_rule_names: Mapping[str, tuple[str, ...]],
    task_ids: Sequence[str],
) -> dict[str, tuple[int, int]]:
    """Build (pooled_solved, pooled_total) per task for the noise model.

    Composition:
    1. Active-baseline trials for the task (always included unless the trial
       carries an error marker).
    2. For each non-crashed record in `recent_candidates` whose
       `decision_reason` is not a baseline-bookkeeping marker, include only
       error-free trials where no rule named in that candidate's
       `new_rule_names` fired. The rule filter removes trials structurally
       affected by the candidate's mechanism; the error filter removes
       crash/skip sentinels.

    Caller is responsible for providing `recent_candidates` (typically the
    most recent N concluded candidates) and mapping each candidate's
    experiment_id to its newly added rule names (extracted from git diff).
    """
    pool: dict[str, tuple[int, int]] = {task_id: (0, 0) for task_id in task_ids}

    def _add(task_id: str, solved: bool) -> None:
        s, n = pool[task_id]
        pool[task_id] = (s + (1 if solved else 0), n + 1)

    def eligible_trial(trial: TaskResult) -> bool:
        # `trial.error` is set only on infrastructure failure (timeout or
        # raised exception in `run_task`); a step-cap exhaustion has
        # `error=None` and `failure_mode="hit_step_cap"`. The `is None`
        # check is load-bearing: `str(exc)` for a bare exception is "",
        # which a truthiness test would silently let through.
        return trial.error is None

    def touched_by_new_rules(
        trial: TaskResult,
        new_rule_names: Sequence[str],
    ) -> bool:
        fires = trial.metrics.rule_fires
        return any(fires.get(name, 0) > 0 for name in new_rule_names)

    for task_id in task_ids:
        baseline_trials = active_baseline.train_task_results.get(task_id)
        if baseline_trials is None:
            continue
        for trial in baseline_trials.finished_trials:
            if not eligible_trial(trial):
                continue
            _add(task_id, trial.solved is True)

    for record in recent_candidates:
        # The active baseline is itself a concluded candidate with a parent
        # (every promoted record is), so load_recent_candidate_records picks
        # it up alongside older candidates. Without this guard the baseline's
        # trials would be counted once via the loop above and again here.
        if record.experiment_id == active_baseline.experiment_id:
            continue
        if record.status == "crash":
            continue
        if record.decision_reason in BASELINE_DECISION_REASONS:
            continue
        new_rules = candidate_new_rule_names.get(record.experiment_id, ())
        for task_id in task_ids:
            trials = record.train_task_results.get(task_id)
            if trials is None:
                continue
            for trial in trials.finished_trials:
                if not eligible_trial(trial):
                    continue
                if touched_by_new_rules(trial, new_rules):
                    continue
                _add(task_id, trial.solved is True)

    return pool


def build_gate_pool(
    *,
    experiments_root: Path,
    workspace_root: Path,
    active_baseline: "ExperimentRecord",
    candidate_experiment_id: str,
    task_ids: Sequence[str],
    window: int = POOLED_CONTROL_WINDOW,
) -> dict[str, tuple[int, int]]:
    """Assemble the (solved, total) pool the promotion gate compares against.

    Includes the active baseline's own trials and the most recent N
    concluded candidates' trials, filtered inside
    :func:`build_pooled_control_samples` to drop crashes, baseline-bookkeeping
    seeds, and trials touched by each candidate's own new rule. Excludes
    the candidate currently being evaluated.
    """
    recent = [
        record
        for record in load_recent_candidate_records(
            experiments_root=experiments_root,
            window=window,
        )
        if record.experiment_id != candidate_experiment_id
    ]
    rule_names_by_id = {
        record.experiment_id: candidate_new_rule_names(
            workspace_root=workspace_root,
            record=record,
        )
        for record in recent
    }
    return build_pooled_control_samples(
        active_baseline=active_baseline,
        recent_candidates=recent,
        candidate_new_rule_names=rule_names_by_id,
        task_ids=task_ids,
    )


@dataclass
class ExperimentState:
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
        return cls(**json.loads(path.read_text()))

    def save(self, *, root: Path) -> None:
        write_json_atomic(self.path(root=root), asdict(self))


def _raise_if_any_trial_errored(record: ExperimentRecord) -> None:
    # `trial.error` always means infrastructure failure (timeout/exception in
    # `run_task`). An errored trial conflated with a "real failed solve"
    # would feed a misattributed verdict to the gate and, for baseline runs,
    # poison every future candidate comparison via the pool. The `is not
    # None` check is load-bearing: `str(exc)` for a bare exception is "",
    # which a truthiness test would silently let through.
    for task_id, trials in record.train_task_results.items():
        for trial in trials.trials:
            if trial.error is not None:
                raise RuntimeError(f"task {task_id} trial failed: {trial.error}")


def _terminal_task_result(*, task_id: str, exc: BaseException) -> TaskResult:
    finished_at = datetime.now(timezone.utc).isoformat()
    if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
        error = "canceled"
    else:
        error = str(exc) or type(exc).__name__
    return TaskResult(
        task_name=task_id,
        reward=0.0,
        solved=False,
        error=error,
        steps_used=0,
        started_at=finished_at,
        finished_at=finished_at,
    )


def _prepare_task_dirs(
    *,
    trial_harbor_config: HarborConfig,
    task_names: Sequence[str],
) -> dict[str, Path]:
    from src.adapters.env import TaskDirectoryResolver

    return dict(TaskDirectoryResolver(trial_harbor_config).resolve(list(task_names)))


async def _run_panel(
    *,
    record: ExperimentRecord,
    experiments_root: Path,
    task_names: Sequence[str],
    task_dirs: Mapping[str, Path],
    harness_config: HarnessConfig,
    make_llm: Callable[[], Any],
    make_env: Callable[..., Any],
) -> None:
    semaphore = asyncio.Semaphore(harness_config.max_concurrency)

    def persist_task_result(task_result: TaskResult) -> None:
        record.record_task_result(task_result)
        record.write(root=experiments_root)

    async def run_one_trial(task_id: str) -> None:
        async with semaphore:
            task_dir = task_dirs[task_id]
            try:
                trial_result = await run_task(
                    task_name=task_id,
                    llm=make_llm(),
                    env=make_env(task_id, task_dir=task_dir),
                    max_steps=harness_config.max_steps,
                    max_output_retries=harness_config.max_output_retries,
                    task_timeout_sec=harness_config.task_timeout_sec,
                )
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                persist_task_result(_terminal_task_result(task_id=task_id, exc=exc))
                if isinstance(exc, Exception):
                    return
                raise
            persist_task_result(trial_result)

    async def run_task_trials(task_id: str) -> None:
        full_trial_count = harness_config.task_trials
        while not record._task_trials(task_id).is_finished:
            trials = record._task_trials(task_id)
            remaining_budget = trials.expected_trial_count - len(trials.finished_trials)
            if remaining_budget <= 0:
                break
            in_flight: set[asyncio.Task[None]] = {
                asyncio.create_task(run_one_trial(task_id))
                for _ in range(remaining_budget)
            }
            try:
                while in_flight:
                    done, in_flight = await asyncio.wait(
                        in_flight, return_when=asyncio.FIRST_COMPLETED
                    )
                    for finished in done:
                        finished.result()
                    trials = record._task_trials(task_id)
                    if (
                        trials.expected_trial_count == 1
                        and full_trial_count > 1
                        and trials.solved_count == 0
                        and len(trials.finished_trials) == 1
                    ):
                        trials.expected_trial_count = full_trial_count
                        record.write(root=experiments_root)
                        break
                    decided = is_majority_decided(
                        solved=trials.solved_count,
                        finished=len(trials.finished_trials),
                        expected_total=trials.expected_trial_count,
                    )
                    if decided and not trials.is_finished:
                        for pending in in_flight:
                            pending.cancel()
                        if in_flight:
                            await asyncio.gather(*in_flight, return_exceptions=True)
                        in_flight = set()
                        trials.expected_trial_count = len(trials.finished_trials)
                        record.write(root=experiments_root)
            except BaseException:
                for pending in in_flight:
                    pending.cancel()
                if in_flight:
                    await asyncio.gather(*in_flight, return_exceptions=True)
                raise

    await asyncio.gather(*(run_task_trials(task_id) for task_id in task_names))


def _make_llm_for_config(*, config: LlmProviderConfig, api_key: str | None):
    match config.provider:
        case "openrouter":
            if api_key is None:
                raise ValueError("OPENROUTER_API_KEY is not set")
            from src.adapters.open_router import OpenRouter

            return OpenRouter(config=config, api_key=api_key)
        case "chatgpt_codex":
            from src.adapters.chatgpt_codex import ChatGptCodex

            return ChatGptCodex(config=config)


class ExperimentRunner:
    @classmethod
    def run_baseline_at_head(
        cls,
        *,
        harness_config: HarnessConfig,
        harbor_config: HarborConfig,
        api_key: str | None,
        decision_reason: Literal["baseline seed", "baseline rerun"] = "baseline rerun",
        experiment_id: str | None = None,
        started_at: str | None = None,
        repo_root: Path | None = None,
    ) -> ExperimentRecord:
        """Run the current HEAD as the active baseline over the full panel."""
        experiments_root = harbor_config.experiments_dir
        state = ExperimentState.load(root=experiments_root)
        baseline_id = state.active_baseline_experiment_id
        baseline = (
            None
            if baseline_id is None
            else ExperimentRecord.load(baseline_id, root=experiments_root)
        )
        if baseline is not None:
            if baseline.status != "keep" or not baseline.is_concluded():
                raise RuntimeError(
                    f"active baseline {baseline.experiment_id} must be a concluded keep record"
                )
        current_id = state.current_experiment_id
        if current_id is not None and current_id != baseline_id:
            current_record = ExperimentRecord.load(current_id, root=experiments_root)
            if not current_record.is_concluded() and any(
                task_trials.trial_count > 0
                for task_trials in current_record.train_task_results.values()
            ):
                raise RuntimeError(
                    "cannot run baseline while the current experiment has recorded task activity"
                )
        control_repo.require_clean_worktree(cwd=repo_root)
        git_commit_hash = control_repo.get_head_commit(cwd=repo_root)
        if baseline is not None:
            if baseline.git_commit_hash == git_commit_hash and set(
                harness_config.train_task_names
            ) == set(baseline.train_task_ids):
                state.current_experiment_id = baseline.experiment_id
                state.updated_at = datetime.now(timezone.utc).isoformat()
                state.save(root=experiments_root)
                return baseline

        baseline_at = (
            datetime.now(timezone.utc).isoformat() if started_at is None else started_at
        )
        resolved_experiment_id = (
            f"baseline-{datetime.fromisoformat(baseline_at).strftime('%Y%m%d-%H%M%S')}"
            if experiment_id is None
            else experiment_id
        )
        if ExperimentRecord.path(
            resolved_experiment_id, root=experiments_root
        ).exists():
            raise RuntimeError(
                f"baseline experiment already exists: {resolved_experiment_id}"
            )
        record = ExperimentRecord.initialize(
            experiment_id=resolved_experiment_id,
            git_commit_hash=git_commit_hash,
            parent_baseline_experiment_id=baseline_id,
            train_task_ids=list(harness_config.train_task_names),
            focus_name=harness_config.focus_name,
            started_at=baseline_at,
            expected_trial_count=harness_config.task_trials,
        )
        experiment_dir = experiments_root / resolved_experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=False)
        record.write(root=experiments_root)

        try:
            from src.adapters.env import Harbor

            trial_harbor_config = harbor_config.model_copy(
                update={"experiments_dir": experiment_dir / "tasks"}
            )
            task_dirs = _prepare_task_dirs(
                trial_harbor_config=trial_harbor_config,
                task_names=harness_config.train_task_names,
            )
            asyncio.run(
                _run_panel(
                    record=record,
                    experiments_root=experiments_root,
                    task_names=harness_config.train_task_names,
                    task_dirs=task_dirs,
                    harness_config=harness_config,
                    make_llm=lambda: _make_llm_for_config(
                        config=harness_config.llm_provider_config,
                        api_key=api_key,
                    ),
                    make_env=lambda task_name, *, task_dir: Harbor(
                        trial_harbor_config,
                        task_name=task_name,
                        task_dir=task_dir,
                    ),
                )
            )
            _raise_if_any_trial_errored(record)
        except BaseException as exc:
            record._complete_unfinished_task_results(exc=exc)
            record.finalize(status="crash", error=str(exc))
            record.refresh_evidence(baseline=baseline)
            record.write(root=experiments_root)
            if isinstance(exc, Exception):
                return record
            raise

        record.finalize(status="keep", decision_reason=decision_reason)
        record.refresh_evidence(baseline=baseline)
        record.write(root=experiments_root)
        state.active_baseline_experiment_id = record.experiment_id
        state.current_experiment_id = record.experiment_id
        state.updated_at = record.finished_at
        state.save(root=experiments_root)
        return record

    def __init__(
        self,
        *,
        harness_config: HarnessConfig,
        harbor_config: HarborConfig,
        api_key: str | None,
        require_clean_worktree: bool = True,
    ) -> None:
        self.harness_config = harness_config
        self.harbor_config = harbor_config
        self.api_key = api_key
        if require_clean_worktree:
            control_repo.require_clean_worktree()
        self.experiments_root = self.harbor_config.experiments_dir
        self.experiment_dir = self.experiments_root / self.harness_config.experiment_id
        self.state = ExperimentState.load(root=self.experiments_root)
        self.frozen_baseline_experiment_id = self.state.active_baseline_experiment_id
        self.record = ExperimentRecord.initialize(
            experiment_id=self.harness_config.experiment_id,
            git_commit_hash=control_repo.get_head_commit(),
            parent_baseline_experiment_id=self.frozen_baseline_experiment_id,
            train_task_ids=list(self.harness_config.train_task_names),
            focus_name=self.harness_config.focus_name,
            started_at=datetime.now(timezone.utc).isoformat(),
            expected_trial_count=self.harness_config.task_trials,
        )
        self._task_dirs: dict[str, Path] = {}
        # Held for tests and terminal paths that need the parent baseline.
        self.baseline: ExperimentRecord | None = None

    def _trial_harbor_config(self):
        return self.harbor_config.model_copy(
            update={"experiments_dir": self.experiment_dir / "tasks"}
        )

    def _make_env(self, task_name: str, task_dir: Path):
        from src.adapters.env import Harbor

        return Harbor(
            self._trial_harbor_config(),
            task_name=task_name,
            task_dir=task_dir,
        )

    def _make_llm(self):
        return _make_llm_for_config(
            config=self.harness_config.llm_provider_config,
            api_key=self.api_key,
        )

    def _apply_baseline_derived_trial_counts(self, baseline: ExperimentRecord) -> None:
        # Where the baseline shows a task as deterministic-solved (every
        # observed trial passed), budget the candidate for a single trial.
        # If that trial fails, run_task_trials' confirm-on-fail expands the
        # budget back to task_trials to verify the suspected regression.
        if self.harness_config.task_trials <= 1:
            return
        changed = False
        for task_id, baseline_trials in baseline.train_task_results.items():
            candidate_trials = self.record.train_task_results.get(task_id)
            if candidate_trials is None:
                continue
            if baseline_trials.is_deterministic_solved:
                candidate_trials.expected_trial_count = 1
                changed = True
        if changed:
            self.record.write(root=self.experiments_root)

    def run(self) -> ExperimentRecord:
        self.experiment_dir.mkdir(parents=True, exist_ok=False)
        baseline_id = self.frozen_baseline_experiment_id
        baseline = (
            None
            if baseline_id is None
            else ExperimentRecord.load(baseline_id, root=self.experiments_root)
        )
        self.baseline = baseline

        try:
            self._validate_setup_contract(
                state=self.state,
                candidate=self.record,
                baseline=baseline,
            )
            self._task_dirs = _prepare_task_dirs(
                trial_harbor_config=self._trial_harbor_config(),
                task_names=self.harness_config.train_task_names,
            )
            self.record.started_at = datetime.now(timezone.utc).isoformat()
            self.record.write(root=self.experiments_root)
            self.state.current_experiment_id = self.record.experiment_id
            self.state.updated_at = self.record.started_at
            self.state.save(root=self.experiments_root)
            return asyncio.run(self._run_experiment(baseline=baseline))
        except BaseException as exc:
            self.record._complete_unfinished_task_results(exc=exc)
            self.record.finalize(status="crash", error=str(exc))
            self.record.refresh_evidence(baseline=baseline)
            self.record.write(root=self.experiments_root)
            self._conclude_experiment()
            if isinstance(exc, Exception):
                return self.record
            raise

    async def _run_experiment(
        self,
        *,
        baseline: ExperimentRecord | None,
    ) -> ExperimentRecord:
        try:
            if baseline is not None:
                self._apply_baseline_derived_trial_counts(baseline)
            await _run_panel(
                record=self.record,
                experiments_root=self.experiments_root,
                task_names=list(self.harness_config.train_task_names),
                task_dirs=self._task_dirs,
                harness_config=self.harness_config,
                make_llm=self._make_llm,
                make_env=self._make_env,
            )
            self._validate_record_ready_for_evaluation(self.record)
            pool = self._build_gate_pool(baseline=baseline)
            verdicts = build_gate_verdicts(candidate=self.record, pool=pool)
            status, decision_reason = decide_from_verdicts(
                candidate=self.record, verdicts=verdicts
            )
            self.record.finalize(status=status, decision_reason=decision_reason)
            # Thread the gate's verdict dict into the persisted evidence so
            # downstream readers, including supervisor artifact selection,
            # read the same per-task verdict the promotion decision used.
            self.record.refresh_evidence(baseline=baseline, verdicts=verdicts)
            self.record.write(root=self.experiments_root)
            self._conclude_experiment()
        except Exception as exc:
            self.record._complete_unfinished_task_results(exc=exc)
            self.record.finalize(status="crash", error=str(exc))
            # Crash path: the gate did not produce verdicts. Evidence
            # labels stay "uncompared" rather than being recomputed via
            # a second mechanism.
            self.record.refresh_evidence(baseline=baseline)
            self.record.write(root=self.experiments_root)
            self._conclude_experiment()

        return self.record

    def _build_gate_pool(
        self,
        *,
        baseline: ExperimentRecord | None,
    ) -> dict[str, tuple[int, int]]:
        # No baseline yet: no-baseline frontier path for every task.
        # compare_candidate_against_baseline handles (0,0) entries by
        # requiring a candidate majority-solve for improvement.
        if baseline is None:
            return {}
        return build_gate_pool(
            experiments_root=self.experiments_root,
            workspace_root=Path.cwd(),
            active_baseline=baseline,
            candidate_experiment_id=self.record.experiment_id,
            task_ids=tuple(self.harness_config.train_task_names),
        )

    def _conclude_experiment(self) -> None:
        status = self.record.status
        if status is None:
            raise RuntimeError(
                "terminal outcome requires a finalized experiment status"
            )
        if self.record.finished_at is None:
            raise RuntimeError("terminal outcome requires a finished timestamp")
        self.state.current_experiment_id = self.record.experiment_id
        self.state.updated_at = self.record.finished_at
        match status:
            case "keep":
                self.state.active_baseline_experiment_id = self.record.experiment_id
                self.state.save(root=self.experiments_root)
                return
            case _:
                preserved_git_ref = failed_experiment_git_ref(self.record.experiment_id)
                control_repo.update_ref(preserved_git_ref, self.record.git_commit_hash)
                self.state.save(root=self.experiments_root)
                return

    def _validate_setup_contract(
        self,
        *,
        state: ExperimentState,
        candidate: ExperimentRecord,
        baseline: ExperimentRecord | None,
    ) -> None:
        if (
            candidate.parent_baseline_experiment_id
            != state.active_baseline_experiment_id
        ):
            raise ValueError(
                "candidate parent baseline must match the frozen active baseline"
            )
        if set(candidate.train_task_ids) != set(candidate.train_task_results):
            raise ValueError(
                "candidate task results must cover exactly the configured task ids"
            )
        if baseline is None:
            return
        if candidate.train_task_ids != baseline.train_task_ids:
            raise ValueError(
                "candidate train panel must match the frozen baseline train panel"
            )

    def _validate_record_ready_for_evaluation(self, record: ExperimentRecord) -> None:
        if any(not trials.is_finished for trials in record.train_task_results.values()):
            raise RuntimeError("all task results must be finished before evaluation")
        if record.error:
            raise RuntimeError(
                "experiment record must not carry a top-level error before evaluation"
            )
        _raise_if_any_trial_errored(record)
