"""Autonomous self-improvement session boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Generator, Iterator
from pathlib import Path

from git import Repo

from src.config import RunConfig, load_config_payload
from src.measurement import MeasurementError, preflight, run_isolated_experiment
from src.rollout.records import (
    ExperimentResult,
    ResultDecision,
    RunKind,
)
from src.rollout.certification import resolve_measurement_identity
from src.rollout.execution import resolve_execution
from src.rollout.store import invoker_repo_root, RunStore
from src.trainer.logger import StdoutLogger
from src.trainer.estimator import Estimator
from src.trainer.parameter import (
    Candidate,
    CandidateValidationError,
    Parameter,
    capture_candidate,
    run_candidate_suite,
    scratch_worktree,
    validate_candidate,
)
from src.trainer.loss import Criterion, Loss, UnmeasurableRun
from src.trainer.optim import Optimizer


class CircuitBreakerTripped(RuntimeError):
    """The session hit the fixed consecutive-epoch skip safety bound."""


MAX_CONSECUTIVE_SKIPS = 2


class Trainer:
    """Deterministic self-improvement session: measurement, persistence, worktrees.

    The epoch concludes when the loop resumes the generator; abandoning the loop
    (break or an exception) leaves the pending epoch unconcluded.

    Loop contract and epoch order: src/trainer/README.md.
    """

    def __init__(
        self,
        *,
        config_path: str | Path,
        estimator: Estimator,
        criterion: Criterion,
        optimizer: Optimizer,
        observer: StdoutLogger | None = None,
        default_root_dir: Path | None = None,
        worktrees_root: Path | None = None,
    ) -> None:
        self.repo_root = invoker_repo_root()
        repo = Repo(self.repo_root)
        config_path = _tracked_repo_relative_path(str(config_path), repo=repo)
        # The worker measures the committed config; one load makes skew unrepresentable.
        payload, source_paths = load_config_payload(self.repo_root / config_path)
        root = self.repo_root.resolve()
        config_sources = tuple(
            source.relative_to(root).as_posix() for source in source_paths
        )
        # --porcelain lists untracked sources (??) too: tracked-ness and dirtiness in one call.
        dirty = repo.git.status("--porcelain", "--", *config_sources)
        if dirty:
            files = ", ".join(line[3:] for line in dirty.splitlines())
            raise ValueError(
                f"measurement config sources have uncommitted changes ({files}); "
                "commit them before training"
            )
        self.run_config = RunConfig.model_validate(payload).model_copy(
            update={"config_path": config_path}
        )
        # Subject declarations that would corrupt the grade.
        target = self.run_config.training_target
        overlap = sorted(set(config_sources) & set(target.patch_paths))
        if overlap:
            raise ValueError(
                "training_target patch paths must not include the measurement "
                f"config sources: {', '.join(overlap)}"
            )
        if not target.proposer_visible:
            raise ValueError(
                "training_target.proposer_visible must not be empty: the "
                "proposer checkout would expose the whole repo, including the "
                "trainer gate"
            )
        self._tracker = RunStore(
            self.repo_root / "experiments"
            if default_root_dir is None
            else default_root_dir
        )
        # Sibling directory: writes must not enter the captured checkout.
        self.worktrees_root = (
            self.repo_root.parent / f"{self.repo_root.name}-worktrees"
            if worktrees_root is None
            else worktrees_root
        )
        self._observer = observer if observer is not None else StdoutLogger()
        self.estimator = estimator
        self.criterion = criterion
        self.optimizer = optimizer
        self._harness = Parameter(repo)
        self.criterion.harness = self._harness
        self.optimizer.harness = self._harness

    def epochs(self, max_epochs: int) -> Iterator[Loss]:
        if max_epochs < 0:
            raise ValueError("max_epochs must be non-negative")
        return self._epochs(max_epochs)

    def _epochs(self, max_epochs: int) -> Generator[Loss, None, None]:
        repo = self._harness.repo
        if repo.is_dirty():
            dirty_files = repo.git.diff("--name-only", "HEAD").splitlines()
            self._observer.log(
                "warning: dirty tracked files: "
                f"{', '.join(dirty_files)}; uncommitted changes will NOT be measured "
                "(worktrees run committed code only)"
            )
        decisions: list[ResultDecision] = []
        final_baseline: ExperimentResult | None = None
        consecutive_skip_causes: list[str] = []
        # Replay adopts each prior task certificate only when identity and chain match.
        certified_baseline: ExperimentResult | None = None
        execution = resolve_execution(self.run_config)
        for epoch_index in range(1, max_epochs + 1):
            baseline_commit = repo.head.commit.hexsha
            if epoch_index == 1:
                # The banner shows the sha the first epoch actually measures.
                self._banner(baseline_commit)
            identity = asyncio.run(
                resolve_measurement_identity(self.run_config, execution)
            )
            baseline = self._tracker.latest_completed_baseline(
                baseline_commit, identity.digest
            )
            self._observer.epoch_started(epoch_index, max_epochs, baseline)
            # Recheck every epoch; the endpoint may scale down, and smoke warms it.
            preflight([self.run_config])
            if baseline is None:
                baseline = self._measure_baseline(repo, baseline_commit)
            baseline, task_ids = asyncio.run(
                execution.certify(
                    tracker=self._tracker,
                    baseline=baseline,
                    log=self._observer.log,
                    inherited=certified_baseline,
                )
            )
            certified_baseline = baseline
            final_baseline = baseline
            skip_cause, finalized = yield from self._epoch(
                repo=repo,
                baseline_commit=baseline_commit,
                baseline=baseline,
                task_ids=task_ids,
            )
            decision = None if finalized is None else finalized.decision
            self._observer.epoch_finished(
                "skipped" if decision is None else decision.outcome
            )
            if decision is not None:
                decisions.append(decision)
            if finalized is not None and finalized.kind == "baseline":
                final_baseline = finalized
            if skip_cause is None:
                consecutive_skip_causes = []
            else:
                consecutive_skip_causes.append(skip_cause)
            if len(consecutive_skip_causes) >= MAX_CONSECUTIVE_SKIPS:
                raise CircuitBreakerTripped(
                    "consecutive epoch skip breaker tripped: "
                    f"{', '.join(consecutive_skip_causes)}"
                )
        self._observer.loop_finished(max_epochs, decisions, final_baseline)

    def _banner(self, baseline_commit: str) -> None:
        config = self.run_config
        llm = config.llm_provider_config
        self._observer.run_started(
            f"train · {config.config_path}",
            (
                (
                    "policy",
                    f"{config.training_target.surface} @ {baseline_commit[:8]}",
                ),
                ("llm", f"{llm.model_name} · {llm.base_url}"),
                (
                    "env",
                    f"{config.environment.kind} · "
                    f"{len(config.environment.task_names)} tasks · "
                    f"concurrency {config.max_rollout_concurrency}",
                ),
                ("criterion", self.criterion.describe()),
            ),
        )

    def _measure_baseline(self, repo: Repo, baseline_commit: str) -> ExperimentResult:
        self._observer.measurement_started(
            tuple(self.run_config.environment.task_names),
            None,
            subject="baseline",
        )
        with scratch_worktree(
            repo,
            commit=baseline_commit,
            root=self.worktrees_root,
            name="measure",
        ) as measure_root:
            baseline = run_isolated_experiment(
                config_path=self.run_config.config_path,
                tracker=self._tracker,
                observer=self._observer,
                measure_root=measure_root,
                task_ids=tuple(self.run_config.environment.task_names),
            )
        baseline = self._tracker.record_finalized_run(baseline, kind="baseline")
        self._observer.experiment_finished(baseline)
        return baseline

    def _epoch(
        self,
        *,
        repo: Repo,
        baseline_commit: str,
        baseline: ExperimentResult,
        task_ids: tuple[str, ...],
    ) -> Generator[Loss, None, tuple[str | None, ExperimentResult | None]]:
        with scratch_worktree(
            repo,
            commit=baseline_commit,
            root=self.worktrees_root,
            name="propose",
            sparse=self.run_config.training_target.proposer_visible,
        ) as propose_root:
            (propose_root / ".venv").symlink_to(self.repo_root / ".venv")
            try:
                self.estimator.propose(
                    repo_root=propose_root,
                    tracker=self._tracker,
                    target=self.run_config.training_target,
                    emit=self._observer.agent_progress,
                )
                propose_repo = Repo(propose_root)
                candidate_ref = capture_candidate(
                    propose_repo,
                    base_commit=baseline_commit,
                )
                target = self.run_config.training_target
                validate_candidate(
                    propose_repo,
                    candidate_ref,
                    surface=target.surface,
                    patch_paths=target.patch_paths,
                )
                # Trusted suite run; the proposer's own pytest is advisory.
                run_candidate_suite(repo, candidate_ref, root=self.worktrees_root)
            except CandidateValidationError as exc:
                self._observer.agent_progress("propose", f"failed · {exc}")
                self._observer.epoch_skipped(exc.cause)
                return exc.cause, None
            if candidate_ref.base_commit != baseline_commit:
                raise RuntimeError(
                    "candidate baseline does not match the epoch baseline: "
                    f"{candidate_ref.base_commit} != {baseline_commit}"
                )
            # Re-measure once; the superseded first attempt stays artifact-only.
            for attempt_index in range(2):
                retry = attempt_index == 0
                try:
                    self._observer.measurement_started(
                        task_ids, baseline, subject="candidate"
                    )
                    with scratch_worktree(
                        repo,
                        commit=candidate_ref.commit,
                        root=self.worktrees_root,
                        name="measure",
                    ) as measure_root:
                        candidate = run_isolated_experiment(
                            config_path=self.run_config.config_path,
                            tracker=self._tracker,
                            observer=self._observer,
                            measure_root=measure_root,
                            task_ids=task_ids,
                        )
                    if candidate.git_commit_hash != candidate_ref.commit:
                        raise RuntimeError(
                            "measured candidate commit does not match the validated "
                            f"candidate: {candidate.git_commit_hash} != "
                            f"{candidate_ref.commit}"
                        )
                    # Judge inside the funnel so infra-sensitive verdicts can
                    # retry; the yielded Loss is the epoch's sole judgment.
                    loss = self.criterion(candidate=candidate, baseline=baseline)
                except MeasurementError as exc:
                    if retry:
                        self._observer.measurement_retrying(
                            f"measurement failed: {exc}"
                        )
                        continue
                    failed_run, decision = exc.result, None
                except UnmeasurableRun as exc:
                    if retry:
                        self._observer.measurement_retrying("unmeasurable run")
                        continue
                    failed_run, decision = candidate, exc.decision
                else:
                    # Zero before opening the window: otherwise an abandoned epoch
                    # satisfies _conclude's guards with the stale verdict.
                    self._harness.grad = None
                    self._harness.applied = None
                    yield loss
                    candidate = self._conclude(
                        baseline=baseline,
                        candidate_ref=candidate_ref,
                        candidate=candidate,
                        propose_root=propose_root,
                    )
                    return None, candidate
                if failed_run is not None:
                    failed_run = self._tracker.record_finalized_run(
                        failed_run,
                        kind="candidate",
                        parent_commit_hash=baseline_commit,
                        baseline_experiment_id=baseline.experiment_id,
                        decision=decision,
                    )
                    self._observer.experiment_finished(failed_run)
                self._observer.epoch_skipped("measurement_failure")
                return "measurement_failure", failed_run
        raise AssertionError("candidate measurement funnel exhausted")

    def _conclude(
        self,
        *,
        baseline: ExperimentResult,
        candidate_ref: Candidate,
        candidate: ExperimentResult,
        propose_root: Path,
    ) -> ExperimentResult:
        loss = self._harness.grad
        if loss is None:
            raise RuntimeError(
                "epoch not trained: loss.backward() never ran — the loop body "
                "must call loss.backward(), optimizer.step() each epoch"
            )
        if loss.candidate.git_commit_hash != candidate_ref.commit:
            raise RuntimeError(
                "recorded verdict does not match this epoch's candidate: "
                f"{loss.candidate.git_commit_hash} != {candidate_ref.commit}"
            )
        if self._harness.applied is None:
            raise RuntimeError(
                "epoch not applied: optimizer.step() never ran — call it after "
                "loss.backward()"
            )
        if candidate.git_commit_hash != candidate_ref.commit:
            raise RuntimeError(
                "measured candidate commit does not match the validated candidate: "
                f"{candidate.git_commit_hash} != {candidate_ref.commit}"
            )
        decision = self.criterion.decision(loss)
        kind: RunKind = "baseline" if decision.outcome == "promoted" else "candidate"
        candidate = self._tracker.record_finalized_run(
            candidate,
            kind=kind,
            parent_commit_hash=candidate_ref.base_commit,
            baseline_experiment_id=baseline.experiment_id,
            decision=decision,
        )
        self._observer.experiment_finished(candidate)
        self._observer.decision_finished(decision)
        try:
            self.estimator.diagnose(
                candidate,
                repo_root=propose_root,
                tracker=self._tracker,
                target=self.run_config.training_target,
                emit=self._observer.agent_progress,
            )
        finally:
            # Epoch boundary: neither the verdict nor the selection may leak forward.
            self._harness.grad = None
            self._harness.applied = None
        return candidate


def _tracked_repo_relative_path(path: str, *, repo: Repo) -> str:
    repo_root = Path(repo.working_tree_dir)
    absolute = Path(path).expanduser()
    if not absolute.is_absolute():
        absolute = repo_root / absolute
    relative = absolute.resolve().relative_to(repo_root.resolve()).as_posix()
    repo.git.ls_files("--error-unmatch", "--", relative)
    return relative
