from __future__ import annotations

import errno
import json
import locale
import os
import pty
import selectors
import shutil
import subprocess
import sys
import tty
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from src.adapters.chatgpt_codex import (
    CODEX_CREDENTIALS_EXPIRED_EXIT_CODE,
    CODEX_CREDENTIALS_EXPIRED_MESSAGE,
    ChatGptCodexCredentialsExpiredError,
)
from src.adapters.env import DEFAULT_HARBOR_CONFIG_PATH, HarborConfig
from src.control import repo as control_repo
from src.control.agent_backend import (
    AgentBackend,
    MissingThreadRollout,
    TurnTimeout,
    create_backend,
)
from src.control.gates import (
    SUPERVISOR_EDITABLE_PATHS,
    build_mechanism_novelty_rejection,
    validate_candidate_config_patch,
    validate_candidate_editable_paths,
    validate_learning_memo_update,
    validate_no_task_ids_in_workspace_diff,
)
from src.control.supervisor_state import (
    DEFAULT_SUPERVISOR_ROOT,
    SupervisorState,
    append_supervisor_event,
    repo_fingerprint,
)
from src.experiment.record import (
    ExperimentAbandoned,
    ExperimentRecord,
    ExperimentState,
    failed_experiment_git_ref,
    write_json_atomic,
)
from src.experiment.runner import ExperimentRunner
from src.harness.config import DEFAULT_HARNESS_CONFIG_PATH, HarnessConfig

DEFAULT_SUPERVISOR_WORKTREE_PARENT = DEFAULT_SUPERVISOR_ROOT
# The candidate may write harness_config.json, but only these proposal fields
# may differ from HEAD during prelaunch.
# Config is structurally validated; only behavior/test diffs are scanned as
# mechanism text for task ids and novelty.
SUPERVISOR_VISIBLE_TRACKED_PATHS = (
    "program.md",
    "pyproject.toml",
    "uv.lock",
    "src/__init__.py",
    "src/serialization.py",
    "src/adapters/llm_base.py",
    "src/experiment/trial.py",
    "src/harness/contracts.py",
    "src/trace.py",
    "src/metrics.py",
    "tests/conftest.py",
    *SUPERVISOR_EDITABLE_PATHS,
)
ABANDONED_EXPERIMENT_REASON = "abandoned after supervisor restart"


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    repo_root: Path
    harness_config: HarnessConfig
    experiments_root: Path
    experiment_state: ExperimentState
    active_baseline_record: ExperimentRecord | None
    current_candidate_record: ExperimentRecord | None


@dataclass(frozen=True, slots=True)
class PreparedCandidate:
    thread_id: str
    experiment_id: str
    changed_paths: tuple[str, ...]
    harness_config: HarnessConfig


def load_runtime_snapshot(*, repo_root: Path | None = None) -> RuntimeSnapshot:
    resolved_repo_root = (
        Path(__file__).resolve().parents[2]
        if repo_root is None
        else repo_root.resolve()
    )
    harness_config_path = DEFAULT_HARNESS_CONFIG_PATH
    if repo_root is not None:
        harness_config_path = resolved_repo_root / "config" / "harness_config.json"
    harbor_config_path = DEFAULT_HARBOR_CONFIG_PATH
    if repo_root is not None:
        harbor_config_path = resolved_repo_root / "config" / "harbor_config.toml"

    harness_config = HarnessConfig.model_validate_json(harness_config_path.read_text())
    harbor_config = HarborConfig.from_toml(harbor_config_path)
    experiment_state = ExperimentState.load(root=harbor_config.experiments_dir)
    active_baseline_record = (
        None
        if experiment_state.active_baseline_experiment_id is None
        else ExperimentRecord.load(
            experiment_state.active_baseline_experiment_id,
            root=harbor_config.experiments_dir,
        )
    )
    current_candidate_record = None
    if (
        experiment_state.current_experiment_id is not None
        and experiment_state.current_experiment_id
        != experiment_state.active_baseline_experiment_id
    ):
        current_candidate_record = ExperimentRecord.load(
            experiment_state.current_experiment_id,
            root=harbor_config.experiments_dir,
        )
    return RuntimeSnapshot(
        repo_root=resolved_repo_root,
        harness_config=harness_config,
        experiments_root=harbor_config.experiments_dir,
        experiment_state=experiment_state,
        active_baseline_record=active_baseline_record,
        current_candidate_record=current_candidate_record,
    )


def current_experiment_record(snapshot: RuntimeSnapshot) -> ExperimentRecord | None:
    current_experiment_id = snapshot.experiment_state.current_experiment_id
    if current_experiment_id is None:
        return None
    candidate = snapshot.current_candidate_record
    if candidate is not None:
        return candidate
    baseline = snapshot.active_baseline_record
    if baseline is not None and baseline.experiment_id == current_experiment_id:
        return baseline
    return ExperimentRecord.load(
        current_experiment_id,
        root=snapshot.experiments_root,
    )


def load_harness_config_for_repo(repo_root: Path) -> HarnessConfig:
    harness_config_path = repo_root.resolve() / "config" / "harness_config.json"
    return HarnessConfig.model_validate_json(harness_config_path.read_text())


def learning_memo_path(*, experiments_root: Path) -> Path:
    return experiments_root / "learning.md"


def read_learning_memo(*, experiments_root: Path) -> str:
    path = learning_memo_path(experiments_root=experiments_root)
    if not path.exists():
        return ""
    return path.read_text()


def write_learning_memo(*, experiments_root: Path, content: str) -> None:
    path = learning_memo_path(experiments_root=experiments_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _existing_artifact_paths(paths: tuple[str | None, ...]) -> tuple[str, ...]:
    return tuple(path for path in paths if path is not None and Path(path).exists())


def latest_evidence_task_artifact_paths(record: ExperimentRecord) -> tuple[str, ...]:
    evidence = record.evidence
    if evidence is None or not evidence.panel_outcomes:
        return ()
    outcomes = [
        outcome
        for panel_id in record.panel_order
        for outcome in evidence.panel_outcomes.get(panel_id, [])
    ]
    relevant_outcomes = [
        outcome
        for outcome in outcomes
        if outcome.outcome in {"new_solve", "regression"}
    ]
    if not relevant_outcomes:
        relevant_outcomes = [
            outcome
            for outcome in outcomes
            if outcome.candidate_solved is not True
            or (outcome.error is not None and outcome.error.strip())
        ]
    if not relevant_outcomes:
        return ()
    return _existing_artifact_paths(
        tuple(
            path
            for outcome in relevant_outcomes
            for path in (
                outcome.trial_dir,
                outcome.agent_steps_path,
                outcome.agent_exec_log_path,
                outcome.metrics_path,
                outcome.verifier_stdout_path,
            )
        )
    )


def _cleanup_orphaned_experiment_artifacts(
    *,
    experiments_root: Path,
    current_experiment_id: str | None,
) -> bool:
    cleaned = False
    for child in experiments_root.iterdir():
        if not child.is_dir():
            continue
        record_path = ExperimentRecord.path(child.name, root=experiments_root)
        if not record_path.exists():
            # We intentionally drop launch artifacts that were created before the
            # experiment became the persisted current candidate.
            shutil.rmtree(child)
            cleaned = True
            continue
        record = ExperimentRecord.load(child.name, root=experiments_root)
        if record.is_concluded() or record.experiment_id == current_experiment_id:
            continue
        shutil.rmtree(child)
        cleaned = True
    return cleaned


def _load_runtime(
    *,
    repo_root: Path,
) -> tuple[HarborConfig, str | None]:
    harbor_config_path = repo_root.resolve() / "config" / "harbor_config.toml"
    harness_config = load_harness_config_for_repo(repo_root)
    api_key: str | None = None
    if harness_config.llm_provider_config.provider == "openrouter":
        from src.adapters.open_router import load_openrouter_api_key

        api_key = load_openrouter_api_key(dotenv_path=repo_root.resolve() / ".env")
    return HarborConfig.from_toml(harbor_config_path), api_key


def build_prelaunch_prompt(
    *,
    workspace_root: Path,
    active_baseline_record: ExperimentRecord | None,
    latest_candidate_record: ExperimentRecord | None,
    evidence_artifact_paths: tuple[str, ...],
    feedback_note: str | None = None,
) -> str:
    root = workspace_root.resolve()
    records_root = root / "experiments"
    lines = [
        "Autonomous prelaunch phase.",
        "Follow program.md.",
        "Read these authoritative files now:",
        f"- {root / 'program.md'}",
        f"- {root / 'config' / 'harness_config.json'}",
        f"- {root / 'experiments' / 'learning.md'}",
    ]
    if active_baseline_record is not None:
        lines.append(
            f"- {ExperimentRecord.path(active_baseline_record.experiment_id, root=records_root)}"
        )
    if latest_candidate_record is not None:
        lines.append(
            f"- {ExperimentRecord.path(latest_candidate_record.experiment_id, root=records_root)}"
        )
    if evidence_artifact_paths:
        lines.append(
            "- candidate evidence artifact paths from `evidence.panel_outcomes`:"
        )
        lines.extend(f"- {path}" for path in evidence_artifact_paths)
    if feedback_note is not None:
        lines.extend(
            [
                "",
                "Supervisor feedback from the previous prelaunch turn:",
                feedback_note,
            ]
        )
    return "\n".join(lines)


def build_experiment_diagnosis_prompt(
    *,
    workspace_root: Path,
    experiment_record: ExperimentRecord,
    evidence_artifact_paths: tuple[str, ...],
    feedback_note: str | None = None,
) -> str:
    root = workspace_root.resolve()
    record_path = ExperimentRecord.path(
        experiment_record.experiment_id,
        root=root / "experiments",
    )
    lines = [
        "Autonomous post-run diagnosis phase.",
        "Follow program.md.",
        "Read these authoritative files now:",
        f"- {root / 'program.md'}",
        f"- {record_path}",
        f"- {root / 'experiments' / 'learning.md'}",
    ]
    if evidence_artifact_paths:
        lines.append(
            "- experiment evidence artifact paths from `evidence.panel_outcomes`:"
        )
        lines.extend(f"- {path}" for path in evidence_artifact_paths)
    if feedback_note is not None:
        lines.extend(
            [
                "",
                "Supervisor feedback from the previous diagnosis turn:",
                feedback_note,
            ]
        )
    return "\n".join(lines)


def default_sparse_workspace_root(*, repo_root: Path) -> Path:
    return (
        DEFAULT_SUPERVISOR_WORKTREE_PARENT / repo_fingerprint(repo_root) / "workspace"
    )


def _ensure_experiments_link(*, repo_root: Path, workspace_root: Path) -> None:
    experiments_root = repo_root / "experiments"
    experiments_root.mkdir(parents=True, exist_ok=True)
    link_path = workspace_root / "experiments"
    if link_path.is_symlink() and link_path.resolve() == experiments_root.resolve():
        return
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(experiments_root.resolve())


def ensure_sparse_workspace(
    *,
    repo_root: Path,
    workspace_root: Path,
) -> Path:
    if not workspace_root.exists():
        workspace_root.parent.mkdir(parents=True, exist_ok=True)
        control_repo.add_worktree(workspace_root, cwd=repo_root, force=True)
    elif not (workspace_root / ".git").exists():
        raise RuntimeError(f"sparse workspace is not a git worktree: {workspace_root}")

    control_repo.sparse_checkout_init_no_cone(cwd=workspace_root)
    control_repo.sparse_checkout_set(
        SUPERVISOR_VISIBLE_TRACKED_PATHS,
        cwd=workspace_root,
    )
    control_repo.sparse_checkout_reapply(cwd=workspace_root)
    _ensure_experiments_link(repo_root=repo_root, workspace_root=workspace_root)
    control_repo.ensure_info_exclude_entry("/experiments", cwd=workspace_root)
    return workspace_root


def sync_sparse_workspace_to_commit(
    *,
    workspace_root: Path,
    commit_hash: str,
) -> None:
    control_repo.hard_reset(commit_hash, cwd=workspace_root)
    control_repo.clean_untracked(cwd=workspace_root, exclude=("experiments",))
    control_repo.sparse_checkout_reapply(cwd=workspace_root)


def workspace_changed_paths(*, workspace_root: Path) -> tuple[str, ...]:
    changed_paths = tuple(
        path
        for path in control_repo.changed_paths(cwd=workspace_root)
        if path != "experiments" and not path.startswith("experiments/")
    )
    validate_candidate_editable_paths(changed_paths=changed_paths)
    return changed_paths


def commit_candidate(*, workspace_root: Path, experiment_id: str) -> str:
    return control_repo.commit_all(f"candidate {experiment_id}", cwd=workspace_root)


def _assign_candidate_experiment_id(
    *,
    workspace_root: Path,
    experiments_root: Path,
    harness_config: HarnessConfig,
) -> HarnessConfig:
    generated_at = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    experiment_id = _next_available_experiment_id(
        experiments_root=experiments_root,
        base_experiment_id=f"exp-{generated_at}",
    )
    config_path = workspace_root.resolve() / "config" / "harness_config.json"
    payload = json.loads(config_path.read_text())
    payload["experiment_id"] = experiment_id
    write_json_atomic(config_path, payload)
    return harness_config.model_copy(update={"experiment_id": experiment_id})


def _load_parent_baseline(
    *,
    record: ExperimentRecord,
    experiments_root: Path,
) -> ExperimentRecord | None:
    parent_id = record.parent_baseline_experiment_id
    if parent_id is None:
        return None
    parent_path = ExperimentRecord.path(parent_id, root=experiments_root)
    if not parent_path.exists():
        return None
    return ExperimentRecord.load(parent_id, root=experiments_root)


def _discard_interrupted_baseline(
    *,
    snapshot: RuntimeSnapshot,
    repo_root: Path,
    supervisor_root: Path = DEFAULT_SUPERVISOR_ROOT,
) -> ExperimentRecord | None:
    """Drop an active baseline that never finalized as a keep.

    A baseline run killed mid-flight leaves ``state.json`` pointing the active
    baseline at a partial, never-graded record. Trusting it would let the loop
    short-circuit the fresh measurement -- and later compare every candidate
    against ungraded evidence -- or make ``run_baseline_at_head`` abort on its
    "active baseline must be a concluded keep" guard. We finalize the stale
    record and fall back to its parent keep (or nothing, seeding fresh), then
    return the baseline the caller should reason about next.
    """
    baseline = snapshot.active_baseline_record
    if baseline is None or (baseline.status == "keep" and baseline.is_concluded()):
        return baseline
    parent = _load_parent_baseline(
        record=baseline,
        experiments_root=snapshot.experiments_root,
    )
    if not baseline.is_concluded():
        baseline.finalize_crash(
            exc=ExperimentAbandoned(ABANDONED_EXPERIMENT_REASON),
            baseline=parent,
            root=snapshot.experiments_root,
        )
    state = snapshot.experiment_state
    parent_id = baseline.parent_baseline_experiment_id
    state.set_active_baseline(experiment_id=parent_id, record=parent)
    if state.current_experiment_id == baseline.experiment_id:
        state.current_experiment_id = parent_id
    state.updated_at = datetime.now(timezone.utc).isoformat()
    state.save(root=snapshot.experiments_root)
    SupervisorState.clear(repo_root=repo_root, root=supervisor_root)
    return parent


def abandon_unfinished_candidate(
    *,
    snapshot: RuntimeSnapshot,
    repo_root: Path,
    supervisor_root: Path = DEFAULT_SUPERVISOR_ROOT,
) -> ExperimentRecord:
    # State-bookkeeping only: mark the candidate as crashed in state.json and
    # preserve its commit under a failed-experiment ref. HEAD is owned by
    # _ensure_baseline_at_head; the user is expected to undo any partial
    # commit they want discarded, otherwise it is treated as deliberate.
    record = snapshot.current_candidate_record
    if record is None or record.is_concluded():
        raise RuntimeError(
            "unfinished candidate cleanup requires a non-concluded current candidate"
        )

    record.finalize_crash(
        exc=ExperimentAbandoned(ABANDONED_EXPERIMENT_REASON),
        baseline=snapshot.active_baseline_record,
        root=snapshot.experiments_root,
    )

    state = snapshot.experiment_state
    state.current_experiment_id = record.experiment_id
    state.updated_at = record.finished_at
    state.save(root=snapshot.experiments_root)

    control_repo.update_ref(
        failed_experiment_git_ref(record.experiment_id),
        record.git_commit_hash,
        cwd=repo_root,
    )
    SupervisorState.clear(repo_root=repo_root, root=supervisor_root)
    return record


def _postrun_original_payload(
    *,
    saved_state: SupervisorState | None,
    experiment_record: ExperimentRecord,
    experiments_root: Path,
) -> dict[str, object]:
    if (
        saved_state is not None
        and saved_state.phase == "postrun"
        and saved_state.postrun_original_payload is not None
    ):
        return saved_state.postrun_original_payload
    return json.loads(
        ExperimentRecord.path(
            experiment_record.experiment_id, root=experiments_root
        ).read_text()
    )


def _postrun_original_learning(
    *,
    saved_state: SupervisorState | None,
    experiments_root: Path,
) -> str:
    if (
        saved_state is not None
        and saved_state.phase == "postrun"
        and saved_state.postrun_original_learning is not None
    ):
        return saved_state.postrun_original_learning
    return read_learning_memo(experiments_root=experiments_root)


def _recover_interrupted_postrun_state(
    *,
    saved_state: SupervisorState | None,
    experiment_record: ExperimentRecord | None,
    experiments_root: Path,
    repo_root: Path,
    supervisor_root: Path = DEFAULT_SUPERVISOR_ROOT,
) -> tuple[SupervisorState | None, ExperimentRecord | None]:
    if (
        saved_state is None
        or saved_state.phase != "postrun"
        or saved_state.postrun_original_payload is None
        or experiment_record is None
        or not experiment_record.is_concluded()
    ):
        return saved_state, experiment_record

    record_path = ExperimentRecord.path(
        experiment_record.experiment_id,
        root=experiments_root,
    )
    current_payload = json.loads(record_path.read_text())
    original_payload = saved_state.postrun_original_payload
    if current_payload == original_payload:
        if saved_state.postrun_original_learning is not None:
            current_learning = read_learning_memo(experiments_root=experiments_root)
            try:
                validate_learning_memo_update(
                    before_payload=original_payload,
                    after_payload=current_payload,
                    before_learning=saved_state.postrun_original_learning,
                    after_learning=current_learning,
                )
            except RuntimeError:
                return saved_state, experiment_record
            completed_state = SupervisorState.prelaunch(
                thread_id=saved_state.thread_id,
                updated_at=saved_state.updated_at,
                postrun_completed_experiment_id=experiment_record.experiment_id,
            )
            completed_state.save(repo_root=repo_root, root=supervisor_root)
            return completed_state, experiment_record
        return saved_state, experiment_record

    try:
        validate_learning_memo_update(
            before_payload=original_payload,
            after_payload=current_payload,
            before_learning=saved_state.postrun_original_learning or "",
            after_learning=read_learning_memo(experiments_root=experiments_root),
        )
    except RuntimeError:
        # Postrun resumes by replaying from the last known-good payload instead of
        # trying to reason about a partially written diagnosis edit.
        write_json_atomic(record_path, original_payload)
        if saved_state.postrun_original_learning is not None:
            write_learning_memo(
                experiments_root=experiments_root,
                content=saved_state.postrun_original_learning,
            )
        return saved_state, ExperimentRecord.load(
            experiment_record.experiment_id,
            root=experiments_root,
        )

    SupervisorState.clear(repo_root=repo_root, root=supervisor_root)
    return None, ExperimentRecord.load(
        experiment_record.experiment_id,
        root=experiments_root,
    )


def complete_postrun_diagnosis(
    *,
    workspace_root: Path,
    repo_root: Path,
    experiments_root: Path,
    experiment_record: ExperimentRecord,
    thread_id: str | None,
    original_payload: dict[str, object],
    original_learning: str,
    sync_before: bool = True,
    supervisor_root: Path = DEFAULT_SUPERVISOR_ROOT,
    backend: AgentBackend | None = None,
) -> str:
    SupervisorState.postrun(
        thread_id=thread_id,
        updated_at=datetime.now(timezone.utc).isoformat(),
        postrun_original_payload=original_payload,
        postrun_original_learning=original_learning,
    ).save(repo_root=repo_root, root=supervisor_root)
    if sync_before:
        sync_sparse_workspace_to_commit(
            workspace_root=workspace_root,
            commit_hash=control_repo.get_head_commit(cwd=repo_root),
        )
    diagnosis_thread_id = run_postrun_diagnosis_phase(
        workspace_root=workspace_root,
        repo_root=repo_root,
        experiments_root=experiments_root,
        experiment_record=experiment_record,
        thread_id=thread_id,
        original_payload=original_payload,
        original_learning=original_learning,
        backend=backend,
        supervisor_root=supervisor_root,
    )
    sync_sparse_workspace_to_commit(
        workspace_root=workspace_root,
        commit_hash=control_repo.get_head_commit(cwd=repo_root),
    )
    if experiment_record.status in {"discard", "crash"}:
        SupervisorState.prelaunch(
            thread_id=diagnosis_thread_id,
            updated_at=datetime.now(timezone.utc).isoformat(),
            postrun_completed_experiment_id=experiment_record.experiment_id,
        ).save(repo_root=repo_root, root=supervisor_root)
    else:
        SupervisorState.clear(repo_root=repo_root, root=supervisor_root)
    append_supervisor_event(
        repo_root=repo_root,
        root=supervisor_root,
        event="postrun_diagnosis_completed",
        experiment_id=experiment_record.experiment_id,
        status=experiment_record.status,
        thread_id=diagnosis_thread_id,
    )
    return diagnosis_thread_id


def recover_interrupted_launch(
    *,
    saved_state: SupervisorState | None,
    snapshot: RuntimeSnapshot,
    repo_root: Path,
    workspace_root: Path,
    supervisor_root: Path = DEFAULT_SUPERVISOR_ROOT,
) -> bool:
    if saved_state is None or saved_state.phase != "launch":
        return False
    experiment_id = saved_state.launch_experiment_id
    baseline_commit = saved_state.launch_baseline_commit
    if experiment_id is None or baseline_commit is None:
        SupervisorState.clear(repo_root=repo_root, root=supervisor_root)
        return True

    record_path = ExperimentRecord.path(experiment_id, root=snapshot.experiments_root)
    if record_path.exists():
        record = ExperimentRecord.load(experiment_id, root=snapshot.experiments_root)
        if not record.is_concluded():
            record.finalize_crash(
                exc=ExperimentAbandoned(ABANDONED_EXPERIMENT_REASON),
                baseline=_load_parent_baseline(
                    record=record,
                    experiments_root=snapshot.experiments_root,
                ),
                root=snapshot.experiments_root,
            )
        state = snapshot.experiment_state
        if record.status == "keep":
            state.set_active_baseline(experiment_id=record.experiment_id, record=record)
        state.current_experiment_id = record.experiment_id
        state.updated_at = record.finished_at
        state.save(root=snapshot.experiments_root)
        SupervisorState.postrun(
            thread_id=saved_state.thread_id,
            updated_at=datetime.now(timezone.utc).isoformat(),
            postrun_original_payload=json.loads(record_path.read_text()),
            postrun_original_learning=read_learning_memo(
                experiments_root=snapshot.experiments_root,
            ),
        ).save(repo_root=repo_root, root=supervisor_root)
        return True

    head_commit = control_repo.get_head_commit(cwd=repo_root)
    if head_commit != baseline_commit:
        control_repo.update_ref(
            failed_experiment_git_ref(experiment_id),
            head_commit,
            cwd=repo_root,
        )
        control_repo.hard_reset(baseline_commit, cwd=repo_root)
        sync_sparse_workspace_to_commit(
            workspace_root=workspace_root,
            commit_hash=baseline_commit,
        )
    SupervisorState.prelaunch(
        thread_id=saved_state.thread_id,
        updated_at=datetime.now(timezone.utc).isoformat(),
        postrun_completed_experiment_id=None,
    ).save(repo_root=repo_root, root=supervisor_root)
    append_supervisor_event(
        repo_root=repo_root,
        root=supervisor_root,
        event="interrupted_launch_recovered",
        experiment_id=experiment_id,
    )
    return True


def launch_tracked_experiment(
    *,
    repo_root: Path,
    experiment_id: str,
    experiments_root: Path,
) -> ExperimentRecord:
    completed = _run_with_live_tty_output(["uv", "run", "exp"], cwd=repo_root)
    if completed.returncode == CODEX_CREDENTIALS_EXPIRED_EXIT_CODE:
        # Dead credentials need a human (`codex login`); halt rather than
        # finalizing a crash record the loop would treat like a discard and
        # advance past, re-launching straight into the same auth wall.
        raise ChatGptCodexCredentialsExpiredError(CODEX_CREDENTIALS_EXPIRED_MESSAGE)
    if completed.returncode != 0:
        raise RuntimeError(
            "uv run exp failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    record_path = ExperimentRecord.path(experiment_id, root=experiments_root)
    if not record_path.exists():
        raise RuntimeError(
            f"tracked experiment record missing after launch: {record_path}"
        )
    return ExperimentRecord.load(experiment_id, root=experiments_root)


def _run_with_live_tty_output(
    args: list[str],
    *,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    encoding = locale.getencoding()
    stdout_master, stdout_slave = pty.openpty()
    stderr_master, stderr_slave = pty.openpty()
    tty.setraw(stdout_slave)
    tty.setraw(stderr_slave)
    process = subprocess.Popen(
        args,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=stdout_slave,
        stderr=stderr_slave,
        close_fds=True,
    )
    os.close(stdout_slave)
    os.close(stderr_slave)

    output_by_fd = {
        stdout_master: (sys.stdout, []),
        stderr_master: (sys.stderr, []),
    }
    selector = selectors.DefaultSelector()
    selector.register(stdout_master, selectors.EVENT_READ)
    selector.register(stderr_master, selectors.EVENT_READ)
    try:
        while selector.get_map():
            for key, _events in selector.select():
                try:
                    chunk = os.read(key.fd, 8192)
                except OSError as exc:
                    if exc.errno != errno.EIO:
                        raise
                    chunk = b""
                if not chunk:
                    selector.unregister(key.fd)
                    os.close(key.fd)
                    continue
                stream, chunks = output_by_fd[key.fd]
                chunks.append(chunk)
                stream.write(chunk.decode(encoding, errors="replace"))
                stream.flush()
        return subprocess.CompletedProcess(
            args=args,
            returncode=process.wait(),
            stdout=b"".join(output_by_fd[stdout_master][1]).decode(
                encoding,
                errors="replace",
            ),
            stderr=b"".join(output_by_fd[stderr_master][1]).decode(
                encoding,
                errors="replace",
            ),
        )
    finally:
        selector.close()
        for fd in (stdout_master, stderr_master):
            try:
                os.close(fd)
            except OSError:
                pass
        if process.poll() is None:
            process.kill()
            process.wait()


def promote_workspace_commit_to_repo(
    *,
    repo_root: Path,
    commit_hash: str,
) -> None:
    control_repo.require_clean_worktree(cwd=repo_root)
    control_repo.hard_reset(commit_hash, cwd=repo_root)


PhaseRunner = Callable[[str | None, str | None], tuple[str | None, str | None]]


def _next_available_experiment_id(
    *,
    experiments_root: Path,
    base_experiment_id: str,
) -> str:
    if not ExperimentRecord.path(base_experiment_id, root=experiments_root).exists():
        return base_experiment_id
    retry_index = 1
    while True:
        candidate = f"{base_experiment_id}_retry_{retry_index}"
        if not ExperimentRecord.path(candidate, root=experiments_root).exists():
            return candidate
        retry_index += 1


def _run_phase_until_valid(
    *,
    initial_thread_id: str | None,
    run_once: PhaseRunner,
    on_turn_complete: Callable[[str], None] | None = None,
    on_feedback: Callable[[str | None, str], None] | None = None,
) -> str:
    # `on_feedback(thread_id, note)` fires once per turn that closes with a
    # non-None validation error (gate trip) or a MissingThreadRollout. Without
    # it the inner retry loop is invisible to events.jsonl: outer-loop audit
    # would see candidate_prepared events and miss the per-gate friction the
    # codex agent paid to clear them.
    feedback_note: str | None = None
    current_thread_id = initial_thread_id
    while True:
        try:
            next_thread_id, validation_error = run_once(
                current_thread_id,
                feedback_note,
            )
        except MissingThreadRollout as exc:
            if current_thread_id is None:
                raise
            note = str(exc)
            if on_feedback is not None:
                on_feedback(current_thread_id, note)
            current_thread_id = None
            feedback_note = note
            continue
        current_thread_id = next_thread_id
        if current_thread_id is None:
            raise RuntimeError("phase turn completed without a thread id")
        if on_turn_complete is not None:
            on_turn_complete(current_thread_id)
        if validation_error is None:
            return current_thread_id
        if on_feedback is not None:
            on_feedback(current_thread_id, validation_error)
        feedback_note = validation_error


def prepare_candidate(
    *,
    workspace_root: Path,
    experiments_root: Path,
    thread_id: str,
) -> PreparedCandidate:
    changed_paths = workspace_changed_paths(
        workspace_root=workspace_root,
    )
    if not changed_paths:
        raise RuntimeError(
            "candidate is not ready to launch: no tracked changes "
            "(program.md Prelaunch requires a focused candidate patch)"
        )
    harness_config = load_harness_config_for_repo(workspace_root)
    validate_candidate_config_patch(
        workspace_root=workspace_root,
        harness_config=harness_config,
    )
    if "src/harness/core.py" not in changed_paths:
        raise RuntimeError(
            "candidate is not ready to launch: no behavioral harness change "
            "(program.md Source-of-truth boundary: src/harness/core.py is the "
            "main behavior surface)"
        )
    harness_config = _assign_candidate_experiment_id(
        workspace_root=workspace_root,
        experiments_root=experiments_root,
        harness_config=harness_config,
    )
    changed_paths = tuple(sorted({*changed_paths, "config/harness_config.json"}))
    return PreparedCandidate(
        thread_id=thread_id,
        experiment_id=harness_config.experiment_id,
        changed_paths=changed_paths,
        harness_config=harness_config,
    )


def run_prelaunch_phase(
    *,
    workspace_root: Path,
    repo_root: Path,
    experiments_root: Path,
    active_baseline_record: ExperimentRecord | None,
    latest_candidate_record: ExperimentRecord | None,
    evidence_artifact_paths: tuple[str, ...],
    thread_id: str | None,
    backend: AgentBackend | None = None,
    on_turn_complete: Callable[[str], None] | None = None,
    supervisor_root: Path = DEFAULT_SUPERVISOR_ROOT,
) -> PreparedCandidate:
    if backend is None:
        backend = create_backend("codex")
    prepared_candidate: PreparedCandidate | None = None

    def run_once(
        current_thread_id: str | None,
        feedback_note: str | None,
    ) -> tuple[str, str | None]:
        nonlocal prepared_candidate
        try:
            turn_result = backend.run_turn(
                prompt=build_prelaunch_prompt(
                    workspace_root=workspace_root,
                    active_baseline_record=active_baseline_record,
                    latest_candidate_record=latest_candidate_record,
                    evidence_artifact_paths=evidence_artifact_paths,
                    feedback_note=feedback_note,
                ),
                repo_root=workspace_root,
                thread_id=current_thread_id,
            )
        except TurnTimeout as exc:
            tid = exc.thread_id or current_thread_id
            if tid is None:
                raise
            return tid, str(exc)
        try:
            prepared = prepare_candidate(
                workspace_root=workspace_root,
                experiments_root=experiments_root,
                thread_id=turn_result.thread_id,
            )
        except Exception as exc:
            return turn_result.thread_id, str(exc)
        if (
            repo_root.resolve() != workspace_root.resolve()
            and (repo_root / ".git").exists()
        ):
            live_repo_dirty_paths = control_repo.changed_paths(cwd=repo_root)
            if live_repo_dirty_paths:
                control_repo.hard_reset(
                    control_repo.get_head_commit(cwd=repo_root),
                    cwd=repo_root,
                )
                return (
                    turn_result.thread_id,
                    "candidate edited the live repo instead of the sparse workspace; "
                    f"edit only within {workspace_root} and do not modify {repo_root} directly",
                )
        # Audit gates: collected together so the agent sees every fixable
        # issue in one feedback turn instead of one issue per agent invocation.
        audit_notes: list[str] = []
        try:
            validate_no_task_ids_in_workspace_diff(
                workspace_root=workspace_root,
                task_ids=tuple(
                    sorted(prepared.harness_config.promotion_panel.task_names)
                ),
            )
        except RuntimeError as exc:
            audit_notes.append(str(exc))
        novelty_rejection = build_mechanism_novelty_rejection(
            workspace_root=workspace_root,
            experiments_root=experiments_root,
            changed_paths=prepared.changed_paths,
        )
        if novelty_rejection is not None:
            audit_notes.append(novelty_rejection)
        if audit_notes:
            return turn_result.thread_id, "\n\n".join(audit_notes)
        prepared_candidate = prepared
        return turn_result.thread_id, None

    def emit_feedback(current_thread_id: str | None, note: str) -> None:
        append_supervisor_event(
            repo_root=repo_root,
            root=supervisor_root,
            event="prelaunch_feedback",
            thread_id=current_thread_id,
            note=note,
        )

    _run_phase_until_valid(
        initial_thread_id=thread_id,
        run_once=run_once,
        on_turn_complete=on_turn_complete,
        on_feedback=emit_feedback,
    )
    if prepared_candidate is None:
        raise RuntimeError("prelaunch phase completed without a prepared candidate")
    return prepared_candidate


def run_postrun_diagnosis_phase(
    *,
    workspace_root: Path,
    repo_root: Path,
    experiments_root: Path,
    experiment_record: ExperimentRecord,
    thread_id: str | None,
    original_payload: dict[str, object] | None = None,
    original_learning: str | None = None,
    backend: AgentBackend | None = None,
    supervisor_root: Path = DEFAULT_SUPERVISOR_ROOT,
) -> str:
    if backend is None:
        backend = create_backend("codex")
    record_path = ExperimentRecord.path(
        experiment_record.experiment_id,
        root=experiments_root,
    )
    if original_payload is None:
        original_payload = json.loads(record_path.read_text())
    if original_learning is None:
        original_learning = read_learning_memo(experiments_root=experiments_root)
    evidence_artifact_paths = latest_evidence_task_artifact_paths(experiment_record)

    def run_once(
        current_thread_id: str | None,
        feedback_note: str | None,
    ) -> tuple[str, str | None]:
        try:
            turn_result = backend.run_turn(
                prompt=build_experiment_diagnosis_prompt(
                    workspace_root=workspace_root,
                    experiment_record=experiment_record,
                    evidence_artifact_paths=evidence_artifact_paths,
                    feedback_note=feedback_note,
                ),
                repo_root=workspace_root,
                thread_id=current_thread_id,
            )
        except TurnTimeout as exc:
            tid = exc.thread_id or current_thread_id
            if tid is None:
                raise
            return tid, str(exc)
        after_payload = json.loads(record_path.read_text())
        after_learning = read_learning_memo(experiments_root=experiments_root)
        try:
            validate_learning_memo_update(
                before_payload=original_payload,
                after_payload=after_payload,
                before_learning=original_learning,
                after_learning=after_learning,
            )
            return turn_result.thread_id, None
        except RuntimeError as exc:
            write_json_atomic(record_path, original_payload)
            write_learning_memo(
                experiments_root=experiments_root,
                content=original_learning,
            )
            return turn_result.thread_id, str(exc)

    def emit_feedback(current_thread_id: str | None, note: str) -> None:
        append_supervisor_event(
            repo_root=repo_root,
            root=supervisor_root,
            event="postrun_feedback",
            thread_id=current_thread_id,
            experiment_id=experiment_record.experiment_id,
            note=note,
        )

    return _run_phase_until_valid(
        initial_thread_id=thread_id,
        run_once=run_once,
        on_feedback=emit_feedback,
    )


def run_supervisor_loop(
    *,
    repo_root: Path | None = None,
    supervisor_root: Path = DEFAULT_SUPERVISOR_ROOT,
    backend: AgentBackend | None = None,
) -> None:
    if backend is None:
        backend = create_backend("codex")
    resolved_repo_root = (
        Path(__file__).resolve().parents[2]
        if repo_root is None
        else repo_root.resolve()
    )

    while True:
        try:
            append_supervisor_event(
                repo_root=resolved_repo_root,
                root=supervisor_root,
                event="loop_iteration_started",
            )
            snapshot = load_runtime_snapshot(repo_root=resolved_repo_root)
            workspace_root = ensure_sparse_workspace(
                repo_root=resolved_repo_root,
                workspace_root=default_sparse_workspace_root(
                    repo_root=resolved_repo_root
                ),
            )
            saved_state = SupervisorState.maybe_load(
                repo_root=resolved_repo_root,
                root=supervisor_root,
            )
            if recover_interrupted_launch(
                saved_state=saved_state,
                snapshot=snapshot,
                repo_root=resolved_repo_root,
                workspace_root=workspace_root,
                supervisor_root=supervisor_root,
            ):
                continue
            if _cleanup_orphaned_experiment_artifacts(
                experiments_root=snapshot.experiments_root,
                current_experiment_id=snapshot.experiment_state.current_experiment_id,
            ):
                append_supervisor_event(
                    repo_root=resolved_repo_root,
                    root=supervisor_root,
                    event="orphan_experiment_artifacts_cleaned",
                )
                continue
            # Reconcile state.json before touching HEAD: an unfinished candidate
            # record must be marked concluded so baseline refresh can proceed if
            # the user advanced HEAD past the active baseline.
            if (
                snapshot.current_candidate_record is not None
                and not snapshot.current_candidate_record.is_concluded()
            ):
                abandon_unfinished_candidate(
                    snapshot=snapshot,
                    repo_root=resolved_repo_root,
                    supervisor_root=supervisor_root,
                )
                continue
            if _ensure_baseline_at_head(
                snapshot=snapshot,
                repo_root=resolved_repo_root,
                workspace_root=workspace_root,
                supervisor_root=supervisor_root,
            ):
                continue
            current_record = current_experiment_record(snapshot)
            saved_state, current_record = _recover_interrupted_postrun_state(
                saved_state=saved_state,
                experiment_record=current_record,
                experiments_root=snapshot.experiments_root,
                repo_root=resolved_repo_root,
                supervisor_root=supervisor_root,
            )
            if (
                current_record is not None
                and current_record.is_concluded()
                and (
                    (saved_state is not None and saved_state.phase == "postrun")
                    or snapshot.current_candidate_record is not None
                )
                and not (
                    saved_state is not None
                    and saved_state.phase == "prelaunch"
                    and saved_state.postrun_completed_experiment_id
                    == current_record.experiment_id
                )
            ):
                complete_postrun_diagnosis(
                    workspace_root=workspace_root,
                    repo_root=resolved_repo_root,
                    experiments_root=snapshot.experiments_root,
                    experiment_record=current_record,
                    thread_id=(
                        saved_state.thread_id
                        if saved_state is not None and saved_state.phase == "postrun"
                        else None
                    ),
                    original_payload=_postrun_original_payload(
                        saved_state=saved_state,
                        experiment_record=current_record,
                        experiments_root=snapshot.experiments_root,
                    ),
                    original_learning=_postrun_original_learning(
                        saved_state=saved_state,
                        experiments_root=snapshot.experiments_root,
                    ),
                    sync_before=True,
                    supervisor_root=supervisor_root,
                    backend=backend,
                )
                continue
            sync_sparse_workspace_to_commit(
                workspace_root=workspace_root,
                commit_hash=control_repo.get_head_commit(cwd=resolved_repo_root),
            )
            prelaunch_thread_id = (
                saved_state.thread_id
                if saved_state is not None and saved_state.phase == "prelaunch"
                else None
            )
            latest_failed_candidate_record = (
                snapshot.current_candidate_record
                if snapshot.current_candidate_record is not None
                and snapshot.current_candidate_record.status in {"discard", "crash"}
                else None
            )
            evidence_artifact_paths = (
                ()
                if latest_failed_candidate_record is None
                else latest_evidence_task_artifact_paths(latest_failed_candidate_record)
            )
            # Capture the postrun_completed marker BEFORE the prelaunch turn
            # callback can overwrite saved state. Without this, every
            # on_turn_complete save during the next prelaunch wiped the
            # `postrun_completed_experiment_id` field — and if the prelaunch
            # then aborted (e.g. codex raised, the supervisor crashed and
            # restarted) before producing a new candidate, the next loop
            # iteration saw saved_state.postrun_completed_experiment_id=None,
            # the postrun-already-completed guard above evaluated False, and
            # the loop re-fired postrun on the same concluded candidate. exp-v4-0506-007
            # logged 69 successive postrun_diagnosis_completed events this
            # way before the next prelaunch eventually stuck.
            prior_postrun_completed = (
                saved_state.postrun_completed_experiment_id
                if saved_state is not None
                else None
            )
            prepared_candidate = run_prelaunch_phase(
                workspace_root=workspace_root,
                repo_root=resolved_repo_root,
                experiments_root=snapshot.experiments_root,
                active_baseline_record=snapshot.active_baseline_record,
                latest_candidate_record=latest_failed_candidate_record,
                evidence_artifact_paths=evidence_artifact_paths,
                thread_id=prelaunch_thread_id,
                backend=backend,
                on_turn_complete=lambda current_thread_id: SupervisorState.prelaunch(
                    thread_id=current_thread_id,
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    postrun_completed_experiment_id=prior_postrun_completed,
                ).save(repo_root=resolved_repo_root, root=supervisor_root),
                supervisor_root=supervisor_root,
            )
            append_supervisor_event(
                repo_root=resolved_repo_root,
                root=supervisor_root,
                event="candidate_prepared",
                thread_id=prepared_candidate.thread_id,
                experiment_id=prepared_candidate.experiment_id,
                changed_paths=list(prepared_candidate.changed_paths),
            )

            baseline_commit = control_repo.get_head_commit(cwd=resolved_repo_root)
            candidate_commit = commit_candidate(
                workspace_root=workspace_root,
                experiment_id=prepared_candidate.experiment_id,
            )
            SupervisorState.launch(
                thread_id=prepared_candidate.thread_id,
                updated_at=datetime.now(timezone.utc).isoformat(),
                launch_experiment_id=prepared_candidate.experiment_id,
                launch_baseline_commit=baseline_commit,
            ).save(repo_root=resolved_repo_root, root=supervisor_root)
            promote_workspace_commit_to_repo(
                repo_root=resolved_repo_root,
                commit_hash=candidate_commit,
            )
            record = launch_tracked_experiment(
                repo_root=resolved_repo_root,
                experiment_id=prepared_candidate.experiment_id,
                experiments_root=snapshot.experiments_root,
            )
            append_supervisor_event(
                repo_root=resolved_repo_root,
                root=supervisor_root,
                event="experiment_completed",
                experiment_id=record.experiment_id,
                status=record.status,
                decision_reason=record.decision_reason,
                git_commit_hash=record.git_commit_hash,
            )
            if record.status in {"discard", "crash"}:
                control_repo.hard_reset(
                    baseline_commit,
                    cwd=resolved_repo_root,
                )
                sync_sparse_workspace_to_commit(
                    workspace_root=workspace_root,
                    commit_hash=baseline_commit,
                )
            else:
                sync_sparse_workspace_to_commit(
                    workspace_root=workspace_root,
                    commit_hash=record.git_commit_hash,
                )
            complete_postrun_diagnosis(
                workspace_root=workspace_root,
                repo_root=resolved_repo_root,
                experiments_root=snapshot.experiments_root,
                experiment_record=record,
                thread_id=prepared_candidate.thread_id,
                original_payload=record.model_dump(mode="json"),
                original_learning=read_learning_memo(
                    experiments_root=snapshot.experiments_root,
                ),
                sync_before=False,
                supervisor_root=supervisor_root,
                backend=backend,
            )
            continue
        except ChatGptCodexCredentialsExpiredError as exc:
            # Terminal auth failure: record why the loop stopped, then let it
            # propagate so the process exits for operator intervention rather
            # than spinning through doomed cycles.
            append_supervisor_event(
                repo_root=resolved_repo_root,
                root=supervisor_root,
                event="loop_halted_credentials_expired",
                error=str(exc),
            )
            raise
        except Exception as exc:
            append_supervisor_event(
                repo_root=resolved_repo_root,
                root=supervisor_root,
                event="loop_iteration_failed",
                error=str(exc),
            )
            raise


def _ensure_baseline_at_head(
    *,
    snapshot: RuntimeSnapshot,
    repo_root: Path,
    workspace_root: Path,
    supervisor_root: Path = DEFAULT_SUPERVISOR_ROOT,
) -> bool:
    # Contract: a clean worktree is a precondition for the supervisor loop.
    # Any uncommitted change is treated as user-in-progress work and aborts.
    # Committed advances past the active baseline trigger a full baseline
    # measurement at HEAD instead of synthesizing evidence from history.
    dirty_paths: tuple[str, ...] = ()
    if (repo_root / ".git").exists():
        dirty_paths = control_repo.changed_paths(cwd=repo_root)
    if dirty_paths:
        raise RuntimeError(
            "supervisor loop requires a clean worktree; uncommitted changes in: "
            + ", ".join(dirty_paths)
        )
    head_commit = control_repo.get_head_commit(cwd=repo_root)
    harness_config = snapshot.harness_config
    # An interrupted baseline run can leave the active baseline pointing at a
    # record that never finalized as a keep. Discard it before deciding whether
    # to short-circuit so we re-measure from the last valid keep instead of
    # trusting partial evidence.
    baseline = _discard_interrupted_baseline(
        snapshot=snapshot,
        repo_root=repo_root,
        supervisor_root=supervisor_root,
    )
    if baseline is not None and head_commit == baseline.git_commit_hash:
        if _baseline_matches_harness_config(
            baseline=baseline,
            harness_config=harness_config,
        ):
            return False
    harbor_config, api_key = _load_runtime(repo_root=repo_root)
    decision_reason: Literal["baseline seed", "baseline rerun"] = (
        "baseline seed" if baseline is None else "baseline rerun"
    )
    baseline_started_at = datetime.now(timezone.utc).isoformat()
    baseline_timestamp = datetime.fromisoformat(baseline_started_at).strftime(
        "%Y%m%d-%H%M%S"
    )
    baseline_experiment_id = f"baseline-{baseline_timestamp}"
    experiment_json_path = str(
        ExperimentRecord.path(
            baseline_experiment_id,
            root=harbor_config.experiments_dir,
        )
    )
    append_supervisor_event(
        repo_root=repo_root,
        root=supervisor_root,
        event="baseline_run_started",
        experiment_id=baseline_experiment_id,
        baseline_experiment_id=None if baseline is None else baseline.experiment_id,
        reason=decision_reason,
        from_commit=None if baseline is None else baseline.git_commit_hash,
        to_commit=head_commit,
        experiment_json_path=experiment_json_path,
    )
    new_baseline = ExperimentRunner.run_baseline_at_head(
        harness_config=harness_config,
        harbor_config=harbor_config,
        api_key=api_key,
        decision_reason=decision_reason,
        experiment_id=baseline_experiment_id,
        started_at=baseline_started_at,
        repo_root=repo_root,
    )
    if new_baseline.status != "keep":
        raise RuntimeError(
            "baseline run failed: "
            f"{new_baseline.experiment_id} ended with status {new_baseline.status}"
        )
    sync_sparse_workspace_to_commit(
        workspace_root=workspace_root,
        commit_hash=new_baseline.git_commit_hash,
    )
    SupervisorState.clear(repo_root=repo_root, root=supervisor_root)
    append_supervisor_event(
        repo_root=repo_root,
        root=supervisor_root,
        event="baseline_run_completed",
        baseline_experiment_id=new_baseline.experiment_id,
        commit=new_baseline.git_commit_hash,
    )
    return True


def _baseline_matches_harness_config(
    *,
    baseline: ExperimentRecord,
    harness_config: HarnessConfig,
) -> bool:
    if baseline.panel_order != [panel.id for panel in harness_config.panels]:
        return False
    return all(
        panel.id in baseline.panels
        and set(baseline.panels[panel.id].task_ids) == set(panel.task_names)
        for panel in harness_config.panels
    )
