"""Agent-facing candidate estimation tasks for the autonomous loop."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from src.config import TrainingTargetConfig
from src.llm.backend import AgentBackend
from src.measurement import PreflightError
from src.rollout.records import ExperimentResult
from src.rollout.store import RunStore

LEARNING_MEMO_MAX_LINES = 150
_MAX_DIAGNOSIS_ATTEMPTS = 2

# (stage, line) progress sink; the trainer routes it into its presentation.
AgentProgress = Callable[[str, str], None]


def _print_agent_progress(stage: str, line: str) -> None:
    print(f"{stage:<9} · {line}")


class Estimator(ABC):
    @abstractmethod
    def propose(
        self,
        *,
        repo_root: Path,
        tracker: RunStore,
        target: TrainingTargetConfig,
        emit: AgentProgress,
    ) -> None: ...

    @abstractmethod
    def diagnose(
        self,
        result: ExperimentResult,
        *,
        repo_root: Path,
        tracker: RunStore,
        target: TrainingTargetConfig,
        emit: AgentProgress,
    ) -> None: ...


class AgenticEstimator(Estimator):
    def __init__(
        self,
        *,
        backend: AgentBackend,
    ) -> None:
        # Fail at the line that names the backend, not hours later at first propose.
        try:
            backend._assert_ready()
        except (RuntimeError, OSError) as exc:
            raise PreflightError(
                f"estimator agent backend is not ready: {exc}"
            ) from exc
        self.backend = backend

    def propose(
        self,
        *,
        repo_root: Path,
        tracker: RunStore,
        target: TrainingTargetConfig,
        emit: AgentProgress = _print_agent_progress,
    ) -> None:
        staged = self._stage_reads(tracker, worktree=repo_root, experiment_id=None)
        program_md_path = repo_root / "program.md"
        learning_md_path = staged.learning_path()
        runs_index_path = staged.runs_index_path()
        prompt = "\n".join(
            [
                "Autonomous candidate proposal phase.",
                "Read program.md; you are the proposer — your role is its 'Propose' and 'Patch' "
                "steps in the 'One cycle' section (diagnose the prior run, then put one bounded "
                "candidate patch in the working tree).",
                # target facts injected from the validate_candidate-enforced config object, never restated in program.md prose
                f"Editable files (machine-enforced): {', '.join(target.patch_paths)}. "
                f"The candidate MUST change {target.surface}.",
                "Read these authoritative files now:",
                f"- {program_md_path}",
                f"- {learning_md_path}",
                f"- {runs_index_path}",
                "",
                "Before stopping, run the test command program.md's 'Patch' step names and keep it green.",
                "Stop once the candidate patch is present in the working tree and that test passes.",
            ]
        )
        turn = self.backend.run_turn(
            prompt=prompt,
            repo_root=repo_root,
            emit=lambda line: emit("propose", line),
        )
        emit("propose", f"done · {turn.progress_summary}")

    def diagnose(
        self,
        result: ExperimentResult,
        *,
        repo_root: Path,
        tracker: RunStore,
        target: TrainingTargetConfig,
        emit: AgentProgress = _print_agent_progress,
    ) -> None:
        staged = self._stage_reads(
            tracker, worktree=repo_root, experiment_id=result.experiment_id
        )
        program_md_path = repo_root / "program.md"
        experiment_json_path = staged.experiment_path(result.experiment_id)
        runs_index_path = staged.runs_index_path()
        learning_md_path = staged.learning_path()
        draft_path = staged.root / "learning.draft.md"
        thread_id: str | None = None
        feedback_note: str | None = None
        for _ in range(_MAX_DIAGNOSIS_ATTEMPTS):
            lines = [
                "Autonomous post-run diagnosis phase.",
                "Read program.md; you are the diagnoser — your role is its 'Diagnose' step in the "
                "'One cycle' section (rewrite the learning memo from the concluded run, per the "
                "method and sections described there).",
                f"The editable policy surface under training is {target.surface}.",
                "Read these authoritative files now:",
                f"- {program_md_path}",
                f"- {experiment_json_path}",
                f"- {runs_index_path}",
                f"- {learning_md_path}",
                "",
                f"Write the full rewritten learning memo to: {draft_path}",
                f"Keep it non-empty and at most {LEARNING_MEMO_MAX_LINES} lines.",
            ]
            if feedback_note is not None:
                lines.extend(
                    [
                        "",
                        "Supervisor feedback from the previous diagnosis turn:",
                        feedback_note,
                    ]
                )
            last_turn = self.backend.run_turn(
                prompt="\n".join(lines),
                repo_root=repo_root,
                emit=lambda line: emit("diagnose", line),
                thread_id=thread_id,
            )
            thread_id = last_turn.thread_id
            feedback_note = _diagnosis_rejection(draft_path)
            if feedback_note is None:
                # The agent's tools are confined to the worktree, so the framework
                # publishes the accepted draft across the boundary.
                tracker.publish_learning(draft_path.read_text())
                emit("diagnose", f"done · {last_turn.progress_summary}")
                return

        raise RuntimeError(
            f"diagnosis memo rejected after {_MAX_DIAGNOSIS_ATTEMPTS} attempts: {feedback_note}"
        )

    @staticmethod
    def _stage_reads(
        source: RunStore, *, worktree: Path, experiment_id: str | None
    ) -> RunStore:
        """Copy the shared store's read-files into a real worktree-local dir the agent reads.

        The agent's tools are confined to the worktree, so it cannot read the store directly.
        A real dir, never a symlink: a headless agent turn materializes worktree symlinks into
        real dirs, severing them from the store."""
        staged = RunStore(worktree / "experiments")
        staged.root.mkdir(parents=True, exist_ok=True)
        reads = [
            (source.learning_path(), staged.learning_path()),
            (source.runs_index_path(), staged.runs_index_path()),
        ]
        if experiment_id is not None:
            reads.append(
                (
                    source.experiment_path(experiment_id),
                    staged.experiment_path(experiment_id),
                )
            )
        for src, dst in reads:
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        return staged


def _diagnosis_rejection(draft_path: Path) -> str | None:
    if not draft_path.exists():
        return f"write the full rewritten learning memo to {draft_path}"
    text = draft_path.read_text()
    if not text.strip():
        return "learning draft is empty"
    line_count = len(text.splitlines())
    if line_count > LEARNING_MEMO_MAX_LINES:
        return (
            f"learning draft is {line_count} lines, over the "
            f"{LEARNING_MEMO_MAX_LINES}-line limit"
        )
    return None
