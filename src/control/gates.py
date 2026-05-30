"""Candidate-validation gates for the supervisor loop.

Pure policy checks run against a candidate's workspace before and after a
tracked experiment: which paths/config fields it may edit, that its diff
embeds no literal task ids, that a post-run turn touches only the learning
memo, and that its mechanism is not a relabel of a recently discarded one.
Extracted from supervisor.py so the orchestration loop stays separate from
the rules it enforces.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.control import repo as control_repo
from src.experiment.gate import load_recent_candidate_records
from src.harness.config import HarnessConfig


SUPERVISOR_EDITABLE_PATHS = (
    "config/harness_config.json",
    "src/harness/core.py",
    "tests/harness/test_core.py",
)
CANDIDATE_EDITABLE_CONFIG_FIELDS: frozenset[str] = frozenset({"focus_name"})
CANDIDATE_DIFF_SCAN_PATHS: tuple[str, ...] = (
    "src/harness/core.py",
    "tests/harness/test_core.py",
)


def validate_candidate_editable_paths(
    *,
    changed_paths: tuple[str, ...],
) -> None:
    allowed_paths = set(SUPERVISOR_EDITABLE_PATHS)
    invalid_paths = sorted(path for path in changed_paths if path not in allowed_paths)
    if invalid_paths:
        raise RuntimeError(
            "candidate modified paths outside supervisor editable paths "
            "(program.md Source-of-truth boundary): "
            + ", ".join(invalid_paths)
            + ". Visible support files are read-only context; express harness "
            "behavior changes through src/harness/core.py with focused tests."
        )


def _validate_no_task_ids_in_added_lines(
    *,
    added_lines: list[str],
    task_ids: tuple[str, ...],
) -> None:
    if not task_ids or not added_lines:
        return
    offenders: dict[str, list[str]] = {}
    for task_id in task_ids:
        pattern = re.compile(r"\b" + re.escape(task_id) + r"\b")
        for line in added_lines:
            if pattern.search(line):
                offenders.setdefault(task_id, []).append(line.strip()[:120])
                break
    if offenders:
        details = "; ".join(
            f"{task_id} -> {samples[0]!r}"
            for task_id, samples in sorted(offenders.items())
        )
        raise RuntimeError(
            "candidate diff embeds literal task ids in harness paths "
            "(program.md Evidence task-agnostic rule; use a generic mechanism "
            f"instead): {details}"
        )


def validate_no_task_ids_in_workspace_diff(
    *,
    workspace_root: Path,
    task_ids: tuple[str, ...],
) -> None:
    if not (workspace_root / ".git").exists():
        return
    _validate_no_task_ids_in_added_lines(
        added_lines=control_repo.git_diff_added_lines_worktree(
            cwd=workspace_root,
            paths=CANDIDATE_DIFF_SCAN_PATHS,
        ),
        task_ids=task_ids,
    )


def validate_learning_memo_update(
    *,
    before_payload: dict[str, object],
    after_payload: dict[str, object],
    before_learning: str,
    after_learning: str,
) -> None:
    changed_keys = sorted(
        key
        for key in set(before_payload) | set(after_payload)
        if before_payload.get(key) != after_payload.get(key)
    )
    if changed_keys:
        raise RuntimeError(
            "diagnosis turn modified experiment.json; update only "
            "`experiments/learning.md`: " + ", ".join(changed_keys)
        )
    if not after_learning.strip():
        raise RuntimeError("learning memo must be non-empty")
    if after_learning == before_learning:
        raise RuntimeError("learning memo was not updated")


def _load_head_harness_config_for_repo(repo_root: Path) -> HarnessConfig:
    payload = control_repo.git_show_at_head("config/harness_config.json", cwd=repo_root)
    return HarnessConfig.model_validate_json(payload)


def validate_candidate_config_patch(
    *,
    workspace_root: Path,
    harness_config: HarnessConfig,
) -> None:
    head_harness_config = _load_head_harness_config_for_repo(workspace_root)
    candidate_payload = harness_config.model_dump(mode="json")
    head_payload = head_harness_config.model_dump(mode="json")
    changed_fields = sorted(
        key
        for key in set(head_payload) | set(candidate_payload)
        if head_payload.get(key) != candidate_payload.get(key)
    )
    contract_fields = [
        key for key in changed_fields if key not in CANDIDATE_EDITABLE_CONFIG_FIELDS
    ]
    if contract_fields:
        raise RuntimeError(
            "candidate changed supervisor-owned harness config fields "
            "(program.md Source-of-truth boundary); config/harness_config.json "
            "candidate edits are limited to focus_name. "
            "Changed contract fields: " + ", ".join(contract_fields)
        )
    if harness_config.focus_name == head_harness_config.focus_name:
        raise RuntimeError(
            "candidate must set config/harness_config.json focus_name to a short "
            "mechanism label for this experiment"
        )


def _mechanism_added_lines(lines: list[str]) -> set[str]:
    return {
        normalized
        for line in lines
        if (normalized := " ".join(line.strip().split()))
        and len(normalized) >= 12
        and not normalized.startswith("#")
        and any(character.isalpha() for character in normalized)
    }


def build_mechanism_novelty_rejection(
    *,
    workspace_root: Path,
    experiments_root: Path,
    changed_paths: tuple[str, ...],
    window: int = 10,
) -> str | None:
    if "src/harness/core.py" not in changed_paths:
        return None
    if not (workspace_root / ".git").exists():
        return None
    candidate_lines = _mechanism_added_lines(
        control_repo.git_diff_added_lines_worktree(
            cwd=workspace_root,
            paths=CANDIDATE_DIFF_SCAN_PATHS,
        )
    )
    if not candidate_lines:
        return None
    recent_discards = [
        r
        for r in load_recent_candidate_records(
            experiments_root=experiments_root,
            window=window,
        )
        if r.status == "discard"
    ]
    for record in recent_discards:
        parent_ref = None
        if record.evidence is not None:
            parent_ref = record.evidence.candidate_change.parent_baseline_commit
        if parent_ref is None:
            continue
        discard_lines = _mechanism_added_lines(
            control_repo.git_diff_added_lines_between(
                cwd=workspace_root,
                base_ref=parent_ref,
                head_ref=record.git_commit_hash,
                paths=CANDIDATE_DIFF_SCAN_PATHS,
            )
        )
        if not discard_lines:
            continue
        if candidate_lines == discard_lines or candidate_lines.issubset(discard_lines):
            return (
                f"Candidate mechanism is identical to recently discarded "
                f"{record.experiment_id} (reason: {record.decision_reason}). "
                f"You must use a structurally different approach — do not "
                f"redeploy the same mechanism with only a label change."
            )
    return None
