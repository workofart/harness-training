"""Measurement identity and rollout provenance."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import RunConfig
from src.env.base import (
    RunAction,
    StepResult,
    VerifyAction,
)
from src import determinism
from src.rollout.execution import Execution
from src.rollout.records import MeasurementIdentity


async def resolve_measurement_identity(
    run_config: RunConfig,
    execution: Execution,
) -> MeasurementIdentity:
    identity_payload = run_config.measurement_identity_payload()
    identity_payload["environment"]["task_names"].sort()
    effective_config_digest = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    lines = await execution.fingerprint(tuple(run_config.environment.task_names))
    replay_regime_digest = hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()
    return MeasurementIdentity(
        effective_config_digest=effective_config_digest,
        provider_revision=run_config.llm_provider_config.provider_revision,
        replay_regime_digest=replay_regime_digest,
    )


def action_payload(action: RunAction | VerifyAction) -> dict[str, Any]:
    """Canonical dict shape for an env action.

    Single source of truth for how an action serializes, shared by cache
    keys, recorded chain rows, and drift diagnostics."""
    if isinstance(action, RunAction):
        return {
            "kind": "run",
            "command": action.command,
            "cwd": action.cwd,
            "timeout_sec": action.timeout_sec,
        }
    return {"kind": "verify"}


def serialize_step_result(result: StepResult) -> str:
    return json.dumps(
        dataclasses.asdict(result),
        sort_keys=True,
        separators=(",", ":"),
    )


def scrubbed_hash(result: StepResult, *, command: str | None = None) -> str:
    """Drift-audit identity for cached StepResult fields, including info."""
    raw = result.raw_env_output
    payload = serialize_step_result(
        dataclasses.replace(
            result,
            raw_env_output=dataclasses.replace(
                raw,
                stdout=determinism.drift_audit_canonical(raw.stdout, command=command),
                stderr=determinism.drift_audit_canonical(raw.stderr, command=command),
            ),
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def verdict_summary(result: StepResult) -> dict[str, Any]:
    """Drift-audit identity for a verify StepResult; compared by equality on replay."""
    verdict = result.verdict
    return {
        "passed": None if verdict is None else verdict.passed,
        "reward": result.reward,
    }


@dataclass(frozen=True, slots=True)
class ChainStep:
    env_action: RunAction | VerifyAction
    step_result: StepResult
    timed_out: bool


# Rollout-relative path of the recorded action chain: written by `write_chain`,
# read back by determinism audit.
DETERMINISM_CHAIN_RELPATH = "infra/determinism_chain.jsonl"


def write_chain(path: Path, steps: Sequence[ChainStep]) -> None:
    rows: list[dict[str, Any]] = []
    for step in steps:
        action = action_payload(step.env_action)
        if isinstance(step.env_action, RunAction):
            rows.append(
                {
                    "action": action,
                    "audit_hash": scrubbed_hash(
                        step.step_result, command=step.env_action.command
                    ),
                    "timed_out": step.timed_out,
                }
            )
        else:
            rows.append(
                {
                    "action": action,
                    "verdict": verdict_summary(step.step_result),
                    "timed_out": step.timed_out,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
