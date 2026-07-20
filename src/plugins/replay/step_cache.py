"""Recorded environment step-cache plugin."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
import shlex
import time

from src.env.base import (
    COMMAND_TIMEOUT_EXIT_CODE,
    RawEnvOutput,
    RunAction,
    StepResult,
    TaskEnv,
    VerifyAction,
    VerifyArtifactWriter,
    VerifyVerdict,
    execute_env_action,
    scrub_step_result,
)
from src.plugins.caching import store as cache
from src.plugins.replay.contract import (
    apply_replay_scope,
    canonical_scope,
)
from src.rollout.certification import (
    action_payload,
    scrubbed_hash,
    serialize_step_result,
)
from src.rollout.execution import ExecutionDriftError

_BLOCKED_VALUE = '{"live_only":true}'
_DEADLINE_RECORDING_FRACTION = 0.8
_DEADLINE_RECORDING_SLACK_SEC = 10.0
_BACKGROUND_PROCESS_RE = re.compile(r"(?<![&>])&(?:\s|$)|(?:^|[;({\s])nohup(?:\s|$)")
_LOGGER = logging.getLogger(__name__)


def _canonical_action(action: RunAction | VerifyAction) -> str:
    """Cache-key bytes for an env action, derived from its canonical shape."""
    return json.dumps(action_payload(action), sort_keys=True, separators=(",", ":"))


@dataclasses.dataclass(frozen=True, slots=True)
class _ReplayedStep:
    action: RunAction
    key: str
    recorded_hash: str


async def make_replay_cache(
    *, content_id: str | None, env: TaskEnv
) -> ReplayCache | None:
    """Replay cache for one rollout attempt; live-only tasks return None."""
    scope = await apply_replay_scope(content_id=content_id, env=env)
    if scope is None:
        return None
    namespace, epoch = scope
    return ReplayCache(namespace=namespace, epoch=epoch, env=env)


class ReplayCache:
    """Serves recorded step results while the action chain is known; live after."""

    def __init__(self, *, namespace: str, epoch: int, env: TaskEnv) -> None:
        self._env = env
        self._namespace = namespace
        self._epoch = epoch
        self._digest = hashlib.sha256(
            canonical_scope(namespace=namespace, epoch=epoch).encode()
        ).hexdigest()
        # Retain prefix metadata to materialize and audit after a cache miss.
        self._replayed: list[_ReplayedStep] = []
        self._live = False
        self._recordable_chain = True
        self._verify_key: str | None = None

    async def provision(self) -> None:
        pass

    async def step_run(self, action: RunAction) -> StepResult:
        key = self._advance(_canonical_action(action))
        replayed = await self._serve_replay(key)
        if replayed is not None:
            self._replayed.append(
                _ReplayedStep(
                    action=action,
                    key=key,
                    recorded_hash=scrubbed_hash(replayed, command=action.command),
                )
            )
            return replayed
        await self._go_live()
        started_at = time.monotonic()
        result = await execute_env_action(self._env, action)
        await self._record_run(key, action, result, time.monotonic() - started_at)
        return result

    async def prepare_submit(self) -> StepResult | None:
        self._verify_key = self._advance(_canonical_action(VerifyAction()))
        replayed = await self._serve_replay(self._verify_key)
        if replayed is not None:
            if isinstance(self._env, VerifyArtifactWriter):
                self._env.write_verify_artifacts(replayed)
            return replayed
        await self._go_live()
        return None

    async def step_verify(self) -> StepResult:
        if self._verify_key is None:
            raise RuntimeError("prepare_submit must be called before step_verify")
        if not self._live:
            raise RuntimeError("prepare_submit must go live before step_verify")
        result = await execute_env_action(self._env, VerifyAction())
        if not self._recordable_chain:
            return result
        await cache.put(self._verify_key, serialize_step_result(result))
        return result

    def _advance(self, canonical: str) -> str:
        self._digest = hashlib.sha256(
            f"{self._digest}|{canonical}".encode()
        ).hexdigest()
        return f"env:{self._digest}"

    async def _serve_replay(self, key: str) -> StepResult | None:
        if self._live:
            return None
        if await cache.get(_blocked_key(key)) is not None:
            self._recordable_chain = False
            return None
        hit = await cache.get(key)
        if hit is None:
            return None
        return _deserialize(hit)

    async def _go_live(self) -> None:
        """Idempotent; first call materializes the replayed prefix."""
        if self._live:
            return
        self._live = True
        await self._materialize()

    async def _record_run(
        self, key: str, action: RunAction, result: StepResult, elapsed_sec: float
    ) -> None:
        if not self._recordable_chain:
            return
        skip_reason = _skip_recording_reason(action, result, elapsed_sec)
        if skip_reason is not None:
            self._recordable_chain = False
            _LOGGER.warning(
                "env step cache skipped recording: namespace=%s reason=%s "
                "elapsed_sec=%.3f timeout_sec=%s exit_code=%s action=%r",
                self._namespace,
                skip_reason,
                elapsed_sec,
                action.timeout_sec,
                result.raw_env_output.exit_code,
                (
                    dataclasses.replace(action, command=action.command[:160] + "...")
                    if len(action.command) > 160
                    else action
                ),
            )
            return
        await cache.put(key, serialize_step_result(result))

    async def _materialize(self) -> None:
        """Re-execute the replayed prefix so the real env reaches chain state.

        Drift fails the rollout: the comparison is on scrubbed (model-visible)
        output, so a mismatch means the model already consumed observations
        the live environment no longer reproduces -- the rollout's suffix would
        be graded against a world that was never real. Crash loud instead.
        """
        for action_index, replayed in enumerate(self._replayed, start=1):
            result = await execute_env_action(self._env, replayed.action)
            live = scrub_step_result(result, command=replayed.action.command)
            live_hash = scrubbed_hash(live, command=replayed.action.command)
            if live_hash != replayed.recorded_hash:
                # Cross-run optimization only; this rollout's retry goes live first.
                await cache.put(_blocked_key(replayed.key), _BLOCKED_VALUE)
                raise ExecutionDriftError(
                    action_index=action_index,
                    diagnostic={
                        "namespace": self._namespace,
                        "epoch": self._epoch,
                        "action": action_payload(replayed.action),
                        "recorded": dataclasses.asdict(
                            await self._recorded_result_for_error(replayed)
                        ),
                        "live": dataclasses.asdict(live),
                    },
                )
        self._replayed.clear()

    async def _recorded_result_for_error(self, replayed: _ReplayedStep) -> StepResult:
        payload = await cache.get(replayed.key)
        if payload is None:
            raise RuntimeError(
                f"recorded env step cache payload disappeared for {replayed.key}"
            )
        result = _deserialize(payload)
        return scrub_step_result(result, command=replayed.action.command)


def _blocked_key(key: str) -> str:
    return f"{key}:live-only"


def _skip_recording_reason(
    action: RunAction, result: StepResult, elapsed_sec: float
) -> str | None:
    # False negatives only cost retries; materialization remains the correctness check.
    if _acquires_live_process_state(action):
        return "background_process"
    if result.raw_env_output.exit_code == COMMAND_TIMEOUT_EXIT_CODE:
        return "command_timeout"
    if action.timeout_sec is not None:
        near_deadline = elapsed_sec > action.timeout_sec * _DEADLINE_RECORDING_FRACTION
        near_budget = elapsed_sec > action.timeout_sec - _DEADLINE_RECORDING_SLACK_SEC
        if near_deadline and near_budget:
            return "near_deadline"
    return None


def _acquires_live_process_state(action: RunAction) -> bool:
    command = _strip_heredoc_bodies(action.command)
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return _BACKGROUND_PROCESS_RE.search(command) is not None
    return "&" in tokens or "nohup" in tokens


def _strip_heredoc_bodies(command: str) -> str:
    # Ignore heredoc bodies so content like `x & 1` is not treated as backgrounding.
    lines: list[str] = []
    pending: list[tuple[str, bool]] = []
    for line in command.splitlines():
        if pending:
            delimiter, strip_tabs = pending[0]
            candidate = line.lstrip("\t") if strip_tabs else line
            if candidate == delimiter:
                pending.pop(0)
            continue
        lines.append(line)
        pending.extend(_heredoc_delimiters(line))
    return "\n".join(lines)


def _heredoc_delimiters(line: str) -> list[tuple[str, bool]]:
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return []
    delimiters: list[tuple[str, bool]] = []
    token_index = 0
    while token_index < len(tokens) - 1:
        if tokens[token_index] != "<<":
            token_index += 1
            continue
        delimiter = tokens[token_index + 1]
        strip_tabs = delimiter.startswith("-")
        delimiter = delimiter.removeprefix("-")
        if delimiter:
            delimiters.append((delimiter, strip_tabs))
        token_index += 2
    return delimiters


def _deserialize(text: str) -> StepResult:
    data = json.loads(text)
    return StepResult(
        raw_env_output=RawEnvOutput(**data["raw_env_output"]),
        reward=data["reward"],
        terminated=data["terminated"],
        truncated=data["truncated"],
        info=data["info"],
        metrics=data["metrics"],
        verdict=None if data["verdict"] is None else VerifyVerdict(**data["verdict"]),
    )
