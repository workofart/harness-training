"""Replay keying contract: schema salt, namespaces, epochs, snapshot tokens."""

from __future__ import annotations

import base64
import json

from src.env.base import NetworkSnapshotEnv, TaskEnv
from src.plugins.caching import store as cache

_EPOCH_KEY_PREFIX = "env:epoch:"
# Bump when key, result serialization, drift audit, or recordability semantics change.
_CACHE_SCHEMA = "step-result-v3-metrics"


def namespace_for(content_id: str | None) -> str | None:
    if content_id is None:
        return None
    return f"{_CACHE_SCHEMA}:{content_id}"


def epoch_counter_key(namespace: str) -> str:
    """Cache-DB counter key for a namespace's epoch; writer and reader agree here."""
    return f"{_EPOCH_KEY_PREFIX}{namespace}"


def canonical_scope(*, namespace: str, epoch: int) -> str:
    """Canonical scope bytes shared by cache-key digests and snapshot tokens."""
    return json.dumps(
        {"namespace": namespace, "epoch": epoch},
        sort_keys=True,
        separators=(",", ":"),
    )


def _network_snapshot_token(*, namespace: str, epoch: int) -> str:
    payload = canonical_scope(namespace=namespace, epoch=epoch).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


async def resolve_scope(content_id: str | None) -> tuple[str, int] | None:
    """One task's replay scope, (namespace, epoch); None means live-only.

    The single derivation shared by identity fingerprints and scope
    application, so what the digest claims is always what execution does.
    """
    namespace = namespace_for(content_id)
    if namespace is None:
        return None
    epoch = await cache.get_counter(epoch_counter_key(namespace))
    return namespace, epoch


async def apply_replay_scope(
    *, content_id: str | None, env: TaskEnv
) -> tuple[str, int] | None:
    """Resolve the replay scope and wire it onto the env before it starts.

    Applied identically for recording, replay, and audit, so the audit always
    re-executes under the regime the chain was recorded under. Only reachable
    under ReplayExecution.
    """
    scope = await resolve_scope(content_id)
    if scope is not None and isinstance(env, NetworkSnapshotEnv):
        namespace, epoch = scope
        env.pin_network_snapshot(
            _network_snapshot_token(namespace=namespace, epoch=epoch)
        )
    return scope
