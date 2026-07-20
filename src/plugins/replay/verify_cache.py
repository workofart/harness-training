"""Verify-cache wrapper: memoizes env's pure grader on the identity env defines."""

from __future__ import annotations

from typing import Any

from src.env import swebench_verify
from src.plugins.caching import store as cache


def cache_wrapper(grader):
    async def cached(
        *, spec: Any, patch: str, rollout_artifact_dir: Any
    ) -> swebench_verify.SweBenchVerifyResult:
        key = swebench_verify.verify_cache_key(spec.instance_id, patch)
        raw = await cache.get(key)
        if raw is not None:
            restored = swebench_verify.restore_cached(
                raw,
                rollout_artifact_dir=rollout_artifact_dir,
                instance_id=spec.instance_id,
                diff=patch,
            )
            if restored is not None:
                return restored
        result = await grader(
            spec=spec, patch=patch, rollout_artifact_dir=rollout_artifact_dir
        )
        # Only completed grades are reproducible enough to memoize.
        if result.completed:
            await cache.put(key, swebench_verify.cache_payload(result))
        return result

    return cached
