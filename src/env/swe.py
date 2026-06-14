"""SWE-bench-Verified `HarnessEnv` adapter.

Wraps one SWE-bench-Verified task in the `HarnessEnv` contract so the existing
`run_trial` / `run_task_loop` drive it unchanged. Validated as a higher-SNR
substrate for the self-improving loop (see plan.md "Phase 2 — EXECUTED": a
char-200 crippled scaffold tanks graded reward by +0.520, p≈5e-7, where Terminal
Bench showed +0.024 non-significant).

Firewall (one-shot hidden verifier). The instance image has the repo at
`base_commit` WITHOUT the held-out test_patch (the F2P tests). The agent sees
only the pre-existing (P2P) tests it may legitimately run. The held-out tests
are injected by `verify()` (terminal, once) via swebench's `eval_script`, which
the agent never sees. Containers run `--network none`, so the agent cannot fetch
the upstream fix, a fixed package version, or the held-out test — the verifier
stays the single, authoritative, one-shot judgment. `--network none` also makes
grading deterministic and matches offline production grading; only tasks that
grade gold->1.0 offline belong in a panel (the gold pre-screen is the filter).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
import uuid

from swebench.harness.constants import FAIL_TO_PASS, PASS_TO_PASS
from swebench.harness.grading import get_eval_tests_report, get_logs_eval
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec

from src.contracts import EnvExecWorkload, RawState


class VerifierCorruptError(RuntimeError):
    """A verify() run swebench could not grade -- a failed held-out test-patch
    apply, repo reset failure, test-runner error/timeout, or missing output
    markers.

    Raised (rather than scored 0.0) so `run_trial` records the trial as infra
    with `error` set and EXCLUDES it from the graded gate, instead of charging
    verifier corruption to the agent as a `verified_rejected` failure.
    """


def _as_list(v) -> list[str]:
    return v if isinstance(v, list) else json.loads(v)


def _score_report(report: dict) -> tuple[float, bool]:
    """Reward + resolution from a swebench eval report (pure, unit-tested).

    reward = fraction of (F2P + P2P) tests passing; resolution = every F2P now
    passes AND every P2P still passes (computed directly, not via the enum
    string). P2P is a high floor that cancels in a per-task differential.
    """
    passed = sum(len(report[k]["success"]) for k in (FAIL_TO_PASS, PASS_TO_PASS))
    total = sum(
        len(report[k]["success"]) + len(report[k]["failure"])
        for k in (FAIL_TO_PASS, PASS_TO_PASS)
    )
    reward = passed / total if total else 0.0
    resolved = (
        len(report[FAIL_TO_PASS]["failure"]) == 0
        and len(report[FAIL_TO_PASS]["success"]) > 0
        and len(report[PASS_TO_PASS]["failure"]) == 0
    )
    return reward, resolved


def _grade_log(
    *,
    spec: TestSpec,
    output: str,
    f2p: list[str],
    p2p: list[str],
    artifacts_dir: str,
) -> tuple[float, bool]:
    """Grade a raw verifier log, failing fast on a corrupt run.

    `output` is swebench's `eval_script` output, with the Start/End test-output
    markers already emitted in order (the script runs under `set -x` and we merge
    its streams). We persist it AS-IS and let `get_logs_eval` parse it -- we do
    NOT re-wrap it in fresh markers. Fabricating those markers is exactly what
    hides a corrupt run: `get_logs_eval` returns `ok=False` when the held-out
    test patch failed to apply, the repo reset failed, the test runner
    errored/timed out, or the markers are absent -- and injecting markers would
    mask all of those as a clean, all-failing zero. Corruption is infra, not an
    agent rejection, so we raise instead of scoring it (see `VerifierCorruptError`).
    """
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", dir=artifacts_dir, delete=False
    ) as fh:
        fh.write(output)
        log_fp = fh.name
    try:
        status_map, ok = get_logs_eval(spec, log_fp)
    finally:
        os.unlink(log_fp)
    if not ok:
        raise VerifierCorruptError(
            f"verifier log not gradable (swebench ok=False) for {spec.instance_id}"
        )
    report = get_eval_tests_report(status_map, {FAIL_TO_PASS: f2p, PASS_TO_PASS: p2p})
    return _score_report(report)


async def _run(*args: str, timeout: float | None = None) -> tuple[int, str, str]:
    """Run a host subprocess (docker CLI) and capture decoded stdout/stderr."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, TimeoutError):
        proc.kill()
        await proc.wait()
        raise
    return (
        proc.returncode if proc.returncode is not None else -1,
        out.decode("utf-8", "replace"),
        err.decode("utf-8", "replace"),
    )


def load_rows(instance_ids: list[str]) -> dict[str, dict]:
    """Map SWE-bench-Verified instance ids -> dataset rows (the task source for a
    SWE-backed panel).

    Loaded once per experiment and indexed by id. Raises on any unknown id so a
    mistyped panel fails fast rather than silently dropping a task. The dataset
    is fetched on the host (cached by `datasets`); the firewall is the container,
    not this lookup.
    """
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    by_id = {r["instance_id"]: r for r in ds}
    missing = [i for i in instance_ids if i not in by_id]
    if missing:
        raise KeyError(f"unknown SWE-bench-Verified instance ids: {missing}")
    return {i: by_id[i] for i in instance_ids}


class SweEnv:
    """One SWE-bench-Verified task behind the `HarnessEnv` protocol.

    `row` is a dataset row (from `princeton-nlp/SWE-bench_Verified`);
    `artifacts_dir` is the host directory the trial writes its trace/metrics/
    verifier artifacts into (exposed as `trial_dir`). `heavy_semaphore`, when
    provided, is shared across trials to bound concurrent heavyweight container
    CPU work (reset/startup, the agent's `run`, verify) -- the runner's
    `max_heavy_action_concurrency` tier -- independently of how many trials are
    in flight.
    """

    def __init__(
        self,
        *,
        row: dict,
        artifacts_dir: str,
        heavy_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self._row = row
        self._spec = make_test_spec(row, namespace="swebench")
        self._image = self._spec.instance_image_key
        self._f2p = _as_list(row[FAIL_TO_PASS])
        self._p2p = _as_list(row[PASS_TO_PASS])
        # Unique container name -> no cross-trial collisions, easy bulk cleanup.
        self._container = f"swe_env_{uuid.uuid4().hex[:12]}"
        self._artifacts_dir = artifacts_dir
        os.makedirs(self._artifacts_dir, exist_ok=True)
        self._verifier_stdout_path: str | None = None
        self._started = False
        self._heavy_semaphore = heavy_semaphore

    def _gate(self, workload: EnvExecWorkload):
        """Acquire the heavy-action gate for heavy work; no-op for light."""
        if workload == "light" or self._heavy_semaphore is None:
            return contextlib.nullcontext()
        return self._heavy_semaphore

    # --- HarnessEnv protocol -------------------------------------------------

    @property
    def trial_dir(self) -> str | None:
        return self._artifacts_dir

    @property
    def verifier_stdout_path(self) -> str | None:
        return self._verifier_stdout_path

    async def reset(self) -> RawState:
        # `--network none`: airtight firewall + deterministic offline grading.
        async with self._gate("heavy"):
            rc, _out, err = await _run(
                "docker",
                "run",
                "-d",
                "--platform",
                "linux/amd64",
                "--network",
                "none",
                "--name",
                self._container,
                self._image,
                "sleep",
                "infinity",
                timeout=600,
            )
        if rc != 0:
            raise RuntimeError(f"container start failed ({rc}): {err[:500]}")
        self._started = True
        return RawState(
            instruction=self._row["problem_statement"],
            working_dir="/testbed",
        )

    async def exec(
        self,
        *,
        command: str,
        cwd: str | None = None,
        timeout_sec: int | None = None,
        workload: EnvExecWorkload = "heavy",
    ) -> RawState:
        workdir = cwd or "/testbed"
        # Light actions (ls/find/read/search/edit) bypass the gate; only the
        # agent's `run` (emulated pytest/python) counts against heavy CPU work.
        async with self._gate(workload):
            rc, out, err = await _run(
                "docker",
                "exec",
                "-w",
                workdir,
                self._container,
                "bash",
                "-lc",
                command,
                timeout=timeout_sec,
            )
        return RawState(return_code=rc, stdout=out, stderr=err)

    async def verify(self) -> RawState:
        """Terminal, one-shot judgment: inject held-out tests, grade, score."""
        async with self._gate("heavy"):
            output = await self._run_eval_script()
        # Persist what the verifier saw (parity with the Harbor env's artifact).
        self._verifier_stdout_path = os.path.join(
            self._artifacts_dir, "verifier_stdout.txt"
        )
        with open(self._verifier_stdout_path, "w") as fh:
            fh.write(output)
        reward, resolved = _grade_log(
            spec=self._spec,
            output=output,
            f2p=self._f2p,
            p2p=self._p2p,
            artifacts_dir=self._artifacts_dir,
        )
        return RawState(reward=reward, passed=resolved, stdout=output)

    async def close(self) -> None:
        if self._started:
            await _run("docker", "rm", "-f", self._container, timeout=120)
            self._started = False

    # --- panel-building helpers (not part of HarnessEnv) ---------------------

    async def apply_patch(self, diff: str) -> tuple[int, str]:
        """Apply a unified diff to /testbed. Used by the gold-offline pre-screen
        that selects deterministically-gradable tasks for a panel."""
        await self._copy_in(diff, "/patch.diff")
        rc, out, err = await _run(
            "docker",
            "exec",
            "-w",
            "/testbed",
            self._container,
            "bash",
            "-lc",
            "git apply -v /patch.diff",
            timeout=120,
        )
        return rc, (err or out)

    # --- internals -----------------------------------------------------------

    async def _copy_in(self, content: str, container_path: str) -> None:
        # docker cp is robust against the slim image's missing shell tools.
        fd, host = tempfile.mkstemp(dir=self._artifacts_dir)
        with os.fdopen(fd, "w") as fh:
            fh.write(content)
        await _run("docker", "cp", host, f"{self._container}:{container_path}")
        os.unlink(host)

    async def _run_eval_script(self) -> str:
        await self._copy_in(self._spec.eval_script, "/eval.sh")
        # Merge streams IN ORDER inside the container: `set -x` echoes the
        # Start/End markers to stderr; separate capture reorders them and breaks
        # marker extraction.
        _rc, out, _err = await _run(
            "docker",
            "exec",
            self._container,
            "bash",
            "-lc",
            "bash /eval.sh 2>&1",
            timeout=1800,
        )
        return out
