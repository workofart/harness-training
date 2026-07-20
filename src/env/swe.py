"""SWE-bench-Verified Docker-shell env."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import determinism
from src.config import EnvironmentConfig
from src.env.docker_shell import DockerShellSession
from src.env.base import (
    DockerTaskEnv,
    MODEL_PATCH_INFO_KEY,
    RawEnvOutput,
    TaskSet,
    VerifyOutcome,
    VerifyVerdict,
    VerifyWrapper,
)
from src.rollout.metrics import GENERIC_SECONDARY_METRICS, SecondaryRewardMetric
from src.rollout.records import ExperimentResult, solved_task_ids
from src.env import swebench_verify

_TASK_WORKDIR = "/testbed"
_F2P_PASSED_KEY = "fail_to_pass_passed"
_P2P_FAILED_KEY = "pass_to_pass_failed"
# Binary artifacts make the whole SWE-bench patch unappliable; include tracked and
# intent-to-add text diffs and exclude numstat binary paths.
_MODEL_PATCH_COMMAND = (
    "set -e; git add -N .; binary_paths=(); "
    "while IFS=$'\\t' read -r -d '' added deleted path; do "
    "old_path=; "
    "if [[ -z $path ]]; then "
    "IFS= read -r -d '' old_path; IFS= read -r -d '' path; "
    "fi; "
    "if [[ $added == - && $deleted == - ]]; then "
    '[[ -z $old_path ]] || binary_paths+=(":(top,exclude,literal)$old_path"); '
    'binary_paths+=(":(top,exclude,literal)$path"); '
    "fi; "
    "done < <(git -c core.fileMode=false diff --numstat -z HEAD --); "
    'git -c core.fileMode=false diff HEAD -- . "${binary_paths[@]}"'
)
_SWE_ENV_FINGERPRINT = hashlib.sha256(
    f"{determinism.PINS_FINGERPRINT}\0{_MODEL_PATCH_COMMAND}".encode()
).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class SweBenchTask:
    instruction: str
    spec: Any
    # SWE-bench defines no agent budget.
    agent_timeout_sec: float | None = None
    replay_id: str = _SWE_ENV_FINGERPRINT


async def load_tasks(
    *,
    task_ids: Sequence[str],
    environment: EnvironmentConfig,
    verify_wrapper: VerifyWrapper | None = None,
) -> TaskSet[SweBenchTask]:
    # Defer heavy datasets/swebench imports until task loading.
    from datasets import load_dataset
    from swebench.harness.test_spec.test_spec import make_test_spec

    # SWE-bench Verified publishes a single "test" split.
    dataset = load_dataset(
        "princeton-nlp/SWE-bench_Verified",
        revision="c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
        split="test",
    )
    rows_by_id = {row["instance_id"]: row for row in dataset}
    missing = [task_id for task_id in task_ids if task_id not in rows_by_id]
    if missing:
        raise KeyError(f"unknown SWE-bench-Verified instance ids: {missing}")
    tasks = {
        task_id: SweBenchTask(
            instruction=rows_by_id[task_id]["problem_statement"],
            spec=make_test_spec(rows_by_id[task_id], namespace="swebench"),
        )
        for task_id in task_ids
    }

    return TaskSet(
        kind=environment.kind,
        tasks=tasks,
        env_factory=lambda task, rollout_dir: SweEnv(
            task=task,
            artifacts_dir=rollout_dir,
            verify_wrapper=verify_wrapper,
        ),
    )


class SweEnv(DockerTaskEnv[SweBenchTask]):
    """One SWE-bench-Verified task behind the `TaskEnv` protocol.

    `task` is a SWE-bench-Verified instruction plus official test spec. The env
    manages the solve container and calls the official SWE-bench grader on submit.
    """

    _task_workdir = _TASK_WORKDIR

    def __init__(
        self,
        *,
        task: SweBenchTask,
        artifacts_dir: Path,
        verify_wrapper: VerifyWrapper | None = None,
    ) -> None:
        # None grades live; replay injects a caching wrapper around the grader.
        self._verify_wrapper = verify_wrapper
        super().__init__(
            task=task,
            artifacts_dir=artifacts_dir,
            verify_timeout_sec=float(swebench_verify.OFFICIAL_EVAL_TIMEOUT_SEC),
        )

    def _build_solve_env(self, task: SweBenchTask) -> DockerShellSession:
        # Omitted network mode keeps solving offline and deterministic.
        return DockerShellSession(
            image=task.spec.instance_image_key,
        )

    async def verify(self) -> VerifyOutcome:
        """Terminal submit: capture the patch, run official SWE-bench, and end."""
        await self.provision()
        # Official grader takes a patch string, not the live container.
        patch_result = await self._solve_env.run(
            command=_MODEL_PATCH_COMMAND,
            cwd=_TASK_WORKDIR,
            timeout=120,
            lossless=True,
        )
        patch = "" if patch_result.exit_code != 0 else patch_result.stdout
        await self._solve_env.close()

        spec = self._task.spec
        grader = swebench_verify.verify
        if self._verify_wrapper is not None:
            grader = self._verify_wrapper(grader)
        verify_result = await grader(
            spec=spec,
            patch=patch,
            rollout_artifact_dir=self._artifacts_dir,
        )
        metrics: dict[str, int | float] = {}
        if verify_result.fail_to_pass_passed is not None:
            metrics[_F2P_PASSED_KEY] = verify_result.fail_to_pass_passed
        if verify_result.pass_to_pass_failed is not None:
            metrics[_P2P_FAILED_KEY] = verify_result.pass_to_pass_failed
        return VerifyOutcome(
            reward=verify_result.reward,
            output=RawEnvOutput(stdout=verify_result.stdout),
            info={MODEL_PATCH_INFO_KEY: patch, "instance_id": spec.instance_id},
            metrics=metrics,
            verdict=VerifyVerdict(
                completed=verify_result.completed,
                passed=verify_result.passed,
                error=verify_result.error,
            ),
        )


class F2pProgressMetric(SecondaryRewardMetric):
    # First unsolved-run tiebreaker: F2P fixed minus P2P broken.
    name = "f2p_progress"
    higher_is_better = True

    def values(
        self, *, baseline: ExperimentResult, candidate: ExperimentResult
    ) -> tuple[int, int]:
        # Non-scorable outcomes drop out symmetrically instead of losing progress.
        scope = self._contested(baseline) & self._contested(candidate)
        return self._progress(baseline, scope), self._progress(candidate, scope)

    def _contested(self, experiment: ExperimentResult) -> set[str]:
        solved = solved_task_ids(experiment)
        return {
            task_id
            for task_id, rollout in experiment.tasks.items()
            if task_id not in solved and self._scorable(rollout)
        }

    def _progress(self, experiment: ExperimentResult, task_ids: set[str]) -> int:
        total = 0
        for task_id in task_ids:
            rollout = experiment.tasks[task_id]
            assert rollout is not None
            total += rollout.metrics.get(_F2P_PASSED_KEY, 0) - rollout.metrics.get(
                _P2P_FAILED_KEY, 0
            )
        return total


SECONDARY_METRICS = (F2pProgressMetric(), *GENERIC_SECONDARY_METRICS)
