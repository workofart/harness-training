"""Public evaluation facade and CLI."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from collections.abc import Callable, Sequence
from pathlib import Path

from dotenv import load_dotenv

from src.config import RunConfig
from src.measurement import MeasurementError, preflight, run_isolated_experiment
from src.rollout.records import ExperimentResult
from src.rollout.store import RunStore, invoker_repo_root
from src.trainer.logger import StdoutLogger, suite_summary

_DEFAULT_CONFIG_PATH = Path("config/quickstart_eval.yaml")


def evaluate(
    run_config: RunConfig,
    *,
    experiment_id: str | None = None,
) -> ExperimentResult:
    return _evaluate(
        run_config,
        experiment_id=experiment_id,
        task_ids=tuple(run_config.environment.task_names),
    )


def _evaluate(
    run_config: RunConfig,
    *,
    experiment_id: str | None,
    task_ids: tuple[str, ...],
    on_experiment_started: Callable[[], None] | None = None,
    on_task_finished: Callable[[str], None] | None = None,
    live: bool | None = None,
) -> ExperimentResult:
    measure_root = invoker_repo_root()
    config_path = run_config.config_path
    assert config_path is not None
    tracker = RunStore(measure_root / "evals")
    stdout = StdoutLogger(show_task_progress=True, live=live)
    observer = (
        stdout
        if on_experiment_started is None and on_task_finished is None
        else _TailObserver(stdout, on_experiment_started, on_task_finished)
    )
    llm = run_config.llm_provider_config
    stdout.run_started(
        f"eval · {config_path}",
        (
            ("policy", run_config.training_target.surface),
            ("llm", f"{llm.model_name} · {llm.base_url}"),
            (
                "env",
                f"{run_config.environment.kind} · {len(task_ids)} tasks · "
                f"concurrency {run_config.max_rollout_concurrency}",
            ),
        ),
    )
    stdout.measurement_started(task_ids, None, subject=f"eval {config_path}")
    try:
        result = run_isolated_experiment(
            config_path=config_path,
            tracker=tracker,
            observer=observer,
            measure_root=measure_root,
            task_ids=task_ids,
            experiment_id=experiment_id,
        )
    except MeasurementError as exc:
        if exc.result is not None:
            crashed = tracker.record_finalized_run(exc.result, kind="eval")
            stdout.experiment_finished(crashed)
        stdout.experiment_failed(exc)
        raise
    result = tracker.record_finalized_run(result, kind="eval")
    stdout.experiment_finished(result)
    return result


class _TailObserver:
    def __init__(
        self,
        stdout: StdoutLogger,
        on_experiment_started: Callable[[], None] | None,
        on_task_finished: Callable[[str], None] | None,
    ) -> None:
        self._stdout = stdout
        self._on_experiment_started = on_experiment_started
        self._on_task_finished = on_task_finished

    def log(self, line: str) -> None:
        self._stdout.log(line)

    def experiment_started(self, experiment_id: str) -> None:
        self._stdout.experiment_started(experiment_id)
        if self._on_experiment_started is not None:
            self._on_experiment_started()

    def task_finished(self, task_id: str, failure_mode: str) -> None:
        self._stdout.task_finished(task_id, failure_mode)
        if self._on_task_finished is not None:
            self._on_task_finished(task_id)

    def measurement_heartbeat(self) -> None:
        self._stdout.measurement_heartbeat()


def _parse_iterations(argv: Sequence[str]) -> tuple[int | None, list[str]]:
    positions = [index for index, argument in enumerate(argv) if argument == "--n"]
    if not positions:
        return None, list(argv)
    if len(positions) != 1 or positions[0] + 1 == len(argv):
        raise ValueError("--n must be a positive integer")
    position = positions[0]
    try:
        iterations = int(argv[position + 1])
    except ValueError:
        raise ValueError("--n must be a positive integer") from None
    if iterations < 1:
        raise ValueError("--n must be a positive integer")
    remaining = list(argv[:position]) + list(argv[position + 2 :])
    return iterations, remaining


@dataclass(frozen=True)
class _ScheduledRun:
    config: RunConfig
    experiment_id: str | None


def _run_overlapping(
    runs: Sequence[_ScheduledRun],
) -> tuple[bool, list[ExperimentResult]]:
    """Launch the next run from the main thread once the current tail is idle."""
    pending = iter(runs)
    active: list[set[str]] = []
    results: list[ExperimentResult] = []
    failed = False

    def run_next() -> None:
        nonlocal failed
        scheduled = next(pending, None)
        if scheduled is None:
            return

        panel = tuple(scheduled.config.environment.task_names)
        conflicts = {task for unfinished in active for task in unfinished}
        ordered = tuple(task for task in panel if task not in conflicts) + tuple(
            task for task in panel if task in conflicts
        )
        unfinished = set(panel)
        active.append(unfinished)
        next_started = False

        def start_next() -> None:
            nonlocal next_started
            if next_started:
                return
            next_started = True
            run_next()

        def maybe_start_next() -> None:
            if len(unfinished) < scheduled.config.max_rollout_concurrency:
                start_next()

        def task_finished(task_id: str) -> None:
            unfinished.remove(task_id)
            maybe_start_next()

        try:
            results.append(
                _evaluate(
                    scheduled.config,
                    experiment_id=scheduled.experiment_id,
                    task_ids=ordered,
                    on_experiment_started=maybe_start_next,
                    on_task_finished=task_finished,
                    # Overlapping runs would fight over one live terminal block.
                    live=None if len(runs) == 1 else False,
                )
            )
        except MeasurementError as exc:
            failed = True
            if exc.result is not None:
                results.append(exc.result)
        finally:
            active.remove(unfinished)
        start_next()

    run_next()
    return failed, results


def main(argv: Sequence[str]) -> int:
    load_dotenv()
    iterations, config_argv = _parse_iterations(argv)
    if not config_argv:
        config_argv = [str(invoker_repo_root() / _DEFAULT_CONFIG_PATH)]

    runnable_configs = [RunConfig.load(path) for path in config_argv]

    preflight(runnable_configs)
    tracker = RunStore(invoker_repo_root() / "evals")
    issued: set[str] = set()
    iteration_numbers: Sequence[int | None] = (
        (None,) if iterations is None else range(1, iterations + 1)
    )
    runs = []
    for iteration in iteration_numbers:
        for config in runnable_configs:
            if iteration is None:
                runs.append(_ScheduledRun(config, None))
                continue
            timestamp = datetime.now(UTC)
            base = (
                f"exp-{timestamp.strftime('%Y%m%d-%H%M%S-%f')}__iteration-{iteration}"
            )
            name, suffix = base, 2
            # Reserve names before their run directories exist.
            while name in issued or tracker.run_dir(name).exists():
                name = f"{base}-{suffix}"
                suffix += 1
            issued.add(name)
            runs.append(_ScheduledRun(config, name))
    failed, results = _run_overlapping(runs)
    if len(runs) > 1:
        suite_summary(results)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
