"""Compact stdout presentation for trainer and evaluation runs.

Progressive disclosure on a TTY: reference info is dimmed, a live block at the
bottom of the screen carries within-measurement progress (bar, the experiment
id, plus a tally of every failure-mode bucket) and in-flight agent meters, and
each epoch collapses to a single scrollback line once its outcome is known.
A bare ``x/y`` always reads solved-out-of-panel; progress counts say ``done``
and independent tallies are ``·``-separated. Unscorable infra
failures mean the grade is untrustworthy, so they stay visible in red even on
collapsed lines. Without a TTY the presentation is append-only: per-event
lines plus periodic heartbeat snapshots are the progress record.
"""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
import time
from typing import get_args

from src.rollout.records import (
    UNSCORABLE_FAILURE_MODES,
    VERDICT_UNTRUSTED_FAILURE_MODES,
    ExperimentResult,
    FailureMode,
    ResultDecision,
    solved_task_ids,
)


_INITIAL_SNAPSHOT_SILENCE_SEC = 120.0
_SNAPSHOT_INTERVAL_SEC = 300.0
# Append-only presentation: per-event lines for every verdict-untrusted mode,
# plus hit_timeout. The live tally has no such notion and shows every bucket.
_NOTEWORTHY_FAILURES = VERDICT_UNTRUSTED_FAILURE_MODES | {"hit_timeout"}
_MODE_ORDER = tuple(mode for mode in get_args(FailureMode) if mode != "solved")

_BAR_WIDTH = 26
_INDENT = "  "
_GREEN = "\x1b[32m"
_RED = "\x1b[31m"
_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"
_RESET = "\x1b[0m"


def _stage(name: str) -> str:
    """Aligned stage column: pads short names, keeps a separator for long ones."""
    return f"{name:<9} · "


class _LiveRegion:
    """Erasable screen lines below the scrollback: settled ones, then a live tail.

    Only fed on a TTY; off one it stays empty and ``print_above`` is a bare
    print. The other operations stay behind a ``live`` check at their call sites
    because the two modes differ in what they say, not just how they draw it.
    """

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._tail = 0  # trailing lines the current activity rewrites in place

    def _erase(self, count: int) -> None:
        if count:
            sys.stdout.write(f"\x1b[{count}A\x1b[0J")
            sys.stdout.flush()
            del self._lines[-count:]

    def _draw(self, lines: Sequence[str]) -> None:
        for line in lines:
            self._lines.append(line)
            print(line, flush=True)

    def set_active(self, lines: Sequence[str]) -> None:
        self._erase(self._tail)
        self._draw(lines)
        self._tail = len(lines)

    def refresh_active(self, lines: Sequence[str]) -> None:
        # Post-settle refreshes would append a stale copy below the settled line.
        if self._tail:
            self.set_active(lines)

    def settle(self, lines: Sequence[str]) -> None:
        self._erase(self._tail)
        self._tail = 0
        self._draw(lines)

    def print_above(self, line: str) -> None:
        held = list(self._lines)
        self._erase(len(held))
        print(line, flush=True)
        self._draw(held)

    def collapse(self, line: str) -> None:
        self._erase(len(self._lines))
        self._tail = 0
        print(line, flush=True)


@dataclass
class _Measurement:
    """One in-flight experiment. Every tally derives from ``results``."""

    subject: str
    task_ids: tuple[str, ...]
    baseline: ExperimentResult | None
    started_at: float
    experiment_id: str | None = None
    snapshot_printed: bool = False
    results: dict[str, str] = field(default_factory=dict)
    baseline_solved: frozenset[str] = field(init=False)
    last_output_at: float = field(init=False)

    def __post_init__(self) -> None:
        self.baseline_solved = (
            frozenset() if self.baseline is None else solved_task_ids(self.baseline)
        )
        self.last_output_at = self.started_at

    @property
    def solved(self) -> int:
        return sum(mode == "solved" for mode in self.results.values())

    @property
    def new_solves(self) -> int:
        if self.baseline is None:
            return 0
        return sum(
            mode == "solved" and task_id not in self.baseline_solved
            for task_id, mode in self.results.items()
        )

    @property
    def regressions(self) -> int:
        return sum(
            mode != "solved" and task_id in self.baseline_solved
            for task_id, mode in self.results.items()
        )

    @property
    def failure_counts(self) -> Counter[str]:
        return Counter(mode for mode in self.results.values() if mode != "solved")

    @property
    def comparison(self) -> str:
        if self.baseline is None:
            return ""
        return f" · +{self.new_solves} new · {self.regressions} regr"

    @property
    def exclusion_text(self) -> str | None:
        # Certification prunes nondeterministic tasks, so the panels can differ.
        # Abbreviated: the preceding "determinism:" log line spells out the cause.
        if self.baseline is None:
            return None
        excluded = len(self.baseline.tasks) - len(self.task_ids)
        return f"{excluded} excluded (nondet)" if excluded else None


class StdoutLogger:
    """Event-driven presentation; ``live`` selects in-place TTY rendering."""

    def __init__(
        self,
        *,
        show_task_progress: bool = False,
        live: bool | None = None,
    ) -> None:
        self._show_task_progress = show_task_progress
        # None = auto: live rendering only when stdout is a TTY.
        self._live = sys.stdout.isatty() if live is None else live
        self._region = _LiveRegion()
        self._run_started_at: float | None = None
        self._initial_score: tuple[int, int] | None = None
        self._epoch_label: str | None = None
        self._epoch_started_at = 0.0
        self._decision: ResultDecision | None = None
        self._skip_cause: str | None = None
        self._baseline_score: tuple[int, int] | None = None
        self._candidate_score: tuple[int, int] | None = None
        self._measurement: _Measurement | None = None

    @property
    def _active(self) -> _Measurement:
        # Measurement-scoped events only; None here means a lifecycle bug.
        assert self._measurement is not None
        return self._measurement

    def _style(self, code: str, text: str) -> str:
        return f"{code}{text}{_RESET}" if self._live else text

    def _emit(self, stage: str, detail: str) -> None:
        self._region.print_above(f"{_stage(stage)}{detail}")
        if stage == "measure" and self._measurement is not None:
            self._measurement.last_output_at = time.monotonic()

    def log(self, line: str) -> None:
        # Determinism arrives as plain log lines; omit zero-exclusion noise.
        if line.startswith("determinism:"):
            if "excluded 0/" in line:
                return
            self._emit("measure", line)
            return
        self._region.print_above(line)

    def run_started(self, title: str, rows: Sequence[tuple[str, str]]) -> None:
        """Orientation banner: what exactly is about to run."""
        self._run_started_at = time.monotonic()
        self._region.print_above(title)
        for label, detail in rows:
            self._region.print_above(self._style(_DIM, f"  {label:<9} {detail}"))

    def epoch_started(
        self,
        epoch: int,
        total_epochs: int,
        baseline: ExperimentResult | None,
    ) -> None:
        self._epoch_label = f"{epoch}/{total_epochs}"
        self._epoch_started_at = time.monotonic()
        self._decision = None
        self._skip_cause = None
        self._baseline_score = self._candidate_score = None
        self._measurement = None
        header = f"epoch {self._epoch_label}"
        if baseline is not None:
            solved, total = _score(baseline)
            if self._initial_score is None:
                self._initial_score = (solved, total)
            header += f" · baseline solved {solved}/{total}"
        if self._live:
            self._region.settle([header])
        else:
            self._region.print_above(header)

    def epoch_finished(self, outcome: str) -> None:
        assert self._epoch_label is not None
        elapsed = _format_duration(time.monotonic() - self._epoch_started_at)
        line = self._epoch_summary(outcome, elapsed) + self._infra_suffix()
        if self._live:
            self._region.collapse(line)
        else:
            self._region.print_above(line)
        self._epoch_label = None

    def _epoch_summary(self, outcome: str, elapsed: str) -> str:
        promoted = outcome == "promoted"
        parts = [f"epoch {self._epoch_label}"]
        if promoted:
            parts.append(self._style(_GREEN + _BOLD, "PROMOTED"))
            if self._baseline_score is not None and self._candidate_score is not None:
                solved, total = self._candidate_score
                movement = f"{self._baseline_score[0]} → {solved}/{total}"
                if self._decision is not None and self._decision.new_solves:
                    names = ", ".join(f"+{t}" for t in self._decision.new_solves)
                    movement += f" ({names})"
                parts.append(movement)
        else:
            parts.append(outcome.replace("_", " "))
            if self._decision is not None:
                parts.append(_decision_cause(self._decision))
            elif self._skip_cause is not None:
                parts.append(self._skip_cause)
        if (
            self._measurement is not None
            and self._measurement.experiment_id is not None
        ):
            parts.append(self._measurement.experiment_id)
        parts.append(elapsed)
        line = " · ".join(parts)
        return line if promoted else self._style(_DIM, line)

    def _infra_suffix(self) -> str:
        if self._measurement is None:
            return ""
        counts = self._measurement.failure_counts
        infra = " · ".join(
            f"{mode} {counts[mode]}"
            for mode in _MODE_ORDER
            if mode in UNSCORABLE_FAILURE_MODES and counts[mode]
        )
        return f" · {self._style(_RED, infra)}" if infra else ""

    def measurement_started(
        self,
        task_ids: tuple[str, ...],
        baseline: ExperimentResult | None,
        *,
        subject: str,
    ) -> None:
        self._measurement = _Measurement(
            subject=subject,
            task_ids=task_ids,
            baseline=baseline,
            started_at=time.monotonic(),
        )
        self._baseline_score = None if baseline is None else _score(baseline)
        if self._live:
            # A retried measurement abandons its predecessor's live lines.
            self._region.set_active(self._measurement_lines())

    def _measurement_lines(self) -> list[str]:
        measurement = self._active
        indent = _INDENT if measurement.subject == "candidate" else ""
        done, total = len(measurement.results), len(measurement.task_ids)
        filled = round(done / total * _BAR_WIDTH) if total else _BAR_WIDTH
        bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
        elapsed = _format_duration(time.monotonic() - measurement.started_at)
        parts = [f"solved {measurement.solved}"]
        if measurement.baseline is not None:
            new = f"+{measurement.new_solves} new"
            regr = f"{measurement.regressions} regr"
            if measurement.new_solves:
                new = self._style(_GREEN, new)
            if measurement.regressions:
                regr = self._style(_RED, regr)
            parts += [new, regr]
        counts = measurement.failure_counts
        parts += [f"{mode} {counts[mode]}" for mode in _MODE_ORDER if counts[mode]]
        gutter = indent + " " * 12
        lines = [
            f"{indent}{_stage(measurement.subject)}{bar}  {done:>2}/{total} done · {elapsed}"
        ]
        # inline overflows 80 cols, and a wrap breaks the region's line accounting.
        trace = []
        if measurement.experiment_id is not None:
            trace.append(measurement.experiment_id.removeprefix("exp-"))
        if (excluded := measurement.exclusion_text) is not None:
            trace.append(excluded)
        if trace:
            lines.append(gutter + self._style(_DIM, " · ".join(trace)))
        lines.append(gutter + self._style(_DIM, " · ".join(parts)))
        return lines

    def experiment_started(self, experiment_id: str) -> None:
        measurement = self._active
        measurement.experiment_id = experiment_id
        if self._live:
            # The id only arrives once the worker starts, after the bar first drew.
            self._region.refresh_active(self._measurement_lines())
            return
        self._emit(
            "measure",
            f"{measurement.subject} · {experiment_id} · "
            f"{len(measurement.task_ids)} tasks",
        )

    def task_finished(self, task_id: str, failure_mode: str) -> None:
        measurement = self._active
        measurement.results[task_id] = failure_mode
        if self._live:
            self._region.set_active(self._measurement_lines())
            return
        baseline_solved = task_id in measurement.baseline_solved
        candidate_solved = failure_mode == "solved"
        if measurement.baseline is None:
            total = len(measurement.task_ids)
            if self._show_task_progress:
                self._emit(
                    "measure",
                    f"{len(measurement.results)}/{total} complete · "
                    f"solved {measurement.solved}/{total} · "
                    f"{task_id} · {failure_mode}",
                )
            elif failure_mode in _NOTEWORTHY_FAILURES:
                self._emit("measure", f"{task_id} · {failure_mode}")
        elif candidate_solved and not baseline_solved:
            self._emit("measure", f"NEW SOLVE · {task_id}")
        elif baseline_solved and not candidate_solved:
            self._emit("measure", f"REGRESSION · {task_id} ({failure_mode})")
        elif failure_mode in _NOTEWORTHY_FAILURES:
            self._emit("measure", f"{task_id} · {failure_mode}")

    def measurement_heartbeat(self) -> None:
        measurement = self._active
        if self._live:
            # Keep the live block's elapsed clock ticking.
            self._region.refresh_active(self._measurement_lines())
            return
        now = time.monotonic()
        interval = (
            _SNAPSHOT_INTERVAL_SEC
            if measurement.snapshot_printed
            else _INITIAL_SNAPSHOT_SILENCE_SEC
        )
        if now - measurement.last_output_at < interval:
            return
        detail = f"{len(measurement.results)}/{len(measurement.task_ids)} complete"
        detail += measurement.comparison
        detail += f" · {_format_duration(now - measurement.started_at)}"
        remaining = [
            task for task in measurement.task_ids if task not in measurement.results
        ]
        if 0 < len(remaining) <= 3:
            detail += f" · waiting: {', '.join(remaining)}"
        self._emit("measure", detail)
        measurement.snapshot_printed = True

    def experiment_finished(self, result: ExperimentResult) -> None:
        measurement = self._active
        solved, total = _score(result)
        line = f"{_stage(measurement.subject)}solved {solved}/{total}"
        line += measurement.comparison
        if (excluded := measurement.exclusion_text) is not None:
            line += f" · {excluded}"
        line += f" · {_format_duration(_elapsed_since(result.started_at, result.finished_at))}"
        line += self._infra_suffix()
        if measurement.subject == "candidate":
            self._candidate_score = (solved, total)
        elif self._initial_score is None:
            # The first freshly-measured baseline is the run's starting score;
            # a cached baseline is captured by epoch_started instead.
            self._initial_score = (solved, total)
        if not self._live:
            self._region.print_above(line)
        elif measurement.subject == "candidate":
            # Stays in the epoch block until the collapse replaces it.
            self._region.settle([_INDENT + line])
        else:
            self._region.settle([])
            self._region.print_above(line)

    def experiment_failed(self, error: BaseException) -> None:
        message = str(error).partition("\n")[0]
        if self._live:
            self._region.settle([])
        self._region.print_above(f"{_stage(self._active.subject)}FAILED — {message}")

    def measurement_retrying(self, reason: str) -> None:
        """A candidate re-measurement follows a transient measurement failure."""
        self._emit("measure", f"retry · {reason}")

    def agent_progress(self, stage: str, line: str) -> None:
        """One in-flight agent turn line; ``running`` lines rewrite in place."""
        text = f"{_stage(stage)}{line}"
        if self._live and line.startswith("running ·"):
            self._region.set_active([_INDENT + text])
        elif self._live and line.startswith("done ·"):
            self._region.settle([_INDENT + text])
        else:
            # Raw agent-CLI output is diagnostics; keep it in scrollback.
            self._region.print_above(text)

    def decision_finished(self, decision: ResultDecision) -> None:
        # Silent: the epoch collapse line carries the outcome and cause.
        self._decision = decision

    def epoch_skipped(self, cause: str) -> None:
        self._skip_cause = cause.replace("_", " ")

    def loop_finished(
        self,
        epochs_run: int,
        decisions: list[ResultDecision],
        baseline: ExperimentResult | None,
    ) -> None:
        if baseline is None:
            epoch_word = "epoch" if epochs_run == 1 else "epochs"
            self._region.print_above(
                f"{_stage('train')}done · {epochs_run} {epoch_word} · baseline unavailable"
            )
            return
        promoted = sum(decision.outcome == "promoted" for decision in decisions)
        rejected = sum(decision.outcome == "rejected" for decision in decisions)
        skipped = epochs_run - len(decisions)
        outcomes = f"{promoted} promoted · {rejected} rejected"
        if skipped:
            outcomes += f" · {skipped} skipped"
        initial = self._initial_score or _score(baseline)
        final = _score(baseline)
        movement = self._style(_BOLD, f"{initial[0]} → {final[0]}/{final[1]}")
        detail = f"done · {movement} · {outcomes}"
        if self._run_started_at is not None:
            detail += f" · {_format_duration(time.monotonic() - self._run_started_at)}"
        self._region.print_above(f"{_stage('train')}{detail}")


def _decision_cause(decision: ResultDecision) -> str:
    detail = {
        "regressed_baseline_tasks": ("regressed", decision.regressions),
        "strict_improvement_without_regression": ("new solves", decision.new_solves),
        "infra_sensitive_verdict": ("invalid infra", decision.invalid_infra_tasks),
    }.get(decision.reason)
    if detail is not None and detail[1]:
        return f"{detail[0]} {', '.join(detail[1])}"
    return decision.reason.replace("_", " ")


def suite_summary(results: Sequence[ExperimentResult]) -> None:
    """One table row per finished run, printed after a multi-run invocation."""
    if not results:
        return
    starts = [r.started_at for r in results if r.started_at is not None]
    ends = [r.finished_at for r in results if r.finished_at is not None]
    wall = _elapsed_since(min(starts), max(ends)) if starts and ends else None
    print()
    print(f"{_stage('summary')}{len(results)} runs · {_format_duration(wall)}")
    width = max(len(r.experiment_id) for r in results)
    for result in results:
        solved, total = _score(result)
        duration = _format_duration(
            _elapsed_since(result.started_at, result.finished_at)
        )
        counts = Counter(
            rollout.failure_mode
            for rollout in result.tasks.values()
            if rollout is not None and rollout.failure_mode != "solved"
        )
        failures = [f"{mode} {count}" for mode, count in counts.most_common()]
        unfinished = sum(rollout is None for rollout in result.tasks.values())
        if unfinished:
            failures.append(f"unfinished {unfinished}")
        if result.crash_reason is not None:
            failures.insert(0, "CRASHED")
        print(
            f"  {result.experiment_id:<{width}}  {solved:>3}/{total:<3}"
            f"  {duration:>7}  {' · '.join(failures) if failures else '-'}"
        )


def _score(result: ExperimentResult) -> tuple[int, int]:
    return len(solved_task_ids(result)), len(result.tasks)


def _elapsed_since(
    started_at: datetime | None,
    finished_at: datetime | None,
) -> float | None:
    if started_at is None:
        return None
    finished = finished_at if finished_at is not None else datetime.now(UTC)
    return max(0.0, (finished - started_at).total_seconds())


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    rounded = int(seconds)
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
