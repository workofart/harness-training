from __future__ import annotations

import asyncio
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.experiment.runner as runner
from src.experiment.record import (
    ExperimentRecord,
    ExperimentState,
    TaskTrials,
    failed_experiment_git_ref,
)
from src.harness.contracts import TaskResult
from src.metrics import TaskMetrics

from conftest import _task_result


@dataclass
class FakeHarnessConfig:
    """Duck-typed stand-in for the pydantic HarnessConfig.

    ExperimentRunner only reads attributes off its config (it imports the
    real type under TYPE_CHECKING), so a plain dataclass is enough and keeps
    these tests free of the real config's construction requirements.
    """

    experiment_id: str
    focus_name: str
    train_task_names: list[str]
    max_steps: int = 30
    max_trial_concurrency: int = 1
    max_heavy_action_concurrency: int = 10
    task_timeout_sec: float = 600.0
    env_setup_timeout_sec: float = 600.0
    max_output_retries: int = 2
    max_disallowed_retries: int = 2
    task_trials: int = 1
    llm_provider_config: object | None = None


def _set_task_dirs(experiment_runner, root: Path, *task_names: str) -> None:
    experiment_runner._task_dirs = {
        task_name: root / "tasks" / task_name for task_name in task_names
    }


def _run_panel_for_experiment_runner(
    runner, experiment_runner, task_names: list[str]
) -> None:
    asyncio.run(
        runner._run_panel(
            record=experiment_runner.record,
            experiments_root=experiment_runner.experiments_root,
            task_names=task_names,
            task_dirs=experiment_runner._task_dirs,
            harness_config=experiment_runner.harness_config,
            make_llm=experiment_runner._make_llm,
            make_env=experiment_runner._make_env,
        )
    )


def make_runner(
    monkeypatch,
    tmp_path: Path,
    *,
    task_dirs: list[str] | None = None,
    harbor_config: object | None = None,
    **config_kwargs,
) -> runner.ExperimentRunner:
    """Construct an ExperimentRunner wired for unit tests.

    Patches the clean-worktree / head-commit git probes, builds the runner
    against a duck-typed FakeHarnessConfig (``config_kwargs`` flow straight
    through) and a minimal harbor_config, and optionally pre-resolves task
    directories under ``tmp_path/tasks``. Tests that drive trials follow up
    with ``stub_trial_factories``.
    """
    monkeypatch.setattr(runner.control_repo, "require_clean_worktree", lambda: None)
    monkeypatch.setattr(runner.control_repo, "get_head_commit", lambda: "abc")
    experiment_runner = runner.ExperimentRunner(
        harness_config=FakeHarnessConfig(**config_kwargs),
        harbor_config=harbor_config
        if harbor_config is not None
        else SimpleNamespace(experiments_dir=tmp_path / "experiments"),
        api_key="key",
    )
    if task_dirs is not None:
        _set_task_dirs(experiment_runner, tmp_path, *task_dirs)
    return experiment_runner


def stub_trial_factories(monkeypatch, experiment_runner) -> None:
    """Replace the per-trial llm/env factories with no-op ``object()`` stubs so
    a test can monkeypatch ``runner.run_task`` directly."""
    monkeypatch.setattr(experiment_runner, "_make_llm", lambda: object(), raising=False)
    monkeypatch.setattr(
        experiment_runner,
        "_make_env",
        lambda task_name, *, task_dir, exec_semaphore=None: object(),
        raising=False,
    )


def _panel_trial_result(
    task_name: str,
    *,
    solved: bool,
    index: int,
    failure_mode: str | None = None,
) -> TaskResult:
    """A finished TaskResult with the boilerplate fields the panel tests share;
    only ``solved`` (and optional ``failure_mode``) carry behavior."""
    return TaskResult(
        task_name=task_name,
        reward=1.0 if solved else 0.0,
        solved=solved,
        error=None,
        steps_used=1,
        trial_dir=f"/tmp/{task_name}/{index}",
        trace_path=None,
        metrics_path=None,
        verifier_stdout_path=None,
        metrics=TaskMetrics(failure_mode=failure_mode),
        started_at="2026-05-05T00:00:00+00:00",
        finished_at="2026-05-05T00:00:01+00:00",
    )


def test_make_llm_for_config_builds_chatgpt_adapter(monkeypatch):
    fake_chatgpt_module = types.ModuleType("src.adapters.chatgpt_codex")
    calls: dict[str, object] = {}

    class FakeChatGptCodex:
        def __init__(self, *, config):
            calls["config"] = config

    fake_chatgpt_module.ChatGptCodex = FakeChatGptCodex
    monkeypatch.setitem(sys.modules, "src.adapters.chatgpt_codex", fake_chatgpt_module)
    config = types.SimpleNamespace(provider="chatgpt_codex")

    llm = runner._make_llm_for_config(config=config, api_key=None)

    assert isinstance(llm, FakeChatGptCodex)
    assert calls["config"] is config


class _FakeHarborConfig(SimpleNamespace):
    def model_copy(self, *, update):
        return _FakeHarborConfig(**{**self.__dict__, **update})


def _stub_baseline_task_environment(monkeypatch, tmp_path: Path) -> None:
    from src.adapters import env as adapter_env

    class FakeTaskDirectoryResolver:
        def __init__(self, config):
            self.config = config

        def resolve(self, task_names):
            return {
                task_name: tmp_path / "tasks" / task_name for task_name in task_names
            }

    monkeypatch.setattr(adapter_env, "TaskDirectoryResolver", FakeTaskDirectoryResolver)
    monkeypatch.setattr(adapter_env, "Harbor", lambda *args, **kwargs: object())


def _write_baseline_record(
    runner,
    experiments_root: Path,
    *,
    experiment_id: str = "baseline",
    git_commit_hash: str = "base123",
    task_ids: list[str] | None = None,
) -> TaskResult:
    resolved_task_ids = ["train-a"] if task_ids is None else task_ids
    baseline = ExperimentRecord.initialize(
        experiment_id=experiment_id,
        git_commit_hash=git_commit_hash,
        parent_baseline_experiment_id=None,
        train_task_ids=resolved_task_ids,
        focus_name="focus",
        started_at="2026-04-10T00:00:00+00:00",
    )
    for task_id in resolved_task_ids:
        baseline.record_task_result(
            TaskResult(
                task_name=task_id,
                reward=1.0,
                solved=True,
                steps_used=1,
                error=None,
                trial_dir=f"/tmp/baseline/{task_id}",
                started_at="2026-04-10T00:00:00+00:00",
                finished_at="2026-04-10T00:00:01+00:00",
            )
        )
    baseline.finalize(status="keep", decision_reason="seed baseline")
    baseline.write(root=experiments_root)
    return baseline


def test_schedule_order_sorts_by_descending_duration_prior():
    duration_priors = {"short": 1, "long": 100, "mid": 10}

    order = runner._schedule_order(["short", "long", "mid"], duration_priors)

    assert order == ["long", "mid", "short"]


def test_schedule_order_falls_back_to_config_order_without_priors():
    assert runner._schedule_order(["a", "b", "c"], {}) == ["a", "b", "c"]


def test_schedule_order_keeps_config_order_for_equal_durations():
    # Stable sort: equal-cost ties (and tasks absent from the prior) keep
    # their config order rather than shuffling.
    duration_priors = {"a": 5, "b": 5}

    assert runner._schedule_order(["a", "b", "unseen"], duration_priors) == [
        "a",
        "b",
        "unseen",
    ]


def test_load_task_duration_priors(tmp_path):
    path = tmp_path / "task_duration_priors.json"
    path.write_text(
        json.dumps(
            {
                "task_duration_seconds": {
                    "slow": 12,
                    "fast": 1.5,
                }
            }
        )
    )

    assert runner._load_task_duration_priors(path) == {
        "slow": 12.0,
        "fast": 1.5,
    }


@pytest.mark.parametrize(
    ("solved", "finished", "expected_total", "admission_count"),
    [
        (0, 0, 5, 3),
        (2, 3, 5, 1),
        (2, 4, 5, 1),
        (3, 3, 5, 0),
        (0, 1, 5, 2),
        (0, 0, 3, 2),
        (1, 2, 3, 1),
        (2, 2, 3, 0),
    ],
)
def test_next_trial_admission_count(solved, finished, expected_total, admission_count):
    assert (
        runner._next_trial_admission_count(
            solved=solved,
            finished=finished,
            expected_total=expected_total,
        )
        == admission_count
    )


def _budget_trials(*, expected, solved=0, failed=0, crashed=0):
    # A TaskTrials with `solved`+`failed` valid (error-free) trials and `crashed`
    # infra-error trials. Crashes count toward finished_trials (the budget) but
    # are excluded from valid_trials/solved_count.
    trials = TaskTrials(task_name="t", expected_trial_count=expected, trials=[])
    for _ in range(solved):
        trials.append(_task_result(task_name="t", reward=1.0))
    for _ in range(failed):
        trials.append(_task_result(task_name="t", reward=0.0))
    for _ in range(crashed):
        trials.append(_task_result(task_name="t", reward=0.0, error="boom"))
    return trials


def test_wants_confirmation_expand_only_for_failed_single_deterministic_trial():
    # The single confirm-on-fail predicate that both the admission loop (which
    # commits the expand) and the priority gate (which reads it speculatively)
    # consult. True iff a single-trial deterministic budget produced exactly one
    # valid failure and the full count exceeds one.
    one_fail = _budget_trials(expected=1, failed=1)
    assert runner._wants_confirmation_expand(one_fail, full_trial_count=3) is True
    # Nothing to expand to when the full count is one.
    assert runner._wants_confirmation_expand(one_fail, full_trial_count=1) is False
    # A passing single trial is deterministic-solved, not a regression.
    assert (
        runner._wants_confirmation_expand(
            _budget_trials(expected=1, solved=1), full_trial_count=3
        )
        is False
    )
    # An already-expanded budget (expected != 1) is no longer the single case.
    assert (
        runner._wants_confirmation_expand(
            _budget_trials(expected=3, failed=1), full_trial_count=3
        )
        is False
    )
    # A crash is not a valid trial: one crash is not one valid failure.
    assert (
        runner._wants_confirmation_expand(
            _budget_trials(expected=1, crashed=1), full_trial_count=3
        )
        is False
    )


@pytest.mark.parametrize(
    ("expected", "solved", "failed", "crashed", "want"),
    [
        # Fresh task: the majority-threshold rule sets the initial admission.
        (5, 0, 0, 0, 3),
        # Crashes consume the budget: remaining caps admission below the
        # threshold count, which is itself blind to crashes.
        (5, 0, 0, 4, 1),
        # Budget fully consumed by a crash -> nothing left to admit.
        (1, 0, 0, 1, 0),
        # Majority already decided -> threshold rule yields zero.
        (3, 2, 0, 0, 0),
    ],
)
def test_planned_admission_count(expected, solved, failed, crashed, want):
    # The shared admission arithmetic: remaining budget capped by the
    # majority-threshold rule, consulted by both the loop and the priority gate.
    trials = _budget_trials(
        expected=expected, solved=solved, failed=failed, crashed=crashed
    )
    assert runner._planned_admission_count(trials, full_trial_count=expected) == want


def test_planned_admission_count_honors_pending_confirmation_expand():
    # A failed single deterministic trial budgets toward the full count (the
    # expand the loop is about to commit), not the committed expected_trial_count
    # of 1. This is the _effective_expected_count seam: the caller passes only
    # config (full_trial_count), never a speculative override of the object's
    # count. The 1-vs-0 contrast is the whole point -- honoring the pending
    # expand admits one more (threshold-paced) trial; ignoring it admits none
    # because the committed single-trial budget is already spent.
    trials = _budget_trials(expected=1, failed=1)
    assert runner._planned_admission_count(trials, full_trial_count=3) == 1
    assert runner._planned_admission_count(trials, full_trial_count=1) == 0


def test_run_panel_early_stops_after_majority_decided_when_trials_agree(
    monkeypatch, tmp_path
):
    experiment_runner = make_runner(
        monkeypatch,
        tmp_path,
        experiment_id="exp-trials",
        focus_name="focus",
        train_task_names=["task-a", "task-b"],
        task_trials=3,
        max_trial_concurrency=1,
        task_dirs=["task-a", "task-b"],
    )
    stub_trial_factories(monkeypatch, experiment_runner)

    call_log: list[str] = []

    async def fake_run_task(*, task_name, **_kwargs):
        # Yield to the event loop so the majority watcher in run_task_trials
        # gets a chance to observe each completion and cancel siblings.
        await asyncio.sleep(0)
        call_log.append(task_name)
        return _panel_trial_result(task_name, solved=True, index=len(call_log))

    monkeypatch.setattr(runner, "run_task", fake_run_task)

    _run_panel_for_experiment_runner(runner, experiment_runner, ["task-a", "task-b"])

    # With k=3 all-pass: trial 1 + trial 2 = majority decided True; trial 3
    # is cancelled by the majority watcher before it can record a result, and
    # expected_trial_count shrinks to len(finished_trials). Cancelled trials
    # leave no synthetic record entry.
    for task_id in ("task-a", "task-b"):
        trials = experiment_runner.record._task_trials(task_id)
        assert trials.trial_count == 2
        assert trials.expected_trial_count == 2
        assert trials.is_finished is True
        assert trials.majority_solved is True
    assert sorted(call_log) == ["task-a"] * 2 + ["task-b"] * 2


def test_run_panel_admits_only_majority_threshold_trials_initially(
    monkeypatch, tmp_path
):
    experiment_runner = make_runner(
        monkeypatch,
        tmp_path,
        experiment_id="exp-adaptive-admission",
        focus_name="focus",
        train_task_names=["task-a"],
        task_trials=5,
        max_trial_concurrency=5,
        task_dirs=["task-a"],
    )
    stub_trial_factories(monkeypatch, experiment_runner)

    started = 0
    release = asyncio.Event()

    async def fake_run_task(*, task_name, **_kwargs):
        nonlocal started
        started += 1
        trial_index = started
        await release.wait()
        return _panel_trial_result(task_name, solved=True, index=trial_index)

    monkeypatch.setattr(runner, "run_task", fake_run_task)

    async def go():
        panel_task = asyncio.create_task(
            runner._run_panel(
                record=experiment_runner.record,
                experiments_root=experiment_runner.experiments_root,
                task_names=["task-a"],
                task_dirs=experiment_runner._task_dirs,
                harness_config=experiment_runner.harness_config,
                make_llm=experiment_runner._make_llm,
                make_env=experiment_runner._make_env,
            )
        )

        async def wait_for_started(count):
            while started < count:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_started(3), timeout=1)
        for _ in range(5):
            await asyncio.sleep(0)
        assert started == 3
        release.set()
        await panel_task

    asyncio.run(go())

    trials = experiment_runner.record._task_trials("task-a")
    assert trials.trial_count == 3
    assert trials.expected_trial_count == 3
    assert trials.majority_solved is True


def test_run_panel_prioritizes_late_admissions_over_lower_priority_waiters(
    monkeypatch, tmp_path
):
    experiment_runner = make_runner(
        monkeypatch,
        tmp_path,
        experiment_id="exp-priority-admission",
        focus_name="focus",
        train_task_names=["high", "low"],
        task_trials=5,
        max_trial_concurrency=1,
        task_dirs=["high", "low"],
    )
    stub_trial_factories(monkeypatch, experiment_runner)

    call_log: list[str] = []
    high_outcomes = iter([True, False, False, False])

    async def fake_run_task(*, task_name, **_kwargs):
        await asyncio.sleep(0)
        call_log.append(task_name)
        solved = next(high_outcomes) if task_name == "high" else True
        return _panel_trial_result(task_name, solved=solved, index=len(call_log))

    monkeypatch.setattr(runner, "run_task", fake_run_task)

    _run_panel_for_experiment_runner(runner, experiment_runner, ["high", "low"])

    assert call_log[:4] == ["high", "high", "high", "high"]
    high_trials = experiment_runner.record._task_trials("high")
    assert high_trials.trial_count == 4
    assert high_trials.majority_solved is False


def test_run_panel_grants_low_priority_waiter_after_higher_priority_result_persists(
    monkeypatch, tmp_path
):
    # Regression for the early-release overlap. run_task hands the slot back
    # (slot_release) *before* its result is persisted, so the grant pass that
    # the release schedules evaluates the not-yet-updated record: "high" still
    # looks like it wants a slot, the gate reserves it and parks "low", and the
    # record is only updated afterwards in persist_task_result. Unless the gate
    # is signaled after that update, "low" is never granted. capacity=1 +
    # task_trials=1 makes "high" finishing the last gate event before the stall,
    # so there is no later acquire/release to mask the missed wakeup -- the
    # panel deadlocks, which wait_for surfaces as a failure.
    experiment_runner = make_runner(
        monkeypatch,
        tmp_path,
        experiment_id="exp-priority-wakeup",
        focus_name="focus",
        train_task_names=["high", "low"],
        task_trials=1,
        max_trial_concurrency=1,
        task_dirs=["high", "low"],
    )
    stub_trial_factories(monkeypatch, experiment_runner)

    call_log: list[str] = []

    async def fake_run_task(*, task_name, slot_release=None, **_kwargs):
        call_log.append(task_name)
        # Mirror run_task's overlap: free the slot, then suspend (the docker
        # teardown await) before returning, so the release-scheduled grant pass
        # runs against the record before persist_task_result records this trial.
        if slot_release is not None:
            slot_release()
        await asyncio.sleep(0)
        return _panel_trial_result(task_name, solved=True, index=len(call_log))

    monkeypatch.setattr(runner, "run_task", fake_run_task)

    async def go():
        await asyncio.wait_for(
            runner._run_panel(
                record=experiment_runner.record,
                experiments_root=experiment_runner.experiments_root,
                task_names=["high", "low"],
                task_dirs=experiment_runner._task_dirs,
                harness_config=experiment_runner.harness_config,
                make_llm=experiment_runner._make_llm,
                make_env=experiment_runner._make_env,
            ),
            timeout=2,
        )

    asyncio.run(go())

    assert call_log == ["high", "low"]
    for task_id in ("high", "low"):
        trials = experiment_runner.record._task_trials(task_id)
        assert trials.trial_count == 1
        assert trials.majority_solved is True


def test_run_panel_calls_run_task_with_current_contract(monkeypatch, tmp_path):
    experiment_runner = make_runner(
        monkeypatch,
        tmp_path,
        experiment_id="exp-run-task-contract",
        focus_name="focus",
        train_task_names=["task-a"],
        task_trials=1,
        max_trial_concurrency=1,
        task_dirs=["task-a"],
    )
    stub_trial_factories(monkeypatch, experiment_runner)

    call_log: list[str] = []

    async def fake_run_task(
        *,
        task_name,
        llm,
        env,
        max_steps,
        max_output_retries=2,
        task_timeout_sec=None,
        env_setup_timeout_sec=None,
        trace_path=None,
        slot_release=None,
    ):
        del (
            llm,
            env,
            max_steps,
            max_output_retries,
            task_timeout_sec,
            env_setup_timeout_sec,
            trace_path,
            slot_release,
        )
        call_log.append(task_name)
        return _panel_trial_result(task_name, solved=True, index=len(call_log))

    monkeypatch.setattr(runner, "run_task", fake_run_task)

    _run_panel_for_experiment_runner(runner, experiment_runner, ["task-a"])

    assert call_log == ["task-a"]
    trials = experiment_runner.record._task_trials("task-a")
    assert trials.trials[0].solved is True
    assert trials.trials[0].error is None


def test_run_panel_passes_task_timeout_sec_to_trial_boundary(monkeypatch, tmp_path):
    experiment_runner = make_runner(
        monkeypatch,
        tmp_path,
        experiment_id="exp-run-task-timeout",
        focus_name="focus",
        train_task_names=["task-a"],
        task_trials=1,
        max_trial_concurrency=1,
        task_timeout_sec=0.01,
        task_dirs=["task-a"],
    )
    stub_trial_factories(monkeypatch, experiment_runner)

    seen_timeouts: list[float | None] = []

    async def fake_run_task(
        *,
        task_name,
        llm,
        env,
        max_steps,
        max_output_retries=2,
        task_timeout_sec=None,
        env_setup_timeout_sec=None,
        trace_path=None,
        slot_release=None,
    ):
        del llm, env, max_steps, max_output_retries, trace_path, slot_release
        seen_timeouts.append(task_timeout_sec)
        return _panel_trial_result(
            task_name, solved=False, index=1, failure_mode="hit_timeout"
        )

    monkeypatch.setattr(runner, "run_task", fake_run_task)

    _run_panel_for_experiment_runner(runner, experiment_runner, ["task-a"])

    trials = experiment_runner.record._task_trials("task-a")
    assert seen_timeouts == [0.01]
    assert trials.trials[0].solved is False
    assert trials.trials[0].error is None
    assert trials.trials[0].metrics.failure_mode == "hit_timeout"


@pytest.mark.parametrize(
    ("solved_by_trial", "expected_calls", "expected_trials", "expected_majority"),
    [
        # First two trials split (pass, fail) -> undecided -> trial 3 runs.
        pytest.param([True, False, True], 3, 3, True, id="split-runs-all"),
        # First two trials fail -> decided False -> trial 3 cancelled.
        pytest.param([False, False, False], 2, 2, False, id="two-fail-early-stop"),
    ],
)
def test_run_panel_majority_decides_when_to_stop(
    monkeypatch,
    tmp_path,
    solved_by_trial,
    expected_calls,
    expected_trials,
    expected_majority,
):
    experiment_runner = make_runner(
        monkeypatch,
        tmp_path,
        experiment_id="exp-majority-stop",
        focus_name="focus",
        train_task_names=["task-a"],
        task_trials=3,
        max_trial_concurrency=1,
        task_dirs=["task-a"],
    )
    stub_trial_factories(monkeypatch, experiment_runner)

    call_log: list[str] = []

    async def fake_run_task(*, task_name, **_kwargs):
        # Yield so the majority watcher can observe each completion and cancel
        # the remaining trial once the outcome is decided.
        await asyncio.sleep(0)
        call_log.append(task_name)
        return _panel_trial_result(
            task_name, solved=solved_by_trial[len(call_log) - 1], index=len(call_log)
        )

    monkeypatch.setattr(runner, "run_task", fake_run_task)

    _run_panel_for_experiment_runner(runner, experiment_runner, ["task-a"])

    assert call_log == ["task-a"] * expected_calls
    trials = experiment_runner.record._task_trials("task-a")
    assert trials.trial_count == expected_trials
    assert trials.expected_trial_count == expected_trials
    assert trials.is_finished is True
    assert trials.majority_solved is expected_majority


def test_run_panel_requires_resolved_task_dir(monkeypatch, tmp_path):
    # No task_dirs are pre-resolved, so the panel must raise on the lookup.
    experiment_runner = make_runner(
        monkeypatch,
        tmp_path,
        experiment_id="exp-missing-task-dir",
        focus_name="focus",
        train_task_names=["task-a"],
        task_trials=1,
        max_trial_concurrency=1,
    )
    monkeypatch.setattr(experiment_runner, "_make_llm", lambda: object(), raising=False)

    with pytest.raises(KeyError, match="task-a"):
        _run_panel_for_experiment_runner(runner, experiment_runner, ["task-a"])


def test_run_experiment_runs_single_trial_for_deterministic_baseline_task(
    monkeypatch, tmp_path
):
    experiment_runner = make_runner(
        monkeypatch,
        tmp_path,
        experiment_id="exp-det",
        focus_name="focus",
        train_task_names=["train-det", "train-noise"],
        task_trials=3,
        max_trial_concurrency=2,
        task_dirs=["train-det", "train-noise"],
    )
    stub_trial_factories(monkeypatch, experiment_runner)

    baseline = ExperimentRecord.initialize(
        experiment_id="baseline",
        git_commit_hash="bca",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-det", "train-noise"],
        started_at="2026-04-10T00:00:00+00:00",
        expected_trial_count=3,
    )
    # train-det: 2/2 all-pass (deterministic-solved).
    for _ in range(2):
        baseline.record_task_result(_task_result(task_name="train-det", reward=1.0))
    baseline.train_task_results["train-det"].expected_trial_count = 2
    # train-noise: 2/3 pass (not deterministic).
    for solved in (True, False, True):
        baseline.record_task_result(
            _task_result(
                task_name="train-noise",
                reward=1.0 if solved else 0.0,
            )
        )

    call_log: list[str] = []
    counter = 0

    async def fake_run_task(*, task_name, **_kwargs):
        # Stagger completions so the majority watcher gets a turn between
        # trial completions and can cancel pending siblings under the
        # parallel trial scheduler.
        nonlocal counter
        counter += 1
        await asyncio.sleep(counter * 0.001)
        call_log.append(task_name)

        # All trials pass for both tasks; we want to see deterministic stop
        # at 1 and non-deterministic stop at 2 (via #3 early-stop).
        return _panel_trial_result(task_name, solved=True, index=len(call_log))

    monkeypatch.setattr(runner, "run_task", fake_run_task)
    monkeypatch.setattr(experiment_runner, "_conclude_experiment", lambda: None)

    asyncio.run(
        experiment_runner._run_experiment(
            baseline=baseline,
        )
    )

    # train-det runs 1 trial (k=1 budget from deterministic baseline, passes).
    # train-noise runs 2 trials (k=3 budget, early-stops on 2/2 pass via #3).
    assert sorted(call_log) == ["train-det"] + ["train-noise"] * 2
    det_trials = experiment_runner.record._task_trials("train-det")
    assert det_trials.expected_trial_count == 1
    assert det_trials.trial_count == 1
    assert det_trials.majority_solved is True
    noise_trials = experiment_runner.record._task_trials("train-noise")
    assert noise_trials.expected_trial_count == 2
    assert noise_trials.trial_count == 2
    assert noise_trials.majority_solved is True


def test_run_experiment_confirms_on_fail_when_deterministic_trial_fails(
    monkeypatch, tmp_path
):
    experiment_runner = make_runner(
        monkeypatch,
        tmp_path,
        experiment_id="exp-confirm",
        focus_name="focus",
        train_task_names=["train-det"],
        task_trials=3,
        max_trial_concurrency=1,
        task_dirs=["train-det"],
    )
    stub_trial_factories(monkeypatch, experiment_runner)

    baseline = ExperimentRecord.initialize(
        experiment_id="baseline",
        git_commit_hash="bca",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-det"],
        started_at="2026-04-10T00:00:00+00:00",
        expected_trial_count=3,
    )
    for _ in range(2):
        baseline.record_task_result(_task_result(task_name="train-det", reward=1.0))
    baseline.train_task_results["train-det"].expected_trial_count = 2

    call_log: list[str] = []
    counter = 0

    async def fake_run_task(*, task_name, **_kwargs):
        # Stagger completions so the majority watcher gets a turn between
        # trial completions and can cancel pending siblings under the
        # parallel trial scheduler.
        nonlocal counter
        counter += 1
        await asyncio.sleep(counter * 0.001)
        call_log.append(task_name)

        # All trials fail to simulate a real regression. Trial 1 fails →
        # confirm-on-fail expands to k=3; trial 2 fails → majority decided
        # False after 2 trials (via #3 early-stop).
        return _panel_trial_result(task_name, solved=False, index=len(call_log))

    monkeypatch.setattr(runner, "run_task", fake_run_task)
    monkeypatch.setattr(experiment_runner, "_conclude_experiment", lambda: None)

    asyncio.run(
        experiment_runner._run_experiment(
            baseline=baseline,
        )
    )

    assert call_log == ["train-det", "train-det"]
    trials = experiment_runner.record._task_trials("train-det")
    assert trials.expected_trial_count == 2
    assert trials.trial_count == 2
    assert trials.majority_solved is False


def test_run_experiment_launches_deterministic_confirmations_in_one_wave(
    monkeypatch, tmp_path
):
    experiment_runner = make_runner(
        monkeypatch,
        tmp_path,
        experiment_id="exp-confirm-wave",
        focus_name="focus",
        train_task_names=["train-det"],
        task_trials=5,
        max_trial_concurrency=5,
        task_dirs=["train-det"],
    )
    stub_trial_factories(monkeypatch, experiment_runner)

    baseline = ExperimentRecord.initialize(
        experiment_id="baseline",
        git_commit_hash="bca",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-det"],
        started_at="2026-04-10T00:00:00+00:00",
        expected_trial_count=3,
    )
    for _ in range(3):
        baseline.record_task_result(_task_result(task_name="train-det", reward=1.0))

    started = 0
    release_confirmations = asyncio.Event()

    async def fake_run_task(*, task_name, **_kwargs):
        nonlocal started
        started += 1
        trial_index = started
        if trial_index == 1:
            await asyncio.sleep(0)
            return _panel_trial_result(task_name, solved=False, index=trial_index)
        await release_confirmations.wait()
        return _panel_trial_result(task_name, solved=True, index=trial_index)

    monkeypatch.setattr(runner, "run_task", fake_run_task)
    monkeypatch.setattr(experiment_runner, "_conclude_experiment", lambda: None)

    async def go():
        panel_task = asyncio.create_task(
            experiment_runner._run_experiment(baseline=baseline)
        )

        async def wait_for_started(count):
            while started < count:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_started(5), timeout=1)
        release_confirmations.set()
        await panel_task

    asyncio.run(go())

    trials = experiment_runner.record._task_trials("train-det")
    assert started == 5
    assert trials.majority_solved is True


def test_run_experiment_prioritizes_late_confirmations_by_duration_prior(
    monkeypatch, tmp_path
):
    experiment_runner = make_runner(
        monkeypatch,
        tmp_path,
        experiment_id="exp-confirm-priority",
        focus_name="focus",
        train_task_names=["short", "long"],
        task_trials=3,
        max_trial_concurrency=1,
        task_dirs=["short", "long"],
    )
    stub_trial_factories(monkeypatch, experiment_runner)
    monkeypatch.setattr(
        runner, "_load_task_duration_priors", lambda: {"long": 100.0, "short": 1.0}
    )

    baseline = ExperimentRecord.initialize(
        experiment_id="baseline",
        git_commit_hash="bca",
        parent_baseline_experiment_id=None,
        train_task_ids=["short", "long"],
        started_at="2026-04-10T00:00:00+00:00",
        expected_trial_count=3,
    )
    for task_id in ("short", "long"):
        for _ in range(3):
            baseline.record_task_result(_task_result(task_name=task_id, reward=1.0))

    call_log: list[str] = []

    async def fake_run_task(*, task_name, **_kwargs):
        call_log.append(task_name)
        return _panel_trial_result(task_name, solved=False, index=len(call_log))

    monkeypatch.setattr(runner, "run_task", fake_run_task)
    monkeypatch.setattr(experiment_runner, "_conclude_experiment", lambda: None)

    asyncio.run(experiment_runner._run_experiment(baseline=baseline))

    assert call_log[:2] == ["long", "long"]


def test_apply_baseline_derived_trial_counts_skips_non_deterministic(
    monkeypatch, tmp_path
):
    experiment_runner = make_runner(
        monkeypatch,
        tmp_path,
        experiment_id="exp-mixed",
        focus_name="focus",
        train_task_names=["task-a", "task-b", "task-c"],
        task_trials=3,
        max_trial_concurrency=2,
    )

    baseline = ExperimentRecord.initialize(
        experiment_id="baseline",
        git_commit_hash="bca",
        parent_baseline_experiment_id=None,
        train_task_ids=["task-a", "task-b", "task-c"],
        started_at="2026-04-10T00:00:00+00:00",
        expected_trial_count=3,
    )
    # task-a: all-pass deterministic at baseline.
    for _ in range(3):
        baseline.record_task_result(_task_result(task_name="task-a", reward=1.0))
    # task-b: mixed (1 pass, 2 fail) — not deterministic.
    baseline.record_task_result(_task_result(task_name="task-b", reward=1.0))
    for _ in range(2):
        baseline.record_task_result(_task_result(task_name="task-b", reward=0.0))
    # task-c: all-pass deterministic.
    for _ in range(2):
        baseline.record_task_result(_task_result(task_name="task-c", reward=1.0))
    baseline.train_task_results["task-c"].expected_trial_count = 2

    experiment_runner._apply_baseline_derived_trial_counts(baseline)

    assert experiment_runner.record._task_trials("task-a").expected_trial_count == 1
    assert experiment_runner.record._task_trials("task-b").expected_trial_count == 3
    assert experiment_runner.record._task_trials("task-c").expected_trial_count == 1


def test_run_experiment_runs_train_when_baseline_absent(monkeypatch, tmp_path):
    experiment_runner = make_runner(
        monkeypatch,
        tmp_path,
        experiment_id="exp-no-baseline",
        focus_name="focus",
        train_task_names=["train-a"],
        task_trials=3,
        max_trial_concurrency=2,
        task_dirs=["train-a"],
    )
    stub_trial_factories(monkeypatch, experiment_runner)

    call_log: list[str] = []
    counter = 0

    async def fake_run_task(*, task_name, **_kwargs):
        # Stagger completions so the majority watcher gets a turn between
        # trial completions and can cancel pending siblings under the
        # parallel trial scheduler.
        nonlocal counter
        counter += 1
        await asyncio.sleep(counter * 0.001)
        call_log.append(task_name)
        return _panel_trial_result(task_name, solved=True, index=len(call_log))

    monkeypatch.setattr(runner, "run_task", fake_run_task)
    monkeypatch.setattr(experiment_runner, "_conclude_experiment", lambda: None)

    asyncio.run(
        experiment_runner._run_experiment(
            baseline=None,
        )
    )

    # With k=3 all-pass: train-a runs 2 trials then early-stops.
    assert sorted(call_log) == ["train-a"] * 2
    trials = experiment_runner.record._task_trials("train-a")
    assert trials.majority_solved is True
    assert trials.expected_trial_count == 2


def test_validate_setup_contract_rejects_train_panel_drift():
    experiment_runner = runner.ExperimentRunner.__new__(runner.ExperimentRunner)
    state = ExperimentState(active_baseline_experiment_id="baseline")
    baseline = ExperimentRecord.initialize(
        experiment_id="baseline",
        git_commit_hash="abc123",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-a"],
        started_at="2026-04-10T00:00:00+00:00",
    )
    candidate = ExperimentRecord.initialize(
        experiment_id="candidate",
        git_commit_hash="def456",
        parent_baseline_experiment_id="baseline",
        train_task_ids=["train-a", "train-b"],
        started_at="2026-04-10T00:00:00+00:00",
    )

    with pytest.raises(ValueError, match="candidate train panel must match"):
        experiment_runner._validate_setup_contract(
            state=state,
            candidate=candidate,
            baseline=baseline,
        )


def test_experiment_runner_requires_clean_worktree_by_default(monkeypatch):
    harness_config_cls = FakeHarnessConfig
    calls: list[str] = []

    monkeypatch.setattr(
        runner.control_repo,
        "require_clean_worktree",
        lambda: calls.append("clean"),
    )
    monkeypatch.setattr(runner.control_repo, "get_head_commit", lambda: "abc123")

    experiment_runner = runner.ExperimentRunner(
        harness_config=harness_config_cls(
            experiment_id="exp-1",
            focus_name="focus",
            train_task_names=["task-a"],
        ),
        harbor_config=type(
            "HarborConfig",
            (),
            {"experiments_dir": Path("/tmp/experiments")},
        )(),
        api_key="test-key",
    )

    assert experiment_runner.record.git_commit_hash == "abc123"
    assert calls == ["clean"]


def test_experiment_runner_allows_dirty_worktree_when_explicitly_disabled(monkeypatch):
    harness_config_cls = FakeHarnessConfig

    def _boom() -> None:
        raise AssertionError("dirty-worktree gate should not run")

    monkeypatch.setattr(runner.control_repo, "require_clean_worktree", _boom)
    monkeypatch.setattr(runner.control_repo, "get_head_commit", lambda: "abc123")

    experiment_runner = runner.ExperimentRunner(
        harness_config=harness_config_cls(
            experiment_id="exp-1",
            focus_name="focus",
            train_task_names=["task-a"],
        ),
        harbor_config=type(
            "HarborConfig",
            (),
            {"experiments_dir": Path("/tmp/experiments")},
        )(),
        api_key="test-key",
        require_clean_worktree=False,
    )

    assert experiment_runner.record.git_commit_hash == "abc123"


def test_conclude_experiment_does_not_hard_reset_on_discard(monkeypatch, tmp_path):
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()
    baseline = ExperimentRecord.initialize(
        experiment_id="baseline",
        git_commit_hash="base123",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-a"],
        started_at="2026-04-10T00:00:00+00:00",
    )
    baseline.record_task_result(
        TaskResult(
            task_name="train-a",
            reward=1.0,
            solved=True,
            steps_used=1,
            error=None,
            trial_dir=None,
            trace_path=None,
            metrics_path=None,
            verifier_stdout_path=None,
            started_at="2026-04-10T00:00:00+00:00",
            finished_at="2026-04-10T00:00:01+00:00",
        )
    )
    baseline.finalize(status="keep")
    baseline.write(root=experiments_root)

    experiment_runner = runner.ExperimentRunner.__new__(runner.ExperimentRunner)
    experiment_runner.experiments_root = experiments_root
    experiment_runner.frozen_baseline_experiment_id = "baseline"
    experiment_runner.state = ExperimentState(
        active_baseline_experiment_id="baseline",
        current_experiment_id="candidate",
        updated_at=None,
    )
    experiment_runner.record = ExperimentRecord.initialize(
        experiment_id="candidate",
        git_commit_hash="candidate123",
        parent_baseline_experiment_id="baseline",
        train_task_ids=["train-a"],
        started_at="2026-04-10T00:00:00+00:00",
    )
    experiment_runner.record.record_task_result(
        TaskResult(
            task_name="train-a",
            reward=0.0,
            solved=False,
            steps_used=1,
            error=None,
            trial_dir=None,
            trace_path=None,
            metrics_path=None,
            verifier_stdout_path=None,
            started_at="2026-04-10T00:00:00+00:00",
            finished_at="2026-04-10T00:00:01+00:00",
        )
    )
    experiment_runner.record.finalize(
        status="discard", decision_reason="no train improvement"
    )

    hard_reset_calls: list[str] = []
    update_ref_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        runner.control_repo,
        "hard_reset",
        lambda commit_hash: hard_reset_calls.append(commit_hash),
    )
    monkeypatch.setattr(
        runner.control_repo,
        "update_ref",
        lambda ref_name, commit_hash: update_ref_calls.append((ref_name, commit_hash)),
    )

    experiment_runner._conclude_experiment()

    assert hard_reset_calls == []
    assert update_ref_calls == [
        (failed_experiment_git_ref("candidate"), "candidate123")
    ]


def test_run_baseline_at_head_returns_existing_baseline_when_unchanged(
    monkeypatch,
    tmp_path,
):
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()
    _write_baseline_record(runner, experiments_root)
    ExperimentState(
        active_baseline_experiment_id="baseline",
        current_experiment_id=None,
    ).save(root=experiments_root)

    monkeypatch.setattr(
        runner.control_repo,
        "require_clean_worktree",
        lambda cwd=None: None,
    )
    monkeypatch.setattr(
        runner.control_repo,
        "get_head_commit",
        lambda cwd=None: "base123",
    )

    baseline = runner.ExperimentRunner.run_baseline_at_head(
        harness_config=SimpleNamespace(train_task_names=["train-a"]),
        harbor_config=SimpleNamespace(experiments_dir=experiments_root),
        api_key="key",
    )

    state = ExperimentState.load(root=experiments_root)
    assert baseline.experiment_id == "baseline"
    assert state.current_experiment_id == "baseline"
    assert state.active_baseline_experiment_id == "baseline"


def test_run_baseline_at_head_runs_full_current_panel(monkeypatch, tmp_path):
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()
    _write_baseline_record(runner, experiments_root)
    ExperimentState(
        active_baseline_experiment_id="baseline",
        current_experiment_id="baseline",
    ).save(root=experiments_root)

    monkeypatch.setattr(
        runner.control_repo,
        "require_clean_worktree",
        lambda cwd=None: None,
    )
    monkeypatch.setattr(
        runner.control_repo,
        "get_head_commit",
        lambda cwd=None: "head456",
    )
    _stub_baseline_task_environment(monkeypatch, tmp_path)
    panel_calls: list[list[str]] = []

    async def fake_run_panel(*, record, experiments_root, task_names, **_kwargs):
        panel_calls.append(list(task_names))
        for task_id in task_names:
            record.record_task_result(
                TaskResult(
                    task_name=task_id,
                    reward=0.0,
                    solved=False,
                    error=None,
                    steps_used=1,
                    started_at="2026-04-11T00:00:00+00:00",
                    finished_at="2026-04-11T00:00:01+00:00",
                )
            )
        record.write(root=experiments_root)

    monkeypatch.setattr(runner, "_run_panel", fake_run_panel)

    record = runner.ExperimentRunner.run_baseline_at_head(
        harness_config=SimpleNamespace(
            focus_name="new-focus",
            train_task_names=["train-a", "train-b"],
            task_trials=1,
            llm_provider_config=None,
            max_trial_concurrency=1,
            max_steps=30,
            task_timeout_sec=600.0,
            max_output_retries=2,
        ),
        harbor_config=_FakeHarborConfig(experiments_dir=experiments_root),
        api_key="key",
        experiment_id="baseline_rerun",
        started_at="2026-04-11T00:00:00+00:00",
    )

    assert panel_calls == [["train-a", "train-b"]]
    assert record.status == "keep"
    assert record.decision_reason == "baseline rerun"
    assert record.parent_baseline_experiment_id == "baseline"
    assert record.git_commit_hash == "head456"
    assert record.train_task_ids == ["train-a", "train-b"]
    state = ExperimentState.load(root=experiments_root)
    assert state.active_baseline_experiment_id == "baseline_rerun"


def test_run_baseline_at_head_seeds_when_no_active_baseline(monkeypatch, tmp_path):
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()
    ExperimentState(active_baseline_experiment_id=None).save(root=experiments_root)

    monkeypatch.setattr(
        runner.control_repo,
        "require_clean_worktree",
        lambda cwd=None: None,
    )
    monkeypatch.setattr(
        runner.control_repo,
        "get_head_commit",
        lambda cwd=None: "head000",
    )
    _stub_baseline_task_environment(monkeypatch, tmp_path)

    async def fake_run_panel(*, record, experiments_root, task_names, **_kwargs):
        for task_id in task_names:
            record.record_task_result(
                TaskResult(
                    task_name=task_id,
                    reward=1.0,
                    solved=True,
                    error=None,
                    steps_used=1,
                    started_at="2026-04-11T00:00:00+00:00",
                    finished_at="2026-04-11T00:00:01+00:00",
                )
            )
        record.write(root=experiments_root)

    monkeypatch.setattr(runner, "_run_panel", fake_run_panel)

    record = runner.ExperimentRunner.run_baseline_at_head(
        harness_config=SimpleNamespace(
            focus_name="seed",
            train_task_names=["train-a"],
            task_trials=1,
            llm_provider_config=None,
            max_trial_concurrency=1,
            max_steps=30,
            task_timeout_sec=600.0,
            max_output_retries=2,
        ),
        harbor_config=_FakeHarborConfig(experiments_dir=experiments_root),
        api_key="key",
        decision_reason="baseline seed",
        experiment_id="baseline_seed",
        started_at="2026-04-11T00:00:00+00:00",
    )

    assert record.parent_baseline_experiment_id is None
    assert record.status == "keep"
    assert record.decision_reason == "baseline seed"
    assert (
        ExperimentState.load(root=experiments_root).active_baseline_experiment_id
        == "baseline_seed"
    )


@pytest.mark.parametrize(
    "trial_error",
    [
        pytest.param("environment reset/bootstrap timed out", id="with-message"),
        # An empty-string error still marks a `crash`. The `error is None`
        # contract is load-bearing: a truthiness check on `trial.error` would
        # misread "" as valid evidence.
        pytest.param("", id="empty-string-crash"),
    ],
)
def test_run_baseline_at_head_crashes_when_no_valid_evidence(
    trial_error, monkeypatch, tmp_path
):
    """When every baseline trial is a `crash`, the run produced no valid
    evidence: it must finalize as `crash` and must NOT update
    `active_baseline_experiment_id`. Promoting an empty baseline would make
    every later candidate compare against an all-zero pool."""
    experiments_root = tmp_path / "experiments"
    experiments_root.mkdir()
    ExperimentState(active_baseline_experiment_id=None).save(root=experiments_root)

    monkeypatch.setattr(
        runner.control_repo,
        "require_clean_worktree",
        lambda cwd=None: None,
    )
    monkeypatch.setattr(
        runner.control_repo,
        "get_head_commit",
        lambda cwd=None: "head000",
    )
    _stub_baseline_task_environment(monkeypatch, tmp_path)

    async def fake_run_panel(*, record, experiments_root, task_names, **_kwargs):
        for task_id in task_names:
            record.record_task_result(
                TaskResult(
                    task_name=task_id,
                    reward=0.0,
                    solved=False,
                    error=trial_error,
                    steps_used=42,
                    started_at="2026-04-11T00:00:00+00:00",
                    finished_at="2026-04-11T00:00:01+00:00",
                )
            )
        record.write(root=experiments_root)

    monkeypatch.setattr(runner, "_run_panel", fake_run_panel)

    record = runner.ExperimentRunner.run_baseline_at_head(
        harness_config=SimpleNamespace(
            focus_name="seed",
            train_task_names=["train-a"],
            task_trials=1,
            llm_provider_config=None,
            max_trial_concurrency=1,
            max_steps=30,
            task_timeout_sec=600.0,
            max_output_retries=2,
        ),
        harbor_config=_FakeHarborConfig(experiments_dir=experiments_root),
        api_key="key",
        decision_reason="baseline seed",
        experiment_id="baseline_errored",
        started_at="2026-04-11T00:00:00+00:00",
    )

    assert record.status == "crash"
    assert "no valid trials" in record.error
    assert (
        ExperimentState.load(root=experiments_root).active_baseline_experiment_id
        is None
    )


@pytest.mark.parametrize(
    "trial_error",
    [
        pytest.param("environment reset/bootstrap timed out", id="with-message"),
        pytest.param("", id="empty-string-crash"),
    ],
)
def test_run_experiment_finalizes_crash_when_no_valid_evidence(
    trial_error, monkeypatch, tmp_path
):
    """When the only task's only trial is a `crash`, the run produced no valid
    evidence and must finalize as `crash`, never `keep`/`discard` — there is
    nothing for the gate to compare. (An isolated crash alongside valid trials
    is instead excluded and tolerated; see the mixed-evidence test.)"""

    monkeypatch.setattr(
        runner.control_repo,
        "update_ref",
        lambda ref_name, commit_hash, *, cwd=None: None,
    )

    experiment_runner = make_runner(
        monkeypatch,
        tmp_path,
        experiment_id="exp-crash-on-trial-error",
        focus_name="focus",
        train_task_names=["task-a"],
        task_trials=1,
        max_trial_concurrency=1,
        task_dirs=["task-a"],
    )
    stub_trial_factories(monkeypatch, experiment_runner)
    experiment_runner.experiment_dir.mkdir(parents=True, exist_ok=True)

    async def fake_run_task(*, task_name, **_kwargs):
        return TaskResult(
            task_name=task_name,
            reward=0.0,
            solved=False,
            error=trial_error,
            steps_used=12,
            trial_dir=None,
            trace_path=None,
            metrics_path=None,
            verifier_stdout_path=None,
            metrics=TaskMetrics(),
            started_at="2026-05-05T00:00:00+00:00",
            finished_at="2026-05-05T00:00:01+00:00",
        )

    monkeypatch.setattr(runner, "run_task", fake_run_task)

    asyncio.run(experiment_runner._run_experiment(baseline=None))

    record = experiment_runner.record
    assert record.status == "crash"
    assert "no valid trials" in record.error


def test_run_experiment_excludes_crash_trial_but_scores_valid_ones(
    monkeypatch, tmp_path
):
    """A lone `crash` trial alongside valid trials must not sink the run: it is
    dropped from the gate's evidence and the surviving valid trials are scored.
    Here trial 1 crashes and trials 2-3 solve, so the frontier task
    majority-solves on 2 valid trials and the run is kept."""

    monkeypatch.setattr(
        runner.control_repo,
        "update_ref",
        lambda ref_name, commit_hash, *, cwd=None: None,
    )

    experiment_runner = make_runner(
        monkeypatch,
        tmp_path,
        experiment_id="exp-mixed-evidence",
        focus_name="focus",
        train_task_names=["task-a"],
        task_trials=3,
        max_trial_concurrency=1,
        task_dirs=["task-a"],
    )
    stub_trial_factories(monkeypatch, experiment_runner)
    experiment_runner.experiment_dir.mkdir(parents=True, exist_ok=True)

    call_count = 0

    async def fake_run_task(*, task_name, **_kwargs):
        # max_trial_concurrency=1 serializes trials, so the counter is deterministic.
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return TaskResult(
                task_name=task_name,
                reward=0.0,
                solved=False,
                error="environment reset/bootstrap timed out",
                steps_used=12,
                metrics=TaskMetrics(failure_mode="crash"),
                started_at="2026-05-05T00:00:00+00:00",
                finished_at="2026-05-05T00:00:01+00:00",
            )
        return TaskResult(
            task_name=task_name,
            reward=1.0,
            solved=True,
            error=None,
            steps_used=5,
            metrics=TaskMetrics(failure_mode="solved"),
            started_at="2026-05-05T00:00:00+00:00",
            finished_at="2026-05-05T00:00:01+00:00",
        )

    monkeypatch.setattr(runner, "run_task", fake_run_task)

    asyncio.run(experiment_runner._run_experiment(baseline=None))

    record = experiment_runner.record
    trials = record.train_task_results["task-a"]
    # All three slots ran; only the two non-crash trials count as evidence.
    assert len(trials.finished_trials) == 3
    assert len(trials.valid_trials) == 2
    assert trials.solved_count == 2
    # 2/2 valid trials solved on the frontier ⇒ improvement ⇒ keep.
    assert record.status == "keep"
    assert record.decision_reason == "train task task-a improved"


def test_run_experiment_discards_task_timeout_without_crashing(monkeypatch, tmp_path):
    """A task episode timeout is a measured unsolved trial, not an
    experiment-level infrastructure crash."""

    monkeypatch.setattr(
        runner.control_repo,
        "update_ref",
        lambda ref_name, commit_hash, *, cwd=None: None,
    )

    experiment_runner = make_runner(
        monkeypatch,
        tmp_path,
        experiment_id="exp-timeout-is-unsolved",
        focus_name="focus",
        train_task_names=["task-a"],
        task_trials=1,
        max_trial_concurrency=1,
        task_dirs=["task-a"],
    )
    stub_trial_factories(monkeypatch, experiment_runner)
    experiment_runner.experiment_dir.mkdir(parents=True, exist_ok=True)

    async def fake_run_task(*, task_name, **_kwargs):
        return TaskResult(
            task_name=task_name,
            reward=0.0,
            solved=False,
            error=None,
            steps_used=12,
            metrics=TaskMetrics(failure_mode="hit_timeout"),
            started_at="2026-05-05T00:00:00+00:00",
            finished_at="2026-05-05T00:00:01+00:00",
        )

    monkeypatch.setattr(runner, "run_task", fake_run_task)

    asyncio.run(experiment_runner._run_experiment(baseline=None))

    record = experiment_runner.record
    assert record.status == "discard"
    assert record.error == ""
    assert record.decision_reason == "no train task improvement reached significance"


@pytest.mark.parametrize("trial_mode", ["result", "exception"])
def test_run_panel_persists_completed_trial_without_refreshing_evidence(
    trial_mode, monkeypatch, tmp_path
):
    experiment_id = f"exp-g4-{trial_mode}"
    experiment_runner = make_runner(
        monkeypatch,
        tmp_path,
        experiment_id=experiment_id,
        focus_name="focus",
        train_task_names=["task-a"],
        task_trials=1,
        max_trial_concurrency=1,
        task_dirs=["task-a"],
    )
    stub_trial_factories(monkeypatch, experiment_runner)

    async def fake_run_task(*, task_name, **_kwargs):
        if trial_mode == "exception":
            raise RuntimeError("simulated env reset crash")
        return TaskResult(
            task_name=task_name,
            reward=1.0,
            solved=True,
            error=None,
            steps_used=1,
            trial_dir=None,
            trace_path=None,
            metrics_path=None,
            verifier_stdout_path=None,
            metrics=TaskMetrics(),
            started_at="2026-05-05T00:00:00+00:00",
            finished_at="2026-05-05T00:00:01+00:00",
        )

    monkeypatch.setattr(runner, "run_task", fake_run_task)

    _run_panel_for_experiment_runner(runner, experiment_runner, ["task-a"])

    record_path = tmp_path / "experiments" / experiment_id / "experiment.json"
    payload = json.loads(record_path.read_text())
    trials = payload["train_task_results"]["task-a"]["trials"]
    assert len(trials) == 1
    if trial_mode == "exception":
        assert trials[0]["solved"] is False
        assert trials[0]["error"] == "simulated env reset crash"
        assert trials[0]["metrics"]["failure_mode"] == "crash"
    else:
        assert trials[0]["solved"] is True
        assert trials[0]["error"] is None
    outcomes = (payload.get("evidence") or {}).get("task_outcomes") or []
    assert outcomes == []


class _FakeStream:
    def __init__(self, *, is_tty: bool) -> None:
        self._is_tty = is_tty
        self.buffer: list[str] = []

    def isatty(self) -> bool:
        return self._is_tty

    def write(self, text: str) -> int:
        self.buffer.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    @property
    def text(self) -> str:
        return "".join(self.buffer)


def test_format_panel_progress_anchors_on_tasks_and_computes_eta():
    line = runner.format_panel_progress(
        tasks_done=25,
        total_tasks=100,
        trials_done=80,
        trials_planned=320,
        solved=17,
        decided=24,
        error_trials=6,
        in_flight=10,
        elapsed_sec=3600,
    )
    assert "25/100 tasks (25%)" in line
    assert "trials 80/320" in line
    assert "solved 17/24" in line
    assert "errors 6" in line
    assert "active 10" in line
    # 25 tasks in 1h -> 75 remaining at 25/h -> 3h elapsed-equivalent left.
    assert "1h00m elapsed" in line
    assert "~3h00m left" in line
    # Bar fill is proportional to task completion.
    assert line.count("#") == int(0.25 * runner.PROGRESS_BAR_WIDTH)


def test_format_panel_progress_unknown_eta_before_first_task():
    line = runner.format_panel_progress(
        tasks_done=0,
        total_tasks=10,
        trials_done=3,
        trials_planned=50,
        solved=0,
        decided=0,
        error_trials=0,
        in_flight=10,
        elapsed_sec=42,
    )
    assert "0/10 tasks (0%)" in line
    assert "~-- left" in line
    assert "#" not in line


def test_panel_progress_reporter_silent_on_non_tty():
    record = ExperimentRecord.initialize(
        experiment_id="exp-progress",
        git_commit_hash="abc123",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-a"],
        started_at="2026-04-10T00:00:00+00:00",
    )
    record.record_task_result(_task_result(task_name="train-a", reward=1.0))
    stream = _FakeStream(is_tty=False)
    reporter = runner.PanelProgressReporter(
        total_tasks=1, max_trial_concurrency=10, stream=stream
    )
    reporter.render(record)
    reporter.close()
    assert stream.text == ""


def test_panel_progress_reporter_draws_and_finalizes_on_tty():
    record = ExperimentRecord.initialize(
        experiment_id="exp-progress",
        git_commit_hash="abc123",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-a", "train-b"],
        started_at="2026-04-10T00:00:00+00:00",
    )
    stream = _FakeStream(is_tty=True)
    reporter = runner.PanelProgressReporter(
        total_tasks=2, max_trial_concurrency=10, stream=stream
    )

    record.record_task_result(_task_result(task_name="train-a", reward=1.0))
    reporter.render(record)
    assert "1/2 tasks" in stream.text
    assert "\r\033[K" in stream.text
    assert not stream.text.endswith("\n")  # bar is still live

    # Completing the panel finalizes the line with a trailing newline.
    record.record_task_result(
        _task_result(task_name="train-b", reward=0.0, solved=False)
    )
    reporter.render(record)
    assert "2/2 tasks (100%)" in stream.text
    assert stream.text.endswith("\n")

    # Further renders are no-ops once finalized.
    before = stream.text
    reporter.render(record)
    assert stream.text == before


def test_panel_progress_reporter_maps_record_counts_to_line():
    record = runner.ExperimentRecord.initialize(
        experiment_id="exp-progress",
        git_commit_hash="abc123",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-a", "train-b"],
        started_at="2026-04-10T00:00:00+00:00",
        expected_trial_count=2,
    )
    stream = _FakeStream(is_tty=True)
    reporter = runner.PanelProgressReporter(
        total_tasks=2, max_trial_concurrency=2, stream=stream
    )

    record.record_task_result(
        _task_result(
            task_name="train-a",
            reward=None,
            error="reset failed",
        )
    )
    reporter.render(record)

    assert "trials 1/4" in stream.text
    assert "solved 0/0" in stream.text
    assert "errors 1" in stream.text
    assert "active 2" in stream.text


def test_panel_progress_reporter_close_terminates_dangling_line():
    record = ExperimentRecord.initialize(
        experiment_id="exp-progress",
        git_commit_hash="abc123",
        parent_baseline_experiment_id=None,
        train_task_ids=["train-a", "train-b"],
        started_at="2026-04-10T00:00:00+00:00",
    )
    stream = _FakeStream(is_tty=True)
    reporter = runner.PanelProgressReporter(
        total_tasks=2, max_trial_concurrency=10, stream=stream
    )
    record.record_task_result(_task_result(task_name="train-a", reward=1.0))
    reporter.render(record)  # one task done, panel incomplete -> no newline yet
    assert not stream.text.endswith("\n")
    reporter.close()  # crash/cancel path terminates the line
    assert stream.text.endswith("\n")
