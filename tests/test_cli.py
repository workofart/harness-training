from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src import cli
from src.experiment.record import ExperimentResult, TaskResult, TrialResult


async def _always_solve(task_id, run_id, heavy_action_semaphore, slot_release):
    return TrialResult(
        run_id=run_id, solved=True, failure_mode="solved", verifier_passed=True
    )


def _harness(train=None, test=None, *, task_trials=1):
    return SimpleNamespace(
        task_trials=task_trials,
        max_trial_concurrency=4,
        max_heavy_action_concurrency=2,
        train=train,
        test=test,
    )


# --- the exp contract (plan.md §2) ------------------------------------------


def test_run_experiment_writes_section5_conformant_record(tmp_path):
    # run_experiment runs the selected task_ids at the uniform full budget into
    # experiment_id -> a §5-conformant ExperimentResult. A stub trial_runner
    # scripts solved trials, so no Harbor/llm/executor is touched.
    result = asyncio.run(
        cli.run_experiment(
            harness_config=_harness(),
            harbor_config=SimpleNamespace(experiments_dir=tmp_path),
            git_commit_hash="c0ffee",
            task_ids=["task-a", "task-b"],
            experiment_id="exp-1",
            trial_runner=_always_solve,
        )
    )

    assert result.run_status == "completed"
    loaded = ExperimentResult.load("exp-1", root=tmp_path)
    assert set(loaded.tasks) == {"task-a", "task-b"}
    assert all(len(task.trials) == 1 for task in loaded.tasks.values())
    assert loaded.git_commit_hash == "c0ffee"
    assert loaded.finished_at is not None


def test_run_experiment_appends_to_an_existing_experiment_id(tmp_path):
    # auto's train-then-test: two `uv run exp` calls share one --experiment-id,
    # so the second call appends to the first's record (plan.md §2/§7).
    common = dict(
        harness_config=_harness(),
        harbor_config=SimpleNamespace(experiments_dir=tmp_path),
        git_commit_hash="abc123",
        experiment_id="exp-shared",
        trial_runner=_always_solve,
    )
    asyncio.run(cli.run_experiment(task_ids=["train-a", "train-b"], **common))
    second = asyncio.run(cli.run_experiment(task_ids=["test-a"], **common))

    assert set(second.tasks) == {"train-a", "train-b", "test-a"}
    loaded = ExperimentResult.load("exp-shared", root=tmp_path)
    assert set(loaded.tasks) == {"train-a", "train-b", "test-a"}
    assert loaded.run_status == "completed" and loaded.finished_at is not None


def test_run_experiment_honors_an_explicit_per_task_budget(tmp_path):
    # `auto` passes the gate's budget_from_baseline (1 confirming trial for a
    # deterministically-solved task) instead of the uniform-full default. With
    # budget=1 the solved task runs exactly one trial, not task_trials=3.
    result = asyncio.run(
        cli.run_experiment(
            harness_config=_harness(task_trials=3),
            harbor_config=SimpleNamespace(experiments_dir=tmp_path),
            git_commit_hash="c0ffee",
            task_ids=["task-a"],
            experiment_id="exp-budget",
            trial_runner=_always_solve,
            budget={"task-a": 1},
        )
    )
    assert result.run_status == "completed"
    loaded = ExperimentResult.load("exp-budget", root=tmp_path)
    assert loaded.tasks["task-a"].expected_trial_count == 1
    assert len(loaded.tasks["task-a"].trials) == 1


def test_selected_task_ids_defaults_to_all_configured_tasks():
    harness = _harness(
        train=SimpleNamespace(task_names=["a", "b"]),
        test=SimpleNamespace(task_names=["c"]),
    )
    assert cli._selected_task_ids(harness, None) == ["a", "b", "c"]


def test_selected_task_ids_honors_tasks_subset():
    harness = _harness(train=SimpleNamespace(task_names=["a", "b", "c"]))
    # The --tasks subset wins over the configured set (auto's train/test selection).
    assert cli._selected_task_ids(harness, "b, c") == ["b", "c"]


def test_selected_experiment_id_honors_the_flag():
    assert cli._selected_experiment_id("exp-existing") == "exp-existing"


def test_selected_experiment_id_generates_fresh_id_by_default():
    fresh = cli._selected_experiment_id(None)
    assert fresh.startswith("exp-") and len(fresh) > len("exp-")


def test_selected_trial_budget_defaults_to_uniform_full_without_the_flag():
    # A standalone exp/baseline run omits --trial-budget -> every selected
    # task gets the uniform-full task_trials count.
    budget = cli._selected_trial_budget(
        task_ids=["a", "b"], harness_config=_harness(task_trials=3), trial_budget=None
    )
    assert budget == {"a": 3, "b": 3}


def test_selected_trial_budget_honors_the_flag():
    # auto's candidate run sets the per-task budget_from_baseline map (§9 #7); the
    # deterministic-baseline single-trial shortcut crosses the seam as JSON.
    budget = cli._selected_trial_budget(
        task_ids=["a", "b"],
        harness_config=_harness(task_trials=3),
        trial_budget='{"a": 1, "b": 3}',
    )
    assert budget == {"a": 1, "b": 3}


def test_selected_trial_budget_rejects_a_task_set_mismatch():
    # The loop is the only writer of --trial-budget, so a budget that does not
    # cover the selected tasks exactly is a bug -> fail fast (§12 strict interface).
    with pytest.raises(ValueError, match="!="):
        cli._selected_trial_budget(
            task_ids=["a", "b"],
            harness_config=_harness(task_trials=3),
            trial_budget='{"a": 1}',
        )


def test_parse_exp_args_accepts_the_auto_seam_flags():
    args = cli._parse_exp_args(
        [
            "--tasks",
            "a,b",
            "--experiment-id",
            "exp-x",
            "--experiments-dir",
            "/abs/experiments",
            "--trial-budget",
            '{"a": 1, "b": 3}',
        ]
    )
    assert args.tasks == "a,b"
    assert args.experiment_id == "exp-x"
    assert args.experiments_dir == "/abs/experiments"
    assert args.trial_budget == '{"a": 1, "b": 3}'


def test_parse_exp_args_defaults_every_flag_to_none():
    args = cli._parse_exp_args([])
    assert (
        args.tasks,
        args.experiment_id,
        args.experiments_dir,
        args.trial_budget,
    ) == (
        None,
        None,
        None,
        None,
    )


def test_experiments_dir_override_anchors_artifacts(tmp_path):
    # §12: auto runs exp in a candidate worktree but passes the absolute main-repo
    # experiments dir, so artifacts + the verifier-context cache stay canonical.
    from src.env.harbor import HarborConfig

    target = tmp_path / "main-experiments"
    overridden = cli._apply_experiments_dir_override(HarborConfig(), str(target))
    assert overridden.experiments_dir == target.resolve()
    # the shared verifier-context cache re-derives under the new root
    assert overridden.verifier_contexts_dir == (target / "_verifier_contexts").resolve()


def test_experiments_dir_override_is_a_noop_without_the_flag():
    from src.env.harbor import HarborConfig

    base = HarborConfig()
    assert cli._apply_experiments_dir_override(base, None) is base  # standalone exp


def test_experiments_dir_override_rejects_a_relative_path():
    from src.env.harbor import HarborConfig

    with pytest.raises(ValueError, match="absolute"):
        cli._apply_experiments_dir_override(HarborConfig(), "relative/experiments")


# --- auto entry point (Step 5: main_auto drives supervisor.loop.run_auto) ----


def test_main_auto_drives_run_auto_with_a_loop_context(monkeypatch):
    # main_auto builds a LoopContext from the real config and drives
    # supervisor.loop.run_auto until its terminal Halt (§6). Stub run_auto so no
    # real loop runs; assert the context it was handed is wired to the primary repo.
    from src.supervisor.policy import Halt

    captured: dict[str, object] = {}

    def fake_run_auto(ctx):
        captured["ctx"] = ctx
        return Halt("nothing to do (test)")

    monkeypatch.setattr("src.supervisor.loop.run_auto", fake_run_auto)

    assert cli.main_auto() == 0
    ctx = captured["ctx"]
    repo_root = cli.Path(__file__).resolve().parents[1]
    assert ctx.repo_root == repo_root
    assert ctx.program_md_path == repo_root / "program.md"
    # §12 path anchoring: the experiments dir handed to every exp run is absolute.
    assert ctx.experiments_dir.is_absolute()


def test_main_auto_returns_130_on_keyboard_interrupt(monkeypatch):
    def fake_run_auto(ctx):
        raise KeyboardInterrupt

    monkeypatch.setattr("src.supervisor.loop.run_auto", fake_run_auto)
    assert cli.main_auto() == 130


def test_main_auto_halts_on_credentials_expired(monkeypatch, capsys):
    from src.llm.codex import (
        CODEX_CREDENTIALS_EXPIRED_EXIT_CODE,
        CODEX_CREDENTIALS_EXPIRED_MESSAGE,
        ChatGptCodexCredentialsExpiredError,
    )

    def fake_run_auto(ctx):
        raise ChatGptCodexCredentialsExpiredError(CODEX_CREDENTIALS_EXPIRED_MESSAGE)

    monkeypatch.setattr("src.supervisor.loop.run_auto", fake_run_auto)

    assert cli.main_auto() == CODEX_CREDENTIALS_EXPIRED_EXIT_CODE
    assert "codex login" in capsys.readouterr().err


def test_main_auto_anchors_experiments_dir_to_the_repo_not_cwd(monkeypatch, tmp_path):
    # §12: experiments_dir must anchor to <main_repo>, not the process cwd. `uv run
    # auto` may be invoked from any directory, but HarborConfig resolves a relative
    # experiments_dir against cwd -- so main_auto must re-anchor it to repo_root
    # (the loop then hands this absolute dir to every exp via --experiments-dir).
    from src.supervisor.policy import Halt

    captured: dict[str, object] = {}

    def fake_run_auto(ctx):
        captured["ctx"] = ctx
        return Halt("nothing to do (test)")

    monkeypatch.setattr("src.supervisor.loop.run_auto", fake_run_auto)
    monkeypatch.chdir(tmp_path)  # invoke from an unrelated cwd

    assert cli.main_auto() == 0
    repo_root = cli.Path(cli.__file__).resolve().parents[1]
    experiments_dir = captured["ctx"].experiments_dir
    assert experiments_dir == (repo_root / "experiments").resolve()
    assert tmp_path.resolve() not in experiments_dir.parents


def test_load_llm_provider_secret_skips_openrouter_key_for_chatgpt():
    harness_config = SimpleNamespace(
        llm_provider_config=SimpleNamespace(provider="chatgpt_codex")
    )

    assert cli._load_llm_provider_secret(harness_config=harness_config) is None


# --- the uv run exp progress bar --------------------------------------------


class _FakeStream:
    def __init__(self, *, tty: bool) -> None:
        self._tty = tty
        self.writes: list[str] = []

    def isatty(self) -> bool:
        return self._tty

    def write(self, text: str) -> None:
        self.writes.append(text)

    def flush(self) -> None:
        pass


def test_format_exp_progress_anchors_on_task_completion():
    line = cli.format_exp_progress(
        tasks_done=12,
        total_tasks=48,
        trials_done=30,
        trials_planned=105,
        solved=11,
        decided=12,
        error_trials=2,
        in_flight=6,
        elapsed_sec=125.0,
    )
    # 12/48 = 25% -> 6 of 24 cells filled.
    assert line.startswith("[" + "#" * 6 + "-" * 18 + "]")
    assert "12/48 tasks (25%)" in line
    assert "trials 30/105" in line
    assert "solved 11/12" in line
    assert "errors 2" in line
    assert "active 6" in line
    assert "2m05s elapsed" in line


def test_exp_progress_bar_is_silent_off_a_tty():
    stream = _FakeStream(tty=False)
    bar = cli._ExpProgressBar(task_ids=["a"], max_trial_concurrency=2, stream=stream)
    bar.render({"a": TaskResult.empty(expected_trial_count=1)})
    bar.close()
    assert stream.writes == []


def test_exp_progress_bar_renders_and_scopes_to_its_own_task_ids():
    stream = _FakeStream(tty=True)
    bar = cli._ExpProgressBar(task_ids=["a"], max_trial_concurrency=2, stream=stream)
    finished = TaskResult(
        expected_trial_count=1,
        trials=[
            TrialResult(
                run_id="r1", solved=True, failure_mode="solved", verifier_passed=True
            )
        ],
    )
    # The live record also holds a prior "train" task (an auto veto append); the
    # bar must count only its own task_ids, so this reads 1/1, not 2/2.
    bar.render({"a": finished, "train": finished})
    assert stream.writes, "a TTY render should write"
    assert "1/1 tasks (100%)" in stream.writes[-1]
    bar.close()
    assert stream.writes[-1] == "\n"  # close caps the dangling line


def test_exp_progress_bar_solved_counts_majority_solved_tasks_not_trials():
    # Regression: `solved` is a task count (majority-solved tasks) matching the
    # `decided` task denominator. Summing per-trial solves instead let the
    # numerator outrun the denominator ("solved 4/2") on any multi-trial task.
    def _solved(run_id: str) -> TrialResult:
        return TrialResult(
            run_id=run_id, solved=True, failure_mode="solved", verifier_passed=True
        )

    def _unsolved(run_id: str) -> TrialResult:
        return TrialResult(
            run_id=run_id,
            solved=False,
            failure_mode="verified_rejected",
            verifier_passed=False,
        )

    stream = _FakeStream(tty=True)
    bar = cli._ExpProgressBar(
        task_ids=["maj", "min"], max_trial_concurrency=4, stream=stream
    )
    majority = TaskResult(
        expected_trial_count=3, trials=[_solved("m0"), _solved("m1"), _solved("m2")]
    )
    # One solve of three valid trials: decided, but not majority-solved.
    minority = TaskResult(
        expected_trial_count=3,
        trials=[_solved("n0"), _unsolved("n1"), _unsolved("n2")],
    )
    bar.render({"maj": majority, "min": minority})
    # Both tasks decided; only `maj` is majority-solved -> 1/2, not the 4/2 a
    # per-trial sum (3 + 1 solved trials) would print.
    assert "solved 1/2" in stream.writes[-1]
