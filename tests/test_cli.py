from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

from src import cli
from src.experiment.record import ExperimentResult, TrialResult


async def _always_solve(task_id, run_id, heavy_action_semaphore, slot_release):
    return TrialResult(
        run_id=run_id, solved=True, failure_mode="solved", verifier_passed=True
    )


def _harness(panels=None, *, task_trials=1):
    return SimpleNamespace(
        task_trials=task_trials,
        max_trial_concurrency=4,
        max_heavy_action_concurrency=2,
        panels=panels or [],
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
    # auto's train-then-test: two `uv run exp` calls share one EXP_EXPERIMENT_ID,
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


def test_selected_task_ids_defaults_to_all_configured_tasks(monkeypatch):
    monkeypatch.delenv("EXP_TASK_IDS", raising=False)
    harness = _harness(
        [
            SimpleNamespace(task_names=["a", "b"]),
            SimpleNamespace(task_names=["c"]),
        ]
    )
    assert cli._selected_task_ids(harness) == ["a", "b", "c"]


def test_selected_task_ids_honors_exp_task_ids_subset(monkeypatch):
    monkeypatch.setenv("EXP_TASK_IDS", "b, c")
    harness = _harness([SimpleNamespace(task_names=["a", "b", "c"])])
    # The env subset wins over the configured set (auto's train/test selection).
    assert cli._selected_task_ids(harness) == ["b", "c"]


def test_selected_experiment_id_honors_exp_experiment_id(monkeypatch):
    monkeypatch.setenv("EXP_EXPERIMENT_ID", "exp-existing")
    assert cli._selected_experiment_id() == "exp-existing"


def test_selected_experiment_id_generates_fresh_id_by_default(monkeypatch):
    monkeypatch.delenv("EXP_EXPERIMENT_ID", raising=False)
    fresh = cli._selected_experiment_id()
    assert fresh.startswith("exp-") and len(fresh) > len("exp-")


def test_experiments_dir_override_anchors_to_exp_experiments_dir(monkeypatch, tmp_path):
    # §12: auto runs exp in a candidate worktree but passes the absolute main-repo
    # experiments dir, so artifacts + the verifier-context cache stay canonical.
    from src.env.harbor import HarborConfig

    target = tmp_path / "main-experiments"
    monkeypatch.setenv("EXP_EXPERIMENTS_DIR", str(target))
    overridden = cli._apply_experiments_dir_override(HarborConfig())
    assert overridden.experiments_dir == target.resolve()
    # the shared verifier-context cache re-derives under the new root
    assert overridden.verifier_contexts_dir == (target / "_verifier_contexts").resolve()


def test_experiments_dir_override_is_a_noop_without_the_env(monkeypatch):
    from src.env.harbor import HarborConfig

    monkeypatch.delenv("EXP_EXPERIMENTS_DIR", raising=False)
    base = HarborConfig()
    assert cli._apply_experiments_dir_override(base) is base  # standalone exp


def test_experiments_dir_override_rejects_a_relative_path(monkeypatch):
    import pytest

    from src.env.harbor import HarborConfig

    monkeypatch.setenv("EXP_EXPERIMENTS_DIR", "relative/experiments")
    with pytest.raises(ValueError, match="absolute"):
        cli._apply_experiments_dir_override(HarborConfig())


# --- auto entry point (unchanged; control rebuilt Steps 3-5) ----------------


def test_main_auto_runs_supervisor_loop(monkeypatch):
    fake_supervisor_module = types.ModuleType("src.control.supervisor")
    calls: dict[str, object] = {}

    def fake_run_supervisor_loop(*, repo_root, **_kwargs):
        calls["repo_root"] = repo_root

    fake_supervisor_module.run_supervisor_loop = fake_run_supervisor_loop
    monkeypatch.setitem(sys.modules, "src.control.supervisor", fake_supervisor_module)

    assert cli.main_auto() == 0
    assert calls["repo_root"] == cli.Path(__file__).resolve().parents[1]


def test_main_auto_returns_130_on_keyboard_interrupt(monkeypatch):
    fake_supervisor_module = types.ModuleType("src.control.supervisor")

    def fake_run_supervisor_loop(*, repo_root, **_kwargs):
        raise KeyboardInterrupt

    fake_supervisor_module.run_supervisor_loop = fake_run_supervisor_loop
    monkeypatch.setitem(sys.modules, "src.control.supervisor", fake_supervisor_module)

    assert cli.main_auto() == 130


def test_main_auto_halts_on_credentials_expired(monkeypatch, capsys):
    from src.llm.codex import (
        CODEX_CREDENTIALS_EXPIRED_EXIT_CODE,
        CODEX_CREDENTIALS_EXPIRED_MESSAGE,
        ChatGptCodexCredentialsExpiredError,
    )

    fake_supervisor_module = types.ModuleType("src.control.supervisor")

    def fake_run_supervisor_loop(*, repo_root, **_kwargs):
        raise ChatGptCodexCredentialsExpiredError(CODEX_CREDENTIALS_EXPIRED_MESSAGE)

    fake_supervisor_module.run_supervisor_loop = fake_run_supervisor_loop
    monkeypatch.setitem(sys.modules, "src.control.supervisor", fake_supervisor_module)

    assert cli.main_auto() == CODEX_CREDENTIALS_EXPIRED_EXIT_CODE
    assert "codex login" in capsys.readouterr().err


def test_load_llm_provider_secret_skips_openrouter_key_for_chatgpt():
    harness_config = types.SimpleNamespace(
        llm_provider_config=types.SimpleNamespace(provider="chatgpt_codex")
    )

    assert cli._load_llm_provider_secret(harness_config=harness_config) is None
