from __future__ import annotations

import sys
import types

from src import cli


def test_main_exp_runs_experiment(monkeypatch, capsys):
    fake_runner_module = types.ModuleType("src.experiment.runner")
    calls: dict[str, object] = {}

    class FakeExperimentRunner:
        def __init__(
            self,
            *,
            harness_config,
            harbor_config,
            api_key,
            require_clean_worktree,
        ):
            self.harness_config = harness_config
            self.harbor_config = harbor_config
            self.api_key = api_key
            calls["require_clean_worktree"] = require_clean_worktree

        def run(self):
            return types.SimpleNamespace(status="keep")

    fake_runner_module.ExperimentRunner = FakeExperimentRunner
    monkeypatch.setitem(sys.modules, "src.experiment.runner", fake_runner_module)
    monkeypatch.setattr(
        cli,
        "load_runtime_config",
        lambda: (
            types.SimpleNamespace(),
            types.SimpleNamespace(
                experiment_id="exp-1",
                panels=[
                    types.SimpleNamespace(
                        id="train",
                        purpose="promotion",
                        task_names=["task-a", "task-b"],
                    ),
                    types.SimpleNamespace(
                        id="test",
                        purpose="regression_veto",
                        task_names=["heldout-a"],
                    ),
                ],
            ),
            "api-key",
        ),
    )

    assert cli.main_exp() == 0

    assert calls["require_clean_worktree"] is True
    output = capsys.readouterr().out
    assert "experiment: exp-1" in output
    assert "panels: train(promotion): 2 tasks; test(regression_veto): 1 task" in output
    assert "evaluation: keep" in output


def test_main_exp_allows_dirty_worktree_with_env_override(monkeypatch, capsys):
    fake_runner_module = types.ModuleType("src.experiment.runner")
    calls: dict[str, object] = {}

    class FakeExperimentRunner:
        def __init__(
            self,
            *,
            harness_config,
            harbor_config,
            api_key,
            require_clean_worktree,
        ):
            calls["require_clean_worktree"] = require_clean_worktree

        def run(self):
            return types.SimpleNamespace(status="keep")

    fake_runner_module.ExperimentRunner = FakeExperimentRunner
    monkeypatch.setitem(sys.modules, "src.experiment.runner", fake_runner_module)
    monkeypatch.setenv("EXP_ALLOW_DIRTY_WORKTREE", "1")
    monkeypatch.setattr(
        cli,
        "load_runtime_config",
        lambda: (
            types.SimpleNamespace(),
            types.SimpleNamespace(
                experiment_id="exp-1",
                panels=[
                    types.SimpleNamespace(
                        id="train",
                        purpose="promotion",
                        task_names=["task-a"],
                    )
                ],
            ),
            "api-key",
        ),
    )

    assert cli.main_exp() == 0

    assert calls["require_clean_worktree"] is False
    output = capsys.readouterr().out
    assert "evaluation: keep" in output


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


def test_load_llm_provider_secret_skips_openrouter_key_for_chatgpt():
    harness_config = types.SimpleNamespace(
        llm_provider_config=types.SimpleNamespace(provider="chatgpt_codex")
    )

    assert cli._load_llm_provider_secret(harness_config=harness_config) is None
