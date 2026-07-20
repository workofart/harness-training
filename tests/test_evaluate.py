from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import docker
from git import Repo
import pytest
from conftest import TEST_MEASUREMENT_IDENTITY

import scripts.evaluate as evaluation
import src.measurement as measurement
from conftest import make_llm_provider_config
from src.config import EnvironmentConfig, RunConfig
from src.measurement import MeasurementError, PreflightError
from src.rollout.records import ExperimentResult, RolloutResult
from src.rollout.store import RunStore, invoker_repo_root


def _config(path: str = "config/example.json") -> RunConfig:
    return RunConfig(
        config_path=path,
        schema_version=13,
        training_target={"module": "src.policy.core"},
        environment=EnvironmentConfig(kind="swe", task_names=["task-a", "task-b"]),
        llm_provider_config=make_llm_provider_config(),
    )


def _result(
    experiment_id: str = "exp-1",
    *,
    solved: tuple[str, ...] = (),
    crash_reason: str | None = None,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        git_commit_hash="commit",
        measurement_identity=TEST_MEASUREMENT_IDENTITY,
        git_dirty=False,
        config_path="config/example.json",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:05+00:00",
        crash_reason=crash_reason,
        tasks={
            task_id: RolloutResult(
                task_id=task_id,
                failure_mode=("solved" if task_id in solved else "verified_rejected"),
                error=None,
                metrics={},
                rollout_dir=None,
                trace_path=None,
                started_at=None,
                finished_at=None,
            )
            for task_id in ("task-a", "task-b")
        },
    )


def test_evaluate_uses_hardened_path_indexes_returns_and_renders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.evaluate as entrypoint

    result = _result(solved=("task-a",))
    root = tmp_path / "evals"
    observed: dict[str, object] = {}

    def fake_run_isolated_experiment(**kwargs):
        observed.update(kwargs)
        kwargs["observer"].experiment_started(result.experiment_id)
        for task_id, rollout in result.tasks.items():
            kwargs["observer"].task_finished(task_id, rollout.failure_mode)
        return result

    config = _config("config/example.json")
    monkeypatch.setattr(
        entrypoint, "run_isolated_experiment", fake_run_isolated_experiment
    )
    monkeypatch.setattr(entrypoint, "invoker_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        entrypoint,
        "preflight",
        lambda configs: pytest.fail("evaluate() must not repeat CLI preflight"),
        raising=False,
    )
    monkeypatch.setattr(
        entrypoint,
        "load_dotenv",
        lambda: pytest.fail("evaluate() must not load dotenv"),
    )

    actual = entrypoint.evaluate(config)

    assert actual.kind == "eval"
    assert actual.experiment_id == result.experiment_id
    assert observed["config_path"] == config.config_path
    assert observed["tracker"].root == root
    assert observed["measure_root"] == tmp_path
    assert RunStore(root).read_index()[0].kind == "eval"
    output = capsys.readouterr().out
    assert "measure   · eval config/example.json · exp-1 · 2 tasks" in output
    assert ("measure   · 1/2 complete · solved 1/2 · task-a · solved") in output
    assert (
        "measure   · 2/2 complete · solved 1/2 · task-b · verified_rejected"
    ) in output
    assert "eval config/example.json · solved 1/2 · 5s" in output


def test_invoker_defaults_follow_git_root_from_subdirectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Repo.init(tmp_path)
    subdirectory = tmp_path / "nested" / "directory"
    subdirectory.mkdir(parents=True)
    monkeypatch.chdir(subdirectory)
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        evaluation,
        "run_isolated_experiment",
        lambda **kwargs: observed.update(kwargs) or _result(),
    )

    result = evaluation.evaluate(_config())

    assert invoker_repo_root() == tmp_path
    assert observed["tracker"].root == tmp_path / "evals"
    assert observed["measure_root"] == tmp_path
    assert result.kind == "eval"


def test_evaluate_main_loads_all_configs_before_preflight_or_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.evaluate as entrypoint

    first = SimpleNamespace()
    events: list[str] = []

    def load(path: str):
        events.append(f"load:{path}")
        if path == "bad.json":
            raise ValueError("invalid schema")
        return first

    monkeypatch.setattr(entrypoint.RunConfig, "load", load)
    monkeypatch.setattr(
        entrypoint, "preflight", lambda configs: events.append("preflight")
    )
    monkeypatch.setattr(
        entrypoint,
        "evaluate",
        lambda *args, **kwargs: events.append("run"),
        raising=False,
    )
    monkeypatch.setattr(entrypoint, "load_dotenv", lambda: events.append("dotenv"))

    with pytest.raises(ValueError, match="invalid schema"):
        entrypoint.main(["first.json", "bad.json"])

    assert events == ["dotenv", "load:first.json", "load:bad.json"]


def test_evaluate_main_keeps_going_indexes_and_uses_callback_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.evaluate as entrypoint

    evals_root = tmp_path / "evals"
    first = _config("first.json")
    second = _config("second.json")
    configs = {"first.json": first, "second.json": second}
    runs: list[str] = []
    results: dict[str, ExperimentResult | MeasurementError] = {
        "first.json": MeasurementError(
            "worker exited\ndiagnostic detail",
            result=_result("exp-crashed", crash_reason="worker exited"),
        ),
        "second.json": _result("exp-ok", solved=("task-a",)),
    }

    def run_isolated_experiment(**kwargs):
        config_path = kwargs["config_path"]
        runs.append(config_path)
        outcome = results[config_path]
        if isinstance(outcome, MeasurementError):
            raise outcome
        return outcome

    monkeypatch.setattr(entrypoint, "invoker_repo_root", lambda: tmp_path)
    monkeypatch.setattr(entrypoint.RunConfig, "load", lambda path: configs[path])
    preflight_calls: list[list[RunConfig]] = []
    monkeypatch.setattr(entrypoint, "preflight", preflight_calls.append)
    monkeypatch.setattr(entrypoint, "run_isolated_experiment", run_isolated_experiment)
    monkeypatch.setattr(entrypoint, "load_dotenv", lambda: None)

    exit_code = entrypoint.main(["first.json", "second.json"])

    assert exit_code == 1
    assert preflight_calls == [[first, second]]
    assert runs == ["first.json", "second.json"]
    rows = RunStore(evals_root).read_index()
    assert [(row.experiment_id, row.kind) for row in rows] == [
        ("exp-crashed", "eval"),
        ("exp-ok", "eval"),
    ]
    output = capsys.readouterr().out
    assert "eval first.json · FAILED — worker exited" in output
    assert "diagnostic detail" not in output
    assert "eval second.json · solved 1/2 · 5s" in output
    assert "first.json:" not in output
    assert "second.json:" not in output
    rows = [" ".join(line.split()) for line in output.splitlines()]
    assert "summary · 2 runs · 5s" in rows
    assert "exp-crashed 0/2 5s CRASHED · verified_rejected 2" in rows
    assert "exp-ok 1/2 5s verified_rejected 1" in rows


def test_evaluate_main_two_successes_exit_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.evaluate as entrypoint

    configs = [_config("first.json"), _config("second.json")]
    monkeypatch.setattr(entrypoint, "invoker_repo_root", lambda: tmp_path)
    monkeypatch.setattr(entrypoint.RunConfig, "load", lambda path: configs.pop(0))
    monkeypatch.setattr(entrypoint, "preflight", lambda configs: None)
    monkeypatch.setattr(
        entrypoint,
        "run_isolated_experiment",
        lambda **kwargs: _result(f"exp-{Path(kwargs['config_path']).stem}"),
    )
    monkeypatch.setattr(entrypoint, "load_dotenv", lambda: None)

    assert entrypoint.main(["first.json", "second.json"]) == 0


def test_evaluate_main_names_each_config_run_for_each_requested_iteration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.evaluate as entrypoint

    configs = {path: _config(path) for path in ("first.json", "second.json")}
    runs: list[tuple[str, str]] = []

    def fake_evaluate(config, **kwargs):
        runs.append((config.config_path, kwargs["experiment_id"]))
        return _result(kwargs["experiment_id"])

    monkeypatch.setattr(entrypoint, "invoker_repo_root", lambda: tmp_path)
    monkeypatch.setattr(entrypoint.RunConfig, "load", configs.__getitem__)
    monkeypatch.setattr(entrypoint, "preflight", lambda configs: None)
    monkeypatch.setattr(entrypoint, "_evaluate", fake_evaluate)
    monkeypatch.setattr(entrypoint, "load_dotenv", lambda: None)

    assert entrypoint.main(["--n", "2", "first.json", "second.json"]) == 0
    assert [(path, name.rsplit("__iteration-", 1)[1]) for path, name in runs] == [
        ("first.json", "1"),
        ("second.json", "1"),
        ("first.json", "2"),
        ("second.json", "2"),
    ]
    assert all(name.startswith("exp-") for _, name in runs)
    assert len({name for _, name in runs}) == 4


def test_evaluate_main_overlaps_tail_and_reorders_conflicting_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.evaluate as entrypoint

    config = _config("first.json").model_copy(
        update={
            "environment": EnvironmentConfig(
                kind="swe", task_names=["t1", "t2", "t3", "t4"]
            ),
            "max_rollout_concurrency": 2,
        }
    )
    second_started = threading.Event()
    calls: list[tuple[str, ...]] = []
    calls_lock = threading.Lock()

    def fake_run_isolated_experiment(**kwargs):
        with calls_lock:
            call_index = len(calls)
            calls.append(tuple(kwargs["task_ids"]))
        experiment_id = f"exp-{call_index}"
        kwargs["tracker"].run_dir(experiment_id).mkdir(parents=True)
        observer = kwargs["observer"]
        if call_index == 0:
            observer.task_finished("t2", "solved")
            observer.task_finished("t3", "solved")
            assert not second_started.wait(timeout=0.2)
            observer.task_finished("t4", "solved")
            assert second_started.wait(timeout=10.0), "launch gate never opened"
            observer.task_finished("t1", "solved")
        else:
            second_started.set()
        return _result(experiment_id)

    monkeypatch.setattr(entrypoint, "invoker_repo_root", lambda: tmp_path)
    monkeypatch.setattr(entrypoint.RunConfig, "load", lambda path: config)
    monkeypatch.setattr(entrypoint, "preflight", lambda configs: None)
    monkeypatch.setattr(
        entrypoint, "run_isolated_experiment", fake_run_isolated_experiment
    )
    monkeypatch.setattr(entrypoint, "load_dotenv", lambda: None)

    assert entrypoint.main(["--n", "2", "first.json"]) == 0
    assert calls == [("t1", "t2", "t3", "t4"), ("t2", "t3", "t4", "t1")]


def test_evaluate_main_immediately_overlaps_panel_below_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.evaluate as entrypoint

    configs = {
        path: _config(path).model_copy(update={"max_rollout_concurrency": 3})
        for path in ("first.json", "second.json")
    }
    second_started = threading.Event()

    def fake_run_isolated_experiment(**kwargs):
        config_path = kwargs["config_path"]
        kwargs["observer"].experiment_started(f"exp-{Path(config_path).stem}")
        if config_path == "first.json":
            assert second_started.wait(timeout=0.2), "initial launch gate never opened"
        else:
            second_started.set()
        return _result(f"exp-{Path(config_path).stem}")

    monkeypatch.setattr(entrypoint, "invoker_repo_root", lambda: tmp_path)
    monkeypatch.setattr(entrypoint.RunConfig, "load", configs.__getitem__)
    monkeypatch.setattr(entrypoint, "preflight", lambda configs: None)
    monkeypatch.setattr(
        entrypoint, "run_isolated_experiment", fake_run_isolated_experiment
    )
    monkeypatch.setattr(entrypoint, "load_dotenv", lambda: None)

    assert entrypoint.main(["first.json", "second.json"]) == 0


def test_evaluate_parent_only_sigint_cleans_up_overlapping_runs(
    tmp_path: Path,
) -> None:
    script = textwrap.dedent(
        """
        import subprocess
        import sys
        import threading
        from pathlib import Path
        from types import SimpleNamespace

        import scripts.evaluate as entrypoint

        root = Path(sys.argv[1])
        children = []

        def fake_run_isolated_experiment(**kwargs):
            index = len(children) + 1
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            children.append(child)
            (root / f"child-{index}.pid").write_text(str(child.pid))
            (root / f"thread-{index}").write_text(threading.current_thread().name)
            try:
                kwargs["observer"].experiment_started(f"exp-{index}")
                if index == 2:
                    (root / "both-started").write_text("yes")
                threading.Event().wait()
            finally:
                child.terminate()
                child.wait(timeout=5)
                (root / f"cleanup-{index}").write_text("yes")

        entrypoint.run_isolated_experiment = fake_run_isolated_experiment
        entrypoint.load_dotenv = lambda: None
        entrypoint.preflight = lambda configs: None
        configs = [
            SimpleNamespace(
                environment=SimpleNamespace(
                    task_names=(f"task-{index}",), kind="swe"
                ),
                max_rollout_concurrency=2,
                config_path=f"config-{index}.yaml",
                llm_provider_config=SimpleNamespace(
                    model_name="model", base_url="http://localhost:8080/v1"
                ),
                training_target=SimpleNamespace(surface="src/policy/core.py"),
            )
            for index in (1, 2)
        ]
        entrypoint.RunConfig.load = lambda path: configs[
            int(Path(path).stem.rsplit("-", 1)[1]) - 1
        ]
        entrypoint.main(["config-1.yaml", "config-2.yaml"])
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child_pids: list[int] = []
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "both-started").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (tmp_path / "both-started").exists(), "overlapping run did not start"
        child_pids = [
            int((tmp_path / f"child-{index}.pid").read_text()) for index in (1, 2)
        ]

        os.kill(process.pid, signal.SIGINT)
        _stdout, stderr = process.communicate(timeout=10)

        assert process.returncode == -signal.SIGINT, stderr
        assert [(tmp_path / f"thread-{index}").read_text() for index in (1, 2)] == [
            "MainThread",
            "MainThread",
        ]
        assert all(
            (tmp_path / f"cleanup-{index}").read_text() == "yes" for index in (1, 2)
        )
        for child_pid in child_pids:
            with pytest.raises(ProcessLookupError):
                os.kill(child_pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        for child_pid in child_pids:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_evaluate_main_keyboard_interrupt_stops_before_later_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.evaluate as entrypoint

    configs = {
        path: _config(path) for path in ("first.json", "second.json", "later.json")
    }
    started: list[str] = []

    def fake_evaluate(config, **kwargs):
        del kwargs
        started.append(config.config_path)
        if config.config_path == "second.json":
            raise KeyboardInterrupt

    monkeypatch.setattr(entrypoint.RunConfig, "load", configs.__getitem__)
    monkeypatch.setattr(entrypoint, "preflight", lambda configs: None)
    monkeypatch.setattr(entrypoint, "_evaluate", fake_evaluate)
    monkeypatch.setattr(entrypoint, "load_dotenv", lambda: None)

    with pytest.raises(KeyboardInterrupt):
        entrypoint.main(["first.json", "second.json", "later.json"])

    assert started == ["first.json", "second.json"]


@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_evaluate_main_rejects_invalid_iterations(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluate as entrypoint

    monkeypatch.setattr(entrypoint, "load_dotenv", lambda: None)
    monkeypatch.setattr(
        entrypoint,
        "preflight",
        lambda configs: pytest.fail("invalid iterations must fail before preflight"),
    )

    with pytest.raises(ValueError, match="--n must be a positive integer"):
        entrypoint.main(["--n", value, "config.json"])


def test_evaluate_hands_scheduled_experiment_id_to_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluate as entrypoint

    scheduled_id = "exp-model__task-panel__iteration-2__20260101-000000"
    observed: dict[str, object] = {}
    result = _result(scheduled_id)
    monkeypatch.setattr(entrypoint, "invoker_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        entrypoint,
        "run_isolated_experiment",
        lambda **kwargs: observed.update(kwargs) or result,
    )

    actual = entrypoint.evaluate(_config(), experiment_id=scheduled_id)

    assert observed["experiment_id"] == scheduled_id
    assert actual.experiment_id == scheduled_id
    assert RunStore(tmp_path / "evals").read_index()[0].experiment_id == scheduled_id


class _FrozenDatetime:
    @staticmethod
    def now(tz):
        from datetime import datetime

        return datetime(2026, 1, 1, tzinfo=tz)


def _capture_scheduled_names(
    entrypoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, names: list[str]
) -> None:
    monkeypatch.setattr(entrypoint, "datetime", _FrozenDatetime)
    monkeypatch.setattr(entrypoint, "invoker_repo_root", lambda: tmp_path)
    monkeypatch.setattr(entrypoint, "preflight", lambda configs: None)
    monkeypatch.setattr(entrypoint, "load_dotenv", lambda: None)

    def fake_evaluate(config, **kwargs):
        name = kwargs["experiment_id"]
        names.append(name)
        (tmp_path / "evals" / name).mkdir(parents=True)
        return _result(name)

    monkeypatch.setattr(entrypoint, "_evaluate", fake_evaluate)


def test_scheduled_run_names_avoid_issued_and_on_disk_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.evaluate as entrypoint

    names: list[str] = []
    _capture_scheduled_names(entrypoint, tmp_path, monkeypatch, names)
    config = _config()
    monkeypatch.setattr(entrypoint.RunConfig, "load", lambda path: config)

    assert entrypoint.main(["--n", "1", "a.json", "b.json"]) == 0
    assert entrypoint.main(["--n", "1", "a.json"]) == 0

    base = "exp-20260101-000000-000000__iteration-1"
    assert names == [base, f"{base}-2", f"{base}-3"]


def test_evaluate_main_defaults_to_eval_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.evaluate as entrypoint

    config = _config()
    loaded: list[str] = []
    monkeypatch.setattr(entrypoint, "invoker_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        entrypoint.RunConfig,
        "load",
        lambda path: loaded.append(path) or config,
    )
    monkeypatch.setattr(entrypoint, "preflight", lambda configs: None)
    monkeypatch.setattr(entrypoint, "_evaluate", lambda *args, **kwargs: None)
    monkeypatch.setattr(entrypoint, "load_dotenv", lambda: None)

    assert entrypoint.main([]) == 0
    assert loaded == [str(tmp_path / "config/quickstart_eval.yaml")]


def test_preflight_docker_unreachable_names_daemon_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class DockerClient:
        def ping(self) -> bool:
            raise docker.errors.DockerException("daemon unavailable")

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(docker, "from_env", lambda **kwargs: DockerClient())

    with pytest.raises(
        PreflightError,
        match="Docker daemon is not reachable -- start Docker/OrbStack",
    ):
        measurement.preflight([_config()])

    assert events == ["close"]


def test_preflight_probes_docker_through_runtime_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class DockerClient:
        def ping(self) -> bool:
            events.append("ping")
            return True

        def close(self) -> None:
            events.append("close")

    def from_env(*, timeout: int) -> DockerClient:
        events.append(("from_env", timeout))
        return DockerClient()

    monkeypatch.setattr(docker, "from_env", from_env)
    monkeypatch.setattr(
        measurement.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("preflight used the Docker CLI"),
    )
    monkeypatch.setattr(
        measurement, "_assert_llm_provider_reachable", lambda config: None
    )

    measurement.preflight([_config()])

    assert events == [("from_env", 10), "ping", "close"]


def test_preflight_smoke_checks_identical_provider_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        docker,
        "from_env",
        lambda **kwargs: SimpleNamespace(ping=lambda: True, close=lambda: None),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    smoke_checks: list[RunConfig] = []
    monkeypatch.setattr(
        measurement, "_assert_llm_provider_reachable", smoke_checks.append
    )
    first = _config()
    second = _config().model_copy(
        update={"environment": EnvironmentConfig(kind="swe", task_names=["task-b"])}
    )

    measurement.preflight([first, second])

    assert smoke_checks == [first]


def test_preflight_smoke_request_has_120_second_outer_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StalledBackend:
        closed = False

        async def complete(self, request):
            del request
            await asyncio.sleep(0.05)
            raise AssertionError("smoke request outlived its outer deadline")

        async def close(self) -> None:
            self.closed = True

    backend = StalledBackend()
    assert measurement._SMOKE_TIMEOUT_SEC == 120.0
    monkeypatch.setattr(measurement, "_SMOKE_TIMEOUT_SEC", 0.001)
    monkeypatch.setattr(measurement, "make_backend", lambda config: backend)

    with pytest.raises(PreflightError, match="TimeoutError"):
        measurement._assert_llm_provider_reachable(_config())

    assert backend.closed is True
