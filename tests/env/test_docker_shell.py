"""Unit tests for docker_shell capture bounding.

Regression for exp-20260703-231416-219263/make-mips-interpreter: a 32MB stdout
was serialized verbatim into a 194MB steps.jsonl line, stalling the event loop
across all concurrent rollouts.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from docker.models.containers import ContainerCollection
from docker.types import EndpointConfig

from src.env.base import RunAction
from src.env.docker_shell import (
    BINARY_STDOUT_PLACEHOLDER,
    COMMAND_TIMEOUT_EXIT_CODE,
    COMMAND_TIMEOUT_KILL_GRACE_SEC,
    MAX_CAPTURED_STREAM_BYTES,
    DockerShellSession,
    ExecResult,
    ManagedDockerNetwork,
    _bounded_decode,
    run_step,
)
from src import determinism
from src.determinism import MTIME_RESET_COMMAND


def test_bounded_decode_passes_small_output_through_verbatim():
    assert _bounded_decode(b"hello\n") == "hello\n"
    assert _bounded_decode(None) == ""
    assert _bounded_decode(b"") == ""


def test_bounded_decode_caps_oversized_output_and_marks_it():
    raw = b"x" * (MAX_CAPTURED_STREAM_BYTES + 100)
    text = _bounded_decode(raw)
    assert text.startswith("x" * 100)
    assert text.endswith(f"...[env output truncated: {len(raw)} bytes total]")
    assert len(text) < MAX_CAPTURED_STREAM_BYTES + 100


class _FakeSession:
    """A DockerShellSession stand-in that returns a canned ExecResult."""

    def __init__(self, result: ExecResult) -> None:
        self._result = result
        self.calls: list[tuple[str, str | None, float | None]] = []

    async def run(self, *, command: str, cwd: str | None, timeout: float | None):
        self.calls.append((command, cwd, timeout))
        return self._result


def test_run_step_timed_out_command_returns_nonterminal_observation():
    # A timed-out command returns an observation without ending the rollout.
    session = _FakeSession(
        ExecResult(exit_code=COMMAND_TIMEOUT_EXIT_CODE, stdout="partial out", stderr="")
    )
    action = RunAction(command="sleep 999", cwd=None, timeout_sec=60)

    result = asyncio.run(run_step(session, action, default_cwd="/app"))  # type: ignore[arg-type]

    assert result.exit_code == COMMAND_TIMEOUT_EXIT_CODE
    assert result.stdout == "partial out"
    assert "timed out" in result.stderr.lower()
    assert "60" in result.stderr


def test_run_step_normal_exit_carries_no_timeout_note():
    session = _FakeSession(ExecResult(exit_code=0, stdout="ok", stderr="warn"))
    action = RunAction(command="true", cwd=None, timeout_sec=60)

    result = asyncio.run(run_step(session, action, default_cwd="/app"))  # type: ignore[arg-type]

    assert result.stderr == "warn"
    assert "timed out" not in result.stderr.lower()


class _FakeExec:
    def __init__(self, exit_code: int, output: tuple[bytes, bytes]) -> None:
        self.exit_code = exit_code
        self.output = output


class _FakeContainer:
    def __init__(self, exec_result: _FakeExec) -> None:
        self._exec_result = exec_result
        self.exec_argv: list[str] | None = None
        self.exec_calls: list[list[str]] = []
        self.exec_kwargs: list[dict] = []
        self.removed = False

    def exec_run(self, cmd, **kwargs):
        self.exec_argv = cmd
        self.exec_calls.append(cmd)
        self.exec_kwargs.append(kwargs)
        return self._exec_result

    def remove(self, **kwargs):
        self.removed = True


def _session_with_container(container: _FakeContainer) -> DockerShellSession:
    class _Containers:
        def get(self, name):
            return container

    class _Client:
        containers = _Containers()

    session = object.__new__(DockerShellSession)
    session._docker_client = _Client()  # type: ignore[attr-defined]
    session.container_name = "c"  # type: ignore[attr-defined]
    session._started = True  # type: ignore[attr-defined]
    session._run_environment = determinism.SOLVE_EXEC_ENV  # type: ignore[attr-defined]
    session._best_effort_warned_commands = set()  # type: ignore[attr-defined]
    return session


def test_run_enforces_timeout_in_container_and_keeps_it_alive():
    # Enforce command timeouts in-container so the container survives.
    container = _FakeContainer(_FakeExec(COMMAND_TIMEOUT_EXIT_CODE, (b"", b"")))
    session = _session_with_container(container)

    result = asyncio.run(session.run(command="sleep 999", cwd="/app", timeout=60))

    assert container.exec_argv is not None
    assert container.exec_argv[0] == "timeout"
    assert str(COMMAND_TIMEOUT_KILL_GRACE_SEC) in container.exec_argv
    assert "60" in container.exec_argv
    assert container.exec_argv[-3:] == ["bash", "-lc", "sleep 999"]
    assert container.exec_calls[0] == ["bash", "-lc", MTIME_RESET_COMMAND]
    assert container.removed is False
    assert session._started is True
    assert result.exit_code == COMMAND_TIMEOUT_EXIT_CODE


def test_run_without_timeout_does_not_wrap():
    # Verify (timeout=None) must run the command directly; its bound is applied a
    # layer up (verify_timeout_sec), not in-container.
    container = _FakeContainer(_FakeExec(0, (b"ok", b"")))
    session = _session_with_container(container)

    asyncio.run(session.run(command="run-verifier", cwd="/app", timeout=None))

    assert container.exec_argv == ["bash", "-lc", "run-verifier"]


def test_run_suppresses_binary_stdout():
    container = _FakeContainer(_FakeExec(0, (b"/\xff\x00A\x7fEXIT: 0\n", b"warn")))
    session = _session_with_container(container)

    result = asyncio.run(session.run(command="emit-binary", cwd="/app", timeout=30))

    assert result.stdout == BINARY_STDOUT_PLACEHOLDER
    assert result.stderr == "warn"


def test_control_run_is_lossless_while_model_visible_run_stays_capped():
    raw = b"x" * (MAX_CAPTURED_STREAM_BYTES + 100)
    container = _FakeContainer(_FakeExec(0, (raw, raw)))
    session = _session_with_container(container)

    control_result = asyncio.run(
        session.run(command="git diff", cwd="/app", timeout=30, lossless=True)
    )
    model_result = asyncio.run(session.run(command="git diff", cwd="/app", timeout=30))

    assert control_result.stdout == raw.decode()
    assert control_result.stderr == raw.decode()
    assert model_result.stdout.endswith(
        f"...[env output truncated: {len(raw)} bytes total]"
    )
    assert model_result.stderr.endswith(
        f"...[env output truncated: {len(raw)} bytes total]"
    )


def test_run_stages_multiline_command_outside_docker_exec_argv():
    command = "cat > plan.py << 'EOF'\nprint('ok')\nEOF\npython3 plan.py"
    container = _FakeContainer(_FakeExec(0, (b"ok\n", b"")))
    session = _session_with_container(container)

    result = asyncio.run(session.run(command=command, cwd="/app", timeout=10))

    run_argv = container.exec_calls[1]
    run_env = container.exec_kwargs[1]["environment"]
    assert run_argv[:4] == ["timeout", "-k", str(COMMAND_TIMEOUT_KILL_GRACE_SEC), "10"]
    assert run_argv[-3:] == [
        "bash",
        "-lc",
        'exec env -u FRAMEWORK_EXEC_COMMAND bash -lc "$FRAMEWORK_EXEC_COMMAND"',
    ]
    assert all(command not in arg for arg in run_argv)
    assert run_env["FRAMEWORK_EXEC_COMMAND"] == command
    assert result.stdout == "ok\n"


def test_best_effort_nonzero_logs_once_per_distinct_command(caplog):
    container = _FakeContainer(_FakeExec(7, (b"", b"pin failed\n")))
    session = _session_with_container(container)

    with caplog.at_level("WARNING", logger="src.env.docker_shell"):
        asyncio.run(session._exec_best_effort(container, "pin-control"))
        asyncio.run(session._exec_best_effort(container, "pin-control"))
        asyncio.run(session._exec_best_effort(container, "pin-control-2"))

    records = [
        record
        for record in caplog.records
        if "best-effort determinism pin" in record.message
    ]
    assert len(records) == 2
    assert "container=c" in records[0].message
    assert "exit_code=7" in records[0].message
    assert "pin failed" in records[0].message


def test_best_effort_exception_logs_and_never_raises(monkeypatch, caplog):
    container = _FakeContainer(_FakeExec(0, (b"", b"")))
    session = _session_with_container(container)

    async def boom(_container, _command):
        raise RuntimeError("control unavailable")

    monkeypatch.setattr(session, "_exec_control_command", boom)

    with caplog.at_level("WARNING", logger="src.env.docker_shell"):
        asyncio.run(session._exec_best_effort(container, "pin-control"))
        asyncio.run(session._exec_best_effort(container, "pin-control"))

    records = [
        record
        for record in caplog.records
        if "best-effort determinism pin" in record.message
    ]
    assert len(records) == 1
    assert "RuntimeError('control unavailable')" in records[0].message


def test_run_backstop_destroys_container_when_exec_wedges(monkeypatch):
    # Force-remove the container if the in-container timeout wedges.
    container = _FakeContainer(_FakeExec(0, (b"", b"")))
    session = _session_with_container(container)

    async def _noop_best_effort(*_args, **_kwargs):
        return None

    monkeypatch.setattr(session, "_exec_best_effort", _noop_best_effort)

    async def _wedge(awaitable, _timeout):
        awaitable.close()  # never awaited; avoid a warning
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", _wedge)

    with pytest.raises(TimeoutError):
        asyncio.run(session.run(command="sleep 999", cwd="/app", timeout=60))

    assert container.removed is True
    assert session._started is False


def test_managed_network_pins_the_requested_static_ip():
    """The requested ipv4_address must survive docker-py's kwargs translation.

    Regression: the endpoint config was pre-wrapped in a NetworkingConfig, whose
    only key is "EndpointsConfig". docker-py's high-level `containers.run` wants
    a plain {network_name: EndpointConfig} dict and checks that the network is a
    key of it, so the pre-wrapped form was silently dropped and the container got
    a DHCP address instead of the pinned one.
    """

    class _StopAfterCreate(Exception):
        pass

    client = MagicMock()
    # Pin the outcome, not the constructor: both routes to an EndpointConfig work.
    client.api._version = "1.44"
    client.api.create_endpoint_config = lambda **kw: EndpointConfig("1.44", **kw)
    client.api.create_container.side_effect = _StopAfterCreate
    client.containers = ContainerCollection(client=client)

    session = object.__new__(DockerShellSession)
    session._docker_client = client
    session.container_name = "tb_env_abc"
    session._image = "img"
    session._read_only_binds = {}
    session._cpu_limit = None
    session._memory_limit_mb = None
    session._managed_network_obj = None
    session._managed_network = ManagedDockerNetwork(
        subnet="10.240.7.0/24", ipv4_address="10.240.7.9"
    )

    with pytest.raises(_StopAfterCreate):
        session._run_container_sync()

    create_kwargs = client.api.create_container.call_args.kwargs
    endpoints = create_kwargs["networking_config"]["EndpointsConfig"]
    assert endpoints["tb_env_abc_net"]["IPAMConfig"]["IPv4Address"] == "10.240.7.9"
    # The container must also actually join that network.
    assert create_kwargs["host_config"]["NetworkMode"] == "tb_env_abc_net"
