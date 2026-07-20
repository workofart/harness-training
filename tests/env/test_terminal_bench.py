from __future__ import annotations

import asyncio
import io
import os
import re
import subprocess
import sys
import tarfile
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from docker.types import EndpointConfig

import src.env.terminal_bench as tb_module
import src.env.docker_shell as docker_shell_module
import src.env.netcache as netcache
import src.plugins.replay.contract as contract
from src.env.base import (
    RawEnvOutput,
    RunAction,
    VerifyAction,
    VerifyVerdict,
    execute_env_action,
    scrub_step_result,
)
from src.env.docker_shell import DockerShellSession, ExecResult, ManagedDockerNetwork
from src.env.base import TaskSet
from src.env.terminal_bench import (
    TerminalBenchEnv,
    TerminalBenchTask,
)
from src.env.base import StepResult
from src import determinism
from src.config import EnvironmentConfig


@pytest.fixture(autouse=True)
def _host_caches_already_started(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests from running a real `docker compose up` on env start."""
    monkeypatch.setattr(netcache, "_host_caches_started", True)


def test_taskset_loads_local_terminal_bench_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    taskset, env, created_sessions = _make_env(
        monkeypatch,
        tmp_path,
        "write-compressor",
        cpus=2,
        memory_mb=4096,
        verifier_timeout_sec=1234.0,
        agent_timeout_sec=2345.0,
        build_timeout_sec=3456.0,
    )

    assert env._task.task_name == "terminal-bench/write-compressor"
    assert env._task.instruction == "Compress data.\n"
    assert env.verify_timeout_sec == 1234.0
    assert env.setup_timeout_sec == 3456.0
    assert taskset.tasks["write-compressor"].agent_timeout_sec == 2345.0
    assert taskset.tasks["write-compressor"].replay_id == tb_module._env_fingerprint()
    assert created_sessions == [
        {
            "image": "example/write-compressor:latest",
            "managed_network": ManagedDockerNetwork(
                subnet="10.240.0.0/24",
                ipv4_address="10.240.0.2",
            ),
            "extra_run_environment": None,
            "read_only_binds": None,
            "name_prefix": "terminal_bench_env",
            "cpu_limit": 2,
            "memory_limit_mb": 4096,
            "setup_timeout_sec": 3456.0,
            "setup_command": (
                "mkdir -p /app\n"
                "export FRAMEWORK_REQUIRE_CACHES=0\n"
                + netcache.CACHE_PROXY_SETUP_SCRIPT.read_text()
            ),
        }
    ]
    assert "FRAMEWORK_NETWORK_CACHE_SCOPE=" not in created_sessions[0]["setup_command"]


def test_env_fingerprint_uses_source_pins_not_hand_copied_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = []
    original_dumps = tb_module.json.dumps

    def capture_dumps(payload, **kwargs):
        payloads.append(payload)
        return original_dumps(payload, **kwargs)

    tb_module._env_fingerprint.cache_clear()
    monkeypatch.setattr(tb_module.json, "dumps", capture_dumps)
    original = tb_module._env_fingerprint()

    revision_inputs = [
        (determinism, "PINS_FINGERPRINT", "abc123def456"),
        (
            tb_module,
            "_WORKDIR_SETUP_COMMAND",
            f"{tb_module._WORKDIR_SETUP_COMMAND} # changed",
        ),
        (tb_module, "_LIVE_ONLY_TASK_NAMES", frozenset({"example-live-only"})),
        (tb_module, "_RESOURCE_ENFORCEMENT_VERSION", 2),
    ]
    for target, name, value in revision_inputs:
        tb_module._env_fingerprint.cache_clear()
        with monkeypatch.context() as patch:
            patch.setattr(target, name, value)
            tb_module._env_fingerprint.cache_clear()
            changed = tb_module._env_fingerprint()
        tb_module._env_fingerprint.cache_clear()
        assert changed != original
        assert tb_module._env_fingerprint() == original

    tb_module._env_fingerprint.cache_clear()
    assert payloads[0]["source_pins"] == determinism.PINS_FINGERPRINT


def test_terminal_bench_task_network_is_panel_independent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "tb"
    _write_task(root, "alpha")
    _write_task(root, "write-compressor")
    _write_task(root, "zeta")

    one_task = _load_taskset(monkeypatch, root, "write-compressor")
    panel = _load_taskset(
        monkeypatch,
        root,
        "alpha",
        "write-compressor",
        "zeta",
    )

    assert one_task.tasks["write-compressor"].network == ManagedDockerNetwork(
        subnet="10.240.1.0/24",
        ipv4_address="10.240.1.2",
    )
    assert (
        one_task.tasks["write-compressor"].network
        == panel.tasks["write-compressor"].network
    )


def test_distribution_search_enables_fakerandom_setup_and_run_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _taskset, _env, created_sessions = _make_env(
        monkeypatch, tmp_path, "distribution-search"
    )

    setup_command = created_sessions[0]["setup_command"]
    assert setup_command is not None
    assert created_sessions[0]["extra_run_environment"] == {
        "LD_PRELOAD": "/opt/framework/libfaketimeMT.so.1",
        "FAKERANDOM_SEED": "0x12345678DEADBEEF",
    }
    assert created_sessions[0]["read_only_binds"] == {
        str(tb_module._FAKERANDOM_LIB_HOST_PATH): "/opt/framework/libfaketimeMT.so.1",
    }
    assert tb_module._FAKERANDOM_LIB_HOST_PATH.is_file()


def _fake_deb(lib_bytes: bytes) -> bytes:
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:xz") as tar:
        info = tarfile.TarInfo(tb_module._FAKERANDOM_DEB_MEMBER)
        info.size = len(lib_bytes)
        tar.addfile(info, io.BytesIO(lib_bytes))
    members = [(b"debian-binary", b"2.0\n"), (b"data.tar.xz", tar_buf.getvalue())]
    ar = io.BytesIO(b"!<arch>\n")
    ar.seek(0, io.SEEK_END)
    for name, data in members:
        header = b"%-16s%-12s%-6s%-6s%-8s%-10d`\n" % (
            name,
            b"0",
            b"0",
            b"0",
            b"100644",
            len(data),
        )
        ar.write(header + data + (b"\n" if len(data) & 1 else b""))
    return ar.getvalue()


def test_fakerandom_lib_fetched_and_sha_verified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import hashlib
    import contextlib

    lib_bytes = b"\x7fELF fake shim bytes"  # odd length exercises ar padding
    deb = _fake_deb(lib_bytes)
    target = tmp_path / "libfaketimeMT.so.1"
    monkeypatch.setattr(tb_module, "_FAKERANDOM_LIB_HOST_PATH", target)
    monkeypatch.setattr(
        tb_module, "_FAKERANDOM_LIB_SHA256", hashlib.sha256(lib_bytes).hexdigest()
    )
    monkeypatch.setattr(
        tb_module, "_FAKERANDOM_DEB_SHA256", hashlib.sha256(deb).hexdigest()
    )
    monkeypatch.setattr(
        tb_module.urllib.request,
        "urlopen",
        lambda url, timeout: contextlib.closing(io.BytesIO(deb)),
    )

    assert tb_module._fakerandom_lib_path() == target
    assert target.read_bytes() == lib_bytes

    # An on-disk file whose bytes drifted must fail loudly, not fork the cache.
    target.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="sha256"):
        tb_module._fakerandom_lib_path()

    # A wrong .deb payload is rejected before anything is written.
    target.unlink()
    monkeypatch.setattr(tb_module, "_FAKERANDOM_DEB_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="sha256"):
        tb_module._fakerandom_lib_path()
    assert not target.exists()


def test_portfolio_optimization_loads_without_env_cache_surface(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    taskset, env, _sessions = _make_env(monkeypatch, tmp_path, "portfolio-optimization")

    assert env._task.task_name == "terminal-bench/portfolio-optimization"
    assert taskset.tasks["portfolio-optimization"].replay_id is None


def test_cache_proxy_setup_contract() -> None:
    setup_command = netcache.CACHE_PROXY_SETUP_SCRIPT.read_text()
    required = (
        'Acquire::Retries "2";',
        'Acquire::http::Timeout "30";',
        'Acquire::https::Timeout "30";',
        "/etc/dpkg/dpkg.cfg.d/00-framework-unsafe-io",
        "force-unsafe-io",
        "require_cache uv-python-mirror 3143",
        "/etc/profile.d/framework-uv-mirror.sh",
        "UV_PYTHON_INSTALL_MIRROR=http://host.docker.internal:3143",
        "HF_HUB_VERBOSITY=error",
        '[ "$network_freeze_enabled" = true ]',
        '[ "${FRAMEWORK_REQUIRE_CACHES:-0}" = "1" ]',
        "require_cache https-proxy 3144",
        "require_cache https-ca 3145",
        "require_cache pypi 3141",
        "/mitmproxy-ca-cert.pem",
        "/usr/local/share/ca-certificates/framework-https-cache.crt",
        "/etc/pki/ca-trust/source/anchors/framework-https-cache.crt",
        "update-ca-certificates",
        "update-ca-trust extract",
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/etc/profile.d/framework-https-proxy.sh",
        "https_proxy=http://${proxy_auth}host.docker.internal:3144",
        "no_proxy=localhost,127.0.0.1,::1,host.docker.internal",
        "SSL_CERT_FILE=$ca_bundle",
        "NODE_EXTRA_CA_CERTS=/usr/local/share/ca-certificates/framework-https-cache.crt",
        "FRAMEWORK_NETWORK_CACHE_SCOPE",
        'proxy_auth="framework:${FRAMEWORK_NETWORK_CACHE_SCOPE}@"',
        'Acquire::http::Proxy "http://${proxy_auth}host.docker.internal:3144"',
        'Acquire::https::Proxy "http://${proxy_auth}host.docker.internal:3144"',
        'Acquire::Check-Valid-Until "false"',
    )
    forbidden = (
        "python-install-mirror",
        "FRAMEWORK_ENABLE_FAKERANDOM",
        "apt-get install",
        "FRAMEWORK_NETWORK_CACHE_SCOPE:-global",
        "host.docker.internal:3142",
    )
    assert not [fragment for fragment in required if fragment not in setup_command]
    assert not [fragment for fragment in forbidden if fragment in setup_command]


def test_cache_proxy_installs_ca_before_writing_proxy_profile() -> None:
    setup_command = netcache.CACHE_PROXY_SETUP_SCRIPT.read_text()
    profile = setup_command.index("cat > /etc/profile.d/framework-https-proxy.sh")
    assert setup_command.index("if fetch_http_file \\\n") < setup_command.index(
        'if [ "$ca_installed" = true ]; then'
    )
    assert setup_command.index("update-ca-certificates") < profile
    assert setup_command.index("update-ca-trust extract") < profile


def test_terminal_bench_network_cache_scope_updates_setup_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _FakeSession()
    monkeypatch.setattr(tb_module, "DockerShellSession", lambda **_: session)
    env = TerminalBenchEnv(
        task=_task(tmp_path),
        artifacts_dir=tmp_path / "rollout",
        host_netcache=True,
    )

    env.pin_network_snapshot(
        contract._network_snapshot_token(namespace="terminal_bench:rev:task-a", epoch=3)
    )

    assert session.setup_command is not None
    assert "FRAMEWORK_NETWORK_CACHE_SCOPE=" in session.setup_command
    assert "terminal_bench:rev:task-a" not in session.setup_command
    assert "export FRAMEWORK_REQUIRE_CACHES=1" in session.setup_command


def test_host_netcache_off_skips_compose_and_cache_exports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _FakeSession()
    created_sessions: list[dict] = []

    def fake_session(**kwargs):
        created_sessions.append(kwargs)
        return session

    monkeypatch.setattr(tb_module, "DockerShellSession", fake_session)
    monkeypatch.setattr(
        netcache,
        "_ensure_host_caches_started",
        lambda: pytest.fail("host caches must not start"),
    )
    env = TerminalBenchEnv(
        task=_task(tmp_path),
        artifacts_dir=tmp_path / "rollout",
        host_netcache=False,
    )

    asyncio.run(env.provision())
    setup_command = created_sessions[0]["setup_command"]

    assert session.start_calls == 1
    assert "export FRAMEWORK_REQUIRE_CACHES=" not in setup_command
    assert "export FRAMEWORK_NETWORK_CACHE_SCOPE=" not in setup_command
    assert 'Acquire::Retries "2";' in setup_command


def _service_block(compose: str, service_name: str) -> str:
    lines = compose.splitlines()
    start = lines.index(f"  {service_name}:")
    end = next(
        (
            index
            for index, line in enumerate(lines[start + 1 :], start + 1)
            if re.fullmatch(r"  [a-z0-9-]+:", line)
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def test_host_cache_compose_contract() -> None:
    compose = netcache._HOST_CACHES_COMPOSE_FILE.read_text()
    required = (
        "framework-uv-python-mirror",
        "127.0.0.1:3143:8000",
        "./uv_python_mirror.py:/opt/uv_python_mirror.py:ro",
        "framework-https-cache",
        "framework-https-cache-ca",
        "--set upstream_cert=false",
        "--set connection_strategy=lazy",
        "--set proxyauth=any",
        "127.0.0.1:3144:8080",
        "127.0.0.1:3145:8000",
        ".:/opt/framework-scripts:ro",
        "-s /opt/framework-scripts/mitm_https_cache.py",
        "mitmproxy-ca-cert.pem",
        "python -m http.server 8000 -d /var/run/framework-ca",
        "FRAMEWORK_DEBIAN_MIRROR_HOST: debian.osuosl.org",
    )
    forbidden = (
        "apt-cacher-ng",
        "3142",
        "apt_index_proxy",
        ":/opt/mitm_https_cache.py",
        "mitmproxy-ca.pem",
    )
    assert not [fragment for fragment in required if fragment not in compose]
    assert not [fragment for fragment in forbidden if fragment in compose]
    pypi = _service_block(compose, "pypi-cache")
    assert "http://127.0.0.1:5000/" in pypi[pypi.index("    healthcheck:") :]
    image_lines = [
        line.strip()
        for line in compose.splitlines()
        if line.strip().startswith("image:")
    ]
    assert image_lines
    assert all("@sha256:" in line for line in image_lines)
    assert all(":latest" not in line for line in image_lines)
    assert not (
        netcache._HOST_CACHES_COMPOSE_FILE.parent / "apt-cacher-ng-harness.conf"
    ).exists()


def test_terminal_bench_provision_starts_host_caches_before_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    events: list[str] = []

    class OrderedSession(_FakeSession):
        async def start(self) -> None:
            events.append("session_start")
            await super().start()

    monkeypatch.setattr(netcache, "_host_caches_started", False)
    monkeypatch.setattr(
        netcache, "_compose_up_host_caches", lambda: events.append("caches_up")
    )
    env = _env_with_session(monkeypatch, tmp_path, OrderedSession())

    output = asyncio.run(env.reset())
    # reset stays infra-free so cache-hit replays never boot a container.
    assert events == []

    asyncio.run(env.provision())

    assert events == ["caches_up", "session_start"]
    assert output.instruction == "Compress data.\n"
    assert output.working_dir == "/app"


def test_terminal_bench_run_defaults_to_task_workdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    session = _FakeSession(run_stdout="ran\n")
    env = _env_with_session(monkeypatch, tmp_path, session)

    result = asyncio.run(env.execute(RunAction(command="pwd")))

    assert result.stdout == "ran\n"
    assert session.runs[-1] == {
        "command": "pwd",
        "cwd": "/app",
        "timeout": None,
    }


def test_terminal_bench_submit_uploads_tests_and_scores_reward(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    session = _FakeSession(
        test_stdout="1 passed\n",
        reward_stdout="1\n",
    )
    env = _env_with_session(monkeypatch, tmp_path, session)
    task = env._task

    result = asyncio.run(env.verify())

    assert result.reward == 1.0
    assert result.info == {
        "task_name": "terminal-bench/write-compressor",
        "verifier_exit_code": 1,
    }
    assert result.verdict == VerifyVerdict(completed=True, passed=True, error=None)
    assert result.output.stdout == "1 passed\n"
    assert session.uploads == [
        {
            "source_dir": task.task_dir / "tests",
            "target_dir": "/tests",
        }
    ]
    verify_run = session.runs[1]
    assert "/tests/test.sh" in verify_run["command"]
    assert "/logs/verifier/test-stdout.txt" in verify_run["command"]
    assert verify_run["timeout"] == 900.0
    verifier_dir = tmp_path / "rollout" / "terminal_bench_verifier"
    assert (verifier_dir / "test-stdout.txt").read_text() == "1 passed\n"
    assert (verifier_dir / "reward.txt").read_text() == "1\n"


def test_terminal_bench_submit_marks_incomplete_when_reward_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    session = _FakeSession(
        test_stdout="failed before reward\n",
        reward_exit_code=2,
        reward_stdout="",
    )
    env = _env_with_session(monkeypatch, tmp_path, session)

    result = asyncio.run(env.verify())

    assert result.reward == 0.0
    assert result.verdict is not None
    assert result.verdict.completed is False
    assert result.verdict.error is not None
    assert "verifier did not write" in result.verdict.error
    assert result.output.stdout == "failed before reward\n"


@pytest.mark.parametrize(
    "reward_stdout",
    [
        "0.5\n",
        '{"reward": 1}\n',
        "nan\n",
    ],
)
def test_terminal_bench_submit_rejects_invalid_reward_format(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reward_stdout: str
) -> None:
    env = _env_with_session(
        monkeypatch,
        tmp_path,
        _FakeSession(test_stdout="done\n", reward_stdout=reward_stdout),
    )

    result = asyncio.run(env.verify())

    assert result.reward == 0.0
    assert result.verdict is not None
    assert result.verdict.completed is False
    assert result.verdict.error == "reward.txt must contain exactly '0' or '1'"


def test_terminal_bench_reward_read_retries_lost_exec_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # docker exec_run can report exit 0 with lost stdout under parallel-run
    # API load; one re-read must rescue the solved verdict.
    session = _FakeSession(
        test_stdout="3 passed\n",
        reward_stdout=["", "1\n"],
    )
    env = _env_with_session(monkeypatch, tmp_path, session)

    result = asyncio.run(env.verify())

    assert result.reward == 1.0
    assert result.verdict == VerifyVerdict(completed=True, passed=True, error=None)
    reward_reads = [
        r for r in session.runs if r["command"] == "cat /logs/verifier/reward.txt"
    ]
    assert len(reward_reads) == 2


def test_terminal_bench_submit_scrubs_verifier_stdout_before_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    session = _FakeSession(
        test_stdout="===== 1 failed in 0.12s =====\n",
        reward_stdout="0\n",
    )
    env = _env_with_session(monkeypatch, tmp_path, session)

    # The episode scrubs every transition before the policy sees it; mirror that
    # here so this test observes exactly what a rollout would.
    transition = scrub_step_result(
        asyncio.run(execute_env_action(env, VerifyAction())), command=None
    )

    assert "0.12s" not in transition.raw_env_output.stdout
    assert "<DUR>" in transition.raw_env_output.stdout


def test_shared_docker_session_supports_terminal_bench_managed_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    networks = _FakeNetworks()
    session, container, containers, client = _docker_session(
        monkeypatch,
        networks=networks,
        managed_network=ManagedDockerNetwork(
            subnet="10.240.0.0/24",
            ipv4_address="10.240.0.2",
        ),
        name_prefix="terminal_bench_env",
    )

    asyncio.run(session.start())
    asyncio.run(session.close())

    assert containers.run_calls[0][0] == (
        "example/write-compressor:latest",
        ["sleep", "infinity"],
    )
    kwargs = containers.run_calls[0][1]
    assert "network_mode" not in kwargs
    assert kwargs["network"].startswith(session.container_name)
    # Unwrapped and keyed by network name: the shape `containers.run` honours.
    assert kwargs["networking_config"][kwargs["network"]]["IPAMConfig"] == {
        "IPv4Address": "10.240.0.2"
    }
    assert kwargs["hostname"] == determinism.CONTAINER_HOSTNAME
    assert networks.create_calls[0]["name"] == kwargs["network"]
    assert networks.create_calls[0]["ipam"]["Config"][0]["Subnet"] == "10.240.0.0/24"
    assert container.remove_calls == [((), {"force": True, "v": True})]
    assert networks.created[0].remove_calls == [((), {})]
    assert client.close_calls == 0


def test_shared_docker_session_applies_extra_env_only_to_run_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, container, _containers, _client = _docker_session(
        monkeypatch,
        setup_command="install faketime",
        extra_run_environment={"LD_PRELOAD": "/opt/framework/libfaketimeMT.so.1"},
    )

    asyncio.run(session.start())
    asyncio.run(session.run(command="python3 solve.py", cwd="/app", timeout=30))

    setup_env = container.exec_run_calls[0][1]["environment"]
    run_env = container.exec_run_calls[-1][1]["environment"]
    assert [args[0][2] for args, _kwargs in container.exec_run_calls[:4]] == [
        "install faketime",
        determinism.GIT_HOOKS_INIT_COMMAND,
        determinism.GDB_INIT_COMMAND,
        determinism.MTIME_RESET_COMMAND,
    ]
    assert "LD_PRELOAD" not in setup_env
    assert run_env["LD_PRELOAD"] == "/opt/framework/libfaketimeMT.so.1"


def test_shared_docker_session_required_setup_failure_is_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = _FakeContainer(
        exec_results=[
            SimpleNamespace(exit_code=7, output=(b"setup out", b"cache down\n"))
        ]
    )
    session, _container, _containers, _client = _docker_session(
        monkeypatch, container=container, setup_command="exit 7"
    )

    with pytest.raises(RuntimeError, match="cache down"):
        asyncio.run(session.start())

    assert container.remove_calls == [((), {"force": True, "v": True})]
    assert len(container.exec_run_calls) == 1


@pytest.mark.parametrize(
    ("read_only_binds", "expected_volumes"),
    [
        pytest.param(None, None, id="default-options"),
        pytest.param(
            {"/host/libfaketimeMT.so.1": "/opt/framework/libfaketimeMT.so.1"},
            {
                "/host/libfaketimeMT.so.1": {
                    "bind": "/opt/framework/libfaketimeMT.so.1",
                    "mode": "ro",
                }
            },
            id="read-only-bind",
        ),
    ],
)
def test_solve_container_run_options(
    monkeypatch: pytest.MonkeyPatch,
    read_only_binds: dict[str, str] | None,
    expected_volumes: dict[str, dict[str, str]] | None,
) -> None:
    session, _container, containers, _client = _docker_session(
        monkeypatch,
        read_only_binds=read_only_binds,
    )

    asyncio.run(session.start())

    run_options = containers.run_calls[0][1]
    assert run_options["labels"] == {
        docker_shell_module.SESSION_OWNER_PID_LABEL: str(os.getpid())
    }
    assert run_options.get("volumes") == expected_volumes
    assert ("volumes" in run_options) is (expected_volumes is not None)
    assert "nano_cpus" not in run_options
    assert "mem_limit" not in run_options


def test_solve_container_enforces_cpu_and_memory_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _container, containers, _client = _docker_session(
        monkeypatch,
        cpu_limit=2,
        memory_limit_mb=4096,
    )

    asyncio.run(session.start())

    run_options = containers.run_calls[0][1]
    assert run_options["nano_cpus"] == 2_000_000_000
    assert run_options["mem_limit"] == 4096 * 1024 * 1024


class _SweepClient:
    def __init__(self, containers, networks) -> None:
        self.list_calls: list[dict[str, object]] = []
        self.network_list_calls: list[dict[str, object]] = []
        self.close_calls = 0
        self._containers = containers
        self._networks = networks
        self.containers = SimpleNamespace(list=self._list_containers)
        self.networks = SimpleNamespace(list=self._list_networks)

    def _list_containers(self, *, all: bool, filters: dict[str, str]):  # noqa: A002
        self.list_calls.append({"all": all, "filters": filters})
        return list(self._containers)

    def _list_networks(self, *, filters: dict[str, str]):
        self.network_list_calls.append({"filters": filters})
        return list(self._networks)

    def close(self) -> None:
        self.close_calls += 1


@pytest.mark.parametrize(
    ("owner_pid", "use_shared_client", "expected_from_env", "expected_close_calls"),
    [
        pytest.param(os.getpid(), True, (), 0, id="current-process-shared-client"),
        pytest.param(
            4242,
            False,
            ({"timeout": 30},),
            1,
            id="dead-child-temporary-client",
        ),
    ],
)
def test_sweep_owner_resources_removes_pid_scoped_containers_and_networks(
    monkeypatch: pytest.MonkeyPatch,
    owner_pid: int,
    use_shared_client: bool,
    expected_from_env: tuple[dict[str, int], ...],
    expected_close_calls: int,
) -> None:
    leaked = [_FakeContainer(), _FakeContainer()]
    leaked_network = _FakeNetwork()
    client = _SweepClient(leaked, [leaked_network])
    from_env_calls: list[dict[str, object]] = []

    def fake_from_env(**kwargs):
        from_env_calls.append(kwargs)
        return client

    monkeypatch.setattr(
        DockerShellSession,
        "_shared_client",
        client if use_shared_client else None,
    )
    monkeypatch.setattr(docker_shell_module.docker, "from_env", fake_from_env)

    DockerShellSession.sweep_owner_resources(owner_pid)

    assert tuple(from_env_calls) == expected_from_env
    label = f"{docker_shell_module.SESSION_OWNER_PID_LABEL}={owner_pid}"
    assert client.list_calls == [{"all": True, "filters": {"label": label}}]
    assert all(
        container.remove_calls == [((), {"force": True, "v": True})]
        for container in leaked
    )
    assert client.network_list_calls == [{"filters": {"label": label}}]
    assert leaked_network.remove_calls == [((), {})]
    assert client.close_calls == expected_close_calls


def test_sweep_owner_resources_own_pid_without_client_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No shared client means this process never created docker resources; the
    # atexit sweep must not spin up a docker client in every exiting process.
    monkeypatch.setattr(DockerShellSession, "_shared_client", None)
    monkeypatch.setattr(
        docker_shell_module.docker,
        "from_env",
        lambda **kwargs: pytest.fail("own-pid sweep must not create a client"),
    )

    DockerShellSession.sweep_owner_resources(os.getpid())


def test_sweep_owner_resources_survives_docker_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from docker.errors import DockerException

    monkeypatch.setattr(DockerShellSession, "_shared_client", None)

    def raise_from_env(**kwargs):
        raise DockerException("docker not reachable")

    monkeypatch.setattr(docker_shell_module.docker, "from_env", raise_from_env)

    DockerShellSession.sweep_owner_resources(4242)


def test_sweep_runs_at_process_exit_even_on_keyboard_interrupt() -> None:
    # atexit covers every entrypoint and unhandled KeyboardInterrupt, which can
    # strand containers.
    script = textwrap.dedent(
        """
        import docker

        class _Container:
            def remove(self, *, force, v):
                print(f"swept container force={force} v={v}")

        class _Containers:
            def list(self, *, all, filters):
                print(f"swept label={filters['label']}")
                return [_Container()]

        class _Networks:
            def list(self, *, filters):
                print(f"swept network label={filters['label']}")
                return []

        docker.from_env = lambda: type(
            "_Client", (), {"containers": _Containers(), "networks": _Networks()}
        )()

        from src.env.docker_shell import DockerShellSession

        DockerShellSession(image="img")  # initialize shared client
        raise KeyboardInterrupt
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[2],
    )
    assert f"swept label={docker_shell_module.SESSION_OWNER_PID_LABEL}=" in proc.stdout
    assert "swept container force=True v=True" in proc.stdout
    assert "cleaned up 1 leftover solve container(s)" in proc.stdout


def test_terminal_bench_host_caches_start_once_across_envs_and_steps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    compose_calls: list[None] = []

    monkeypatch.setattr(netcache, "_host_caches_started", False)
    monkeypatch.setattr(
        netcache, "_compose_up_host_caches", lambda: compose_calls.append(None)
    )
    monkeypatch.setattr(tb_module, "DockerShellSession", lambda **_: _FakeSession())
    task = _task(tmp_path)
    first = TerminalBenchEnv(
        task=task,
        artifacts_dir=tmp_path / "rollout-1",
        host_netcache=True,
    )
    second = TerminalBenchEnv(
        task=task,
        artifacts_dir=tmp_path / "rollout-2",
        host_netcache=True,
    )

    async def run_lifecycle() -> None:
        await first.reset()
        await second.reset()
        await first.execute(RunAction(command="ls"))
        await first.close()
        await second.close()

    asyncio.run(run_lifecycle())

    assert len(compose_calls) == 1


@pytest.mark.parametrize(
    ("returncode", "stderr"),
    [
        pytest.param(0, "", id="success"),
        pytest.param(1, "no docker daemon\n", id="failure-detail"),
    ],
)
def test_compose_up_host_caches_runs_idempotent_docker_compose(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stderr: str,
):
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=returncode, stdout="", stderr=stderr)

    monkeypatch.setattr(netcache.subprocess, "run", fake_run)

    if returncode:
        with pytest.raises(RuntimeError, match="no docker daemon"):
            netcache._compose_up_host_caches()
    else:
        netcache._compose_up_host_caches()

    assert calls == [
        [
            "docker",
            "compose",
            "-f",
            str(netcache._HOST_CACHES_COMPOSE_FILE),
            "up",
            "-d",
            "--no-recreate",
            "--wait",
        ]
    ]


def test_terminal_bench_dataset_ref_is_a_pinned_commit_sha() -> None:
    # The dataset repo ships the verifier scripts; a branch ref would let the
    # graders drift silently between runs (see comment at the constant).
    assert re.fullmatch(r"[0-9a-f]{40}", tb_module._TERMINAL_BENCH_REPO_REF)


def test_terminal_bench_sync_git_dataset_uses_gitpython(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []

    class FakeRemote:
        def fetch(self, repo_ref: str, *, depth: int) -> None:
            calls.append(("fetch", repo_ref, depth))

    class FakeRepo:
        def __init__(self, path: Path) -> None:
            calls.append(("open", Path(path)))
            self.git = SimpleNamespace(
                checkout=lambda *args: calls.append(("checkout", args))
            )

        @classmethod
        def clone_from(cls, repo_url: str, target: Path, *, depth: int) -> "FakeRepo":
            target.mkdir()
            (target / ".git").mkdir()
            calls.append(("clone", repo_url, target, depth))
            return cls(target)

        def remote(self, name: str) -> FakeRemote:
            calls.append(("remote", name))
            return FakeRemote()

    monkeypatch.setattr(tb_module, "Repo", FakeRepo)
    target = tmp_path / "terminal-bench-cache"
    monkeypatch.setattr(
        tb_module,
        "_TERMINAL_BENCH_REPO_URL",
        "https://example.test/terminal-bench.git",
    )
    monkeypatch.setattr(tb_module, "_TERMINAL_BENCH_REPO_REF", "main")
    monkeypatch.setattr(tb_module, "_TERMINAL_BENCH_CACHE_DIR", target)

    result = tb_module._sync_git_dataset()

    resolved = target.resolve()
    assert result == resolved
    assert calls == [
        ("clone", "https://example.test/terminal-bench.git", resolved, 1),
        ("open", resolved),
        ("remote", "origin"),
        ("fetch", "main", 1),
        ("checkout", ("--detach", "FETCH_HEAD")),
    ]


def test_shared_docker_upload_tar_is_metadata_deterministic(tmp_path: Path) -> None:
    source_dir = tmp_path / "tests"
    source_dir.mkdir()
    test_file = source_dir / "test.sh"
    test_file.write_text("#!/bin/bash\necho ok\n")
    os.utime(test_file, (1234567890, 1234567890))
    first = DockerShellSession._tar_directory_contents(source_dir)

    os.utime(test_file, (987654321, 987654321))
    second = DockerShellSession._tar_directory_contents(source_dir)

    assert first == second
    with tarfile.open(fileobj=io.BytesIO(first), mode="r") as tar:
        member = tar.getmember("test.sh")
    assert member.uid == 0
    assert member.gid == 0
    assert member.uname == ""
    assert member.gname == ""
    assert member.mtime == 0


def _load_taskset(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    *task_ids: str,
) -> TaskSet[TerminalBenchTask]:
    monkeypatch.setattr(tb_module, "_sync_git_dataset", lambda: root)
    return asyncio.run(
        tb_module.load_tasks(
            task_ids=list(task_ids),
            environment=EnvironmentConfig(
                kind="terminal_bench", task_names=list(task_ids)
            ),
        )
    )


def _make_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    task_name: str,
    **write_task_kwargs,
) -> tuple[TaskSet[TerminalBenchTask], TerminalBenchEnv, list[dict]]:
    """Load a single-task taskset and its env, capturing DockerShellSession kwargs."""
    root = tmp_path / "tb"
    _write_task(root, task_name, **write_task_kwargs)
    created_sessions: list[dict] = []

    def fake_session(**kwargs):
        created_sessions.append(kwargs)
        return _FakeSession()

    monkeypatch.setattr(tb_module, "DockerShellSession", fake_session)
    taskset = _load_taskset(monkeypatch, root, task_name)
    env = taskset.task(task_name).make_env(tmp_path / "rollout")
    return taskset, env, created_sessions


def _write_task(
    root: Path,
    name: str,
    *,
    extra_toml: str = "",
    allow_internet_line: str | None = "allow_internet = true",
    verifier_timeout_sec: float | None = 900.0,
    agent_timeout_sec: float = 900.0,
    build_timeout_sec: float = 600.0,
    cpus: int | float | None = 1,
    memory_mb: int | float | None = 2048,
) -> Path:
    task_dir = root / name
    allow_internet_toml = (
        "" if allow_internet_line is None else f"{allow_internet_line}\n"
    )
    verifier_timeout_toml = (
        ""
        if verifier_timeout_sec is None
        else f"timeout_sec = {verifier_timeout_sec}\n"
    )
    resource_toml = "".join(
        f"{field} = {value}\n"
        for field, value in (("cpus", cpus), ("memory_mb", memory_mb))
        if value is not None
    )
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("Compress data.\n")
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\necho 1\n")
    (task_dir / "task.toml").write_text(
        f"""{extra_toml}schema_version = "1.1"
[task]
name = "terminal-bench/{name}"

[verifier]
{verifier_timeout_toml}

[agent]
timeout_sec = {agent_timeout_sec}

[environment]
build_timeout_sec = {build_timeout_sec}
docker_image = "example/{name}:latest"
{allow_internet_toml}{resource_toml}gpus = 0
mcp_servers = []
"""
    )
    return task_dir


def _env_with_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    session: "_FakeSession",
) -> TerminalBenchEnv:
    monkeypatch.setattr(tb_module, "DockerShellSession", lambda **_: session)
    return TerminalBenchEnv(
        task=_task(tmp_path),
        artifacts_dir=tmp_path / "rollout",
        host_netcache=True,
    )


def _task(tmp_path: Path) -> TerminalBenchTask:
    task_dir = _write_task(tmp_path / "tb", "write-compressor")
    return TerminalBenchTask(
        task_name="terminal-bench/write-compressor",
        instruction="Compress data.\n",
        task_dir=task_dir,
        docker_image="example/write-compressor:latest",
        agent_timeout_sec=900.0,
        build_timeout_sec=600.0,
        verify_timeout_sec=900.0,
        cpus=1,
        memory_mb=2048,
        replay_id="test-fingerprint",
        network=ManagedDockerNetwork(subnet="10.240.0.0/24", ipv4_address="10.240.0.2"),
    )


class _FakeSession:
    def __init__(
        self,
        *,
        run_stdout: str = "",
        test_stdout: str = "",
        reward_stdout: str | list[str] = "0\n",
        reward_exit_code: int = 0,
    ) -> None:
        self.start_calls = 0
        self.run_stdout = run_stdout
        self.test_stdout = test_stdout
        self.reward_stdouts = (
            list(reward_stdout) if isinstance(reward_stdout, list) else [reward_stdout]
        )
        self.reward_exit_code = reward_exit_code
        self.runs: list[dict] = []
        self.uploads: list[dict] = []
        self.closed = False
        self.setup_command: str | None = None

    def set_setup_command(self, setup_command: str | None) -> None:
        self.setup_command = setup_command

    async def start(self) -> None:
        self.start_calls += 1

    async def run(
        self,
        *,
        command: str,
        cwd: str,
        timeout: float | None,
    ) -> ExecResult:
        self.runs.append(
            {
                "command": command,
                "cwd": cwd,
                "timeout": timeout,
            }
        )
        if "cat /logs/verifier/test-stdout.txt" in command:
            return ExecResult(exit_code=0, stdout=self.test_stdout, stderr="")
        if "/tests/test.sh" in command:
            return ExecResult(exit_code=1, stdout="", stderr="")
        if command == "cat /logs/verifier/reward.txt":
            stdout = (
                self.reward_stdouts.pop(0)
                if len(self.reward_stdouts) > 1
                else self.reward_stdouts[0]
            )
            return ExecResult(
                exit_code=self.reward_exit_code,
                stdout=stdout,
                stderr="",
            )
        return ExecResult(exit_code=0, stdout=self.run_stdout, stderr="")

    async def upload_dir(self, *, source_dir: Path, target_dir: str) -> None:
        self.uploads.append({"source_dir": source_dir, "target_dir": target_dir})

    async def close(self) -> None:
        self.closed = True


class _FakeContainer:
    def __init__(self, *, exec_results: list[SimpleNamespace] | None = None) -> None:
        self.exec_run_calls = []
        self.remove_calls = []
        self.exec_results = list(exec_results or [])

    def exec_run(self, *args, **kwargs):
        self.exec_run_calls.append((args, kwargs))
        if self.exec_results:
            return self.exec_results.pop(0)
        return SimpleNamespace(exit_code=0, output=(b"", b""))

    def remove(self, *args, **kwargs):
        self.remove_calls.append((args, kwargs))


class _FakeContainers:
    def __init__(self, container: _FakeContainer) -> None:
        self.container = container
        self.run_calls = []

    def run(self, *args, **kwargs):
        self.run_calls.append((args, kwargs))
        return self.container

    def get(self, _name):
        return self.container


class _FakeNetwork:
    def __init__(self) -> None:
        self.remove_calls = []

    def remove(self, *args, **kwargs):
        self.remove_calls.append((args, kwargs))


class _FakeNetworks:
    def __init__(self) -> None:
        self.create_calls = []
        self.created = []

    def create(self, name, **kwargs):
        network = _FakeNetwork()
        self.created.append(network)
        self.create_calls.append({"name": name, **kwargs})
        return network


class _FakeDockerClient:
    def __init__(self, containers: _FakeContainers) -> None:
        self.containers = containers
        self.images = SimpleNamespace(pull=lambda _image: None)
        self.api = SimpleNamespace(
            _version="1.43",
            create_endpoint_config=lambda **kw: EndpointConfig("1.43", **kw),
        )
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _docker_session(monkeypatch, *, container=None, networks=None, **kwargs):
    container = container or _FakeContainer()
    containers = _FakeContainers(container)
    client = _FakeDockerClient(containers)
    if networks is not None:
        client.networks = networks
    monkeypatch.setattr(docker_shell_module.docker, "from_env", lambda: client)
    session = DockerShellSession(
        image="example/write-compressor:latest",
        **kwargs,
    )
    return session, container, containers, client


def test_write_verify_artifacts_reconstructs_from_replayed_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # The env step cache calls this hook when a verify verdict is replayed;
    # the artifact files a live verifier would have written must reappear.
    import json

    env = _env_with_session(monkeypatch, tmp_path, _FakeSession())
    result = StepResult(
        raw_env_output=RawEnvOutput(exit_code=0, stdout="test output\n"),
        reward=1.0,
        terminated=True,
        truncated=False,
        info={
            "task_name": "terminal-bench/write-compressor",
            "verifier_exit_code": 0,
        },
        verdict=VerifyVerdict(completed=True, passed=True, error=None),
    )

    env.write_verify_artifacts(result)

    verifier_dir = tmp_path / "rollout" / "terminal_bench_verifier"
    assert (verifier_dir / "test-stdout.txt").read_text() == "test output\n"
    assert json.loads((verifier_dir / "result.json").read_text())["passed"] is True
    assert (verifier_dir / "reward.txt").read_text() == "1\n"
