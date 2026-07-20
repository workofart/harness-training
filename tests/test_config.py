from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from src.config import RunConfig, LlmProviderConfig

_DELETE = object()
type _Edits = dict[tuple[str, ...], object]
type _InvalidPayloadCase = tuple[_Edits, tuple[str, ...]]


def environment_payload(
    task_names: list[str] | None = None,
    *,
    kind: str = "swe",
) -> dict[str, object]:
    return {
        "kind": kind,
        "task_names": ["task-a"] if task_names is None else task_names,
    }


def llm_provider_payload(**updates: object) -> dict[str, object]:
    return {
        "model_name": "gpt-5.5",
        "base_url": "http://127.0.0.1:18000/v1",
        "api_key_env": "OPENAI_API_KEY",
        "max_context_length": 200000,
        "max_tokens": 8192,
    } | updates


def minimal_config_payload(**updates: object) -> dict[str, object]:
    return {
        "schema_version": 13,
        "training_target": {"module": "src.policy.core"},
        "environment": environment_payload(),
        "llm_provider_config": llm_provider_payload(),
    } | updates


def _validate_provider(payload: dict[str, object]) -> LlmProviderConfig:
    return RunConfig.model_validate(
        minimal_config_payload(llm_provider_config=payload)
    ).llm_provider_config


def test_run_config_rejects_removed_chatgpt_codex_provider() -> None:
    with pytest.raises(ValueError, match="chatgpt_codex"):
        _validate_provider(
            {
                "provider": "chatgpt_codex",
                "model_name": "gpt-5.5",
                "max_context_length": 200000,
            }
        )


def _apply_edits(payload: dict[str, object], edits: _Edits) -> None:
    for path, value in edits.items():
        target = payload
        for key in path[:-1]:
            nested = target[key]
            assert isinstance(nested, dict)
            target = nested
        if value is _DELETE:
            del target[path[-1]]
        else:
            target[path[-1]] = deepcopy(value)


def _write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def test_config_import_does_not_load_llm_backend() -> None:
    code = """
import sys
import src.config
if "src.llm.backend" in sys.modules:
    raise SystemExit("src.config imported src.llm.backend")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_run_config_load_composes_llm_config_and_task_panel(
    tmp_path: Path,
) -> None:
    parent_provider = llm_provider_payload(
        model_name="Qwen/Qwen3.6-35B-A3B",
        api_key_env="TEST_API_KEY",
        max_context_length=32768,
        temperature=1.0,
        top_p=0.95,
        max_tokens=4096,
        enable_thinking=True,
    )
    parent = {
        "schema_version": 13,
        "max_rollout_concurrency": 2,
        "llm_provider_config": parent_provider,
    }
    _write_yaml(tmp_path / "llm" / "qwen.yaml", parent)
    _write_yaml(
        tmp_path / "task_panels" / "swebench.yaml",
        environment_payload(["task-a", "task-b"]),
    )
    _write_yaml(
        tmp_path / "run_config.yaml",
        {
            "$extends": ["llm/qwen.yaml"],
            "training_target": {"module": "src.policy.core"},
            "environment": {"$include": "task_panels/swebench.yaml"},
            "max_steps": 110,
            "llm_provider_config": {
                "max_tokens": 8192,
                "thinking_budget_tokens": 6144,
            },
        },
    )

    config = RunConfig.load(tmp_path / "run_config.yaml")

    assert config.config_path == str(tmp_path / "run_config.yaml")
    assert config.environment.kind == "swe"
    assert config.training_target.module == "src.policy.core"
    assert config.environment.task_names == ["task-a", "task-b"]
    assert config.max_steps == 110
    assert config.max_rollout_concurrency == 2
    assert config.llm_provider_config.model_name == "Qwen/Qwen3.6-35B-A3B"
    assert config.llm_provider_config.max_tokens == 8192
    assert config.llm_provider_config.thinking_budget_tokens == 6144
    assert config.llm_provider_config.top_p == 0.95


def test_run_config_load_stores_cwd_independent_config_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_yaml(tmp_path / "run_config.yaml", minimal_config_payload())
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    monkeypatch.chdir(subdir)

    config = RunConfig.load("../run_config.yaml")

    assert config.config_path is not None
    assert Path(config.config_path).is_absolute()
    # The worker reopens config_path from a different cwd (measure_root).
    monkeypatch.chdir(tmp_path)
    assert Path(config.config_path).is_file()


def test_run_config_rejects_duplicate_task_names() -> None:
    payload = minimal_config_payload(
        environment=environment_payload(["task-a", "task-a"])
    )

    with pytest.raises(ValueError, match="task_names must be unique"):
        RunConfig.model_validate(payload)


def test_shipped_configs_own_root_fields_outside_llm_fragments() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "config"
    for path in sorted((config_dir / "llm").glob("*.yaml")):
        payload = yaml.safe_load(path.read_text())
        assert "schema_version" not in payload, path
        assert "training_target" not in payload, path

    for path in sorted(config_dir.glob("*.yaml")):
        if path.name.endswith(".template.yaml"):
            continue
        payload = yaml.safe_load(path.read_text())
        assert payload.get("schema_version") == 13, path
        target = payload.get("training_target")
        assert isinstance(target, dict), path
        assert isinstance(target.get("module"), str), path
        assert RunConfig.load(path).training_target.module == target["module"]


def test_shipped_run_config_template_loads() -> None:
    path = Path(__file__).resolve().parents[1] / "config/run_config.template.yaml"

    config = RunConfig.load(path)

    assert config.environment.kind == "swe"
    assert config.environment.task_names


def test_run_config_load_replaces_lists_and_accepts_null_overrides(
    tmp_path: Path,
) -> None:
    _write_yaml(
        tmp_path / "base.yaml",
        minimal_config_payload(
            environment=environment_payload(["base-a", "base-b"]),
            llm_provider_config=llm_provider_payload(temperature=1.0),
        ),
    )
    _write_yaml(
        tmp_path / "run_config.yaml",
        {
            "$extends": ["base.yaml"],
            "environment": {"task_names": ["child"]},
            "llm_provider_config": {"temperature": None},
        },
    )

    config = RunConfig.load(tmp_path / "run_config.yaml")

    assert config.environment.kind == "swe"
    assert config.environment.task_names == ["child"]
    assert config.llm_provider_config.temperature is None


@pytest.mark.parametrize("composition", ["extends", "include"])
def test_run_config_load_rejects_parent_escape(
    tmp_path: Path, composition: str
) -> None:
    _write_yaml(tmp_path / "outside.yaml", minimal_config_payload())
    config_dir = tmp_path / "config"
    payload = (
        {"$extends": ["../outside.yaml"]}
        if composition == "extends"
        else minimal_config_payload(environment={"$include": "../outside.yaml"})
    )
    _write_yaml(config_dir / "run_config.yaml", payload)

    with pytest.raises(ValueError, match="must stay within config directory"):
        RunConfig.load(config_dir / "run_config.yaml")


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param("swe", id="swe"),
        pytest.param("terminal_bench", id="terminal-bench"),
    ],
)
def test_environment_accepts_supported_kinds(kind: str) -> None:
    config = RunConfig.model_validate(
        minimal_config_payload(environment=environment_payload(kind=kind))
    )

    assert config.environment.kind == kind
    assert not hasattr(config, "terminal_bench_repo_url")
    assert not hasattr(config, "terminal_bench_dataset_path")


def test_environment_rejects_framework_owned_task_timeout() -> None:
    payload = minimal_config_payload()
    environment = payload["environment"]
    assert isinstance(environment, dict)
    environment["task_timeout_sec"] = 4500.0

    with pytest.raises(ValueError, match="task_timeout_sec"):
        RunConfig.model_validate(payload)


def test_run_config_accepts_literal_narrow_loop_shape() -> None:
    config = RunConfig.model_validate(
        minimal_config_payload(
            environment=environment_payload(),
            max_steps=20,
            max_rollout_concurrency=3,
        )
    )

    assert config.environment.task_names == ["task-a"]
    assert config.max_steps == 20
    assert config.max_rollout_concurrency == 3
    assert config.environment.kind == "swe"
    assert "max_epochs" not in RunConfig.model_fields


def test_plugins_block_is_optional_with_behavior_preserving_defaults() -> None:
    payload = minimal_config_payload()

    config = RunConfig.model_validate(payload)

    assert config.plugins.model_dump() == {
        "llm_cache": True,
        "execution": "eager",
    }


def test_with_task_panel_swaps_only_the_panel() -> None:
    payload = minimal_config_payload(environment=environment_payload(["a", "b"]))
    config = RunConfig.model_validate(payload)

    staged = config.with_task_panel(("b",))

    assert staged.environment.task_names == ["b"]
    assert staged.environment.kind == config.environment.kind
    assert config.environment.task_names == ["a", "b"]

    with pytest.raises(ValueError, match="task_names must be unique"):
        config.with_task_panel(("b", "b"))


def test_old_schema_payload_is_rejected() -> None:
    payload = minimal_config_payload(schema_version=11)

    with pytest.raises(ValueError, match="schema_version"):
        RunConfig.model_validate(payload)


def test_plugins_block_rejects_unknown_keys() -> None:
    payload = minimal_config_payload(plugins={"unknown_cache": True})

    with pytest.raises(ValueError, match="unknown_cache"):
        RunConfig.model_validate(payload)


def test_swe_rejects_non_default_host_netcache() -> None:
    environment = environment_payload(kind="swe")
    environment["host_netcache"] = False
    payload = minimal_config_payload(environment=environment)

    with pytest.raises(ValueError, match="host_netcache"):
        RunConfig.model_validate(payload)


def test_terminal_bench_replay_requires_host_netcache() -> None:
    environment = environment_payload(kind="terminal_bench")
    environment["host_netcache"] = False
    payload = minimal_config_payload(
        environment=environment,
        llm_provider_config=llm_provider_payload(seed=7),
        plugins={"execution": "replay"},
    )

    with pytest.raises(
        ValueError,
        match=r"plugins\.execution.*environment\.host_netcache",
    ):
        RunConfig.model_validate(payload)


def test_replay_requires_deterministic_provider() -> None:
    payload = minimal_config_payload(
        llm_provider_config=llm_provider_payload(),
        plugins={"execution": "replay"},
    )

    with pytest.raises(ValueError, match="deterministic"):
        RunConfig.model_validate(payload)


def test_replay_accepts_seeded_provider() -> None:
    payload = minimal_config_payload(
        llm_provider_config=llm_provider_payload(seed=7),
        plugins={"execution": "replay"},
    )

    config = RunConfig.model_validate(payload)

    assert config.plugins.execution == "replay"


def test_agent_timeout_multiplier_defaults_and_accepts_override() -> None:
    default = RunConfig.model_validate(minimal_config_payload())
    overridden = RunConfig.model_validate(
        minimal_config_payload(agent_timeout_multiplier=4.0)
    )

    assert default.agent_timeout_multiplier == 1.0
    assert overridden.agent_timeout_multiplier == 4.0


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_run_config_rejects_invalid_agent_timeout_multiplier(value: float) -> None:
    with pytest.raises(ValueError, match="agent_timeout_multiplier"):
        RunConfig.model_validate(minimal_config_payload(agent_timeout_multiplier=value))


def test_run_config_load_stamps_provenance_but_excludes_from_dumps(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "run_config.yaml"
    payload = minimal_config_payload()
    _write_yaml(config_path, payload)

    config = RunConfig.load(config_path)

    assert config.config_path == str(config_path)
    assert "config_path" not in config.model_dump()
    assert "config_path" not in config.model_dump_json()


_INVALID_PAYLOAD_CASES: dict[str, _InvalidPayloadCase] = {
    "missing-max-tokens": (
        {("llm_provider_config", "max_tokens"): _DELETE},
        ("max_tokens",),
    ),
    "composition-key-without-loader": (
        {("$extends",): ["llm/qwen.yaml"]},
        ("$extends",),
    ),
    "missing-environment-kind": ({("environment", "kind"): _DELETE}, ("kind",)),
    "unsupported-harbor-environment": (
        {("environment", "kind"): "harbor"},
        ("harbor",),
    ),
    "flat-terminal-bench-field": (
        {("terminal_bench_repo_url",): "https://example.test/tb.git"},
        ("terminal_bench_repo_url",),
    ),
    "v7-task-timeout-payload": (
        {
            ("schema_version",): 7,
            ("environment", "task_timeout_sec"): 4500.0,
        },
        ("schema_version", "task_timeout_sec"),
    ),
    "v6-split-tasks-payload": (
        {
            ("schema_version",): 6,
            ("environment",): {"kind": "swe"},
            ("tasks",): {"task_names": ["task-a"], "task_timeout_sec": 600.0},
        },
        ("schema_version", "tasks"),
    ),
    "missing-schema-version": ({("schema_version",): _DELETE}, ("schema_version",)),
    "agent-backend-field": (
        {("agent_backend",): {"kind": "claude"}},
        ("agent_backend",),
    ),
    "max-epochs-field": ({("max_epochs",): 2}, ("max_epochs",)),
    "nested-provider-model-payload": (
        {
            ("llm_provider_config",): {
                "provider": {
                    "kind": "openai_compatible",
                    "base_url": "http://127.0.0.1:18000/v1",
                    "api_key_env": "OPENAI_API_KEY",
                },
                "model": {
                    "adapter": "none",
                    "name": "gpt-5.5",
                    "max_context_length": 200000,
                },
            }
        },
        ("provider", "model"),
    ),
    "thinking-budget-without-channel": (
        {("llm_provider_config", "thinking_budget_tokens"): 6144},
        ("thinking_budget_tokens requires enable_thinking: true",),
    ),
    "thinking-budget-disabled-channel": (
        {
            ("llm_provider_config", "enable_thinking"): False,
            ("llm_provider_config", "thinking_budget_tokens"): 6144,
        },
        ("thinking_budget_tokens requires enable_thinking: true",),
    ),
    "missing-api-key-environment-variable": (
        {("llm_provider_config", "api_key_env"): _DELETE},
        ("api_key_env",),
    ),
    "literal-api-key": ({("llm_provider_config", "api_key"): "test-key"}, ("api_key",)),
    "empty-model-name": ({("llm_provider_config", "model_name"): ""}, ("model_name",)),
}


@pytest.mark.parametrize(
    ("edits", "error_paths"),
    _INVALID_PAYLOAD_CASES.values(),
    ids=_INVALID_PAYLOAD_CASES,
)
def test_run_config_rejects_invalid_payloads(
    edits: _Edits,
    error_paths: tuple[str, ...],
) -> None:
    payload = minimal_config_payload()
    _apply_edits(payload, edits)

    with pytest.raises(ValueError) as raised:
        RunConfig.model_validate(payload)

    for error_path in error_paths:
        assert error_path in str(raised.value)


_ACCEPTED_PROVIDER_CASES: dict[str, tuple[dict[str, object], dict[str, object]]] = {
    "minimal-llm-config": (
        llm_provider_payload(),
        {
            "provider": "openai_compatible",
            "model_name": "gpt-5.5",
            "max_context_length": 200000,
            "enable_thinking": None,
        },
    ),
    "thinking-budget-when-thinking-enabled": (
        llm_provider_payload(enable_thinking=True, thinking_budget_tokens=6144),
        {"thinking_budget_tokens": 6144},
    ),
}


@pytest.mark.parametrize(
    ("provider_payload", "expected_attrs"),
    _ACCEPTED_PROVIDER_CASES.values(),
    ids=_ACCEPTED_PROVIDER_CASES,
)
def test_validate_provider_accepts_payloads(
    provider_payload: dict[str, object],
    expected_attrs: dict[str, object],
) -> None:
    provider = _validate_provider(provider_payload)

    for attr, expected in expected_attrs.items():
        assert getattr(provider, attr) == expected


def test_run_config_defaults_reasoning_effort_to_none() -> None:
    provider = _validate_provider(llm_provider_payload())

    assert provider.reasoning_effort is None


def test_run_config_accepts_openai_compatible_reasoning_effort() -> None:
    provider = _validate_provider(llm_provider_payload(reasoning_effort="high"))

    assert provider.reasoning_effort == "high"


def test_run_config_accepts_tokenizer_name() -> None:
    provider = _validate_provider(
        llm_provider_payload(tokenizer_name="openai-tokenizer", max_tokens=4096)
    )

    assert provider.tokenizer_name == "openai-tokenizer"
    assert provider.max_tokens == 4096


def test_training_target_surface_and_patch_paths_derive_from_module() -> None:
    config = RunConfig.model_validate(
        minimal_config_payload(
            training_target={
                "module": "src.policy.core",
                "extra_patch_paths": ["tests/policy/test_core_impl.py"],
            }
        )
    )

    target = config.training_target
    assert target.surface == "src/policy/core.py"
    assert target.patch_paths == (
        "src/policy/core.py",
        "tests/policy/test_core_impl.py",
    )


def test_training_target_module_is_required() -> None:
    payload = minimal_config_payload(training_target={})

    with pytest.raises(ValueError, match="module"):
        RunConfig.model_validate(payload)
