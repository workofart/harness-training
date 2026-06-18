from __future__ import annotations

import pytest

from src.config import DEFAULT_HARNESS_CONFIG_PATH, HarnessConfig


def train_panel(
    task_names: list[str] | None = None,
    *,
    task_timeout_sec: float = 600.0,
) -> dict[str, object]:
    return {
        "task_names": ["task-a"] if task_names is None else task_names,
        "task_timeout_sec": task_timeout_sec,
    }


def held_out_panel(task_names: list[str] | None = None) -> dict[str, object]:
    return {
        "task_names": ["task-b"] if task_names is None else task_names,
        "task_timeout_sec": 1200.0,
    }


def minimal_config_payload(**updates) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 3,
        "train": train_panel(),
        "llm_provider_config": {
            "provider": "openrouter",
            "model_name": "openrouter/openai/gpt-oss-20b",
        },
    }
    payload.update(updates)
    return payload


def test_train_and_test_tasks_derive_from_the_panels():
    config = HarnessConfig.model_validate(
        minimal_config_payload(
            train=train_panel(["task-a", "task-b"]),
            test=held_out_panel(["task-c"]),
        )
    )
    assert config.train_tasks == frozenset({"task-a", "task-b"})
    assert config.test_tasks == frozenset({"task-c"})
    # Disjoint by construction (task_groups_are_disjoint enforces it).
    assert config.train_tasks.isdisjoint(config.test_tasks)


def test_test_tasks_is_empty_without_a_veto_panel():
    # A config with only a promotion panel has no test panel; test_tasks is empty
    # (scan() rejects that at World-build time per §12, not config-load).
    config = HarnessConfig.model_validate(minimal_config_payload())
    assert config.train_tasks == frozenset({"task-a"})
    assert config.test_tasks == frozenset()


def test_env_backend_defaults_to_harbor():
    config = HarnessConfig.model_validate(minimal_config_payload())
    assert config.env_backend == "harbor"


def test_env_backend_accepts_swe():
    config = HarnessConfig.model_validate(minimal_config_payload(env_backend="swe"))
    assert config.env_backend == "swe"


def test_env_backend_rejects_unknown_value():
    with pytest.raises(ValueError):
        HarnessConfig.model_validate(minimal_config_payload(env_backend="docker"))


def test_harness_config_accepts_literal_narrow_loop_shape():
    payload = minimal_config_payload(
        train=train_panel(task_timeout_sec=30.0),
        max_steps=20,
        max_trial_concurrency=3,
        max_output_retries=4,
    )

    config = HarnessConfig.model_validate(payload)

    assert config.train.task_names == ["task-a"]
    assert config.test is None
    assert config.max_steps == 20
    assert config.max_trial_concurrency == 3
    assert config.train.task_timeout_sec == 30.0
    assert config.env_setup_timeout_sec == 600.0  # default applied when omitted
    assert config.max_output_retries == 4
    assert config.llm_provider_config.service_tier is None


def test_default_harness_config_matches_baseline_run_profile():
    config = HarnessConfig.model_validate_json(DEFAULT_HARNESS_CONFIG_PATH.read_text())
    held_out = config.test

    assert config.env_backend == "swe"
    assert len(config.train.task_names) == 30
    assert held_out is not None
    assert len(held_out.task_names) == 9
    assert set(config.train.task_names).isdisjoint(held_out.task_names)
    assert "django__django-10973" in config.train.task_names
    assert "sympy__sympy-24443" in config.train.task_names
    assert "django__django-10999" in held_out.task_names
    assert "sympy__sympy-18763" in held_out.task_names
    assert config.max_trial_concurrency == 12
    assert config.max_heavy_action_concurrency == 8
    assert config.train.task_timeout_sec == 1800.0
    assert held_out.task_timeout_sec == 1800.0
    assert config.env_setup_timeout_sec == 600.0
    assert config.task_trials == 3
    assert config.llm_provider_config.provider == "chatgpt_codex"
    assert config.llm_provider_config.model_name == "gpt-5.5"
    assert config.llm_provider_config.max_context_length == 200000
    assert config.llm_provider_config.reasoning_effort == "low"


def test_harness_config_rejects_v1_payload():
    payload = {
        "experiment_id": "exp-1",
        "train_task_names": ["task-a"],
        "llm_provider_config": {
            "model_name": "openrouter/openai/gpt-oss-20b",
        },
    }

    with pytest.raises(ValueError) as exc:
        HarnessConfig.model_validate(payload)

    message = str(exc.value)
    assert "schema_version" in message
    assert "train_task_names" in message


def test_harness_config_rejects_v2_panels_payload():
    # The v2 shape (panels[] + experiment_id) is not accepted: schema_version is
    # a strict Literal[3] and the retired fields are extra-forbidden.
    payload = minimal_config_payload(
        schema_version=2,
        experiment_id="exp-1",
        panels=[{"id": "train", "purpose": "promotion", "task_names": ["task-a"]}],
    )

    with pytest.raises(ValueError) as exc:
        HarnessConfig.model_validate(payload)

    message = str(exc.value)
    assert "schema_version" in message
    assert "panels" in message
    assert "experiment_id" in message


def test_harness_config_accepts_provider_require_parameters():
    payload = minimal_config_payload(
        llm_provider_config={
            "provider": "openrouter",
            "model_name": "openrouter/openai/gpt-oss-20b",
            "provider_kwargs": {
                "require_parameters": True,
            },
        }
    )

    config = HarnessConfig.model_validate(payload)

    assert config.llm_provider_config.provider_kwargs.require_parameters is True


def test_harness_config_defaults_missing_provider_to_openrouter():
    payload = minimal_config_payload(
        llm_provider_config={
            "model_name": "openrouter/openai/gpt-oss-20b",
        }
    )

    config = HarnessConfig.model_validate(payload)

    assert config.llm_provider_config.provider == "openrouter"


def test_harness_config_rejects_overlapping_task_groups():
    payload = minimal_config_payload(
        train=train_panel(["task-a"]),
        test=held_out_panel(["task-a"]),
    )

    with pytest.raises(ValueError) as exc:
        HarnessConfig.model_validate(payload)

    assert "train.task_names and test.task_names must be disjoint" in str(exc.value)


def test_harness_config_rejects_missing_schema_version():
    payload = minimal_config_payload()
    del payload["schema_version"]

    with pytest.raises(ValueError) as exc:
        HarnessConfig.model_validate(payload)

    assert "schema_version" in str(exc.value)


def test_harness_config_accepts_provider_routing_with_ignore():
    payload = minimal_config_payload(
        llm_provider_config={
            "provider": "openrouter",
            "model_name": "openrouter/openai/gpt-oss-20b",
            "provider_kwargs": {
                "provider": {
                    "order": ["gmicloud"],
                    "allow_fallbacks": True,
                    "ignore": ["siliconflow", "parasail"],
                },
            },
        }
    )

    config = HarnessConfig.model_validate(payload)
    routing = config.llm_provider_config.provider_kwargs.provider
    assert routing is not None
    assert routing.order == ("gmicloud",)
    assert routing.allow_fallbacks is True
    assert routing.ignore == ("siliconflow", "parasail")


def test_provider_routing_ignore_defaults_to_empty_tuple():
    payload = minimal_config_payload(
        llm_provider_config={
            "provider": "openrouter",
            "model_name": "openrouter/openai/gpt-oss-20b",
            "provider_kwargs": {
                "provider": {
                    "order": ["gmicloud"],
                    "allow_fallbacks": True,
                },
            },
        }
    )

    config = HarnessConfig.model_validate(payload)
    routing = config.llm_provider_config.provider_kwargs.provider
    assert routing is not None
    assert routing.ignore == ()


def test_harness_config_accepts_chatgpt_codex_provider():
    payload = minimal_config_payload(
        llm_provider_config={
            "provider": "chatgpt_codex",
            "model_name": "gpt-5.5",
            "max_context_length": 200000,
            "reasoning_effort": "low",
            "service_tier": "standard",
        }
    )

    config = HarnessConfig.model_validate(payload)

    assert config.llm_provider_config.provider == "chatgpt_codex"
    assert config.llm_provider_config.model_name == "gpt-5.5"
    assert config.llm_provider_config.max_context_length == 200000
    assert config.llm_provider_config.service_tier == "standard"


def test_harness_config_rejects_openrouter_priority_service_tier():
    payload = minimal_config_payload(
        llm_provider_config={
            "provider": "openrouter",
            "model_name": "openrouter/openai/gpt-oss-20b",
            "service_tier": "priority",
        }
    )

    with pytest.raises(ValueError) as exc:
        HarnessConfig.model_validate(payload)

    assert "service_tier" in str(exc.value)
