from __future__ import annotations

from src.harness.config import DEFAULT_HARNESS_CONFIG_PATH, HarnessConfig


def test_harness_config_accepts_literal_narrow_loop_shape():
    payload = {
        "experiment_id": "exp-1",
        "focus_name": "action-set",
        "train_task_names": ["task-a"],
        "max_steps": 20,
        "max_concurrency": 3,
        "task_timeout_sec": 30.0,
        "max_output_retries": 4,
        "llm_provider_config": {
            "provider": "openrouter",
            "model_name": "openrouter/openai/gpt-oss-20b",
        },
    }

    config = HarnessConfig.model_validate(payload)

    assert config.experiment_id == "exp-1"
    assert config.focus_name == "action-set"
    assert config.train_task_names == ["task-a"]
    assert config.max_steps == 20
    assert config.max_concurrency == 3
    assert config.task_timeout_sec == 30.0
    assert config.max_output_retries == 4
    assert config.llm_provider_config.service_tier is None


def test_default_harness_config_accepts_provider_require_parameters():
    config = HarnessConfig.model_validate_json(DEFAULT_HARNESS_CONFIG_PATH.read_text())

    assert config.llm_provider_config.provider_kwargs.require_parameters is True


def test_harness_config_accepts_provider_require_parameters():
    payload = {
        "experiment_id": "exp-1",
        "train_task_names": ["task-a"],
        "llm_provider_config": {
            "provider": "openrouter",
            "model_name": "openrouter/openai/gpt-oss-20b",
            "provider_kwargs": {
                "require_parameters": True,
            },
        },
    }

    config = HarnessConfig.model_validate(payload)

    assert config.llm_provider_config.provider_kwargs.require_parameters is True


def test_harness_config_accepts_provider_routing_with_ignore():
    payload = {
        "experiment_id": "exp-1",
        "train_task_names": ["task-a"],
        "llm_provider_config": {
            "provider": "openrouter",
            "model_name": "openrouter/openai/gpt-oss-20b",
            "provider_kwargs": {
                "provider": {
                    "order": ["gmicloud"],
                    "allow_fallbacks": True,
                    "ignore": ["siliconflow", "parasail"],
                },
            },
        },
    }

    config = HarnessConfig.model_validate(payload)
    routing = config.llm_provider_config.provider_kwargs.provider
    assert routing is not None
    assert routing.order == ("gmicloud",)
    assert routing.allow_fallbacks is True
    assert routing.ignore == ("siliconflow", "parasail")


def test_provider_routing_ignore_defaults_to_empty_tuple():
    payload = {
        "experiment_id": "exp-1",
        "train_task_names": ["task-a"],
        "llm_provider_config": {
            "provider": "openrouter",
            "model_name": "openrouter/openai/gpt-oss-20b",
            "provider_kwargs": {
                "provider": {
                    "order": ["gmicloud"],
                    "allow_fallbacks": True,
                },
            },
        },
    }

    config = HarnessConfig.model_validate(payload)
    routing = config.llm_provider_config.provider_kwargs.provider
    assert routing is not None
    assert routing.ignore == ()
