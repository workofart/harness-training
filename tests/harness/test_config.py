from __future__ import annotations

from src.harness.config import DEFAULT_HARNESS_CONFIG_PATH, HarnessConfig


def test_harness_config_accepts_literal_narrow_loop_shape():
    payload = {
        "experiment_id": "exp-1",
        "focus_name": "action-set",
        "train_task_names": ["task-a"],
        "max_steps": 20,
        "max_trial_concurrency": 3,
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
    assert config.max_trial_concurrency == 3
    assert config.task_timeout_sec == 30.0
    assert config.env_setup_timeout_sec == 600.0  # default applied when omitted
    assert config.max_output_retries == 4
    assert config.llm_provider_config.service_tier is None


def test_default_harness_config_matches_baseline_run_profile():
    config = HarnessConfig.model_validate_json(DEFAULT_HARNESS_CONFIG_PATH.read_text())

    assert len(config.train_task_names) == 67
    assert len(config.slow_task_names) == 22
    # slow tasks are held out of the panel, never overlapping it
    assert set(config.train_task_names).isdisjoint(config.slow_task_names)
    assert len(set(config.train_task_names) | set(config.slow_task_names)) == 89
    # the two refusal-prone tasks were moved out of the panel into storage
    assert "vulnerable-secret" in config.slow_task_names
    assert "model-extraction-relu-logits" in config.slow_task_names
    assert config.max_trial_concurrency == 16
    assert config.max_env_concurrency == 10
    assert config.task_timeout_sec == 1200.0
    assert config.env_setup_timeout_sec == 600.0
    assert config.task_trials == 5
    assert config.llm_provider_config.provider == "chatgpt_codex"
    assert config.llm_provider_config.model_name == "gpt-5.5"
    assert config.llm_provider_config.max_context_length == 200000
    assert config.llm_provider_config.reasoning_effort == "high"


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


def test_harness_config_defaults_missing_provider_to_openrouter():
    payload = {
        "experiment_id": "exp-1",
        "train_task_names": ["task-a"],
        "llm_provider_config": {
            "model_name": "openrouter/openai/gpt-oss-20b",
        },
    }

    config = HarnessConfig.model_validate(payload)

    assert config.llm_provider_config.provider == "openrouter"


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


def test_harness_config_accepts_chatgpt_codex_provider():
    payload = {
        "experiment_id": "exp-1",
        "train_task_names": ["task-a"],
        "llm_provider_config": {
            "provider": "chatgpt_codex",
            "model_name": "gpt-5.5",
            "max_context_length": 200000,
            "reasoning_effort": "low",
        },
    }

    config = HarnessConfig.model_validate(payload)

    assert config.llm_provider_config.provider == "chatgpt_codex"
    assert config.llm_provider_config.model_name == "gpt-5.5"
    assert config.llm_provider_config.max_context_length == 200000
