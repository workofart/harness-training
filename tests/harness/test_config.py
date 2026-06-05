from __future__ import annotations

import pytest

from src.harness.config import DEFAULT_HARNESS_CONFIG_PATH, HarnessConfig


def promotion_panel(
    task_names: list[str] | None = None,
    *,
    task_timeout_sec: float = 600.0,
) -> dict[str, object]:
    return {
        "id": "train",
        "purpose": "promotion",
        "task_names": ["task-a"] if task_names is None else task_names,
        "task_timeout_sec": task_timeout_sec,
        "run": {"when": "always"},
        "baseline": {"required": False},
    }


def regression_veto_panel(task_names: list[str] | None = None) -> dict[str, object]:
    return {
        "id": "test",
        "purpose": "regression_veto",
        "task_names": ["task-b"] if task_names is None else task_names,
        "task_timeout_sec": 1200.0,
        "run": {"after_panel": "train", "when_status": "keep"},
        "baseline": {"required": True},
    }


def minimal_config_payload(**updates) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 2,
        "experiment_id": "exp-1",
        "focus_name": "action-set",
        "panels": [promotion_panel()],
        "llm_provider_config": {
            "provider": "openrouter",
            "model_name": "openrouter/openai/gpt-oss-20b",
        },
    }
    payload.update(updates)
    return payload


def test_harness_config_accepts_literal_narrow_loop_shape():
    payload = minimal_config_payload(
        panels=[promotion_panel(task_timeout_sec=30.0)],
        max_steps=20,
        max_trial_concurrency=3,
        max_output_retries=4,
    )

    config = HarnessConfig.model_validate(payload)

    assert config.experiment_id == "exp-1"
    assert config.focus_name == "action-set"
    assert config.promotion_panel.task_names == ["task-a"]
    assert config.regression_veto_panel is None
    assert config.max_steps == 20
    assert config.max_trial_concurrency == 3
    assert config.promotion_panel.task_timeout_sec == 30.0
    assert config.env_setup_timeout_sec == 600.0  # default applied when omitted
    assert config.max_output_retries == 4
    assert config.llm_provider_config.service_tier is None


def test_default_harness_config_matches_baseline_run_profile():
    config = HarnessConfig.model_validate_json(DEFAULT_HARNESS_CONFIG_PATH.read_text())
    test_panel = config.regression_veto_panel
    ignored_group = config.excluded_task_groups["ignored"]
    slow_group = config.excluded_task_groups["slow"]
    contamination_group = config.excluded_task_groups["contamination_risk"]

    assert [panel.id for panel in config.panels] == ["train", "test"]
    assert len(config.promotion_panel.task_names) == 48
    assert test_panel is not None
    assert len(test_panel.task_names) == 10
    assert len(ignored_group.task_names) == 2
    assert len(slow_group.task_names) == 28
    assert contamination_group.task_names == ["reshard-c4-data"]
    assert set(config.promotion_panel.task_names).isdisjoint(test_panel.task_names)
    assert set(config.promotion_panel.task_names).isdisjoint(ignored_group.task_names)
    assert set(config.promotion_panel.task_names).isdisjoint(slow_group.task_names)
    assert set(test_panel.task_names).isdisjoint(ignored_group.task_names)
    assert set(test_panel.task_names).isdisjoint(slow_group.task_names)
    assert set(ignored_group.task_names).isdisjoint(slow_group.task_names)
    assert (
        len(
            set(config.promotion_panel.task_names)
            | set(test_panel.task_names)
            | set(ignored_group.task_names)
            | set(slow_group.task_names)
            | set(contamination_group.task_names)
        )
        == 89
    )
    assert "gpt2-codegolf" in test_panel.task_names
    assert "configure-git-webserver" in test_panel.task_names
    assert "break-filter-js-from-html" in ignored_group.task_names
    assert "pytorch-model-cli" in ignored_group.task_names
    assert "vulnerable-secret" in slow_group.task_names
    assert "model-extraction-relu-logits" in slow_group.task_names
    assert "qemu-startup" in slow_group.task_names
    assert "financial-document-processor" in slow_group.task_names
    assert "dna-assembly" in slow_group.task_names
    assert config.max_trial_concurrency == 24
    assert config.max_heavy_action_concurrency == 10
    assert config.promotion_panel.task_timeout_sec == 1200.0
    assert test_panel.task_timeout_sec == 1800.0
    assert config.env_setup_timeout_sec == 600.0
    assert config.task_trials == 5
    assert config.llm_provider_config.provider == "chatgpt_codex"
    assert config.llm_provider_config.model_name == "gpt-5.5"
    assert config.llm_provider_config.max_context_length == 200000
    assert config.llm_provider_config.reasoning_effort == "high"


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
        panels=[
            promotion_panel(["task-a"]),
            regression_veto_panel(["task-a"]),
        ]
    )

    with pytest.raises(ValueError) as exc:
        HarnessConfig.model_validate(payload)

    assert "panels.train.task_names and panels.test.task_names must be disjoint" in str(
        exc.value
    )


def test_harness_config_rejects_duplicate_panel_ids():
    payload = minimal_config_payload(
        panels=[
            promotion_panel(["task-a"]),
            {
                **regression_veto_panel(["task-b"]),
                "id": "train",
            },
        ]
    )

    with pytest.raises(ValueError) as exc:
        HarnessConfig.model_validate(payload)

    assert "panel ids must be unique: train" in str(exc.value)


@pytest.mark.parametrize(
    "panels",
    [
        pytest.param(
            [regression_veto_panel(["task-b"]), promotion_panel(["task-a"])],
            id="regression-veto-before-promotion",
        ),
        pytest.param(
            [
                promotion_panel(["task-a"]),
                {
                    **regression_veto_panel(["task-b"]),
                    "run": {"after_panel": "test", "when_status": "keep"},
                },
            ],
            id="regression-veto-self-reference",
        ),
    ],
)
def test_harness_config_rejects_regression_veto_not_after_promotion(panels):
    payload = minimal_config_payload(panels=panels)

    with pytest.raises(ValueError) as exc:
        HarnessConfig.model_validate(payload)

    assert "regression_veto panel must run after the promotion panel (train)" in str(
        exc.value
    )


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
            "service_tier": "priority",
        }
    )

    config = HarnessConfig.model_validate(payload)

    assert config.llm_provider_config.provider == "chatgpt_codex"
    assert config.llm_provider_config.model_name == "gpt-5.5"
    assert config.llm_provider_config.max_context_length == 200000
    assert config.llm_provider_config.service_tier == "priority"


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
