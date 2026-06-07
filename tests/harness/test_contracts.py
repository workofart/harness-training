from __future__ import annotations

from src.contracts import TaskMetrics


def test_task_metrics_from_dict_coerces_counter_values():
    # The trial-output contract (one trial's result) now lives in
    # `record.TrialResult`; `contracts` owns only telemetry. Pin the one
    # non-obvious deserialization behavior: custom_counters values coerce to int.
    metrics = TaskMetrics.model_validate(
        {
            "steps_total": 3,
            "token_input_total": 99,
            "token_output_total": 11,
            "custom_counters": {"extra": "2"},
        }
    )

    assert metrics == TaskMetrics(
        steps_total=3,
        token_input_total=99,
        token_output_total=11,
        custom_counters={"extra": 2},
    )
