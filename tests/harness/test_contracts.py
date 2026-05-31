from __future__ import annotations

from src.harness.contracts import TaskResult
from src.metrics import TaskMetrics


def test_task_result_from_dict_loads_metric_fields():
    task_result = TaskResult.model_validate(
        {
            "task_name": "task-a",
            "reward": 1.0,
            "steps_used": 3,
            "error": None,
            "trial_dir": "/tmp/trial",
            "trace_path": "/tmp/trial/agent/steps.jsonl",
            "metrics_path": "/tmp/trial/agent/metrics.json",
            "verifier_stdout_path": "/tmp/verifier.txt",
            "started_at": "2026-04-10T00:00:00+00:00",
            "finished_at": "2026-04-10T00:00:01+00:00",
            "solved": True,
            "metrics": {
                "steps_total": 3,
                "token_input_total": 99,
                "token_output_total": 11,
                "custom_counters": {"extra": "2"},
            },
        }
    )

    assert task_result.metrics == TaskMetrics(
        steps_total=3,
        token_input_total=99,
        token_output_total=11,
        custom_counters={"extra": 2},
    )
