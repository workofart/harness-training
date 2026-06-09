from __future__ import annotations

from src.contracts import TaskMetrics, is_majority_decided, is_majority_solved


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


def test_is_majority_decided():
    # k=3: outcome locks in once one side reaches ceil(3/2)=2 or can no longer reach it.
    assert is_majority_decided(solved=0, finished=0, expected_total=3) is False
    assert is_majority_decided(solved=1, finished=1, expected_total=3) is False
    assert is_majority_decided(solved=2, finished=2, expected_total=3) is True
    assert is_majority_decided(solved=0, finished=2, expected_total=3) is True
    assert is_majority_decided(solved=1, finished=2, expected_total=3) is False
    # k=1: any single trial is the final word.
    assert is_majority_decided(solved=1, finished=1, expected_total=1) is True
    # finished >= expected_total is always decided.
    assert is_majority_decided(solved=0, finished=3, expected_total=3) is True


def test_is_majority_solved():
    # total <= 0 -> False (no trials, no majority).
    assert is_majority_solved(solved=0, total=0) is False
    # total=3, threshold=ceil(3/2)=2.
    assert is_majority_solved(solved=1, total=3) is False
    assert is_majority_solved(solved=2, total=3) is True
    assert is_majority_solved(solved=3, total=3) is True
    # total=2, threshold=1.
    assert is_majority_solved(solved=0, total=2) is False
    assert is_majority_solved(solved=1, total=2) is True
    # total=1, threshold=1: any single solve is a majority.
    assert is_majority_solved(solved=0, total=1) is False
    assert is_majority_solved(solved=1, total=1) is True
    # ceil thresholds: total=4 -> 2, total=5 -> 3.
    assert is_majority_solved(solved=2, total=4) is True
    assert is_majority_solved(solved=2, total=5) is False
    assert is_majority_solved(solved=3, total=5) is True


def test_task_metrics_owned_by_contracts_module():
    assert TaskMetrics.__module__ == "src.contracts"
