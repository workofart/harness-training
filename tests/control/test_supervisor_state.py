import pytest

from src.control.supervisor_state import SupervisorState


def test_prelaunch_factory_sets_phase_and_carries_marker():
    state = SupervisorState.prelaunch(
        thread_id="t1",
        updated_at="2026-05-31T00:00:00+00:00",
        postrun_completed_experiment_id="exp-1",
    )
    assert state.phase == "prelaunch"
    assert state.postrun_completed_experiment_id == "exp-1"
    # Other phases' fields stay at their inert defaults.
    assert state.launch_experiment_id is None
    assert state.postrun_original_payload is None


def test_prelaunch_factory_requires_the_postrun_completed_marker():
    # The marker must be passed explicitly (even as None). A prelaunch save that
    # silently dropped a still-pending postrun-completed id is what let postrun
    # re-fire repeatedly on one concluded record; requiring it here makes that
    # omission a construction-time error instead of a latent default.
    with pytest.raises(TypeError):
        SupervisorState.prelaunch(  # type: ignore[call-arg]
            thread_id="t1",
            updated_at="2026-05-31T00:00:00+00:00",
        )


def test_launch_factory_requires_launch_fields():
    with pytest.raises(TypeError):
        SupervisorState.launch(  # type: ignore[call-arg]
            thread_id="t1",
            updated_at="2026-05-31T00:00:00+00:00",
        )
    state = SupervisorState.launch(
        thread_id="t1",
        updated_at="2026-05-31T00:00:00+00:00",
        launch_experiment_id="exp-1",
        launch_baseline_commit="abc123",
    )
    assert state.phase == "launch"
    assert state.launch_experiment_id == "exp-1"
    assert state.launch_baseline_commit == "abc123"


def test_postrun_factory_requires_payload_and_learning():
    with pytest.raises(TypeError):
        SupervisorState.postrun(  # type: ignore[call-arg]
            thread_id="t1",
            updated_at="2026-05-31T00:00:00+00:00",
        )
    state = SupervisorState.postrun(
        thread_id="t1",
        updated_at="2026-05-31T00:00:00+00:00",
        postrun_original_payload={"k": "v"},
        postrun_original_learning="memo",
    )
    assert state.phase == "postrun"
    assert state.postrun_original_payload == {"k": "v"}
    assert state.postrun_original_learning == "memo"


def test_factory_state_round_trips_through_disk(tmp_path):
    # The factories keep the single flat persisted object: a saved factory state
    # reloads identically (no on-disk shape change vs direct construction).
    root = tmp_path / "supervisor"
    state = SupervisorState.prelaunch(
        thread_id="t1",
        updated_at="2026-05-31T00:00:00+00:00",
        postrun_completed_experiment_id="exp-1",
    )
    state.save(repo_root=tmp_path, root=root)
    assert SupervisorState.maybe_load(repo_root=tmp_path, root=root) == state
