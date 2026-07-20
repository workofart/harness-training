from __future__ import annotations

from math import inf
from types import SimpleNamespace

import pytest

from src.trainer.loss import Loss
from src.trainer.optim import GreedyMonotonic


class _RecordingRepo:
    def __init__(self) -> None:
        self.merges: list[tuple[str, ...]] = []
        self.git = SimpleNamespace(merge=lambda *args: self.merges.append(args))


class _StubHarness:
    def __init__(self, *, data: str, grad: Loss | None) -> None:
        self.repo = _RecordingRepo()
        self.data = data
        self.grad: Loss | None = grad
        self.applied: str | None = None


def _loss(value: float, harness: _StubHarness | None = None) -> Loss:
    if harness is None:
        harness = _StubHarness.__new__(_StubHarness)
    return Loss(
        value,
        "strict_pareto",
        SimpleNamespace(git_commit_hash="candidate-commit"),  # type: ignore[arg-type]
        SimpleNamespace(git_commit_hash="baseline-commit"),  # type: ignore[arg-type]
        harness,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        (-0.1, True),
        (0.0, False),
        (inf, False),
    ],
)
def test_greedy_monotonic_step_selects_and_merges_once(
    value: float, accepted: bool
) -> None:
    loss = _loss(value)
    harness = _StubHarness(data="incumbent-commit", grad=loss)
    optimizer = GreedyMonotonic()
    optimizer.harness = harness  # type: ignore[assignment]

    assert optimizer.step() is None
    target = "candidate-commit" if accepted else "incumbent-commit"
    assert harness.repo.merges == [("--ff-only", target)]
    assert harness.applied == target
    # Grad persists until the epoch boundary.
    assert harness.grad is loss


def test_step_without_backward_fails_loudly() -> None:
    harness = _StubHarness(data="incumbent-commit", grad=None)
    optimizer = GreedyMonotonic()
    optimizer.harness = harness  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="loss.backward"):
        optimizer.step()
    assert harness.repo.merges == []
    assert harness.applied is None
