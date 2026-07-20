"""Optimizer step policies for candidate promotion decisions."""

from __future__ import annotations

from src.trainer.loss import Loss
from src.trainer.parameter import Parameter


class Optimizer:
    """Applies the parameter update: ff-only merges the selected commit into harness."""

    harness: Parameter

    def step(self) -> None:
        """Fast-forward the harness to the candidate commit iff the recorded loss selects it."""
        loss = self.harness.grad
        if loss is None:
            raise RuntimeError(
                "nothing to apply: run criterion and loss.backward() before "
                "optimizer.step()"
            )
        # Only two reachable targets, so no subclass can select an unrelated commit.
        target = (
            loss.candidate.git_commit_hash
            if self._should_promote(loss)
            else self.harness.data
        )
        # --ff-only preserves unrelated WIP and refuses overlap/HEAD drift; grad clears at the epoch boundary.
        self.harness.repo.git.merge("--ff-only", target)
        self.harness.applied = target

    def _should_promote(self, loss: Loss) -> bool:
        raise NotImplementedError


class GreedyMonotonic(Optimizer):
    def _should_promote(self, loss: Loss) -> bool:
        return float(loss) < 0
