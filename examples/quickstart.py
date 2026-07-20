"""Small local training run with held-out evaluation.

Same shape as scripts/train.py, shrunk to two Terminal-Bench tasks and two
epochs. Prerequisites, model-server setup, and first-run cost are in the README's
"quick start" section.

Run from the repo root:

    uv run python examples/quickstart.py
"""

from __future__ import annotations

from dotenv import load_dotenv
from git import Repo

from src.config import RunConfig
from src.env.base import benchmark
from scripts.evaluate import evaluate
from src.llm.agent_backend import CodexAgentBackend
from src.measurement import preflight
from src.rollout.records import ExperimentResult, solved_task_ids
from src.rollout.store import invoker_repo_root
from src.trainer.estimator import AgenticEstimator
from src.trainer.loss import StrictPareto
from src.trainer.optim import GreedyMonotonic
from src.trainer.trainer import Trainer

# To swap tasks or model, edit these configs and commit — the trainer only
# measures git-tracked configs.
_TRAIN_CONFIG = "config/quickstart_train.yaml"  # two easy tasks
_EVAL_CONFIG = "config/quickstart_eval.yaml"  # two held-out tasks
_EPOCHS = 4


def _summary(result: ExperimentResult) -> str:
    solved = solved_task_ids(result)
    names = ", ".join(sorted(solved)) or "none"
    return f"{len(solved)}/{len(result.tasks)} solved ({names})"


def main() -> None:
    load_dotenv()
    root = invoker_repo_root()
    train_config = RunConfig.load(root / _TRAIN_CONFIG)
    eval_config = RunConfig.load(root / _EVAL_CONFIG)
    preflight([train_config, eval_config])

    repo = Repo(root)
    # Run on a branch: promotions fast-forward this checkout past start_commit.
    start_commit = repo.head.commit.hexsha

    criterion = StrictPareto(
        secondary_metrics=benchmark(train_config.environment.kind).secondary_metrics
    )
    optimizer = GreedyMonotonic()
    # Built before the held-out run so an unusable estimator CLI fails here, not after it.
    trainer = Trainer(
        config_path=root / _TRAIN_CONFIG,
        estimator=AgenticEstimator(
            # Needs the `codex` CLI installed and authenticated.
            backend=CodexAgentBackend(
                trace_dir=root / "experiments" / "codex-traces",
                model="gpt-5.6-sol",
                effort="medium",  # for faster quickstart turnaround
            )
            # For Claude:
            # backend=ClaudeAgentBackend(
            #     model="claude-opus-4-8",
            # )
        ),
        criterion=criterion,
        optimizer=optimizer,
    )

    print("Step 1: evaluating the untrained harness on the held-out panel")
    before = evaluate(eval_config)

    print("Step 2: train the harness on training panel")
    for loss in trainer.epochs(_EPOCHS):
        loss.backward()
        optimizer.step()

    if repo.head.commit.hexsha == start_commit:
        print("no candidate promoted; held-out result unchanged")
        return
    print("Step 3: evaluate the trained harness on held-out panel")
    after = evaluate(eval_config)
    print(f"held-out after training:  {_summary(after)}")
    print(f"held-out before training: {_summary(before)}")


if __name__ == "__main__":
    main()
