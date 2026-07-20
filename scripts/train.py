from dotenv import load_dotenv

from src.config import RunConfig
from src.env.base import benchmark
from src.llm.agent_backend import ClaudeAgentBackend
from src.rollout.store import invoker_repo_root
from src.trainer.estimator import AgenticEstimator
from src.trainer.loss import StrictPareto
from src.trainer.optim import GreedyMonotonic
from src.trainer.trainer import Trainer


def main() -> None:
    load_dotenv()
    config_path = invoker_repo_root() / "config/train_harness.yaml"
    config = RunConfig.load(config_path)
    criterion = StrictPareto(
        secondary_metrics=benchmark(config.environment.kind).secondary_metrics
    )
    optimizer = GreedyMonotonic()
    trainer = Trainer(
        config_path=config_path,
        estimator=AgenticEstimator(backend=ClaudeAgentBackend(turn_timeout_sec=3600.0)),
        criterion=criterion,
        optimizer=optimizer,
    )

    for loss in trainer.epochs(30):
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    main()
