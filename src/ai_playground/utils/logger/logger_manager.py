import json
from typing import TYPE_CHECKING

from ai_playground.utils.logger import console_logger, tensorboard_logger
from ai_playground.utils import config_to_dict, convert_paths

if TYPE_CHECKING:
    from ai_playground.configs.config import TrainerConfig
    from ai_playground.distributed.base import Parallel
    from ai_playground.utils.logger.base_logger import Logger
    from typing import List, Dict, Any

# Default metric templates
BASELINE_METRICS: Dict[str, float] = {
    "train_loss": 0.0,
    "val_loss": 0.0,
    "lr": 0.0,
    "tps": 0.0,
}

FULL_METRICS: Dict[str, float] = {
    "train_loss": 0.0,
    "lr": 0.0,
    "grad_norm": 0.0,
    "weight_norm": 0.0,
    "update_ratio": 0.0,
    "tps": 0.0,
    "avg_step_time": 0.0,
    "avg_data_time": 0.0,
}


class LoggerManager:
    """
    Manages multiple loggers in a distributed training setup.

    Handles logging of metrics, hyperparameters, and configuration to all attached loggers,
    with support for rank-zero-only logging in distributed settings.
    """

    def __init__(
        self,
        loggers: List["Logger"],
        strategy: "Parallel",
        trainer_config: "TrainerConfig",
    ):
        """
        Args:
            loggers (List[Logger]): List of logger instances to manage.
            strategy (Parallel): Distributed training strategy to determine rank-zero logging.
            config (ConfigProtocol): Training configuration object.
        """
        self.loggers: List["Logger"] = loggers
        self.strategy: "Parallel" = strategy
        self.log_frequency: int = trainer_config.log_interval

    def log_config(self, config: "TrainerConfig") -> None:
        """
        Log the configuration to all loggers and print to console.

        Args:
            config (ConfigProtocol): Training configuration object.
        """
        cfg_dict: Dict[str, Any] = convert_paths(config_to_dict(config))
        print("\n" + "=" * 20 + " ConfigProtocol " + "=" * 20)
        print(json.dumps(cfg_dict, indent=2))
        print("=" * 50 + "\n")

        for logger in self.loggers:
            if hasattr(logger, "log_config"):
                logger.log_config(cfg_dict)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log metrics to all attached loggers respecting rank-zero-only settings.

        Args:
            metrics (Dict[str, float]): Metric name-value pairs.
            step (int): Current training step.
        """
        for logger in self.loggers:
            if not logger.rank_zero_only or (
                self.strategy and self.strategy.is_main_process()
            ):
                logger.log_metrics(metrics, step)

    def log_hparams(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters to all attached loggers respecting rank-zero-only settings.

        Args:
            params (Dict[str, Any]): Hyperparameter name-value pairs.
        """
        for logger in self.loggers:
            if not logger.rank_zero_only or (
                self.strategy and self.strategy.is_main_process()
            ):
                logger.log_hparams(params)

    def finalize(self) -> None:
        """
        Finalize all loggers.
        """
        for logger in self.loggers:
            logger.finalize()


def create_loggers(
    strategy: "Parallel", trainer_config: "TrainerConfig"
) -> LoggerManager:
    """
    Factory function to create a LoggerManager from configuration.

    Args:
        strategy (Parallel): Distributed training strategy.
        config (ConfigProtocol): Training configuration object.

    Returns:
        LoggerManager: Manager containing the requested loggers.
    """
    logger_configs: List[str] = list(trainer_config.logger)
    loggers: List["Logger"] = []

    for logger_name in logger_configs:
        if logger_name == "console":
            loggers.append(console_logger.ConsoleLogger(trainer_config))
        elif logger_name == "tensorboard":
            loggers.append(tensorboard_logger.TensorBoardLogger(trainer_config))

    return LoggerManager(loggers, strategy, trainer_config)
