from ai_playground.utils.logger import console_logger, tensorboard_logger
import json
from ai_playground.utils.load_yaml_config import config_to_dict

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ai_playground.configs.config import ConfigProtocol
    from ai_playground.distributed.base import Parallel
    from ai_playground.utils.logger.base_logger import Logger

BASELINE_METRICS: dict = {
    "train_loss": 0.0,
    "val_loss": 0.0,
    "lr": 0.0,
    "tps": 0.0,
}

FULL_METRICS: dict = {
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
    def __init__(
        self, logger: List[Logger], strategy: Parallel, config: ConfigProtocol
    ):
        self.loggers: List[Logger] = logger
        self.strategy = strategy
        self.log_frequency: int = config.trainer.log_interval

    def log_config(self, config: ConfigProtocol):
        cfg = config_to_dict(config)
        print("\n" + "=" * 20 + " ConfigProtocol " + "=" * 20)
        print(json.dumps(cfg, indent=2))
        print("=" * 50 + "\n")

        for logger in self.loggers:
            if hasattr(logger, "log_config"):
                logger.log_config(cfg)  # type: ignore

    def log_metrics(self, metrics: dict, step: int):
        for logger in self.loggers:
            if not logger.rank_zero_only or (
                self.strategy and self.strategy.is_main_process()
            ):
                logger.log_metrics(metrics, step)

    def log_hparams(self, params: dict):
        for logger in self.loggers:
            if not logger.rank_zero_only or (
                self.strategy and self.strategy.is_main_process()
            ):
                logger.log_hparams(params)

    def finalize(self):
        for logger in self.loggers:
            logger.finalize()


def create_loggers(strategy: Parallel, config: ConfigProtocol):
    logger_configs = config.trainer.logger
    loggers: List[Logger] = []
    for logger in logger_configs:
        if logger == "console":
            loggers.append(console_logger.ConsoleLogger(config))
        elif logger == "tensorboard":
            loggers.append(tensorboard_logger.TensorBoardLogger(config))
    return LoggerManager(loggers, strategy, config)
