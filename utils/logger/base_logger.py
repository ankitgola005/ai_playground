from pathlib import Path
from abc import ABC, abstractmethod
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_playground.configs.config import ConfigProtocol


class Logger(ABC):
    def __init__(self, config: ConfigProtocol):
        self.log_frequency = config.trainer.log_interval
        self.rank_zero_only = True
        self.log_dir = config.trainer.log_dir
        if config.experimental.experiment_name != "":
            self.log_dir = Path(self.log_dir) / config.experimental.experiment_name
        os.makedirs(self.log_dir, exist_ok=True)

    def log_config(self, config: dict):
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict, step: int):
        pass

    @abstractmethod
    def log_hparams(self, params: dict):
        pass

    @abstractmethod
    def finalize(self):
        pass
