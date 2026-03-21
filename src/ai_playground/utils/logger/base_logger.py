from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, Any
    from ai_playground.configs.config import Config


class Logger(ABC):
    """
    Abstract base logger class.

    Attributes:
        log_frequency (int): How often to log metrics (in steps).
        rank_zero_only (bool): Whether to log only from rank 0 in distributed setups.
        log_dir (Path): Directory where logs will be stored.
    """

    def __init__(self, config: "Config"):
        """
        Args:
            config (ConfigProtocol): Training configuration object.
        """
        self.log_frequency: int = config.trainer.log_interval
        self.rank_zero_only: bool = True
        self.log_dir: Path = Path(config.trainer.log_dir)

        if getattr(config.experimental, "experiment_name", "") != "":
            self.log_dir = self.log_dir / config.experimental.experiment_name

        os.makedirs(self.log_dir, exist_ok=True)

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log the training configuration.

        Subclasses can override this to save configs to file or console.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """
        Log training/validation metrics.

        Args:
            metrics (Dict[str, Any]): Metric name-value pairs.
            step (int): Training step.
        """
        pass

    @abstractmethod
    def log_hparams(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters.

        Args:
            params (Dict[str, Any]): Hyperparameter name-value pairs.
        """
        pass

    @abstractmethod
    def finalize(self) -> None:
        """
        Finalize logging, flush files, close connections.
        """
        pass
