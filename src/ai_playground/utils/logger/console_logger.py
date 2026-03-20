from typing import TYPE_CHECKING

from ai_playground.utils.logger.base_logger import Logger

if TYPE_CHECKING:
    from typing import Dict, Any
    from ai_playground.configs.config import ConfigProtocol


class ConsoleLogger(Logger):
    """
    Logger that outputs metrics and hyperparameters to the console.
    """

    def __init__(self, config: "ConfigProtocol") -> None:
        """
        Initialize the ConsoleLogger.

        Args:
            config (ConfigProtocol): Training configuration object.
        """
        super().__init__(config)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Print metrics to the console.

        Args:
            metrics (Dict[str, float]): Metric name-value pairs.
            step (int): Current training step.
        """
        metrics_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"[step {step}] {metrics_str}")

    def log_hparams(self, params: Dict[str, Any]) -> None:
        """
        Print hyperparameters to the console.

        Args:
            params (Dict[str, Any]): Hyperparameter name-value pairs.
        """
        print("Hyperparameters:")
        for k, v in params.items():
            print(f"{k}: {v}")

    def finalize(self) -> None:
        """
        Finalize logging. No-op for console logger.
        """
        pass
