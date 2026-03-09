from typing import TYPE_CHECKING

from ai_playground.utils.logger.base_logger import Logger

if TYPE_CHECKING:
    from ai_playground.configs.config import ConfigProtocol


class ConsoleLogger(Logger):
    def __init__(self, config: ConfigProtocol) -> None:
        super().__init__(config)

    def log_metrics(self, metrics: dict, step: int):
        metrics_str = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"[step {step}] {metrics_str}")

    def log_hparams(self, params: dict):
        print("Hyperparameters:")
        for k, v in params.items():
            print(f"{k}: {v}")

    def finalize(self):
        pass
