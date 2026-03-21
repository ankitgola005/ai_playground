from typing import TYPE_CHECKING
from torch.utils.tensorboard import SummaryWriter

from ai_playground.utils.logger import Logger

if TYPE_CHECKING:
    from typing import Dict, Any
    from ai_playground.configs import TrainerConfig


class TensorBoardLogger(Logger):
    """
    TensorBoard Logger.
    """

    def __init__(self, config: "TrainerConfig") -> None:
        """
        Initialize the TensorBoard logger.

        Args:
            config (ConfigProtocol): Training configuration object.
        """
        super().__init__(config)
        self.writer: SummaryWriter = SummaryWriter(self.log_dir)

    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log the training configuration as JSON text in TensorBoard.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        import json

        self.writer.add_text("config", json.dumps(config, indent=2))

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log training/validation metrics to TensorBoard.

        Args:
            metrics (Dict[str, float]): Metric name-value pairs.
            step (int): Current training step.
        """
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)

    def log_hparams(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters to TensorBoard.

        Args:
            params (Dict[str, Any]): Hyperparameter name-value pairs.
        """
        for k, v in params.items():
            self.writer.add_text(f"hparams/{k}", str(v))

    def finalize(self) -> None:
        """
        Close the TensorBoard writer.
        """
        self.writer.close()
