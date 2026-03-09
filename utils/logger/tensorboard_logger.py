from torch.utils.tensorboard import SummaryWriter

from ai_playground.configs.config import Config
from ai_playground.utils.logger.base_logger import Logger


class TensorBoardLogger(Logger):
    def __init__(self, config):
        super().__init__(config)
        self.writer = SummaryWriter(self.log_dir)

    def log_config(self, config: dict):
        import json

        self.writer.add_text("config", json.dumps(config, indent=2))

    def log_metrics(self, metrics: dict, step: int):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)

    def log_hparams(self, params: dict):
        for k, v in params.items():
            self.writer.add_text("hparams/" + k, str(v))

    def finalize(self):
        self.writer.close()
