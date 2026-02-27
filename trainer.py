from typing import Optional
from utils.logger import Logger

import torch


class Trainer:
    def __init__(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        logger: Optional[Logger] = None,
        device: str = "cpu",
        max_steps: int = 1,
    ):
        self.max_steps = max_steps
        self.device = device
        self.optimizer = optimizer
        self.logger = logger

    def configure_optimizer(self, model):
        self.optimizer = (
            self.optimizer
            if self.optimizer is not None
            else torch.optim.AdamW(model.parameters(), lr=0.01)
        )

    def fit(self, model, dataloader):
        if self.optimizer is None:
            self.configure_optimizer(model)
        assert (
            self.optimizer is not None
        ), "Optimizer must be configured before training"

        model.to(self.device)
        model.train()

        step = 0
        loss = []
        while step < self.max_steps:
            for xb, yb in dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits, loss = model(xb, yb)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                if self.logger:
                    self.logger.update(
                        loss=loss.item(),
                        lr=self.optimizer.param_groups[0]["lr"],
                    )
                    self.logger.log(step)

                step += 1
                if step >= self.max_steps:
                    break

    def generate(self, model, tokenizer, context=None, num_tokens=500):
        model.eval()
        context = (
            context
            if context is not None
            else torch.zeros((1, 1), dtype=torch.long, device=self.device)
        )
        return tokenizer.decode(
            model.generate(context, max_new_tokens=num_tokens)[0].tolist()
        )
