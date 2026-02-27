from pathlib import Path
import shutil
from typing import Optional

import torch

from config import Config
from utils.logger import Logger


class Trainer:
    def __init__(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        logger: Optional[Logger] = None,
        device: str = "cpu",
        config: Optional[Config] = None,
    ):
        self.device = device
        self.optimizer = optimizer
        self.logger = logger
        self.global_step = 0
        self.config = config
        self.max_steps = 1 if config is None else config.max_steps
        self.save_interval = 100 if config is None else config.save_interval
        self.val_interval = 10 if config is None else config.val_interval

    def configure_optimizer(self, model):
        self.optimizer = (
            self.optimizer
            if self.optimizer is not None
            else torch.optim.AdamW(model.parameters(), lr=0.01)
        )

    def fit(self, model, train_dataloader, val_dataloader=None):
        if self.optimizer is None:
            self.configure_optimizer(model)
        assert (
            self.optimizer is not None
        ), "Optimizer must be configured before training"

        model.to(self.device)
        model.train()

        while self.global_step < self.max_steps:
            for xb, yb in train_dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits, loss = model(xb, yb)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()

                if self.logger:
                    self.logger.update(
                        loss=loss.item(),
                        lr=self.optimizer.param_groups[0]["lr"],
                    )
                    self.logger.log(self.global_step)

                if self.global_step > 0 and self.global_step % self.save_interval == 0:
                    self.save_checkpoint(model)

                self.global_step += 1
                if self.global_step >= self.max_steps:
                    break
            if val_dataloader and self.global_step % self.val_interval == 0:
                val_loss = self.validate(model, val_dataloader)
                if self.logger:
                    self.logger.update(val_loss=val_loss)
                    self.logger.log(self.global_step)

    def validate(self, model, val_dataloader):
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for xb, yb in val_dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                _, loss = model(xb, yb)
                total_loss += loss.item() * xb.size(0)
                count += xb.size(0)

            return total_loss / count if count > 0 else 0.0

    def generate(self, model, tokenizer, context=None, num_tokens=500):
        model.eval()
        with torch.no_grad():
            context = (
                context
                if context is not None
                else torch.zeros((1, 1), dtype=torch.long, device=self.device)
            )
            return tokenizer.decode(
                model.generate(context, max_new_tokens=num_tokens)[0].tolist()
            )

    def save_checkpoint(self, model, path: Optional[str] = None):
        path = path or f"checkpoints"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "step": self.global_step,
            "config": self.config.__dict__ if self.config else None,
        }

        step_path = Path(path) / f"ckpt_step_{self.global_step}.pt"
        torch.save(checkpoint, step_path)

        latest_path = Path(path) / "ckpt_latest.pt"
        temp_path = Path(path) / "ckpt_latest.pt_"
        shutil.copy2(step_path, temp_path)
        temp_path.replace(latest_path)

    def load_checkpoint(self, model, path: Optional[str] = None):
        path = path or "checkpoints/ckpt_latest.pt"
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint["model"])

        if not self.optimizer:
            self.configure_optimizer(model)
        assert (
            self.optimizer is not None
        ), "Optimizer must be configured before loading checkpoint"
        if checkpoint.get("optimizer") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.global_step = checkpoint.get("step", 0)

        return self.global_step
