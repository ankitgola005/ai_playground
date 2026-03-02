from pathlib import Path
import shutil
from contextlib import nullcontext
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, ProfilerActivity

if TYPE_CHECKING:
    from configs.config import Config
    from torch.utils.data import DataLoader
    from torch.optim import Optimizer


class Trainer:
    def __init__(
        self,
        config: "Config",
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.optimizer: Optional[torch.optim.Optimizer] = optimizer
        self.global_step: int = 0

        self.config: "Config" = config
        self.device: str = config.experimental.device
        self.max_epochs: int = config.trainer.max_epochs
        self.max_steps: int = config.trainer.max_steps
        self.save_interval: int = config.trainer.save_interval
        self.val_interval: int = config.trainer.val_interval
        self.experiment_name: str = config.experimental.experiment_name

        self.logger: Optional[str] = config.trainer.logger
        if self.logger == "tensorboard":
            log_dir = config.trainer.log_dir
            if self.experiment_name != "":
                log_dir = Path(log_dir) / self.experiment_name
            self.writer = SummaryWriter(log_dir=log_dir)

        self.use_profiler: bool = config.trainer.use_profiler
        self.profiler: Optional[torch.profiler.profile] = None
        if self.use_profiler:
            activities = (
                [ProfilerActivity.CPU, ProfilerActivity.CUDA]
                if self.device == "cuda"
                else [ProfilerActivity.CPU]
            )
            self.profiler = profile(
                activities=activities,
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
                schedule=torch.profiler.schedule(
                    wait=config.trainer.profiler_wait,
                    warmup=config.trainer.profiler_warmup,
                    active=config.trainer.profiler_active,
                    repeat=config.trainer.profiler_repeat,
                ),
            )

    def configure_optimizer(self, model: nn.Module):
        self.optimizer = (
            self.optimizer
            if self.optimizer is not None
            else torch.optim.AdamW(model.parameters(), lr=0.01)
        )

    def fit(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ):
        if self.optimizer is None:
            self.configure_optimizer(model)
        assert (
            self.optimizer is not None
        ), "Optimizer must be configured before training"

        model.to(self.device)
        profiler_context = self.profiler if self.profiler else nullcontext()
        with profiler_context:
            for epoch in range(self.max_epochs):
                if self._should_stop():
                    break

                model.train()
                self._train_one_epoch(
                    model, self.optimizer, train_dataloader, val_dataloader
                )

        if self.logger == "tensorboard":
            self.writer.close()

    def _train_one_epoch(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
    ):
        for xb, yb in train_dataloader:
            if self._should_stop():
                break
            xb, yb = xb.to(self.device), yb.to(self.device)
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            self.global_step += 1

            # Logging and profiling
            if self.profiler is not None:
                self.profiler.step()

            # grads = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
            # total_norm = (
            #     torch.norm(torch.stack(grads))
            #     if grads
            #     else torch.tensor(0.0, device=self.device)
            # )
            if self.logger == "tensorboard":
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                # self.writer.add_scalar(
                #    "train/grad_norm", total_norm.item(), self.global_step
                # )

            # Validation
            if (
                val_dataloader is not None
                and self.val_interval > 0
                and (self.global_step) % self.val_interval == 0
            ):
                val_loss = self._validate(model, val_dataloader)
                if self.logger == "tensorboard":
                    self.writer.add_scalar("val/loss", val_loss, self.global_step)

            # Checkpointing
            if self.save_interval > 0 and (self.global_step) % self.save_interval == 0:
                self.save_checkpoint(model)

    def _validate(self, model: nn.Module, val_dataloader: DataLoader):
        is_training = model.training
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for xb, yb in val_dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                _, loss = model(xb, yb)
                total_loss += loss.item() * xb.size(0)
                count += xb.size(0)
            if is_training:
                model.train()
            return total_loss / count if count > 0 else 0.0

    def _should_stop(self) -> bool:
        return self.max_steps > 0 and self.global_step >= self.max_steps

    def generate(self, model, tokenizer, context=None, num_tokens: int = 500):
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

    def save_checkpoint(self, model: nn.Module):
        path = self.config.trainer.save_path
        if self.experiment_name != "":
            path = Path(path) / self.experiment_name
        Path(path).mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "step": self.global_step,
            "config": self.config.__dict__,
        }

        step_path = Path(path) / f"ckpt_step_{self.global_step}.pt"
        torch.save(checkpoint, step_path)

        latest_path = Path(path) / "ckpt_latest.pt"
        temp_path = Path(path) / "ckpt_latest.pt_"
        shutil.copy2(step_path, temp_path)
        temp_path.replace(latest_path)

    def load_checkpoint(self, model: nn.Module) -> int:
        path = self.config.trainer.save_path
        if self.experiment_name != "":
            path = Path(path) / self.experiment_name
        path = Path(path) / "ckpt_latest.pt"
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
