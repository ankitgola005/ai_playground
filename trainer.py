from pathlib import Path
from contextlib import nullcontext
import shutil
from typing import Optional, TYPE_CHECKING
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, ProfilerActivity

from utils.utils import precision_to_dtype, _build_lr_scheduler

if TYPE_CHECKING:
    from configs.config import Config
    from torch.utils.data import DataLoader
    from torch.optim import Optimizer, lr_scheduler


class Trainer:
    def __init__(
        self,
        config: "Config",
        optimizer: Optional[Optimizer] = None,
    ):
        self.optimizer: Optional[Optimizer] = optimizer
        self.lr_scheduler: Optional[lr_scheduler.LambdaLR] = None
        self.global_step: int = 0
        self.step_time_accumulator: float = 0.0
        self.tokens_accumulator: int = 0

        self.config: "Config" = config
        self.device: str = (
            "cuda"
            if config.experimental.device == "cuda" and torch.cuda.is_available()
            else "cpu"
        )

        self.precision: str = config.trainer.precision
        self.precision_dtype: torch.dtype = precision_to_dtype(self.precision)
        self.use_amp: bool = self.device == "cuda" and self.precision in (
            "fp16",
            "bf16",
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp and self.precision == "fp16"
        )

        self.max_epochs: int = config.trainer.max_epochs
        self.max_steps: int = config.trainer.max_steps
        self.save_interval: int = config.trainer.save_interval
        self.val_interval: int = config.trainer.val_interval
        self.log_interval = config.trainer.log_interval
        self.experiment_name: str = config.experimental.experiment_name

        self.logger: Optional[str] = config.trainer.logger
        if self.logger == "tensorboard":
            log_dir = config.trainer.log_dir
            if self.experiment_name != "":
                log_dir = Path(log_dir) / self.experiment_name
            self.writer = SummaryWriter(log_dir=log_dir)

        self.use_profiler: bool = config.trainer.use_profiler
        self.profiler: Optional[profile] = None
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
        decay = []
        no_decay = []

        for name, param in model.named_parameters():
            if param.ndim == 1 or name.endswith("bias"):
                no_decay.append(param)
            else:
                decay.append(param)

        self.optimizer = (
            self.optimizer
            if self.optimizer is not None
            else torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": self.config.trainer.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=self.config.trainer.lr,
                betas=self.config.trainer.betas,
            )
        )
        self.lr_scheduler = _build_lr_scheduler(
            self.optimizer,
            warmup_steps=self.config.trainer.warmup_steps,
            max_steps=self.max_steps,
            min_lr_ratio=self.config.trainer.min_lr_ratio,
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
        assert (
            self.lr_scheduler is not None
        ), "LR Scheduler must be configured before training"

        model.to(self.device)
        profiler_context = self.profiler if self.profiler else nullcontext()
        with profiler_context:
            for epoch in range(self.max_epochs):
                if self._should_stop():
                    break

                model.train()
                self._train_one_epoch(
                    model,
                    self.optimizer,
                    self.lr_scheduler,
                    train_dataloader,
                    val_dataloader,
                )

        if self.logger == "tensorboard":
            self.writer.close()

    def _train_one_epoch(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: lr_scheduler.LambdaLR,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
    ):
        for xb, yb in train_dataloader:
            if self._should_stop():
                break

            if self.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            # Forward and backward pass
            logits, loss = self._train_step(model, xb, yb, optimizer, scheduler)

            if self.device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            self.tokens_accumulator += xb.numel()
            self.step_time_accumulator += end - start

            # Profiling
            if self.profiler is not None:
                self.profiler.step()

            # Logging
            self._maybe_log(model, scheduler, loss)

            # Validation
            self._maybe_validate(model, val_dataloader)

            # Checkpointing
            self._maybe_checkpoint(model)

    def _train_step(
        self,
        model: nn.Module,
        xb: torch.Tensor,
        yb: torch.Tensor,
        optimizer: Optimizer,
        scheduler: lr_scheduler.LambdaLR,
    ):
        optimizer.zero_grad(set_to_none=True)
        xb, yb = xb.to(self.device), yb.to(self.device)
        with torch.autocast(
            device_type=self.device,
            dtype=self.precision_dtype,
            enabled=self.use_amp,
        ):
            logits, loss = model(xb, yb)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.scaler.step(optimizer)
        self.scaler.update()
        scheduler.step()

        self.global_step += 1
        return logits, loss

    def _maybe_log(
        self, model: nn.Module, scheduler: lr_scheduler.LambdaLR, loss: torch.Tensor
    ):
        if self.log_interval > 0 and self.global_step % self.log_interval == 0:
            avg_step_time = self.step_time_accumulator / self.log_interval
            grads = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
            total_norm = (
                torch.norm(torch.stack(grads))
                if grads
                else torch.tensor(0.0, device=self.device)
            )
            kwargs = {
                "loss": loss.item(),
                "lr": scheduler.get_last_lr()[0],
                "scaler_scale": self.scaler.get_scale() if self.use_amp else None,
                "total_norm": total_norm,
                "tps": (
                    self.tokens_accumulator / self.step_time_accumulator
                    if self.step_time_accumulator > 0
                    else 0.0
                ),
                "avg_step_time": avg_step_time,
            }

            self._log(kwargs)
            self.step_time_accumulator = 0.0
            self.tokens_accumulator = 0

    def _log(self, metrics: dict, prefix: str = "train"):
        if self.logger != "tensorboard":
            return
        for key, value in metrics.items():
            if value is not None:
                self.writer.add_scalar(f"{prefix}_{key}", value, self.global_step)

    def _maybe_validate(self, model, val_dataloader):
        if (
            val_dataloader is not None
            and self.val_interval > 0
            and (self.global_step) % self.val_interval == 0
        ):
            val_loss = self._validate(model, val_dataloader)
            if self.logger == "tensorboard":
                self.writer.add_scalar("val/loss", val_loss, self.global_step)

    def _maybe_checkpoint(self, model):
        if self.save_interval > 0 and (self.global_step) % self.save_interval == 0:
            self.save_checkpoint(model)

    def _validate(self, model: nn.Module, val_dataloader: DataLoader):
        is_training = model.training
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad(), torch.autocast(
            device_type=self.device, dtype=self.precision_dtype, enabled=self.use_amp
        ):
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

    def predict(
        self, model: nn.Module, tokenizer, prompts: list[str], max_tokens: int = 500
    ):
        model.eval()
        with torch.inference_mode():
            token_list = [tokenizer.encode(c) for c in prompts]
            max_len = max(len(t) for t in token_list)
            batch_context = torch.zeros(
                len(prompts), max_len, dtype=torch.long, device=self.device
            )
            for i, tokens in enumerate(token_list):
                batch_context[i, : len(tokens)] = torch.tensor(
                    tokens, device=self.device
                )

            output_tokens = batch_context.clone()
            for _ in range(max_tokens):
                logits, _ = model(output_tokens)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                output_tokens = torch.cat([output_tokens, next_token], dim=1)

        return [tokenizer.decode(seq.tolist()) for seq in output_tokens]

    def save_checkpoint(self, model: nn.Module):
        path = self.config.trainer.save_path
        if self.experiment_name != "":
            path = Path(path) / self.experiment_name
        Path(path).mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "step": self.global_step,
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
        if self.lr_scheduler and checkpoint.get("scheduler") is not None:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler"])

        if self.scaler and checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.global_step = checkpoint.get("step", 0)

        return self.global_step
