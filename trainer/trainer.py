from pathlib import Path
from contextlib import nullcontext
import shutil
from typing import Optional, TYPE_CHECKING, List
import time

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

from ai_playground.distributed.base import Parallel
from ai_playground.utils.logger.logger_manager import (
    LoggerManager,
    create_loggers,
    BASELINE_METRICS,
)
from ai_playground.utils.utils import (
    precision_to_dtype,
    build_lr_scheduler,
    get_norm_info,
    setup_progress_bar,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from torch.optim import Optimizer, lr_scheduler
    from tqdm import tqdm
    from ai_playground.configs.config import ConfigProtocol


class Trainer:
    def __init__(
        self,
        config: "ConfigProtocol",
        strategy: Parallel,
        optimizer: Optional[Optimizer] = None,
    ):
        self.optimizer: Optional[Optimizer] = optimizer
        self.lr_scheduler: Optional[lr_scheduler.LambdaLR] = None
        self.global_step: int = 0
        self.step_time_accumulator: float = 0.0
        self.data_time_accumulator: float = 0.0
        self.tokens_accumulator: int = 0

        self.strategy: Parallel = strategy

        self.config: "ConfigProtocol" = config
        self.set_seed(
            config.experimental.seed + (self.strategy.world_size * self.strategy.rank)
        )
        self.device_type = self.strategy.device_type
        self.device = self.strategy.device

        self.precision: str = config.trainer.precision
        self.precision_dtype: torch.dtype = precision_to_dtype(self.precision)
        self.use_amp: bool = self.device_type == "cuda" and self.precision in (
            "fp16",
            "bf16",
        )
        self.scaler = torch.amp.grad_scaler.GradScaler(
            "cuda", enabled=self.use_amp and self.precision == "fp16"
        )

        self.max_epochs: int = config.trainer.max_epochs
        self.max_steps: int = config.trainer.max_steps
        self.save_interval: int = config.trainer.save_interval
        self.val_interval: int = config.trainer.val_interval
        self.log_interval = config.trainer.log_interval
        self.logger_manager: LoggerManager = create_loggers(self.strategy, config)
        self.logger_metrics: set = set(BASELINE_METRICS)

        self.use_progress_bar: bool = config.trainer.use_progress_bar
        self.progress_bar: Optional[tqdm] = None
        self.progress_bar_metrics = BASELINE_METRICS if self.use_progress_bar else None

        self.use_profiler: bool = config.trainer.use_profiler
        self.profiler: Optional[profile] = None
        if self.use_profiler:
            activities = (
                [ProfilerActivity.CPU, ProfilerActivity.CUDA]
                if self.device_type == "cuda"
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

        self.experiment_name: str = config.experimental.experiment_name
        self.val_loss_history: List[dict] = []

    def set_seed(self, seed: int = 42):
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def configure_optimizer_and_scheduler(self, model: nn.Module):
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
        self.lr_scheduler = build_lr_scheduler(self.optimizer, self.config)

    def _pre_fit(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
    ):
        self.logger_manager.log_config(self.config)
        self.strategy.setup_environment()
        model = self.strategy.wrap_model(model)

        # Configure optimizer and scheduler
        if self.optimizer is None:
            self.configure_optimizer_and_scheduler(model)

        # Try to load checkpoint
        step = 0
        latest_ckpt = self._latest_checkpoint_path()
        if latest_ckpt is not None:
            print(f"Loading checkpoint from: {latest_ckpt}")
            step = self.load_checkpoint(model)
            print(f"Resuming training from step: {step}")
        else:
            print("No checkpoint found, starting training from scratch")

        # Setup progress bar
        if self.use_progress_bar and self.progress_bar is None:
            self.progress_bar = setup_progress_bar(
                initial_step=step, total_steps=self.max_steps
            )

        return model, train_dataloader, val_dataloader

    def fit(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
    ):
        try:
            # Pretraining setup
            model, train_dataloader, val_dataloader = self._pre_fit(
                model, train_dataloader, val_dataloader
            )
            assert (
                self.optimizer is not None
            ), "Optimizer must be configured before training"
            assert (
                self.lr_scheduler is not None
            ), "LR Scheduler must be configured before training"

            # Train
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
        finally:
            self._cleanup()

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

            if self.strategy.device_type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            # Forward and backward pass
            logits, loss = self._train_step(model, xb, yb, optimizer, scheduler)

            if self.strategy.device_type == "cuda":
                torch.cuda.synchronize()
            self.step_time_accumulator += time.perf_counter() - start
            self.tokens_accumulator += xb.numel()

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
        data_start = time.perf_counter()
        xb, yb = xb.to(self.strategy.device), yb.to(self.strategy.device)
        self.data_time_accumulator += time.perf_counter() - data_start

        with torch.autocast(
            device_type=self.strategy.device_type,
            dtype=self.precision_dtype,
            enabled=self.use_amp,
        ):
            logits, loss = model(xb, yb)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss detected at step {self.global_step}: {loss}"
            )

        self.strategy.backward(self.scaler.scale(loss))
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise RuntimeError(
                    f"Non-finite gradient detected at step {self.global_step} for parameter {p.shape}"
                )
        self.scaler.unscale_(optimizer)
        if self.config.trainer.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.config.trainer.grad_clip
            )
        self.strategy.optimizer_step(optimizer, self.scaler)
        scheduler.step()

        self.global_step += 1
        if self.progress_bar is not None:
            self.progress_bar.update(1)
        return logits, loss

    def _maybe_log(self, model: nn.Module, scheduler, loss: torch.Tensor):
        """Logs training metrics if we hit the logging interval."""
        if self.logger_manager.log_frequency <= 0:
            return

        if self.global_step % self.logger_manager.log_frequency != 0:
            return

        metrics = {}
        if "train_loss" in self.logger_metrics:
            metrics["train_loss"] = loss.item()

        if "lr" in self.logger_metrics:
            metrics["lr"] = scheduler.get_last_lr()[0]

        if {"grad_norm", "weight_norm", "update_ratio"} & self.logger_metrics:
            grad_norm, weight_norm, update_ratio = get_norm_info(
                model, scheduler.get_last_lr()[0]
            )
            if grad_norm in self.logger_metrics:
                metrics["grad_norm"] = grad_norm
            if weight_norm in self.logger_metrics:
                metrics["weight_norm"] = weight_norm
            if update_ratio in self.logger_metrics:
                metrics["update_ratio"] = update_ratio

        if "tps" in self.logger_metrics:
            avg_step_time = (
                self.step_time_accumulator / self.logger_manager.log_frequency
            )
            avg_data_time = (
                self.data_time_accumulator / self.logger_manager.log_frequency
            )
            tps = (
                self.tokens_accumulator
                / (self.step_time_accumulator + self.data_time_accumulator)
                if (self.step_time_accumulator + self.data_time_accumulator) > 0
                else 0.0
            )
            metrics["tps"] = tps
            metrics["avg_step_time"] = avg_step_time
            metrics["avg_data_time"] = avg_data_time

        # Include CUDA memory metrics if on GPU
        if self.strategy.device_type == "cuda":
            if "gpu_mem_alloc" in self.logger_metrics:
                metrics["gpu_mem_alloc"] = torch.cuda.memory_allocated() / (1024**3)
            if "gpu_mem_reserved" in self.logger_metrics:
                metrics["gpu_mem_reserved"] = torch.cuda.memory_reserved() / (1024**3)

        # Log metrics to all configured loggers (console, tensorboard, etc.)
        self.logger_manager.log_metrics(metrics, step=self.global_step)

        # Update tqdm / progress bar metrics
        if self.progress_bar_metrics is not None and self.progress_bar is not None:
            for key, value in metrics.items():
                self.progress_bar_metrics[key] = value
            self.progress_bar.set_postfix(self.progress_bar_metrics)

        # Reset accumulators
        self.step_time_accumulator = 0.0
        self.data_time_accumulator = 0.0
        self.tokens_accumulator = 0

    def _maybe_validate(self, model: nn.Module, val_dataloader):
        """Run validation and log metrics if we hit the validation interval."""
        if val_dataloader is None or self.val_interval == 0:
            return

        if (
            self.global_step % self.val_interval != 0
            and not self.global_step == self.max_steps
        ):
            return

        # Compute validation loss
        val_loss = self._validate(model, val_dataloader)
        self.val_loss_history.append({"val_loss": val_loss, "step": self.global_step})

        if "val_loss" in self.logger_metrics:
            metrics = {"val_loss": val_loss}
            self.logger_manager.log_metrics(metrics, step=self.global_step)

            # Update tqdm / progress bar
            if self.progress_bar is not None and self.progress_bar_metrics is not None:
                self.progress_bar_metrics.update(metrics)
                self.progress_bar.set_postfix(self.progress_bar_metrics)

    def _maybe_checkpoint(self, model):
        if self.save_interval > 0 and (self.global_step) % self.save_interval == 0:
            self.save_checkpoint(model)

    def _validate(self, model: nn.Module, val_dataloader: DataLoader):
        was_training = model.training
        model.eval()
        total_loss = torch.zeros(1, device=self.strategy.device)
        count = 0
        with torch.no_grad(), torch.autocast(
            device_type=self.strategy.device_type,
            dtype=self.precision_dtype,
            enabled=self.use_amp,
        ):
            for xb, yb in val_dataloader:
                xb, yb = xb.to(self.strategy.device), yb.to(self.strategy.device)
                _, loss = model(xb, yb)
                total_loss += loss.detach() * xb.size(0)
                count += xb.size(0)
            if was_training:
                model.train()
            return (total_loss / count).item() if count > 0 else 0.0

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
                len(prompts), max_len, dtype=torch.long, device=self.strategy.device
            )
            for i, tokens in enumerate(token_list):
                batch_context[i, : len(tokens)] = torch.tensor(
                    tokens, device=self.strategy.device
                )

            output_tokens = batch_context.clone()
            for _ in range(max_tokens):
                logits, _ = model(output_tokens)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                output_tokens = torch.cat([output_tokens, next_token], dim=1)

        return [tokenizer.decode(seq.tolist()) for seq in output_tokens]

    def save_checkpoint(self, model: nn.Module):
        if self.strategy.is_main_process():
            path = self.config.trainer.save_path
            if self.experiment_name != "":
                path = Path(path) / self.experiment_name
            Path(path).mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "model": self.strategy.unwrap_model(model).state_dict(),
                "optimizer": self.optimizer.state_dict() if self.optimizer else None,
                "scheduler": (
                    self.lr_scheduler.state_dict() if self.lr_scheduler else None
                ),
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
        if not path.exists():
            print(f"No checkpoint found at {path}")
            return 0

        checkpoint = torch.load(path, map_location=self.strategy.device)
        model.load_state_dict(checkpoint["model"])

        if not self.optimizer:
            self.configure_optimizer_and_scheduler(model)
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

    def _latest_checkpoint_path(self):
        path = self.config.trainer.save_path
        if self.experiment_name != "":
            path = Path(path) / self.experiment_name

        ckpt = Path(path) / "ckpt_latest.pt"

        return ckpt if ckpt.exists() else None

    def _cleanup(self):
        self.logger_manager.finalize()
        if self.progress_bar is not None:
            self.progress_bar.close()
