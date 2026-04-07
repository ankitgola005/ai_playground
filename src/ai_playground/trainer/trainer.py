from contextlib import nullcontext
from typing import TYPE_CHECKING, List, Dict, Any, Iterable
import time

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler

from ai_playground.inference.generator import Generator
from ai_playground.utils.logger.logger_manager import (
    LoggerManager,
    create_loggers,
    BASELINE_METRICS,
)
from ai_playground.utils import (
    set_seed,
    precision_to_dtype,
    build_lr_scheduler,
    get_norm_info,
    setup_progress_bar,
    get_profiler,
    save_checkpoint,
    load_checkpoint,
    create_infinite_loader,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from torch.profiler.profiler import profile as Profiler
    from torch.optim import Optimizer, lr_scheduler
    from tqdm import tqdm
    from ai_playground.configs.config import Config
    from ai_playground.distributed.base import Parallel


class Trainer:
    """
    Handles training, validation, checkpointing, and inference.

    Supports:
    - Distributed strategies
    - Mixed precision (fp16/bf16)
    - Gradient scaling and clipping
    - Logging, profiling, checkpointing
    """

    def __init__(
        self,
        config: "Config",
        strategy: Parallel,
        optimizer: Optimizer | None = None,
        logger_metrics: Iterable[str] | None = None,
    ):
        self.optimizer: Optimizer | None = optimizer
        self.lr_scheduler: lr_scheduler.LambdaLR | None = None

        self.global_step: int = 0
        self.step_time_accumulator: float = 0.0
        self.data_time_accumulator: float = 0.0
        self.tokens_accumulator: int = 0

        self.strategy: Parallel = strategy
        self.config: "Config" = config

        set_seed(config.trainer.seed + (self.strategy.world_size * self.strategy.rank))

        self.device_type: str = self.strategy.device_type
        self.device: torch.device = self.strategy.device
        self.compile: bool = self.config.model.compile

        self.precision: str = config.trainer.precision
        self.precision_dtype: torch.dtype = precision_to_dtype(self.precision)
        self.use_amp: bool = self.precision in ("fp16", "bf16")

        self.scaler: GradScaler | None = (
            GradScaler(self.device_type, enabled=True)
            if self.use_amp and self.precision == "fp16"
            else None
        )

        self._prepared_models: dict[tuple[int, str], nn.Module] = {}

        self.warmup_steps: int = config.trainer.warmup_steps
        self.max_steps: int = config.trainer.max_steps
        self.save_interval: int = config.trainer.save_interval
        self.val_interval: int = config.trainer.val_interval

        self.log_interval = config.trainer.log_interval
        self.logger_manager: LoggerManager = create_loggers(
            self.strategy, config.trainer
        )

        base_metrics: set = set(BASELINE_METRICS)
        extra_metrics: set = set(logger_metrics) if logger_metrics else set()
        self.logger_metrics: set[str] = base_metrics | extra_metrics

        self.use_progress_bar: bool = config.trainer.use_progress_bar
        self.progress_bar: tqdm | None = None
        self.progress_bar_metrics_names = (
            BASELINE_METRICS if self.use_progress_bar else set()
        )
        self.progress_bar_metrics = {}

        self.use_profiler: bool = config.trainer.use_profiler
        self.profiler: Profiler | None = None
        if self.use_profiler:
            assert self.config.trainer.log_dir is not None
            self.profiler = get_profiler(
                self.config.trainer.profiler_config,
                device_type=self.strategy.device_type,
                log_dir=self.config.trainer.log_dir,
            )

        self.run_name: str = config.trainer.run_name
        self.generator: Generator | None = None
        self.val_loss_history: List[dict] = []
        self.check_finite_grads: bool = False
        self.history: List = []

    def configure_optimizer_and_scheduler(self, model: nn.Module):
        decay, no_decay = [], []

        for name, param in model.named_parameters():
            if param.ndim == 1 or name.endswith("bias"):
                no_decay.append(param)
            else:
                decay.append(param)

        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": self.config.trainer.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=self.config.trainer.lr_config.lr,
                betas=self.config.trainer.betas,
            )

        self.lr_scheduler = build_lr_scheduler(
            optimizer=self.optimizer,
            lr_config=self.config.trainer.lr_config,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
        )

    def _prepare_model(self, model: nn.Module, stage: str = "train"):
        self.strategy.setup_environment(stage=stage)
        raw_model = self._unwrap_model(model)
        cache_key = (id(raw_model), stage)
        if cache_key in self._prepared_models:
            return self._prepared_models[cache_key]

        wrapped_model = self.strategy.wrap_model(raw_model, stage=stage)
        prepared_model = torch.compile(wrapped_model) if self.compile else wrapped_model
        self._prepared_models[cache_key] = prepared_model
        return prepared_model

    def _pre_fit(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None,
    ):
        self.logger_manager.log_config(self.config.trainer)

        # Try to load checkpoint
        step = 0
        print("Looking for saved checkpoints")
        step = self.load_checkpoint(model)
        if step < 0:
            print("No checkpoint found, starting training from scratch")
        else:
            print(f"Resuming training from step: {step}")

        model = self._prepare_model(model)  # type: ignore

        # Configure optimizer and scheduler
        if self.optimizer is None:
            self.configure_optimizer_and_scheduler(model)

        # Setup progress bar
        if self.use_progress_bar and self.progress_bar is None:
            initial_step = max(step, 0)
            self.progress_bar = setup_progress_bar(
                initial_step=initial_step, total_steps=self.max_steps
            )

        return model, train_dataloader, val_dataloader

    def fit(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None,
    ):
        """
        Main training loop.

        Args:
            model (nn.Module): model
            train_dataloader (DataLoader): train dataloader
            val_dataloader (Optional[DataLoader]): val dataloader
        """
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
                model.train()
                train_iter = create_infinite_loader(train_dataloader)

                while not self._should_stop():
                    xb, yb = next(train_iter)

                    if self.strategy.device_type == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()

                    # Forward and backward pass
                    logits, loss, aux_metrics = self._train_step(
                        model, xb, yb, self.optimizer, self.lr_scheduler
                    )
                    if aux_metrics and "moe" in aux_metrics:
                        loss += aux_metrics.get("load_balance_loss", 0.0)

                    if self.strategy.device_type == "cuda":
                        torch.cuda.synchronize()
                    self.step_time_accumulator += time.perf_counter() - start
                    self.tokens_accumulator += xb.numel()

                    # Profiling
                    if self.profiler is not None:
                        self.profiler.step()

                    # Logging
                    self._maybe_log(model, self.lr_scheduler, loss, aux_metrics)

                    # Validation
                    self._maybe_validate(model, val_dataloader)

                    # Checkpointing
                    self._maybe_checkpoint(model)

        finally:
            self._cleanup()

    def _train_step(
        self,
        model: nn.Module,
        xb: torch.Tensor,
        yb: torch.Tensor,
        optimizer: Optimizer,
        scheduler: lr_scheduler.LambdaLR,
    ):
        """
        Single training step

        Args:
            model (nn.Module): model
            xb (torch.Tensor): input data
            yb (torch.Tensor): target
            optimizer (Optimizer): optimizer
            scheduler (lr_scheduler.LambdaLR): learning rate scheduler

        Raises:
            RuntimeError: Non finite loss or grads
        """
        optimizer.zero_grad(set_to_none=True)
        data_start = time.perf_counter()
        xb, yb = xb.to(self.strategy.device), yb.to(self.strategy.device)
        self.data_time_accumulator += time.perf_counter() - data_start

        with torch.autocast(
            device_type=self.strategy.device_type,
            dtype=self.precision_dtype,
            enabled=self.use_amp,
        ):
            out = model(xb, yb)
            logits = out["logits"]
            loss = out["loss"]
            aux_metrics = out["aux_metrics"]

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss detected at step {self.global_step}: {loss}"
            )

        if self.scaler is not None:
            self.strategy.backward(self.scaler.scale(loss))
        else:
            self.strategy.backward(loss)

        if self.check_finite_grads:
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise RuntimeError(
                        f"Non-finite gradient detected at step {self.global_step} for parameter {p.shape}"
                    )

        if self.scaler is not None:
            self.scaler.unscale_(optimizer)

        if self.config.trainer.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.config.trainer.grad_clip
            )

        self.strategy.optimizer_step(optimizer, self.scaler)
        if self.scaler is not None:
            self.scaler.update()

        scheduler.step()

        self.global_step += 1
        if self.progress_bar is not None:
            self.progress_bar.update(1)
        return logits, loss, aux_metrics

    def _maybe_log(
        self,
        model: nn.Module,
        scheduler,
        loss: torch.Tensor,
        aux_metrics: Dict[str, Any],
    ):
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

        # Auxillary metrics to log
        if aux_metrics:
            # MoE metrics
            if "moe" in self.logger_metrics and aux_metrics:
                for block_name, block_data in aux_metrics.items():
                    if not isinstance(block_data, dict):
                        continue
                    moe_stats = block_data.get("moe", None)
                    if moe_stats is None:
                        continue

                    for key, value in moe_stats.items():
                        if value is None:
                            continue

                        metric_prefix = f"{block_name}_{key}"

                        # Normalize
                        if key == "expert_load":
                            value = value.float() / (value.sum() + 1e-9)
                        if key == "expert_importance":
                            value = value / (value.sum() + 1e-9)

                        if torch.is_tensor(value):
                            if value.numel() == 1:
                                metrics[metric_prefix] = value.item()
                            else:
                                for i, v in enumerate(value):
                                    metrics[f"{metric_prefix}_{i}"] = v.item()
                        else:
                            metrics[metric_prefix] = value

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
                if key in self.progress_bar_metrics_names:
                    self.progress_bar_metrics[key] = value
            self.progress_bar.set_postfix(self.progress_bar_metrics)

        self.history.append({"step": self.global_step, **metrics})

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
        """Save checkpoints"""
        if self.save_interval > 0 and (self.global_step) % self.save_interval == 0:
            self.save_checkpoint(model)

    def _validate(self, model: nn.Module, val_dataloader: DataLoader):
        """
        Validation

        Args:
            model (nn.Module): model
            val_dataloader (DataLoader): validation dataloader

        Returns:
            average val loss (int): loss
        """
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
                out = model(xb, yb)
                total_loss += out["loss"].detach() * xb.size(0)
                count += xb.size(0)
            if was_training:
                model.train()
            return (total_loss / count).item() if count > 0 else 0.0

    def _should_stop(self) -> bool:
        return self.max_steps > 0 and self.global_step >= self.max_steps

    def predict(
        self,
        model: nn.Module,
        tokenizer,
        prompts: list[str],
        max_tokens: int = 500,
        use_cache: bool = True,
    ):
        """
        Run inference.

        Args:
            model (nn.Module): model
            tokenizer (_type_): tokenizer
            prompts (list[str]): input prompt
            max_tokens (int, optional): Max tokens to predict. Defaults to 500.
            use_cache (bool, optional): use KV cache. Defaults to True.

        Returns:
            generated tokens
        """
        model = self._prepare_model(model, stage="infer")  # type: ignore
        model.eval()

        self.generator = Generator(
            config=self.config.trainer,
            model=model,
            tokenizer=tokenizer,
            device=self.strategy.device,
        )

        return self.generator.generate(
            prompts=prompts,
            max_tokens=max_tokens,
            use_cache=use_cache,
        )

    def _unwrap_model(self, model):
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        model = self.strategy.unwrap_model(model)
        return model

    def save_checkpoint(self, model: nn.Module):
        model = self._unwrap_model(model)
        if self.strategy.is_main_process():
            save_checkpoint(
                self.config.trainer,
                model,
                self.optimizer,
                self.lr_scheduler,
                self.scaler,
                self.global_step,
            )

    def load_checkpoint(self, model: nn.Module) -> int:
        model = self._unwrap_model(model)
        return load_checkpoint(
            self.config.trainer,
            model,
            self.device,
            self.optimizer,
            self.lr_scheduler,
            self.scaler,
        )

    def _cleanup(self):
        self.logger_manager.finalize()
        if self.progress_bar is not None:
            self.progress_bar.close()
