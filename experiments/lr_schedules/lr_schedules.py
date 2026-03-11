from __future__ import annotations
import copy
from typing import Any, Dict, List
import matplotlib.pyplot as plt

import torch
from ai_playground.utils.load_yaml_config import load_yaml_config
from ai_playground.utils.utils import build_data_pipeline, build_model, get_strategy
from ai_playground.trainer.trainer import Trainer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_playground.configs.config import ConfigProtocol


def run_lr_sweep(
    base_config: ConfigProtocol, schedules: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Run multiple LR schedules and record validation loss curves.
    """
    results: List[Dict[str, Any]] = []

    for schedule_name, cfg in schedules.items():
        print(f"\nRunning LR schedule: {schedule_name}")
        config = copy.deepcopy(base_config)
        config.trainer.lr_config = cfg

        tokenizer, train_loader, val_loader = build_data_pipeline(config)
        model_cls = build_model(config)
        model = model_cls(tokenizer.vocab_size, config)
        model = torch.compile(model)

        trainer = Trainer(config, strategy=get_strategy(config.distributed))
        trainer.fit(model, train_loader, val_loader)  # type: ignore

        val_curve: List[float] = trainer.val_loss_history
        results.append(
            {
                "schedule": schedule_name,
                "val_loss_curve": val_curve,
                "final_val_loss": val_curve[-1],
            }
        )

    return results


def plot_lr_results(results: List[Dict[str, Any]]) -> None:
    """
    Plot validation loss curves for multiple LR schedules.
    """
    plt.figure(figsize=(10, 6))

    for r in results:
        steps = list(range(1, len(r["val_loss_curve"]) + 1))
        plt.plot(
            steps,
            r["val_loss_curve"],
            label=f"{r['schedule']} (final: {r['final_val_loss']:.4f})",
        )

    plt.xlabel("Validation Step")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs Step for Different LR Schedules")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    base_config: ConfigProtocol = load_yaml_config("gpt_config.yaml")  # type: ignore

    steps = base_config.trainer.max_steps
    schedules = {
        "constant": {"scheduler": "constant", "lr": base_config.trainer.lr},
        "one_cycle": {
            "scheduler": "one_cycle",
            "lr": base_config.trainer.lr,
            "one_cycle_pct": 0.3,
        },
        "cosine": {
            "scheduler": "cosine",
            "lr": base_config.trainer.lr,
            "min_lr_ratio": base_config.trainer.lr_config["min_lr_ratio"],
        },
        "cosine_restart": {
            "scheduler": "cosine_restart",
            "lr": base_config.trainer.lr,
            "cycle_steps": 200,
        },
        "exponential_decay": {
            "scheduler": "exponential_decay",
            "lr": base_config.trainer.lr,
            "gamma": 0.95,
        },
        "polynomial_decay": {
            "scheduler": "polynomial_decay",
            "lr": base_config.trainer.lr,
            "power": 2,
        },
        "linear_decay": {"scheduler": "linear_decay", "lr": base_config.trainer.lr},
    }

    results = run_lr_sweep(base_config, schedules)
    plot_lr_results(results)
