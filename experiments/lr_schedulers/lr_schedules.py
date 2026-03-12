from __future__ import annotations
import copy
import json
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt

import torch
from ai_playground.utils.load_yaml_config import load_yaml_config
from ai_playground.utils.utils import build_data_pipeline, build_model, get_strategy
from ai_playground.trainer.trainer import Trainer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_playground.configs.config import ConfigProtocol
    import torch.nn as nn


RESULT_DIR = Path("experiment_results/lr_schedulers")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def run_lr_sweep(
    base_config: ConfigProtocol, schedules: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Run multiple LR schedules and record validation loss curves.
    Resume if results already exist.

    Args:
        base_config (ConfigProtocol): Base configuration
        schedules (Dict[str, Dict[str, Any]]): Schedulers config

    Returns:
        List[Dict[str, Any]]: List of results
    """
    results: List[Dict[str, Any]] = []

    for schedule_name, cfg in schedules.items():
        save_file = RESULT_DIR / f"{schedule_name}.json"

        # Resume if result already exists
        if save_file.exists():
            print(f"\nLoading cached result for {schedule_name}")
            with open(save_file) as f:
                results.append(json.load(f))
            continue

        print(f"\nRunning LR schedule: {schedule_name}")
        config = copy.deepcopy(base_config)
        config.trainer.lr_config = cfg

        tokenizer, train_loader, val_loader = build_data_pipeline(config)
        model_cls = build_model(config)
        model: nn.Module = model_cls(tokenizer.vocab_size, config)
        model = torch.compile(model)  # type: ignore

        trainer = Trainer(config, strategy=get_strategy(config.distributed))
        trainer.fit(model, train_loader, val_loader)  # type: ignore
        history = trainer.val_loss_history

        # supports both formats
        val_curve = [x["val_loss"] for x in history]
        steps = [x["step"] for x in history]
        result = {
            "schedule": schedule_name,
            "val_loss_curve": val_curve,
            "steps": steps,
            "final_val_loss": val_curve[-1],
        }
        results.append(result)

        # Save result
        with open(save_file, "w") as f:
            json.dump(result, f, indent=2)

    return results


def plot_lr_results(results: List[Dict[str, Any]]) -> None:
    """
    Plot validation loss curves for multiple LR schedules.

    Args:
        [List[Dict]]: List of results
    """
    plt.figure(figsize=(10, 6))

    for r in results:
        steps = r["steps"]
        plt.plot(
            steps,
            r["val_loss_curve"],
            label=f"{r['schedule']} (final: {r['final_val_loss']:.4f})",
        )

    plt.xlabel("Training Steps")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs Step for Different LR Schedules")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(
        "lr_schedulers_vs_val_loss.png",
        dpi=600,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    """Run a scheduler config and plot validation loss trends."""
    base_config: ConfigProtocol = load_yaml_config("gpt_config.yaml")  # type: ignore

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
            "cycle_steps": 2000,
        },
        "exponential_decay": {
            "scheduler": "exponential_decay",
            "lr": base_config.trainer.lr,
            "gamma": 0.995,
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
