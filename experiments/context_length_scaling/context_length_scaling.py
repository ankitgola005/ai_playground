import torch
import copy
import os
import matplotlib.pyplot as plt

from ai_playground.utils import get_config
from ai_playground.utils import build_data_pipeline, build_model, get_strategy
from ai_playground.trainer import Trainer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_playground.configs.config import Config
    import torch.nn as nn
    from typing import List, Dict


TARGET_TOKENS = 25_000_000


def run_context_sweep(base_config: "Config", block_sizes: List[int]) -> List[Dict]:
    results: List[Dict] = []

    batch_size = base_config.trainer.batch_size
    for block_size in block_sizes:
        print(f"\nRunning context length experiment block_size={block_size}")
        config = copy.deepcopy(base_config)
        config.model.model_kwargs["block_size"] = block_size

        # normalize training tokens
        tokens_per_step = batch_size * block_size
        max_steps = TARGET_TOKENS // tokens_per_step
        config.trainer.max_steps = int(max_steps)
        print(f"Tokens per step: {tokens_per_step}")
        print(f"Adjusted max_steps: {max_steps}")

        tokenizer, train_loader, val_loader = build_data_pipeline(
            config.data, config.trainer.batch_size, config.trainer.seed
        )
        model_cls = build_model(config.model)
        model: nn.Module = model_cls(
            config.model, tokenizer.vocab_size, config.data.block_size
        )
        trainer = Trainer(config, strategy=get_strategy(config.distributed))
        try:
            trainer.fit(model, train_loader, val_loader)  # type: ignore
        except RuntimeError as e:
            if "Non-finite gradient detected" in str(e):
                print("Warning: Non-finite gradient detected.")
                for p in model.parameters():  # type: ignore
                    if torch.any(torch.isnan(p)) or torch.any(torch.isinf(p)):
                        p.data = torch.nan_to_num(
                            p.data, nan=0.0, posinf=1e3, neginf=-1e3
                        )

        last_val_loss = [r["val_loss"] for r in trainer.val_loss_history]
        results.append(
            {
                "block_size": block_size,
                "val_loss": last_val_loss[-1],
            }
        )

    return results


def plot_context_results(results: List[Dict], save_dir: str = "plots") -> None:
    """
    Creates and saves plots for context length scaling.
    """

    os.makedirs(save_dir, exist_ok=True)
    results = sorted(results, key=lambda x: x["block_size"])
    block_sizes = [r["block_size"] for r in results]
    val_loss = [r["val_loss"] for r in results]

    plt.figure()
    plt.plot(block_sizes, val_loss, marker="o")
    plt.xlabel("Context Length (block_size)")
    plt.ylabel("Validation Loss")
    plt.title("Context Length Scaling (Linear)")
    plt.grid(True)
    plt.tight_layout()
    linear_path = os.path.join(save_dir, "context_linear.png")
    plt.savefig(
        linear_path,
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    base_config = get_config("minigpt_config.yaml")
    block_sizes = [32, 48, 64, 96, 128, 192, 256, 384, 512]

    results = run_context_sweep(base_config, block_sizes)  # type: ignore
    plot_context_results(results)
