import torch
import copy
import matplotlib.pyplot as plt
from ai_playground.utils.load_yaml_config import load_yaml_config
from ai_playground.utils.utils import build_data_pipeline, build_model, get_strategy
from ai_playground.trainer.trainer import Trainer

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ai_playground.configs.config import ConfigProtocol


def run_context_sweep(base_config: ConfigProtocol, block_sizes: List[int]):
    results = []

    for block_size in block_sizes:
        print(f"\nRunning context length experiment block_size={block_size}")
        config = copy.deepcopy(base_config)
        config.model.model_kwargs["block_size"] = block_size

        tokenizer, train_loader, val_loader = build_data_pipeline(config)
        model_cls = build_model(config)
        model = model_cls(tokenizer.vocab_size, config)
        model = torch.compile(model)
        trainer = Trainer(config, strategy=get_strategy(config.distributed))
        trainer.fit(model, train_loader, val_loader)  # type: ignore

        results.append({"block_size": block_size, "val_loss": trainer.last_val_loss})

    return results


def plot_context_results(results: List[dict], save_dir: str = "plots"):
    """
    Creates and saves linear and log2 x-axis plots for context length scaling.
    `results` is a list of dicts: [{"block_size": int, "val_loss": float}, ...]
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    block_sizes = [r["block_size"] for r in results]
    val_loss = [r["val_loss"] for r in results]

    # ----- Linear plot -----
    plt.figure()
    plt.plot(block_sizes, val_loss, marker="o", color="blue")
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
    print(f"Saved linear plot: {linear_path}")

    # ----- Log2 plot -----
    plt.figure()
    plt.plot(block_sizes, val_loss, marker="o", color="green")
    plt.xscale("log", base=2)
    plt.xlabel("Context Length (block_size, log2)")
    plt.ylabel("Validation Loss")
    plt.title("Context Length Scaling (Log2 x-axis)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    log_path = os.path.join(save_dir, "context_log2.png")
    plt.savefig(
        log_path,
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved log2 plot: {log_path}")


if __name__ == "__main__":
    base_config = load_yaml_config("gpt_config.yaml")
    block_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    block_sizes.reverse()

    results = run_context_sweep(base_config, block_sizes)  # type: ignore
    plot_context_results(results)
