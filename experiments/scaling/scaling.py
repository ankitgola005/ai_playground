import argparse
import copy
import matplotlib.pyplot as plt

from ai_playground.utils.load_yaml_config import load_yaml_config
from ai_playground.utils.utils import build_data_pipeline, build_model, get_strategy
from ai_playground.runner.trainer import Trainer

from typing import TYPE_CHECKING, List, Dict, Tuple

if TYPE_CHECKING:
    from ai_playground.configs.config import ConfigProtocol
    import torch.nn as nn


def count_params(model: nn.Module) -> int:
    """Counts params in a nn.Module
    Args:
        model (nn.Module)

    Returns:
        int: number of parameters in nn.Module
    """
    return sum(p.numel() for p in model.parameters())


def run_single(config: "ConfigProtocol") -> Tuple[int, float]:
    """Single run

    Args:
        config (ConfigProtocol): Config to run

    Returns:
        Tuple containing:
            - Number of params in model
            - Final val_loss
    """
    tokenizer, train_loader, val_loader = build_data_pipeline(config)

    model_cls = build_model(config)
    model = model_cls(tokenizer.vocab_size, config)

    trainer = Trainer(config, strategy=get_strategy(config.distributed))
    trainer.fit(model, train_loader, val_loader)

    result = trainer.val_loss_history
    val_loss = [r["val_loss"] for r in result]
    params = count_params(model)

    return params, val_loss[-1]


def width_scaling(
    base_config: "ConfigProtocol",
) -> Tuple[List[int], List[int], List[float]]:
    """Run a width scaling experiment by sweeping embedding sizes.

    Args:
        base_config (ConfigProtocol): Base configuration.

    Returns:
        Tuple containing:
            - List of embedding sizes
            - List of parameter dicts for each run
            - List of final validation losses for each run
    """
    embed_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    params_list, loss_list = [], []

    for embed in embed_sizes:
        print(f"\nRunning width experiment embed={embed}")
        config = copy.deepcopy(base_config)
        config.model.model_kwargs["n_embed"] = embed
        config.model.model_kwargs["hidden_dim"] = 4 * embed
        config.model.model_kwargs["n_layer"] = 4

        params, loss = run_single(config)
        params_list.append(params)
        loss_list.append(loss)

    return embed_sizes, params_list, loss_list


def depth_scaling(
    base_config: "ConfigProtocol",
) -> Tuple[List[int], List[int], List[float]]:
    """Run a depth scaling experiment by sweeping number of layers sizes.

    Args:
        base_config (ConfigProtocol): Base configuration.

    Returns:
        Tuple containing:
            - List of embedding sizes
            - List of parameter dicts for each run
            - List of final validation losses for each run
    """
    layers = [1, 2, 4, 6, 8, 16, 32]
    params_list, loss_list = [], []

    for layer in layers:
        print(f"\nRunning depth experiment layers={layer}")
        config = copy.deepcopy(base_config)
        config.model.model_kwargs["n_layer"] = layer
        config.model.model_kwargs["n_embed"] = 128
        config.model.model_kwargs["hidden_dim"] = 128 * 4

        params, loss = run_single(config)
        params_list.append(params)
        loss_list.append(loss)

    return layers, params_list, loss_list


def depth_width_scaling(base_config: "ConfigProtocol") -> List[Dict]:
    """Run a depth + width scaling experiment.

    Args:
        base_config (ConfigProtocol): Base configuration.

    Returns:
        Tuple containing:
            - List of embedding sizes
            - List of parameter dicts for each run
            - List of final validation losses for each run
    """
    embed_sizes = [32, 64, 128, 256]
    layers = [2, 4, 8, 16]
    results = []

    for layer in layers:
        for embed in embed_sizes:
            print(f"\nRunning depth×width experiment layers={layer}, embed={embed}")
            config = copy.deepcopy(base_config)
            config.model.model_kwargs["n_layer"] = layer
            config.model.model_kwargs["n_embed"] = embed
            config.model.model_kwargs["hidden_dim"] = 4 * embed

            params, loss = run_single(config)
            results.append(
                {"layers": layer, "embed": embed, "params": params, "val_loss": loss}
            )

    return results


def plot_results(x: List[int], loss: List[float], title: str, xlabel: str) -> None:
    """Plot validation loss against a given parameter sweep.

    Args:
        x (List[int]): List of parameter values (e.g., embedding sizes).
        loss (List[float]): Corresponding validation losses.
        title (str): Plot title.
        xlabel (str): Label for the x-axis.
    """
    plt.figure()
    plt.plot(x, loss, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Validation Loss")
    plt.grid(True)
    plt.show()


def main():
    """Run scaling experiment and plot results."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--law",
        choices=["width", "depth", "depth_width"],
        default="depth",
        help="Which scaling law to run",
    )
    args = parser.parse_args()

    base_config = load_yaml_config("gpt_config.yaml")

    if args.law == "width":
        embeds, params_w, loss_w = width_scaling(base_config)  # type: ignore
        plot_results(embeds, loss_w, "Width Scaling", "Embedding Size")

    elif args.law == "depth":
        layers, params_d, loss_d = depth_scaling(base_config)  # type: ignore
        plot_results(layers, loss_d, "Depth Scaling", "Number of Layers")

    elif args.law == "depth_width":
        results = depth_width_scaling(base_config)  # type: ignore
        # optional: heatmap or scatter plot
        layers_list = [r["layers"] for r in results]
        embed_list = [r["embed"] for r in results]
        loss_list = [r["val_loss"] for r in results]

        plt.figure()
        plt.scatter(embed_list, layers_list, c=loss_list, cmap="viridis", s=200)
        plt.colorbar(label="Validation Loss")
        plt.xlabel("Embedding Size")
        plt.ylabel("Number of Layers")
        plt.title("Depth × Width Scaling")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()
