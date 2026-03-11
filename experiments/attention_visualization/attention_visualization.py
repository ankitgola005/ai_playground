from typing import List

import torch
from torch import Tensor, nn
import matplotlib.pyplot as plt

from ai_playground.utils.load_yaml_config import load_yaml_config
from ai_playground.utils.utils import build_data_pipeline, build_model, get_strategy
from ai_playground.trainer.trainer import Trainer
from ai_playground.configs.config import ConfigProtocol

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


def get_attention_maps(model: nn.Module, input_ids: Tensor) -> List[Tensor]:
    """
    Runs a forward pass and collects attention maps from each layer.

    Assumes each block stores the last attention map in:
        block.attention.last_attn
    """
    attn_maps: List[Tensor] = []
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    def hook(module: nn.Module, _input: tuple[Tensor, ...], _output: Tensor) -> None:
        if hasattr(module, "last_attn"):
            attn_maps.append(module.last_attn.detach().cpu())  # type: ignore[attr-defined]

    for block in model.transformer_blocks:  # type: ignore[attr-defined]
        hooks.append(block.attention.register_forward_hook(hook))  # type: ignore[attr-defined]

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    return attn_maps


def plot_attention(attn_maps: List[Tensor], tokens: List[str], save_path: str) -> None:
    """
    Plot all attention heads for all layers in a single figure.
    Rows = layers
    Cols = heads
    """
    plt.rcParams["image.cmap"] = "magma"  # "magma", "viridis"
    num_layers = len(attn_maps)
    num_heads = attn_maps[0].shape[1]
    seq_len = len(tokens)

    fig, axes = plt.subplots(
        num_layers,
        num_heads,
        figsize=(4 * num_heads, 4 * num_layers),
        squeeze=False,
    )

    for layer_idx, attn in enumerate(attn_maps):
        attn = attn[0]  # remove batch

        for head_idx in range(num_heads):
            ax = axes[layer_idx][head_idx]

            ax.imshow(attn[head_idx], interpolation="nearest")

            # titles
            if layer_idx == 0:
                ax.set_title(f"Head {head_idx}", fontsize=10)

            if head_idx == 0:
                ax.set_ylabel(f"Layer {layer_idx}", fontsize=10)

            # remove internal ticks
            ax.set_xticks([])
            ax.set_yticks([])

    # ---- add global token axes ----

    # bottom tokens
    for ax in axes[-1]:
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=90, fontsize=8)

    # left tokens
    for ax in axes[:, 0]:
        ax.set_yticks(range(seq_len))
        ax.set_yticklabels(tokens, fontsize=8)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight",
    )


def main() -> None:
    config: ConfigProtocol = load_yaml_config("gpt_config.yaml")  # type: ignore

    tokenizer, train_loader, val_loader = build_data_pipeline(config)

    layer_sweep = [1, 2, 4, 6, 8, 12, 16, 24, 32, 40]
    layer_sweep = [40]

    sample_text: str = "The quick brown fox jumps over the lazy dog"
    tokens: List[int] = tokenizer.encode(sample_text)
    decoded_tokens: List[str] = [tokenizer.decode([t]) for t in tokens]

    for n_layer in layer_sweep:

        print(f"\nRunning experiment with n_layer = {n_layer}")

        # update config
        config.model.model_kwargs["n_layer"] = n_layer

        model_cls = build_model(config)
        model: nn.Module = model_cls(tokenizer.vocab_size, config)

        model = torch.compile(model)  # type: ignore

        trainer = Trainer(config, strategy=get_strategy(config.distributed))
        trainer.fit(model, train_loader, val_loader)

        model.eval()

        device = next(model.parameters()).device
        input_ids: Tensor = (
            torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        )

        base_model: nn.Module = model._orig_mod  # type: ignore
        attn_maps = get_attention_maps(base_model, input_ids)

        save_path = f"attention_layers_{n_layer}.png"

        plot_attention(attn_maps, decoded_tokens, save_path)

        print(f"Saved attention map → {save_path}")


if __name__ == "__main__":
    main()
