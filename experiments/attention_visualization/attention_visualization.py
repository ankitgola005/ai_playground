from typing import List

import torch
from torch import Tensor, nn
import matplotlib.pyplot as plt

from ai_playground.utils.load_yaml_config import load_yaml_config
from ai_playground.utils import build_data_pipeline, build_model, get_strategy
from ai_playground.trainer import Trainer
from ai_playground.configs.config import Config

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
            attn_maps.append(module.last_attn.detach().cpu())  # type: ignore

    for block in model.transformer_blocks:  # type: ignore
        hooks.append(block.attention.register_forward_hook(hook))  # type: ignore

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    return attn_maps


def plot_attention(
    attn_maps: List[Tensor], tokens: List[str], save_prefix: str
) -> None:
    """
    Plot attention heads for every layer.
    Each layer gets its own figure with heads arranged in a grid.
    """
    plt.rcParams["image.cmap"] = "magma"

    num_heads = attn_maps[0].shape[1]
    seq_len = len(tokens)
    grid_cols = 2
    grid_rows = (num_heads + 1) // 2

    for layer_idx, attn in enumerate(attn_maps):
        attn = attn[0]  # remove batch
        fig, axes = plt.subplots(
            grid_rows,
            grid_cols,
            figsize=(8, 8),
            squeeze=False,
        )
        axes_flat = axes.flatten()
        for head_idx in range(num_heads):
            ax = axes_flat[head_idx]
            ax.imshow(
                attn[head_idx],
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            ax.set_title(f"Head {head_idx}", fontsize=10)
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(tokens, rotation=90, fontsize=7)
            ax.set_yticks(range(seq_len))
            ax.set_yticklabels(tokens, fontsize=7)

        for ax in axes_flat[num_heads:]:
            ax.axis("off")

        plt.suptitle(f"Layer {layer_idx}", fontsize=14)
        plt.tight_layout()
        save_path = f"{save_prefix}_layer_{layer_idx}.png"
        plt.savefig(
            save_path,
            dpi=120,
            bbox_inches="tight",
        )
        plt.close()


def main() -> None:
    config: Config = load_yaml_config("gpt_config.yaml")  # type: ignore

    tokenizer, train_loader, val_loader = build_data_pipeline(config)

    # Depth sweep
    layer_sweep = [4, 12, 24, 40]
    layer_sweep.reverse()

    sample_text: str = "The quick brown fox jumps over the lazy dog"

    tokens: List[int] = tokenizer.encode(sample_text)
    decoded_tokens: List[str] = [tokenizer.decode([t]) for t in tokens]
    config.model.model_kwargs["use_flash_attention"] = False

    for n_layer in layer_sweep:
        print(f"\nRunning experiment with n_layer = {n_layer}")
        config.model.model_kwargs["n_layer"] = n_layer
        model_cls = build_model(config.model)
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
        plot_attention(
            attn_maps,
            decoded_tokens,
            f"attention_layers_{n_layer}",
        )


if __name__ == "__main__":
    main()
