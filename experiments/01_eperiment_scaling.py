import copy
import matplotlib.pyplot as plt

from ai_playground.utils.load_yaml_config import load_yaml_config
from ai_playground.utils.utils import build_data_pipeline, build_model, get_strategy
from ai_playground.trainer.trainer import Trainer

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_playground.configs.config import ConfigProtocol


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run_single(config: ConfigProtocol):

    tokenizer, train_loader, val_loader = build_data_pipeline(config)

    model_cls = build_model(config)
    model = model_cls(tokenizer.vocab_size, config)

    trainer = Trainer(config, strategy=get_strategy(config.distributed))

    trainer.fit(model, train_loader, val_loader)

    train_loss = trainer.last_train_loss
    val_loss = trainer.last_val_loss
    params = count_params(model)

    return params, val_loss


# -------------------------
# WIDTH SCALING
# -------------------------


def width_scaling(base_config: ConfigProtocol):

    embed_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

    params_list = []
    loss_list = []

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


# -------------------------
# DEPTH SCALING
# -------------------------


def depth_scaling(base_config: ConfigProtocol):

    layers = [1, 2, 4, 6, 8, 16, 32]

    params_list = []
    loss_list = []

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


# -------------------------
# PLOTTING
# -------------------------


def plot_results(x, loss, title, xlabel):

    plt.figure()

    plt.plot(x, loss, marker="o")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Validation Loss")

    plt.grid(True)

    plt.show()


# -------------------------
# MAIN
# -------------------------


def main():

    base_config = load_yaml_config(
        "/home/kitkat/Desktop/ishtudy/ml_system/ai_playground/configs/gpt_config.yaml"
    )

    # width scaling
    # embeds, params_w, loss_w = width_scaling(base_config)

    # plot_results(
    #     embeds,
    #     loss_w,
    #     "Width Scaling",
    #     "Embedding Size",
    # )

    # depth scaling
    layers, params_d, loss_d = depth_scaling(base_config)  # type: ignore

    plot_results(
        layers,
        loss_d,
        "Depth Scaling",
        "Number of Layers",
    )


if __name__ == "__main__":
    main()
