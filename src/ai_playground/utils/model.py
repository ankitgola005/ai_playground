from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn
    from typing import Type
    from ai_playground.configs.config import ModelConfigProtocol


def build_model(config: "ModelConfigProtocol") -> Type["nn.Module"]:
    """
    Factory function to get a PyTorch model class based on the configuration.

    Args:
        config (ConfigProtocol): Configuration object with model specifications.
            Must have `config.model.model_name` defined.

    Returns:
        Type[nn.Module]: A PyTorch model class (not an instance).

    Raises:
        NotImplementedError: If the specified model name is not supported.

    Supported model names:
        - "minigpt" : MiniGPT transformer model
        - "bigram"  : Bi-gram language model
        - "mnist"   : Feedforward classifier for MNIST
    """
    model: Type["nn.Module"] | None = None

    if config.model_name == "minigpt":
        from ai_playground.models.miniGPT import MiniGPT

        model = MiniGPT
    elif config.model_name == "bigram":
        from ai_playground.models.bigram import BiGram

        model = BiGram
    elif config.model_name == "mnist":
        from ai_playground.models.mnist import MNIST

        model = MNIST
    else:
        raise NotImplementedError(
            f"Model '{config.model_name}' is currently not supported."
        )

    return model
