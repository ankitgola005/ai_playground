from pathlib import Path
from typing import TYPE_CHECKING

import torch
from ai_playground.data import dataset, CharTokenizer

if TYPE_CHECKING:
    from ai_playground.configs.config import ConfigProtocol, DataConfigProtocol
    from typing import Tuple


DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "datasets"


def get_dataset_path(config: "DataConfigProtocol") -> Path:
    """
    Get path to a dataset based on the dataset name in config.

    Args:
        config (DataConfigProtocol): Data configuration object.
            Must have attribute `dataset`.

    Returns:
        Path: Path to the dataset file.

    Raises:
        NotImplementedError: If the dataset name is not supported.
    """
    if config.dataset == "shakespeare":
        return DATA_DIR / "text_datasets/shakespeare.txt"
    else:
        raise NotImplementedError(
            f"Dataset '{config.dataset}' is currently not supported."
        )


def build_data_pipeline(
    config: "ConfigProtocol",
) -> Tuple[CharTokenizer, "torch.utils.data.DataLoader", "torch.utils.data.DataLoader"]:
    """
    Build the tokenizer and PyTorch data loaders for training and validation.

    Args:
        config (ConfigProtocol): Configuration object with a `data` attribute
            of type DataConfigProtocol.

    Returns:
        Tuple[CharTokenizer, DataLoader, DataLoader]:
            - tokenizer: CharTokenizer instance
            - train_loader: DataLoader for training data
            - val_loader: DataLoader for validation data
    """
    dataset_path = get_dataset_path(config.data)

    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Tokenize
    tokenizer = CharTokenizer(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Split
    train_data, val_data = dataset.train_val_split(encoded, split=config.data.split)

    # Build dataloaders
    train_loader = dataset.build_dataloader(config, train_data)
    val_loader = dataset.build_dataloader(config, val_data)

    return tokenizer, train_loader, val_loader
