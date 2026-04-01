from pathlib import Path
from typing import TYPE_CHECKING

import torch
from ai_playground.data import dataset, CharTokenizer

if TYPE_CHECKING:
    from ai_playground.configs.config import DataConfig
    from typing import Tuple


DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "datasets"


def get_dataset_path(dataset: str) -> Path:
    """
    Get path to a dataset based on the dataset name in config.

    Args:
        dataset (str): Dataset name.

    Returns:
        Path: Path to the dataset file.

    Raises:
        NotImplementedError: If the dataset name is not supported.
    """
    if dataset == "shakespeare":
        return DATA_DIR / "text_datasets/shakespeare.txt"
    else:
        raise NotImplementedError(f"Dataset '{dataset}' is currently not supported.")


def build_data_pipeline(
    data_config: "DataConfig",
    batch_size: int,
    seed: int = 42,
    shuffle: bool = True,
    drop_last: bool = True,
) -> Tuple[CharTokenizer, "torch.utils.data.DataLoader", "torch.utils.data.DataLoader"]:
    """
    Build the tokenizer and PyTorch data loaders for training and validation.

    Args:
        config (Config): Configuration object with a `data` attribute
            of type DataConfigProtocol.

    Returns:
        Tuple[CharTokenizer, DataLoader, DataLoader]:
            - tokenizer: CharTokenizer instance
            - train_loader: DataLoader for training data
            - val_loader: DataLoader for validation data
    """
    dataset_path = get_dataset_path(data_config.dataset)

    cache_dir = DATA_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    encoded_cache_path = cache_dir / f"{data_config.dataset}.pt"
    tokenizer_cache_path = cache_dir / f"{data_config.dataset}_tokenizer.pt"

    dataset_mtime = dataset_path.stat().st_mtime
    if (
        encoded_cache_path.exists()
        and tokenizer_cache_path.exists()
        and encoded_cache_path.stat().st_mtime >= dataset_mtime
        and tokenizer_cache_path.stat().st_mtime >= dataset_mtime
    ):
        encoded = torch.load(encoded_cache_path)
        tokenizer_state = torch.load(tokenizer_cache_path)
        tokenizer = CharTokenizer.from_state(tokenizer_state)
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokenizer = CharTokenizer(text)

        lines = text.split("\n")
        all_tokens = []
        for line in lines:
            tokens = tokenizer.encode(line)
            if len(tokens) > 0:
                all_tokens.extend(tokens + [tokenizer.eos_token_id])
        encoded = torch.tensor(all_tokens, dtype=torch.long)

        torch.save(encoded, encoded_cache_path)
        torch.save(tokenizer.state_dict(), tokenizer_cache_path)

    # Split
    split_idx = int(len(encoded) * data_config.split)
    while encoded[split_idx] != tokenizer.eos_token_id:
        split_idx += 1

    # If no EOS found, fallback
    if split_idx == len(encoded):
        split_idx = int(len(encoded) * data_config.split)

    train_data = encoded[:split_idx]
    val_data = encoded[split_idx:]
    # train_data, val_data = dataset.train_val_split(encoded, split=data_config.split)

    # Build dataloaders
    train_loader = dataset.build_dataloader(
        data_config=data_config,
        encoded_data=train_data,
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    val_loader = dataset.build_dataloader(
        data_config=data_config,
        encoded_data=val_data,
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    return tokenizer, train_loader, val_loader


def create_infinite_loader(dl):
    while True:
        for batch in dl:
            yield batch
