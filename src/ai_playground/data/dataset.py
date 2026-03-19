from torch.utils.data import Dataset, DataLoader
import torch
import random
import numpy as np

from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ai_playground.configs.config import ConfigProtocol


class TextDataset(Dataset):
    """
    Autoregressive language modeling dataset.

    Produces (x, y) pairs where:
        x = tokens [t, ..., t+block_size-1]
        y = tokens [t+1, ..., t+block_size]
    """

    def __init__(self, data: torch.Tensor, block_size: int) -> None:
        """
        Args:
            data: 1D tensor of token IDs (T,)
            block_size: context length
        """
        self.data: torch.Tensor = data
        self.block_size: int = block_size

    def __len__(self) -> int:
        return self.data.size(0) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: (block_size,)
            y: (block_size,)
        """
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def seed_worker(worker_id: int) -> None:
    """
    Ensure reproducibility across DataLoader workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(
    config: "ConfigProtocol", encoded_data: torch.Tensor
) -> DataLoader:
    """
    Build training dataloader.

    Args:
        config: Global config
        encoded_data: 1D tensor of token IDs

    Returns:
        PyTorch DataLoader
    """
    dataset = TextDataset(
        encoded_data,
        block_size=int(config.model.model_kwargs["block_size"]),
    )

    generator = torch.Generator().manual_seed(config.experimental.seed)

    dataloader = DataLoader(
        dataset,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True,
        pin_memory=True,  # 🔥 useful for GPU training
    )
    return dataloader


def train_val_split(
    encoded_data: torch.Tensor, split: float = 0.9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split dataset into train and validation.

    Args:
        encoded_data: 1D tensor of token IDs
        split: fraction for training

    Returns:
        (train_data, val_data)
    """
    split_idx = int(len(encoded_data) * split)
    train_data = encoded_data[:split_idx]
    val_data = encoded_data[split_idx:]
    return train_data, val_data
