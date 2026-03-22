from torch.utils.data import Dataset, DataLoader
import torch
import random
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_playground.configs import DataConfig
    from typing import Tuple


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
    data_config: "DataConfig",
    encoded_data: torch.Tensor,
    batch_size: int,
    seed: int = 42,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Build dataloader.

    Args:
        data_config: Data config
        encoded_data: 1D tensor of token IDs
        batch_size: Batch size
        seed: Random seed
        shuffle: Whether to shuffle data
        drop_last: Whether drop last incomplete batch

    Returns:
        PyTorch DataLoader
    """
    dataset = TextDataset(
        encoded_data,
        block_size=data_config.block_size,
    )

    generator = torch.Generator().manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=data_config.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last,
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
    split_idx: int = int(len(encoded_data) * split)
    train_data = encoded_data[:split_idx]
    val_data = encoded_data[split_idx:]
    return train_data, val_data
