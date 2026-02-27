from torch.utils.data import Dataset, DataLoader
import torch
import random
import numpy as np


class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(config, encoded_data):
    dataset = TextDataset(encoded_data, block_size=config.block_size)
    generator = torch.Generator().manual_seed(config.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True,
    )
    return dataloader


def train_val_split(encoded_data, split=0.9):
    split_idx = int(len(encoded_data) * split)
    train_data = encoded_data[:split_idx]
    val_data = encoded_data[split_idx:]
    return train_data, val_data
