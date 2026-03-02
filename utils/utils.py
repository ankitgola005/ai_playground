import random
import torch
import numpy as np

from configs.config import Config
from data import dataset
from data.char_tokenizer import CharTokenizer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_data_pipeline(config: Config):
    with open(config.data.data_path, "r") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_data, val_data = dataset.train_val_split(encoded, split=config.data.split)
    train_loader = dataset.build_dataloader(config, train_data)
    val_loader = dataset.build_dataloader(config, val_data)

    return tokenizer, train_loader, val_loader
