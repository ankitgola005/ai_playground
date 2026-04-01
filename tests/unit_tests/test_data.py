import random

import numpy as np
import pytest
import torch

from ai_playground.configs.config import DataConfig
from ai_playground.data.char_tokenizer import CharTokenizer
from ai_playground.data.dataset import (
    TextDataset,
    build_dataloader,
    seed_worker,
    train_val_split,
)


def test_char_tokenizer_encode_decode():
    text = "hello"
    tokenizer = CharTokenizer(text)

    eos_token = tokenizer.eos_token
    expected_vocab = set(text) | {eos_token}

    # Vocabulary checks
    assert tokenizer.vocab_size == len(expected_vocab)
    assert set(tokenizer.stoi.keys()) == expected_vocab
    assert set(tokenizer.itos.values()) == expected_vocab

    encoded = tokenizer.encode(text)
    assert isinstance(encoded, list)
    expected_encoded = [tokenizer.stoi[c] for c in text]
    assert encoded == expected_encoded

    decoded = tokenizer.decode(encoded)
    assert decoded.startswith(text)


def test_char_tokenizer_unknown_character_raises_keyerror():
    tokenizer = CharTokenizer("abc")
    with pytest.raises(KeyError):
        tokenizer.encode("d")


def test_char_tokenizer_invalid_decode_id_raises_keyerror():
    tokenizer = CharTokenizer("abc")
    with pytest.raises(KeyError):
        tokenizer.decode([10])


def test_text_dataset_getitem_and_len():
    data = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    block_size = 2
    ds = TextDataset(data, block_size)

    assert len(ds) == 3

    x0, y0 = ds[0]
    assert torch.equal(x0, torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(y0, torch.tensor([1, 2], dtype=torch.long))

    x2, y2 = ds[2]
    assert torch.equal(x2, torch.tensor([2, 3], dtype=torch.long))
    assert torch.equal(y2, torch.tensor([3, 4], dtype=torch.long))


def test_train_val_split_boundary_and_values():
    encoded = torch.arange(10, dtype=torch.long)

    train, val = train_val_split(encoded, split=0.7)
    assert len(train) == 7
    assert len(val) == 3
    assert torch.equal(train, torch.arange(7, dtype=torch.long))
    assert torch.equal(val, torch.arange(7, 10, dtype=torch.long))

    train_zero, val_all = train_val_split(encoded, split=0.0)
    assert len(train_zero) == 0
    assert torch.equal(val_all, encoded)

    train_all, val_zero = train_val_split(encoded, split=1.0)
    assert torch.equal(train_all, encoded)
    assert len(val_zero) == 0


def test_build_dataloader_iteration_no_shuffle():
    encoded = torch.arange(8, dtype=torch.long)
    data_config = DataConfig(dataset="test", split=0.9, num_workers=0, block_size=3)

    dl = build_dataloader(
        data_config=data_config,
        encoded_data=encoded,
        batch_size=2,
        seed=123,
        shuffle=False,
        drop_last=False,
    )

    all_x = []
    all_y = []
    for x, y in dl:
        all_x.append(x)
        all_y.append(y)

    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)

    expected_dataset = TextDataset(encoded, block_size=3)
    expected_x = torch.stack(
        [expected_dataset[i][0] for i in range(len(expected_dataset))]
    )
    expected_y = torch.stack(
        [expected_dataset[i][1] for i in range(len(expected_dataset))]
    )

    assert torch.equal(all_x, expected_x)
    assert torch.equal(all_y, expected_y)


def test_seed_worker_reproducible():
    torch.manual_seed(1)
    seed_worker(0)
    a = np.random.randint(0, 1_000_000)
    b = random.randint(0, 1_000_000)

    torch.manual_seed(1)
    seed_worker(0)
    c = np.random.randint(0, 1_000_000)
    d = random.randint(0, 1_000_000)

    assert a == c
    assert b == d


def test_text_dataset_block_size_ge_data_length():
    # block_size >= data length
    data = torch.arange(2, dtype=torch.long)
    ds = TextDataset(data, block_size=2)
    assert len(ds) == 0

    # empty data
    data_empty = torch.tensor([], dtype=torch.long)
    ds_empty = TextDataset(data_empty, block_size=2)
    assert len(ds_empty) == 0
