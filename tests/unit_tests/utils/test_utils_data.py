from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import pytest

from ai_playground.utils.data import (
    get_dataset_path,
    build_data_pipeline,
    create_infinite_loader,
)
from ai_playground.configs.config import DataConfig


def test_get_dataset_path_shakespeare():
    path = get_dataset_path("shakespeare")
    assert isinstance(path, Path)
    assert "shakespeare.txt" in str(path)


def test_get_dataset_path_unknown():
    with pytest.raises(NotImplementedError):
        get_dataset_path("unknown")


def test_build_data_pipeline():
    config = DataConfig(dataset="shakespeare", split=0.9, num_workers=0, block_size=10)
    text = "hello world"

    with patch("builtins.open", mock_open(read_data=text)):
        with patch("ai_playground.utils.data.dataset.build_dataloader") as mock_build:
            # Return mock train and validation loaders
            mock_train_dl = MagicMock(name="train_loader")
            mock_val_dl = MagicMock(name="val_loader")
            mock_build.side_effect = [mock_train_dl, mock_val_dl]

            tokenizer, train_loader, val_loader = build_data_pipeline(
                config, batch_size=2
            )

            # Check tokenizer
            expected_vocab = set(text)
            assert tokenizer.vocab_size == len(expected_vocab)
            assert set(tokenizer.stoi.keys()) == expected_vocab
            assert set(tokenizer.itos.values()) == expected_vocab

            # Check loaders
            assert train_loader is mock_train_dl
            assert val_loader is mock_val_dl


def test_create_infinite_loader():
    data = [1, 2, 3]
    infinite_loader = create_infinite_loader(data)

    seen = []
    for _ in range(6):
        val = next(infinite_loader)
        assert val in data
        seen.append(val)

    # Ensure cycling behavior
    assert seen[:3] == data
    assert seen[3:] == data
