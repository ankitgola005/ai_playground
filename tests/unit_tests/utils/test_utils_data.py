import pytest
import torch
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from ai_playground.utils.data import (
    get_dataset_path,
    build_data_pipeline,
    create_infinite_loader,
)
from ai_playground.configs.config import DataConfig
from ai_playground.data.char_tokenizer import CharTokenizer


# Fixtures
@pytest.fixture
def config():
    return DataConfig(dataset="shakespeare", split=0.9, num_workers=0, block_size=4)


@pytest.fixture
def fake_bytes():
    return b"hello\nworld\nthis\nworks\nwell"


@pytest.fixture
def open_mock(fake_bytes):
    m = mock_open(read_data=fake_bytes)
    m.return_value.read.return_value = fake_bytes
    return m


@pytest.fixture
def mock_dataloaders():
    train = MagicMock(name="train_loader")
    val = MagicMock(name="val_loader")
    return train, val


# get_dataset_path
def test_get_dataset_path_shakespeare():
    path = get_dataset_path("shakespeare")
    assert isinstance(path, Path)
    assert "shakespeare.txt" in str(path)


@pytest.mark.parametrize("dataset", ["unknown", "invalid", ""])
def test_get_dataset_path_invalid(dataset):
    with pytest.raises(NotImplementedError):
        get_dataset_path(dataset)


@pytest.mark.parametrize("cache_exists", [True, False])
def test_build_data_pipeline_cache_modes(
    config, open_mock, mock_dataloaders, cache_exists
):
    train_dl, val_dl = mock_dataloaders
    dataset_text = "hello\nworld\nthis\nis\ntest"
    tokenizer = CharTokenizer(dataset_text)
    tokenizer_state = tokenizer.state_dict()

    encoded = []
    for line in ["hello", "world", "this", "is", "test"]:
        encoded.extend(tokenizer.encode(line))
        encoded.append(tokenizer.eos_token_id)
    encoded = torch.tensor(encoded, dtype=torch.long)

    with patch("builtins.open", open_mock), patch(
        "pathlib.Path.exists", return_value=cache_exists
    ), patch(
        "ai_playground.utils.data.dataset.build_dataloader",
        side_effect=[train_dl, val_dl],
    ) as mock_build, patch(
        "torch.load"
    ) as mock_load, patch(
        "torch.save"
    ) as mock_save:

        if cache_exists:
            mock_load.side_effect = [encoded, tokenizer_state]

        tokenizer, train_loader, val_loader = build_data_pipeline(config, batch_size=2)

    assert train_loader is train_dl
    assert val_loader is val_dl
    assert mock_build.call_count == 2

    if cache_exists:
        assert mock_load.call_count == 2
        assert tokenizer.eos_token_id == tokenizer_state["eos_token_id"]
        mock_save.assert_not_called()
    else:
        mock_save.assert_called()


# dataset too small
@pytest.mark.parametrize(
    "fake_bytes,block_size",
    [
        (b"short", 50),
        (b"a\nb", 10),
    ],
)
def test_dataset_too_small(fake_bytes, block_size):
    config = DataConfig(
        dataset="shakespeare", split=0.9, num_workers=0, block_size=block_size
    )

    m = mock_open(read_data=fake_bytes)
    m.return_value.read.return_value = fake_bytes

    with patch("builtins.open", m), patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(ValueError):
            build_data_pipeline(config, batch_size=2)


# EOS-aware split
def test_split_respects_eos(open_mock, config):
    captured = {}

    def fake_build(data_config, encoded_data, *args, **kwargs):
        if "train" not in captured:
            captured["train"] = encoded_data
        else:
            captured["val"] = encoded_data
        return MagicMock()

    with patch("builtins.open", open_mock), patch(
        "pathlib.Path.exists", return_value=False
    ), patch("torch.save"), patch(
        "ai_playground.utils.data.dataset.build_dataloader", side_effect=fake_build
    ):

        tokenizer, _, _ = build_data_pipeline(config, batch_size=2)

    train = captured["train"]
    val = captured["val"]

    assert train[-1] == tokenizer.eos_token_id or len(val) == 0


# infinite loader
@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
        ["a", "b"],
    ],
)
def test_create_infinite_loader(data):
    loader = create_infinite_loader(data)

    out = [next(loader) for _ in range(len(data) * 2)]

    assert out[: len(data)] == data
    assert out[len(data) :] == data
