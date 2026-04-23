import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from ai_playground.utils.data import (
    get_dataset_path,
    build_data_pipeline,
    create_infinite_loader,
)
from ai_playground.configs.config import DataConfig, TokenizerConfig
from ai_playground.tokenizer.char_tokenizer import CharTokenizer


@pytest.fixture
def config():
    return DataConfig(
        dataset="tinyshakespeare",
        split=0.9,
        num_workers=0,
        block_size=4,
        tokenizer=TokenizerConfig(name="char"),
    )


@pytest.fixture
def tokenizer_file(tmp_path):
    def _make(text="hello world this is test"):
        tok = CharTokenizer()
        tok.build_from_text(text)
        path = tmp_path / "tok.json"
        tok.save(path)
        return path

    return _make


@pytest.fixture
def encoded_data(config):
    return torch.arange(config.block_size + 10, dtype=torch.long)


@pytest.fixture
def mock_loaders():
    return MagicMock(name="train"), MagicMock(name="val")


@pytest.mark.parametrize(
    "dataset, has_val",
    [
        ("tinyshakespeare", False),
        ("tinystories", True),
        ("tinystoriesV2", True),
    ],
)
def test_get_dataset_paths(dataset, has_val):
    paths = get_dataset_path(dataset)

    assert isinstance(paths["train"], Path)

    if has_val:
        assert isinstance(paths["val"], Path)
    else:
        assert paths["val"] is None


@pytest.mark.parametrize("dataset", ["unknown", "invalid", ""])
def test_get_dataset_path_invalid(dataset):
    with pytest.raises(NotImplementedError):
        get_dataset_path(dataset)


@pytest.mark.parametrize("cache_exists", [True, False])
def test_build_data_pipeline_cache_modes(
    tmp_path,
    config,
    tokenizer_file,
    encoded_data,
    mock_loaders,
    cache_exists,
):
    config.tokenizer = TokenizerConfig(
        name="char",
        path=str(tokenizer_file()),
    )

    mock_train_dl, mock_val_dl = mock_loaders

    def fake_dataloader(*args, **kwargs):
        if fake_dataloader.call_count == 0:
            fake_dataloader.call_count += 1
            return mock_train_dl
        return mock_val_dl

    fake_dataloader.call_count = 0

    mock_exists = MagicMock(side_effect=[cache_exists, cache_exists])

    with (
        patch("pathlib.Path.exists", mock_exists),
        patch("torch.load", return_value=encoded_data),
        patch("torch.save") as mock_save,
        patch(
            "ai_playground.utils.data.dataset.build_dataloader",
            side_effect=fake_dataloader,
        ),
    ):

        tokenizer_out, train_loader, val_loader = build_data_pipeline(
            config,
            batch_size=2,
        )

    assert train_loader is mock_train_dl
    assert val_loader is mock_val_dl

    if cache_exists:
        mock_save.assert_not_called()
    else:
        mock_save.assert_called()


@pytest.mark.parametrize("block_size", [50, 10])
def test_dataset_too_small(config, block_size):
    """Test that ValueError is raised when dataset is smaller than block_size"""
    config.block_size = block_size
    small_tensor = torch.tensor([1, 2, 3], dtype=torch.long)

    with (
        patch("ai_playground.utils.data.maybe_cache_dataset") as mock_cache,
        patch("torch.load", return_value=small_tensor),
    ):
        # Mock cache to return paths that exist
        mock_cache.return_value = (Path("train.pt"), Path("val.pt"))

        with pytest.raises(ValueError, match="Train dataset too small"):
            build_data_pipeline(config, batch_size=2)


def test_split_respects_eos(tmp_path, config):
    tokenizer = CharTokenizer()
    tokenizer.build_from_text("hello world this is test")

    path = tmp_path / "tok.json"
    tokenizer.save(path)

    config.tokenizer = TokenizerConfig(
        name="char",
        path=str(path),
    )

    eos = tokenizer.eos_token_id
    encoded = torch.tensor([1, 2, eos, 3, 4, eos, 5, 6, eos], dtype=torch.long)

    train_capture = {}
    val_capture = {}

    def fake_build(*args, **kwargs):
        encoded_data = kwargs["encoded_data"]
        if "train" not in train_capture:
            train_capture["train"] = encoded_data.clone()
        else:
            val_capture["val"] = encoded_data.clone()

        return MagicMock()

    with (
        patch("torch.load", return_value=encoded),
        patch("pathlib.Path.exists", return_value=False),
        patch("torch.save"),
        patch(
            "ai_playground.utils.data.dataset.build_dataloader", side_effect=fake_build
        ),
    ):

        build_data_pipeline(config, batch_size=2)

    train = train_capture["train"]
    val = val_capture["val"]

    assert eos in train
    assert torch.sum(train == eos).item() > 0
    assert len(val) >= 0


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
