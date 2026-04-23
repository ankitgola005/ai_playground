import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ai_playground.configs.config import TokenizerConfig
from ai_playground.tokenizer.tokenizer_factory import (
    build_tokenizer,
    resolve_tokenizer_path,
    _TOKENIZER_CACHE,
)
from ai_playground.tokenizer.char_tokenizer import CharTokenizer
from ai_playground.tokenizer.bpe_tokenizer import BPETokenizer


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear tokenizer cache before and after each test"""
    _TOKENIZER_CACHE.clear()
    yield
    _TOKENIZER_CACHE.clear()


def test_build_char_tokenizer_from_file(tmpdir):
    """Test building char tokenizer from saved file"""
    # Create and save a tokenizer
    tok = CharTokenizer()
    tok.build_from_text("hello world test")
    tok_path = tmpdir / "tok.json"
    tok.save(tok_path)

    # Build from config
    config = TokenizerConfig(name="char", filename=None)
    with patch(
        "ai_playground.tokenizer.tokenizer_factory.resolve_tokenizer_path"
    ) as mock_resolve:
        mock_resolve.return_value = str(tok_path)
        with patch(
            "ai_playground.tokenizer.tokenizer_factory.CharTokenizer.load"
        ) as mock_load:
            mock_load.return_value = tok
            result = build_tokenizer(config)
            assert isinstance(result, CharTokenizer)
            mock_load.assert_called_once_with(str(tok_path))


def test_build_char_tokenizer_from_text():
    """Test building char tokenizer from dataset text"""
    config = TokenizerConfig(name="char", filename=None)

    with (
        patch(
            "ai_playground.tokenizer.tokenizer_factory.resolve_tokenizer_path",
            return_value=None,
        ),
        patch("ai_playground.utils.data.get_dataset_path") as mock_get_path,
    ):

        mock_get_path.return_value = {
            "train": Path("/fake/train.txt"),
            "val": Path("/fake/val.txt"),
        }

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.side_effect = [
                "hello ",
                "world",
            ]

            with patch.object(CharTokenizer, "build_from_text") as mock_build:
                result = build_tokenizer(config, dataset="tinyshakespeare")

                mock_build.assert_called_once_with("hello \nworld")
                assert isinstance(result, CharTokenizer)


def test_build_char_tokenizer_requires_dataset():
    """Test that char tokenizer from text requires dataset parameter"""
    config = TokenizerConfig(name="char", filename=None)

    with patch(
        "ai_playground.tokenizer.tokenizer_factory.resolve_tokenizer_path",
        return_value=None,
    ):
        with pytest.raises(ValueError, match="Dataset must be provided"):
            build_tokenizer(config, dataset=None)


def test_build_bpe_tokenizer(tmp_path):
    """Test building BPE tokenizer"""
    bpe_path = tmp_path / "bpe.json"
    bpe_path.write_text("{}", encoding="utf-8")

    config = TokenizerConfig(name="bpe", filename="dummy.json")

    with patch(
        "ai_playground.tokenizer.tokenizer_factory.resolve_tokenizer_path"
    ) as mock_resolve:
        mock_resolve.return_value = str(bpe_path)
        with patch(
            "ai_playground.tokenizer.tokenizer_factory.BPETokenizer.load"
        ) as mock_load:
            mock_tok = MagicMock(spec=BPETokenizer)
            mock_load.return_value = mock_tok

            result = build_tokenizer(config)

            assert result is mock_tok
            mock_load.assert_called_once_with(str(bpe_path))


def test_build_bpe_tokenizer_requires_file():
    """Test that BPE tokenizer requires a file"""
    config = TokenizerConfig(name="bpe", filename=None)

    with patch(
        "ai_playground.tokenizer.tokenizer_factory.resolve_tokenizer_path",
        return_value=None,
    ):
        with pytest.raises(ValueError, match="BPE tokenizer requires a tokenizer file"):
            build_tokenizer(config)


def test_unknown_tokenizer_raises_error():
    """Test that unknown tokenizer name raises error"""
    config = TokenizerConfig(name="unknown_tokenizer", filename=None)

    with patch(
        "ai_playground.tokenizer.tokenizer_factory.resolve_tokenizer_path",
        return_value=None,
    ):
        with pytest.raises(ValueError, match="Unknown tokenizer"):
            build_tokenizer(config)


def test_tokenizer_caching():
    """Test that tokenizers are cached"""
    config = TokenizerConfig(name="char", filename=None)
    cache_key = ("char", None, "test_dataset")

    mock_tok = MagicMock(spec=CharTokenizer)
    _TOKENIZER_CACHE[cache_key] = mock_tok

    result = build_tokenizer(config, dataset="test_dataset")

    assert result is mock_tok


def test_resolve_tokenizer_path_none():
    """Test resolve_tokenizer_path returns None when filename is None"""
    config = TokenizerConfig(name="char", filename=None)
    result = resolve_tokenizer_path(config)
    assert result is None


def test_resolve_tokenizer_path_with_file(tmp_path):
    """Test resolve_tokenizer_path finds file in pretrained directory"""
    # Create a temporary pretrained directory
    tok_file = tmp_path / "test_tok.json"
    tok_file.write_text("{}", encoding="utf-8")

    config = TokenizerConfig(name="char", filename="test_tok.json")

    with patch("ai_playground.tokenizer.tokenizer_factory.TOKENIZER_DIR", tmp_path):
        result = resolve_tokenizer_path(config)
        assert result == str(tok_file)


def test_resolve_tokenizer_path_file_not_found():
    """Test resolve_tokenizer_path raises error when file not found"""
    config = TokenizerConfig(name="char", filename="nonexistent.json")

    with patch(
        "ai_playground.tokenizer.tokenizer_factory.TOKENIZER_DIR", Path("/fake/path")
    ):
        with pytest.raises(FileNotFoundError, match="Tokenizer file not found"):
            resolve_tokenizer_path(config)
