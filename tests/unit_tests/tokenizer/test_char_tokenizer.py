from ai_playground.tokenizer.char_tokenizer import CharTokenizer
from contextlib import nullcontext

import pytest


def test_char_tokenizer_init():
    tokenizer = CharTokenizer()
    assert tokenizer.stoi is None
    assert tokenizer.itos is None


def test_tokenizer_not_built():
    tokenizer = CharTokenizer()

    with pytest.raises(AssertionError):
        tokenizer.encode("abc")

    with pytest.raises(AssertionError):
        tokenizer.decode([1, 2])


def test_char_tokenizer_setup():
    tokenizer = CharTokenizer()
    assert tokenizer.stoi is None
    assert tokenizer.itos is None
    tokenizer.build_from_text("hello")

    eos_token = tokenizer._eos_token
    expected_vocab = set("hello") | {eos_token}

    assert tokenizer.vocab_size == len(expected_vocab)
    assert tokenizer.stoi is not None
    assert tokenizer.itos is not None


def test_eos_token_present():
    tokenizer = CharTokenizer()
    tokenizer.build_from_text("abc")

    assert tokenizer._eos_token is not None
    assert tokenizer.stoi is not None
    assert tokenizer._eos_token in tokenizer.stoi
    assert tokenizer.eos_token_id == tokenizer.stoi[tokenizer._eos_token]


def test_char_tokenizer_encode_decode():
    text = "hello"
    tokenizer = CharTokenizer()
    tokenizer.build_from_text(text)

    eos_token = tokenizer._eos_token
    expected_vocab = set(text) | {eos_token}

    # Vocabulary checks
    assert tokenizer.vocab_size == len(expected_vocab)
    assert tokenizer.stoi is not None
    assert tokenizer.itos is not None
    assert set(tokenizer.stoi.keys()) == expected_vocab
    assert set(tokenizer.itos.values()) == expected_vocab

    encoded = tokenizer.encode(text)
    assert isinstance(encoded, list)
    expected_encoded = [tokenizer.stoi[c] for c in text]
    assert encoded == expected_encoded

    decoded = tokenizer.decode(encoded)
    assert decoded.startswith(text)


def test_char_tokenizer_unknown_character():
    tokenizer = CharTokenizer()
    tokenizer.build_from_text("abc")
    assert tokenizer.stoi is not None

    token = tokenizer.encode("ad")
    assert token == [tokenizer.stoi["a"]]  # 'd' dropped


def test_char_tokenizer_invalid_decode_id_does_not_raise():
    tokenizer = CharTokenizer()
    tokenizer.build_from_text("abc")
    with nullcontext():
        output = tokenizer.decode([10])
    assert output == ""


def test_tokenizer_save_load(tmp_path):
    tokenizer = CharTokenizer()
    tokenizer.build_from_text("abc")

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)

    loaded = CharTokenizer.load(path)

    assert loaded.stoi == tokenizer.stoi
    assert loaded.itos == tokenizer.itos
    assert loaded.eos_token_id == tokenizer.eos_token_id
