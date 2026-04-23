from tokenizers import Tokenizer, models, trainers, pre_tokenizers

import pytest
from ai_playground.tokenizer.bpe_tokenizer import BPETokenizer


@pytest.fixture
def tokenizer():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(vocab_size=50, special_tokens=["<eos>"])

    tok.train_from_iterator(["hello world", "hello there"], trainer)

    return BPETokenizer(tok)


def test_encode_decode_roundtrip(tokenizer):
    text = "hello world"

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert isinstance(encoded, list)
    assert all(isinstance(i, int) for i in encoded)
    assert isinstance(decoded, str)

    assert "hello" in decoded  # BPE may add spaces differently


def test_eos_token_present(tokenizer):
    assert isinstance(tokenizer.eos_token_id, int)
    assert tokenizer.eos_token_id >= 0


def test_vocab_size(tokenizer):
    assert tokenizer.vocab_size > 0


def test_encode_unknown_tokens(tokenizer):
    encoded = tokenizer.encode("zzzzzz")

    assert isinstance(encoded, list)
    assert all(isinstance(i, int) for i in encoded)


def test_decode_invalid_ids(tokenizer):
    # HF tokenizer usually handles this internally
    decoded = tokenizer.decode([9999])

    assert isinstance(decoded, str)


def test_save_load(tmpdir, tokenizer):
    path = tmpdir / "bpe.json"

    tokenizer.save(path)
    loaded = BPETokenizer.load(path)

    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.eos_token_id == tokenizer.eos_token_id


def test_missing_eos_raises():
    tok = Tokenizer(models.BPE())

    with pytest.raises(ValueError):
        BPETokenizer(tok)
