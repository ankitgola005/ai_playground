import pytest
from ai_playground.tokenizer.base_tokenizer import BaseTokenizer


class DummyTokenizer(BaseTokenizer):
    def __init__(self):
        self._built = True

    def encode(self, text: str):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    @property
    def eos_token_id(self):
        return 0

    @property
    def vocab_size(self):
        return 256

    def save(self, path: str):
        with open(path, "w") as f:
            f.write("dummy")

    @classmethod
    def load(cls, path: str):
        return cls()


def test_cannot_instantiate_base_tokenizer():
    with pytest.raises(TypeError):
        BaseTokenizer()


def test_encode_returns_list_of_ints():
    tokenizer = DummyTokenizer()
    result = tokenizer.encode("abc")

    assert isinstance(result, list)
    assert all(isinstance(i, int) for i in result)


def test_decode_returns_string():
    tokenizer = DummyTokenizer()
    result = tokenizer.decode([97, 98, 99])

    assert isinstance(result, str)
    assert result == "abc"


def test_eos_token_id_type():
    tokenizer = DummyTokenizer()
    assert isinstance(tokenizer.eos_token_id, int)


def test_vocab_size_type():
    tokenizer = DummyTokenizer()
    assert isinstance(tokenizer.vocab_size, int)
    assert tokenizer.vocab_size > 0


def test_save_creates_file(tmp_path):
    tokenizer = DummyTokenizer()
    path = tmp_path / "tok.txt"

    tokenizer.save(path)

    assert path.exists()


def test_load_returns_instance(tmp_path):
    path = tmp_path / "tok.txt"
    path.write_text("dummy")

    tokenizer = DummyTokenizer.load(path)

    assert isinstance(tokenizer, DummyTokenizer)


def test_encode_decode_roundtrip():
    tokenizer = DummyTokenizer()

    text = "hello"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text
