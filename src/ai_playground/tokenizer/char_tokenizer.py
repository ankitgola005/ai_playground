from ai_playground.tokenizer import BaseTokenizer
from typing import Dict, List, Optional, ClassVar
import json


class CharTokenizer(BaseTokenizer):
    """
    A simple character-level tokenizer.

    Builds a vocabulary from unique characters in input text and assigns
    each character a unique integer ID.
    Unknown characters are ignored during encoding and decoding
    Includes a special end-of-sequence (EOS) token.
    """

    _DEFAULT_EOS_TOKEN: ClassVar[str] = "<eos>"

    def __init__(self) -> None:
        """
        Initialize an unbuilt tokenizer.

        Call `build_from_text` before using encode/decode.
        """
        self.stoi: Optional[Dict[str, int]] = None
        self.itos: Optional[Dict[int, str]] = None
        self._eos_token: str = self._DEFAULT_EOS_TOKEN
        self._eos_token_id: Optional[int] = None

    def build_from_text(self, text: str) -> None:
        """
        Build vocabulary from input text. Should be called before encode / decode.

        Args:
            text: Input corpus used to extract unique characters.

        Side Effects:
            Populates `stoi`, `itos`, and `_eos_token_id`.
        """
        chars = sorted(set(text))

        if self._eos_token not in chars:
            chars.append(self._eos_token)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self._eos_token_id = self.stoi[self._eos_token]

    def encode(self, text: str) -> List[int]:
        """
        Convert text into a list of token IDs.

        Unknown characters are silently ignored.

        Args:
            text: Input string.

        Returns:
            List of token IDs.
        """
        assert self.stoi is not None, "Tokenizer not built"
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        """
        Convert token IDs back to string.

        Invalid IDs are silently ignored.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded string.
        """
        assert self.itos is not None, "Tokenizer not built"
        return "".join(self.itos.get(i, "") for i in ids)

    @property
    def eos_token_id(self) -> int:
        """
        ID corresponding to the EOS token.

        Returns:
            Integer token ID.
        """
        assert self._eos_token_id is not None, "Tokenizer not built"
        return self._eos_token_id

    @property
    def vocab_size(self) -> int:
        """
        Total number of tokens in the vocabulary.

        Returns:
            Vocabulary size.
        """
        assert self.stoi is not None, "Tokenizer not built"
        return len(self.stoi)

    def save(self, path: str) -> None:
        """
        Save tokenizer state to disk as JSON.

        Args:
            path: File path to save tokenizer.
        """
        assert self.stoi is not None, "Tokenizer not built"

        with open(path, "w") as f:
            json.dump(
                {
                    "stoi": self.stoi,
                    "itos": self.itos,
                    "eos_token": self._eos_token,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """
        Load tokenizer state from disk.

        Args:
            path: File path of saved tokenizer.

        Returns:
            Reconstructed CharTokenizer instance.
        """
        with open(path, "r") as f:
            data = json.load(f)

        obj = cls()
        obj.stoi = data["stoi"]
        obj.itos = {int(k): v for k, v in data["itos"].items()}
        obj._eos_token = data["eos_token"]
        obj._eos_token_id = obj.stoi[obj._eos_token]

        return obj
