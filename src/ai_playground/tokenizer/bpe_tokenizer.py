from tokenizers import Tokenizer
from typing import List, Type, TypeVar
from ai_playground.tokenizer.base_tokenizer import BaseTokenizer
from pathlib import Path


T = TypeVar("T", bound="BPETokenizer")


class BPETokenizer(BaseTokenizer):
    """
    Wrapper around HuggingFace `tokenizers.Tokenizer` implementing BaseTokenizer.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        """
        Initialize with a pre-built HuggingFace tokenizer.

        Args:
            tokenizer: A trained `tokenizers.Tokenizer` instance.
        """
        self.tokenizer: Tokenizer = tokenizer

        eos_id = tokenizer.token_to_id("<eos>")
        if eos_id is None:
            raise ValueError("Tokenizer must contain '<eos>' token")

        self._eos_token_id: int = eos_id

    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.

        Args:
            text: Input string.

        Returns:
            List of token IDs.
        """
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded string.
        """
        return self.tokenizer.decode(ids)

    @property
    def eos_token_id(self) -> int:
        """
        ID of the end-of-sequence token.

        Returns:
            EOS token ID.
        """
        return self._eos_token_id

    @property
    def vocab_size(self) -> int:
        """
        Vocabulary size.

        Returns:
            Number of tokens in vocabulary.
        """
        return self.tokenizer.get_vocab_size()

    def save(self, path: str | Path) -> None:
        """
        Save tokenizer to file.

        Args:
            path: File path.
        """
        self.tokenizer.save(str(path))

    @classmethod
    def load(cls: Type[T], path: str | Path) -> T:
        """
        Load tokenizer from file.

        Args:
            path: File path.

        Returns:
            BPETokenizer instance.
        """
        tok = Tokenizer.from_file(str(path))
        return cls(tok)
