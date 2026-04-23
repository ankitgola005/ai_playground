from abc import ABC, abstractmethod
from typing import List, TypeVar, Type


T = TypeVar("T", bound="BaseTokenizer")


class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizers.

    Defines the interface for converting between text and token IDs,
    along with serialization and vocabulary metadata.
    """

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Convert input text into a sequence of token IDs.

        Args:
            text: Input string.

        Returns:
            List of integer token IDs.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """
        Convert a sequence of token IDs back into text.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded string.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        """
        ID corresponding to the end-of-sequence (EOS) token.

        Returns:
            Integer token ID.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """
        Total number of tokens in the vocabulary.

        Returns:
            Vocabulary size.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Persist tokenizer state to disk.

        Args:
            path: File path to save tokenizer.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls: Type[T], path: str) -> T:
        """
        Load tokenizer state from disk.

        Args:
            path: File path of saved tokenizer.

        Returns:
            An instance of the tokenizer.
        """
        raise NotImplementedError
