from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List, Dict


class CharTokenizer:
    """
    Simple character-level tokenizer.
    Builds a vocabulary from a given text corpus and provides
    encode/decode utilities.

    Notes:
        - No special tokens (e.g., <pad>, <unk>) by default
        - Assumes all input text at inference time was seen during training
    """

    def __init__(self, text: str) -> None:
        """
        Initialize tokenizer.

        Args:
            text: Input corpus used to build vocabulary
        """
        chars = sorted(set(text))
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size: int = len(chars)

    def encode(self, text: str) -> List[int]:
        """
        Convert text into token IDs.

        Args:
            text: Input string

        Returns:
            List of token IDs

        Raises:
            KeyError: If character not in vocabulary
        """
        return [self.stoi[c] for c in text]

    def decode(self, ids: List[int]) -> str:
        """
        Convert token IDs back to string.

        Args:
            ids: List of token IDs

        Returns:
            Decoded string

        Raises:
            KeyError: If ID not in vocabulary
        """
        return "".join(self.itos[i] for i in ids)
