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

        # Add EOS token
        self.eos_token = "<EOS>"
        self.eos_token_id = len(self.stoi)

        self.stoi[self.eos_token] = self.eos_token_id
        self.itos[self.eos_token_id] = self.eos_token

        self.vocab_size: int = len(self.stoi)

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
            Decoded string, stripping EOS

        Raises:
            KeyError: If ID not in vocabulary
        """
        return "".join(self.itos[i] for i in ids if i != self.eos_token_id)

    def state_dict(self) -> dict[str, object]:
        return {
            "stoi": self.stoi,
            "itos": self.itos,
            "eos_token_id": self.eos_token_id,
            "eos_token": self.eos_token,
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "CharTokenizer":
        tokenizer = cls.__new__(cls)
        tokenizer.stoi = state["stoi"]
        tokenizer.itos = state["itos"]
        tokenizer.eos_token_id = state["eos_token_id"]
        tokenizer.eos_token = state["eos_token"]
        tokenizer.vocab_size = state["vocab_size"]
        return tokenizer
