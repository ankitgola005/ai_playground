import torch
import torch.nn as nn
from torch.nn import functional as F


class BiGram(nn.Module):
    """
    A simple Bigram language model.
    Predicts the next token based solely on the previous token using an embedding
    table of shape (vocab_size, vocab_size).
    """

    def __init__(self, vocab_size: int = 32000) -> None:
        """
        Initialize the Bigram model.

        Args:
            vocab_size (int): Size of the vocabulary.
        """
        super().__init__()
        self.vocab_size: int = vocab_size
        # Token embeddings: each token directly predicts next token probabilities
        self.tok_emb: nn.Embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass of the model.

        Args:
            idx (Tensor): Input token indices of shape (B, T).
            targets (Optional[Tensor]): Target token indices of shape (B, T).

        Returns:
            logits (Tensor): Predicted logits of shape (B, T, C).
            loss (Optional[Tensor]): Cross-entropy loss if targets provided, else None.
        """
        logits: torch.Tensor = self.tok_emb(idx)

        if targets is None:
            loss: torch.Tensor | None = None
        else:
            B, T, C = logits.shape
            logits_flat: torch.Tensor = logits.view(B * T, C)
            targets_flat: torch.Tensor = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Generate new tokens from the model autoregressively.

        Args:
            idx (Tensor): Starting token indices of shape (B, T).
            max_new_tokens (int): Number of new tokens to generate.

        Returns:
            Tensor: Generated token indices of shape (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            # Focus on the last timestep
            logits_last: torch.Tensor = logits[:, -1, :]  # (B, C)
            # Convert logits to probabilities
            probs: torch.Tensor = F.softmax(logits_last, dim=-1)  # (B, C)
            # Sample the next token
            idx_next: torch.Tensor = torch.multinomial(
                probs, num_samples=1, generator=generator
            )  # (B, 1)
            # Append to the sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
