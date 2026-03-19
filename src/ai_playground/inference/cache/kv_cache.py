import torch
from typing import TYPE_CHECKING

from ai_playground.inference.cache.base_kv_cache import BaseKVCache

if TYPE_CHECKING:
    from typing import Tuple, Iterator


class KVCache(BaseKVCache):
    """
    KV cache for attention.
    Stores keys and values in preallocated tensors of shape (B, H, max_len, D),
    and appends new tokens along the sequence dimension.

    Attributes:
        k (torch.Tensor): Key cache of shape (B, H, max_len, D)
        v (torch.Tensor): Value cache of shape (B, H, max_len, D)
        idx (int): Number of tokens currently stored
        max_len (int): Maximum capacity of the cache
    """

    def __init__(
        self,
        B: int,
        H: int,
        head_dim: int,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """
        Initialize KV cache.

        Args:
            B: Batch size
            H: Number of KV heads
            head_dim: Dimension per head
            max_len: Maximum sequence length supported
            device: Torch device
            dtype: Tensor dtype
        """
        self.k = torch.empty(B, H, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.empty(B, H, max_len, head_dim, device=device, dtype=dtype)
        self.idx = 0
        self.max_len = max_len

    def append(self, k: torch.Tensor, v: torch.Tensor):
        """
        Append new key/value tensors to the cache.

        Args:
            k: Key tensor of shape (B, H, T, D)
            v: Value tensor of shape (B, H, T, D)

        Raises:
            RuntimeError: If appending exceeds max_len
        """
        T = k.size(2)
        t = self.idx
        if t + T > self.max_len:
            raise RuntimeError("KVCache overflow")

        self.k[:, :, t : t + T] = k
        self.v[:, :, t : t + T] = v
        self.idx += T

    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all cached keys and values.

        Returns:
            Tuple of:
                k: (B, H, T, D)
                v: (B, H, T, D)
        """
        return self.k[:, :, : self.idx], self.v[:, :, : self.idx]

    def iter_kv(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate over KV blocks. For KVCache, this yields a single full block.

        Yields:
            Tuple of:
                k: (B, H, T, D)
                v: (B, H, T, D)
        """
        yield self.k[:, :, : self.idx], self.v[:, :, : self.idx]

    def reset(self) -> None:
        """Clear the cache."""
        self.idx = 0

    def __len__(self) -> int:
        """Return number of tokens stored."""
        return self.idx
