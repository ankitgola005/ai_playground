from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from typing import Tuple, Iterator


class BaseKVCache(ABC):
    @abstractmethod
    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append new KV of shape (B, H, T, D)"""
        pass

    @abstractmethod
    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return full KV tensors"""
        pass

    def iter_kv(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield KV chunks of shape (B, H, T_chunk, D)"""
        raise NotImplementedError

    def get_blocks(self):
        """Optional: only for paged cache"""
        raise NotImplementedError

    def supports_blocks(self) -> bool:
        return False

    def __len__(self) -> int:
        """Total tokens stored"""
        raise NotImplementedError
