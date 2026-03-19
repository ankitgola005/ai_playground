from abc import ABC, abstractmethod
from typing import Tuple
import torch


class BaseKVCache(ABC):
    @abstractmethod
    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Append new KV of shape (B, H, T, D)"""
        pass

    @abstractmethod
    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return full KV tensors"""
        pass

    def get_blocks(self):
        """Optional: only for paged cache"""
        raise NotImplementedError

    def supports_blocks(self) -> bool:
        return False

    def __len__(self) -> int:
        """Total tokens stored"""
        raise NotImplementedError
