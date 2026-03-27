import torch
from ai_playground.inference.cache.base_kv_cache import BaseKVCache

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple, Iterator, List


class PagedKVCache(BaseKVCache):
    """
    Paged (block-wise) KV cache for efficient autoregressive decoding.
    Stores keys and values in fixed-size blocks to avoid large tensor
    reallocations and enable streaming attention.

    Attributes:
        B (int): Batch size
        H (int): Number of KV heads
        head_dim (int): Dimension per head
        block_size (int): Tokens per block
        device (torch.device): Storage device
        dtype (torch.dtype): Tensor dtype

        blocks_k (List[torch.Tensor]): List of key blocks
        blocks_v (List[torch.Tensor]): List of value blocks

        curr_block_k (torch.Tensor): Active key block
        curr_block_v (torch.Tensor): Active value block
        offset (int): Write position within current block

        total_tokens (int): Total tokens stored across all blocks
    """

    def __init__(
        self,
        B: int,
        H: int,
        head_dim: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """
        Initialize Paged KV Cache

        Args:
            B (int): Batch size
            H (int): Number of KV heads
            head_dim (int): Dimension per head
            block_size (int): Tokens per block
            device (torch.device): Storage device
            dtype (torch.dtype): Tensor dtype
        """
        self.B: int = B
        self.H: int = H
        self.head_dim: int = head_dim
        self.block_size: int = block_size
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype

        # list of blocks
        self.blocks_k: List[torch.Tensor] = []
        self.blocks_v: List[torch.Tensor] = []

        # current write state
        self.curr_block_k: torch.Tensor | None = None
        self.curr_block_v: torch.Tensor | None = None
        self.offset: int = 0

        self.total_tokens: int = 0

        self._alloc_new_block()

    def _alloc_new_block(self) -> None:
        """
        Allocate a new KV block and set it as current.
        """
        k_block = torch.empty(
            self.B,
            self.H,
            self.block_size,
            self.head_dim,
            device=self.device,
            dtype=self.dtype,
        )
        v_block = torch.empty_like(k_block)

        self.blocks_k.append(k_block)
        self.blocks_v.append(v_block)

        self.curr_block_k = k_block
        self.curr_block_v = v_block
        self.offset = 0

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Append KV tensors to the cache.

        Args:
            k: Key tensor of shape (B, H, T, D)
            v: Value tensor of shape (B, H, T, D)
        """
        B, H, T, D = k.shape
        t_start = 0

        while t_start < T:
            space = self.block_size - self.offset

            if space == 0:
                self._alloc_new_block()
                space = self.block_size

            t_chunk = min(space, T - t_start)
            assert self.curr_block_k is not None and self.curr_block_v is not None
            self.curr_block_k[:, :, self.offset : self.offset + t_chunk] = k[
                :, :, t_start : t_start + t_chunk
            ]
            self.curr_block_v[:, :, self.offset : self.offset + t_chunk] = v[
                :, :, t_start : t_start + t_chunk
            ]

            self.offset += t_chunk
            self.total_tokens += t_chunk
            t_start += t_chunk

    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return concatenated KV tensors.

        Returns:
            Tuple of:
                k: (B, H, T_total, D)
                v: (B, H, T_total, D)
        """
        if len(self.blocks_k) == 1:
            # fast path: no concat
            k = self.blocks_k[0][:, :, : self.offset]
            v = self.blocks_v[0][:, :, : self.offset]
            return k, v

        # concat all full blocks + partial last block
        k_list: List[torch.Tensor] = []
        v_list: List[torch.Tensor] = []

        for i in range(len(self.blocks_k)):
            if i == len(self.blocks_k) - 1:
                # last block: only valid portion
                k_list.append(self.blocks_k[i][:, :, : self.offset])
                v_list.append(self.blocks_v[i][:, :, : self.offset])
            else:
                k_list.append(self.blocks_k[i])
                v_list.append(self.blocks_v[i])

        k = torch.cat(k_list, dim=2)
        v = torch.cat(v_list, dim=2)

        return k, v

    def iter_kv(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Yield KV blocks in order.

        Yields:
            Tuples of:
                k: (B, H, T_block, D)
                v: (B, H, T_block, D)
        """
        for i in range(len(self.blocks_k)):
            if i == len(self.blocks_k) - 1:
                # last block: only valid tokens
                yield (
                    self.blocks_k[i][:, :, : self.offset],
                    self.blocks_v[i][:, :, : self.offset],
                )
            else:
                yield self.blocks_k[i], self.blocks_v[i]

    def supports_blocks(self) -> bool:
        """Return True: paged cache supports block iteration."""
        return True

    def get_blocks(self) -> Tuple[list[torch.Tensor], list[torch.Tensor], int]:
        """
        Return raw block storage (legacy / debug use).

        Returns:
            blocks_k: list of key blocks
            blocks_v: list of value blocks
            offset: valid tokens in last block
        """
        return self.blocks_k, self.blocks_v, self.offset

    def __len__(self) -> int:
        """Return total number of tokens stored."""
        return self.total_tokens

    def reset(self) -> None:
        """
        Clear the cache and reinitialize with a fresh block.
        """
        self.blocks_k.clear()
        self.blocks_v.clear()
        self.total_tokens = 0
        self._alloc_new_block()

    def gather(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather KV for given token indices.

        Args:
            idx: (B, H, T_q, K) global token indices

        Returns:
            k_sel, v_sel: (B, H, T_q, K, D)
        """
        B, H, T_q, K = idx.shape
        D = self.head_dim

        device = idx.device

        # map to block + offset
        block_id = idx // self.block_size
        offset = idx % self.block_size

        # prepare output
        k_sel = torch.empty(B, H, T_q, K, D, device=device, dtype=self.dtype)
        v_sel = torch.empty_like(k_sel)

        # iterate blocks (clean + safe, optimize later if needed)
        for b_id, (k_block, v_block) in enumerate(self.iter_kv()):
            # k_block: (B, H, T_block, D)

            mask = block_id == b_id  # (B, H, T_q, K)
            if not mask.any():
                continue

            # get offsets for this block
            off = offset[mask]  # (N,)

            # gather values
            k_vals = k_block.reshape(-1, k_block.shape[2], D)
            v_vals = v_block.reshape(-1, v_block.shape[2], D)

            # flatten B,H dims for indexing
            bh_idx = torch.nonzero(mask, as_tuple=False)[:, :2]  # (N, 2)

            b_idx = bh_idx[:, 0]
            h_idx = bh_idx[:, 1]

            # gather per element
            gathered_k = k_block[b_idx, h_idx, off]  # (N, D)
            gathered_v = v_block[b_idx, h_idx, off]  # (N, D)

            # place into output
            k_sel[mask] = gathered_k
            v_sel[mask] = gathered_v

        return k_sel, v_sel
