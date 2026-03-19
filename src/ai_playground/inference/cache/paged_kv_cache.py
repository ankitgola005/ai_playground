import torch
from ai_playground.inference.cache.base_kv_cache import BaseKVCache


class PagedKVCache(BaseKVCache):
    def __init__(self, B, H, head_dim, block_size, device, dtype):
        self.B = B
        self.H = H
        self.block_size = block_size
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # list of blocks
        self.blocks_k = []
        self.blocks_v = []

        # current write state
        self.curr_block_k = None
        self.curr_block_v = None
        self.offset = 0  # position inside current block

        self.total_tokens = 0

        self._alloc_new_block()

    def _alloc_new_block(self):
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

    def append(self, k, v):
        """
        k, v: (B, H, T, D)
        """
        B, H, T, D = k.shape
        t_start = 0

        while t_start < T:
            space = self.block_size - self.offset

            if space == 0:
                self._alloc_new_block()
                space = self.block_size

            t_chunk = min(space, T - t_start)

            self.curr_block_k[:, :, self.offset : self.offset + t_chunk] = k[
                :, :, t_start : t_start + t_chunk
            ]
            self.curr_block_v[:, :, self.offset : self.offset + t_chunk] = v[
                :, :, t_start : t_start + t_chunk
            ]

            self.offset += t_chunk
            self.total_tokens += t_chunk
            t_start += t_chunk

    def get_kv(self):
        """
        Returns:
            k, v: (B, H, total_tokens, D)
        """
        if len(self.blocks_k) == 1:
            # fast path (no concat)
            k = self.blocks_k[0][:, :, : self.offset]
            v = self.blocks_v[0][:, :, : self.offset]
            return k, v

        # concat all full blocks + partial last block
        k_list = []
        v_list = []

        for i in range(len(self.blocks_k)):
            if i == len(self.blocks_k) - 1:
                # last block → only valid portion
                k_list.append(self.blocks_k[i][:, :, : self.offset])
                v_list.append(self.blocks_v[i][:, :, : self.offset])
            else:
                k_list.append(self.blocks_k[i])
                v_list.append(self.blocks_v[i])

        k = torch.cat(k_list, dim=2)
        v = torch.cat(v_list, dim=2)

        return k, v

    def supports_blocks(self) -> bool:
        return True

    def get_blocks(self):
        return self.blocks_k, self.blocks_v, self.offset

    def __len__(self):
        return self.total_tokens
