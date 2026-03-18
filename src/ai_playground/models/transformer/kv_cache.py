import torch


class KVCache:
    def __init__(self, B, H, head_dim, max_len, device, dtype):
        self.k = torch.empty(B, H, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.empty(B, H, max_len, head_dim, device=device, dtype=dtype)
        self.idx = 0  # total tokens stored
        self.max_len = max_len

    def append(self, k, v):
        T = k.size(2)
        t = self.idx

        self.k[:, :, t : t + T] = k
        self.v[:, :, t : t + T] = v
        self.idx += T

    def get_kv(self):
        return self.k[:, :, : self.idx], self.v[:, :, : self.idx]


class PagedKVCache:
    def __init__(self, B, H, head_dim, block_size, device, dtype):
        self.B = B
        self.H = H
        self.D = head_dim
        self.block_size = block_size
        self.device = device
        self.dtype = dtype

        self.k_blocks = []
        self.v_blocks = []

        self.idx = 0

    def _allocate_block(self):
        k_block = torch.empty(
            self.B,
            self.H,
            self.block_size,
            self.D,
            device=self.device,
            dtype=self.dtype,
        )
        v_block = torch.empty_like(k_block)

        self.k_blocks.append(k_block)
        self.v_blocks.append(v_block)

    def append(self, k, v):
        """
        k, v: (B, H, T, D)
        """
        B, H, T, D = k.shape

        for t in range(T):
            block_id = self.idx // self.block_size
            offset = self.idx % self.block_size

            if block_id == len(self.k_blocks):
                self._allocate_block()

            self.k_blocks[block_id][:, :, offset] = k[:, :, t]
            self.v_blocks[block_id][:, :, offset] = v[:, :, t]

            self.idx += 1

    def get_kv(self):
        """
        Returns concatenated KV up to current idx
        """
        if self.idx == 0:
            return None, None

        full_blocks = self.idx // self.block_size
        rem = self.idx % self.block_size

        k_list = []
        v_list = []

        for i in range(full_blocks):
            k_list.append(self.k_blocks[i])
            v_list.append(self.v_blocks[i])

        if rem > 0:
            k_list.append(self.k_blocks[full_blocks][:, :, :rem])
            v_list.append(self.v_blocks[full_blocks][:, :, :rem])

        k = torch.cat(k_list, dim=2)
        v = torch.cat(v_list, dim=2)

        return k, v
