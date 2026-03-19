import torch
from ai_playground.inference.cache.base_kv_cache import BaseKVCache


class KVCache(BaseKVCache):
    def __init__(self, B, H, head_dim, max_len, device, dtype):
        self.k = torch.empty(B, H, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.empty(B, H, max_len, head_dim, device=device, dtype=dtype)
        self.idx = 0  # total tokens stored
        self.max_len = max_len

    def append(self, k, v):
        T = k.size(2)
        t = self.idx
        if t + T > self.max_len:
            raise RuntimeError("KVCache overflow")

        self.k[:, :, t : t + T] = k
        self.v[:, :, t : t + T] = v
        self.idx += T

    def get_kv(self):
        return self.k[:, :, : self.idx], self.v[:, :, : self.idx]

    def iter_kv(self):
        yield self.k[:, :, : self.idx], self.v[:, :, : self.idx]
