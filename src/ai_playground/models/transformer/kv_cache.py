import torch


class KVCache:
    def __init__(self, B, H, max_len, head_dim, device, dtype):
        self.k = torch.empty(B, H, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.empty(B, H, max_len, head_dim, device=device, dtype=dtype)
        self.idx = 0
        self.max_len = max_len

    def append(self, k, v):
        T = k.size(2)
        t = self.idx

        self.k[:, :, t:t+T] = k
        self.v[:, :, t:t+T] = v
        self.idx += T

    def get_kv(self):
        return self.k[:, :, :self.idx], self.v[:, :, :self.idx]