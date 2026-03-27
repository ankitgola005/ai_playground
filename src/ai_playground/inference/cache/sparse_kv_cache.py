import torch
import heapq
from typing import Tuple, Iterator


class SparseKVCache:
    def __init__(
        self, B, H, head_dim, max_len, device, dtype, mode="stride", stride=2, topk=4
    ):
        self.B, self.H, self.D = B, H, head_dim
        self.max_len = max_len
        self.mode = mode
        self.stride = stride
        self.topk = topk

        self.k = torch.empty(B, H, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.empty(B, H, max_len, head_dim, device=device, dtype=dtype)

        self.idx_total = 0
        self.idx_sparse = 0

        # Track metadata
        self.positions = []

        # Use heap for efficient min tracking (top-k)
        if self.mode == "topk":
            self.heap = []  # (score, idx_in_cache)

    def append(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        score: torch.Tensor | None = None,
    ):
        B, H, T, D = k.shape

        for t in range(T):
            if self.mode == "stride":
                if (self.idx_total % self.stride) == 0:
                    if self.idx_sparse >= self.max_len:
                        raise RuntimeError("SparseKVCache overflow")

                    self.k[:, :, self.idx_sparse] = k[:, :, t]
                    self.v[:, :, self.idx_sparse] = v[:, :, t]
                    self.positions.append(self.idx_total)

                    self.idx_sparse += 1

                self.idx_total += 1

            elif self.mode == "topk":
                assert score is not None, "Top-k mode requires score"
                s = float(score[t])

                if self.idx_sparse < self.max_len:
                    # append
                    self.k[:, :, self.idx_sparse] = k[:, :, t]
                    self.v[:, :, self.idx_sparse] = v[:, :, t]
                    self.positions.append(self.idx_total)

                    heapq.heappush(self.heap, (s, self.idx_sparse))
                    self.idx_sparse += 1

                else:
                    # replace smallest
                    min_score, min_idx = self.heap[0]

                    if s > min_score:
                        heapq.heapreplace(self.heap, (s, min_idx))

                        self.k[:, :, min_idx] = k[:, :, t]
                        self.v[:, :, min_idx] = v[:, :, t]
                        self.positions[min_idx] = self.idx_total

                self.idx_total += 1

    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.idx_sparse == 0:
            return torch.empty(0), torch.empty(0)

        # 🔥 FIX: enforce temporal order
        pos = torch.tensor(self.positions, device=self.k.device)
        order = torch.argsort(pos)

        k = self.k[:, :, order]
        v = self.v[:, :, order]

        return k, v

    def iter_kv(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        yield self.get_kv()

    def reset(self):
        self.idx_total = 0
        self.idx_sparse = 0
        self.positions = []

        if self.mode == "topk":
            self.heap = []

    def __len__(self) -> int:
        return self.idx_sparse
