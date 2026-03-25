import torch
from typing import TYPE_CHECKING
from ai_playground.inference.cache.base_kv_cache import BaseKVCache

if TYPE_CHECKING:
    from typing import Tuple, Iterator


class SparseKVCache(BaseKVCache):
    """
    Sparse KV Cache: stores only a subset of past keys/values.
    Supports:
      - stride-based sparsity: keep every `stride`-th token
      - top-k sparsity: keep only the top-k tokens based on a score

    Args:
        B: batch size
        H: number of KV heads
        head_dim: dimension per head
        max_len: max number of KV tokens to store
        device, dtype
        mode: "stride" or "topk"
        stride: for stride mode
        topk: for top-k mode
    """

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

        # counters
        self.idx_total = 0  # total tokens seen
        self.idx_sparse = 0  # number of stored tokens

        # track positions and scores for top-k
        self.positions = []
        if self.mode == "topk":
            self.scores = []

    def append(self, k: torch.Tensor, v: torch.Tensor, score: torch.Tensor = None):
        """
        Append new KV tensors sparsely.
        For stride mode, `score` is ignored.
        For top-k mode, `score` must be provided with shape (T,)
        """
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
                assert score is not None, "Top-k mode requires a score tensor"
                s = score[t].item()
                if self.idx_sparse < self.max_len:
                    # cache not full, append
                    self.k[:, :, self.idx_sparse] = k[:, :, t]
                    self.v[:, :, self.idx_sparse] = v[:, :, t]
                    self.positions.append(self.idx_total)
                    self.scores.append(s)
                    self.idx_sparse += 1
                else:
                    # cache full: replace min score if current > min
                    min_idx = self.scores.index(min(self.scores))
                    if s > self.scores[min_idx]:
                        self.k[:, :, min_idx] = k[:, :, t]
                        self.v[:, :, min_idx] = v[:, :, t]
                        self.scores[min_idx] = s
                        self.positions[min_idx] = self.idx_total
                self.idx_total += 1

    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.idx_sparse == 0:
            return torch.empty(0), torch.empty(0)
        return self.k[:, :, : self.idx_sparse], self.v[:, :, : self.idx_sparse]

    def iter_kv(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        yield self.get_kv()

    def reset(self):
        self.idx_total = 0
        self.idx_sparse = 0
        self.positions = []
        if self.mode == "topk":
            self.scores = []

    def __len__(self) -> int:
        return self.idx_sparse
