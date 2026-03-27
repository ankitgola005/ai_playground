import torch


class SparseSelector:
    def select(
        self,
        q: torch.Tensor,  # (B, H, Tq, D)
        k: torch.Tensor,  # (B, H, Tk, D)
        scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns indices: (B, H, Tq, K)
        """
        raise NotImplementedError


class TopKSelector(SparseSelector):
    def __init__(self, topk: int):
        self.topk = topk

    def select(self, q, k, scores=None):
        if scores is None:
            scores = torch.matmul(q, k.transpose(-2, -1))

        _, idx = torch.topk(scores, self.topk, dim=-1)
        return idx


class StrideSelector(SparseSelector):
    def __init__(self, stride: int):
        self.stride = stride

    def select(self, q, k, scores=None):
        Tk = k.size(2)
        idx = torch.arange(0, Tk, self.stride, device=k.device)
        return idx.view(1, 1, 1, -1).expand(q.size(0), q.size(1), q.size(2), -1)
