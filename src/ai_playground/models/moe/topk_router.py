import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    """
    Top-K router for Mixture-of-Experts (MoE) models.

    Computes routing probabilities over experts for each token
    and selects the top-k experts along with their corresponding probabilities.
    """

    def __init__(self, d_model: int, num_experts: int, topk: int = 1) -> None:
        """
        Initilaizes TopKRouter.

        Args:
            d_model (int): Input embeddings dimension.
            num_experts (int): Total number of experts.
            topk (int, optional): Number of experts to route each token to.
                Defaults to 2.

        Attributes:
            topk (int): Number of selected experts per token.
            linear (nn.Linear): Projection layer mapping input embeddings
                to expert logits of shape (..., num_experts).
        """
        super().__init__()
        self.topk = min(topk, num_experts)
        self.linear = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D),

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - topk_idx (torch.Tensor): Indices of selected experts, shape (B, T, topk)
                - topk_vals (torch.Tensor): Corresponding routing probabilities, shape (B, T, topk)

        Notes:
            - Routing probabilities are computed using softmax over experts.
            - `topk_vals` are not renormalized after top-k selection.
            - No load balancing or auxiliary loss is applied here.
        """
        # x: [B, T, D]
        logits = self.linear(x)  # [B, T, E]
        probs = F.softmax(logits, dim=-1)

        # top-k experts
        topk_vals, topk_idx = torch.topk(probs, k=self.topk, dim=-1)  # [B, T, k]

        return topk_idx, topk_vals
