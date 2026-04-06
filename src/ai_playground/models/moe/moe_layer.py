import torch
from torch import nn
from ai_playground.models.moe.expert import Expert
from ai_playground.models.moe.topk_router import TopKRouter

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, Tuple


class MoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts (MoE) layer with Top-K routing.

    This layer routes each token to a subset of experts (Top-K) using a learned
    router, applies the selected experts independently, and combines their
    outputs using routing weights.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        dropout: float,
        capacity_factor: float = 1.2,
        logging: bool = False,
    ):
        """
        Initialize MoELayer.

        Args:
            d_model (int): Input and output dimensionality.
            d_ff (int): Hidden dimension of each expert (FFN expansion).
            num_experts (int): Total number of experts.
            dropout (float): Dropout probability used inside each expert.

        Attributes:
            num_experts (int): Number of experts.
            router (TopKRouter): Module that computes routing probabilities and
                selects top-k experts per token.
            experts (nn.ModuleList): List of expert networks.
        """
        super().__init__()
        self.num_experts = num_experts
        self.router = TopKRouter(d_model, num_experts)
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff, dropout) for _ in range(num_experts)]
        )
        self.capacity_factor = capacity_factor
        self.logging = logging

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict | None]:
        """
        Forward pass of the MoE layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D),

        Returns:
            torch.Tensor: Output tensor of shape (B, T, D).

        Workflow:
            1. Flatten tokens → (N, D), where N = B * T
            2. Route tokens using Top-K router:
               - topk_idx: (N, K) expert indices
               - topk_vals: (N, K) routing probabilities
            3. For each expert:
               - Select assigned tokens
               - Apply expert FFN
               - Weight outputs using routing probabilities
            4. Accumulate outputs and reshape back to (B, T, D)
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # [N, D], N = B*T

        # routing
        topk_idx, topk_vals, probs = self.router(x)  # [B, T, K]
        assert topk_idx.shape == topk_vals.shape
        assert topk_idx.dim() == 3  # (B, T, K)
        if self.training:
            topk_idx, topk_vals = self.apply_capacity(topk_idx, topk_vals)

        K = topk_idx.shape[-1]
        topk_idx = topk_idx.view(-1, K)  # (N, K)
        topk_vals = topk_vals.view(-1, K)  # (N, K)
        output = torch.zeros_like(x_flat)

        # dispatch per expert
        for expert_id in range(self.num_experts):
            # mask tokens routed to this expert
            mask = topk_idx == expert_id  # [N, K]
            if not mask.any():
                continue

            # get token indices
            token_idx, topk_pos = torch.where(mask)
            selected_x = x_flat[token_idx]  # [num_tokens, D]
            expert_out = self.experts[expert_id](selected_x)

            # weight outputs
            weights = topk_vals[token_idx, topk_pos].unsqueeze(-1)
            output.index_add_(0, token_idx, expert_out * weights)

        covered = torch.zeros(x_flat.size(0), dtype=torch.bool, device=x.device)
        for expert_id in range(self.num_experts):
            mask = topk_idx == expert_id
            token_idx, _ = torch.where(mask)
            covered[token_idx] = True
        if not self.training:
            assert covered.all(), "Some tokens were never routed!"

        loss_lb = None
        load = None
        if self.training:
            # importance: soft distribution
            importance = probs.mean(dim=(0, 1))  # (num_experts,)

            # load: hard routing counts
            load = torch.zeros(self.num_experts, device=x.device)
            load = torch.bincount(topk_idx.view(-1), minlength=self.num_experts).float()

            # normalize
            importance = importance / importance.sum()
            load = load / load.sum()

            # load balancing loss
            loss_lb = (importance * load).sum() * (self.num_experts**2)

        moe_stats: Dict | None = None

        if self.logging and self.training:
            expert_load = load
            expert_importance = torch.zeros(self.num_experts, device=x.device)
            expert_importance.scatter_add_(0, topk_idx.view(-1), topk_vals.view(-1))
            moe_stats = {
                "expert_load": expert_load,
                "expert_importance": expert_importance,
            }
            if loss_lb is not None:
                moe_stats["load_balance_loss"] = loss_lb.detach()

        return output.view(B, T, D), moe_stats

    def apply_capacity(self, topk_idx, topk_vals):
        B, T, K = topk_idx.shape
        N = B * T
        E = self.num_experts
        capacity = int(self.capacity_factor * (N / E))

        topk_idx_flat = topk_idx.view(N, K).clone()
        topk_vals_flat = topk_vals.view(N, K).clone()

        expert_count = torch.zeros(E, dtype=torch.long, device=topk_idx.device)
        keep = torch.zeros(N, K, dtype=torch.bool, device=topk_idx.device)

        for n in range(N):
            seen = set()
            for k in range(K):
                e = topk_idx_flat[n, k].item()
                if e in seen:
                    continue
                seen.add(e)
                if expert_count[e] < capacity:
                    keep[n, k] = True
                    expert_count[e] += 1

        # Zero weights for dropped slots
        topk_vals_flat[~keep] = 0.0

        # Redirect dropped slot *indices* to a kept expert for this token (no-op for bool mask)
        # For tokens with at least one kept slot: redirect to that expert
        # For fully-dropped tokens: leave idx[n,0] as-is (all weights are 0 anyway)
        has_kept = keep.any(dim=1)  # [N]
        first_kept_k = keep.long().argmax(dim=1)  # [N]
        safe_expert = topk_idx_flat[
            torch.arange(N, device=topk_idx.device), first_kept_k
        ]

        topk_idx_out = topk_idx_flat.clone()
        for n in range(N):
            if has_kept[n]:
                topk_idx_out[n][~keep[n]] = safe_expert[n]
            # else: fully dropped token — keep original indices, weights are all 0

        # Renormalize ONLY tokens that have at least one kept slot
        denom = topk_vals_flat.sum(dim=-1, keepdim=True)
        active = has_kept & (denom.squeeze(-1) > 0)
        topk_vals_flat[active] = topk_vals_flat[active] / denom[active]
        # Fully-dropped tokens keep all-zero weights (they contribute nothing to output)

        return topk_idx_out.view(B, T, K), topk_vals_flat.view(B, T, K)
