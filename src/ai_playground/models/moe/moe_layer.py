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
        moe_topk: int = 1,
        dropout: float = 0.0,
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
        self.router = TopKRouter(
            d_model=d_model,
            num_experts=num_experts,
            topk=moe_topk,
            router_noise=0.15,
            temperature=2.5,
        )
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
        device = topk_idx.device

        capacity = int(self.capacity_factor * (N / E))

        idx = topk_idx.view(N, K)
        vals = topk_vals.view(N, K)

        # Flatten
        flat_expert = idx.reshape(-1)  # [N*K]
        flat_vals = vals.reshape(-1)  # [N*K]

        # Sort by expert, then by score descending
        sort_key = flat_expert * 1e6 - flat_vals
        order = torch.argsort(sort_key)

        flat_expert = flat_expert[order]
        flat_vals = flat_vals[order]

        # Identify new expert segments
        is_new = torch.ones_like(flat_expert, dtype=torch.bool)
        is_new[1:] = flat_expert[1:] != flat_expert[:-1]

        # Compute position within each segment
        pos_in_segment = torch.arange(len(flat_expert), device=device)
        segment_start = torch.zeros_like(pos_in_segment)
        segment_start[is_new] = pos_in_segment[is_new]

        segment_start = torch.cummax(segment_start, dim=0).values
        rank = pos_in_segment - segment_start

        # Keep top capacity per expert
        keep_sorted = rank < capacity

        # Undo sort
        inv_order = torch.argsort(order)
        keep_mask = keep_sorted[inv_order].view(N, K)

        # Apply mask
        vals = vals * keep_mask

        # Renormalize safely
        denom = vals.sum(dim=-1, keepdim=True)
        vals = vals / denom.clamp(min=1e-9)
        vals = vals * (denom > 0)

        # Redirect dropped indices
        has_kept = keep_mask.any(dim=1)
        first_kept = keep_mask.float().argmax(dim=1)

        safe_expert = idx[torch.arange(N, device=device), first_kept]

        idx_out = idx.clone()
        # Only redirect for tokens that actually have a kept expert
        redirect_mask = (~keep_mask) & has_kept.unsqueeze(1)
        idx_out[redirect_mask] = safe_expert.unsqueeze(1).expand_as(idx_out)[
            redirect_mask
        ]

        return idx_out.view(B, T, K), vals.view(B, T, K)
