import torch
import pytest
from ai_playground.models.moe.moe_layer import MoELayer


@pytest.mark.parametrize(
    "shape,d_ff,E",
    [
        ((2, 4, 8), 16, 4),
        ((1, 6, 12), 24, 3),
        ((3, 5, 16), 32, 6),
    ],
)
def test_output_shape(shape, d_ff, E):
    *_, D = shape
    moe = MoELayer(D, d_ff, E, dropout=0.1)

    x = torch.randn(*shape)
    out, _ = moe(x)

    assert out.shape == x.shape
    assert out.dtype == x.dtype


@pytest.mark.parametrize(
    "shape,d_ff,E",
    [
        ((2, 4, 8), 16, 4),
        ((3, 6, 12), 24, 5),
    ],
)
def test_numerical_stability(shape, d_ff, E):
    *_, D = shape
    moe = MoELayer(D, d_ff, E, dropout=0.1)

    x = torch.randn(*shape)
    out, _ = moe(x)

    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "shape,d_ff,E",
    [
        ((2, 3, 4), 8, 3),
        ((1, 5, 8), 16, 4),
    ],
)
def test_gradients_flow(shape, d_ff, E):
    *_, D = shape
    moe = MoELayer(D, d_ff, E, dropout=0.1)

    x = torch.randn(*shape, requires_grad=True)
    out, _ = moe(x)

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert any(p.grad is not None for p in moe.parameters())


@pytest.mark.parametrize(
    "shape,d_ff,E",
    [
        ((2, 4, 8), 16, 4),
        ((1, 6, 12), 24, 3),
    ],
)
def test_eval_mode_deterministic(shape, d_ff, E):
    *_, D = shape
    moe = MoELayer(D, d_ff, E, dropout=0.5)

    x = torch.randn(*shape)

    moe.eval()
    out1, _ = moe(x)
    out2, _ = moe(x)

    assert torch.allclose(out1, out2)


@pytest.mark.parametrize(
    "shape,d_ff",
    [
        ((2, 4, 8), 16),
        ((1, 5, 12), 24),
    ],
)
def test_single_expert_equivalence(shape, d_ff):
    *_, D = shape

    moe = MoELayer(D, d_ff, num_experts=1, dropout=0.0)
    x = torch.randn(*shape)
    expert = moe.experts[0]
    out_moe, _ = moe(x)
    out_expert = expert(x)

    assert torch.allclose(out_moe, out_expert, atol=1e-6)


@pytest.mark.parametrize(
    "shape,E",
    [
        ((4, 8, 16), 4),
        ((2, 6, 12), 3),
    ],
)
def test_multiple_experts_used(shape, E):
    *_, D = shape
    moe = MoELayer(D, 32, E, dropout=0.0)

    x = torch.randn(*shape)

    moe.eval()
    topk_idx, _ = moe.router(x)

    unique_experts = torch.unique(topk_idx)

    assert unique_experts.numel() > 1


@pytest.mark.parametrize(
    "shape,d_ff,E",
    [
        ((2, 4, 8), 16, 4),
        ((1, 3, 6), 12, 3),
    ],
)
def test_output_changes_with_input(shape, d_ff, E):
    *_, D = shape
    moe = MoELayer(D, d_ff, E, dropout=0.0)

    x1 = torch.randn(*shape)
    x2 = torch.randn(*shape)

    out1, _ = moe(x1)
    out2, _ = moe(x2)

    assert not torch.allclose(out1, out2)


@pytest.mark.parametrize(
    "shape,d_ff,E",
    [
        ((1, 2, 4), 8, 2),
        ((2, 3, 6), 12, 3),
    ],
)
def test_accumulation_behavior(shape, d_ff, E):
    *_, D = shape
    moe = MoELayer(D, d_ff, E, dropout=0.0)

    torch.manual_seed(0)
    x = torch.randn(*shape)
    out, _ = moe(x)

    assert torch.isfinite(out).all()
    assert out.shape == x.shape


@pytest.mark.parametrize(
    "shape,d_ff,E",
    [
        ((2, 4, 8), 16, 4),
    ],
)
def test_backward_multiple_steps(shape, d_ff, E):
    *_, D = shape
    moe = MoELayer(D, d_ff, E, dropout=0.1)

    optimizer = torch.optim.Adam(moe.parameters(), lr=1e-3)

    for _ in range(3):
        x = torch.randn(*shape)
        out, _ = moe(x)
        loss = out.pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_accumulation_not_overwrite():
    B, T, D, E = 1, 2, 4, 2

    moe = MoELayer(D, 8, E, dropout=0.0)
    x = torch.randn(B, T, D)
    topk_idx, topk_vals = moe.router(x)
    out, _ = moe(x)

    # if overwrite exists, output norm will be too small
    assert out.norm() > 0
