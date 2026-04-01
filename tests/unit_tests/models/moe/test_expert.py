import torch
import pytest
from ai_playground.models.moe.expert import Expert


@pytest.mark.parametrize(
    "shape,d_ff",
    [
        ((4, 8, 16), 32),
        ((2, 4, 8), 16),
        ((1, 10, 32), 64),
    ],
)
def test_output_shape(shape, d_ff):
    *dims, D = shape
    expert = Expert(d_model=D, d_ff=d_ff, dropout=0.1)

    x = torch.randn(*shape)
    out = expert(x)

    assert out.shape == x.shape
    assert out.dtype == x.dtype


@pytest.mark.parametrize(
    "N,D,d_ff",
    [
        (32, 16, 64),
        (10, 8, 32),
    ],
)
def test_flattened_input(N, D, d_ff):
    expert = Expert(D, d_ff, dropout=0.1)

    x = torch.randn(N, D)
    out = expert(x)

    assert out.shape == (N, D)


@pytest.mark.parametrize(
    "shape,d_ff",
    [
        ((2, 4, 8), 16),
        ((3, 5, 12), 24),
    ],
)
def test_numerical_stability(shape, d_ff):
    *_, D = shape
    expert = Expert(D, d_ff, dropout=0.1)

    x = torch.randn(*shape)
    out = expert(x)

    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "shape,d_ff",
    [
        ((2, 3, 4), 8),
        ((1, 6, 8), 16),
    ],
)
def test_gradients_flow(shape, d_ff):
    *_, D = shape
    expert = Expert(D, d_ff, dropout=0.1)

    x = torch.randn(*shape, requires_grad=True)
    out = expert(x)

    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert all(p.grad is not None for p in expert.parameters())


@pytest.mark.parametrize(
    "shape,d_ff",
    [
        ((2, 4, 8), 16),
        ((1, 6, 12), 24),
    ],
)
def test_dropout_behavior(shape, d_ff):
    *_, D = shape
    expert = Expert(D, d_ff, dropout=0.5)

    x = torch.randn(*shape)

    expert.train()
    out1 = expert(x)
    out2 = expert(x)

    assert not torch.allclose(out1, out2)

    expert.eval()
    out3 = expert(x)
    out4 = expert(x)

    assert torch.allclose(out3, out4)


@pytest.mark.parametrize(
    "shape,d_ff",
    [
        ((2, 4, 8), 16),
        ((1, 3, 6), 12),
    ],
)
def test_zero_input(shape, d_ff):
    *_, D = shape
    expert = Expert(D, d_ff, dropout=0.0)

    x = torch.zeros(*shape)
    out = expert(x)

    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "D,d_ff",
    [
        (8, 16),
        (16, 32),
    ],
)
def test_parameter_count(D, d_ff):
    expert = Expert(D, d_ff, dropout=0.1)

    param_count = sum(p.numel() for p in expert.parameters())

    expected = (D * d_ff + d_ff) + (d_ff * D + D)

    assert param_count == expected


def test_large_input_stability():
    D, d_ff = 16, 64
    expert = Expert(D, d_ff, dropout=0.0)

    x = torch.randn(2, 4, D) * 1e3
    out = expert(x)

    assert torch.isfinite(out).all()
