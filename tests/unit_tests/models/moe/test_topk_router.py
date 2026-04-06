import torch
import pytest
from ai_playground.models.moe.topk_router import TopKRouter


@pytest.mark.parametrize(
    "B,T,D,E,K",
    [
        (4, 8, 16, 6, 2),
        (2, 5, 8, 4, 2),
        (1, 10, 32, 8, 4),
    ],
)
def test_output_shapes(B, T, D, E, K):
    router = TopKRouter(d_model=D, num_experts=E, topk=K)
    x = torch.randn(B, T, D)

    idx, vals, _ = router(x)

    assert idx.shape == (B, T, K)
    assert vals.shape == (B, T, K)
    assert idx.dtype == torch.long
    assert vals.dtype == torch.float32


@pytest.mark.parametrize(
    "B,T,D,E,K",
    [
        (2, 5, 8, 4, 2),
        (3, 7, 16, 6, 3),
    ],
)
def test_indices_range(B, T, D, E, K):
    router = TopKRouter(D, E, topk=K)
    x = torch.randn(B, T, D)

    idx, _, _ = router(x)

    assert torch.all(idx >= 0)
    assert torch.all(idx < E)


@pytest.mark.parametrize(
    "B,T,D,E,K",
    [
        (2, 5, 8, 4, 2),
        (3, 6, 12, 5, 3),
    ],
)
def test_values_range(B, T, D, E, K):
    router = TopKRouter(D, E, topk=K)
    x = torch.randn(B, T, D)

    _, vals, _ = router(x)

    assert torch.all(vals >= 0)
    assert torch.all(vals <= 1)


@pytest.mark.parametrize(
    "B,T,D,E,K",
    [
        (2, 3, 4, 5, 2),
        (1, 4, 6, 7, 3),
    ],
)
def test_topk_correctness(B, T, D, E, K):
    torch.manual_seed(0)

    router = TopKRouter(D, E, topk=K)
    x = torch.randn(B, T, D)
    idx, vals, probs = router(x)

    # Each token should have exactly K selected experts
    assert idx.shape == (B, T, K)
    assert vals.shape == (B, T, K)

    # All indices are valid expert ids
    assert torch.all((idx >= 0) & (idx < E))

    # Softmax probabilities sum <= 1
    prob_sums = vals.sum(dim=-1)
    assert torch.all(prob_sums <= 1.0 + 1e-6)
    load = torch.bincount(idx.view(-1), minlength=E)
    assert load.sum() == B * T * K


@pytest.mark.parametrize(
    "B,T,D,E,K",
    [
        (2, 4, 8, 6, 2),
        (3, 5, 10, 7, 3),
    ],
)
def test_topk_sum_less_than_one(B, T, D, E, K):
    router = TopKRouter(D, E, topk=K)
    x = torch.randn(B, T, D)

    _, vals, _ = router(x)
    sums = vals.sum(dim=-1)

    assert torch.all(sums <= 1.0 + 1e-6)


@pytest.mark.parametrize(
    "B,T,D,E,K",
    [
        (2, 3, 4, 5, 2),
        (1, 2, 8, 4, 1),
    ],
)
def test_gradients_flow(B, T, D, E, K):
    router = TopKRouter(D, E, topk=K)
    x = torch.randn(B, T, D, requires_grad=True)

    _, vals, _ = router(x)

    loss = vals.sum()
    loss.backward()

    assert x.grad is not None
    assert router.linear.weight.grad is not None


@pytest.mark.parametrize(
    "B,T,D,E",
    [
        (2, 3, 4, 5),
        (1, 6, 8, 3),
    ],
)
def test_topk_equals_one(B, T, D, E):
    router = TopKRouter(D, E, topk=1)
    x = torch.randn(B, T, D)

    idx, vals, _ = router(x)

    assert idx.shape[-1] == 1
    assert vals.shape[-1] == 1


@pytest.mark.parametrize(
    "B,T,D,E",
    [
        (2, 3, 4, 5),
        (1, 4, 6, 6),
    ],
)
def test_topk_equals_num_experts(B, T, D, E):
    router = TopKRouter(D, E, topk=E)
    x = torch.randn(B, T, D)

    idx, vals, _ = router(x)

    assert idx.shape[-1] == E

    sums = vals.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_no_duplicate_experts_in_topk():
    B, T, D, E, K = 2, 4, 8, 6, 3

    router = TopKRouter(D, E, topk=K)
    x = torch.randn(B, T, D)
    idx, _, _ = router(x)  # (B, T, K)

    sorted_idx, _ = torch.sort(idx, dim=-1)
    no_duplicates = sorted_idx[..., 1:] != sorted_idx[..., :-1]

    assert torch.all(no_duplicates)
