import torch
import pytest
import torch.nn as nn

from ai_playground.utils.norms import get_grad_norm, get_weight_norm, get_norm_info


def test_get_grad_norm_positive():
    model = nn.Linear(2, 1)
    # Set gradients manually
    model.weight.grad = torch.tensor([[1.0, 2.0]])
    model.bias.grad = torch.tensor([0.5])

    norm = get_grad_norm(model)
    expected = (1**2 + 2**2 + 0.5**2) ** 0.5
    assert torch.isclose(torch.tensor(norm), torch.tensor(expected))


def test_get_grad_norm_zero():
    model = nn.Linear(2, 1)
    # No gradients
    norm = get_grad_norm(model)
    assert norm == 0.0


def test_get_weight_norm_positive():
    model = nn.Linear(2, 1)
    model.weight.data = torch.tensor([[1.0, 2.0]])
    model.bias.data = torch.tensor([0.5])

    norm = get_weight_norm(model)
    expected = (1**2 + 2**2 + 0.5**2) ** 0.5
    assert torch.isclose(torch.tensor(norm), torch.tensor(expected))


def test_get_weight_norm_zero():
    model = nn.Linear(2, 1)
    # All weights zero
    model.weight.data.zero_()
    model.bias.data.zero_()

    norm = get_weight_norm(model)
    assert norm == 0.0


def test_get_norm_info_positive():
    model = nn.Linear(2, 1)
    model.weight.data = torch.tensor([[1.0, 2.0]])
    model.bias.data = torch.tensor([0.5])
    model.weight.grad = torch.tensor([[1.0, 2.0]])
    model.bias.grad = torch.tensor([0.5])

    lr = 0.01
    grad_norm, weight_norm, update_ratio = get_norm_info(model, lr)

    expected_grad = (1**2 + 2**2 + 0.5**2) ** 0.5
    expected_weight = (1**2 + 2**2 + 0.5**2) ** 0.5
    expected_ratio = (lr * expected_grad) / expected_weight

    assert torch.isclose(torch.tensor(grad_norm), torch.tensor(expected_grad))
    assert torch.isclose(torch.tensor(weight_norm), torch.tensor(expected_weight))
    assert torch.isclose(torch.tensor(update_ratio), torch.tensor(expected_ratio))


def test_get_norm_info_zero_weight():
    model = nn.Linear(2, 1)
    model.weight.data.zero_()
    model.bias.data.zero_()
    model.weight.grad = torch.tensor([[1.0, 2.0]])
    model.bias.grad = torch.tensor([0.5])

    lr = 0.01
    grad_norm, weight_norm, update_ratio = get_norm_info(model, lr)

    assert weight_norm == 0.0
    assert grad_norm > 0.0
    # update_ratio should handle zero weight gracefully
    assert update_ratio == 0.0


@pytest.mark.parametrize("layer", [nn.Linear(2, 1), nn.Conv2d(1, 1, 1)])
def test_grad_and_weight_norm_param(layer):
    model = layer
    # Make weights and gradients positive
    model.weight.grad = torch.ones_like(model.weight)
    if model.bias is not None:
        model.bias.grad = torch.ones_like(model.bias)
        model.bias.data = torch.ones_like(model.bias)
    model.weight.data = torch.ones_like(model.weight)

    grad_norm = get_grad_norm(model)
    weight_norm = get_weight_norm(model)

    assert grad_norm > 0
    assert weight_norm > 0
