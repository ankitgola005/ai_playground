from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn
    from typing import Tuple


def get_grad_norm(model: nn.Module) -> float:
    """
    Compute the total L2 norm of all gradients in the model.

    Args:
        model (nn.Module): PyTorch model.

    Returns:
        float: Total gradient norm.
    """
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm(2).item() ** 2
    return total**0.5


def get_weight_norm(model: nn.Module) -> float:
    """
    Compute the total L2 norm of all model parameters.

    Args:
        model (nn.Module): PyTorch model.

    Returns:
        float: Total weight norm.
    """
    total = 0.0
    for p in model.parameters():
        total += p.norm(2).item() ** 2
    return total**0.5


def get_norm_info(model: nn.Module, lr: float) -> Tuple[float, float, float]:
    """
    Compute gradient norm, weight norm, and update ratio for a model.

    Args:
        model (nn.Module): PyTorch model.
        lr (float): Current learning rate.

    Returns:
        Tuple[float, float, float]: (grad_norm, weight_norm, update_ratio)
    """
    grad_norm = get_grad_norm(model)
    weight_norm = get_weight_norm(model)
    update_ratio = lr * grad_norm / weight_norm if weight_norm > 0 else 0.0

    return grad_norm, weight_norm, update_ratio
