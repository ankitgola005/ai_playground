import pytest
from unittest.mock import patch, MagicMock

from ai_playground.utils.strategy import get_strategy
from ai_playground.configs.config import DistributedConfig


@pytest.mark.parametrize(
    "dist_type,world_size,expected_class",
    [
        ("single", 1, "ai_playground.distributed.single.SingleDevice"),
        ("ddp", 2, "ai_playground.distributed.ddp.DDParallel"),
    ],
)
def test_get_strategy_known(dist_type, world_size, expected_class):
    config = DistributedConfig(
        device="cpu", distributed=dist_type, rank=0, world_size=world_size
    )

    with patch(expected_class) as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        strategy = get_strategy(config)
        assert strategy == mock_instance


def test_get_strategy_unknown():
    config = DistributedConfig(device="cpu", distributed="single", rank=0, world_size=1)
    config.distributed = "unknown"
    with pytest.raises(NotImplementedError):
        get_strategy(config)
