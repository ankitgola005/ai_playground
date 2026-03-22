from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_playground.configs.config import DistributedConfig
    from ai_playground.distributed import Parallel


def get_strategy(config: "DistributedConfig") -> "Parallel":
    """
    Factory function to create a distributed training strategy based on configuration.

    Args:
        config (DistributedConfigProtocol): Distributed configuration object.
            Must have `config.distributed` and `config.world_size`.

    Returns:
        Parallel: An instance of a Parallel strategy (SingleDevice or DDParallel).

    Raises:
        NotImplementedError: If the specified distributed strategy is not supported.

    Supported strategies:
        - "single" : SingleDevice (no distributed training)
        - "ddp"    : DistributedDataParallel (DDParallel)
    """
    strategy: Parallel | None = None

    if config.distributed == "single":
        from ai_playground.distributed.single import SingleDevice

        strategy = SingleDevice(config)
    elif config.distributed == "ddp":
        from ai_playground.distributed.ddp import DDParallel

        strategy = DDParallel(config)
    else:
        raise NotImplementedError(
            f"Strategy '{config.distributed}' is currently not supported."
        )

    return strategy
