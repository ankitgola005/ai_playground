from ai_playground.distributed.base import Parallel
from ai_playground.distributed.single import SingleDevice
from ai_playground.distributed.ddp import DDParallel

__all__ = ["Parallel", "SingleDevice", "DDParallel"]
