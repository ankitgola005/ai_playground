from torch.profiler import profile, schedule, ProfilerActivity
from torch.profiler import tensorboard_trace_handler
from ai_playground.configs import ProfilerConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from torch.profiler.profiler import profile as Profiler
    from typing import Literal


def get_profiler(
    profiler_config: ProfilerConfig, device_type: Literal["cpu", "cuda"], log_dir: Path
) -> Profiler:
    """
    Create a PyTorch profiler
    Args:
        profiler_config: ProfilerConfig.
        device_type: Device type ("cpu" or "cuda").

    Returns:
        Configured PyTorch Profiler instance.
    """
    activities = (
        [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        if device_type == "cuda"
        else [ProfilerActivity.CPU]
    )
    profiler = profile(
        activities=activities,
        record_shapes=profiler_config.record_shapes,
        with_stack=profiler_config.with_stack,
        profile_memory=profiler_config.profile_memory,
        schedule=schedule(
            wait=profiler_config.wait,
            warmup=profiler_config.warmup,
            active=profiler_config.active,
            repeat=profiler_config.repeat,
        ),
        on_trace_ready=tensorboard_trace_handler(str(log_dir)),
    )
    return profiler
