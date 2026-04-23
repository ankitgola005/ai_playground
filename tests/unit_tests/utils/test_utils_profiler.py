from unittest.mock import patch, MagicMock

import pytest

from ai_playground.utils.profiler import get_profiler
from ai_playground.configs.config import ProfilerConfig


@pytest.mark.parametrize("record_shapes", [True, False])
@pytest.mark.parametrize("with_stack", [True, False])
@pytest.mark.parametrize("profile_memory", [True, False])
def test_get_profiler_parametrized(tmpdir, record_shapes, with_stack, profile_memory):
    config = ProfilerConfig(
        record_shapes=record_shapes,
        with_stack=with_stack,
        profile_memory=profile_memory,
        wait=1,
        warmup=2,
        active=3,
        repeat=0,
    )
    log_dir = tmpdir

    with patch("ai_playground.utils.profiler.profile") as mock_profile:
        mock_profiler = MagicMock()
        mock_profile.return_value = mock_profiler

        profiler = get_profiler(config, "cuda", log_dir)

        mock_profile.assert_called_once()
        _, kwargs = mock_profile.call_args
        assert kwargs["record_shapes"] == record_shapes
        assert kwargs["with_stack"] == with_stack
        assert kwargs["profile_memory"] == profile_memory

        assert profiler == mock_profiler
