import pytest
from pathlib import Path
from unittest.mock import patch

from ai_playground.utils.paths import resolve_run_name, resolve_dirs, convert_paths
from ai_playground.configs.config import TrainerConfig, LRConfig, ProfilerConfig


def test_resolve_run_name_provided():
    name = "test_run"
    assert resolve_run_name(name) == name


def test_resolve_run_name_empty():
    with patch("ai_playground.utils.paths.datetime") as mock_datetime:
        mock_datetime.now.return_value.strftime.return_value = "run_20260101_120000"
        assert resolve_run_name("") == "run_20260101_120000"


@pytest.mark.parametrize(
    "log_dir,ckpt_dir",
    [
        ("provided", "provided"),
        (None, None),
    ],
)
def test_resolve_dirs(tmpdir, log_dir, ckpt_dir):
    base_dir = Path(tmpdir) / "base"

    log_path = Path(tmpdir) / "log" if log_dir == "provided" else None
    ckpt_path = Path(tmpdir) / "ckpt" if ckpt_dir == "provided" else None

    cfg = TrainerConfig(
        seed=42,
        auto_restart=False,
        batch_size=1,
        max_steps=1,
        val_interval=1,
        lr_config=LRConfig(scheduler="constant", lr=0.1),
        betas=(0.9, 0.95),
        warmup_steps=0,
        weight_decay=0.0,
        grad_clip=1.0,
        precision="fp32",
        use_progress_bar=False,
        logger=[],
        log_interval=1,
        base_dir=base_dir,
        run_name="test",
        use_profiler=False,
        profiler_config=ProfilerConfig(wait=0, warmup=0, active=0, repeat=0),
        ckpt_dir=ckpt_path,
        save_interval=1,
        log_dir=log_path,
    )

    run_dir, resolved_log, resolved_ckpt = resolve_dirs(cfg)

    assert run_dir == base_dir / "test"
    assert resolved_log == (log_path if log_path else run_dir / "logs")
    assert resolved_ckpt == (ckpt_path if ckpt_path else run_dir / "checkpoints")
    assert run_dir.exists()
    assert resolved_log.exists()
    assert resolved_ckpt.exists()


def test_convert_paths(tmpdir):
    data = {
        "path": tmpdir / "test",
        "list": [tmpdir / "a", "string"],
        "nested": {"p": tmpdir / "b"},
    }
    converted: dict = convert_paths(data)

    assert converted["path"] == str(tmpdir / "test")
    assert converted["list"] == [str(tmpdir / "a"), "string"]
    assert converted["nested"]["p"] == str(tmpdir / "b")
