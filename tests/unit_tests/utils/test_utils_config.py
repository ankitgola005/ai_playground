from unittest.mock import patch, mock_open
from pathlib import Path
import yaml

from ai_playground.utils.config import (
    load_config,
    config_to_dict,
    update_config,
    preprocess_config,
)
from ai_playground.configs.config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainerConfig,
    LRConfig,
    ProfilerConfig,
    DistributedConfig,
)
from ai_playground.utils.paths import convert_paths


def make_minimal_config(tmpdir) -> Config:
    base_dir = Path(tmpdir) / "test_run"
    cfg = Config(
        data=DataConfig(dataset="test", split=0.9, num_workers=0, block_size=10),
        model=ModelConfig(model_name="bigram", compile=False),
        trainer=TrainerConfig(
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
            profiler_config=ProfilerConfig(
                record_shapes=False,
                with_stack=True,
                profile_memory=True,
                wait=0,
                warmup=0,
                active=0,
                repeat=0,
            ),
            ckpt_dir=base_dir / "ckpt",
            save_interval=1,
            log_dir=base_dir / "logs",
        ),
        distributed=DistributedConfig(
            device="cpu", distributed="single", rank=0, world_size=1
        ),
    )
    return cfg


def test_load_config_yaml(monkeypatch, tmpdir):
    cfg = make_minimal_config(tmpdir)
    cfg_dict = convert_paths(config_to_dict(cfg))
    yaml_content = yaml.safe_dump(cfg_dict)
    monkeypatch.setattr(
        "ai_playground.utils.config.CONFIG_BASE_PATH", Path(tmpdir) / "configs"
    )

    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            loaded_cfg = load_config("dummy.yaml")
            assert isinstance(loaded_cfg, Config)
            assert loaded_cfg.data.dataset == "test"
            assert isinstance(loaded_cfg.trainer.base_dir, Path)
            assert isinstance(loaded_cfg.trainer.log_dir, Path)
            assert isinstance(loaded_cfg.trainer.ckpt_dir, Path)


def test_config_to_dict(tmpdir):
    cfg = make_minimal_config(tmpdir)
    d = config_to_dict(cfg)
    assert isinstance(d, dict)
    assert d["data"]["dataset"] == "test"
    assert d["trainer"]["lr_config"]["lr"] == 0.1


def test_update_config(tmpdir):
    cfg = make_minimal_config(tmpdir)
    updates = {"data": {"dataset": "new_test"}}
    new_cfg = update_config(cfg, updates)
    assert new_cfg.data.dataset == "new_test"
    assert cfg.data.dataset == "test"  # original unchanged


def test_preprocess_config(tmpdir):
    cfg = make_minimal_config(tmpdir)
    processed = preprocess_config(cfg)
    assert processed.trainer.log_dir is not None
    assert processed.trainer.ckpt_dir is not None
    # directories should exist or be Path objects
    assert isinstance(processed.trainer.log_dir, Path)
    assert isinstance(processed.trainer.ckpt_dir, Path)
