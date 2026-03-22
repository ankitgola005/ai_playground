from copy import deepcopy
from pathlib import Path
from pydantic import BaseModel
import yaml
import ai_playground
from ai_playground.configs import Config
from ai_playground.utils.paths import resolve_dirs, convert_paths

from typing import TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, Any

T = TypeVar("T", bound=Config)

# Base path for all config files
CONFIG_BASE_PATH: Path = Path(ai_playground.__file__).parent / "configs"


def load_config(filename: str) -> Config:
    """
    Load YAML config and return validated Pydantic Config.

    Raises:
        ValidationError: If config is invalid
        FileNotFoundError: If file does not exist
    """
    path = CONFIG_BASE_PATH / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    return Config(**cfg_dict)


def config_to_dict(cfg: BaseModel) -> Dict[str, Any]:
    """
    Convert Pydantic config to a plain Python dictionary.

    Args:
        cfg: Pydantic Config object

    Returns:
        Dict representation of config
    """
    return cfg.model_dump()


def update_config(cfg: T, updates: Dict[str, Any]) -> T:
    """
    Deep update a Pydantic config with a dictionary.

    Args:
        cfg: Existing config object
        updates: Nested dictionary of updates

    Returns:
        New validated config instance
    """
    base = deepcopy(cfg.model_dump())

    def deep_update(orig: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in new.items():
            if isinstance(v, dict) and isinstance(orig.get(k), dict):
                orig[k] = deep_update(orig[k], v)
            else:
                orig[k] = v
        return orig

    new_dict = deep_update(base, updates)
    return type(cfg)(**new_dict)


def preprocess_config(config: Config) -> Config:
    """
    Preprocess config.
    1. Resolve log_dir and ckpt_dir based on base_dir and run_name.
    """
    run_dir, log_dir, ckpt_dir = resolve_dirs(config.trainer)

    return update_config(
        config,
        {
            "trainer": {
                "base_dir": run_dir,
                "log_dir": log_dir,
                "ckpt_dir": ckpt_dir,
            }
        },
    )


def save_config_snapshot(config: Config) -> None:
    """
    Persist the resolved config to disk.
    This should be called after all preprocessing steps (e.g., path resolution,
    default filling), so the saved config reflects the exact state used for
    the run.
    The config is saved as `config.yaml` inside `trainer.log_dir`.

    Args:
        config: Config object.

    Raises:
        ValueError: If `trainer.log_dir` is not set.
        OSError: If the file cannot be written.
    """
    if config.trainer.log_dir is None:
        raise ValueError("Unable to resolve log_dir.")

    path = config.trainer.log_dir / "config.yaml"
    cfg_dict = convert_paths(config.model_dump())

    with open(path, "w") as f:
        yaml.safe_dump(cfg_dict, f)


def get_config(filename: str) -> Config:
    """
    Load, preprocess, and finalize a config for training.

    Pipeline:
    1. Load raw YAML config
    2. Pydantic validation
    3. Preprocess config: resolve paths
    4. Save a snapshot of the final config for reproducibility

    Args:
        filename: Name of the YAML config file.

    Returns:
        Config object.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValidationError: If config is invalid.
        ValueError: If preprocessing fails (e.g., invalid paths).
        OSError: If snapshot saving fails.
    """
    cfg = load_config(filename=filename)
    cfg = preprocess_config(config=cfg)
    save_config_snapshot(config=cfg)
    return cfg
