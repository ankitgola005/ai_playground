from copy import deepcopy
from pathlib import Path
import yaml
import ai_playground
from ai_playground.configs import Config
from typing import TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, Any

T = TypeVar("T", bound=Config)

# Base path for all config files
CONFIG_BASE_PATH: Path = Path(ai_playground.__file__).parent / "configs"


def load_yaml_config(filename: str) -> Config:
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


def config_to_dict(cfg: Config) -> Dict[str, Any]:
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
