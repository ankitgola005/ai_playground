from typing import TYPE_CHECKING
from pathlib import Path
import yaml
import ai_playground

if TYPE_CHECKING:
    from typing import Any, Mapping, Sequence

# Base path for all config files
CONFIG_BASE_PATH: Path = Path(ai_playground.__file__).parent / "configs"


class ConfigNode:
    """
    Wrapper around a dictionary that allows attribute-style access.

    Example:
        cfg = ConfigNode({"model": {"n_layer": 12}})
        print(cfg.model.n_layer)  # 12
    """

    def __init__(self, data: Mapping[str, Any]):
        for k, v in data.items():
            if isinstance(v, dict):
                v = ConfigNode(v)
            setattr(self, k, v)

    def get(self, key: str, default: Any = None) -> Any:
        """Get an attribute, returning default if not found."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        """Allow setting attributes via dict-style access."""
        setattr(self, key, value)

    def __repr__(self) -> str:
        return f"ConfigNode({self.__dict__})"


def load_yaml_config(filename: str) -> ConfigNode:
    """
    Load a YAML config file from ai_playground/configs and return a ConfigNode.

    Args:
        filename (str): Name of the YAML file (relative to configs directory), e.g., "gpt_config.yaml"

    Returns:
        ConfigNode: Top-level config object with attribute access.

    Raises:
        FileNotFoundError: If the file does not exist at the expected path.
    """
    path: Path = CONFIG_BASE_PATH / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg_dict: dict = yaml.safe_load(f)

    return ConfigNode(cfg_dict)


def config_to_dict(obj: Any) -> Any:
    """
    Recursively convert a ConfigNode, or object with __dict__ into a dictionary.

    Args:
        obj: Object to convert

    Returns:
        dict or list: Fully converted dictionary/list representation
    """
    if hasattr(obj, "__dict__"):
        return {k: config_to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, (list, tuple, Sequence)):
        return [config_to_dict(v) for v in obj]
    else:
        return obj
