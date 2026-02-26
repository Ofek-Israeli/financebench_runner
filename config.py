"""
Load and validate YAML config for financebench_runner.
"""

from pathlib import Path
from typing import Any, Dict, List

import yaml


REQUIRED_KEYS = [
    "model_id",
    "temperature",
    "sglang.base_url",
    "sglang.timeout_s",
    "sglang.max_retries",
    "max_new_tokens",
    "top_p",
    "seed",
    "prompt_template",
]


def _get(cfg: Dict[str, Any], path: str) -> Any:
    v = cfg
    for k in path.split("."):
        if not isinstance(v, dict):
            return None
        v = v.get(k)
    return v


def _is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, (str, list, dict)) and len(v) == 0:
        return True
    return False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config and validate required keys. Raises ValueError on missing/invalid."""
    path = Path(config_path)
    if not path.exists():
        raise ValueError(f"Config not found: {config_path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML object (dict)")
    missing: List[str] = []
    for key in REQUIRED_KEYS:
        if _is_empty(_get(cfg, key)):
            missing.append(key)
    if missing:
        raise ValueError(
            "Config missing or empty required keys: " + ", ".join(missing)
        )
    return cfg
