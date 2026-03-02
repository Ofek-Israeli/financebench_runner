"""
Load and validate config for financebench_runner.

Supports two formats:
- YAML (legacy): .config with YAML structure (model_id:, sglang:, etc.).
- Kconfig: .config from 'make menuconfig' (CONFIG_MODEL_ID=..., CONFIG_SGLANG_BASE_URL=...).
  Same style as compressor_2. load_config() auto-detects format and returns the same nested dict.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

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

# Default prompt template when not set in Kconfig
_DEFAULT_PROMPT_TEMPLATE = """  Context:
  {context}

  Question:
  {query}

  Answer:
"""


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


def _is_kconfig_file(path: Path) -> bool:
    """Return True if the file looks like a Kconfig .config (CONFIG_*=...)."""
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("CONFIG_") and "=" in line:
                    return True
                if ":" in line and not line.startswith("CONFIG_"):
                    return False
        return False
    except Exception:
        return False


def _parse_kconfig_file(config_path: Path) -> Dict[str, Any]:
    """Parse a Kconfig .config into flat dict (keys without CONFIG_ prefix)."""
    values: Dict[str, Any] = {}
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, raw = line.partition("=")
            key = key.strip()
            if key.startswith("CONFIG_"):
                key = key[len("CONFIG_"):]
            raw = raw.strip()
            if raw.startswith('"') and raw.endswith('"'):
                raw = raw[1:-1].replace("\\n", "\n")
            if raw == "y":
                values[key] = True
            elif raw == "n":
                values[key] = False
            else:
                try:
                    values[key] = int(raw)
                except ValueError:
                    try:
                        values[key] = float(raw)
                    except ValueError:
                        values[key] = raw
    return values


def _kconfig_to_runner_dict(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Convert flat Kconfig keys to the nested dict structure the runner expects."""
    def _str(key: str, default: str = "") -> str:
        v = flat.get(key)
        return str(v).strip() if v is not None else default

    def _float(key: str, default: float = 0.0) -> float:
        v = flat.get(key)
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _int(key: str, default: int = 0) -> int:
        v = flat.get(key)
        if v is None:
            return default
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    example_indices: Optional[List[int]] = None
    raw_indices = _str("EXAMPLE_INDICES").strip()
    if raw_indices:
        parts = [p.strip() for p in raw_indices.split(",") if p.strip()]
        if parts:
            try:
                example_indices = [int(p) for p in parts]
            except ValueError:
                pass

    prompt = _str("PROMPT_TEMPLATE")
    if not prompt or "{context}" not in prompt or "{query}" not in prompt:
        prompt = _DEFAULT_PROMPT_TEMPLATE

    cfg: Dict[str, Any] = {
        "model_id": _str("MODEL_ID") or "meta-llama/Llama-3.1-8B-Instruct",
        "temperature": _float("TEMPERATURE", 0.0),
        "sglang": {
            "base_url": _str("SGLANG_BASE_URL") or "http://localhost:8000/v1",
            "timeout_s": _float("SGLANG_TIMEOUT_S", 120.0),
            "max_retries": _int("SGLANG_MAX_RETRIES", 3),
        },
        "max_new_tokens": _int("MAX_NEW_TOKENS", 512),
        "top_p": _float("TOP_P", 1.0),
        "seed": _int("SEED", 42),
        "prompt_template": prompt,
        "correctness_model": _str("CORRECTNESS_MODEL") or "gpt-4o",
        "correctness_tolerance": _float("CORRECTNESS_TOLERANCE", 0.10),
    }
    if example_indices is not None:
        cfg["example_indices"] = example_indices
    return cfg


def load_config(config_path: str) -> Dict[str, Any]:
    """Load config (YAML or Kconfig .config) and validate required keys. Raises ValueError on missing/invalid."""
    path = Path(config_path)
    if not path.exists():
        raise ValueError(f"Config not found: {config_path}")

    if _is_kconfig_file(path):
        flat = _parse_kconfig_file(path)
        cfg = _kconfig_to_runner_dict(flat)
    else:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError("Config must be a YAML object (dict)")
        if "sglang" not in cfg or not isinstance(cfg.get("sglang"), dict):
            cfg["sglang"] = cfg.get("sglang") or {}
        for key in ("correctness_model", "correctness_tolerance"):
            if key not in cfg:
                cfg[key] = "gpt-4o" if key == "correctness_model" else 0.10

    missing: List[str] = []
    for key in REQUIRED_KEYS:
        if _is_empty(_get(cfg, key)):
            missing.append(key)
    if missing:
        raise ValueError(
            "Config missing or empty required keys: " + ", ".join(missing)
        )
    return cfg
