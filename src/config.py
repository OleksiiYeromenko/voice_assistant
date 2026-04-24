"""Config loader — reads YAML config with env var overrides."""

import os
from pathlib import Path

import yaml

_DEFAULT_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def _apply_env_override(cfg: dict, parts: list[str], val: str):
    """Walk config tree, greedily matching existing keys with underscores.

    For VA_LLM_LOCAL_THINK_MODEL, tries "local_think" before "local"+"think"
    so that keys containing underscores (like ``local_think``) are matched correctly.
    """
    d = cfg
    i = 0
    while i < len(parts) - 1:
        # Try longest compound key first (e.g. "local_think" before "local")
        matched = False
        for end in range(len(parts) - 1, i, -1):
            candidate = "_".join(parts[i:end])
            if candidate in d and isinstance(d[candidate], dict):
                d = d[candidate]
                i = end
                matched = True
                break
        if not matched:
            d = d.setdefault(parts[i], {})
            i += 1
    # Set the leaf value
    leaf = "_".join(parts[i:])
    d[leaf] = val


def load_config(path: str | Path | None = None) -> dict:
    """Load config from YAML. Environment variables override nested keys via VA_ prefix.

    Example: VA_LLM_LOCAL_MODEL=qwen3:1.7b overrides llm.local.model
    """
    path = Path(path) if path else _DEFAULT_PATH
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Apply env overrides: VA_STT_MODEL -> cfg["stt"]["model"]
    for key, val in os.environ.items():
        if not key.startswith("VA_"):
            continue
        parts = key[3:].lower().split("_")
        _apply_env_override(cfg, parts, val)

    return cfg
