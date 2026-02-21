"""Config loader — reads YAML config with env var overrides."""

from pathlib import Path
import os
import yaml

_DEFAULT_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


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
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val

    return cfg