"""Config loader — reads YAML config with env var overrides."""

import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
_DOTENV_PATH = Path(__file__).parent.parent / ".env"


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from .env into os.environ (does not overwrite existing vars)."""
    if not _DOTENV_PATH.exists():
        return
    for line in _DOTENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().removeprefix("export").strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()


def _coerce_env_value(val: str, existing: Any) -> Any:
    """Convert an env string to the type the config expects.

    Existing string settings stay strings (so a model named "3" is not turned into an
    int); everything else goes through YAML so "0.4", "true", "null" become proper values.
    """
    if isinstance(existing, str):
        return val
    try:
        return yaml.safe_load(val)
    except yaml.YAMLError:
        return val


def _set_env_override(d: dict[str, Any], parts: list[str], val: str) -> bool:
    """Set the config key addressed by ``parts`` under ``d``; return False if no key matches.

    Config keys contain underscores (``local_think``, ``aplay_device``), so the env name
    is ambiguous. Try the longest joined key first, at every level, including the final
    leaf: VA_TTS_APLAY_DEVICE -> tts -> "aplay_device".
    """
    for end in range(len(parts), 0, -1):
        key = "_".join(parts[:end])
        if key not in d:
            continue
        if end == len(parts):
            if not isinstance(d[key], dict):
                d[key] = _coerce_env_value(val, d[key])
                return True
        elif isinstance(d[key], dict) and _set_env_override(d[key], parts[end:], val):
            return True
    return False


def _apply_env_override(cfg: dict[str, Any], parts: list[str], val: str) -> None:
    """Apply one VA_* override. Unknown keys are created as nested dicts, one level per part."""
    if _set_env_override(cfg, parts, val):
        return
    d = cfg
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = _coerce_env_value(val, None)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
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
