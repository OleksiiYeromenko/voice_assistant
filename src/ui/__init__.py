"""UI package — PyQt6 display for Raspberry Pi Touch Display 2.

Install the optional dependency group before use:
    uv sync --extra ui

Launch:
    python -m src.main --ui [--no-wake] [--windowed]

The ``--windowed`` flag keeps the window at 800×480 instead of fullscreen,
useful when developing on a desktop machine.
"""

from src.ui.app import run_ui

__all__ = ["run_ui"]
