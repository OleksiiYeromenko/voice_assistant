#!/usr/bin/env python3
"""Manual integration test for the vacuum tool.

Requires the Valetudo robot to be reachable at the configured IP.

Usage:
  uv run scripts/test_vacuum.py
"""

import sys

sys.path.insert(0, ".")

from src.config import load_config
from src.tools.vacuum import init_vacuum, start_vacuum


def main():
    cfg = load_config()
    init_vacuum(cfg)

    print("Sending start command to vacuum...")
    result = start_vacuum()
    print(f"Result: {result}")

    if result.startswith("ERROR:"):
        sys.exit(1)


if __name__ == "__main__":
    main()
