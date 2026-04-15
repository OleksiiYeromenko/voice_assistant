#!/usr/bin/env python3
"""Manual test for Todoist shopping list tools.

Usage:
  uv run scripts/test_todoist.py list
  uv run scripts/test_todoist.py add "milk"
  uv run scripts/test_todoist.py add "eggs" "butter" "bread"
"""

import sys

sys.path.insert(0, ".")

from pathlib import Path
import os

# Load .env manually (no python-dotenv dependency)
_env = Path(".env")
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from src.tools.executor import get_shopping_list, add_to_shopping_list


def cmd_list():
    print("Fetching shopping list...")
    result = get_shopping_list()
    print(result)


def cmd_add(items: list[str]):
    for item in items:
        result = add_to_shopping_list(item)
        print(result)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("list", "add"):
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    if command == "list":
        cmd_list()
    elif command == "add":
        if len(sys.argv) < 3:
            print("Error: provide at least one item to add.")
            sys.exit(1)
        cmd_add(sys.argv[2:])


if __name__ == "__main__":
    main()
