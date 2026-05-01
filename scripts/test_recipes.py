#!/usr/bin/env python3
"""Manual test for recipe tools: web fetch → Notion add.

Usage:
  uv run scripts/test_recipes.py search <query>
  uv run scripts/test_recipes.py fetch <recipe name>
  uv run scripts/test_recipes.py add <json_string>
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

from src.tools.recipes import search_recipes, fetch_recipe_from_web, add_recipe_to_notion


def cmd_search(args: list[str]) -> None:
    query = " ".join(args)
    print(f"Searching Notion Recipe Library for: {query!r}\n")
    result = search_recipes(query=query)
    print(result)


def cmd_fetch(args: list[str]) -> None:
    name = " ".join(args)
    print(f"Fetching recipe from web: {name!r}\n")

    from src.llm.backends import LlamaCppBackend

    backend = LlamaCppBackend(num_predict=2000)
    result = fetch_recipe_from_web(recipe_name=name, backend=backend)
    print(result)

    if not result.startswith("DRAFT_JSON:"):
        return

    draft_json = result.split("DRAFT_JSON:", 1)[1].split("\n", 1)[0].strip()

    confirm = input("\nAdd this recipe to Notion? [y/N]: ").strip().lower()
    if confirm == "y":
        print("\nSaving to Notion...")
        add_result = add_recipe_to_notion(recipe_draft=draft_json, backend=backend)
        print(add_result)


def cmd_add(args: list[str]) -> None:
    draft_json = " ".join(args)
    print("Writing recipe draft to Notion...\n")

    from src.llm.backends import LlamaCppBackend

    backend = LlamaCppBackend(num_predict=2000)
    result = add_recipe_to_notion(recipe_draft=draft_json, backend=backend)
    print(result)


COMMANDS = {"search": cmd_search, "fetch": cmd_fetch, "add": cmd_add}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    rest = sys.argv[2:]

    if not rest and command in ("search", "fetch", "add"):
        print(f"Error: provide arguments for '{command}'.")
        print(__doc__)
        sys.exit(1)

    COMMANDS[command](rest)


if __name__ == "__main__":
    main()
