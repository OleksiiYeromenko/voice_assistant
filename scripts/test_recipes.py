#!/usr/bin/env python3
"""Test the get_recipe tool against the live Notion family recipe library.

Tests:
  1. Direct get_recipe() call — verifies Notion API connectivity
  2. execute_tool() dispatch — verifies tool registration and dispatch chain
  3. Missing recipe — verifies graceful "not found" error handling

Usage:
  uv run scripts/test_recipes.py <recipe name>
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

for line in Path(".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from src.tools.executor import execute_tool  # noqa: E402
from src.tools.recipes import get_recipe     # noqa: E402

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

name = " ".join(sys.argv[1:])
ok = True

# ── Test 1: Direct Notion API call ───────────────────────────────────────────
print(f"[1/3] Direct get_recipe({name!r})")
result1 = get_recipe(recipe_name=name)
print(result1)
if result1.startswith("ERROR:"):
    print("FAIL — API error")
    ok = False
else:
    print("OK\n")

# ── Test 2: execute_tool() dispatch ──────────────────────────────────────────
print(f"[2/3] execute_tool('get_recipe', {{'name': {name!r}}})")
result2 = execute_tool("get_recipe", {"recipe_name": name})
if result2 != result1:
    print(
        f"FAIL — dispatch result differs from direct call"
        f"\n  direct:   {result1[:80]}"
        f"\n  executor: {result2[:80]}"
    )
    ok = False
else:
    print("OK — matches direct call\n")

# ── Test 3: Missing recipe → expect "not found" ───────────────────────────────
print("[3/3] Missing recipe → expect 'not found' message")
missing = execute_tool("get_recipe", {"recipe_name": "xxxxnosuchrecipexxxx"})
if "not found" in missing.lower() or "ERROR" in missing:
    print(f"OK — got: {missing}")
else:
    print(f"FAIL — unexpected response: {missing}")
    ok = False

print()
print("PASS" if ok else "FAIL — see above")
sys.exit(0 if ok else 1)
