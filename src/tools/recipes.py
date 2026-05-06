"""Recipe Library — read family recipes from Notion."""

import logging
import os
from typing import Any

import httpx

log = logging.getLogger(__name__)

_NOTION_API = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"


# ---------------------------------------------------------------------------
# Notion helpers
# ---------------------------------------------------------------------------


def _notion_headers() -> dict[str, str]:
    key = os.getenv("NOTION_API_KEY", "")
    return {
        "Authorization": f"Bearer {key}",
        "Notion-Version": _NOTION_VERSION,
        "Content-Type": "application/json",
    }


def _db_id() -> str:
    return os.getenv("NOTION_RECIPES_DB_ID", "")


def _notion_text(rich_text: list) -> str:
    return "".join(t.get("plain_text", "") for t in rich_text)


def _parse_blocks(blocks: list[dict]) -> dict[str, list[str]]:
    """Parse Notion page blocks into named sections keyed by H2 heading text."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for block in blocks:
        btype = block.get("type", "")
        data = block.get(btype, {})
        if btype == "heading_2":
            current = _notion_text(data.get("rich_text", [])).strip()
            sections.setdefault(current, [])
        elif btype in ("paragraph", "bulleted_list_item", "numbered_list_item"):
            text = _notion_text(data.get("rich_text", [])).strip()
            if text and current is not None:
                sections[current].append(text)
    return sections


def _all_blocks(page_id: str) -> list[dict]:
    """Fetch all child blocks for a Notion page, following pagination cursors."""
    blocks: list[dict] = []
    cursor: str | None = None
    with httpx.Client(timeout=12) as client:
        while True:
            params: dict[str, Any] = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            r = client.get(
                f"{_NOTION_API}/blocks/{page_id}/children",
                headers=_notion_headers(),
                params=params,
            )
            r.raise_for_status()
            body = r.json()
            blocks.extend(body.get("results", []))
            if not body.get("has_more"):
                break
            cursor = body.get("next_cursor")
    return blocks


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

GET_RECIPE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_recipe",
        "description": (
            "Look up a recipe in the family Recipe Library by English name."
            " Returns ingredients and cooking instructions in English."
            " Call ONLY when the user explicitly asks how to make a dish,"
            " asks for a recipe, or asks for cooking instructions."
            " Do NOT call for general questions like 'what is X?' or 'tell me about X'."
        ),
        "parameters": {
            "type": "object",
            "required": ["recipe_name"],
            "properties": {
                "recipe_name": {
                    "type": "string",
                    "description": "English recipe name or keyword to search for",
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Tool function
# ---------------------------------------------------------------------------


def get_recipe(recipe_name: str) -> str:
    """Search Notion family recipes by English title and return the recipe."""
    api_key = os.getenv("NOTION_API_KEY", "")
    db_id = _db_id()
    if not api_key or not db_id:
        return (
            "ERROR: Recipe library not configured"
            " (missing NOTION_API_KEY or NOTION_RECIPES_DB_ID)."
        )

    try:
        with httpx.Client(timeout=10) as client:
            r = client.post(
                f"{_NOTION_API}/databases/{db_id}/query",
                headers=_notion_headers(),
                json={
                    "filter": {"property": "Title EN", "rich_text": {"contains": recipe_name}},
                    "page_size": 1,
                },
            )
            r.raise_for_status()
            results = r.json().get("results", [])

        if not results:
            return f"Recipe '{recipe_name}' not found in the Recipe Library."

        page_id = results[0]["id"]
        title = (
            _notion_text(
                results[0].get("properties", {}).get("Title EN", {}).get("rich_text", [])
            )
            or recipe_name
        )

        sections = _parse_blocks(_all_blocks(page_id))
        ingredients = sections.get("Ingredients", [])
        instructions = sections.get("Instructions", [])

        if not ingredients and not instructions:
            return f"Recipe '{title}' found but has no English content yet."

        lines = [f"Recipe: {title}", "", "Ingredients:"]
        lines.extend(f"  - {item}" for item in ingredients)
        lines += ["", "Instructions:"]
        lines.extend(f"  {i}. {step}" for i, step in enumerate(instructions, 1))
        return "\n".join(lines)

    except Exception as e:
        log.error(f"get_recipe error: {e}")
        return f"ERROR: Could not fetch recipe: {e}."


def parse_recipe_result(text: str) -> dict | None:
    """Parse the formatted string from get_recipe into a structured dict.

    Returns {"title": str, "ingredients": list[str], "instructions": list[str]}
    or None if the text looks like an error or has no content.
    """
    if not text or text.startswith("ERROR") or text.startswith("Recipe '"):
        return None

    title = ""
    ingredients: list[str] = []
    instructions: list[str] = []
    section: str | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if raw.startswith("Recipe: "):
            title = raw[len("Recipe: "):]
        elif line == "Ingredients:":
            section = "ingredients"
        elif line == "Instructions:":
            section = "instructions"
        elif section == "ingredients" and line.startswith("- "):
            ingredients.append(line[2:])
        elif section == "instructions" and line and line[0].isdigit():
            parts = line.split(". ", 1)
            if len(parts) == 2:
                instructions.append(parts[1])

    if not title or (not ingredients and not instructions):
        return None

    return {"title": title, "ingredients": ingredients, "instructions": instructions}
