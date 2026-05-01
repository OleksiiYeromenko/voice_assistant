"""Recipe Library tools — Notion-backed family recipe store.

Four tools for the LLM:
  search_recipes         — filter the Notion Recipe Library database
  get_recipe             — full page with EN Translation Cache (translate + write if absent)
  fetch_recipe_from_web  — scrape klopotenko.com / smachno.in.ua and LLM-parse into a draft
  add_recipe_to_notion   — write a confirmed Recipe Draft to Notion after user confirms
"""

import json
import logging
import os
import re
from typing import Any, Callable

import httpx

log = logging.getLogger(__name__)

_NOTION_API = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"

# ---------------------------------------------------------------------------
# UI callback (recipe step display on IdleScreen)
# ---------------------------------------------------------------------------

_recipe_step_callback: Callable[[str, int, int], None] | None = None


def register_recipe_step_callback(cb: Callable[[str, int, int], None]) -> None:
    """Register a callback invoked when a recipe is fetched.

    Args:
        cb: callable(step_text, step_num, total_steps)
            step_num == 0  → step_text contains all steps (≤ 5 total)
            step_num > 0   → step_text is only the current step
    """
    global _recipe_step_callback
    _recipe_step_callback = cb


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


def _fallback_url() -> str:
    """Find an available local LLM server (used only when no backend is passed)."""
    try:
        from src.config import load_config
        cfg = load_config().get("llm", {})
        local = cfg.get("local_base_url", "http://localhost:8080")
        remote = cfg.get("remote_base_url", "http://192.168.1.74:11434")
    except Exception:
        local, remote = "http://localhost:8080", "http://192.168.1.74:11434"
    from src.llm.backends import check_llama_cpp_connectivity
    return local if check_llama_cpp_connectivity(local) else remote


def _llm_call(prompt: str, backend=None, max_tokens: int = 2048) -> str:
    """Single-turn LLM call using the active backend.

    Cloud backends (Claude/Gemini): streams via their SDK.
    Local backends (Ollama/llama.cpp): direct HTTP with custom max_tokens so
    recipe parsing isn't capped by the low num_predict tuned for conversation.
    """
    from src.llm.backends import ClaudeBackend, GeminiBackend

    if isinstance(backend, (ClaudeBackend, GeminiBackend)):
        result = ""
        for chunk in backend.stream([{"role": "user", "content": prompt}]):
            result += chunk.text
        return result.strip()

    base_url = getattr(backend, "_base_url", None) or _fallback_url()
    model = getattr(backend, "_model", "")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    try:
        with httpx.Client(timeout=900) as client:
            r = client.post(f"{base_url}/v1/chat/completions", json=payload)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            # Thinking models (e.g. Gemma-IT) put output in reasoning_content when
            # max_tokens is exhausted during thought — fall back to extract JSON from there.
            content = msg.get("content", "").strip()
            if not content:
                content = msg.get("reasoning_content", "").strip()
            return content
    except Exception as e:
        log.error(f"LLM call via {base_url} failed: {e}")
        raise RuntimeError(f"LLM backend unavailable: {e}") from e


def _translate_to_en(text_uk: str, backend=None) -> str:
    prompt = (
        "Translate the following Ukrainian recipe text to English. "
        "Preserve formatting: keep each line as a separate line. "
        "Output only the translation, no commentary.\n\n"
        f"{text_uk}"
    )
    return _llm_call(prompt, backend=backend, max_tokens=1024)


def _write_en_sections_to_notion(page_id: str, ingredients_en: str, instructions_en: str) -> None:
    """Append EN Translation Cache sections to an existing Notion page."""
    children: list[dict] = []

    def _h2(text: str) -> dict:
        return {
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": text}}]},
        }

    def _bullet(text: str) -> dict:
        return {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": text}}]},
        }

    def _numbered(text: str) -> dict:
        return {
            "object": "block",
            "type": "numbered_list_item",
            "numbered_list_item": {"rich_text": [{"type": "text", "text": {"content": text}}]},
        }

    children.append(_h2("Ingredients (EN)"))
    for line in ingredients_en.splitlines():
        if line.strip():
            children.append(_bullet(line.strip()))

    children.append(_h2("Instructions (EN)"))
    for line in instructions_en.splitlines():
        if line.strip():
            children.append(_numbered(line.strip()))

    with httpx.Client(timeout=15) as client:
        r = client.patch(
            f"{_NOTION_API}/blocks/{page_id}/children",
            headers=_notion_headers(),
            json={"children": children},
        )
        r.raise_for_status()


def _check_notion_config() -> str | None:
    """Return an error string if Notion env vars are missing, else None."""
    if not os.getenv("NOTION_API_KEY"):
        return (
            "ERROR: NOTION_API_KEY is not configured."
            " Tell the user recipe features are unavailable."
        )
    if not os.getenv("NOTION_RECIPES_DB_ID"):
        return (
            "ERROR: NOTION_RECIPES_DB_ID is not configured."
            " Tell the user recipe features are unavailable."
        )
    return None


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI/Ollama format)
# ---------------------------------------------------------------------------

SEARCH_RECIPES_TOOL = {
    "type": "function",
    "function": {
        "name": "search_recipes",
        "description": (
            "Search the family Recipe Library (Notion) for recipes by name, category,"
            " ingredients, cook time, or dietary preference."
            " Use this before get_recipe to find the exact recipe name."
        ),
        "parameters": {
            "type": "object",
            "required": [],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Recipe name or keyword — searched in both Ukrainian and English titles",
                },
                "category": {
                    "type": "string",
                    "description": "Dish category, e.g. 'Soup', 'Main', 'Dessert', 'Salad'",
                },
                "tags": {
                    "type": "string",
                    "description": "Comma-separated ingredient keywords, e.g. 'chicken, mushroom'",
                },
                "max_cook_time": {
                    "type": "integer",
                    "description": "Maximum cook time in minutes",
                },
                "dietary": {
                    "type": "string",
                    "description": "Dietary tag, e.g. 'vegetarian', 'vegan', 'gluten-free'",
                },
            },
        },
    },
}

GET_RECIPE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_recipe",
        "description": (
            "Fetch the full recipe (ingredients and step-by-step instructions in English)"
            " from the Recipe Library. Use after search_recipes."
            " Translates Ukrainian source on first access and caches the result in Notion."
        ),
        "parameters": {
            "type": "object",
            "required": ["title_en"],
            "properties": {
                "title_en": {
                    "type": "string",
                    "description": "English recipe name as returned by search_recipes",
                },
            },
        },
    },
}

FETCH_RECIPE_FROM_WEB_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_recipe_from_web",
        "description": (
            "Search klopotenko.com and smachno.in.ua for a recipe, scrape and parse it."
            " Returns a Recipe Draft for confirmation — does NOT write to Notion."
            " Only use when the recipe is not found in the Recipe Library."
            " Always present the draft to the user and wait for confirmation before calling"
            " add_recipe_to_notion."
        ),
        "parameters": {
            "type": "object",
            "required": ["recipe_name"],
            "properties": {
                "recipe_name": {
                    "type": "string",
                    "description": "Recipe name to search for",
                },
            },
        },
    },
}

ADD_RECIPE_TO_NOTION_TOOL = {
    "type": "function",
    "function": {
        "name": "add_recipe_to_notion",
        "description": (
            "Write a confirmed Recipe Draft to the Notion Recipe Library."
            " ONLY call this AFTER the user explicitly confirms the Confirmation Step."
            " Pass the DRAFT_JSON string exactly as returned by fetch_recipe_from_web."
        ),
        "parameters": {
            "type": "object",
            "required": ["recipe_draft"],
            "properties": {
                "recipe_draft": {
                    "type": "string",
                    "description": "JSON string of the recipe draft from fetch_recipe_from_web",
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def search_recipes(
    query: str = "",
    category: str = "",
    tags: str = "",
    max_cook_time: int = 0,
    dietary: str = "",
) -> str:
    err = _check_notion_config()
    if err:
        return err

    filters: list[dict] = []

    if query:
        filters.append({
            "or": [
                {"property": "Name", "title": {"contains": query}},
                {"property": "Title (EN)", "rich_text": {"contains": query}},
            ]
        })
    if category:
        filters.append({"property": "Category", "select": {"equals": category}})
    if tags:
        for tag in re.split(r"[,\s]+", tags.strip()):
            if tag:
                filters.append({"property": "Tags", "multi_select": {"contains": tag}})
    if max_cook_time > 0:
        filters.append({
            "property": "Cook Time (min)",
            "number": {"less_than_or_equal_to": max_cook_time},
        })
    if dietary:
        filters.append({"property": "Dietary", "multi_select": {"contains": dietary}})

    body: dict[str, Any] = {"page_size": 10}
    if len(filters) == 1:
        body["filter"] = filters[0]
    elif len(filters) > 1:
        body["filter"] = {"and": filters}

    try:
        with httpx.Client(timeout=10) as client:
            r = client.post(
                f"{_NOTION_API}/databases/{_db_id()}/query",
                headers=_notion_headers(),
                json=body,
            )
            r.raise_for_status()
            results = r.json().get("results", [])

        if not results:
            return "No recipes found in the Recipe Library matching your search."

        names = []
        for page in results:
            props = page.get("properties", {})
            title_en = _notion_text(props.get("Title (EN)", {}).get("rich_text", []))
            name_uk = _notion_text(props.get("Name", {}).get("title", []))
            display = title_en or name_uk
            cook_min = (props.get("Cook Time (min)") or {}).get("number")
            if cook_min:
                display += f" ({cook_min} min)"
            names.append(display)

        return f"Found {len(names)} recipe(s): " + "; ".join(names) + "."

    except Exception as e:
        log.error(f"search_recipes error: {e}")
        return (
            f"ERROR: Recipe search failed: {e}."
            " Tell the user the search failed; do not invent results."
        )


def get_recipe(title_en: str, backend=None) -> str:
    err = _check_notion_config()
    if err:
        return err

    try:
        # Search by English title first
        body: dict[str, Any] = {
            "filter": {"property": "Title (EN)", "rich_text": {"contains": title_en}},
            "page_size": 1,
        }
        with httpx.Client(timeout=10) as client:
            r = client.post(
                f"{_NOTION_API}/databases/{_db_id()}/query",
                headers=_notion_headers(),
                json=body,
            )
            r.raise_for_status()
            results = r.json().get("results", [])

        # Fall back to Ukrainian title search
        if not results:
            body2: dict[str, Any] = {
                "filter": {"property": "Name", "title": {"contains": title_en}},
                "page_size": 1,
            }
            with httpx.Client(timeout=10) as client:
                r = client.post(
                    f"{_NOTION_API}/databases/{_db_id()}/query",
                    headers=_notion_headers(),
                    json=body2,
                )
                r.raise_for_status()
                results = r.json().get("results", [])

        if not results:
            return (
                f"Recipe '{title_en}' not found in the Recipe Library."
                " Offer to use fetch_recipe_from_web to search Ukrainian recipe sites."
            )

        page = results[0]
        page_id = page["id"]
        props = page.get("properties", {})
        name_en = _notion_text(props.get("Title (EN)", {}).get("rich_text", [])) or title_en
        cook_min = (props.get("Cook Time (min)") or {}).get("number")
        category = ((props.get("Category") or {}).get("select") or {}).get("name", "")

        blocks = _all_blocks(page_id)
        sections = _parse_blocks(blocks)

        has_en_ing = bool(sections.get("Ingredients (EN)"))
        has_en_inst = bool(sections.get("Instructions (EN)"))

        if not has_en_ing or not has_en_inst:
            ing_uk = "\n".join(sections.get("Ingredients", []))
            inst_uk = "\n".join(sections.get("Instructions", []))
            try:
                ing_en_text = _translate_to_en(ing_uk, backend) if (ing_uk and not has_en_ing) else "\n".join(sections.get("Ingredients (EN)", []))
                inst_en_text = _translate_to_en(inst_uk, backend) if (inst_uk and not has_en_inst) else "\n".join(sections.get("Instructions (EN)", []))
                _write_en_sections_to_notion(page_id, ing_en_text, inst_en_text)
                log.info(f"Cached EN translation for: {name_en}")
                en_ingredients = [l for l in ing_en_text.splitlines() if l.strip()]
                en_instructions = [l for l in inst_en_text.splitlines() if l.strip()]
            except Exception as e:
                log.warning(f"Translation failed, using Ukrainian source: {e}")
                en_ingredients = sections.get("Ingredients", [])
                en_instructions = sections.get("Instructions", [])
        else:
            en_ingredients = sections.get("Ingredients (EN)", [])
            en_instructions = sections.get("Instructions (EN)", [])

        # Emit recipe step signal for IdleScreen display
        steps = [s for s in en_instructions if s.strip()]
        if _recipe_step_callback and steps:
            if len(steps) <= 5:
                all_text = "\n".join(f"Step {i}: {s}" for i, s in enumerate(steps, 1))
                _recipe_step_callback(all_text, 0, len(steps))
            else:
                _recipe_step_callback(steps[0], 1, len(steps))

        # Build voice-friendly response
        header = name_en
        if category:
            header += f" ({category})"
        if cook_min:
            header += f", {cook_min} minutes"

        ing_text = "; ".join(en_ingredients) if en_ingredients else "No ingredients listed."
        steps_text = " ".join(f"Step {i}: {s}" for i, s in enumerate(steps, 1))
        notes = sections.get("Notes", [])
        notes_text = " Note: " + " ".join(notes) if notes else ""

        return f"Recipe: {header}. Ingredients: {ing_text}. {steps_text}{notes_text}"

    except Exception as e:
        log.error(f"get_recipe error: {e}")
        return (
            f"ERROR: Could not fetch recipe: {e}."
            " Tell the user the lookup failed; do not invent recipe content."
        )


def fetch_recipe_from_web(recipe_name: str, backend=None) -> str:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        ddgs = DDGS(timeout=10)
        url: str = ""

        # Try each site separately so ddgs site: filtering is reliable
        for site in ("klopotenko.com", "smachno.in.ua"):
            results = ddgs.text(f"site:{site} {recipe_name}", max_results=2)
            if results:
                url = results[0].get("href", "")
                if url:
                    break

        if not url:
            return (
                f"No results found for '{recipe_name}' on klopotenko.com or smachno.in.ua."
                " Try a different recipe name."
            )

        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            html = resp.text

        # Strip HTML tags and truncate for LLM context
        clean = re.sub(r"<[^>]+>", " ", html)
        clean = re.sub(r"\s+", " ", clean).strip()[:6000]

        parse_prompt = (
            "You are parsing a Ukrainian recipe page. "
            "Extract the recipe and return ONLY a valid JSON object with exactly these keys:\n"
            '  "title_uk": Ukrainian recipe name (string)\n'
            '  "title_en": English translation of the name (string)\n'
            '  "category": dish category in English (string, e.g. "Soup", "Main", "Dessert")\n'
            '  "cook_time_min": total cook time in minutes (integer or null)\n'
            '  "dietary": obvious dietary tags (list of strings, e.g. ["vegetarian"])\n'
            '  "tags": main ingredient keywords in English (list of strings, max 8)\n'
            '  "ingredients_uk": ingredient lines in Ukrainian (list of strings)\n'
            '  "instructions_uk": instruction steps in Ukrainian (list of strings)\n'
            "Return only the JSON object, no commentary.\n\n"
            f"Page URL: {url}\n\nPage content:\n{clean}"
        )

        content = _llm_call(parse_prompt, backend=backend, max_tokens=2000)

        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            return (
                f"Could not parse recipe from '{url}'."
                " Try a more specific recipe name or different spelling."
            )

        draft = json.loads(json_match.group())

        title_en = draft.get("title_en", recipe_name)
        title_uk = draft.get("title_uk", "")
        category = draft.get("category", "")
        cook_time = draft.get("cook_time_min")
        ing_count = len(draft.get("ingredients_uk", []))
        step_count = len(draft.get("instructions_uk", []))

        summary_parts = [f'Found: "{title_en}"']
        if title_uk:
            summary_parts.append(f"(Ukrainian: \"{title_uk}\")")
        if category:
            summary_parts.append(f"category: {category}")
        if cook_time:
            summary_parts.append(f"{cook_time} min")
        summary_parts.append(f"{ing_count} ingredients, {step_count} steps")

        summary = " | ".join(summary_parts) + "."
        draft_json = json.dumps(draft, ensure_ascii=False)

        return (
            f"DRAFT_JSON: {draft_json}\n\n"
            f"SUMMARY: {summary} "
            "Read the ingredients and first few steps to the user, then ask: "
            "'Shall I add this recipe to the Recipe Library?' "
            "If confirmed, call add_recipe_to_notion with the DRAFT_JSON above."
        )

    except json.JSONDecodeError as e:
        return f"ERROR: Could not parse recipe JSON: {e}. Try a different recipe name."
    except Exception as e:
        log.error(f"fetch_recipe_from_web error: {e}")
        return (
            f"ERROR: Web recipe fetch failed: {e}."
            " Tell the user the fetch failed; do not invent recipe content."
        )


def add_recipe_to_notion(recipe_draft: str, backend=None) -> str:
    err = _check_notion_config()
    if err:
        return err

    try:
        draft = json.loads(recipe_draft)
    except (json.JSONDecodeError, TypeError) as e:
        return f"ERROR: Invalid recipe draft — could not parse JSON: {e}."

    title_uk = draft.get("title_uk", "")
    title_en = draft.get("title_en", title_uk)
    category = draft.get("category", "")
    cook_time = draft.get("cook_time_min")
    dietary = draft.get("dietary", [])
    tags = draft.get("tags", [])
    ingredients_uk = draft.get("ingredients_uk", [])
    instructions_uk = draft.get("instructions_uk", [])

    try:
        # Translate to build Translation Cache at add time
        ing_en_text = ""
        inst_en_text = ""
        try:
            if ingredients_uk:
                ing_en_text = _translate_to_en("\n".join(ingredients_uk), backend)
            if instructions_uk:
                inst_en_text = _translate_to_en("\n".join(instructions_uk), backend)
        except Exception as e:
            log.warning(f"Translation failed during add, skipping EN cache: {e}")

        # Notion page properties
        properties: dict[str, Any] = {
            "Name": {
                "title": [{"type": "text", "text": {"content": title_uk or title_en}}]
            },
            "Title (EN)": {
                "rich_text": [{"type": "text", "text": {"content": title_en}}]
            },
        }
        if category:
            properties["Category"] = {"select": {"name": category}}
        if cook_time:
            properties["Cook Time (min)"] = {"number": int(cook_time)}
        if dietary:
            properties["Dietary"] = {"multi_select": [{"name": d} for d in dietary]}
        if tags:
            properties["Tags"] = {"multi_select": [{"name": t} for t in tags]}

        # Page body blocks
        children: list[dict] = []

        def _h2(text: str) -> dict:
            return {
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": [{"type": "text", "text": {"content": text}}]},
            }

        def _bullet(text: str) -> dict:
            return {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": text}}]},
            }

        def _numbered(text: str) -> dict:
            return {
                "object": "block",
                "type": "numbered_list_item",
                "numbered_list_item": {"rich_text": [{"type": "text", "text": {"content": text}}]},
            }

        # Ukrainian source
        children.append(_h2("Ingredients"))
        for ing in ingredients_uk:
            children.append(_bullet(ing))

        children.append(_h2("Instructions"))
        for step in instructions_uk:
            children.append(_numbered(step))

        # Translation Cache
        if ing_en_text:
            children.append(_h2("Ingredients (EN)"))
            for line in ing_en_text.splitlines():
                if line.strip():
                    children.append(_bullet(line.strip()))

        if inst_en_text:
            children.append(_h2("Instructions (EN)"))
            for line in inst_en_text.splitlines():
                if line.strip():
                    children.append(_numbered(line.strip()))

        with httpx.Client(timeout=15) as client:
            r = client.post(
                f"{_NOTION_API}/pages",
                headers=_notion_headers(),
                json={
                    "parent": {"database_id": _db_id()},
                    "properties": properties,
                    "children": children,
                },
            )
            r.raise_for_status()

        en_note = " English translation cached." if ing_en_text else " No EN translation cached — fix in Notion if needed."
        return f"Recipe '{title_en}' added to the Recipe Library.{en_note}"

    except Exception as e:
        log.error(f"add_recipe_to_notion error: {e}")
        return (
            f"ERROR: Could not save recipe to Notion: {e}."
            " Tell the user the save failed; the recipe was NOT added."
        )
