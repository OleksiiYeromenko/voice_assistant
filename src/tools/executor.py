"""Tool definitions and executor for LLM function calling.

Each tool is:
  - A schema dict (OpenAI/Ollama format) for the LLM
  - A Python function that executes the tool
"""

import json
import logging
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI/Ollama format)
# ---------------------------------------------------------------------------

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city. Use this when the user asks about weather.",
        "parameters": {
            "type": "object",
            "required": ["city"],
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'London' or 'New York'",
                },
            },
        },
    },
}

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information. Use this when you need facts you don't know.",
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
            },
        },
    },
}

SHOPPING_LIST_TOOL = {
    "type": "function",
    "function": {
        "name": "add_to_shopping_list",
        "description": "Add an item to the shopping list. Use this when the user wants to add something to buy.",
        "parameters": {
            "type": "object",
            "required": ["item"],
            "properties": {
                "item": {
                    "type": "string",
                    "description": "Item to add, e.g. 'milk', '2 kg potatoes'",
                },
            },
        },
    },
}


REMEMBER_TOOL = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": (
            "Store a fact about the user for future reference. "
            "Use this when the user says 'remember that...', 'my name is...', "
            "'I like...', or shares important personal information."
        ),
        "parameters": {
            "type": "object",
            "required": ["fact"],
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "The fact to remember, e.g. 'User prefers metric units'",
                },
            },
        },
    },
}


# All available tool schemas
ALL_TOOLS = [WEATHER_TOOL, WEB_SEARCH_TOOL, SHOPPING_LIST_TOOL, REMEMBER_TOOL]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def get_weather(city: str) -> str:
    """Fetch weather from Open-Meteo (free, no API key)."""
    import httpx

    try:
        # Step 1: Geocode city name
        geo = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=5,
        ).json()

        if not geo.get("results"):
            return f"Could not find city: {city}"

        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]
        name = geo["results"][0].get("name", city)

        # Step 2: Get current weather
        weather = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            },
            timeout=5,
        ).json()

        cur = weather["current"]
        temp = cur["temperature_2m"]
        humidity = cur["relative_humidity_2m"]
        wind = cur["wind_speed_10m"]
        code = cur["weather_code"]
        desc = _weather_code_to_text(code)

        return f"{name}: {temp}°C, {desc}, humidity {humidity}%, wind {wind} km/h"
    except Exception as e:
        log.error(f"Weather error: {e}")
        return f"Weather lookup failed: {e}"


def web_search(query: str) -> str:
    """Web search via DuckDuckGo. Returns top 3 results."""
    from duckduckgo_search import DDGS

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))

        if not results:
            return f"No results found for: {query}"

        summaries = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            summaries.append(f"{title}: {body}"[:200])

        return " | ".join(summaries)
    except Exception as e:
        log.error(f"Search error: {e}")
        return f"Search failed: {e}"


# --- Shopping list: simple local file for now, Google Keep later ---
_SHOPPING_LIST_FILE = "data/shopping_list.txt"


def add_to_shopping_list(item: str) -> str:
    """Add item to local shopping list file."""
    from pathlib import Path

    path = Path(_SHOPPING_LIST_FILE)
    path.parent.mkdir(exist_ok=True)
    with open(path, "a") as f:
        f.write(f"- {item}\n")

    log.info(f"Added to shopping list: {item}")
    return f"Added '{item}' to shopping list."


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

_TOOL_FUNCTIONS: dict[str, callable] = {
    "get_weather": get_weather,
    "web_search": web_search,
    "add_to_shopping_list": add_to_shopping_list,
}


def register_tool(name: str, fn: callable):
    """Register a dynamic tool function (e.g., memory-backed tools).

    The tool schema must already be in ALL_TOOLS for the LLM to know about it.
    """
    _TOOL_FUNCTIONS[name] = fn
    log.info(f"Registered dynamic tool: {name}")


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool by name. Returns result string."""
    fn = _TOOL_FUNCTIONS.get(name)
    if not fn:
        return f"Unknown tool: {name}"

    log.info(f"Executing tool: {name}({arguments})")
    try:
        return fn(**arguments)
    except Exception as e:
        log.error(f"Tool {name} failed: {e}")
        return f"Tool error: {e}"


def _weather_code_to_text(code: int) -> str:
    """Convert WMO weather code to human-readable text."""
    codes = {
        0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
        45: "foggy", 48: "depositing rime fog",
        51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
        61: "slight rain", 63: "moderate rain", 65: "heavy rain",
        71: "slight snow", 73: "moderate snow", 75: "heavy snow",
        80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
        95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
    }
    return codes.get(code, f"weather code {code}")