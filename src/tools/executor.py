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
        "description": (
            "Get current weather or a multi-day forecast for a city. "
            "Use forecast_days=0 for current conditions, forecast_days=1 for tomorrow, "
            "forecast_days=3 for the next 3 days, forecast_days=7 for the week ahead."
        ),
        "parameters": {
            "type": "object",
            "required": ["city"],
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'London' or 'New York'",
                },
                "forecast_days": {
                    "type": "integer",
                    "description": (
                        "0 = current conditions (default), "
                        "1 = tomorrow, 2-7 = that many days ahead starting tomorrow"
                    ),
                },
            },
        },
    },
}

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information. Use for: current prices, "
            "live data, recent news, sports scores, stock prices, or any factual "
            "question requiring up-to-date information."
        ),
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — be specific, e.g. 'gold price per ounce today'",
                },
            },
        },
    },
}

TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": (
            "Get the current date and time. Use when the user asks 'what time is it', "
            "'what's the date', or asks about the time in another city/timezone."
        ),
        "parameters": {
            "type": "object",
            "required": [],
            "properties": {
                "location": {
                    "type": "string",
                    "description": (
                        "Optional city or timezone. Leave empty for local time. "
                        "Examples: 'London', 'Tokyo', 'New York', 'America/Chicago'"
                    ),
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
            "Store information about the user for future reference. "
            "Use this when the user says 'remember that...', 'my name is...', "
            "'I like...', 'always use...', 'prefer...', 'never...', or shares "
            "important personal information or preferences."
        ),
        "parameters": {
            "type": "object",
            "required": ["fact"],
            "properties": {
                "fact": {
                    "type": "string",
                    "description": (
                        "The information to remember, e.g. 'User prefers metric units' "
                        "or 'always use 24h time format'"
                    ),
                },
            },
        },
    },
}


RECALL_TOOL = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": (
            "Search past conversation summaries. Use when the user asks about "
            "previous conversations, e.g. 'what did we talk about yesterday?', "
            "'do you remember when we discussed...?', 'check our previous conversations'."
        ),
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Keywords to search for in past conversations, "
                        "e.g. 'recipe', 'weather london', 'yesterday'"
                    ),
                },
            },
        },
    },
}


MY_MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "my_memory",
        "description": (
            "List everything the assistant knows about the user: preferences and facts. "
            "Use when the user asks 'what are my preferences?', 'what do you know about me?', "
            "'list my settings', 'what have I told you?'."
        ),
        "parameters": {
            "type": "object",
            "required": [],
            "properties": {},
        },
    },
}


# All available tool schemas
ALL_TOOLS = [
    WEATHER_TOOL, WEB_SEARCH_TOOL, TIME_TOOL, SHOPPING_LIST_TOOL,
    REMEMBER_TOOL, RECALL_TOOL, MY_MEMORY_TOOL,
]

# Tools whose results go stale immediately (e.g., time changes every minute).
# Responses using these tools are replaced with placeholders in conversation
# history so the LLM doesn't parrot old values on subsequent calls.
VOLATILE_TOOLS: set[str] = {"get_time"}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def get_weather(city: str, forecast_days: int = 0) -> str:
    """Fetch current weather or multi-day forecast from Open-Meteo (free, no API key).

    forecast_days=0  → current conditions
    forecast_days=1  → tomorrow
    forecast_days=N  → next N days starting tomorrow (max 7)
    """
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

        if forecast_days <= 0:
            # Current conditions
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

        else:
            # Daily forecast — request N+1 days so index 0 (today) can be skipped
            days_capped = min(forecast_days, 7)
            weather = httpx.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
                    "forecast_days": days_capped + 1,
                },
                timeout=5,
            ).json()

            daily = weather["daily"]
            # Slice off today (index 0), take the requested number of future days
            dates  = daily["time"][1 : days_capped + 1]
            codes  = daily["weather_code"][1 : days_capped + 1]
            highs  = daily["temperature_2m_max"][1 : days_capped + 1]
            lows   = daily["temperature_2m_min"][1 : days_capped + 1]
            precip = daily["precipitation_sum"][1 : days_capped + 1]
            winds  = daily["wind_speed_10m_max"][1 : days_capped + 1]

            if days_capped == 1:
                day_str = _format_day(codes[0], highs[0], lows[0], precip[0], winds[0])
                return f"{name} tomorrow: {day_str}."

            parts = []
            for i in range(len(dates)):
                day_name = _day_name(dates[i])
                day_str = _format_day(codes[i], highs[i], lows[i], precip[i], winds[i])
                parts.append(f"{day_name}: {day_str}")
            return f"{name} {days_capped}-day forecast. " + ". ".join(parts) + "."

    except Exception as e:
        log.error(f"Weather error: {e}")
        return f"Weather lookup failed: {e}"


def get_time(location: str = "") -> str:
    """Get current date/time locally or for a given city/timezone."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    if not location or not location.strip():
        now = datetime.now()
        return now.strftime("Local time: %A, %B %d, %Y — %I:%M %p")

    # Try as IANA timezone first (e.g. "America/New_York")
    try:
        tz = ZoneInfo(location.strip())
        now = datetime.now(tz)
        return now.strftime(f"{location}: %A, %B %d, %Y — %I:%M %p")
    except (KeyError, ValueError):
        pass

    # Fall back to geocoding (same API as weather) to resolve city → timezone
    import httpx
    try:
        geo = httpx.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location.strip(), "count": 1},
            timeout=5,
        ).json()

        if not geo.get("results"):
            return f"Could not find location: {location}"

        tz_name = geo["results"][0].get("timezone", "")
        city_name = geo["results"][0].get("name", location)

        if not tz_name:
            return f"No timezone data for: {location}"

        tz = ZoneInfo(tz_name)
        now = datetime.now(tz)
        return now.strftime(f"{city_name} ({tz_name}): %A, %B %d, %Y — %I:%M %p")
    except Exception as e:
        log.error(f"Time lookup error: {e}")
        return f"Time lookup failed: {e}"


def web_search(query: str) -> str:
    """Web search via ddgs (multi-engine). Returns top 3 results, truncated for LLM context."""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    try:
        ddgs = DDGS(timeout=10)

        # Try news first for time-sensitive queries, fall back to general text
        results = ddgs.text(query, max_results=3, timelimit="m")
        if not results:
            results = ddgs.text(query, max_results=3)

        if not results:
            return f"No results found for: {query}"

        summaries = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "").strip()[:80]
            body = r.get("body", "").strip()[:200]
            summaries.append(f"[{i}] {title}: {body}")

        return "\n".join(summaries)
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
    "get_time": get_time,
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


def _day_name(date_str: str) -> str:
    """Return weekday name from an ISO date string, e.g. '2024-01-15' → 'Monday'."""
    from datetime import date
    return date.fromisoformat(date_str).strftime("%A")


def _format_day(code, high, low, precip, wind) -> str:
    """Format a single forecast day as a concise voice-friendly string."""
    high_v = float(high) if high is not None else 0.0
    low_v  = float(low)  if low  is not None else 0.0
    parts = [f"high {high_v:.0f}°C, low {low_v:.0f}°C, {_weather_code_to_text(int(code or 0))}"]
    if wind:
        parts.append(f"wind up to {float(wind):.0f} km/h")
    if precip and float(precip) > 0:
        parts.append(f"{float(precip):.1f}mm rain")
    return ", ".join(parts)


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