"""Tool definitions and executor for LLM function calling.

Each tool is:
  - A schema dict (OpenAI/Ollama format) for the LLM
  - A Python function that executes the tool
"""

import inspect
import logging
from typing import Any

from src.tools.http_utils import http_get, http_post
from src.tools.player import play_radio, set_volume, stop_playback
from src.tools.recipes import GET_RECIPE_TOOL, get_recipe
from src.tools.timers import cancel_timer, set_timer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI/Ollama format)
# ---------------------------------------------------------------------------

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather or forecast for a city.",
        "parameters": {
            "type": "object",
            "required": ["city"],
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "forecast_days": {
                    "type": "integer",
                    "description": "0=current(default), 1=tomorrow, 2-7=days ahead",
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
            "Search the web for recent events, news, or rapidly-changing information."
            " Do NOT use for general knowledge or facts that do not change."
        ),
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
        },
    },
}

TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get current date and time, optionally for another city/timezone.",
        "parameters": {
            "type": "object",
            "required": [],
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or IANA timezone. Empty = local time.",
                },
            },
        },
    },
}

SHOPPING_LIST_TOOL = {
    "type": "function",
    "function": {
        "name": "add_to_shopping_list",
        "description": "Add an item to the Todoist shopping list.",
        "parameters": {
            "type": "object",
            "required": ["item"],
            "properties": {
                "item": {"type": "string", "description": "Item to add"},
            },
        },
    },
}

GET_SHOPPING_LIST_TOOL = {
    "type": "function",
    "function": {
        "name": "get_shopping_list",
        "description": "Show all items currently on the shopping list.",
        "parameters": {
            "type": "object",
            "required": [],
            "properties": {},
        },
    },
}


REMEMBER_TOOL = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": (
            "Store a fact or preference about the user for future reference. "
            "For radio: use 'Likes radio station: [name]' to accumulate liked stations, "
            "or 'My favorite radio station is [name]' to set the default (overwrites). "
            "Always use the full station name from <radio_status> context."
        ),
        "parameters": {
            "type": "object",
            "required": ["fact"],
            "properties": {
                "fact": {"type": "string", "description": "Information to store"},
            },
        },
    },
}


RECALL_TOOL = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": "Search past conversations or list what is known about the user.",
        "parameters": {
            "type": "object",
            "required": [],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords to search, or 'profile' for all known user info",
                },
            },
        },
    },
}


TIMER_TOOL = {
    "type": "function",
    "function": {
        "name": "set_timer",
        "description": (
            "Set a countdown timer (spoken alert when done). Convert durations to seconds."
        ),
        "parameters": {
            "type": "object",
            "required": ["duration_seconds"],
            "properties": {
                "duration_seconds": {
                    "type": "integer",
                    "description": "Duration in seconds",
                },
                "label": {
                    "type": "string",
                    "description": "Timer name, e.g. 'pasta'. Defaults to 'timer'.",
                },
            },
        },
    },
}

CANCEL_TIMER_TOOL = {
    "type": "function",
    "function": {
        "name": "cancel_timer",
        "description": "Cancel an active timer by label.",
        "parameters": {
            "type": "object",
            "required": [],
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Timer label. Defaults to 'timer'.",
                },
            },
        },
    },
}

PLAY_RADIO_TOOL = {
    "type": "function",
    "function": {
        "name": "play_radio",
        "description": (
            "Play an internet radio station. Pass a genre keyword "
            "(e.g. 'rock', 'jazz', 'classical'), optionally prefixed with "
            "'random' for a random pick (e.g. 'random jazz') and/or a country "
            "(e.g. 'rock Ukraine', 'random pop France'). For a specific station "
            "pass its name, optionally with country (e.g. 'BBC Radio 1', 'KISS FM Spain'). "
            "When the user says 'play some radio' with no genre, use their saved favorite "
            "station name. "
            "Do NOT call when the user asks what is currently playing — "
            "answer from <radio_status> context instead."
        ),
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Genre keyword optionally with 'random' and/or country (e.g. 'random rock Ukraine'). For a specific station use its name, optionally with country (e.g. 'KISS FM Spain').",  # noqa: E501
                },
            },
        },
    },
}

STOP_PLAYBACK_TOOL = {
    "type": "function",
    "function": {
        "name": "stop_radio",
        "description": "Stop radio or music playback. Safe to call when nothing is playing.",
        "parameters": {"type": "object", "required": [], "properties": {}},
    },
}

SET_VOLUME_TOOL = {
    "type": "function",
    "function": {
        "name": "set_volume",
        "description": (
            "Set music/radio volume 0-100. Rough mapping: quiet=30, normal=70,"
            " loud=90, max=100. Only affects music; the assistant's own voice is unchanged."
        ),
        "parameters": {
            "type": "object",
            "required": ["level"],
            "properties": {
                "level": {"type": "integer", "description": "Volume 0-100"},
            },
        },
    },
}


# All available tool schemas
ALL_TOOLS = [
    WEATHER_TOOL,
    WEB_SEARCH_TOOL,
    TIME_TOOL,
    SHOPPING_LIST_TOOL,
    GET_SHOPPING_LIST_TOOL,
    REMEMBER_TOOL,
    RECALL_TOOL,
    TIMER_TOOL,
    CANCEL_TIMER_TOOL,
    PLAY_RADIO_TOOL,
    STOP_PLAYBACK_TOOL,
    SET_VOLUME_TOOL,
    GET_RECIPE_TOOL,
]

# Tools whose results go stale immediately (e.g., time changes every minute).
# Responses using these tools are replaced with placeholders in conversation
# history so the LLM doesn't parrot old values on subsequent calls.
VOLATILE_TOOLS: set[str] = {"get_time", "get_weather", "get_shopping_list"}


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def get_weather(city: str, forecast_days: int = 0) -> str:
    """Fetch current weather or multi-day forecast from Open-Meteo (free, no API key).

    forecast_days=0  → current conditions
    forecast_days=1  → tomorrow
    forecast_days=N  → next N days starting tomorrow (max 7)
    """
    try:
        # Step 1: Geocode city name
        geo = http_get(
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
            weather = http_get(
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
            weather = http_get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": (
                        "weather_code,temperature_2m_max,temperature_2m_min,"
                        "precipitation_sum,wind_speed_10m_max"
                    ),
                    "forecast_days": days_capped + 1,
                },
                timeout=5,
            ).json()

            daily = weather["daily"]
            # Slice off today (index 0), take the requested number of future days
            dates = daily["time"][1 : days_capped + 1]
            codes = daily["weather_code"][1 : days_capped + 1]
            highs = daily["temperature_2m_max"][1 : days_capped + 1]
            lows = daily["temperature_2m_min"][1 : days_capped + 1]
            precip = daily["precipitation_sum"][1 : days_capped + 1]
            winds = daily["wind_speed_10m_max"][1 : days_capped + 1]

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
        return (
            f"ERROR: Weather lookup failed: {e}."
            " Tell the user the lookup failed; do not invent data."
        )


def get_time(location: str = "") -> str:
    """Get current date/time locally or for a given city/timezone."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    if not location or not location.strip():
        now = datetime.now()
        return now.strftime("Local time: %A, %B %d, %Y — %H:%M")

    # Try as IANA timezone first (e.g. "America/New_York")
    try:
        tz = ZoneInfo(location.strip())
        now = datetime.now(tz)
        return now.strftime(f"{location}: %A, %B %d, %Y — %H:%M")
    except (KeyError, ValueError):
        pass

    # Fall back to geocoding (same API as weather) to resolve city → timezone
    try:
        geo = http_get(
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
        return now.strftime(f"{city_name} ({tz_name}): %A, %B %d, %Y — %H:%M")
    except Exception as e:
        log.error(f"Time lookup error: {e}")
        return (
            f"ERROR: Time lookup failed: {e}. Tell the user the lookup failed; do not invent data."
        )


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
        return (
            f"ERROR: Web search failed: {e}."
            " Tell the user the search failed; do not invent results."
        )


_TODOIST_API = "https://api.todoist.com/api/v1"


def _todoist_headers() -> dict:
    import os

    token = os.getenv("TODOIST_API_TOKEN", "")
    return {"Authorization": f"Bearer {token}"}


def _todoist_project_id() -> str:
    import os

    return os.getenv("TODOIST_PROJECT_ID", "")


def get_shopping_list() -> str:
    """Return all open tasks in the Todoist shopping list project."""
    import os

    if not os.getenv("TODOIST_API_TOKEN"):
        return (
            "ERROR: TODOIST_API_TOKEN is not configured."
            " Tell the user the shopping list is unavailable until Todoist is set up."
        )

    try:
        project_id = _todoist_project_id()
        params = {"project_id": project_id} if project_id else {}
        r = http_get(f"{_TODOIST_API}/tasks", headers=_todoist_headers(), params=params, timeout=8)
        body = r.json()
        tasks = body.get("results", body) if isinstance(body, dict) else body
        if not tasks:
            return "The shopping list is empty."
        items = [t["content"] for t in tasks]
        log.info(f"Fetched {len(items)} items from Todoist")
        return "Shopping list: " + ", ".join(items) + "."
    except Exception as e:
        log.error(f"Todoist fetch failed: {e}")
        return (
            f"ERROR: Could not fetch shopping list: {e}."
            " Tell the user the fetch failed; do not invent items."
        )


def add_to_shopping_list(item: str) -> str:
    """Add item to Todoist shopping list."""
    import os

    if not os.getenv("TODOIST_API_TOKEN"):
        return (
            "ERROR: TODOIST_API_TOKEN is not configured."
            " Tell the user the shopping list is unavailable until Todoist is set up."
        )

    project_id = _todoist_project_id()
    payload: dict = {"content": item}
    if project_id:
        payload["project_id"] = project_id

    try:
        http_post(f"{_TODOIST_API}/tasks", headers=_todoist_headers(), json=payload, timeout=3)
        log.info(f"Added to Todoist: {item}")
        return f"Added '{item}' to the shopping list."
    except Exception as e:
        log.error(f"Todoist add failed: {e}")
        return (
            f"ERROR: Could not add '{item}' to shopping list: {e}."
            " Tell the user the item was NOT saved."
        )


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

_TOOL_FUNCTIONS: dict[str, callable] = {
    "get_weather": get_weather,
    "get_time": get_time,
    "web_search": web_search,
    "get_shopping_list": get_shopping_list,
    "add_to_shopping_list": add_to_shopping_list,
    "set_timer": set_timer,
    "cancel_timer": cancel_timer,
    "play_radio": play_radio,
    "stop_radio": stop_playback,
    "set_volume": set_volume,
    "get_recipe": get_recipe,
}


def register_tool(name: str, fn: callable):
    """Register a dynamic tool function (e.g., memory-backed tools).

    The tool schema must already be in ALL_TOOLS for the LLM to know about it.
    """
    _TOOL_FUNCTIONS[name] = fn
    log.info(f"Registered dynamic tool: {name}")


def execute_tool(name: str, arguments: dict[str, Any], backend=None) -> str:
    """Execute a tool by name. Returns result string."""
    fn = _TOOL_FUNCTIONS.get(name)
    if not fn:
        return f"Unknown tool: {name}"

    log.info(f"Executing tool: {name}({arguments})")
    try:
        if backend is not None and "backend" in inspect.signature(fn).parameters:
            return fn(**arguments, backend=backend)
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
    low_v = float(low) if low is not None else 0.0
    parts = [f"high {high_v:.0f}°C, low {low_v:.0f}°C, {_weather_code_to_text(int(code or 0))}"]
    if wind:
        parts.append(f"wind up to {float(wind):.0f} km/h")
    if precip and float(precip) > 0:
        parts.append(f"{float(precip):.1f}mm rain")
    return ", ".join(parts)


def _weather_code_to_text(code: int) -> str:
    """Convert WMO weather code to human-readable text."""
    codes = {
        0: "clear sky",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "foggy",
        48: "depositing rime fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        61: "slight rain",
        63: "moderate rain",
        65: "heavy rain",
        71: "slight snow",
        73: "moderate snow",
        75: "heavy snow",
        80: "slight rain showers",
        81: "moderate rain showers",
        82: "violent rain showers",
        95: "thunderstorm",
        96: "thunderstorm with slight hail",
        99: "thunderstorm with heavy hail",
    }
    return codes.get(code, f"weather code {code}")
