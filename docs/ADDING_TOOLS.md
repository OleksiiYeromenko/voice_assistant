# Adding Tools to the Voice Assistant

This guide walks you through adding a new tool to the voice assistant. Tools are functions that the LLM can call to fetch information, control devices, or perform actions.

## Architecture Overview

The tool system consists of three parts:

1. **Tool Schema** — JSON specification that the LLM sees (name, description, parameters)
2. **Implementation Function** — Python async function that executes the tool
3. **Registration** — Wiring the schema and function together

## Step-by-Step Example: Create a "Get Sunrise/Sunset" Tool

Let's create a tool that tells you the sunrise and sunset times.

### Step 1: Create the Implementation Function

Add a new file `src/tools/astronomy.py`:

```python
"""Astronomy tools — sunrise, sunset, moon phase."""

import httpx


async def get_sunrise_sunset(latitude: float, longitude: float) -> str:
    """Fetch sunrise and sunset times for a location.
    
    Args:
        latitude: Location latitude (e.g., 40.7128 for NYC)
        longitude: Location longitude (e.g., -74.0060 for NYC)
    
    Returns:
        Human-readable string with sunrise/sunset times
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.sunrise-sunset.org/json",
                params={"lat": latitude, "lng": longitude},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            
            if data["status"] != "OK":
                return f"Error: {data.get('results', {})}"
            
            results = data["results"]
            sunrise = results["sunrise"]
            sunset = results["sunset"]
            return f"Sunrise at {sunrise}, sunset at {sunset}"
    except Exception as e:
        return f"Failed to fetch sunrise/sunset: {e}"
```

### Step 2: Define the Tool Schema

The schema tells the LLM what parameters the tool accepts. Add this to `src/tools/executor.py` in the `TOOLS` dict:

```python
TOOLS = {
    "get_weather": { ... },  # existing tools
    
    "get_sunrise_sunset": {
        "name": "get_sunrise_sunset",
        "description": "Get sunrise and sunset times for a specific location",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Location latitude (e.g., 40.7128 for New York)"
                },
                "longitude": {
                    "type": "number",
                    "description": "Location longitude (e.g., -74.0060 for New York)"
                }
            },
            "required": ["latitude", "longitude"]
        },
        "func": get_sunrise_sunset
    }
}
```

### Step 3: Import and Register the Function

At the top of `src/tools/executor.py`, add:

```python
from src.tools.astronomy import get_sunrise_sunset
```

Then register it in the `register_tool()` call or add it to `TOOLS` as shown above.

### Step 4: Test the Tool

```bash
# Test with the component test script
uv run scripts/test_components.py all

# Or test directly in the REPL
uv run python
>>> from src.tools.astronomy import get_sunrise_sunset
>>> import asyncio
>>> asyncio.run(get_sunrise_sunset(40.7128, -74.0060))
'Sunrise at 6:45:32 AM, sunset at 8:12:15 PM'
```

### Step 5: Update System Prompt (Optional)

If the tool needs special usage instructions, add context to the system prompt in `memory/PERSONA.md`:

```markdown
You have access to the following tools:
- get_sunrise_sunset: Get sunrise/sunset times by latitude/longitude
  (Automatically available in the NYC area)
```

## Schema Format Reference

Tools use **OpenAI function-calling format**. Here's the template:

```python
"tool_name": {
    "name": "tool_name",
    "description": "Brief description of what this tool does",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",  # or "number", "boolean", "array", "object"
                "description": "What this parameter is for"
            },
            "param2": {
                "type": "number",
                "description": "Another parameter"
            }
        },
        "required": ["param1"]  # List params that must be provided
    },
    "func": implementation_function  # Reference to the actual function
}
```

### Parameter Types

| Type | Python Example | Notes |
|------|---|---|
| `"string"` | `"hello"` | Text input |
| `"number"` | `42.5` | Float or int |
| `"integer"` | `42` | Integer only |
| `"boolean"` | `True` | True/False |
| `"array"` | `["a", "b"]` | List of items; specify `items` property |
| `"object"` | `{"key": "value"}` | Nested object; specify `properties` |

## Best Practices

### 1. **Handle Errors Gracefully**
Always return a string result, even on error:

```python
async def my_tool(query: str) -> str:
    try:
        result = await some_api_call(query)
        return f"Success: {result}"
    except TimeoutError:
        return "Request timed out — please try again"
    except Exception as e:
        return f"Error: {str(e)}"
```

### 2. **Keep Responses Brief**
The LLM sees the first 80 characters of tool results. Be concise:

```python
# ❌ Too long
return "The weather today is sunny with a high of 75°F and a low of 62°F with a 10% chance of precipitation and winds from the southwest at 5-10 mph"

# ✅ Better
return "Sunny, high 75°F, low 62°F, 10% rain chance"
```

### 3. **Add Type Hints**
Help other developers understand your tool:

```python
async def my_tool(name: str, count: int = 1) -> str:
    """Do something with name, repeated count times."""
    ...
```

### 4. **Describe Parameters Clearly**
The LLM will read your parameter descriptions to decide how to call the tool:

```python
"name": {
    "type": "string",
    "description": "The person's full name (e.g., 'Alice Johnson')"  # Good
}

"name": {
    "type": "string",
    "description": "name"  # ❌ Too vague
}
```

### 5. **Use Async Functions**
All tools should be `async` to avoid blocking the event loop:

```python
# ✅ Correct
async def my_tool(query: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    return str(response.text)

# ❌ Wrong
def my_tool(query: str) -> str:
    response = requests.get(url)  # Blocks the entire assistant
    return response.text
```

## Advanced: Stateful Tools

Some tools need to maintain state (e.g., shopping list, timer list). Use the `MarkdownMemoryStore`:

```python
from src.memory import MarkdownMemoryStore

memory = MarkdownMemoryStore()

async def get_shopping_list() -> str:
    items = memory.recall("shopping list")
    return items or "Shopping list is empty"

async def add_to_shopping_list(item: str) -> str:
    memory.remember(f"Shopping list item: {item}")
    return f"Added {item} to shopping list"
```

## Examples in the Codebase

Check out existing tools for reference:

- **Simple tool**: `src/tools/executor.py` → `get_time` (no API calls)
- **API tool**: `src/tools/executor.py` → `web_search` (HTTP request)
- **Stateful tool**: `src/tools/executor.py` → `add_to_shopping_list` (memory interaction)
- **Complex tool**: `src/tools/player.py` → `play_radio` (mpv subprocess control)

## Troubleshooting

### Tool Not Called by LLM
- **Check the schema**: Does the description clearly explain when to use it?
- **Check the system prompt**: Is it mentioned in `memory/PERSONA.md`?
- **Test directly**: Call the function manually to confirm it works

### Tool Returns Error to LLM
- **Read the error**: The error message is sent back to the LLM
- **Add logging**: Insert `log.info(f"Debug: {variable}")` to trace execution
- **Check timeouts**: API calls may timeout; use explicit `timeout=10`

### Tool Takes Too Long
- **Use async**: Ensure you're not blocking with sync I/O
- **Add timeout**: Limit API calls to 10 seconds max
- **Consider caching**: For tools that return the same data repeatedly

## See Also

- **[API Documentation](./API.md)** — Backend interface and tool execution flow
- **[Architecture](./ARCHITECTURE.md)** — System design and tool calling loop
- **[Development Guide](./DEVELOPMENT.md)** — Setup, testing, and coding standards
