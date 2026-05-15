# Voice Assistant — Phase 2 Plan

## Already shipped (Phase 1)
- `src/tools/http_utils.py` — retry-aware `http_get`/`http_post` (2 attempts, 0.5s backoff)
- `VOLATILE_TOOLS` expanded to `{"get_time", "get_weather", "get_shopping_list"}` in `src/tools/executor.py`

---

## Features in this plan

### Feature 1 — n8n Outbound Webhook Tool (~2h)

**What:** New `trigger_automation(workflow_name, payload?)` tool so the LLM can fire n8n workflows
by voice. Covers both general automation triggers and smart home control — same tool, different
workflow names configured in n8n.

**New file:** `src/tools/automation.py`

```python
import os
from src.tools.http_utils import http_post

def trigger_automation(workflow_name: str, payload: dict | None = None) -> str:
    cfg = _load_n8n_cfg()          # reads config/config.yaml tools.n8n section
    url = f"{cfg['base_url']}/webhook/{workflow_name}"
    token = cfg.get("token", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    http_post(url, json=payload or {}, headers=headers, timeout=5)
    return f"Done. '{workflow_name}' triggered."
```

**Config additions in `config/config.yaml`:**
```yaml
tools:
  n8n:
    base_url: "http://localhost:5678"
    token: ""          # set via VA_TOOLS_N8N_TOKEN env var or .env
```

**Wire-up in `src/tools/executor.py`:**
- Import `trigger_automation` from `src.tools.automation`
- Add schema `TRIGGER_AUTOMATION_TOOL` to `ALL_TOOLS` list
- Add `"trigger_automation": trigger_automation` to `_TOOL_FUNCTIONS` dict

**Wire-up in `src/main.py`:**
- Add `"trigger_automation"` to `ACTION_TOOLS` frozenset

**Example voice commands:**
- "turn on the kitchen lights" → `trigger_automation("lights_kitchen_on")`
- "start the coffee maker" → `trigger_automation("coffee_maker")`
- "run my morning routine" → `trigger_automation("morning_routine")`

**Verification:**
1. Create a test webhook in n8n (Trigger node → Webhook, path = `test_webhook`)
2. Say "trigger test_webhook" or ask the assistant to run it
3. Confirm execution appears in n8n's execution history

---

### Feature 2 — Inbound HTTP API Server (~1 day)

**What:** A small FastAPI server running in a daemon thread so n8n (or any local service) can push
messages into the voice assistant — speaking alerts, triggering queries.

**New file:** `src/api/server.py`

```python
import queue, threading
from fastapi import FastAPI, HTTPException, Request
import uvicorn

def start(push_queue: queue.Queue, token: str, port: int) -> None:
    app = FastAPI()

    def _check_auth(req: Request):
        if token and req.headers.get("Authorization") != f"Bearer {token}":
            raise HTTPException(status_code=401)

    @app.post("/speak")
    async def speak(body: dict, req: Request):
        _check_auth(req)
        push_queue.put({"type": "speak", "text": body["message"]})
        return {"ok": True}

    @app.post("/query")
    async def query(body: dict, req: Request):
        _check_auth(req)
        push_queue.put({"type": "query", "text": body["text"]})
        return {"ok": True}

    def _run():
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

    threading.Thread(target=_run, daemon=True).start()
```

**`src/main.py` changes:**
```python
import queue
push_queue = queue.Queue()
if cfg.get("api", {}).get("enabled", False):
    from src.api.server import start as start_api
    start_api(push_queue, cfg["api"]["token"], cfg["api"]["port"])
# pass push_queue into StateMachine constructor
```

**`src/state_machine.py` changes:**
1. Constructor: `def __init__(self, ..., push_queue: queue.Queue | None = None)`
   - `self.push_queue = push_queue or queue.Queue()`
2. `_state_idle` WAKE_WORD loop — on each heartbeat tick add:
```python
try:
    msg = self.push_queue.get_nowait()
    if msg["type"] == "speak":
        self.tts.speak_text(msg["text"])   # or use existing speak mechanism
    elif msg["type"] == "query":
        return State.SESSION_CHECK, {"text": msg["text"]}
except queue.Empty:
    pass
```
Note: `speak_text` needs to be verified against the TTS engine API.
Check `src/tts/engine.py` for the right method to call for a one-shot speak.

**Config additions:**
```yaml
api:
  enabled: true
  port: 9876
  token: "change-me"        # VA_API_TOKEN env var
```

**Add to `pyproject.toml` dependencies:**
```
"fastapi>=0.100",
"uvicorn>=0.24",
```

**Verification:**
```bash
curl -s -X POST http://localhost:9876/speak \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello from n8n"}'
# → assistant speaks the message

curl -s -X POST http://localhost:9876/query \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{"text": "What is the weather in Wroclaw?"}'
# → full LLM pipeline runs, response spoken
```

---

### Feature 3 — Morning Briefing n8n Workflow (~1h, n8n config only)

No code changes to voice assistant. Requires Feature 2 (HTTP API server) to be running.

**n8n workflow steps:**
1. **Schedule Trigger** — 7:00am, Monday–Friday (or every day)
2. **HTTP Request** — GET `https://api.open-meteo.com/v1/forecast?latitude=51.107&longitude=17.038&current_weather=true&temperature_unit=celsius`
   (Wroclaw coordinates: lat 51.107, lon 17.038)
3. **HTTP Request** — GET `https://api.todoist.com/api/v1/tasks?project_id={{TODOIST_PROJECT_ID}}`
   Headers: `Authorization: Bearer {{TODOIST_API_TOKEN}}`
4. **Code node** — compose message:
   ```js
   const weather = $('HTTP Request').item.json;
   const tasks = $('HTTP Request1').item.json;
   const temp = weather.current_weather.temperature;
   const count = Array.isArray(tasks) ? tasks.length : 0;
   const today = new Date().toLocaleDateString('en-GB', {weekday:'long', day:'numeric', month:'long'});
   return [{json: {
     message: `Good morning Oleksii! Today is ${today}. The weather in Wrocław is ${temp}°C. You have ${count} item${count !== 1 ? 's' : ''} on your shopping list. Have a great day!`
   }}];
   ```
5. **HTTP Request** — POST `http://localhost:9876/speak`
   Body: `{"message": "{{$json.message}}"}`
   Headers: `Authorization: Bearer change-me`

**Save workflow export:** `docs/n8n_workflows/morning_briefing.json`
(Export from n8n → Download as JSON, save to this path for future reference)

---

### Feature 4 — Printer Tool (~2h)

**What:** Two new tools — `print_recipe(recipe_name)` and `print_shopping_list()` — that send
formatted output to the Canon PIXMA registered in CUPS as `Raspberry_pi_enabled_Canon_printserver`.

**New file:** `src/tools/printer.py`

```python
import subprocess
import logging
from src.config import load_config

log = logging.getLogger(__name__)

def _printer_name() -> str:
    cfg = load_config()
    return cfg.get("tools", {}).get("printer", {}).get(
        "queue_name", "Raspberry_pi_enabled_Canon_printserver"
    )

def _lp(text: str, title: str) -> str:
    proc = subprocess.run(
        ["lp", "-d", _printer_name(), "-t", title, "-"],
        input=text.encode(),
        capture_output=True,
        timeout=10,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode().strip()
        log.error(f"Printer error: {err}")
        return f"ERROR: Printer failed — {err}"
    return f"Sent '{title}' to printer."

def print_recipe(recipe_name: str) -> str:
    from src.tools.recipes import get_recipe
    text = get_recipe(recipe_name)
    if text.startswith("ERROR") or text.startswith("Recipe '"):
        return text
    return _lp(text, recipe_name)

def print_shopping_list() -> str:
    from src.tools.executor import get_shopping_list
    text = get_shopping_list()
    if text.startswith("ERROR"):
        return text
    return _lp(text, "Shopping List")
```

**Config additions in `config/config.yaml`:**
```yaml
tools:
  printer:
    queue_name: "Raspberry_pi_enabled_Canon_printserver"
```

**Wire-up in `src/tools/executor.py`:**
- Import `print_recipe, print_shopping_list` from `src.tools.printer`
- Add schemas `PRINT_RECIPE_TOOL` and `PRINT_SHOPPING_LIST_TOOL` to `ALL_TOOLS`
- Add both to `_TOOL_FUNCTIONS` dict

**Wire-up in `src/main.py`:**
- Add `"print_recipe"` and `"print_shopping_list"` to `ACTION_TOOLS` frozenset

**Potential circular import:** `print_shopping_list` calls `get_shopping_list` from `executor.py`,
but `executor.py` imports from `printer.py`. Resolve by importing `get_shopping_list`'s HTTP logic
directly in `printer.py` rather than via `executor.py`. Specifically:
- Copy the Todoist GET logic into printer.py as a private helper, OR
- Move `get_shopping_list` to its own module `src/tools/todoist.py` first

**Verification:**
```bash
# Test printer connectivity first
lp -d Raspberry_pi_enabled_Canon_printserver -t "test" - <<< "Hello from voice assistant"
lpq -P Raspberry_pi_enabled_Canon_printserver

# Then test via voice
# Say: "print my shopping list"
# Say: "print the carbonara recipe"
```

---

## Implementation order

1. Feature 1 (n8n outbound) — self-contained, no new deps, lowest risk
2. Feature 4 (printer) — self-contained, note circular import issue above
3. Feature 2 (HTTP API server) — new deps (fastapi, uvicorn), FSM changes
4. Feature 3 (morning briefing) — n8n config only, after Feature 2 is verified

## Files to create/modify

| File | Action |
|------|--------|
| `src/tools/automation.py` | CREATE |
| `src/tools/printer.py` | CREATE |
| `src/api/__init__.py` | CREATE (empty) |
| `src/api/server.py` | CREATE |
| `src/tools/executor.py` | MODIFY — add 4 new tools + imports |
| `src/main.py` | MODIFY — ACTION_TOOLS, push_queue, API startup |
| `src/state_machine.py` | MODIFY — push_queue param + IDLE heartbeat check |
| `config/config.yaml` | MODIFY — add api, tools.n8n, tools.printer sections |
| `pyproject.toml` | MODIFY — add fastapi, uvicorn |
| `docs/n8n_workflows/morning_briefing.json` | CREATE (exported from n8n) |
