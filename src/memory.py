"""Persistent memory — stores user facts and session summaries as JSON.

Data is stored in a simple JSON file that survives reboots.
Memory is injected into the system prompt so the LLM has context about the user.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/memory.json")
MAX_PROMPT_FACTS = 20
MAX_SESSION_SUMMARIES = 10


class MemoryStore:
    def __init__(self, path: Path = DEFAULT_PATH):
        self._path = path
        self._data = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                log.warning(f"Failed to load memory: {e}")
        return {
            "user_profile": {"name": None, "preferences": {}},
            "facts": [],
            "session_summaries": [],
        }

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2, default=str))

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------
    def add_fact(self, text: str, source: str = "user_requested") -> str:
        """Add a fact to long-term memory. Returns confirmation message."""
        self._data["facts"].append({
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "source": source,
        })
        self._save()
        log.info(f"Memory: stored fact '{text}'")
        return f"I'll remember that: {text}"

    def get_facts(self, limit: int = MAX_PROMPT_FACTS) -> list[dict]:
        """Get most recent facts."""
        return self._data["facts"][-limit:]

    # ------------------------------------------------------------------
    # User profile
    # ------------------------------------------------------------------
    def set_user_name(self, name: str):
        self._data["user_profile"]["name"] = name
        self._save()

    def set_preference(self, key: str, value: str):
        self._data["user_profile"]["preferences"][key] = value
        self._save()

    # ------------------------------------------------------------------
    # Session summaries
    # ------------------------------------------------------------------
    def add_session_summary(self, summary: str):
        self._data["session_summaries"].append({
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        })
        # Keep only the most recent summaries
        self._data["session_summaries"] = self._data["session_summaries"][-MAX_SESSION_SUMMARIES:]
        self._save()

    # ------------------------------------------------------------------
    # System prompt injection
    # ------------------------------------------------------------------
    def build_prompt_section(self) -> str:
        """Build a memory section to append to the system prompt.

        Returns an empty string if there is nothing to inject.
        Keeps output compact to fit within the context window (~300 tokens max).
        """
        parts: list[str] = []

        # User profile
        profile = self._data["user_profile"]
        if profile.get("name"):
            parts.append(f"User's name: {profile['name']}")
        for key, val in profile.get("preferences", {}).items():
            parts.append(f"User preference — {key}: {val}")

        # Known facts
        facts = self.get_facts()
        if facts:
            parts.append("Known facts about the user:")
            for f in facts:
                parts.append(f"- {f['text']}")

        # Recent session summaries (last 3)
        summaries = self._data.get("session_summaries", [])[-3:]
        if summaries:
            parts.append("Recent conversation summaries:")
            for s in summaries:
                parts.append(f"- {s['summary']}")

        if not parts:
            return ""
        return "\n\n## Memory\n" + "\n".join(parts)
