"""Persistent memory — markdown files for personality/profile/facts, SQLite for sessions.

Architecture (CLAUDE.md-inspired):
  memory/PERSONA.md  — Assistant identity, rules, tool instructions (read-only at runtime)
  memory/PROFILE.md  — User preferences as key-value pairs (written by remember tool)
  memory/FACTS.md    — Known facts about the user (appended by remember tool)
  data/memory.db     — Session summaries only (SQLite, WAL mode)

On first run, migrates data from old SQLite preferences/facts tables if present.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_MEMORY_DIR = Path("memory")
DEFAULT_DB_PATH = Path("data/memory.db")
MAX_PROMPT_FACTS = 10
MAX_STORED_FACTS = 100
MAX_STORED_SESSIONS = 50

# ---------------------------------------------------------------------------
# Regex patterns for classifying remembered text as preference vs fact
# ---------------------------------------------------------------------------

# Name detection: "my name is Alex", "call me Alex"
_NAME_PATTERNS = [
    re.compile(r"\bmy name is (\w+)", re.I),
    re.compile(r"\bcall me (\w+)", re.I),
]

# Preference detection: (pattern, preference_key)
_PREFERENCE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Time format
    (re.compile(r"\b24[- ]?h(our)?\b.*\btime\b", re.I), "time_format"),
    (re.compile(r"\btime\b.*\b24[- ]?h(our)?\b", re.I), "time_format"),
    (re.compile(r"\b12[- ]?h(our)?\b.*\btime\b", re.I), "time_format"),
    (re.compile(r"\btime\b.*\b12[- ]?h(our)?\b", re.I), "time_format"),
    (re.compile(r"\bmilitary time\b", re.I), "time_format"),
    # Temperature units
    (re.compile(r"\b(celsius|centigrade)\b", re.I), "temperature_unit"),
    (re.compile(r"\bfahrenheit\b", re.I), "temperature_unit"),
    # Unit system
    (re.compile(r"\b(use|prefer|always)\b.*\b(metric|imperial)\b", re.I), "unit_system"),
    (re.compile(r"\b(use|prefer|always)\b.*\b(kilometers?|miles)\b", re.I), "unit_system"),
    # Generic "always/never/prefer" — catch-all for unrecognized preferences
    (re.compile(r"^(always|never|prefer)\b", re.I), "_generic_preference"),
]

# Lookup table: preference key → {value_substring → imperative instruction}
_PREFERENCE_INSTRUCTIONS: dict[str, dict[str, str]] = {
    "time_format": {
        "24": "ALWAYS use 24-hour time format (e.g., 14:30, not 2:30 PM)",
        "12": "ALWAYS use 12-hour AM/PM time format",
        "military": "ALWAYS use 24-hour time format (e.g., 14:30, not 2:30 PM)",
    },
    "temperature_unit": {
        "celsius": "ALWAYS show temperatures in Celsius",
        "centigrade": "ALWAYS show temperatures in Celsius",
        "fahrenheit": "ALWAYS show temperatures in Fahrenheit",
    },
    "unit_system": {
        "metric": "ALWAYS use metric units (km, kg, etc.)",
        "imperial": "ALWAYS use imperial units (miles, lbs, etc.)",
        "kilometer": "ALWAYS use metric units (km, kg, etc.)",
        "mile": "ALWAYS use imperial units (miles, lbs, etc.)",
    },
}

_SESSIONS_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    summary    TEXT,
    topics     TEXT,
    started_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f','now','localtime')),
    ended_at   TEXT,
    status     TEXT NOT NULL DEFAULT 'open'
);
"""


class MarkdownMemoryStore:
    """Markdown-file + SQLite memory store.

    .md files hold personality, user profile, and facts (human-readable, git-friendly).
    SQLite holds session data only.
    """

    def __init__(
        self,
        memory_dir: Path = DEFAULT_MEMORY_DIR,
        db_path: Path = DEFAULT_DB_PATH,
    ):
        self._memory_dir = Path(memory_dir)
        self._persona_path = self._memory_dir / "PERSONA.md"
        self._profile_path = self._memory_dir / "PROFILE.md"
        self._facts_path = self._memory_dir / "FACTS.md"

        # Ensure directories exist
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Load persona (cached, immutable at runtime)
        if self._persona_path.exists():
            self._persona = self._persona_path.read_text().strip()
        else:
            self._persona = ""
            log.warning(f"Persona file not found: {self._persona_path}")

        # SQLite for sessions only
        self._conn = self._connect(db_path)
        self._create_tables()

    # ------------------------------------------------------------------
    # Internal: SQLite connection
    # ------------------------------------------------------------------
    def _connect(self, db_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        self._conn.executescript(_SESSIONS_SQL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # PROFILE.md — user preferences (key-value)
    # ------------------------------------------------------------------
    def _read_profile(self) -> dict[str, str]:
        """Parse PROFILE.md into a dict of key-value pairs."""
        if not self._profile_path.exists():
            return {}

        data: dict[str, str] = {}
        for line in self._profile_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("- ") and ":" in line:
                key, _, value = line[2:].partition(":")
                data[key.strip()] = value.strip()
        return data

    def _write_profile(self, data: dict[str, str]):
        """Atomic write of profile data to PROFILE.md."""
        lines = ["# User Profile\n"]
        for key, val in data.items():
            lines.append(f"- {key}: {val}")
        content = "\n".join(lines) + "\n"
        self._atomic_write(self._profile_path, content)
        log.info(f"Updated PROFILE.md ({len(data)} entries)")

    def _set_preference(self, key: str, value: str):
        """Update a single preference in PROFILE.md."""
        profile = self._read_profile()
        profile[key] = value
        self._write_profile(profile)

    # ------------------------------------------------------------------
    # FACTS.md — known facts about the user
    # ------------------------------------------------------------------
    def _read_all_facts(self) -> list[str]:
        """Read all fact lines from FACTS.md."""
        if not self._facts_path.exists():
            return []

        facts = []
        for line in self._facts_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("- "):
                facts.append(line[2:])
        return facts

    def _read_recent_facts(self, limit: int = MAX_PROMPT_FACTS) -> list[str]:
        """Read the most recent N facts from FACTS.md."""
        all_facts = self._read_all_facts()
        return all_facts[-limit:]

    def _append_fact(self, text: str) -> str:
        """Append a fact to FACTS.md with today's date."""
        date = datetime.now().strftime("%Y-%m-%d")
        line = f"- {text} ({date})\n"

        # Ensure file exists with header
        if not self._facts_path.exists():
            self._facts_path.write_text("# Known Facts\n\n")

        with open(self._facts_path, "a") as f:
            f.write(line)

        log.info(f"Memory: stored fact '{text}'")

        # Prune if over limit
        all_facts = self._read_all_facts()
        if len(all_facts) > MAX_STORED_FACTS:
            pruned = all_facts[-MAX_STORED_FACTS:]
            content = "# Known Facts\n\n" + "\n".join(f"- {f}" for f in pruned) + "\n"
            self._atomic_write(self._facts_path, content)
            log.info(f"Pruned FACTS.md to {MAX_STORED_FACTS} entries")

        return f"I'll remember that: {text}"

    # ------------------------------------------------------------------
    # Atomic file write (crash-safe on Pi)
    # ------------------------------------------------------------------
    @staticmethod
    def _atomic_write(path: Path, content: str):
        """Write content to a file atomically using temp file + rename."""
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
            os.replace(tmp_path, path)
        except BaseException:
            os.unlink(tmp_path)
            raise

    # ------------------------------------------------------------------
    # Sessions (SQLite — same as before)
    # ------------------------------------------------------------------
    def open_session(self) -> int:
        """Open a new session. Returns session ID."""
        cur = self._conn.execute("INSERT INTO sessions (status) VALUES ('open')")
        self._conn.commit()
        log.info(f"Opened session {cur.lastrowid}")
        return cur.lastrowid

    def close_session(
        self,
        session_id: int,
        summary: str | None = None,
        topics: str | None = None,
    ):
        """Close a session with optional summary and topics."""
        self._conn.execute(
            "UPDATE sessions SET status='closed', summary=?, topics=?, ended_at=? WHERE id=?",
            (summary, topics, datetime.now().isoformat(), session_id),
        )
        self._conn.commit()
        log.info(f"Session {session_id} closed. Summary: {summary}")

    def close_stale_sessions(self):
        """Close any sessions left open from a previous run (clean startup)."""
        rows = self._conn.execute(
            "SELECT id FROM sessions WHERE status = 'open'"
        ).fetchall()
        for row in rows:
            self.close_session(row["id"], summary=None)
            log.info(f"Closed stale session {row['id']} from previous run")

    def get_latest_summary(self) -> dict | None:
        """Get the most recent closed session with a summary."""
        row = self._conn.execute(
            "SELECT summary, topics, ended_at FROM sessions "
            "WHERE status='closed' AND summary IS NOT NULL "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def search_sessions(self, query: str, limit: int = 5) -> list[dict]:
        """Search session summaries and topics for keywords."""
        rows = self._conn.execute(
            "SELECT summary, topics, ended_at FROM sessions "
            "WHERE status='closed' AND summary IS NOT NULL "
            "AND (summary LIKE ? OR topics LIKE ?) "
            "ORDER BY id DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Tool: remember — classify and store preference or fact
    # ------------------------------------------------------------------
    def remember(self, text: str) -> str:
        """Classify text as name, preference, or fact, then store accordingly.

        Called by the 'remember' tool. Uses zero-cost regex patterns to detect
        preferences before falling back to generic fact storage.
        """
        # 1. Check for name
        for pattern in _NAME_PATTERNS:
            m = pattern.search(text)
            if m:
                name = m.group(1).strip().title()
                self._set_preference("user_name", name)
                return f"Got it, I'll remember your name is {name}."

        # 2. Check for preference
        for pattern, pref_key in _PREFERENCE_PATTERNS:
            if pattern.search(text):
                if pref_key == "_generic_preference":
                    pref_key = f"pref_{abs(hash(text.lower().strip())) % 10000}"
                self._set_preference(pref_key, text)
                return f"Preference saved: {text}"

        # 3. Default: store as a fact
        return self._append_fact(text)

    # ------------------------------------------------------------------
    # Tool: recall — search past sessions OR list user profile
    # ------------------------------------------------------------------
    def recall(self, query: str = "") -> str:
        """Search past sessions or list all known user info.

        Called by the 'recall' tool.
        - Empty query or 'profile'/'all'/'me' → list all preferences + facts
        - Keyword query → search past session summaries
        """
        q = query.strip().lower()

        if not q or q in ("profile", "all", "me", "my profile", "preferences"):
            return self._format_full_memory()

        # Search past sessions
        results = self.search_sessions(query, limit=5)
        if not results:
            return "I don't have any past conversations matching that query."

        lines = []
        for r in results:
            date = (r.get("ended_at") or "")[:10]
            summary = r.get("summary", "")
            topics = r.get("topics", "")
            entry = f"[{date}] {summary}"
            if topics:
                entry += f" (topics: {topics})"
            lines.append(entry)

        return "Past conversations:\n" + "\n".join(lines)

    def _format_full_memory(self) -> str:
        """Format all stored preferences and facts for voice output."""
        parts: list[str] = []

        profile = self._read_profile()
        if profile:
            parts.append("Your preferences:")
            for key, val in profile.items():
                label = key.replace("_", " ")
                parts.append(f"- {label}: {val}")

        facts = self._read_all_facts()
        if facts:
            parts.append("Things I remember about you:")
            for f in facts:
                parts.append(f"- {f}")

        if not parts:
            return "I don't have any stored preferences or facts about you yet."

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # System prompt assembly
    # ------------------------------------------------------------------
    def build_system_prompt(self) -> str:
        """Assemble the full system prompt from .md files + session data.

        Token budget: ~370 tokens total
          PERSONA.md:  ~200 tokens
          Profile:     ~40 tokens (preferences as imperatives)
          Facts:       ~100 tokens (last 10)
          Session:     ~30 tokens (latest summary)
        """
        parts = [self._persona]

        # User profile as imperative instructions
        profile = self._read_profile()
        user_name = profile.pop("user_name", None)
        user_lines: list[str] = []
        if user_name:
            user_lines.append(f"User's name: {user_name}")
        if profile:
            user_lines.append("User preferences (ALWAYS follow these):")
            for key, val in profile.items():
                instruction = self._preference_to_instruction(key, val)
                user_lines.append(f"- {instruction}")
        if user_lines:
            parts.append("\n## User\n" + "\n".join(user_lines))

        # Known facts (last N)
        facts = self._read_recent_facts()
        if facts:
            fact_lines = "\n".join(f"- {f}" for f in facts)
            parts.append(f"\n## Known Facts\n{fact_lines}")

        # Latest session summary
        latest = self.get_latest_summary()
        if latest and latest.get("summary"):
            parts.append(f"\n## Last Conversation\n{latest['summary']}")

        return "\n".join(parts)

    @staticmethod
    def _preference_to_instruction(key: str, value: str) -> str:
        """Convert a stored preference into an imperative system prompt instruction."""
        if key in _PREFERENCE_INSTRUCTIONS:
            val_lower = value.lower()
            for sub_key, instruction in _PREFERENCE_INSTRUCTIONS[key].items():
                if sub_key in val_lower:
                    return instruction

        if value.lower().startswith(("always", "never")):
            return value
        return f"ALWAYS: {value}"
