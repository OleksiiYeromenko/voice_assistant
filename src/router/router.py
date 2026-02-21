"""Model router — picks the right LLM backend for each query.

Priority:
  1. Explicit user trigger ("use claude to...")
  2. Session preference ("switch to gemini" persists until reset)
  3. Automatic rules (default: local)
  4. Fallback: if chosen backend fails, fall back to local
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

from src.llm.backends import LLMBackend, ClaudeBackend, GeminiBackend, OllamaBackend

log = logging.getLogger(__name__)

DEFAULT_TRIGGERS: dict[str, list[str]] = {
    "claude": ["use claude", "ask claude", "switch to claude", "with claude"],
    "gemini": ["use gemini", "ask gemini", "switch to gemini", "with gemini"],
    "local": ["use local", "go offline", "use ollama", "go back to automatic"],
}


@dataclass
class RouteDecision:
    backend_key: str        # "local", "claude", "gemini"
    reason: str             # Why this backend was chosen
    cleaned_text: str       # User text with trigger phrase removed


class ModelRouter:
    def __init__(
        self,
        backends: dict[str, LLMBackend],
        triggers: dict[str, list[str]] | None = None,
    ):
        self.backends = backends
        self.triggers = triggers or DEFAULT_TRIGGERS
        self.session_preference: str | None = None  # Sticky preference

    def route(self, text: str) -> RouteDecision:
        """Decide which backend to use for this text."""
        text_lower = text.lower().strip()

        # 1. Check explicit triggers
        for key, patterns in self.triggers.items():
            for pattern in patterns:
                if pattern in text_lower:
                    # "switch to X" sets session preference
                    if "switch to" in pattern:
                        if key == "local":
                            self.session_preference = None
                            log.info("Session preference cleared → automatic routing")
                        else:
                            self.session_preference = key
                            log.info(f"Session preference set → {key}")

                    # Remove trigger phrase from the text
                    cleaned = re.sub(re.escape(pattern), "", text_lower, count=1).strip()
                    # Capitalize first letter if needed
                    cleaned = cleaned[0].upper() + cleaned[1:] if cleaned else text

                    # "go back to automatic" resets preference
                    if "automatic" in pattern or "offline" in pattern:
                        self.session_preference = None

                    backend_key = key if key != "local" else "local"
                    return RouteDecision(
                        backend_key=backend_key,
                        reason=f"explicit trigger: '{pattern}'",
                        cleaned_text=cleaned if cleaned else text,
                    )

        # 2. Session preference
        if self.session_preference and self.session_preference in self.backends:
            return RouteDecision(
                backend_key=self.session_preference,
                reason=f"session preference: {self.session_preference}",
                cleaned_text=text,
            )

        # 3. Default: local
        return RouteDecision(
            backend_key="local",
            reason="default → local",
            cleaned_text=text,
        )

    def get_backend(self, key: str) -> LLMBackend:
        """Get backend by key, with fallback to local."""
        if key in self.backends:
            return self.backends[key]
        log.warning(f"Backend '{key}' not available, falling back to local")
        return self.backends["local"]

    def get_fallback(self, failed_key: str) -> LLMBackend | None:
        """Get fallback backend when primary fails."""
        if failed_key != "local" and "local" in self.backends:
            log.info(f"Falling back from {failed_key} → local")
            return self.backends["local"]
        return None