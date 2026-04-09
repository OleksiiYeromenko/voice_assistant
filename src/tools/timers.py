"""Timer management for the voice assistant.

Timers run in background threads. When a timer fires, it calls the registered
alert callback (wired to tts.speak at startup) to speak the alert aloud.
"""

import logging
import threading
from collections.abc import Callable

log = logging.getLogger(__name__)

_active_timers: dict[str, threading.Timer] = {}
_alert_callback: Callable[[str], None] | None = None


def register_alert_callback(fn: Callable[[str], None]) -> None:
    """Register the function called when a timer fires (e.g. tts.speak)."""
    global _alert_callback
    _alert_callback = fn


def set_timer(duration_seconds: int, label: str = "timer") -> str:
    """Schedule a spoken alert after duration_seconds. Returns confirmation string."""
    if label in _active_timers:
        _active_timers[label].cancel()

    def _fire():
        _active_timers.pop(label, None)
        msg = f"Time's up! Your {label} is done."
        log.info(f"Timer fired: {label}")
        if _alert_callback:
            _alert_callback(msg)

    t = threading.Timer(duration_seconds, _fire)
    t.daemon = True
    t.start()
    _active_timers[label] = t

    minutes, seconds = divmod(duration_seconds, 60)
    if minutes and seconds:
        human = (
            f"{minutes} minute{'s' if minutes > 1 else ''} "
            f"and {seconds} second{'s' if seconds > 1 else ''}"
        )
    elif minutes:
        human = f"{minutes} minute{'s' if minutes > 1 else ''}"
    else:
        human = f"{seconds} second{'s' if seconds > 1 else ''}"

    return f"Timer set: '{label}' will fire in {human}."


def cancel_timer(label: str = "timer") -> str:
    """Cancel an active timer by label."""
    t = _active_timers.pop(label, None)
    if t:
        t.cancel()
        return f"Timer '{label}' cancelled."
    return f"No active timer named '{label}'."


def list_timers() -> str:
    """Return a string listing all active timer labels."""
    if not _active_timers:
        return "No active timers."
    return "Active timers: " + ", ".join(_active_timers.keys()) + "."
