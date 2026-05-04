"""Qt bridge for BackendHealthMonitor.

Subscribes to health updates from src.health.BackendHealthMonitor and re-emits
them as Qt signals so UI widgets can connect without touching threads directly.
"""
from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, pyqtSignal

from src.health import BackendStatus

log = logging.getLogger(__name__)


class BackendMonitor(QObject):
    """Thin Qt wrapper around BackendHealthMonitor.

    Pass the shared BackendHealthMonitor instance from main.py so there is
    a single polling source for both the router and the UI.
    """

    status_updated = pyqtSignal(list)   # list[BackendStatus]
    net_updated = pyqtSignal(bool)      # internet reachable (not used here, kept for compat)

    def __init__(self, health, parent=None):
        super().__init__(parent)
        self._health = health
        health.add_listener(self._on_update)
        # Emit current state immediately so the UI is populated before first poll completes
        self.status_updated.emit(health.get_statuses())

    def set_active(self, key: str) -> None:
        self._health.set_active(key)

    def stop(self) -> None:
        pass  # health monitor lifecycle managed by main.py

    def _on_update(self, statuses: list[BackendStatus]) -> None:
        # Called from a background thread — Qt auto-queues cross-thread signals.
        self.status_updated.emit(statuses)
        online = any(s.online for s in statuses if s.key != "local")
        self.net_updated.emit(online)
