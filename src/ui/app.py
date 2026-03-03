"""PyQt6 application: MainWindow, FSMWorker, and run_ui() entry point.

Thread model:
  Main thread  — QApplication + all widgets (Qt requirement)
  Worker thread — FSMWorker(QThread) runs AssistantFSM.run() until shutdown

Signals from UIEventBus are automatically queued across thread boundaries,
so no manual locking is required.
"""

import logging
import sys

from PyQt6.QtCore import QThread, QTimer, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from src.ui.signals import UIEventBus
from src.ui.theme import MAIN_STYLESHEET
from src.ui.widgets.chat_view import ChatView
from src.ui.widgets.state_bar import StateBar
from src.ui.widgets.sys_bar import SysBar
from src.ui.widgets.tool_strip import ToolStrip

log = logging.getLogger(__name__)

_RESOURCE_POLL_INTERVAL_MS = 5_000


class FSMWorker(QThread):
    """Runs AssistantFSM.run() in a Qt worker thread.

    On exit (normal or exception), emits shutdown_requested so the window closes.
    """

    def __init__(self, fsm, bus: UIEventBus):
        super().__init__()
        self._fsm = fsm
        self._bus = bus

    def run(self):
        try:
            self._fsm.run()
        except Exception as exc:
            log.error(f"FSM crashed: {exc}", exc_info=True)
        finally:
            self._bus.shutdown_requested.emit()


class MainWindow(QMainWindow):
    """800×480 fullscreen window for Raspberry Pi Touch Display 2."""

    def __init__(self, bus: UIEventBus):
        super().__init__()
        self._bus = bus
        self.setWindowTitle("Voice Assistant")
        self.setStyleSheet(MAIN_STYLESHEET)

        # Central widget + vertical layout
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Assemble panels
        self._state_bar = StateBar(bus, central)
        self._chat_view = ChatView(bus, central)
        self._tool_strip = ToolStrip(bus, central)
        self._sys_bar = SysBar(bus, central)

        layout.addWidget(self._state_bar)
        layout.addWidget(self._chat_view, stretch=1)
        layout.addWidget(self._tool_strip)
        layout.addWidget(self._sys_bar)

        # Resource polling (runs in main thread — 100ms block every 5s is fine)
        self._res_timer = QTimer(self)
        self._res_timer.setInterval(_RESOURCE_POLL_INTERVAL_MS)
        self._res_timer.timeout.connect(self._poll_resources)
        self._res_timer.start()

        bus.shutdown_requested.connect(self._on_shutdown)

    def _poll_resources(self):
        try:
            from src.monitor import snapshot
            snap = snapshot()
            self._bus.resource_updated.emit(
                snap.cpu_percent,
                snap.ram_used_mb,
                snap.ram_total_mb,
                snap.temp_celsius,
            )
        except Exception as exc:
            log.debug(f"Resource poll error: {exc}")

    def _on_shutdown(self):
        self._res_timer.stop()
        self.close()

    def show_fullscreen_rpi(self):
        """Show fullscreen on RPi; windowed on dev machines (DISPLAY env present)."""
        import os
        if os.environ.get("DISPLAY") and "--windowed" in sys.argv:
            self.show()
        else:
            self.showFullScreen()


def run_ui(fsm) -> int:
    """Launch the Qt UI and run the FSM in a worker thread.

    Call this from assistant_loop() when --ui is in sys.argv.
    Returns the QApplication exit code for sys.exit().
    """
    # Guard: --text mode uses input() which blocks the worker thread.
    # In UI mode, keyboard/wake-word input is used instead.
    if "--text" in sys.argv:
        log.warning("--text mode is not compatible with --ui; ignoring --text")
        sys.argv.remove("--text")

    app = QApplication(sys.argv)
    app.setApplicationName("VoiceAssistant")

    bus = UIEventBus()

    # Inject the signal bus into the FSM before starting the worker
    fsm.ui_bus = bus

    window = MainWindow(bus)
    window.show_fullscreen_rpi()

    worker = FSMWorker(fsm, bus)
    worker.start()

    exit_code = app.exec()

    # Give the FSM thread a moment to finish cleanly
    worker.wait(3000)

    return exit_code
