"""PyQt6 application: MainWindow, FSMWorker, and run_ui() entry point.

Thread model:
  Main thread  — QApplication + all widgets (Qt requirement)
  Worker thread — FSMWorker(QThread) runs AssistantFSM.run() until shutdown

Signals from UIEventBus are automatically queued across thread boundaries,
so no manual locking is required.
"""

import logging
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import QEvent, Qt, QThread, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.tools.player import register_radio_callback
from src.ui.signals import UIEventBus
from src.ui.theme import MAIN_STYLESHEET
from src.ui.widgets.chat_view import ChatView
from src.ui.widgets.idle_screen import IdleScreen
from src.ui.widgets.state_bar import StateBar
from src.ui.widgets.sys_bar import SysBar

log = logging.getLogger(__name__)

_RESOURCE_POLL_INTERVAL_MS = 5_000
_IDLE_SWITCH_DELAY_MS = 20_000   # stay on active screen 20s after returning to IDLE
_BACKLIGHT_ROOT = Path("/sys/class/backlight")

_IDLE_STATES = {"IDLE", "SESSION_CHECK"}


class BacklightController:
    """Controls the RPi Touch Display backlight via sysfs.

    Auto-detects the first device under /sys/class/backlight/. Silently no-ops
    on platforms without a backlight (dev laptops, --windowed mode).
    """

    def __init__(self, dim_level: int):
        self._brightness_path: Path | None = None
        self._normal: int | None = None
        self._dim_level = max(0, dim_level)

        try:
            device = next(_BACKLIGHT_ROOT.iterdir(), None) if _BACKLIGHT_ROOT.exists() else None
            if device is None:
                return
            self._brightness_path = device / "brightness"
            # Use max_brightness as "on" level. Reading current brightness is unsafe:
            # if a prior run was killed while dimmed, sysfs still holds the dim value.
            max_path = device / "max_brightness"
            self._normal = int(max_path.read_text().strip())
        except (OSError, ValueError) as exc:
            log.debug(f"Backlight detect failed: {exc}")
            self._brightness_path = None

        # Own the display state: restore full brightness on startup regardless
        # of what we inherited from a previous process.
        self.restore()

    def _write(self, value: int):
        if self._brightness_path is None:
            return
        try:
            self._brightness_path.write_text(f"{value}\n")
        except OSError as exc:
            log.warning(f"Backlight write failed ({value}): {exc}")

    def dim(self):
        self._write(self._dim_level)

    def restore(self):
        if self._normal is not None:
            self._write(self._normal)


def _wake_screen():
    """Un-blank the X display when the wake word fires."""
    try:
        subprocess.run(
            ["xset", "-display", ":0", "dpms", "force", "on"],
            check=False, timeout=1, capture_output=True,
        )
    except Exception:
        pass  # xset not available or display already on — ignore


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
    """800×480 fullscreen window for Raspberry Pi Touch Display 2.

    Uses a QStackedWidget to switch between two layouts:
      Page 0 — IdleScreen: ambient clock + date + slim sys strip
      Page 1 — Active:     StateBar / ChatView / SysBar
    """

    def __init__(self, bus: UIEventBus, dim_after_s: int = 60, dim_level: int = 0):
        super().__init__()
        self._bus = bus
        self.setWindowTitle("Voice Assistant")
        self.setStyleSheet(MAIN_STYLESHEET)
        self._backlight = BacklightController(dim_level=dim_level)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Stacked widget ────────────────────────────────────────────────────
        self._stack = QStackedWidget(central)
        layout.addWidget(self._stack)

        # Page 0 — idle ambient screen
        self._idle_screen = IdleScreen()
        self._stack.addWidget(self._idle_screen)

        # Page 1 — active conversation panels
        active = QWidget()
        active_layout = QVBoxLayout(active)
        active_layout.setContentsMargins(0, 0, 0, 0)
        active_layout.setSpacing(0)

        self._state_bar = StateBar(bus, active)
        self._chat_view = ChatView(bus, active)
        self._sys_bar = SysBar(bus, active)

        active_layout.addWidget(self._state_bar)
        active_layout.addWidget(self._chat_view, stretch=1)
        active_layout.addWidget(self._sys_bar)

        self._stack.addWidget(active)

        # Start on idle screen
        self._stack.setCurrentIndex(0)

        # ── Idle-switch delay timer ───────────────────────────────────────────
        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.setInterval(_IDLE_SWITCH_DELAY_MS)
        self._idle_timer.timeout.connect(lambda: self._stack.setCurrentIndex(0))

        # ── Backlight dim timer ───────────────────────────────────────────────
        self._dim_timer = QTimer(self)
        self._dim_timer.setSingleShot(True)
        self._dim_after_s = max(0, dim_after_s)
        if self._dim_after_s > 0:
            self._dim_timer.setInterval(self._dim_after_s * 1000)
            self._dim_timer.timeout.connect(self._backlight.dim)

        # Accept touch events so event() receives TouchBegin/TouchUpdate
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)

        # ── Signal connections ────────────────────────────────────────────────
        bus.state_changed.connect(self._on_state_changed)
        bus.resource_updated.connect(self._on_resource_updated)
        bus.shutdown_requested.connect(self._on_shutdown)
        bus.radio_changed.connect(self._idle_screen.on_radio_changed)

        # ── Resource polling ──────────────────────────────────────────────────
        self._res_timer = QTimer(self)
        self._res_timer.setInterval(_RESOURCE_POLL_INTERVAL_MS)
        self._res_timer.timeout.connect(self._poll_resources)
        self._res_timer.start()

        # FSM starts in IDLE but emits state_changed only on transitions, so
        # arm the dim timer here — otherwise the screen never dims until the
        # user invokes the assistant at least once.
        if self._dim_after_s > 0:
            self._dim_timer.start()

    # ── Touch / mouse wake ─────────────────────────────────────────────
    def event(self, ev: QEvent) -> bool:
        """Restore backlight on any touch or mouse interaction."""
        if ev.type() in (
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.TouchBegin,
            QEvent.Type.TouchUpdate,
        ):
            self._wake_backlight()
        return super().event(ev)

    def _wake_backlight(self):
        """Restore backlight and restart the dim timer."""
        self._backlight.restore()
        if self._dim_after_s > 0:
            self._dim_timer.start()  # restart countdown
        _wake_screen()

    def _on_state_changed(self, state_name: str):
        self._idle_screen.on_state_changed(state_name)
        if state_name in _IDLE_STATES:
            self._idle_timer.start()        # switch to clock after delay
            if self._dim_after_s > 0:
                self._dim_timer.start()     # dim backlight after longer delay
        else:
            self._idle_timer.stop()         # cancel pending idle switch
            self._dim_timer.stop()          # cancel pending dim
            self._backlight.restore()
            self._stack.setCurrentIndex(1)  # show active immediately
            _wake_screen()

    def _on_resource_updated(
        self,
        cpu_pct: float,
        ram_used: float,
        ram_total: float,
        temp: object,
    ):
        self._idle_screen.update_resources(cpu_pct, ram_used, ram_total, temp)

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
        self._dim_timer.stop()
        self._backlight.restore()
        self.close()

    def show_fullscreen_rpi(self):
        """Fullscreen on RPi; fixed 800×480 window in --windowed dev mode."""
        if "--windowed" in sys.argv:
            self.setFixedSize(800, 480)
            self.show()
        else:
            self.showFullScreen()


def run_ui(fsm) -> int:
    """Launch the Qt UI and run the FSM in a worker thread.

    Call this from assistant_loop() when --ui is in sys.argv.
    Returns the QApplication exit code for sys.exit().
    """
    # Guard: --text mode uses input() which blocks the worker thread.
    if "--text" in sys.argv:
        log.warning("--text mode is not compatible with --ui; ignoring --text")
        sys.argv.remove("--text")

    # Ensure display env vars are set when running from SSH
    import os
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        os.environ["DISPLAY"] = ":0"
    if not os.environ.get("XDG_RUNTIME_DIR"):
        os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    app = QApplication(sys.argv)
    app.setApplicationName("VoiceAssistant")
    app.setOverrideCursor(Qt.CursorShape.BlankCursor)

    bus = UIEventBus()
    fsm.ui_bus = bus

    register_radio_callback(lambda station: bus.radio_changed.emit(station))

    display_cfg = {}
    try:
        from src.config import load_config
        display_cfg = load_config().get("display") or {}
    except Exception as exc:
        log.debug(f"Display config load failed, using defaults: {exc}")

    window = MainWindow(
        bus,
        dim_after_s=int(display_cfg.get("dim_after_idle_s", 60)),
        dim_level=int(display_cfg.get("dim_min_level", 0)),
    )
    window.show_fullscreen_rpi()

    worker = FSMWorker(fsm, bus)
    worker.start()

    exit_code = app.exec()
    worker.wait(3000)

    return exit_code
