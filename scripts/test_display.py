"""PyQt6 display diagnostic for Raspberry Pi.

Tries progressively simpler rendering to identify what's blocking display.

Usage:
    uv run python scripts/test_display.py            # windowed 800x480 (safe default)
    uv run python scripts/test_display.py --fs       # fullscreen
    uv run python scripts/test_display.py --nostyle  # no stylesheet (plain Qt)
"""

import os
import sys

# ── SSH / display env-var bootstrap (must run before Qt imports) ─────────────
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ["DISPLAY"] = ":0"
    print("[bootstrap] DISPLAY not set — defaulting to :0 (XWayland)")

if not os.environ.get("XDG_RUNTIME_DIR"):
    os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
    print(f"[bootstrap] XDG_RUNTIME_DIR set to {os.environ['XDG_RUNTIME_DIR']}")

# Force X11/XWayland to avoid 'propagateSizeHints' Wayland compositor warning
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# ── Print diagnostic info before Qt initializes ─────────────────────────────
print("=== Display Diagnostic ===")
print(f"Python:           {sys.version.split()[0]}")
print(f"DISPLAY:          {os.environ.get('DISPLAY', '<not set>')}")
print(f"WAYLAND_DISPLAY:  {os.environ.get('WAYLAND_DISPLAY', '<not set>')}")
print(f"QT_QPA_PLATFORM:  {os.environ.get('QT_QPA_PLATFORM', '<not set>')}")
print(f"XDG_SESSION_TYPE: {os.environ.get('XDG_SESSION_TYPE', '<not set>')}")
print()

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

app = QApplication(sys.argv)

from PyQt6.QtCore import QLibraryInfo
print(f"Qt version:       {app.platformName()} platform")
print(f"Qt libs:          {QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)}")
print()

# ── Build window ─────────────────────────────────────────────────────────────
win = QWidget()
win.setWindowTitle("RPi Display Test")

layout = QVBoxLayout(win)
layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
layout.setSpacing(16)

use_style = "--nostyle" not in sys.argv

if use_style:
    win.setStyleSheet("background-color: #0D0D1A;")

def lbl(text, color="#FFFFFF", size=20, bold=False):
    l = QLabel(text)
    l.setAlignment(Qt.AlignmentFlag.AlignCenter)
    if use_style:
        weight = "bold;" if bold else ""
        l.setStyleSheet(f"color: {color}; font-size: {size}px; {weight}")
    layout.addWidget(l)
    return l

lbl("RPi PyQt6 Display Test", "#4FC3F7", 26, bold=True)
lbl("If you see this, rendering works!", "#69F0AE", 18)
lbl(f"Platform: {app.platformName()}", "#FFB300", 15)
lbl(f"DISPLAY={os.environ.get('DISPLAY','?')}  WAYLAND={os.environ.get('WAYLAND_DISPLAY','?')}", "#9090A0", 13)

tick_label = lbl("Tick: 0", "#FF7043", 16)

tick = [0]
states = ["IDLE", "LISTENING", "THINKING"]
colors = ["#718096", "#69F0AE", "#FFB300"]

def on_tick():
    tick[0] += 1
    i = tick[0] % len(states)
    tick_label.setText(f"Tick: {tick[0]}  →  {states[i]}")
    if use_style:
        tick_label.setStyleSheet(f"color: {colors[i]}; font-size: 16px;")
    print(f"tick {tick[0]}", flush=True)

timer = QTimer()
timer.setInterval(1000)
timer.timeout.connect(on_tick)
timer.start()

# ── Show window ───────────────────────────────────────────────────────────────
if "--fs" in sys.argv:
    print("Showing fullscreen...")
    win.showFullScreen()
else:
    print("Showing windowed 800x480...")
    win.setFixedSize(800, 480)
    win.show()

print("Qt event loop starting — watch for 'tick N' output every second")
sys.exit(app.exec())
