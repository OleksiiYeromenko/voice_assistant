"""Minimal PyQt6 display test for Raspberry Pi Touch Display 2.

Run with:
    uv run python scripts/test_display.py
    uv run python scripts/test_display.py --windowed   # keep as window
"""

import sys

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget


def main():
    app = QApplication(sys.argv)

    win = QWidget()
    win.setStyleSheet("background-color: #0D0D1A;")

    layout = QVBoxLayout(win)
    layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.setSpacing(20)

    title = QLabel("PyQt6 Display Test")
    title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    title.setStyleSheet("color: #4FC3F7; font-size: 28px; font-weight: bold;")

    status = QLabel("IDLE")
    status.setAlignment(Qt.AlignmentFlag.AlignCenter)
    status.setStyleSheet(
        "color: #69F0AE; background-color: #1B5E20; font-size: 20px; "
        "font-weight: bold; border-radius: 12px; padding: 8px 24px;"
    )

    info = QLabel("If you can read this, PyQt6 is working!")
    info.setAlignment(Qt.AlignmentFlag.AlignCenter)
    info.setStyleSheet("color: #9090A0; font-size: 16px;")

    counter_label = QLabel("Tick: 0")
    counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    counter_label.setStyleSheet("color: #FFB300; font-family: 'DejaVu Sans Mono'; font-size: 16px;")

    layout.addWidget(title)
    layout.addWidget(status)
    layout.addWidget(info)
    layout.addWidget(counter_label)

    # Cycle state colors to verify timer + stylesheet updates work
    states = [
        ("IDLE",        "#2D3748", "#718096"),
        ("LISTENING",   "#1B5E20", "#69F0AE"),
        ("THINKING",    "#4A3000", "#FFB300"),
        ("INTERRUPTED", "#4A1500", "#FF7043"),
    ]
    tick = [0]

    def on_tick():
        tick[0] += 1
        s = states[tick[0] % len(states)]
        status.setText(s[0])
        status.setStyleSheet(
            f"color: {s[2]}; background-color: {s[1]}; font-size: 20px; "
            f"font-weight: bold; border-radius: 12px; padding: 8px 24px;"
        )
        counter_label.setText(f"Tick: {tick[0]}")

    timer = QTimer()
    timer.setInterval(1000)
    timer.timeout.connect(on_tick)
    timer.start()

    if "--windowed" in sys.argv:
        win.resize(800, 480)
        win.show()
    else:
        win.showFullScreen()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
