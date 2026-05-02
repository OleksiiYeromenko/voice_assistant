"""Two-column recipe display for the 800×480 RPi Touch Display."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.ui import theme

_TITLE_SIZE = 22
_SECTION_SIZE = 14
_ITEM_SIZE = 18

_LIST_STYLE = (
    f"QListWidget {{"
    f"  background-color: {theme.BG_SURFACE};"
    f"  color: {theme.RETRO_TEXT_PRIMARY};"
    f"  font-family: 'DejaVu Sans';"
    f"  font-size: {_ITEM_SIZE}px;"
    f"  border: 1px solid {theme.RETRO_BORDER};"
    f"  padding: 4px;"
    f"}}"
    f"QListWidget::item {{ padding: 3px 6px; }}"
    f"QScrollBar:vertical {{ width: 6px; background: {theme.BG_ELEVATED}; }}"
    f"QScrollBar::handle:vertical {{"
    f"  background: {theme.RETRO_BORDER}; border-radius: 3px; }}"
)

_TEXT_STYLE = (
    f"QTextEdit {{"
    f"  background-color: {theme.BG_SURFACE};"
    f"  color: {theme.RETRO_TEXT_PRIMARY};"
    f"  font-family: 'DejaVu Sans';"
    f"  font-size: {_ITEM_SIZE}px;"
    f"  border: 1px solid {theme.RETRO_BORDER};"
    f"  padding: 4px 6px;"
    f"}}"
    f"QScrollBar:vertical {{ width: 6px; background: {theme.BG_ELEVATED}; }}"
    f"QScrollBar::handle:vertical {{"
    f"  background: {theme.RETRO_BORDER}; border-radius: 3px; }}"
)


class RecipeView(QWidget):
    """Full-screen two-column recipe card: ingredients (left, 1/3) | instructions (right, 2/3).

    Call load_recipe(data) to populate where data = {title, ingredients, instructions}.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {theme.BG_DEEPEST};")

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(8)

        # Title bar
        self._title = QLabel("", self)
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setWordWrap(True)
        self._title.setStyleSheet(
            f"color: {theme.RETRO_ACCENT};"
            f"font-family: 'Press Start 2P', 'DejaVu Sans';"
            f"font-size: {_TITLE_SIZE}px;"
            f"padding: 4px 0;"
        )
        root.addWidget(self._title)

        # Thin separator
        sep = QFrame(self)
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {theme.RETRO_BORDER}; border: none;")
        root.addWidget(sep)

        # Two-column area: ingredients (1/3) | instructions (2/3)
        cols = QHBoxLayout()
        cols.setContentsMargins(0, 0, 0, 0)
        cols.setSpacing(8)

        self._ingredients_list = self._make_ingredients_col(cols)
        cols.addWidget(self._make_divider())
        self._instructions_text = self._make_instructions_col(cols)

        root.addLayout(cols, stretch=1)

    def _make_section_header(self, text: str, parent: QWidget) -> QLabel:
        lbl = QLabel(text, parent)
        lbl.setStyleSheet(
            f"color: {theme.RETRO_TEXT_SECONDARY};"
            f"font-family: 'Press Start 2P', 'DejaVu Sans';"
            f"font-size: {_SECTION_SIZE}px;"
            f"padding-bottom: 2px;"
        )
        return lbl

    def _make_ingredients_col(self, parent_layout: QHBoxLayout) -> QListWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        layout.addWidget(self._make_section_header("INGREDIENTS", container))

        lst = QListWidget(container)
        lst.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        lst.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        lst.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        lst.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lst.setStyleSheet(_LIST_STYLE)
        layout.addWidget(lst, stretch=1)

        parent_layout.addWidget(container, stretch=1)
        return lst

    def _make_instructions_col(self, parent_layout: QHBoxLayout) -> QTextEdit:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        layout.addWidget(self._make_section_header("INSTRUCTIONS", container))

        txt = QTextEdit(container)
        txt.setReadOnly(True)
        txt.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        txt.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        txt.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        txt.setStyleSheet(_TEXT_STYLE)
        layout.addWidget(txt, stretch=1)

        parent_layout.addWidget(container, stretch=2)
        return txt

    def _make_divider(self) -> QFrame:
        div = QFrame(self)
        div.setFrameShape(QFrame.Shape.VLine)
        div.setFixedWidth(1)
        div.setStyleSheet(f"background-color: {theme.RETRO_BORDER}; border: none;")
        return div

    def load_recipe(self, data: dict):
        """Populate the view with recipe data dict {title, ingredients, instructions}."""
        self._title.setText(data.get("title", ""))

        self._ingredients_list.clear()
        for item in data.get("ingredients", []):
            self._ingredients_list.addItem(QListWidgetItem(f"• {item}"))

        steps = data.get("instructions", [])
        self._instructions_text.setPlainText(
            "\n\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))
        )
