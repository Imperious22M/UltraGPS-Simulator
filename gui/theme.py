"""
Theme — the simulator's look: a 2D vector-display style of thin, glowing lines
on a dark field (``vector``), or the same drawing in ink on paper (``paper``).

Every colour the windows use comes from here, so a palette swap restyles the
whole application.  ``apply(app, name)`` sets the Qt palette and style sheet.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

from PyQt6.QtGui import QColor, QFont, QFontDatabase, QPalette

ALGO_ORDER = ("OLS", "LM", "CEP", "CEP4", "ROBUST", "BIAS", "RANSAC", "EKF")

# Marker drawn for each algorithm's estimate
ALGO_SHAPES = {"OLS": "square", "LM": "circle", "CEP": "diamond", "CEP4": "triangle",
               "ROBUST": "triangle_down", "BIAS": "hexagon", "RANSAC": "star", "EKF": "ring"}


class StatusColours(dict):
    """``{receiver status: colour}``.  A status without a colour of its own
    (the hardware model may gain new ones) gets ``fallback``, never a KeyError."""

    def __init__(self, colours: dict, fallback: str):
        super().__init__(colours)
        self.fallback = fallback

    def __missing__(self, key) -> str:
        return self.fallback


@dataclass(frozen=True)
class Palette:
    name: str
    bg: str             # arena background
    panel: str          # widget background
    panel_alt: str      # alternate rows, inputs
    border: str
    grid_minor: str
    grid_major: str
    text: str
    text_dim: str
    text_muted: str
    accent: str         # selection, focus, highlights
    wall: str
    obstacle: str
    receiver: str
    truth: str          # true transmitter position
    ray: str            # sound paths
    wave: str           # wavefront
    glow: bool          # draw lines with a soft glow
    algo: dict          # {algorithm: colour}
    status: StatusColours   # {receiver status: colour}


_STATUS_DARK = StatusColours({
    "ok": "#5BE37D", "walk": "#9CCC65", "near": "#FFE066", "echo": "#FFB347", "bent": "#C792EA",
    "crosstalk": "#FF6AD5", "fade": "#4FC9B0", "noise": "#C8B07A", "lost": "#FF5C5C",
    "missed": "#FF5C5C", "radio": "#FF5C5C", "angle": "#FF8C66", "range": "#FF8C66", "off": "#4A5563",
    "dropout": "#FF5C5C", "timeout": "#FF5C5C", "spike": "#FF3B30", "log": "#9FB3C8",
}, fallback="#8FA1B3")

_STATUS_LIGHT = StatusColours({
    "ok": "#1E9E4A", "walk": "#5C8A1F", "near": "#B08900", "echo": "#D2691E", "bent": "#7B3FA0",
    "crosstalk": "#C2188B", "fade": "#00897B", "noise": "#8A6D2B", "lost": "#D32F2F",
    "missed": "#D32F2F", "radio": "#D32F2F", "angle": "#E64A19", "range": "#E64A19", "off": "#9AA5B1",
    "dropout": "#D32F2F", "timeout": "#D32F2F", "spike": "#B71C1C", "log": "#455A64",
}, fallback="#52606D")

PALETTES = {
    "vector": Palette(
        "vector", bg="#0A0E14", panel="#0F151C", panel_alt="#151D26", border="#243041",
        grid_minor="#141C26", grid_major="#1F2B3A", text="#D8E1EA", text_dim="#8FA1B3",
        text_muted="#56677A", accent="#4FD1FF", wall="#7A8B99", obstacle="#C9A26B",
        receiver="#A9C7E8", truth="#FFFFFF", ray="#4FD1FF", wave="#4FD1FF", glow=True,
        algo={"OLS": "#FF9F43", "LM": "#4DA3FF", "CEP": "#3DDC97", "CEP4": "#B388FF",
              "ROBUST": "#FF6B9A", "BIAS": "#E8C547", "RANSAC": "#29D2E4", "EKF": "#C6FF00"},
        status=_STATUS_DARK),
    "paper": Palette(
        "paper", bg="#F7F5F0", panel="#EFECE5", panel_alt="#E6E2D9", border="#C9C3B6",
        grid_minor="#E9E5DC", grid_major="#D6D0C3", text="#1F2933", text_dim="#52606D",
        text_muted="#8A94A0", accent="#0B6BCB", wall="#3E4C59", obstacle="#9C6B2E",
        receiver="#2F4858", truth="#111111", ray="#0B6BCB", wave="#0B6BCB", glow=False,
        algo={"OLS": "#E8710A", "LM": "#1A66C9", "CEP": "#138A52", "CEP4": "#7340C2",
              "ROBUST": "#D0306B", "BIAS": "#A07E00", "RANSAC": "#00838F", "EKF": "#5E8C00"},
        status=_STATUS_LIGHT),
}

current: Palette = PALETTES["vector"]


def color(c: str, alpha: float = 1.0) -> QColor:
    q = QColor(c)
    q.setAlphaF(max(0.0, min(1.0, alpha)))
    return q


def mono_font(size: float = 9.0, bold: bool = False) -> QFont:
    for family in ("JetBrains Mono", "Cascadia Mono", "Fira Code", "DejaVu Sans Mono", "Menlo", "Consolas"):
        if family in QFontDatabase.families():
            f = QFont(family)
            break
    else:
        f = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    f.setPointSizeF(size)
    f.setBold(bold)
    return f


def ui_font(size: float = 9.5) -> QFont:
    for family in ("Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue", "Noto Sans", "DejaVu Sans"):
        if family in QFontDatabase.families():
            f = QFont(family)
            break
    else:
        f = QFont()
    f.setPointSizeF(size)
    return f


def stylesheet(p: Palette) -> str:
    return f"""
    QWidget {{ background: {p.panel}; color: {p.text}; }}
    QMainWindow::separator {{ background: {p.border}; width: 1px; height: 1px; }}
    QDockWidget {{ color: {p.text_dim}; }}
    QDockWidget::title {{ background: {p.panel}; padding: 5px 8px; border-bottom: 1px solid {p.border}; }}
    QToolBar {{ background: {p.panel}; border: none; border-bottom: 1px solid {p.border}; spacing: 2px; padding: 2px; }}
    QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 3px; padding: 4px 8px; color: {p.text}; }}
    QToolButton:hover {{ border-color: {p.border}; background: {p.panel_alt}; }}
    QToolButton:checked {{ border-color: {p.accent}; color: {p.accent}; }}
    QPushButton {{ background: {p.panel_alt}; border: 1px solid {p.border}; border-radius: 3px; padding: 4px 10px; }}
    QPushButton:hover {{ border-color: {p.accent}; }}
    QPushButton:checked {{ border-color: {p.accent}; color: {p.accent}; }}
    QPushButton:disabled {{ color: {p.text_muted}; }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: {p.panel_alt}; border: 1px solid {p.border};
        border-radius: 3px; padding: 2px 4px; selection-background-color: {p.accent}; }}
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled, QLabel:disabled,
    QCheckBox:disabled, QRadioButton:disabled, QToolButton:disabled {{ color: {p.text_muted}; }}
    QComboBox QAbstractItemView {{ background: {p.panel_alt}; border: 1px solid {p.border}; }}
    QGroupBox {{ border: 1px solid {p.border}; border-radius: 4px; margin-top: 14px; padding-top: 6px; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; color: {p.text_dim}; }}
    QSlider::groove:horizontal {{ height: 2px; background: {p.border}; }}
    QSlider::sub-page:horizontal {{ background: {p.accent}; height: 2px; }}
    QSlider::handle:horizontal {{ background: {p.panel}; border: 1px solid {p.accent}; width: 10px;
        height: 10px; margin: -5px 0; border-radius: 5px; }}
    QSlider::sub-page:horizontal:disabled {{ background: {p.text_muted}; }}
    QSlider::handle:horizontal:disabled {{ border-color: {p.text_muted}; }}
    QCheckBox::indicator, QRadioButton::indicator {{ width: 12px; height: 12px; border: 1px solid {p.text_dim};
        background: {p.panel}; }}
    QRadioButton::indicator {{ border-radius: 6px; }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{ background: {p.accent}; border-color: {p.accent}; }}
    QTableWidget, QTableView, QTreeView, QListWidget, QTextBrowser, QPlainTextEdit {{ background: {p.panel};
        alternate-background-color: {p.panel_alt}; gridline-color: {p.border}; border: 1px solid {p.border}; }}
    QHeaderView::section {{ background: {p.panel}; color: {p.text_dim}; border: none;
        border-bottom: 1px solid {p.border}; padding: 3px 6px; }}
    QTabWidget::pane {{ border: 1px solid {p.border}; top: -1px; }}
    QTabBar::tab {{ background: {p.panel}; color: {p.text_dim}; padding: 5px 12px; border: 1px solid transparent; }}
    QTabBar::tab:selected {{ color: {p.text}; border-bottom: 2px solid {p.accent}; }}
    QScrollBar:vertical {{ background: {p.panel}; width: 10px; }}
    QScrollBar::handle:vertical {{ background: {p.border}; min-height: 24px; border-radius: 4px; }}
    QScrollBar:horizontal {{ background: {p.panel}; height: 10px; }}
    QScrollBar::handle:horizontal {{ background: {p.border}; min-width: 24px; border-radius: 4px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
    QMenuBar {{ background: {p.panel}; border-bottom: 1px solid {p.border}; }}
    QMenuBar::item:selected, QMenu::item:selected {{ background: {p.panel_alt}; color: {p.accent}; }}
    QMenu {{ background: {p.panel}; border: 1px solid {p.border}; }}
    QStatusBar {{ background: {p.panel}; color: {p.text_dim}; border-top: 1px solid {p.border}; }}
    QToolTip {{ background: {p.panel_alt}; color: {p.text}; border: 1px solid {p.border}; padding: 4px; }}
    QProgressBar {{ border: 1px solid {p.border}; border-radius: 3px; text-align: center; background: {p.panel_alt}; }}
    QProgressBar::chunk {{ background: {p.accent}; }}
    QLabel[role="dim"] {{ color: {p.text_dim}; }}
    QLabel[role="muted"] {{ color: {p.text_muted}; }}
    QLabel[role="heading"] {{ color: {p.text_dim}; font-weight: bold; }}
    """


def apply(app, name: str = "vector") -> Palette:
    """Switch the whole application to palette ``name``."""
    global current
    current = PALETTES[name]
    p = current
    pal = QPalette()
    for role, c in ((QPalette.ColorRole.Window, p.panel), (QPalette.ColorRole.Base, p.panel),
                    (QPalette.ColorRole.AlternateBase, p.panel_alt), (QPalette.ColorRole.Text, p.text),
                    (QPalette.ColorRole.WindowText, p.text), (QPalette.ColorRole.Button, p.panel_alt),
                    (QPalette.ColorRole.ButtonText, p.text), (QPalette.ColorRole.Highlight, p.accent),
                    (QPalette.ColorRole.HighlightedText, p.bg), (QPalette.ColorRole.ToolTipBase, p.panel_alt),
                    (QPalette.ColorRole.ToolTipText, p.text), (QPalette.ColorRole.PlaceholderText, p.text_muted),
                    (QPalette.ColorRole.Link, p.accent), (QPalette.ColorRole.LinkVisited, p.accent)):
        pal.setColor(role, QColor(c))
    # Each call below walks every widget, calling into Python on the way.  A
    # garbage collection there can delete a closed window Qt is about to visit,
    # and Qt then crashes: let it wait until the walk is over.
    collecting = gc.isenabled()
    gc.disable()
    try:
        app.setStyle("Fusion")             # also tells every widget to restyle (StyleChange)
        app.setPalette(pal)
        app.setFont(ui_font())
        app.setStyleSheet(stylesheet(p))
    finally:
        if collecting:
            gc.enable()
    return p
