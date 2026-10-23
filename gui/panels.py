"""
Panels — the control panels docked round the arena.

Each panel takes ``(sim, runner)``, writes every change straight into the
simulation and has ``refresh(frame)``, which the main window calls for each
new frame to update read-outs and to pick up changes made elsewhere.

Most controls are rows of a ``ParamForm``, built from a table of ``Field``s:
a number is a slider and a spin box kept in step, a bool a check box, a choice
a combo box.  The tables below are also where each setting's tooltip lives.

Writes follow the runner's rule: an ``Environment`` or ``Faults`` change swaps
in a new object in one assignment; anything needing several assignments holds
``runner.lock``.  Reads never take the lock, so the windows never wait for a
frame to finish.
"""

from __future__ import annotations

import dataclasses
import math
from functools import lru_cache
from typing import Callable, NamedTuple

import numpy as np
from PyQt6.QtCore import QEvent, QObject, QPointF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFontMetrics, QIcon, QPainter, QPainterPath, QPixmap
from PyQt6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox,
                             QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                             QPushButton, QRadioButton, QScrollArea, QSlider, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from engine import acoustics as ac
from engine.hardware import PRESETS, STATUS, Environment
from engine.motion import MODES
from engine.solvers import ALGORITHMS
from gui import draw, theme
from gui.theme import color


# ─── Generic form ────────────────────────────────────────────────────────────

class _WheelGuard(QObject):
    """Event filter: the mouse wheel over a control without the keyboard focus
    scrolls the panel instead of changing the control's value."""

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()                 # passes the wheel on to the scroll area
            return True
        return False


def _wheel_needs_focus(*widgets) -> None:
    """Only take the mouse wheel once clicked (or tabbed to), so scrolling a
    long panel never changes a setting the pointer happens to pass over."""
    for w in widgets:
        w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        w.installEventFilter(_WheelGuard(w))


class Field(NamedTuple):
    """One row of a ``ParamForm``."""
    attr: str
    label: str
    kind: str                       # "float", "int", "bool" or "choice"
    lo: float = 0.0
    hi: float = 1.0
    step: float = 1.0
    unit: str = ""
    tip: str = ""
    choices: tuple = ()             # "choice": ((value, text), ...)
    scale: float = 1.0              # shown = stored × scale (100 shows a rate as %)
    needs: Callable | None = None   # needs(get) false: the row does not apply and is disabled


class SliderSpin(QWidget):
    """A compact slider and a spin box showing the same number."""

    valueChanged = pyqtSignal(float)

    def __init__(self, lo: float, hi: float, step: float, unit: str = "", parent=None):
        super().__init__(parent)
        self.lo, self.step = lo, step
        decimals = len(f"{step:g}".partition(".")[2])
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, round((hi - lo) / step))
        self.slider.setMinimumWidth(50)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals)
        self.spin.setSuffix(f" {unit}" if unit else "")
        self.spin.setKeyboardTracking(False)          # commit on Enter, not on every keystroke
        self.spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        widest = max((f"{v:.{decimals}f}{self.spin.suffix()}" for v in (lo, hi)), key=len)
        self.spin.setFixedWidth(self.spin.fontMetrics().horizontalAdvance(widest) + 16)
        _wheel_needs_focus(self.slider, self.spin)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, v: float) -> None:
        for w in (self.slider, self.spin):
            w.blockSignals(True)
        self.spin.setValue(v)
        self.slider.setValue(round((v - self.lo) / self.step))
        for w in (self.slider, self.spin):
            w.blockSignals(False)

    def editing(self) -> bool:
        """True while the user is dragging the slider or typing in the box."""
        return self.slider.isSliderDown() or self.spin.hasFocus()

    def setToolTip(self, text: str) -> None:
        for w in (self, self.slider, self.spin):
            QWidget.setToolTip(w, text)

    def _from_slider(self, k: int) -> None:
        v = self.lo + k * self.step
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self.valueChanged.emit(self.spin.value())

    def _from_spin(self, v: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(round((v - self.lo) / self.step))
        self.slider.blockSignals(False)
        self.valueChanged.emit(v)


class ParamForm(QWidget):
    """Labelled controls built from ``Field`` rows, bound to ``get(attr)`` and
    ``set(attr, value)``.  ``sync()`` shows the current values."""

    changed = pyqtSignal(str, object)      # attr, new value as stored

    def __init__(self, fields, get: Callable, set_: Callable, parent=None):
        super().__init__(parent)
        self.fields  = list(fields)
        self._get    = get
        self._set    = set_
        self.editors = {}                 # attr -> its editor widget
        self._labels = {}                 # attr -> its name label (none for a check box)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        for row, f in enumerate(self.fields):
            self._add_row(grid, row, f)
        self.sync()

    def _add_row(self, grid: QGridLayout, row: int, f: Field) -> None:
        if f.kind == "bool":
            w = QCheckBox(f.label)
            w.toggled.connect(lambda on, f=f: self._commit(f, on))
            grid.addWidget(w, row, 0, 1, 2)
        else:
            name = QLabel(f.label)
            name.setToolTip(f.tip)
            grid.addWidget(name, row, 0)
            self._labels[f.attr] = name
            if f.kind == "choice":
                w = QComboBox()
                w.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
                w.setMinimumContentsLength(8)             # long item texts must not widen the dock
                for value, text in f.choices:
                    w.addItem(text, value)
                _wheel_needs_focus(w)
                w.currentIndexChanged.connect(lambda i, f=f, w=w: self._commit(f, w.itemData(i)))
            else:
                w = SliderSpin(f.lo, f.hi, f.step, f.unit)
                w.valueChanged.connect(lambda v, f=f: self._commit(f, v / f.scale))
            grid.addWidget(w, row, 1)
        w.setToolTip(f.tip)
        self.editors[f.attr] = w

    def _commit(self, f: Field, value) -> None:
        if f.kind == "int":
            value = int(round(value))
        self._set(f.attr, value)
        self.changed.emit(f.attr, value)
        self._enable()

    def sync(self) -> None:
        """Show the current values (leaving a control the user is editing alone)."""
        for f in self.fields:
            w, v = self.editors[f.attr], self._get(f.attr)
            w.blockSignals(True)
            if f.kind == "bool":
                w.setChecked(bool(v))
            elif f.kind == "choice":
                i = next((i for i in range(w.count()) if w.itemData(i) == v), -1)
                if i >= 0:
                    w.setCurrentIndex(i)
            elif not w.editing() and abs(w.value() - v * f.scale) > f.step / 2:
                w.setValue(v * f.scale)
            w.blockSignals(False)
        self._enable()

    def _enable(self) -> None:
        """Disable (and dim) the rows that do not apply with the current settings."""
        for f in self.fields:
            if f.needs is None:
                continue
            on = bool(f.needs(self._get))
            self.editors[f.attr].setEnabled(on)
            name = self._labels.get(f.attr)
            if name is not None and name.isEnabled() != on:
                name.setEnabled(on)
                name.setProperty("role", "" if on else "muted")
                name.style().unpolish(name)
                name.style().polish(name)


def align_spins(*forms: ParamForm) -> None:
    """Give every spin box in ``forms`` the width of the widest, so the
    sliders line up."""
    spins = [w.spin for f in forms for w in f.editors.values() if isinstance(w, SliderSpin)]
    widest = max((sp.maximumWidth() for sp in spins), default=0)
    for sp in spins:
        sp.setFixedWidth(widest)


# ─── Small helpers ───────────────────────────────────────────────────────────

class _Panel(QWidget):
    """Base of the dock panels: keeps ``sim`` and ``runner``, gives the panel a
    scrolling body (``self.body``, a vertical layout), and shows the last frame
    again when the palette changes, since ``refresh`` colours table cells and
    icons with the palette's colours."""

    _restyling = False

    def __init__(self, sim, runner, parent=None):
        super().__init__(parent)
        self.sim, self.runner = sim, runner
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()                 # a short or narrow dock still shows everything
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)
        inner = QWidget()
        scroll.setWidget(inner)
        self.body = QVBoxLayout(inner)
        self.body.setContentsMargins(8, 8, 8, 8)
        self.body.setSpacing(8)

    def refresh(self, frame) -> None:
        raise NotImplementedError

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.StyleChange and not self._restyling:
            self._restyling = True             # theme.apply sends several: refresh once
            QTimer.singleShot(0, self._restyle)

    def _restyle(self) -> None:
        self._restyling = False
        self.refresh(self.sim.last)


def _wrap(layout) -> QWidget:
    """A plain widget holding ``layout``."""
    w = QWidget()
    w.setLayout(layout)
    layout.setContentsMargins(0, 0, 0, 0)
    return w


def _label(text: str = "", role: str = "", tip: str = "", mono: bool = False) -> QLabel:
    """A wrapping label, or with ``mono`` a one-line read-out in the monospace font."""
    w = QLabel(text)
    if role:
        w.setProperty("role", role)
    if tip:
        w.setToolTip(tip)
    if mono:
        w.setFont(theme.mono_font(8.5))
    w.setWordWrap(not mono)
    return w


def _group(title: str, *widgets) -> QGroupBox:
    box = QGroupBox(title)
    lay = QVBoxLayout(box)
    lay.setContentsMargins(8, 8, 8, 8)
    lay.setSpacing(6)
    for w in widgets:
        lay.addWidget(w)
    return box


@lru_cache(maxsize=256)
def _icon(shape: str, colour: str, w: int = 10, h: int = 10, line: bool = False,
          fill: str = "") -> QIcon:
    """A marker glyph (on a short line, like a chart legend, the marker filled
    with ``fill`` to hide the line) or a solid "dot", as an icon."""
    pm = QPixmap(w * 2, h * 2)                         # drawn at 2x for a crisp downscale
    pm.fill(color(colour, 0.0))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(2, 2)
    c = QPointF(w / 2, h / 2)
    if line:
        path = QPainterPath(QPointF(1, h / 2))
        path.lineTo(w - 1, h / 2)
        draw.stroke(p, path, colour, 1.2, glow=False)
    if shape == "dot":
        p.setPen(draw.pen(colour, 1.0))
        p.setBrush(color(colour))
        p.drawEllipse(c, 3.2, 3.2)
    else:
        p.setBrush(color(fill or colour, 1.0 if fill else 0.0))
        p.setPen(draw.pen(colour, 1.3))
        p.drawPath(draw.marker_path(c, shape, 7.0))
    p.end()
    return QIcon(pm)


class Column(NamedTuple):
    """One column of a read-only table."""
    head: str
    align: str                      # "left", "center" or "right"
    sample: str                     # widest cell text expected (sets the width)
    tip: str


ALIGN = {"left": Qt.AlignmentFlag.AlignLeft, "center": Qt.AlignmentFlag.AlignHCenter,
         "right": Qt.AlignmentFlag.AlignRight}


def _table(columns, rows: int, mono_size: float = 8.0, stretch: int = 0) -> QTableWidget:
    """A compact read-only table: each column just wide enough for its header
    and sample text, column ``stretch`` taking any spare width.  The table is
    never narrower than that, so a narrow dock scrolls rather than clipping it."""
    t = QTableWidget(rows, len(columns))
    t.setFont(theme.mono_font(mono_size))
    head = t.horizontalHeader()
    head.setMinimumSectionSize(14)                  # before the widths, or they are clamped
    head.setHighlightSections(False)
    head_fm, cell_fm = QFontMetrics(theme.ui_font(8)), t.fontMetrics()
    widths = [max(head_fm.horizontalAdvance(c.head) + 13, cell_fm.horizontalAdvance(c.sample) + 8)
              for c in columns]
    for j, (c, w) in enumerate(zip(columns, widths)):
        item = QTableWidgetItem(c.head)
        item.setToolTip(c.tip)
        item.setFont(theme.ui_font(8))
        item.setTextAlignment(ALIGN[c.align] | Qt.AlignmentFlag.AlignVCenter)
        t.setHorizontalHeaderItem(j, item)
        t.setColumnWidth(j, w)
    head.setSectionResizeMode(stretch, QHeaderView.ResizeMode.Stretch)
    t.setMinimumWidth(sum(widths) + 2)
    t.verticalHeader().hide()
    t.verticalHeader().setDefaultSectionSize(20)
    t.verticalHeader().setMinimumSectionSize(12)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    t.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    t.setShowGrid(False)
    t.setWordWrap(False)
    t.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    t.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    return t


def _span_row(t: QTableWidget, row: int) -> QTableWidgetItem:
    """Make ``row`` one short line of text spanning the table."""
    t.setSpan(row, 0, 1, t.columnCount())
    t.setRowHeight(row, 15)
    item = _cell(t, row, 0)
    item.setFont(theme.mono_font(7.5))
    item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    return item


def _fit_height(t: QTableWidget) -> None:
    """Size the table to show every row without scrolling."""
    rows = sum(t.rowHeight(i) for i in range(t.rowCount()))
    t.setFixedHeight(t.horizontalHeader().sizeHint().height() + rows + 2 * t.frameWidth())


def _cell(t: QTableWidget, row: int, col: int, text: str = "", colour: str | None = None,
          tip: str = "") -> QTableWidgetItem:
    """Set a cell's text and colour (aligned like its column's header)."""
    item = t.item(row, col)
    if item is None:
        item = QTableWidgetItem()
        item.setTextAlignment(t.horizontalHeaderItem(col).textAlignment())
        t.setItem(row, col, item)
    item.setText(text)
    item.setForeground(color(colour or theme.current.text))
    if tip:
        item.setToolTip(tip)
    return item


def _num(v, decimals: int = 1, signed: bool = False) -> str:
    """A number of cm in at most five characters ("—" for none)."""
    if v is None or not math.isfinite(v):
        return "—"
    a, sign = abs(v), "+" if signed else ""
    if a < 100:
        return f"{v:{sign}.{decimals}f}"
    return (f"{v:{sign}.0f}" if a < 1e4 else f"{v / 1e3:{sign}.0f}k" if a < 1e6
            else f"{v / 1e6:{sign}.0f}M")


# ─── Scenario ────────────────────────────────────────────────────────────────

MODE_NAMES = {"target": "Go to clicked point", "circle": "Circle", "figure8": "Figure eight",
              "lawnmower": "Lawnmower sweep", "random": "Random waypoints"}

SCENARIO_FIELDS = [
    Field("mode", "Motion", "choice", choices=tuple((m, MODE_NAMES[m]) for m in MODES),
          tip="How the transmitter moves. Clicking the arena always switches to 'Go to clicked point' "
              "and sends it there."),
    Field("speed", "Speed", "float", 0, 300, 5, "cm/s",
          tip="How fast the transmitter drives. Fast movement makes consecutive ranges jump, which "
              "the library's jump filter may reject."),
    Field("radius", "Path radius", "float", 20, 200, 5, "cm",
          tip="Size of the circle and figure-eight paths (centred on the arena).",
          needs=lambda v: v("mode") in ("circle", "figure8")),
    Field("dt", "Timestep", "float", 0.02, 0.5, 0.01, "s",
          tip="Simulated seconds per frame (one ping). The real base station pings at 10 Hz: 0.10 s. "
              "Longer steps mean bigger jumps between readings."),
    Field("time_scale", "Time scale", "choice",
          choices=((0.25, "0.25×"), (0.5, "0.5×"), (1.0, "1× real time"), (2.0, "2×"), (4.0, "4×"),
                   (8.0, "8×")),
          tip="Simulated seconds per real second. Faster than the computer can solve runs flat out."),
]


class ScenarioPanel(_Panel):
    """How the transmitter moves and how fast the simulation runs, with a live read-out."""

    def __init__(self, sim, runner, parent=None):
        super().__init__(sim, runner, parent)
        lay = self.body
        self.form = ParamForm(SCENARIO_FIELDS, self._get, self._set)
        align_spins(self.form)
        lay.addWidget(self.form)
        lay.addWidget(_label("Click the arena to send the transmitter to a point.", "dim"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(2)
        self.values = {}
        for k, (key, name, tip) in enumerate((
                ("x", "x", "Transmitter's true x position"),
                ("y", "y", "Transmitter's true y position"),
                ("heading", "heading", "Direction the transmitter is driving (0° = +x, 90° = +y)"),
                ("compute", "compute", "Time to simulate and solve one frame (all algorithms)"),
                ("rate", "rate", "Frames simulated per real second"),
                ("time", "time", "Simulated time since the last reset"))):
            row, col = k % 3, 2 * (k // 3)
            grid.addWidget(_label(name, "muted", tip, mono=True), row, col)
            self.values[key] = _label("—", tip=tip, mono=True)
            self.values[key].setAlignment(Qt.AlignmentFlag.AlignRight)
            grid.addWidget(self.values[key], row, col + 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        lay.addWidget(_group("Transmitter", _wrap(grid)))
        lay.addStretch(1)

    def _get(self, attr):
        m = self.sim.motion
        return {"mode": m.mode, "speed": m.speed, "radius": m.radius, "dt": self.runner.dt,
                "time_scale": self.runner.speed}[attr]

    def _set(self, attr, value) -> None:
        if attr == "mode":
            with self.runner.lock:
                self.sim.motion.set_mode(value)
        elif attr in ("speed", "radius"):
            setattr(self.sim.motion, attr, value)
        elif attr == "dt":
            self.runner.dt = value
        else:
            self.runner.speed = value

    def refresh(self, frame) -> None:
        self.form.sync()
        if frame is None:
            return
        pos = frame.true
        v   = self.values
        v["x"].setText("—" if pos is None else f"{pos[0]:.1f} cm")
        v["y"].setText("—" if pos is None else f"{pos[1]:.1f} cm")
        v["heading"].setText(f"{frame.heading:.0f}°")
        v["compute"].setText(f"{frame.compute_ms:.1f} ms")
        v["rate"].setText(f"{self.runner.real_rate:.1f} fps")
        v["time"].setText(f"{frame.time:.1f} s")


# ─── Receivers ───────────────────────────────────────────────────────────────

STATES = ("on", "blocked", "off")

RECEIVER_COLUMNS = (
    Column("on", "center", "oo", "On: working normally"),
    Column("blk", "center", "oo", "Blocked: something stands in the line of sight to the transmitter. "
           "Sound can still reach it by bouncing off walls or bending round corners, so it may read "
           "long or lose the signal"),
    Column("off", "center", "oo", "Off: unplugged, the base station reads 0 for it"),
    Column("tick", "right", "1510", "Tick count the base station printed: its sampling-loop count "
           "when the echo line dropped"),
    Column("dist", "right", "-139", "Distance the positioning library makes of the tick with "
           "config.toml's calibration (cm); grey when outside the library's bounds"),
    Column("true", "right", "424", "True straight-line distance from the transmitter (cm)"),
    Column("err", "right", "+1029", "Library distance minus true distance (cm): green within 2 cm, "
           "yellow within 10 cm, red beyond"),
    Column("KC", "center", "KC", "K: kept by the library's filter (bounds and jump check).  "
           "C: in the CEP best subset"),
    Column("dB", "right", "+25", "Strength of the detected arrival over the detection threshold (dB). "
           "A small margin means late detection (range walk); below about 3 dB the signal is lost"),
)

SURVEY_TIP = ("Survey error: moves each receiver's REAL position a random distance (standard "
              "deviation σ) away from where config.toml says it is, as if the receivers had been "
              "measured in place carelessly. The library still uses config.toml's positions, so "
              "every fix is slightly wrong. 0 puts them back.")


class ReceiverPanel(_Panel):
    """Two table rows per receiver: what happened to its ping (status dot and
    words), then its on / blocked / off switch and its numbers."""

    def __init__(self, sim, runner, parent=None):
        super().__init__(sim, runner, parent)
        self._sigma = 0.0
        self._seed  = 1
        lay = self.body

        n = len(sim.receivers)
        self.table = t = _table(RECEIVER_COLUMNS, 2 * n, 7.5, stretch=3)
        t.setIconSize(QSize(10, 10))
        self.groups = []
        for i in range(n):
            _span_row(t, 2 * i).setFont(theme.mono_font(7.5, bold=True))
            group = QButtonGroup(self)
            for j, state in enumerate(STATES):
                rb = QRadioButton()
                rb.setToolTip(f"R{i} {state}. {RECEIVER_COLUMNS[j].tip}.")
                group.addButton(rb, j)
                holder = QWidget()
                hl = QHBoxLayout(holder)
                hl.setContentsMargins(0, 0, 0, 0)
                hl.addWidget(rb, 0, Qt.AlignmentFlag.AlignCenter)
                t.setCellWidget(2 * i + 1, j, holder)
            group.idClicked.connect(lambda j, i=i: self.sim.set_receiver_state(i, STATES[j]))
            self.groups.append(group)
        _fit_height(t)
        lay.addWidget(t)

        self.survey = ParamForm([Field("sigma", "Survey error σ", "float", 0, 20, 0.5, "cm", SURVEY_TIP)],
                                lambda _: self._sigma, self._set_sigma)
        reroll = QPushButton("Re-roll")
        reroll.setToolTip("Draw new random survey errors of the same size σ.")
        reroll.clicked.connect(self._reroll)
        self.offset = _label("", "dim", SURVEY_TIP)
        row = QHBoxLayout()
        row.addWidget(self.offset, 1)
        row.addWidget(reroll, 0, Qt.AlignmentFlag.AlignTop)
        lay.addWidget(_group("Receiver positions", self.survey, _wrap(row)))
        lay.addStretch(1)
        self.refresh(None)

    def _set_sigma(self, _attr, sigma: float) -> None:
        self._sigma = sigma
        with self.runner.lock:
            self.sim.survey_error(sigma, seed=self._seed)
        self._show_offset()

    def _reroll(self) -> None:
        self._seed += 1
        self._set_sigma("sigma", self._sigma)

    def _show_offset(self) -> None:
        """How far the receivers really are from config.toml's positions (the
        survey error, or receivers dragged on the arena)."""
        off = np.hypot(*(self.sim.coords - self.sim.believed).T)
        self.offset.setText("Where config.toml says." if off.max() < 1e-9 else
                            f"Up to {off.max():.1f} cm (mean {off.mean():.1f} cm) from where "
                            "config.toml says.")

    def _true_distance(self, i: int, r, frame) -> float:
        if r is not None:
            return r.true_cm
        if frame.true is None:
            return float("nan")
        return float(np.hypot(*(self.sim.coords[i] - frame.true)))

    def refresh(self, frame) -> None:
        pal, t, sim = theme.current, self.table, self.sim
        for group, rx in zip(self.groups, sim.receivers):      # states changed elsewhere too
            group.button(STATES.index(rx.state)).setChecked(True)
        self._show_offset()
        if frame is None:
            for i in range(len(self.groups)):
                _cell(t, 2 * i, 0, f"R{i}", pal.text).setIcon(_icon("dot", pal.text_muted))
            return
        sol      = frame.solution
        cep_used = {int(k) for k in frame.estimates["CEP"].used} if "CEP" in frame.estimates else set()
        for i in range(len(self.groups)):
            r      = frame.readings[i] if frame.readings else None      # none when replaying
            status = r.status if r else "log"
            true   = self._true_distance(i, r, frame)
            tick   = frame.ticks[i]
            dist   = sol.distances[i]
            err    = dist - true if tick > 0 else float("nan")
            kept   = i in sol.sane
            det    = r.detected if r else None
            sc     = pal.status.get(status, pal.text)
            text   = STATUS.get(status, "replayed from a tick log")
            head   = _cell(t, 2 * i, 0, f"R{i}  {status} · {text}", sc, f"R{i} {status}: {text}")
            head.setIcon(_icon("dot", sc))
            row    = 2 * i + 1
            _cell(t, row, 3, str(tick), pal.text if tick else pal.text_muted)
            _cell(t, row, 4, f"{dist:.0f}", pal.text if i in sol.inbounds else pal.text_muted)
            _cell(t, row, 5, _num(true, 0), pal.text_dim)
            _cell(t, row, 6, _num(err, signed=True),
                  pal.text_muted if not math.isfinite(err) else pal.status["ok"] if abs(err) <= 2
                  else pal.status["near"] if abs(err) <= 10 else pal.status["lost"])
            _cell(t, row, 7, ("K" if kept else "·") + ("C" if i in cep_used else "·"),
                  pal.text if kept else pal.text_muted)
            _cell(t, row, 8, "—" if det is None else f"{det.ratio_db:+.0f}",
                  pal.text_dim if det is None or det.ratio_db >= 6 else pal.status["walk"])


# ─── Algorithms ──────────────────────────────────────────────────────────────

ALGO_COLUMNS = (
    Column("show", "left", "", "Tick to draw the algorithm on the arena and the error chart"),
    Column("now", "right", "99999", "This frame's distance from the true position (cm)"),
    Column("RMS", "right", "99999", "Root-mean-square error over the recent history (cm)"),
    Column("p95", "right", "99999", "95 % of the recent errors are below this (cm)"),
    Column("fails", "right", "9999", "Frames the algorithm could not solve (it held its last fix)"),
)

SOLVER_FIELDS = [
    Field("filter_mode", "Receivers used", "choice",
          choices=(("library", "Library filter"), ("bounds", "Bounds only")),
          tip="Which receivers the simulator-only algorithms (CEP4, ROBUST, BIAS, RANSAC) get. "
              "Library filter: those the library kept (bounds and jump check). Bounds only: every "
              "plausible reading, so their own outlier handling has to do the work. The EKF always "
              "uses bounds only (its gate replaces the jump check)."),
    Field("robust_scale_cm", "Robust scale", "float", 0.5, 30, 0.5, "cm",
          tip="ROBUST: residuals smaller than this count fully (least squares), larger ones only "
              "linearly, so a bad reading pulls the fix less."),
    Field("ransac_tol_cm", "RANSAC tolerance", "float", 1, 50, 0.5, "cm",
          tip="RANSAC: a receiver agrees with a 3-receiver fit when its range is within this."),
    Field("ekf_accel", "EKF acceleration", "float", 0, 1000, 10, "cm/s²",
          tip="EKF process noise: how hard the transmitter may accelerate. Low: smooth but slow to "
              "follow turns. High: follows quickly but noisier."),
    Field("ekf_sigma_cm", "EKF range σ", "float", 0.2, 20, 0.1, "cm",
          tip="Range noise the EKF assumes. Larger trusts each range less."),
    Field("ekf_gate", "EKF gate", "float", 1, 10, 0.1, "σ",
          tip="EKF rejects a range more than this many standard deviations from its prediction "
              "(an echo or bad value)."),
]


class AlgorithmPanel(_Panel):
    """One row per algorithm: shown or not, error statistics and what it did this frame."""

    shownChanged = pyqtSignal(set)

    def __init__(self, sim, runner, parent=None):
        super().__init__(sim, runner, parent)
        lay = self.body

        self.table = t = _table(ALGO_COLUMNS, 2 * len(theme.ALGO_ORDER))
        t.setIconSize(QSize(22, 10))
        self.checks = {}
        for k, key in enumerate(theme.ALGO_ORDER):
            info = ALGORITHMS[key]
            box  = QCheckBox(key)
            box.setChecked(True)
            box.setFont(theme.mono_font(8, bold=True))
            box.setToolTip(f"{info.label}\n{info.summary}\n"
                           + ("From the positioning library (ultragps_position)." if info.source == "library"
                              else "Simulator only, for comparison (not in the library)."))
            box.toggled.connect(lambda _on: self.shownChanged.emit(self.shown))
            self.checks[key] = box
            t.setCellWidget(2 * k, 0, box)
            _span_row(t, 2 * k + 1)
        _fit_height(t)
        lay.addWidget(t)

        nan = float("nan")
        self._stats   = {k: (nan, nan, nan, 0) for k in theme.ALGO_ORDER}
        self.settings = ParamForm(SOLVER_FIELDS, lambda a: getattr(sim.solvers.settings, a),
                                  lambda a, v: setattr(sim.solvers.settings, a, v))
        align_spins(self.settings)
        reset = QPushButton("Reset stats")
        reset.setToolTip("Clear the error statistics, trails and every solver's memory (the library's "
                         "jump filter history and the EKF's track).")
        reset.clicked.connect(self._reset)
        lay.addWidget(_group("Solver settings", self.settings, reset))
        lay.addStretch(1)
        self.refresh(None)

    @property
    def shown(self) -> set:
        return {k for k, box in self.checks.items() if box.isChecked()}

    def set_shown(self, keys) -> None:
        for k, box in self.checks.items():
            box.blockSignals(True)
            box.setChecked(k in keys)
            box.blockSignals(False)
        self.shownChanged.emit(self.shown)

    def _reset(self) -> None:
        with self.runner.lock:
            self.sim.reset()
        self.refresh(None)

    def _read_stats(self) -> dict:
        """``sim.stats`` of every algorithm, read without the runner's lock; a
        read that collides with the runner appending keeps the last numbers."""
        try:
            self._stats = {k: self.sim.stats(k) for k in theme.ALGO_ORDER}
        except RuntimeError:                       # deque mutated during iteration
            pass
        return self._stats

    def refresh(self, frame) -> None:
        pal, t = theme.current, self.table
        self.settings.sync()
        for key, box in self.checks.items():           # in the current palette's colours
            box.setIcon(_icon(theme.ALGO_SHAPES[key], pal.algo[key], 22, 10, True, pal.panel))
        stats = self._read_stats()
        for k, key in enumerate(theme.ALGO_ORDER):
            now, rms, p95, fails = stats[key]
            est = frame.estimates.get(key) if frame is not None else None
            if frame is not None:                      # the frame on show, not the runner's latest
                now = frame.errors.get(key, float("nan"))
            _cell(t, 2 * k, 1, _num(now), pal.text if est is None or est.ok else pal.text_muted)
            _cell(t, 2 * k, 2, _num(rms), pal.text)
            _cell(t, 2 * k, 3, _num(p95), pal.text)
            _cell(t, 2 * k, 4, str(fails), pal.text if fails else pal.text_muted)
            label = ALGORITHMS[key].label
            note  = est.note if est is not None else ""
            _cell(t, 2 * k + 1, 0, f"{label} · {note}" if note else label, pal.text_dim,
                  self.checks[key].toolTip() + (f"\nThis frame: {note}" if note else ""))


# ─── Environment ─────────────────────────────────────────────────────────────

ENV_GROUPS = [
    ("Air", [
        Field("temperature_c", "Temperature", "float", -10, 45, 0.5, "°C",
              tip="Air temperature. Sound travels about 0.6 m/s faster per °C, so in air warmer than "
                  "during calibration every range reads short: about 0.18 % per °C, 0.5 cm at 3 m. "
                  "It also changes how fast ultrasound fades."),
        Field("humidity_pct", "Humidity", "float", 0, 100, 1, "%",
              tip="Relative humidity. Moist air carries sound slightly faster (about 0.6 m/s from dry "
                  "to saturated) and changes how strongly 40 kHz ultrasound is absorbed (ISO 9613-1)."),
        Field("cal_temperature_c", "Calibrated at", "float", -10, 45, 0.5, "°C",
              tip="Air temperature when the receivers were calibrated. Times are turned into distances "
                  "at that speed of sound, so any change since shows up as a range error that grows "
                  "with distance."),
        Field("cal_humidity_pct", "Calibrated humidity", "float", 0, 100, 1, "%",
              tip="Relative humidity when the receivers were calibrated."),
        Field("wind_mps", "Air flow", "float", 0, 5, 0.1, "m/s",
              tip="Steady air flow from a fan, air conditioning or an open window. Sound going with "
                  "the flow arrives early, against it late (about 0.3 % of the range per m/s), "
                  "differently for each receiver, so calibration cannot remove it."),
        Field("wind_dir_deg", "Flow direction", "float", -180, 180, 5, "°",
              tip="Direction the air flows toward (0° = +x, 90° = +y). Applies when there is air flow.",
              needs=lambda v: v("wind_mps") > 0),
    ]),
    ("Transducers", [
        Field("beam_patterns", "Beam patterns", "bool",
              tip="Model how directional the transducers are (a circular piston, −6 dB at the "
                  "half-angle below). A transmitter off a receiver's axis is heard weaker, so it is "
                  "detected later or lost. Off: every transducer hears equally in all directions."),
        Field("rx_beam_deg", "RX half-angle", "float", 5, 80, 0.5, "°",
              tip="Angle off a receiver's axis where its sensitivity has halved (−6 dB). The "
                  "HC-SR04's transducers are about ±27°; the datasheet's 15° 'measuring angle' is "
                  "narrower than the real beam. Applies with beam patterns on.",
              needs=lambda v: v("beam_patterns")),
        Field("tx_mode", "TX beam", "choice",
              choices=(("omni", "All round"), ("heading", "Along its heading"),
                       ("fixed", "Fixed direction")),
              tip="All round: the transmitter radiates equally in every direction (e.g. a cone "
                  "reflector). Along its heading: one transducer pointing where it drives. "
                  "Fixed direction: one transducer pointing one way. Applies with beam patterns on.",
              needs=lambda v: v("beam_patterns")),
        Field("tx_beam_deg", "TX half-angle", "float", 5, 80, 0.5, "°",
              tip="Angle off the transmitter's axis where its output has halved (−6 dB).",
              needs=lambda v: v("beam_patterns") and v("tx_mode") != "omni"),
        Field("tx_fixed_deg", "TX facing", "float", -180, 180, 5, "°",
              tip="Direction the transmitter points in 'Fixed direction' mode (0° = +x, 90° = +y).",
              needs=lambda v: v("beam_patterns") and v("tx_mode") == "fixed"),
        Field("fov_cutoff", "Hard field of view", "bool",
              tip="A cruder alternative to beam patterns: nothing at all is heard more than the "
                  "angle below off a receiver's axis, and the reading is 0. Matches the datasheet's "
                  "15° claim rather than the real beam."),
        Field("fov_half_deg", "Field of view ±", "float", 1, 90, 0.5, "°",
              tip="Half-width of the hard field of view.", needs=lambda v: v("fov_cutoff")),
    ]),
    ("Detection", [
        Field("threshold", "Threshold detection", "bool",
              tip="Model the HC-SR04's comparator: its echo line drops on the first 40 kHz cycle at "
                  "which the received burst crosses a fixed threshold. A weak burst crosses a few "
                  "cycles late (range walk, 0.86 cm per 25 µs cycle) or never (signal lost: the module "
                  "times out). Off: ideal timing, every arrival detected at once."),
        Field("snr_db", "Signal margin", "float", 0, 60, 0.5, "dB",
              tip="How far the burst is above the detection threshold, on axis at 1 m. Sound fades "
                  "as 1/distance plus about 1.3 dB/m absorption, so a smaller margin loses distant "
                  "and off-axis receivers first. Models a weak transmitter, low receiver gain or a "
                  "noisy room.", needs=lambda v: v("threshold")),
        Field("rise_cycles", "Ring-up time", "float", 1, 15, 0.5, "cycles",
              tip="Time constant of the resonant transducers ringing up, in 40 kHz cycles (25 µs "
                  "each). Slower ring-up makes a weak burst cross the threshold later (more range "
                  "walk). The burst stops growing after its 8 cycles.", needs=lambda v: v("threshold")),
        Field("amp_jitter_db", "Turbulence", "float", 0, 6, 0.1, "dB",
              tip="Random ping-to-ping change in strength from air turbulence and small movements. "
                  "Makes marginal arrivals flicker between detected, late and lost."),
        Field("ref_distance_cm", "Calibrated range", "float", 20, 400, 5, "cm",
              tip="Detection always takes a cycle or so; the delay of an on-axis burst at this "
                  "distance is absorbed by the calibration. Only range walk beyond it shows as error.",
              needs=lambda v: v("threshold")),
        Field("near_noise", "Near-field noise", "bool",
              tip="Extra noise when the transmitter is very close to a receiver, where the received "
                  "signal is distorted."),
        Field("near_cm", "Near-field distance", "float", 0, 100, 1, "cm",
              tip="Within this distance the extra noise applies (default 4 % of the 400 cm range).",
              needs=lambda v: v("near_noise")),
        Field("near_noise_cm", "Near-field noise σ", "float", 0, 30, 0.5, "cm",
              tip="Standard deviation of the extra near-field noise.", needs=lambda v: v("near_noise")),
        Field("max_range_cm", "Maximum range", "float", 50, 1000, 10, "cm",
              tip="With threshold detection off: nothing is heard beyond this distance (the HC-SR04 "
                  "datasheet says 400 cm). With it on, the range limit comes from the signal margin.",
              needs=lambda v: not v("threshold")),
    ]),
    ("Propagation", [
        Field("reflections", "Reflections", "bool",
              tip="Sound bounces off walls and obstacles (image-source method). A reflection always "
                  "travels further than the direct sound, so it only matters when the direct sound "
                  "is blocked or too weak: then the receiver reads long."),
        Field("bounces", "Bounces", "int", 1, 2, 1, "",
              tip="Most bounces per path. Each bounce keeps only the surface's reflectivity of the "
                  "amplitude, so two-bounce paths are weak.", needs=lambda v: v("reflections")),
        Field("diffraction", "Diffraction", "bool",
              tip="Sound bends round the corners of an obstacle in the way (knife-edge diffraction), "
                  "weaker the sharper the bend. A receiver behind an obstacle may still hear the bent "
                  "sound and read a little long."),
    ]),
    ("Timing & electronics", [
        Field("burst_lead_ms", "TX delay", "float", 0, 10, 0.1, "ms",
              tip="The base station triggers all receivers at once; the transmitter fires this long "
                  "later on its radio trigger, so the receivers are already listening."),
        Field("rx_ready_ms", "RX ready", "float", 0, 10, 0.1, "ms",
              tip="An HC-SR04 raises its echo line about this long after its trigger, once it has "
                  "sent its own burst; anything arriving earlier is not heard."),
        Field("timeout_ms", "Module timeout", "float", 10, 200, 1, "ms",
              tip="With nothing detected, an HC-SR04 gives up after this long and drops its echo line "
                  "anyway: the base station reads a very high value (signal lost)."),
        Field("window_ms", "Sampling window", "float", 10, 100, 1, "ms",
              tip="The base station times the echo lines for this long (80 % of the 100 ms frame). A "
                  "line still high at the end reads 0."),
        Field("sync_offset_us", "Trigger offset", "float", -500, 500, 5, "µs",
              tip="The transmitter fires late by this constant amount (radio and PLL latency). Every "
                  "range reads long by the same 3.4 cm per 100 µs: the BIAS algorithm can absorb it."),
        Field("sync_jitter_us", "Trigger jitter", "float", 0, 200, 1, "µs",
              tip="Random extra trigger delay, different each ping but the same for every receiver."),
        Field("radio_loss", "Radio loss", "float", 0, 20, 0.5, "%", scale=100.0,
              tip="Chance the transmitter misses its radio trigger (there are no retries): it does "
                  "not fire and every receiver times out."),
        Field("tick_dither", "Tick dither", "bool",
              tip="The base station's sampling loop starts at a random phase, so a reading can be "
                  "one tick off."),
        Field("crosstalk", "Crosstalk", "bool",
              tip="Each receiver HC-SR04 also sends its own burst when triggered (unmodified "
                  "modules). Other receivers can hear it, and it reads short."),
    ]),
]

PRESET_TIP = ("Ideal: straight-line ranges, no physics (as in the first simulator). Realistic: "
              "the default HC-SR04 model. Harsh: weaker signal, more turbulence, two-bounce echoes, "
              "trigger jitter, radio loss and a draught.")


class EnvironmentPanel(_Panel):
    """Every ``Environment`` setting, grouped, with presets and the derived speed of sound."""

    def __init__(self, sim, runner, parent=None):
        super().__init__(sim, runner, parent)
        lay = self.body

        self.preset = QComboBox()
        self.preset.addItems(list(PRESETS) + ["Custom"])
        self.preset.setToolTip(PRESET_TIP)
        _wheel_needs_focus(self.preset)
        self.preset.activated.connect(self._load_preset)
        row = QHBoxLayout()
        row.addWidget(_label("Preset", tip=PRESET_TIP))
        row.addWidget(self.preset, 1)
        lay.addLayout(row)
        self.derived = _label("", "dim", "Derived from the air settings above: the speed of sound now "
                                         "and at calibration, the resulting scale error of every "
                                         "range, and how fast 40 kHz ultrasound fades.")
        lay.addWidget(self.derived)

        self.forms = []
        for title, fields in ENV_GROUPS:
            form = ParamForm(fields, lambda a: getattr(self.sim.env, a), self._set)
            form.changed.connect(lambda *_: self.refresh(None))
            self.forms.append(form)
            lay.addWidget(_group(title, form))
        lay.addStretch(1)
        align_spins(*self.forms)
        self.refresh(None)

    def _set(self, attr, value) -> None:
        self.sim.env = self.sim.env.copy(**{attr: value})

    def _load_preset(self, index: int) -> None:
        name = self.preset.itemText(index)
        if name in PRESETS:
            self.sim.env = Environment(**PRESETS[name])
        self.refresh(None)

    def refresh(self, frame) -> None:
        env = self.sim.env
        for form in self.forms:
            form.sync()
        now   = env.as_dict()
        match = next((name for name, values in PRESETS.items() if values == now), "Custom")
        self.preset.setCurrentText(match)
        c     = ac.speed_of_sound(env.temperature_c, env.humidity_pct)
        c_cal = ac.speed_of_sound(env.cal_temperature_c, env.cal_humidity_pct)
        self.derived.setText(f"Sound {c:.1f} m/s (calibrated at {c_cal:.1f}): ranges read "
                             f"{(c_cal / c - 1) * 100:+.2f} %.  Absorption "
                             f"{ac.absorption_db_per_m(env.temperature_c, env.humidity_pct):.2f} dB/m "
                             f"at 40 kHz.")


# ─── Faults and the library filter ───────────────────────────────────────────

FAULT_FIELDS = [
    Field("noise_cm", "Range noise σ", "float", 0, 10, 0.1, "cm",
          tip="Gaussian noise added to every measured range (electrical noise, the base station's "
              "sampling jitter)."),
    Field("dropout_rate", "Dropouts", "float", 0, 50, 0.5, "%", scale=100.0,
          tip="Share of readings missed altogether: the base station reads 0 ticks, which the "
              "library's bounds check rejects."),
    Field("timeout_rate", "Lost signal", "float", 0, 50, 0.5, "%", scale=100.0,
          tip="Share of readings where the signal is lost and the HC-SR04 times out: a very high "
              "value, which the top-band reject should catch."),
    Field("spike_rate", "Bad values", "float", 0, 50, 0.5, "%", scale=100.0,
          tip="Share of readings replaced by a wrong value, off by up to the range below. The jump "
              "filter and the robust algorithms should catch them."),
    Field("spike_pct", "Bad value range ±", "float", 0, 100, 1, "%",
          tip="How wrong a bad value can be: the range is scaled by a random amount within ± this "
              "percentage.", needs=lambda v: v("spike_rate") > 0),
]

FILTER_FIELDS = [
    Field("max_distance", "Top-band reject", "float", 50, 1500, 5, "cm",
          tip="The library discards readings longer than this. Default: the largest distance between "
              "two receivers, since nothing in the arena can be further. Catches lost-signal "
              "timeouts."),
    Field("max_differential", "Jump filter", "float", 1, 500, 1, "cm",
          tip="The library discards a receiver whose reading jumped more than this since the previous "
              "frame (an echo or a bad value). Too tight and a fast transmitter gets rejected."),
]


class FaultsPanel(_Panel):
    """Synthetic faults injected after the physics, and the library's reading filter."""

    def __init__(self, sim, runner, parent=None):
        super().__init__(sim, runner, parent)
        lay = self.body
        self.faults = ParamForm(FAULT_FIELDS, lambda a: getattr(self.sim.faults, a), self._set_fault)
        lay.addWidget(_group("Injected faults", self.faults,
                             _label("Added on top of the physics, to see how each algorithm copes.",
                                    "muted")))
        self.filter = ParamForm(FILTER_FIELDS, self._get_filter,
                                lambda a, v: setattr(self.sim.solvers.lib, a, v))
        lay.addWidget(_group("Library filter", self.filter))
        lay.addStretch(1)
        align_spins(self.faults, self.filter)

    def _set_fault(self, attr, value) -> None:
        self.sim.faults = dataclasses.replace(self.sim.faults, **{attr: value})

    def _get_filter(self, attr):
        lib = self.sim.solvers.lib
        return self.sim.solvers.top_band if attr == "max_distance" else lib.max_differential

    def refresh(self, frame) -> None:
        self.faults.sync()
        self.filter.sync()
