"""
Algorithm lab — watch the positioning algorithms work on one scenario, step
by step and animated, with the library's own equations and numbers.

A scenario is a transmitter position and the distance each receiver
reports.  It is a snapshot of a simulator frame (``load_frame``) or built
here: click the CEP or Blocking canvas to place the transmitter and give each
receiver an override in the Scenario panel (clean, blocked, an extra offset,
or off).  Lab readings come from a private copy of the simulator's hardware
with its random parts turned off, so a scenario always gives the same ticks.

  CEP subsets   the sanity filter, then every receiver subset the library's
                ``_find_best_subset`` scores (OLS seed, LM fit, residuals,
                weights, covariance, CEP) and the box check of
                ``_validate_position``.  The search goes as far as the
                simulator's CEP subsets setting lets it (the first 15, as
                UltraGPS-Control, by default); the subsets past that show
                faintly in the chart, for comparison
  Blocking      each algorithm's fix in turn, then a sweep: how every fix
                moves as one receiver's reading grows (as when it hears a
                reflection), with the library's bounds check at every offset
  LM optimiser  the cost surface Levenberg-Marquardt walks down, every point
                MINPACK evaluates from an initial guess, and where many
                different starts end up

Every number comes from ``engine.lab``, which watches the real library on a
private ``UltraGPSPositionLib``; the simulator's own library (and its filter
memory) is never touched.  The solver settings (filter mode, CEP subsets) are
the simulator's, taken up again whenever the window is activated.  Space,
Left/Right and Home/End drive the current story, except in a focused control
that uses the key itself.
"""

from __future__ import annotations

import contextlib
import copy
import html
import math
import threading
import time
from dataclasses import dataclass, replace
from itertools import combinations

import numpy as np
import scipy
from PyQt6 import sip
from PyQt6.QtCore import QEvent, QObject, QPointF, QRect, QRectF, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QFontMetricsF, QKeySequence, QPainter, QPainterPath, QTransform
from PyQt6.QtWidgets import (QAbstractButton, QAbstractItemView, QAbstractSlider, QApplication, QButtonGroup,
                             QCheckBox, QComboBox, QDockWidget, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
                             QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter,
                             QStackedWidget, QTabBar, QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser,
                             QToolButton, QToolTip, QVBoxLayout, QWidget)

from engine import lab
from engine import solvers as S
from engine.hardware import STATUS, Faults, Hardware
from gui import draw, theme
from gui.theme import color
from ultragps_position import UltraGPSPositionLib

OVERRIDES     = {"clean": "Clean", "blocked": "Blocked", "offset": "+offset", "off": "Off"}
STATE         = {"clean": "on", "blocked": "blocked", "offset": "on", "off": "off"}   # receiver state in the hardware
# What engine.lab.static_solve returns (not EKF: it needs a history).
STATIC_ALGOS  = ("OLS", "LM", "CEP", "CEP4", "ROBUST", "BIAS", "RANSAC")
# Settings.cep_subsets, as the solver panel names them
CEP_SUBSETS   = {"control": "First 15, as UltraGPS-Control", "all": "All"}
# The solver settings engine.lab.static_solve reads (the EKF ones it never uses)
LAB_SETTINGS  = ("filter_mode", "cep_subsets", "robust_scale_cm", "ransac_tol_cm")
MAX_OFFSET_CM = 150
SWEEP_STEPS   = 61
STARTS_SIDE   = 6          # "many starts": a 6 × 6 grid of initial guesses
GRID_N        = 150        # cost surface resolution (columns)
LEVELS        = 12         # contour lines
SAME_MIN_CM   = 2.0        # LM finals closer than this are the same minimum
MINIMA_KEYS   = ("LM", "OLS", "CEP", "ROBUST", "BIAS", "CEP4", "RANSAC", "EKF")   # colours of minima A, B, ...
# Since scipy 1.16, least_squares(method="lm") lets MINPACK scale by the Jacobian's columns
# (D = diag(JᵀJ) in the LM update); before, it passed no scaling (D = I).
LM_JAC_SCALING = tuple(int(v) for v in scipy.__version__.split(".")[:2]) >= (1, 16)


# ─── Small helpers ───────────────────────────────────────────────────────────

def ease(t: float) -> float:
    """Smooth 0 → 1 (smoothstep), for animations."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def num(v, signed: bool = False) -> str:
    """A number to about three significant figures, with a real minus sign."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    v = float(v)
    if math.isinf(v):
        return "∞" if v > 0 else "−∞"
    a, sign = abs(v), "+" if signed else ""
    if a >= 100:
        s = f"{v:{sign}.1f}"
    elif a >= 1:
        s = f"{v:{sign}.2f}"
    elif a >= 0.01 or a == 0:
        s = f"{v:{sign}.3f}"
    elif a >= 1e-4:
        s = f"{v:{sign}.4f}"
    else:
        mantissa, power = f"{v:{sign}.1e}".split("e")
        s = f"{mantissa}×10" + str(int(power)).translate(SUPERSCRIPT)
    return s.replace("-", "−")


SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def pt(p) -> str:
    return "—" if p is None else f"({num(p[0])}, {num(p[1])})"


def names(idx) -> str:
    return " ".join(f"R{i}" for i in idx) or "none"


def span(text: str, c: str) -> str:
    return f"<span style='color:{c}'>{text}</span>"


def html_table(head, rows, align: str = "") -> str:
    """A compact table; ``align`` has an "l" or "r" per column (default: first
    column left, the rest right)."""
    align = align or "l" + "r" * (len(head) - 1)

    def cells(row, tag):
        return "".join(f"<{tag} align='{'left' if align[k] == 'l' else 'right'}'>{c}</{tag}>"
                       for k, c in enumerate(row))
    body = "".join(f"<tr>{cells(r, 'td')}</tr>" for r in rows)
    return f"<table cellspacing='0' cellpadding='2'><tr>{cells(head, 'th')}</tr>{body}</table>"


def matrix(name: str, m) -> str:
    """A 2×2 matrix (or a 2-vector) as aligned monospace lines."""
    m = np.atleast_2d(np.asarray(m, dtype=float))
    if m.shape[0] == 1:
        m = m.T
    cells = [[num(v) for v in row] for row in m]
    w     = max(len(c) for row in cells for c in row)
    pad   = " " * len(name)
    lines = [f"{name if r == 0 else pad} {'=' if r == 0 else ' '} [ " + "  ".join(c.rjust(w) for c in row) + " ]"
             for r, row in enumerate(cells)]
    return "<pre>" + "\n".join(lines) + "</pre>"


def css(p: theme.Palette) -> str:
    return f"""
    body {{ color: {p.text}; }}
    h3 {{ color: {p.text}; font-size: 10pt; margin-top: 0px; margin-bottom: 4px; }}
    p {{ margin-top: 3px; margin-bottom: 5px; }}
    pre {{ margin-top: 2px; margin-bottom: 4px; }}
    th {{ color: {p.text_dim}; font-weight: normal; white-space: nowrap; padding-left: 4px; padding-right: 4px; }}
    td {{ white-space: nowrap; padding-left: 4px; padding-right: 4px; }}
    .dim {{ color: {p.text_dim}; }}
    .muted {{ color: {p.text_muted}; }}
    .eq {{ color: {p.accent}; }}
    .ok {{ color: {p.status['ok']}; }}
    .bad {{ color: {p.status['lost']}; }}
    .warn {{ color: {p.status['echo']}; }}
    """


def nice_ceil(v: float) -> float:
    """The next 1-2-5 number at or above ``v``."""
    if not math.isfinite(v) or v <= 0:
        return 1.0
    mag = 10.0 ** math.floor(math.log10(v))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if v <= m * mag * 1.0001:
            return m * mag
    return 10.0 * mag


def collinear(coords) -> bool:
    """Do these receivers lie on one straight line (within 1 cm)?"""
    c = np.asarray(coords, dtype=float)
    return len(c) >= 2 and np.linalg.svd(c - c.mean(axis=0), compute_uv=False)[-1] < 1.0


def make_table(headers) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.verticalHeader().setDefaultSectionSize(19)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    t.setAlternatingRowColors(True)
    t.setFont(theme.mono_font(8.5))
    t.horizontalHeader().setStretchLastSection(True)
    # It follows the story: clicking a row jumps there, and the keys keep stepping the story
    t.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return t


def fill_table(t: QTableWidget, rows, bold=(), headers=None) -> None:
    """New rows, and columns as wide as every row needs (sized before the story hides any)."""
    t.blockSignals(True)
    if headers is not None:
        t.setHorizontalHeaderLabels(headers)
    t.setRowCount(len(rows))
    for r, row in enumerate(rows):
        t.setRowHidden(r, False)
        for c, text in enumerate(row):
            it = QTableWidgetItem(text)
            it.setTextAlignment((Qt.AlignmentFlag.AlignLeft if c <= 1 else Qt.AlignmentFlag.AlignRight)
                                | Qt.AlignmentFlag.AlignVCenter)
            if r in bold:
                f = it.font()
                f.setBold(True)
                it.setFont(f)
            t.setItem(r, c, it)
    t.resizeColumnsToContents()
    t.blockSignals(False)


def show_rows(t: QTableWidget, last: int, current: int | None) -> None:
    """Show rows up to ``last`` (the animation's progress) and select ``current``."""
    t.blockSignals(True)
    for r in range(t.rowCount()):
        t.setRowHidden(r, r > last)
    if current is not None and 0 <= current < t.rowCount():
        t.selectRow(current)
        t.doItemsLayout()                  # the scroll range must count the rows just shown
        t.scrollToItem(t.item(current, 0))
    else:
        t.clearSelection()
    t.blockSignals(False)


# ─── Scenario ────────────────────────────────────────────────────────────────

@dataclass
class Scenario:
    """One ping to replay: where the transmitter is and what each receiver reported."""
    source: str                     # "lab", or "frame 123" for a snapshot
    truth: np.ndarray | None        # true transmitter position (None: a log without truth)
    ticks: list                     # what the library is given (a lab offset folds in as a fractional tick)
    distances: np.ndarray           # the library's distances, cm
    filter: lab.FilterTrace         # bounds and jump verdicts
    readings: list                  # [Reading] from the hardware (empty for a log)
    states: list                    # receiver state: "on", "blocked" or "off"
    offsets: np.ndarray             # cm the lab added to each reading
    coords: np.ndarray              # receiver positions the library believes
    facing: list                    # receiver facing (deg), for drawing
    world: object                   # copy of the room
    settings: S.Settings            # copy of the simulator's solver settings
    held: np.ndarray | None = None  # the fix the library fell back to in that frame (snapshot)
    previous: str = ""              # what the jump check compared each reading with

    @property
    def sane(self) -> list:
        return list(self.filter.sane)

    @property
    def status(self) -> list:
        return [r.status for r in self.readings] if self.readings else ["log"] * len(self.ticks)

    def error(self, i: int) -> float | None:
        """How far receiver ``i``'s distance is from the true straight-line range."""
        return float(self.distances[i] - self.readings[i].true_cm) if self.readings else None

    def bounds(self, margin: float = 15.0) -> tuple:
        """The arena box: walls, receivers and the transmitter."""
        pts = [self.coords]
        if len(self.world.walls) >= 3:
            pts.append(np.asarray(self.world.walls, dtype=float))
        if self.truth is not None:
            pts.append(np.asarray(self.truth, dtype=float)[None])
        a = np.vstack(pts)
        (x0, y0), (x1, y1) = a.min(axis=0) - margin, a.max(axis=0) + margin
        return float(x0), float(y0), float(x1), float(y1)


def verdict(row: lab.FilterRow) -> str:
    """The filter's verdict on one receiver, in a word or two."""
    if row.kept:
        return "kept"
    if not row.in_bounds:
        return "no reading" if row.distance <= 0 else "too far"
    return "jumped"


def filter_html(sc: Scenario) -> str:
    """The sanity filter with every receiver's numbers."""
    f, pal = sc.filter, theme.current
    rows = []
    for r in f.rows:
        i     = r.receiver
        tick  = sc.ticks[i]
        tick  = f"{tick:.1f}" if abs(tick - round(tick)) > 1e-9 else f"{int(round(tick))}"
        sign  = "+" if r.intercept >= 0 else "−"
        st    = sc.status[i]
        jump  = "—" if r.jump is None else num(r.jump) + (" ✓" if r.jump < f.max_differential else " ✗")
        bound = "✓" if r.in_bounds else "✗"
        v     = verdict(r)
        rows.append([f"R{i}", span(st, pal.status.get(st, pal.text_dim)),
                     f"{tick}·{r.slope:.3f} {sign} {abs(r.intercept):.1f}", f"<b>{num(r.distance)}</b>",
                     bound, jump, span(v, pal.status["ok"] if r.kept else pal.status["lost"])])
    offs = [f"R{i} +{num(o)} cm" for i, o in enumerate(sc.offsets) if o]
    jump_note = (f"compared with {sc.previous}" if sc.previous else
                 "this frame's jumps are not known any more: the receivers the simulator's filter kept are shown")
    return (
        "<h3>The sanity filter</h3>"
        "<p><span class='dim'>_ticks_to_distances</span> &nbsp;"
        "<span class='eq'>d<sub>i</sub> = tick<sub>i</sub> · slope<sub>i</sub> + intercept<sub>i</sub></span>"
        + (f"<br><span class='muted'>lab offsets folded into the tick: {', '.join(offs)}</span>" if offs else "")
        + "</p><p><span class='dim'>_filter_receivers</span> keeps a receiver when "
        f"<span class='eq'>0 &lt; d ≤ {num(f.top_band)}</span> (the top band; 0 ticks reads ≤ 0: "
        "<i>no reading</i>, a timed-out HC-SR04 reads very long: <i>too far</i>) and "
        f"<span class='eq'>|d − d<sub>prev</sub>| &lt; {num(f.max_differential)}</span> (else <i>jumped</i>), "
        f"<span class='muted'>{jump_note}</span></p>"
        + html_table(["", "heard", "tick · slope + icpt", "d cm", "bounds", "jump", "verdict"], rows)
        + f"<p>Kept: <b>{names(f.sane)}</b> ({len(f.sane)} of {len(f.rows)}). "
        + (cep_search_html(sc) + "</p>" if len(f.sane) >= 3 else
           span("Fewer than 3: the library cannot solve and holds its last fix.", pal.status["lost"]) + "</p>")
    )


def cep_search_html(sc: Scenario) -> str:
    """Which subsets of the kept receivers the library's CEP scores, under the CEP subsets setting."""
    n       = len(sc.sane)
    total   = 1 + sum(math.comb(n, k) for k in range(3, n))      # the full set, then 3, 4, … receivers
    scored  = lab.searched_count(n, S.cep_max_subsets(sc.settings, len(sc.filter.rows)))
    setting = CEP_SUBSETS.get(sc.settings.cep_subsets, sc.settings.cep_subsets)
    if total == 1:
        return "The library's CEP has one subset to score: all three."
    if scored < total:
        return (f"The library's CEP scores the first {scored} of the {total} subsets of 3 or more of them "
                f"(CEP subsets: {setting}).")
    return f"The library's CEP scores every subset of 3 or more of them, all {total} (CEP subsets: {setting})."


# ─── Playback ────────────────────────────────────────────────────────────────

class Player(QObject):
    """Steps through ``count`` steps.  Moving into a step takes part of its
    time (``phase`` runs 0 → 1, for the canvases to interpolate); while
    playing, the step is then held for the rest."""

    changed    = pyqtSignal()
    MOVE_SHARE = 0.45

    def __init__(self, parent=None):
        super().__init__(parent)
        self.count        = 0
        self.index        = 0
        self.phase        = 1.0
        self.playing      = False
        self.speed        = 1.0
        self.step_seconds = 1.5          # at speed 1
        self._held        = 0.0
        self._last        = 0.0
        self._timer       = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._tick)

    def load(self, count: int, step_seconds: float, index: int = 0) -> None:
        self.count, self.step_seconds, self.playing = max(int(count), 0), step_seconds, False
        self.seek(index)

    def seek(self, index: int, phase: float = 1.0) -> None:
        """Jump to step ``index``; ``phase`` < 1 animates the move into it."""
        self.index = max(0, min(int(index), self.count - 1)) if self.count else 0
        self.phase = min(max(phase, 0.0), 1.0)
        self._held = 0.0
        self._run()
        self.changed.emit()

    def go(self, index: int) -> None:
        if 0 <= index < self.count and index != self.index:
            self.seek(index, 0.0)

    def next(self) -> None:
        self.go(self.index + 1)

    def prev(self) -> None:
        if self.index > 0:
            self.seek(self.index - 1)

    def first(self) -> None:
        self.playing = False
        self.seek(0)

    def last(self) -> None:
        self.playing = False
        self.seek(self.count - 1)

    def play(self) -> None:
        if self.count < 2:
            return
        if self.index >= self.count - 1:
            self.seek(0, 0.0)
        self.playing = True
        self._run()
        self.changed.emit()

    def pause(self) -> None:
        self.playing = False
        self.changed.emit()

    def toggle(self) -> None:
        self.pause() if self.playing else self.play()

    def advance(self, seconds: float) -> None:
        """Move the animation on by ``seconds`` of real time."""
        dur  = self.step_seconds / max(self.speed, 1e-3)
        move = self.MOVE_SHARE * dur
        if self.phase < 1.0:
            self.phase = min(1.0, self.phase + seconds / move)
        elif self.playing:
            self._held += seconds
            if self._held < dur - move:
                return
            if self.index < self.count - 1:
                self.index, self.phase, self._held = self.index + 1, 0.0, 0.0
            else:
                self.playing = False
        else:
            return
        self.changed.emit()

    def _run(self) -> None:
        if (self.playing or self.phase < 1.0) and not self._timer.isActive():
            self._last = time.perf_counter()
            self._timer.start()

    def _tick(self) -> None:
        now = time.perf_counter()
        dt, self._last = now - self._last, now
        self.advance(min(dt, 0.1))
        if not self.playing and self.phase >= 1.0:
            self._timer.stop()


class GlyphButton(QToolButton):
    """A transport button drawn as a vector glyph: first, prev, play, pause, next, last."""

    def __init__(self, glyph: str, parent=None):
        super().__init__(parent)
        self.glyph = glyph
        self.setFixedSize(30, 24)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        pal = theme.current
        p   = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        x, y, s = self.width() / 2, self.height() / 2, 5.0
        path = QPainterPath()

        def chevron(cx, d):            # an open ">" (d = 1) or "<" (d = -1)
            path.moveTo(cx - d * s * 0.5, y - s)
            path.lineTo(cx + d * s * 0.5, y)
            path.lineTo(cx - d * s * 0.5, y + s)

        def bar(cx):
            path.moveTo(cx, y - s)
            path.lineTo(cx, y + s)

        if self.glyph == "play":
            path.moveTo(x - s * 0.7, y - s)
            path.lineTo(x + s, y)
            path.lineTo(x - s * 0.7, y + s)
            path.closeSubpath()
        elif self.glyph == "pause":
            bar(x - 2.5)
            bar(x + 2.5)
        elif self.glyph == "first":
            bar(x - 4)
            chevron(x + 1.5, -1)
        elif self.glyph == "last":
            bar(x + 4)
            chevron(x - 1.5, 1)
        else:
            chevron(x, 1 if self.glyph == "next" else -1)
        draw.stroke(p, path, pal.text if self.isEnabled() else pal.text_muted, 1.4)


class PlaybackBar(QWidget):
    """|< < play/pause > >|, a speed slider and "step k of n: what happens"."""

    def __init__(self, player: Player, parent=None):
        super().__init__(parent)
        self.player  = player
        self.buttons = {}
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 3, 6, 3)
        lay.setSpacing(2)
        for glyph, tip, slot in (("first", "Go to the first step (Home)", player.first),
                                 ("prev", "Back one step (Left arrow)", player.prev),
                                 ("play", "Play or pause the animation (Space)", player.toggle),
                                 ("next", "Forward one step, animated (Right arrow)", player.next),
                                 ("last", "Go to the last step (End)", player.last)):
            b = GlyphButton(glyph)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            lay.addWidget(b)
            self.buttons[glyph] = b
        lay.addSpacing(8)
        self.speed = QSlider(Qt.Orientation.Horizontal)
        self.speed.setRange(0, 10)
        self.speed.setValue(4)
        self.speed.setFixedWidth(80)
        self.speed.setToolTip("Animation speed: left is slower, right is faster")
        self.speed.valueChanged.connect(self._on_speed)
        self.speed_label = QLabel("1×")
        self.speed_label.setProperty("role", "dim")
        self.speed_label.setFont(theme.mono_font(8.5))
        self.speed_label.setFixedWidth(36)
        lay.addWidget(self.speed)
        lay.addWidget(self.speed_label)
        lay.addSpacing(6)
        self.full_text = ""
        self.status = QLabel("")
        self.status.setFont(theme.mono_font(8.5))
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        lay.addWidget(self.status, 1)
        player.changed.connect(self._sync)
        self._sync()

    def set_text(self, text: str) -> None:
        n = self.player.count
        self.full_text = f"step {self.player.index + 1} of {n} · {text}" if n else text
        self.status.setToolTip(self.full_text)
        self._fit_text()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_text()

    def _fit_text(self) -> None:
        """The status line, cut short with "…" where the bar is too narrow (the tooltip has it all)."""
        self.status.setText(self.status.fontMetrics().elidedText(self.full_text, Qt.TextElideMode.ElideRight,
                                                                 self.status.width()))

    def _on_speed(self, v: int) -> None:
        self.player.speed = 2.0 ** ((v - 4) / 2.0)
        self.speed_label.setText(f"{self.player.speed:.2g}×")

    def _sync(self) -> None:
        pl = self.player
        self.buttons["play"].glyph = "pause" if pl.playing else "play"
        self.buttons["play"].update()
        for g, on in (("first", pl.index > 0), ("prev", pl.index > 0),
                      ("next", pl.index < pl.count - 1), ("last", pl.index < pl.count - 1),
                      ("play", pl.count > 1)):
            self.buttons[g].setEnabled(on)


class Explanation(QTextBrowser):
    """The equations and numbers of the current step.  ``present(render)``
    takes a function returning HTML, so a palette change can redraw it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._render = None
        self.setFont(theme.mono_font(8.5))
        self.setOpenLinks(False)
        self.setMinimumWidth(240)
        self.setToolTip("")

    def present(self, render) -> None:
        self._render = render
        self._refresh(keep_scroll=False)

    def _refresh(self, keep_scroll: bool = True) -> None:
        if self._render is None:
            return
        pos = self.verticalScrollBar().value()
        self.document().setDefaultStyleSheet(css(theme.current))
        self.setHtml(self._render())
        if keep_scroll:
            self.verticalScrollBar().setValue(pos)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in (QEvent.Type.StyleChange, QEvent.Type.PaletteChange):
            self._refresh()


# ─── Canvas and shared drawing ───────────────────────────────────────────────

class LabCanvas(QWidget):
    """A world view (cm, y up) that a tab paints its scene into.  Click →
    ``clicked``; the wheel zooms, right- or middle-drag pans, double-click fits.
    A click is reported once the double-click time has passed, so the first
    click of a double-click does not count."""

    clicked = pyqtSignal(float, float)

    def __init__(self, paint_scene, hint: str = "", parent=None):
        super().__init__(parent)
        self.paint_scene = paint_scene
        self.hint        = hint
        self.view        = draw.View()
        self.bounds      = (-200.0, -250.0, 200.0, 250.0)
        self._auto       = True        # keep fitting the bounds until the user zooms or pans
        self._press      = None
        self._pan        = None
        self._double     = False       # the release that ends a double-click is not a click
        self._click_at   = (0.0, 0.0)
        self._click      = QTimer(self)
        self._click.setSingleShot(True)
        self._click.setInterval(QApplication.doubleClickInterval())
        self._click.timeout.connect(lambda: self.clicked.emit(*self._click_at))
        self.setMinimumSize(260, 200)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip(f"{hint + ' · ' if hint else ''}wheel: zoom · right-drag: pan · double-click: fit")

    def set_bounds(self, bounds) -> None:
        self.bounds = tuple(bounds)
        if self._auto:
            self.fit()

    def fit(self) -> None:
        self._auto = True
        self.view.resize(self.width(), self.height())
        self.view.fit(self.bounds, 26.0)
        self.update()

    def resizeEvent(self, event) -> None:
        self.view.resize(self.width(), self.height())
        if self._auto:
            self.view.fit(self.bounds, 26.0)
        super().resizeEvent(event)

    def paintEvent(self, _event) -> None:
        pal = theme.current
        p   = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), color(pal.bg))
        draw.grid(p, self.view)
        self.paint_scene(p, self.view)

    def mousePressEvent(self, e) -> None:
        self._press, self._double = e.position(), False
        if e.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan = e.position()

    def mouseMoveEvent(self, e) -> None:
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan, self._auto = e.position(), False
            self.view.pan(d.x(), d.y())
            self.update()

    def mouseReleaseEvent(self, e) -> None:
        moved = self._press is not None and (e.position() - self._press).manhattanLength() > 4
        if e.button() == Qt.MouseButton.LeftButton and not moved and not self._double:
            self._click_at = self.view.world(e.position().x(), e.position().y())
            self._click.start()
        self._press = self._pan = None
        self._double = False

    def wheelEvent(self, e) -> None:
        self._auto = False
        self.view.zoom_at(1.15 ** (e.angleDelta().y() / 120.0), e.position().x(), e.position().y())
        self.update()

    def mouseDoubleClickEvent(self, _e) -> None:
        self._click.stop()
        self._double = True
        self.fit()


def draw_world(p: QPainter, view: draw.View, world) -> None:
    """The room, faint: it is scenery here."""
    pal = theme.current
    p.save()
    p.setOpacity(0.5)
    if len(world.walls) >= 3:
        draw.wall_outline(p, view, world.walls, pal.wall)
    for o in world.obstacles:
        draw.hatched_polygon(p, view, o.points, pal.obstacle)
    p.restore()


def info_anchor(view: draw.View, pos, facing: float, line: int) -> tuple[QPointF, str]:
    """Where line ``line`` of extra text goes beside a receiver's name label
    (behind the receiver, away from the arena), and how to align it."""
    c  = view.px(*pos)
    t  = math.radians(facing)
    ox, oy = -math.cos(t), math.sin(t)          # outward, in screen coordinates
    name = QPointF(c.x() + ox * 23.4, c.y() + oy * 23.4)
    step = -13.0 if oy < -0.5 else 13.0
    y    = name.y() + step * (line + 1)
    if ox < -0.3:
        return QPointF(name.x() + 10, y), "right"
    if ox > 0.3:
        return QPointF(name.x() - 10, y), "left"
    return QPointF(name.x(), y), "center"


def draw_receivers(p: QPainter, view: draw.View, sc: Scenario, lit=None, rings=None,
                   info=None, thick=()) -> None:
    """Every receiver where the library believes it is.  Receivers not in
    ``lit`` are faded; ``rings`` {i: colour} draws status rings, ``info``
    {i: [(text, colour)]} adds lines beside the name."""
    pal = theme.current
    for i, pos in enumerate(sc.coords):
        p.save()
        if lit is not None and i not in lit:
            p.setOpacity(0.3)
        draw.receiver_glyph(p, view, pos, sc.facing[i], pal.receiver, (rings or {}).get(i),
                            sc.states[i], label_text=f"R{i}", used=i in thick)
        for k, (text, c) in enumerate((info or {}).get(i, [])):
            at, align = info_anchor(view, pos, sc.facing[i], k)
            draw.label(p, at, text, c, 7.5, align)
        p.restore()


def draw_truth(p: QPainter, view: draw.View, truth) -> None:
    if truth is not None:
        draw.crosshair(p, view.px(*truth), theme.current.truth, 9.0, 1.3)


def receiver_points(view: draw.View, sc: Scenario) -> list:
    """Pixels the legend and magnifier should keep clear of: receivers, their
    names (drawn behind the transducer face, as draw.receiver_glyph does) and
    the truth."""
    pts = [view.px(*c) for c in sc.coords]
    for q, facing in zip(list(pts), sc.facing):
        t = math.radians(facing)
        pts.append(QPointF(q.x() - 23.4 * math.cos(t), q.y() + 23.4 * math.sin(t)))
    return pts + ([view.px(*sc.truth)] if sc.truth is not None else [])


def draw_offscreen(p: QPainter, view: draw.View, pos, c: str, text: str) -> None:
    """An arrow at the canvas edge pointing to ``pos`` when it is out of sight."""
    if pos is None or not np.all(np.isfinite(pos)):
        return
    q     = view.px(*pos)
    inner = QRectF(0, 0, view.width, view.height).adjusted(24, 24, -24, -24)
    if inner.contains(q):
        return
    o  = QPointF(view.width / 2, view.height / 2)
    d  = q - o
    ts = [t for t in ((inner.left() - o.x()) / d.x() if d.x() < 0 else None,
                      (inner.right() - o.x()) / d.x() if d.x() > 0 else None,
                      (inner.top() - o.y()) / d.y() if d.y() < 0 else None,
                      (inner.bottom() - o.y()) / d.y() if d.y() > 0 else None) if t is not None]
    tip = o + d * min(ts)
    L   = math.hypot(d.x(), d.y())
    tail = tip - d * (20.0 / L)
    draw.arrow(p, tail, tip, c, 1.4, 6.0)
    back = tail - d * (8.0 / L)
    draw.label(p, back, text, c, 7.5, "right" if d.x() > 0 else "left" if d.x() < 0 else "center")


def draw_ranges(p: QPainter, view: draw.View, sc: Scenario, which, grow: float = 1.0,
                dropped_too: bool = True) -> None:
    """Range circles ``|p − Rᵢ| = dᵢ``: solid for kept receivers, dotted for dropped ones."""
    pal = theme.current
    for i in which:
        d = float(sc.distances[i])
        if d <= 0 or d > 3000:
            continue
        kept = i in sc.filter.sane
        if not kept and not dropped_too:
            continue
        draw.circle(p, view, sc.coords[i], d * grow, pal.ray if kept else pal.status["lost"], 1.0,
                    0.5 if kept else 0.45, None if kept else draw.DOT)


def draw_filter(p: QPainter, view: draw.View, sc: Scenario, t: float) -> None:
    """Each receiver's range circle growing to its distance, with the verdict."""
    pal = theme.current
    draw_ranges(p, view, sc, range(len(sc.coords)), ease(t))
    info = {}
    for r in sc.filter.rows:
        c = pal.status["ok"] if r.kept else pal.status["lost"]
        info[r.receiver] = [(f"{num(r.distance)} cm", pal.text_dim), (verdict(r), c)]
    rings = {i: pal.status.get(s, pal.text_dim) for i, s in enumerate(sc.status)}
    draw_receivers(p, view, sc, rings=rings, info=info if t >= 0.5 else None)


def filter_keys(sc: Scenario) -> list:
    """Legend of the filter step: ranges, and the status of each reading."""
    pal  = theme.current
    keys = [Key("line", pal.ray, "range dᵢ, kept")]
    if len(sc.sane) < len(sc.filter.rows):
        keys.append(Key("dot", pal.status["lost"], "range, dropped"))
    keys += [Key("circle", pal.status.get(st, pal.text_dim), st) for st in dict.fromkeys(sc.status)]
    if sc.truth is not None:
        keys.append(Key("crosshair", pal.truth, "true position"))
    return keys


def arc_points(centre, radius: float, near, span_cm: float) -> np.ndarray:
    """The part of a circle within about ``span_cm`` of the point ``near``."""
    c = np.asarray(centre, dtype=float)
    v = np.asarray(near, dtype=float) - c
    a = math.atan2(v[1], v[0])
    h = min(span_cm / max(radius, 1e-9), math.pi)
    t = np.linspace(a - h, a + h, 48)
    return c + radius * np.column_stack([np.cos(t), np.sin(t)])


def small_scale_bar(p: QPainter, view: draw.View, at: QPointF) -> None:
    """A scale bar for any zoom: the 1-2-5 length nearest 50 px."""
    pal    = theme.current
    length = nice_ceil(45.0 / view.scale)
    w      = length * view.scale
    path   = QPainterPath(QPointF(at.x(), at.y() - 3))
    path.lineTo(at.x(), at.y())
    path.lineTo(at.x() + w, at.y())
    path.lineTo(at.x() + w, at.y() - 3)
    draw.stroke(p, path, pal.text_dim, 1.0, glow=False)
    draw.label(p, QPointF(at.x() + w / 2, at.y() - 9), f"{length:g} cm", pal.text_dim, 7, "center", halo=False)


def inset_frame(p: QPainter, rect: QRectF) -> None:
    """Background and border of a magnifier or chart inset."""
    pal = theme.current
    p.setPen(draw.pen(pal.border, 1.0))
    p.setBrush(color(pal.bg, 0.94))
    p.drawRect(rect)


def inset_title(p: QPainter, rect: QRectF, title: str) -> None:
    draw.label(p, QPointF(rect.left() + 7, rect.top() + 10), title, theme.current.text_dim, 7.5)


def free_spot(view: draw.View, w: float, h: float, avoid, busy=(), prefer: str = "tl") -> QRectF:
    """A ``w`` × ``h`` box at a canvas edge covering as few of the ``avoid``
    points (pixels) as it can and none of the ``busy`` boxes: the corners and
    edge middles first (``prefer`` before the rest), then anywhere along the
    edges."""
    xs = {"l": 34.0, "c": (view.width - w) / 2, "r": view.width - w - 8.0}
    ys = {"t": 8.0, "m": (view.height - h) / 2, "b": view.height - h - 22.0}
    spots = [prefer] + [c for c in ("tl", "tr", "bl", "br", "tc", "bc", "ml", "mr") if c != prefer]
    rects = [QRectF(xs[c[1]], ys[c[0]], w, h) for c in spots]
    rects += [QRectF(x, y, w, h) for y in np.arange(ys["t"], ys["b"], 12.0) for x in (xs["l"], xs["r"])]
    rects += [QRectF(x, y, w, h) for x in np.arange(xs["l"], xs["r"], 12.0) for y in (ys["t"], ys["b"])]
    best = None
    for r in rects:
        cost = spot_cost(r, avoid, busy)
        if cost == 0:
            return r
        if best is None or cost < best[0]:
            best = (cost, r)
    return best[1]


def spot_cost(r: QRectF, avoid, busy=()) -> int:
    """How many ``avoid`` points a box covers (a busy box counts 100)."""
    near = r.adjusted(-14, -14, 14, 14)
    return sum(near.contains(q) for q in avoid) + 100 * sum(r.intersects(b) for b in busy)


@dataclass
class Key:
    """One line of a canvas legend."""
    glyph: str              # a draw.marker shape, or line, dash, dot, tick, arrow, ellipse,
                            # crosshair; a "faint:" prefix fades it
    colour: str
    text: str
    value: str = ""
    strong: bool = False    # text in the glyph's colour, bold (the current item)
    spare: bool = False     # may be left out where space is short (a chart beside the canvas shows it)


LEGEND_ROW = 14.0


def legend_glyph(p: QPainter, at: QPointF, glyph: str, c: str) -> None:
    alpha = 0.45 if glyph.startswith("faint:") else 1.0
    glyph = glyph.removeprefix("faint:")
    x, y  = at.x(), at.y()
    path  = QPainterPath()
    if glyph in ("line", "dash", "dot", "tick"):
        path.moveTo(x - 7, y)
        path.lineTo(x + 7, y)
        dash = {"dash": draw.DASH, "dot": draw.DOT}.get(glyph)
        draw.stroke(p, path, c, 2.2 if glyph == "tick" else 1.2, alpha, dash)
    elif glyph == "arrow":
        draw.arrow(p, QPointF(x - 7, y), QPointF(x + 7, y), c, 1.2, 5.0, alpha)
    elif glyph == "ellipse":
        path.addEllipse(at, 7.0, 4.0)
        draw.stroke(p, path, c, 1.0, alpha)
    elif glyph == "crosshair":
        draw.crosshair(p, at, c, 6.0, 1.1, 2.0)
    else:
        draw.marker(p, at, glyph, c, 8, 1.2, alpha)


def draw_legend(p: QPainter, view: draw.View, keys, hint: str, avoid, busy=()) -> QRectF:
    """A key to the canvas in the freest corner, ending with what a click does.
    Where the full key would cover a receiver, a narrower one without the
    values and the hint is tried, then one without the spare keys, then two
    columns."""
    pal = theme.current
    fm  = QFontMetricsF(theme.mono_font(7.5))
    bare = [Key(k.glyph, k.colour, k.text, "", k.strong) for k in keys]
    core = [b for k, b in zip(keys, bare) if not k.spare]
    best = None
    for shown, note, cols in ((keys, hint, 1), (bare, "", 1), (core, "", 1), (bare, "", 2)):
        if not shown:
            continue
        per  = math.ceil(len(shown) / cols)
        colw = max(fm.horizontalAdvance(k.text) + (fm.horizontalAdvance(k.value) + 14 if k.value else 0)
                   for k in shown) + 34
        w    = max(cols * colw, fm.horizontalAdvance(note) + 16)
        rect = free_spot(view, w, 8 + LEGEND_ROW * (per + bool(note)), avoid, busy, "tl")
        cost = spot_cost(rect, avoid, busy)
        if best is None or cost < best[0]:
            best = (cost, rect, shown, note, per, colw)
        if cost == 0:
            break
    _, rect, keys, hint, per, colw = best
    inset_frame(p, rect)
    for n, k in enumerate(keys):
        x = rect.left() + (n // per) * colw
        y = rect.top() + 4 + LEGEND_ROW * (n % per + 0.5)
        legend_glyph(p, QPointF(x + 12, y), k.glyph, k.colour)
        draw.label(p, QPointF(x + 25, y), k.text, k.colour if k.strong else pal.text_dim, 7.5,
                   bold=k.strong, halo=False)
        if k.value:
            draw.label(p, QPointF(rect.right() - 6, y), k.value, pal.text if k.strong else pal.text_dim, 7.5,
                       "right", halo=False)
    if hint:
        draw.label(p, QPointF(rect.left() + 7, rect.bottom() - 4 - LEGEND_ROW / 2), hint, pal.text_muted, 7,
                   halo=False)
    return rect


def magnifier(p: QPainter, view: draw.View, centre, radius_cm: float, paint, avoid, busy=()) -> QRectF | None:
    """A zoomed inset round ``centre``, for what is too small to see at arena
    scale: ``paint(p, inset_view)`` draws into it.  None when not needed."""
    big  = min(max(0.34 * min(view.width, view.height), 120.0), 210.0)
    best = None
    for size in (big, 0.8 * big, 0.66 * big):         # shrink rather than hide a receiver
        rect = free_spot(view, size, size, avoid, busy, "br")
        cost = spot_cost(rect, avoid, busy)
        if best is None or cost < best[0]:
            best = (cost, size, rect)
        if cost == 0:
            break
    cost, size, rect = best
    scale = (size / 2 - 14) / max(radius_cm, 1e-6)
    if cost >= 100 or scale < 3.0 * view.scale:          # never over the legend or a banner
        return None
    inset_frame(p, rect)
    p.save()
    p.setClipRect(rect.adjusted(1, 1, -1, -1))
    p.translate(rect.topLeft())
    v = draw.View()
    v.resize(int(size), int(size))
    v.scale, v.cx, v.cy = scale, float(centre[0]), float(centre[1])
    paint(p, v)
    small_scale_bar(p, v, QPointF(10, size - 10))
    p.restore()
    inset_title(p, rect, f"×{scale / view.scale:.0f}")
    return rect


def banner(p: QPainter, view: draw.View, text: str, c: str) -> QRectF:
    """A verdict across the top of the canvas, smaller where it would not fit."""
    wide = QFontMetricsF(theme.mono_font(9, True)).horizontalAdvance(text)
    size = max(7.0, min(9.0, 9.0 * (view.width - 24) / max(wide, 1.0)))
    return draw.label(p, QPointF(view.width / 2, 16), text, c, size, "center", bold=True)


def side_label(p: QPainter, view: draw.View, q: QPointF, text: str, c: str, size: float = 7.5) -> None:
    """A label beside point ``q``, flipped to the left near the right edge."""
    fm = QFontMetricsF(theme.mono_font(size))
    if q.x() + 12 + fm.horizontalAdvance(text) > view.width - 4:
        draw.label(p, QPointF(q.x() - 12, q.y() - 12), text, c, size, "right")
    else:
        draw.label(p, QPointF(q.x() + 12, q.y() - 12), text, c, size)


# ─── Tabs ────────────────────────────────────────────────────────────────────

class LabTab(QWidget):
    """What every tab shares: the scenario, a canvas, a player with its
    playback bar and an explanation pane.  A tab builds its steps in
    ``compute`` (run when it is shown and the scenario has changed) and
    describes step ``k`` in ``describe`` (one line), ``html`` (the pane) and
    ``paint_scene`` (the canvas, at ``phase`` 0 → 1 of the move into it)."""

    STEP_SECONDS = 1.5
    HINT         = ""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sc: Scenario | None = None
        self.lib    = None
        self._dirty = False
        self._shown = None
        self.player = Player(self)
        self.bar    = PlaybackBar(self.player)
        self.text   = Explanation()
        self.canvas = LabCanvas(self._paint, self.HINT)
        self.player.changed.connect(self._on_player)

    def set_scenario(self, sc: Scenario, lib) -> None:
        self.sc, self.lib, self._dirty = sc, lib, True
        if self.isVisible():
            self.ensure_computed()

    def ensure_computed(self) -> None:
        if self._dirty and self.sc is not None:
            self._dirty = False
            self.canvas.set_bounds(self.sc.bounds())
            self.compute()
            self._on_player()

    def restart(self, count: int, step_seconds: float | None = None, play: bool = False) -> None:
        """Load ``count`` new steps; stay at the end if the player was there."""
        at_end = self.player.count > 1 and self.player.index == self.player.count - 1 and not play
        self._shown = None
        self.player.load(count, step_seconds or self.STEP_SECONDS, count - 1 if at_end else 0)
        if play:
            self.player.play()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.ensure_computed()

    def _on_player(self) -> None:
        if self.sc is None or self._dirty:
            return
        k = self.player.index
        if k != self._shown and self.player.count:
            self._shown = k
            self.bar.set_text(self.describe(k))
            self.text.present(lambda k=k: self.html(k))
            self.on_step(k)
        self.on_phase()

    def _paint(self, p: QPainter, view: draw.View) -> None:
        if self.sc is not None and not self._dirty and self.player.count:
            self.paint_scene(p, view, self.player.index, self.player.phase)

    def on_step(self, k: int) -> None:
        pass

    def on_phase(self) -> None:
        self.canvas.update()


def splitter(orientation, widgets, sizes) -> QSplitter:
    s = QSplitter(orientation)
    for w in widgets:
        s.addWidget(w)
    s.setChildrenCollapsible(False)
    s.setSizes(sizes)
    for i, w in enumerate(sizes):
        s.setStretchFactor(i, w)
    return s


class FlowRow(QWidget):
    """A row of controls, left-aligned, that wraps onto another line where the
    tab is too narrow for it, so it never sets the window's minimum width.  A
    widget that places its controls itself rather than a QLayout subclass: Qt
    asks a layout for its items while it tears the window down, which Python
    can no longer answer at exit."""

    def __init__(self, widgets, margins=(8, 4, 8, 4), spacing: int = 6, parent=None):
        super().__init__(parent)
        self.items, self.margins, self.spacing = list(widgets), margins, spacing
        for w in self.items:
            w.setParent(self)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._place(width, move=False)

    def sizeHint(self) -> QSize:                  # everything on one line
        left, top, right, bottom = self.margins
        hints = [w.sizeHint() for w in self.items]
        width = sum(h.width() for h in hints) + self.spacing * max(len(hints) - 1, 0)
        return QSize(left + width + right, top + max((h.height() for h in hints), default=0) + bottom)

    def minimumSizeHint(self) -> QSize:           # one control per line at the most
        left, top, right, bottom = self.margins
        hints = [w.sizeHint() for w in self.items]
        return QSize(left + max((h.width() for h in hints), default=0) + right,
                     top + max((h.height() for h in hints), default=0) + bottom)

    def event(self, e) -> bool:
        if e.type() == QEvent.Type.LayoutRequest:          # a control's size changed (text, font, style)
            self.updateGeometry()
            self._place(self.width(), move=True)
        return super().event(e)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place(self.width(), move=True)

    def _place(self, width: int, move: bool) -> int:
        """Lay the controls out in lines ``width`` wide, each centred in its line; the height used."""
        left, top, right, bottom = self.margins
        lines, line, x = [], [], left
        for w in self.items:
            hint = w.sizeHint()
            if line and x + hint.width() > width - right:
                lines.append(line)
                line, x = [], left
            line.append((w, hint))
            x += hint.width() + self.spacing
        if line:
            lines.append(line)
        y = top
        for n, line in enumerate(lines):
            h, x = max(hint.height() for _, hint in line), left
            for w, hint in line:
                if move:
                    w.setGeometry(QRect(x, y + (h - hint.height()) // 2, hint.width(), hint.height()))
                x += hint.width() + self.spacing
            y += h + (self.spacing if n < len(lines) - 1 else 0)
        return y + bottom


# ─── Tab 1: CEP subsets ──────────────────────────────────────────────────────

class CepChart(QWidget):
    """The CEP of every subset in the library's order, as bars on a log scale,
    with a dot matrix of each subset's receivers underneath.  Bars appear as
    the animation reaches them; click one to jump to it.  Where the library's
    search stops short (``max_subsets``), a dashed line marks it, and the
    subsets past it show faintly at the end, for comparison."""

    picked = pyqtSignal(int)
    LEFT, RIGHT, TOP, ROW = 46, 10, 22, 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self.steps    = []
        self.rows     = []         # receivers that can be members
        self.winner   = None
        self.best_all = None       # the winner had every subset been scored
        self.searched = 0          # subsets the library scores
        self.shown    = 0
        self.current  = None
        self.phase    = 1.0
        self.final    = False
        self.setMinimumHeight(170)
        self.setMouseTracking(True)

    def set_trace(self, trace: lab.CepTrace) -> None:
        self.steps, self.rows, self.winner = trace.steps, list(trace.sane), trace.best
        self.best_all, self.searched = trace.best_all, trace.searched
        self.update()

    def set_progress(self, shown: int, current, phase: float, final: bool) -> None:
        self.shown, self.current, self.phase, self.final = shown, current, phase, final
        self.update()

    def _columns(self):
        """x centre of every column and the bar width (subset sizes are grouped)."""
        n      = len(self.steps)
        sizes  = [len(s.subset) for s in self.steps]
        breaks = [i for i in range(1, n) if sizes[i] != sizes[i - 1]]
        units  = n + 0.8 * len(breaks)
        w      = (self.width() - self.LEFT - self.RIGHT) / max(units, 1)
        xs, gap = [], 0
        for i in range(n):
            gap += 0.8 if i in breaks else 0
            xs.append(self.LEFT + (i + gap + 0.5) * w)
        return xs, w

    def _plot_bottom(self) -> float:
        return self.height() - 18 - self.ROW * len(self.rows) - 6

    def _column_at(self, x: float) -> int | None:
        if not self.steps:
            return None
        xs, w = self._columns()
        i = int(np.argmin([abs(x - c) for c in xs]))
        return i if abs(x - xs[i]) <= w * 0.6 else None

    def paintEvent(self, _event) -> None:
        pal = theme.current
        p   = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), color(pal.bg))
        draw.label(p, QPointF(self.LEFT, 11), "CEP of each subset, cm (log) · library order",
                   pal.text_dim, 8, halo=False)
        if not self.steps:
            draw.label(p, QPointF(self.width() / 2, self.height() / 2),
                       "fewer than 3 receivers kept: no subsets to score", pal.text_muted, 8.5, "center")
            return
        xs, w  = self._columns()
        top, bottom = float(self.TOP + 4), self._plot_bottom()
        ceps   = np.array([s.cep for s in self.steps], dtype=float)
        good   = ceps[np.isfinite(ceps) & (ceps > 0)]
        lo     = 10.0 ** math.floor(math.log10(good.min())) if len(good) else 0.1
        hi     = 10.0 ** math.ceil(math.log10(good.max())) if len(good) else 10.0
        hi     = max(hi, lo * 10)

        def Y(v):
            return bottom - (math.log10(max(v, lo)) - math.log10(lo)) / math.log10(hi / lo) * (bottom - top)

        for e in range(round(math.log10(lo)), round(math.log10(hi)) + 1):
            y = Y(10.0 ** e)
            p.setPen(draw.pen(pal.grid_major if e == round(math.log10(lo)) else pal.grid_minor, 1.0))
            p.drawLine(QPointF(self.LEFT - 3, y), QPointF(self.width() - self.RIGHT, y))
            draw.label(p, QPointF(self.LEFT - 6, y), f"{10.0 ** e:g}", pal.text_muted, 7, "right", halo=False)

        # The library's best so far (the winner once every subset is in)
        best = self.winner if self.final else next(
            (i for i in reversed(range(self.shown)) if self.steps[i].best and np.isfinite(ceps[i])), None)
        cc   = pal.algo["CEP"]
        for i in range(self.shown):
            cep  = ceps[i]
            cur  = i == self.current
            past = i >= self.searched          # the library never scores it
            v    = hi if not np.isfinite(cep) else cep
            if cur and self.phase < 1:
                v = lo * (v / lo) ** ease(self.phase)
            rect = QRectF(xs[i] - w * 0.35, Y(v), w * 0.7, bottom - Y(v))
            path = QPainterPath()
            path.addRect(rect)
            c = pal.accent if cur else pal.text_muted if past else cc
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color(c, 0.35 if cur else 0.06 if past else 0.12))
            p.drawPath(path)
            draw.stroke(p, path, c, 1.6 if cur else 1.0, 1.0 if cur else 0.5 if past else 0.75,
                        dash=draw.DASH if past or not np.isfinite(cep) else None, glow=False if past else None)
        if self.searched < len(self.steps):
            x = (xs[self.searched - 1] + xs[self.searched]) / 2 if self.searched else self.LEFT
            p.setPen(draw.pen(pal.text_dim, 1.0, 0.8, draw.DASH))
            p.drawLine(QPointF(x, top - 2), QPointF(x, self.height() - 16))
            draw.label(p, QPointF(x + 5, top + 6), f"library stops: {self.searched} of {len(self.steps)}",
                       pal.text_dim, 7)
        a = self.best_all
        if self.final and a is not None and a != self.winner:
            q = QPointF(xs[a], Y(ceps[a]) - 9)
            draw.marker(p, q, "star", pal.text_dim, 9, 1.2, 0.8)
            draw.label(p, QPointF(q.x(), q.y() - 11), f"best of all {len(self.steps)}", pal.text_dim, 7, "center")
        if best is not None and (self.current is None or self.phase >= 1 or best != self.current):
            q = QPointF(xs[best], Y(ceps[best]) - 9)
            draw.marker(p, q, "star", pal.algo["CEP"], 9, 1.2, fill=0.4)
            draw.label(p, QPointF(q.x(), q.y() - 11), "winner" if self.final else "best", pal.algo["CEP"],
                       7, "center")

        # Dot matrix: which receivers each subset uses
        mtop = bottom + 6
        for r, rx in enumerate(self.rows):
            y = mtop + r * self.ROW + self.ROW / 2
            draw.label(p, QPointF(self.LEFT - 6, y), f"R{rx}", pal.text_muted, 7, "right", halo=False)
            for i, s in enumerate(self.steps):
                member, live = rx in s.subset, i < min(self.shown, self.searched)
                c = pal.accent if i == self.current else pal.text_dim
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(color(c if member else pal.text_muted, (0.9 if member else 0.35) * (1 if live else 0.3)))
                p.drawEllipse(QPointF(xs[i], y), 2.6 if member else 1.1, 2.6 if member else 1.1)
        ylab, start = self.height() - 8, 0
        for i in range(1, len(self.steps) + 1):
            if i == len(self.steps) or len(self.steps[i].subset) != len(self.steps[start].subset):
                k    = len(self.steps[start].subset)
                text = "all" if start == 0 else f"{k} of {len(self.rows)}"
                draw.label(p, QPointF((xs[start] + xs[i - 1]) / 2, ylab), text, pal.text_muted, 7, "center",
                           halo=False)
                start = i

    def mousePressEvent(self, e) -> None:
        i = self._column_at(e.position().x())
        if i is not None and i < self.searched and e.button() == Qt.MouseButton.LeftButton:
            self.picked.emit(i)

    def event(self, e) -> bool:
        if e.type() == QEvent.Type.ToolTip:
            i = self._column_at(e.position().x() if hasattr(e, "position") else e.pos().x())
            if i is None:
                QToolTip.hideText()
            else:
                s    = self.steps[i]
                what = ("click to jump to this step" if i < self.searched else
                        f"past the library's search (it stops after {self.searched}): shown for comparison")
                QToolTip.showText(e.globalPos(), f"subset {i + 1}: {names(s.subset)}\nCEP {num(s.cep)} cm\n{what}",
                                  self)
            return True
        return super().event(e)


class CepTab(LabTab):
    """The library's CEP search, one subset at a time."""

    STEP_SECONDS = 1.6
    HINT         = "click: move the transmitter"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.trace = None
        self.steps = []
        self.chart = CepChart()
        self.chart.picked.connect(lambda j: self.player.seek(1 + j))
        self.chart.setToolTip("CEP of every subset the library scored; click a bar to jump to it")
        self.table = make_table(["#", "receivers", "CEP cm", "x", "y", "error cm"])
        self.table.setToolTip("Every subset the library scored, in its order; click one to jump to it")
        self.table.cellClicked.connect(lambda r, _c: self.player.seek(1 + r))
        left  = splitter(Qt.Orientation.Vertical, [self.canvas, self.chart], [500, 220])
        right = splitter(Qt.Orientation.Vertical, [self.text, self.table], [480, 230])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(splitter(Qt.Orientation.Horizontal, [left, right], [640, 430]), 1)
        lay.addWidget(self.bar)

    def compute(self) -> None:
        sc = self.sc
        self.trace = tr = lab.trace_cep(self.lib, sc.distances, sc.sane,
                                        S.cep_max_subsets(sc.settings, self.lib.receiver_count))
        self.steps = ["filter"] + list(range(tr.searched)) + ["validate"]
        rows = []
        for j, s in enumerate(tr.steps[:tr.searched]):
            err = num(float(np.hypot(*(s.pos - sc.truth)))) if sc.truth is not None else "—"
            rows.append([f"{j + 1}{' ★' if j == tr.best else ''}", names(s.subset), num(s.cep),
                         num(s.pos[0]), num(s.pos[1]), err])
        fill_table(self.table, rows, bold={tr.best})
        self.chart.set_trace(tr)
        self.restart(len(self.steps))

    # ─── Steps ───────────────────────────────────────────────────────────────

    def describe(self, k: int) -> str:
        kind, tr = self.steps[k], self.trace
        if kind == "filter":
            return f"filter: {len(self.sc.sane)} of {len(self.sc.coords)} receivers kept"
        if kind == "validate":
            if tr.best is None:
                return "no subset of 3 receivers: the library holds its last fix"
            w = tr.steps[tr.best]
            return (f"validation: winner {names(w.subset)} (CEP {num(w.cep)} cm) "
                    + ("is outside the receivers' box: rejected" if tr.rejected else "is inside the box: accepted"))
        s = tr.steps[kind]
        return (f"subset {kind + 1} of {tr.searched}: {names(s.subset)} → CEP {num(s.cep)} cm"
                + (" · best so far" if s.best else ""))

    def on_step(self, k: int) -> None:
        kind = self.steps[k]
        last = self.trace.searched - 1 if kind == "validate" else -1 if kind == "filter" else kind
        show_rows(self.table, last, kind if isinstance(kind, int) else self.trace.best if kind == "validate" else None)

    def on_phase(self) -> None:
        kind = self.steps[self.player.index]
        n    = len(self.trace.steps)
        if kind == "filter":
            self.chart.set_progress(0, None, 1.0, False)
        elif kind == "validate":
            self.chart.set_progress(n, None, 1.0, True)
        else:
            self.chart.set_progress(kind + 1, kind, self.player.phase, False)
        self.canvas.update()

    # ─── Canvas ──────────────────────────────────────────────────────────────

    def paint_scene(self, p: QPainter, view: draw.View, k: int, t: float) -> None:
        draw_world(p, view, self.sc.world)
        kind = self.steps[k]
        if kind == "filter":
            draw_filter(p, view, self.sc, t)
            draw_truth(p, view, self.sc.truth)
            draw_legend(p, view, filter_keys(self.sc), self.HINT, receiver_points(view, self.sc))
        elif kind == "validate":
            self._paint_validate(p, view, t)
        else:
            self._paint_subset(p, view, kind, t)

    def _ghost(self, j: int):
        """The best subset before subset ``j`` was scored."""
        prior = [i for i in range(j) if self.trace.steps[i].best]
        return self.trace.steps[prior[-1]] if prior else None

    def _keys(self, s, fix, ghost, winner: bool = False) -> list:
        pal, cc = theme.current, theme.current.algo["CEP"]
        keys = [Key("plus", pal.algo["OLS"], "OLS seed"),
                Key("diamond", cc, "winner" if winner else "LM fit of the subset"),
                Key("tick", pal.status["echo"], "residual r"),
                Key("ellipse", cc, "1σ ellipse (Σ)"),
                Key("dash", cc, "CEP circle", f"{num(s.cep)} cm", strong=True)]
        if ghost is not None:
            keys.append(Key("faint:diamond", cc, "best so far", f"{num(ghost.cep)} cm"))
        if self.sc.truth is not None:
            keys.append(Key("crosshair", pal.truth, "true position"))
        return keys

    def _paint_subset(self, p, view, j: int, t: float) -> None:
        pal, sc = theme.current, self.sc
        s, ghost = self.trace.steps[j], self._ghost(j)
        a   = ease(t)
        fix = s.seed + (s.pos - s.seed) * a
        members = set(s.subset)
        draw_receivers(p, view, sc, lit=members, rings={i: pal.ray for i in members}, thick=members)
        draw_ranges(p, view, sc, s.subset)
        self._fit_marks(p, view, s, fix, a, ghost, labels=True)
        draw_truth(p, view, sc.truth)
        draw_offscreen(p, view, fix, pal.algo["CEP"], f"fix {pt(fix)}")
        avoid = receiver_points(view, sc) + [view.px(*fix), view.px(*s.seed)]
        box   = draw_legend(p, view, self._keys(s, fix, ghost), self.HINT, avoid)
        self._zoom(p, view, s, fix, a, ghost, avoid, [box])

    def _fit_marks(self, p, view, s, fix, a: float, ghost, labels: bool) -> None:
        """Seed, fit, residual ticks, 1σ ellipse, CEP circle and the best-so-far ghost."""
        pal, sc = theme.current, self.sc
        cc = pal.algo["CEP"]
        if ghost is not None:
            draw.marker(p, view.px(*ghost.pos), "diamond", cc, 11, 1.2, alpha=0.4)
        for i in s.subset:
            c = sc.coords[i]
            u = fix - c
            L = float(np.hypot(*u))
            if L < 1e-9:
                continue
            on = c + u / L * sc.distances[i]
            draw.polyline(p, view, [on, fix], pal.status["echo"], 2.0, 0.9)
            if labels and abs(L - sc.distances[i]) * view.scale > 18:
                draw.label(p, view.px(*((on + fix) / 2)), num(L - sc.distances[i], True), pal.status["echo"], 7,
                           "center")
        q, seed = view.px(*fix), view.px(*s.seed)
        draw.marker(p, seed, "plus", pal.algo["OLS"], 11, 1.4)
        if math.hypot(q.x() - seed.x(), q.y() - seed.y()) > 6:
            draw.arrow(p, seed, q, cc, 1.2, 6.0, 0.8)
        k = min(max((a - 0.55) / 0.45, 0.0), 1.0)
        if k > 0 and math.isfinite(s.cep):          # CEP ∞: Σ is the library's 1000·I placeholder
            draw.ellipse_cov(p, view, fix, s.cov, cc, 1.0, 1.0, 0.85 * k)
            draw.circle(p, view, fix, s.cep, cc, 1.2, k, draw.DASH)
        draw.marker(p, q, "diamond", cc, 11, 1.5, fill=0.25)

    def _zoom(self, p, view, s, fix, a: float, ghost, avoid, busy) -> None:
        """The magnifier round the fix, where the CEP circle and residuals show."""
        pal, sc = theme.current, self.sc
        R = 1.0
        if math.isfinite(s.cep):
            R = max(R, 1.6 * s.cep, 1.4 * math.sqrt(max(np.linalg.eigvalsh((s.cov + s.cov.T) / 2).max(), 0.0)))
        res = np.abs(np.hypot(*(fix - sc.coords[s.subset]).T) - sc.distances[s.subset])
        R = max(R, 1.2 * float(res.max()) if len(res) else 0.0)
        if sc.truth is not None:
            e = float(np.hypot(*(fix - sc.truth)))
            if e < 5 * R:
                R = max(R, 1.25 * e)

        def paint(q, v):
            for i in s.subset:
                draw.polyline(q, v, arc_points(sc.coords[i], sc.distances[i], fix, 3 * R), pal.ray, 1.0, 0.6)
            self._fit_marks(q, v, s, fix, a, ghost, labels=False)
            draw_truth(q, v, sc.truth)

        magnifier(p, view, fix, R, paint, avoid, busy)

    def _paint_validate(self, p, view, t: float) -> None:
        pal, sc, tr = theme.current, self.sc, self.trace
        x0, y0, x1, y1 = tr.box
        draw.polyline(p, view, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], pal.text_dim, 1.2, 0.85 * ease(t),
                      draw.DASH, closed=True)
        box_key = Key("dash", pal.text_dim, "receivers' box")
        if tr.best is None:
            draw_receivers(p, view, sc, lit=set(sc.sane))
            draw_truth(p, view, sc.truth)
            top = banner(p, view, "fewer than 3 receivers kept: no fix", pal.status["lost"])
            draw_legend(p, view, [box_key], self.HINT, receiver_points(view, sc), [top])
            return
        s = tr.steps[tr.best]
        members = set(s.subset)
        draw_receivers(p, view, sc, lit=members, rings={i: pal.ray for i in members}, thick=members)
        draw_ranges(p, view, sc, s.subset)
        self._fit_marks(p, view, s, s.pos, 1.0, None, labels=True)
        draw_truth(p, view, sc.truth)
        held = sc.held if sc.held is not None and np.hypot(*(sc.held - s.pos)) > 1e-6 else None
        keys = [box_key] + self._keys(s, s.pos, None, winner=True)
        if tr.rejected and held is not None:
            draw.arrow(p, view.px(*s.pos), view.px(*held), pal.text_dim, 1.0, 6.0, 1.0, draw.DOT)
            draw.marker(p, view.px(*held), "diamond", pal.algo["CEP"], 11, 1.2, alpha=0.5)
            keys.append(Key("faint:diamond", pal.algo["CEP"], "last good fix", pt(held)))
        ok   = not tr.rejected
        top  = banner(p, view, f"{names(s.subset)}: {'inside' if ok else 'outside'} the receivers' box → "
                               f"{'accepted' if ok else 'rejected'}", pal.status["ok"] if ok else pal.status["lost"])
        draw_offscreen(p, view, s.pos, pal.algo["CEP"], f"fix {pt(s.pos)}")
        avoid = receiver_points(view, sc) + [view.px(*s.pos)]
        box   = draw_legend(p, view, keys, self.HINT, avoid, [top])
        self._zoom(p, view, s, s.pos, 1.0, None, avoid, [box, top])

    # ─── Explanation ─────────────────────────────────────────────────────────

    def html(self, k: int) -> str:
        kind = self.steps[k]
        if kind == "filter":
            return filter_html(self.sc)
        if kind == "validate":
            return self._validate_html()
        return self._subset_html(kind)

    def _subset_html(self, j: int) -> str:
        pal, sc, tr = theme.current, self.sc, self.trace
        s    = tr.steps[j]
        idx  = list(s.subset)
        c, d = sc.coords[idx], sc.distances[idx]
        rng  = np.hypot(*(s.pos - c).T)
        J    = (s.pos - c) / np.maximum(rng, 1e-9)[:, None]
        rows = [[f"R{i}", num(d[n]), num(rng[n]), span(num(s.residuals[n], True), pal.status["echo"]),
                 num(s.weights[n]), f"{num(J[n, 0], True)} {num(J[n, 1], True)}"] for n, i in enumerate(idx)]
        cov  = s.cov
        sx, sy = math.sqrt(abs(cov[0, 0])), math.sqrt(abs(cov[1, 1]))
        if math.isfinite(s.cep):
            info = matrix("JᵀWJ", np.linalg.inv(cov))
        else:        # the library caught a LinAlgError and put 1000·I in Σ's place
            info = ("<p class='bad'>JᵀWJ is singular: these receivers cannot fix both x and y, so the "
                    "library sets Σ = 1000·I and CEP = ∞.</p>")
        prior = [i for i in range(j) if tr.steps[i].best]
        if not prior:
            lead = span("first subset: best so far", pal.status["ok"])
        elif s.best:
            b = tr.steps[prior[-1]]
            lead = span(f"yes: beats {num(b.cep)} cm of {names(b.subset)}", pal.status["ok"])
        else:
            b = tr.steps[prior[-1]]
            lead = span(f"no: {names(b.subset)} still leads with {num(b.cep)} cm", pal.text_dim)
        whole = " (every kept receiver: the library tries the full set first)" if j == 0 else ""
        return (
            f"<h3>Subset {j + 1} of {tr.searched}: {names(idx)}</h3>"
            f"<p class='muted'>_compute_cep(distances, {idx}){whole}<br>{self._search_line()}</p>"
            f"<p><b>1 · seed</b> <span class='dim'>_ols</span>: squaring and subtracting the range equations "
            f"makes them linear; one least-squares solve.<br>p<sub>0</sub> = {pt(s.seed)}</p>"
            f"<p><b>2 · fit</b> <span class='dim'>_nlls</span>: least_squares(method=\"lm\") from p<sub>0</sub> "
            f"→ <b>p = {pt(s.pos)}</b>, {num(float(np.hypot(*(s.pos - s.seed))))} cm from the seed</p>"
            "<p><b>3 · residuals and weights</b><br>"
            "<span class='eq'>r<sub>i</sub> = |p − R<sub>i</sub>| − d<sub>i</sub></span> &nbsp; "
            "<span class='eq'>w<sub>i</sub> = 1 / (|r<sub>i</sub>| + 0.1)</span> &nbsp; "
            "<span class='eq'>J<sub>i</sub> = (p − R<sub>i</sub>) / |p − R<sub>i</sub>|</span></p>"
            + html_table(["", "d", "|p − R|", "r", "w", "J (x, y)"], rows)
            + "<p><b>4 · covariance</b> <span class='eq'>Σ = (JᵀWJ)⁻¹</span>, W = diag(w)</p>"
            + info + matrix("Σ", cov)
            + f"<p>σ<sub>x</sub> = √Σ<sub>xx</sub> = {num(sx)} cm, σ<sub>y</sub> = √Σ<sub>yy</sub> = {num(sy)} cm</p>"
            "<p><b>5 · CEP</b> <span class='eq'>CEP = 0.59 · √(σ<sub>x</sub>² + σ<sub>y</sub>²) · √2</span><br>"
            + (f"= 0.59 · √({num(sx * sx)} + {num(sy * sy)}) · 1.414 = <b>{num(s.cep)} cm</b></p>"
               if math.isfinite(s.cep) else "= <b>∞</b>: this subset can never win</p>")
            + f"<p><b>Best so far?</b> {lead}</p>"
            "<p class='muted'>Receivers that agree with the fit get large weights (w = 10 for an exact fit, "
            "≈ 0.1 for one 10 cm off), so a subset whose readings agree scores a small CEP. "
            "So Σ ranks the subsets; it is not the fix's covariance in cm² (the weights come from how well "
            "the subset fits, not from the noise). The simulator's CEP ellipse uses the winning subset's "
            "geometry and the noise LM sees over every kept receiver instead. "
            "The library takes J from the optimiser's result (result.jac, by finite differences); the rows "
            "above are the exact values it approximates.</p>"
        )

    def _search_line(self) -> str:
        """How far the library's subset search goes under the simulator's CEP subsets setting."""
        tr, setting = self.trace, CEP_SUBSETS.get(self.sc.settings.cep_subsets, self.sc.settings.cep_subsets)
        m = S.cep_max_subsets(self.sc.settings, self.lib.receiver_count)
        if tr.searched < len(tr.steps):
            return (f"The library scores the first {tr.searched} of the {len(tr.steps)} subsets: "
                    f"max_subsets = {m} (CEP subsets: {setting}).")
        if len(tr.steps) == 1:
            return (f"The library scores the only subset, all three kept receivers (CEP subsets: {setting}, "
                    f"max_subsets = {m}).")
        return f"The library scores all {len(tr.steps)} subsets (CEP subsets: {setting}, max_subsets = {m})."

    def _limit_html(self) -> str:
        """What the search limit leaves out, and which subset would win without it (or with it)."""
        sc, tr = self.sc, self.trace
        n, m = len(tr.steps), S.CONTROL_MAX_SUBSETS
        if tr.searched < n:
            sizes, skipped = [], []
            for k in sorted({len(s.subset) for s in tr.steps[1:]}):
                total  = sum(len(s.subset) == k for s in tr.steps)
                scored = sum(len(s.subset) == k for s in tr.steps[:tr.searched])
                if scored == 0:
                    skipped.append(f"{k}-receiver")
                else:
                    sizes.append(f"all {total} {k}-receiver subsets" if scored == total else
                                 f"{scored} of the {total} {k}-receiver subsets")
            what  = (f"the full set, then {', '.join(sizes)}" if sizes else "the full set")
            what += f"; it never scores a {' or '.join(skipped)} subset" if skipped else ""
            other, label = tr.best_all, f"Scoring all {n} (CEP subsets: All)"
            out = f"<p>{self._search_line()} That is {what}."
        elif n > m:
            other, label = lab.winner(tr.steps, m), f"UltraGPS-Control (max_subsets = {m}, the library default)"
            out = f"<p>{self._search_line()} {label} stops after {m}."
        else:
            return f"<p class='muted'>{self._search_line()}</p>"
        if other is None or tr.best is None:
            return out + "</p>"
        o = tr.steps[other]
        if other == tr.best:
            return out + f" {label} would pick the same winner.</p>"
        err = (f", error {num(float(np.hypot(*(o.pos - sc.truth))))} cm"
               f" (this winner: {num(float(np.hypot(*(tr.steps[tr.best].pos - sc.truth))))} cm)"
               if sc.truth is not None else "")
        return out + f" {label} would pick <b>{names(o.subset)}</b> instead: CEP {num(o.cep)} cm{err}.</p>"

    def _validate_html(self) -> str:
        pal, sc, tr = theme.current, self.sc, self.trace
        if tr.best is None:
            return ("<h3>Validation</h3><p>Fewer than 3 receivers were kept, so <span class='dim'>"
                    "_find_best_subset</span> has nothing to score. The library returns its last good CEP "
                    "fix and reports success = False.</p>")
        s = tr.steps[tr.best]
        x0, y0, x1, y1 = tr.box
        okx, oky = x0 <= s.pos[0] <= x1, y0 <= s.pos[1] <= y1
        def tick(ok):
            return span("✓", pal.status["ok"]) if ok else span("✗", pal.status["lost"])
        # A snapshot's library had no last good fix when it handed back the rejected fix itself
        held = sc.held if sc.held is not None and np.hypot(*(sc.held - s.pos)) > 1e-6 else None
        if not tr.rejected:
            err  = f" Error from the true position: <b>{num(float(np.hypot(*(s.pos - sc.truth))))} cm</b>." \
                   if sc.truth is not None else ""
            what = span("Inside: accepted.", pal.status["ok"]) + f" This is the library's CEP fix.{err}"
        elif held is not None:
            what = (span("Outside: rejected.", pal.status["lost"]) +
                    f" The library returns its last good CEP fix {pt(held)} and success = False.")
        elif sc.source == "lab":
            what = (span("Outside: rejected.", pal.status["lost"]) +
                    " In the running simulator the library would return its last good CEP fix; the lab has "
                    "none, so it hands back this fix with success = False.")
        else:
            what = (span("Outside: rejected.", pal.status["lost"]) +
                    " The library had no last good CEP fix in that frame, so it returned this fix itself, with "
                    "success = False.")
        order = sorted(range(tr.searched), key=lambda i: tr.steps[i].cep)[:6]
        rows  = [[f"{i + 1}", names(tr.steps[i].subset), num(tr.steps[i].cep), pt(tr.steps[i].pos)] for i in order]
        return (
            "<h3>Validation: _validate_position</h3>"
            f"<p>Winner: <b>{names(s.subset)}</b>, CEP {num(s.cep)} cm, fix {pt(s.pos)}</p>"
            f"<p>The fix must lie in the receivers' bounding box:<br>"
            f"<span class='eq'>{num(x0)} ≤ x ≤ {num(x1)}</span> &nbsp; x = {num(s.pos[0])} {tick(okx)}<br>"
            f"<span class='eq'>{num(y0)} ≤ y ≤ {num(y1)}</span> &nbsp; y = {num(s.pos[1])} {tick(oky)}</p>"
            f"<p>{what}</p>" + self._limit_html() + "<p class='dim'>Smallest CEPs:</p>"
            + html_table(["#", "receivers", "CEP cm", "fix"], rows)
        )


# ─── Tab 2: Blocking ─────────────────────────────────────────────────────────

class SweepWorker(QThread):
    """Runs ``engine.lab.sweep`` (a few seconds) off the GUI thread, on its own
    library instance so the tabs can keep using theirs.  Setting ``cancel``
    stops it within one offset."""

    done = pyqtSignal(int, object)      # token, Sweep, None (cancelled) or an error message

    def __init__(self, token: int, config_path: str, limits, distances, idx, receiver: int,
                 max_offset: float, steps: int, settings, parent=None):
        super().__init__(parent)
        self.token  = token
        self.cancel = threading.Event()
        self.args   = (config_path, limits, np.array(distances, dtype=float), list(idx), receiver,
                       max_offset, steps, settings)

    def run(self) -> None:
        config_path, limits, d, idx, receiver, max_offset, steps, settings = self.args
        lib = UltraGPSPositionLib(config_path)
        lib.max_distance, lib.max_differential = limits
        try:
            result = lab.sweep(lib, d, idx, receiver, max_offset, steps, settings, self.cancel)
        except Exception as exc:        # report it: a dead worker would leave the tab busy forever
            result = f"sweep failed: {exc}"
        self.done.emit(self.token, result)


def no_cep_fix(sweep: lab.Sweep, k: int) -> str:
    """Why the library's CEP has no fix at step ``k`` of a sweep."""
    if sweep.kept and len(sweep.kept[k]) < 3:
        return "fewer than 3 receivers kept"
    return "no subset fixes both x and y"


def cep_switches(sweep: lab.Sweep) -> list:
    """Where the library's CEP changes subset, loses its fix or gets it back: ``[(step, offset, text)]``."""
    out = []
    for k in range(1, len(sweep.offsets)):
        a, b = set(sweep.cep_subsets[k - 1]), set(sweep.cep_subsets[k])
        at   = f"+{sweep.offsets[k]:.0f} cm"
        if a and not b:              # not a choice of CEP's: the library has nothing to solve with
            out.append((k, float(sweep.offsets[k]), f"CEP has no fix at {at}: {no_cep_fix(sweep, k)}"))
        elif b and not a:
            out.append((k, float(sweep.offsets[k]), f"CEP has a fix again at {at}: {names(sorted(b))}"))
        elif a != b:
            parts = (["drops " + names(sorted(a - b))] if a - b else []) + \
                    (["takes " + names(sorted(b - a))] if b - a else [])
            out.append((k, float(sweep.offsets[k]), f"CEP {' and '.join(parts)} at {at}"))
        if a and b and sweep.cep_rejected[k] != sweep.cep_rejected[k - 1]:
            where = "leaves" if sweep.cep_rejected[k] else "comes back into"
            out.append((k, float(sweep.offsets[k]), f"CEP fix {where} the box at {at}"))
    return out


class ErrorBars(QWidget):
    """Each algorithm's distance from the true position, a bar appearing per step."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows    = []          # [(algorithm, error cm or None, note)]
        self.truth   = True
        self.shown   = 0
        self.current = None
        self.phase   = 1.0
        self.ranked  = False
        self.setMinimumHeight(110)
        self.setToolTip("How far each algorithm's fix is from the true transmitter position")

    def set_rows(self, rows, truth: bool) -> None:
        self.rows, self.truth = rows, truth
        self.update()

    def set_progress(self, shown: int, current, phase: float, ranked: bool) -> None:
        self.shown, self.current, self.phase, self.ranked = shown, current, phase, ranked
        self.update()

    def paintEvent(self, _event) -> None:
        pal = theme.current
        p   = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), color(pal.bg))
        draw.label(p, QPointF(10, 11), "distance from the true position, cm" if self.truth else
                   "no true position in this snapshot: errors unknown", pal.text_dim, 8, halo=False)
        if not self.rows:
            return
        top   = 24.0
        h     = (self.height() - top - 4) / len(self.rows)
        x0    = 84.0
        bw    = max(self.width() - x0 - 92, 20)
        errs  = [e for _, e, _ in self.rows if e is not None and math.isfinite(e)]
        hi    = nice_ceil(max(errs)) if errs else 1.0
        order = sorted((r for r in self.rows if r[1] is not None and math.isfinite(r[1])), key=lambda r: r[1])
        rank  = {a: n + 1 for n, (a, _, _) in enumerate(order)}      # ties keep the rows' order, as the ranking
        p.setPen(draw.pen(pal.grid_major, 1.0))
        p.drawLine(QPointF(x0, top - 2), QPointF(x0, self.height() - 4))
        p.setPen(draw.pen(pal.grid_minor, 1.0))
        p.drawLine(QPointF(x0 + bw, top - 2), QPointF(x0 + bw, self.height() - 4))
        draw.label(p, QPointF(x0 + bw, top - 8), f"{hi:g} cm", pal.text_muted, 7, "center", halo=False)
        for r, (a, e, note) in enumerate(self.rows):
            y, c = top + (r + 0.5) * h, pal.algo[a]
            live = r < self.shown
            draw.marker(p, QPointF(16, y), theme.ALGO_SHAPES[a], c, 8, 1.2, 1.0 if live else 0.3)
            draw.label(p, QPointF(27, y), a, c, 8, bold=r == self.current, halo=False, alpha=1.0 if live else 0.3)
            if not live:
                continue
            if e is None or not math.isfinite(e):
                draw.label(p, QPointF(x0 + 6, y), note or "no fix", pal.text_muted, 7.5, halo=False)
                continue
            grow = ease(self.phase) if r == self.current else 1.0
            path = QPainterPath()
            path.addRect(QRectF(x0, y - h * 0.28, max(e / hi * bw * grow, 1.0), h * 0.56))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color(c, 0.32 if r == self.current else 0.15))
            p.drawPath(path)
            draw.stroke(p, path, c, 1.5 if r == self.current else 1.0)
            text = f"{num(e)} cm" + (f"  #{rank[a]}" if self.ranked else "")
            draw.label(p, QPointF(x0 + e / hi * bw * grow + 6, y), text, c, 7.5, halo=False)


class SweepCharts(QWidget):
    """Small multiples: each algorithm's error as one receiver's reading grows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sweep    = None
        self.errors   = {}
        self.switches = []
        self.cursor   = 0.0
        self.title    = ""
        self.setMinimumHeight(110)
        self.setToolTip("Each algorithm's error against the offset added to the swept receiver; "
                        "dotted lines: where the library's CEP changes subset; dashed red line: where the "
                        "reading passes the top band and the library's bounds check drops it")

    def set_sweep(self, sweep: lab.Sweep | None, truth, switches) -> None:
        self.sweep, self.switches, self.errors = sweep, switches, {}
        if sweep is not None:
            for a, path in sweep.paths.items():
                ref = truth if truth is not None else path[0]
                self.errors[a] = np.hypot(*(path - np.asarray(ref, dtype=float)).T)
            what = "error" if truth is not None else "shift from the offset-0 fix"
            self.title = f"{what}, cm, against the offset on R{sweep.receiver}"
        self.update()

    def set_cursor(self, offset: float) -> None:
        self.cursor = offset
        self.update()

    def columns(self, count: int) -> int:
        """The grid whose plots are biggest, aiming at plots twice as wide as tall."""
        def size(cols):
            rows = math.ceil(count / cols)
            plot_w = (self.width() - 8) / cols - 38
            plot_h = (self.height() - 24) / rows - 29
            return min(plot_w / 2, plot_h)
        return max(range(1, max(count, 1) + 1), key=size)

    def paintEvent(self, _event) -> None:
        pal = theme.current
        p   = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), color(pal.bg))
        if self.sweep is None:
            draw.label(p, QPointF(self.width() / 2, self.height() / 2),
                       "pick a receiver and press Run sweep", pal.text_muted, 8.5, "center")
            return
        title = self.title
        if self.sweep.dropped is not None:
            title += f" · red dashes: R{self.sweep.receiver} too far, dropped"
        draw.label(p, QPointF(10, 11), title, pal.text_dim, 8, halo=False)
        algos = [a for a in STATIC_ALGOS if a in self.errors]
        fm    = QFontMetricsF(theme.mono_font(7.5))
        cols  = self.columns(len(algos))
        rows  = math.ceil(len(algos) / cols)
        cw, ch = (self.width() - 8) / cols, (self.height() - 24) / rows
        finite = np.concatenate([v[np.isfinite(v)] for v in self.errors.values()])
        hi    = nice_ceil(float(finite.max())) if len(finite) else 1.0
        offs  = self.sweep.offsets
        xmax  = float(offs[-1]) or 1.0
        for n, a in enumerate(algos):
            cell = QRectF(4 + (n % cols) * cw, 22 + (n // cols) * ch, cw, ch)
            plot = cell.adjusted(30, 15, -8, -14)
            c    = pal.algo[a]
            err  = self.errors[a]
            now  = float(np.interp(self.cursor, offs, np.nan_to_num(err, nan=np.inf)))
            draw.marker(p, QPointF(cell.left() + 10, cell.top() + 7), theme.ALGO_SHAPES[a], c, 7, 1.1)
            head = draw.label(p, QPointF(cell.left() + 18, cell.top() + 7), a, c, 7.5, bold=True, halo=False)
            for value in (f"{num(now)} cm", num(now)):          # drop the unit where it would not fit
                if fm.horizontalAdvance(value) + 6 < plot.right() - head.right():
                    draw.label(p, QPointF(plot.right(), cell.top() + 7), value, c, 7.5, "right", halo=False)
                    break

            def X(v):
                return plot.left() + v / xmax * plot.width()

            def Y(v):
                return plot.bottom() - min(v, hi) / hi * plot.height()

            p.setPen(draw.pen(pal.grid_major, 1.0))
            p.drawLine(plot.bottomLeft(), plot.bottomRight())
            p.drawLine(plot.bottomLeft(), plot.topLeft())
            p.setPen(draw.pen(pal.grid_minor, 1.0))
            p.drawLine(QPointF(plot.left(), Y(hi)), QPointF(plot.right(), Y(hi)))
            draw.label(p, QPointF(plot.left() - 4, Y(hi)), f"{hi:g}", pal.text_muted, 6.5, "right", halo=False)
            draw.label(p, QPointF(plot.left() - 4, plot.bottom()), "0", pal.text_muted, 6.5, "right", halo=False)
            draw.label(p, QPointF(plot.right(), plot.bottom() + 8), f"+{xmax:g}", pal.text_muted, 6.5, "right",
                       halo=False)
            for _, off, _ in self.switches:
                p.setPen(draw.pen(pal.algo["CEP"] if a == "CEP" else pal.text_muted,
                                  1.0, 0.8 if a == "CEP" else 0.4, draw.DOT))
                p.drawLine(QPointF(X(off), plot.top()), QPointF(X(off), plot.bottom()))
            if self.sweep.dropped is not None:
                x = X(float(offs[self.sweep.dropped]))
                p.setPen(draw.pen(pal.status["lost"], 1.0, 0.7, draw.DASH))
                p.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            full = draw.polyline_path([QPointF(X(o), Y(v)) for o, v in zip(offs, err) if math.isfinite(v)])
            draw.stroke(p, full, c, 1.0, 0.3, glow=False)
            done = [QPointF(X(o), Y(v)) for o, v in zip(offs, err) if math.isfinite(v) and o <= self.cursor]
            if math.isfinite(now):
                done.append(QPointF(X(self.cursor), Y(now)))
            draw.stroke(p, draw.polyline_path(done), c, 1.4)
            p.setPen(draw.pen(pal.accent, 1.0, 0.5))
            p.drawLine(QPointF(X(self.cursor), plot.top()), QPointF(X(self.cursor), plot.bottom()))
            if math.isfinite(now):
                draw.marker(p, QPointF(X(self.cursor), Y(now)), "circle", c, 5, 1.2, fill=0.6)


ALGO_HOW = {
    "OLS":    "Every kept receiver counts equally in one linear solve, so a bad range shifts it directly.",
    "LM":     "Least squares over every kept receiver: one bad range's error is shared out among all the "
              "residuals, dragging the fix toward it.",
    "CEP":    "Scores subsets of 3+ (only the first 15 with CEP subsets First 15, as UltraGPS-Control) and keeps "
              "the most self-consistent one, so it can leave a bad receiver out; must land inside the receivers' "
              "box.",
    "CEP4":   "Like CEP but subsets of 4+ only, with the textbook CEP: needs 4 receivers that agree.",
    "ROBUST": "Soft-L1 loss: residuals beyond the knee count linearly, not squared, so an outlier pulls less.",
    "BIAS":   "Also fits one offset shared by all receivers: absorbs a common timing error, not one bad range.",
    "RANSAC": "Every 3-receiver fit votes; receivers within the tolerance agree; the best-supported fit is "
              "refitted on its agreeing receivers.",
}


class BlockingTab(LabTab):
    """Each algorithm's fix in turn, and a sweep of one receiver going wrong."""

    STEP_SECONDS  = 1.4
    SWEEP_SECONDS = 0.14
    HINT          = "click: move the transmitter"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fixes       = {}
        self.sweep       = None
        self.switches    = []
        self.story       = "fixes"
        self.steps       = []
        self.sweep_steps = SWEEP_STEPS
        self._token      = 0
        self._worker     = None

        self.b_fixes = QPushButton("Fixes")
        self.b_fixes.setToolTip("Step through the filter and each algorithm's fix on this scenario")
        self.b_sweep = QPushButton("Sweep")
        self.b_sweep.setToolTip("Replay the last sweep: every fix moving as one receiver's reading grows")
        self.b_sweep.setEnabled(False)
        group = QButtonGroup(self)
        for b in (self.b_fixes, self.b_sweep):
            b.setCheckable(True)
            group.addButton(b)
        self.b_fixes.clicked.connect(lambda: self._set_story("fixes"))
        self.b_sweep.clicked.connect(lambda: self._set_story("sweep", play=True))
        self.rx = QComboBox()
        self.rx.setToolTip("Receiver whose reading the sweep makes longer, as when it hears a reflection")
        self.max_off = QSpinBox()
        self.max_off.setRange(10, 300)
        self.max_off.setValue(MAX_OFFSET_CM)
        self.max_off.setSuffix(" cm")
        self.max_off.setToolTip("Largest offset the sweep adds to the reading")
        self.run_button = QPushButton("Run sweep")
        self.run_button.setToolTip("Add 0 up to the largest offset to the receiver's reading and solve with "
                                   "every algorithm at each step (runs in the background)")
        self.run_button.clicked.connect(self.run_sweep)
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setFixedWidth(90)
        self.busy.setTextVisible(False)
        self.busy.setToolTip("Sweeping: the library solves every algorithm at each offset")
        self.busy.hide()
        controls = QHBoxLayout()
        controls.setContentsMargins(8, 4, 8, 4)
        controls.addWidget(self.b_fixes)
        controls.addWidget(self.b_sweep)
        controls.addSpacing(16)
        sweep_label = QLabel("Sweep")
        sweep_label.setProperty("role", "dim")
        controls.addWidget(sweep_label)
        controls.addWidget(self.rx)
        up_to = QLabel("up to")
        up_to.setProperty("role", "dim")
        controls.addWidget(up_to)
        controls.addWidget(self.max_off)
        controls.addWidget(self.run_button)
        controls.addWidget(self.busy)
        controls.addStretch(1)

        self.bars   = ErrorBars()
        self.charts = SweepCharts()
        self.stack  = QStackedWidget()
        self.stack.addWidget(self.bars)
        self.stack.addWidget(self.charts)
        left = splitter(Qt.Orientation.Vertical, [self.canvas, self.stack], [520, 200])
        lay  = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addLayout(controls)
        lay.addWidget(splitter(Qt.Orientation.Horizontal, [left, self.text], [640, 430]), 1)
        lay.addWidget(self.bar)

    # ─── Computing ───────────────────────────────────────────────────────────

    def set_scenario(self, sc: Scenario, lib) -> None:
        self.stop()                      # a sweep still running belongs to the old scenario
        super().set_scenario(sc, lib)

    def compute(self) -> None:
        sc = self.sc
        self.fixes   = lab.static_solve(self.lib, sc.distances, sc.sane, sc.settings)
        self._token += 1
        self.sweep   = None
        self.charts.set_sweep(None, None, [])
        self.b_sweep.setEnabled(False)
        keep  = self.rx.currentData()
        sweep = self.inputs("RANSAC")    # with Bounds only, the simulator algorithms see more than the library
        self.rx.blockSignals(True)
        self.rx.clear()
        for i in sweep:
            self.rx.addItem(f"R{i}" if i in sc.sane else f"R{i} (jumped)", i)
        bad  = [i for i in sweep if sc.offsets[i] or sc.states[i] != "on"]   # the story's suspect first
        pick = keep if keep in bad else bad[0] if bad else keep if keep in sweep else None
        if pick is not None:
            self.rx.setCurrentIndex(self.rx.findData(pick))
        self.rx.blockSignals(False)
        self._enable_run()
        self.bars.set_rows([(a, self._error(a), self._note(a)) for a in STATIC_ALGOS], sc.truth is not None)
        self._set_story("fixes")

    def inputs(self, a: str) -> list:
        """The receivers algorithm ``a`` is given: the library filter's kept ones,
        or with Bounds only, every one within the bounds for the simulator's own."""
        sc = self.sc
        if S.ALGORITHMS[a].source == "library" or sc.settings.filter_mode == "library":
            return sc.sane
        return [r.receiver for r in sc.filter.rows if r.in_bounds]

    def _error(self, a: str) -> float | None:
        e = self.fixes.get(a)
        if e is None or e.pos is None or self.sc.truth is None:
            return None
        return float(np.hypot(*(np.asarray(e.pos) - self.sc.truth)))

    def ranking(self) -> list:
        """The algorithms, smallest error first (unknown errors last; ties in STATIC_ALGOS order)."""
        return sorted(STATIC_ALGOS, key=lambda a: (self._error(a) is None, self._error(a) or 0.0))

    def _note(self, a: str) -> str:
        e = self.fixes.get(a)
        return "needs 3 receivers" if e is None else e.note

    def _set_story(self, story: str, play: bool = False) -> None:
        if story == "sweep" and self.sweep is None:
            story = "fixes"
        self.story = story
        self.b_fixes.setChecked(story == "fixes")
        self.b_sweep.setChecked(story == "sweep")
        self.stack.setCurrentWidget(self.charts if story == "sweep" else self.bars)
        if story == "sweep":
            self.steps = list(range(len(self.sweep.offsets)))
            self.restart(len(self.steps), self.SWEEP_SECONDS, play)
        else:
            self.steps = ["filter", *STATIC_ALGOS, "summary"]
            self.restart(len(self.steps), self.STEP_SECONDS, play)

    def _enable_run(self) -> None:
        self.run_button.setEnabled(self._worker is None and self.rx.count() > 0
                                   and self.sc is not None and len(self.inputs("RANSAC")) >= 3)
        self.run_button.setText("Sweeping…" if self._worker is not None else "Run sweep")

    def run_sweep(self) -> None:
        """Start the sweep in a worker thread; the tab stays usable meanwhile."""
        if self._worker is not None or self.sc is None or self.rx.currentData() is None:
            return
        sc, lib = self.sc, self.lib
        self._worker = SweepWorker(self._token, lib._config_path, (lib.max_distance, lib.max_differential),
                                   sc.distances, sc.sane, self.rx.currentData(), float(self.max_off.value()),
                                   self.sweep_steps, copy.copy(sc.settings), self)
        self._worker.done.connect(self._on_sweep)     # a method, not a lambda: queued safely if the tab goes first
        self.busy.show()
        self._enable_run()
        self.bar.set_text(f"sweeping R{self.rx.currentData()} from 0 to +{self.max_off.value()} cm…")
        self._worker.start()

    def _on_sweep(self, token: int, result) -> None:
        worker = self.sender()
        if worker is None or worker is not self._worker:      # stop() has already let it go
            return
        self._release()
        if token != self._token or self._dirty or result is None:   # the scenario changed while it ran
            return
        if isinstance(result, str):
            self.bar.set_text(result)
            return
        self.sweep    = result
        self.switches = cep_switches(result)
        self.charts.set_sweep(result, self.sc.truth, self.switches)
        self.b_sweep.setEnabled(True)
        self._set_story("sweep", play=self.isVisible())

    def _release(self) -> None:
        """Wait for the finished (or cancelled) worker and let it go."""
        self._worker.wait()
        self._worker.deleteLater()
        self._worker = None
        self.busy.hide()
        if self.sc is not None:
            self._enable_run()

    def stop(self) -> None:
        """Cancel a running sweep: it stops within one offset, and its result is dropped."""
        if self._worker is not None:
            self._worker.cancel.set()
            self._token += 1
            self._release()
            self._shown = None               # the bar said "sweeping…": describe the step again
            self._on_player()

    # ─── Steps ───────────────────────────────────────────────────────────────

    def _offset(self, k: int, t: float) -> float:
        offs = self.sweep.offsets
        return float(offs[k] if k == 0 else offs[k - 1] + (offs[k] - offs[k - 1]) * ease(t))

    def describe(self, k: int) -> str:
        if self.story == "sweep":
            sw  = self.sweep
            now = [text for sk, _, text in self.switches if sk == k]
            out = "" if sw.in_band[k] else (", past the top band: the bounds check drops it" if k == sw.dropped
                                            else ", past the top band: dropped")
            if sw.cep_subsets[k]:
                cep = f"CEP uses {names(sw.cep_subsets[k])}"
            else:                                       # the only change it can have here is losing the fix
                cep, now = f"CEP has no fix ({no_cep_fix(sw, k)})", []
            return f"R{sw.receiver} +{sw.offsets[k]:.1f} cm{out} · {cep}" + (f" · {now[0]}" if now else "")
        kind = self.steps[k]
        if kind == "filter":
            return f"filter: kept {names(self.sc.sane)}"
        if kind == "summary":
            ranked = [a for a in self.ranking() if self._error(a) is not None]
            if not ranked:
                return "summary: no true position to rank by"
            best, worst = ranked[0], ranked[-1]
            return (f"ranking: best {best} ({num(self._error(best))} cm), "
                    f"worst {worst} ({num(self._error(worst))} cm)")
        e = self.fixes.get(kind)
        if e is None or e.pos is None:
            return f"{kind}: no fix ({self._note(kind)})"
        err = self._error(kind)
        return f"{kind}: fix {pt(e.pos)}" + (f" · error {num(err)} cm" if err is not None else "")

    def on_phase(self) -> None:
        k, t = self.player.index, self.player.phase
        if self.story == "sweep":
            self.charts.set_cursor(self._offset(k, t))
        else:
            kind = self.steps[k]
            m    = STATIC_ALGOS.index(kind) if kind in STATIC_ALGOS else (-1 if kind == "filter" else 99)
            self.bars.set_progress(min(m + 1, len(STATIC_ALGOS)), m if 0 <= m < 99 else None, t, kind == "summary")
        self.canvas.update()

    # ─── Canvas ──────────────────────────────────────────────────────────────

    def paint_scene(self, p: QPainter, view: draw.View, k: int, t: float) -> None:
        draw_world(p, view, self.sc.world)
        self._paint_sound(p, view)
        if self.story == "sweep":
            self._paint_sweep(p, view, k, t)
        elif self.steps[k] == "filter":
            draw_filter(p, view, self.sc, t)
            draw_truth(p, view, self.sc.truth)
            draw_legend(p, view, filter_keys(self.sc) + self._sound_keys(), self.HINT,
                        receiver_points(view, self.sc))
        else:
            self._paint_fixes(p, view, self.steps[k], t)

    def _rings(self) -> dict:
        pal = theme.current
        return {i: pal.status.get(s, pal.text_dim) for i, s in enumerate(self.sc.status)}

    def _paint_sound(self, p, view) -> None:
        """The path the detected sound took to each receiver; dotted where the line of sight is blocked."""
        pal, sc = theme.current, self.sc
        for i, r in enumerate(sc.readings):
            c = pal.status.get(r.status, pal.text_dim)
            if r.detected is not None and r.detected.source == "TX":
                path = r.detected.path
                draw.polyline(p, view, path.points, c, 1.0, 0.5, None if path.kind == "direct" else draw.DASH)
                for b in path.points[1:-1]:
                    draw.marker(p, view.px(*b), "circle", c, 4, 1.0, 0.7)
            if not r.los and sc.truth is not None:
                draw.polyline(p, view, [sc.truth, sc.coords[i]], pal.status["lost"], 1.0, 0.45, draw.DOT)

    def _sound_keys(self) -> list:
        pal, rd = theme.current, self.sc.readings
        keys = []
        if any(r.detected is not None and r.detected.path.kind == "direct" for r in rd):
            keys.append(Key("line", pal.status["ok"], "sound heard, direct"))
        if any(r.detected is not None and r.detected.path.kind != "direct" for r in rd):
            keys.append(Key("dash", pal.status["echo"], "bounced or bent"))
        if any(not r.los for r in rd):
            keys.append(Key("dot", pal.status["lost"], "line of sight blocked"))
        return keys

    def _fix_marks(self, p, view, shown, kind: str, g: float) -> None:
        """Every fix shown so far, with its error line to the truth; the current one appearing."""
        pal, truth = theme.current, self.sc.truth
        for a in shown:
            e = self.fixes.get(a)
            if e is None or e.pos is None:
                continue
            c, now, pos = pal.algo[a], a == kind, np.asarray(e.pos, dtype=float)
            if truth is not None:
                end = truth + (pos - truth) * (g if now else 1.0)
                draw.polyline(p, view, [truth, end], c, 1.2 if now else 1.0, 0.9 if now else 0.35, draw.DASH)
            if now:
                draw.marker(p, view.px(*pos), theme.ALGO_SHAPES[a], c, 12 * (1.8 - 0.8 * g), 1.6, g, fill=0.25)
            else:
                draw.marker(p, view.px(*pos), theme.ALGO_SHAPES[a], c, 9, 1.2, 1.0 if kind == "summary" else 0.55)

    def _paint_fixes(self, p, view, kind: str, t: float) -> None:
        pal, sc = theme.current, self.sc
        shown = STATIC_ALGOS if kind == "summary" else STATIC_ALGOS[:STATIC_ALGOS.index(kind) + 1]
        cur   = self.fixes.get("CEP" if kind == "summary" else kind)    # the ranking keeps CEP's subset lit
        used  = set(cur.used) if cur is not None and cur.used else None
        g     = ease(t)
        draw_ranges(p, view, sc, range(len(sc.coords)))
        draw_receivers(p, view, sc, lit=used, rings=self._rings(), thick=used or ())
        self._fix_marks(p, view, shown, kind, g)
        draw_truth(p, view, sc.truth)
        fixes = [np.asarray(e.pos, dtype=float) for a in shown if (e := self.fixes.get(a)) is not None
                 and e.pos is not None]
        for a in shown:
            e = self.fixes.get(a)
            if e is not None and e.pos is not None:
                draw_offscreen(p, view, e.pos, pal.algo[a], a)
        keys  = [Key(theme.ALGO_SHAPES[a], pal.algo[a], a, "—" if (e := self._error(a)) is None else f"{num(e)} cm",
                     strong=a == kind, spare=True) for a in shown]
        if sc.truth is not None:
            keys.append(Key("crosshair", pal.truth, "true position"))
        avoid = receiver_points(view, sc) + [view.px(*f) for f in fixes]
        box   = draw_legend(p, view, keys, self.HINT, avoid)
        if fixes:
            centre = sc.truth if sc.truth is not None else np.mean(fixes, axis=0)
            near   = [float(np.hypot(*(f - centre))) for f in fixes]
            near   = [d for d in near if d <= 50.0] or [1.0]
            def paint(q, v):
                self._fix_marks(q, v, shown, kind, g)
                draw_truth(q, v, sc.truth)

            magnifier(p, view, centre, 1.25 * max(near) + 0.5, paint, avoid, [box])

    def _sweep_marks(self, p, view, k: int, t: float) -> list:
        """Every algorithm's path (faint), the part travelled (bold) and where it is now: [(algorithm, head)]."""
        pal, sw = theme.current, self.sweep
        heads = []
        for a in STATIC_ALGOS:
            path = sw.paths.get(a)
            if path is None:
                continue
            c    = pal.algo[a]
            head = path[k] if k == 0 else path[k - 1] + (path[k] - path[k - 1]) * ease(t)
            draw.polyline(p, view, path, c, 1.0, 0.22)
            draw.polyline(p, view, np.vstack([path[:k], head[None]]), c, 1.4, 0.9)
            if np.all(np.isfinite(head)):
                draw.marker(p, view.px(*head), theme.ALGO_SHAPES[a], c, 10, 1.5, fill=0.25)
            heads.append((a, head))
        for sk, _, _ in (s for s in self.switches if s[0] <= k):
            at = sw.paths["CEP"][sk]
            if np.all(np.isfinite(at)):
                draw.marker(p, view.px(*at), "cross", pal.algo["CEP"], 7, 1.2)
        return heads

    def _paint_sweep(self, p, view, k: int, t: float) -> None:
        pal, sc, sw = theme.current, self.sc, self.sweep
        i   = sw.receiver
        off = self._offset(k, t)
        out = not sw.in_band[k]                 # past the top band: the bounds check drops the reading
        draw_ranges(p, view, sc, [j for j in sc.sane if j != i], dropped_too=False)
        d = float(sc.distances[i])
        draw.circle(p, view, sc.coords[i], d, pal.text_dim, 1.0, 0.6, draw.DOT)
        draw.circle(p, view, sc.coords[i], d + off, pal.status["lost"] if out else pal.status["echo"], 1.5, 0.95,
                    draw.DOT if out else None)
        draw_receivers(p, view, sc, lit=set(sc.sane) - ({i} if out else set()), rings=self._rings(),
                       thick=set() if out else {i})
        heads = self._sweep_marks(p, view, k, t)
        draw_truth(p, view, sc.truth)
        for n, (sk, _, text) in enumerate(s for s in self.switches if s[0] <= k):
            at = sw.paths["CEP"][sk]
            if np.all(np.isfinite(at)):
                q = view.px(*at)
                side_label(p, view, QPointF(q.x(), q.y() + 26 + 13 * (n % 3)), text, pal.algo["CEP"])
        for a, head in heads:
            draw_offscreen(p, view, head, pal.algo[a], a)
        def error(h) -> str:
            if sc.truth is None or not np.all(np.isfinite(h)):
                return "—"
            return f"{num(float(np.hypot(*(h - sc.truth))))} cm"

        keys = [Key(theme.ALGO_SHAPES[a], pal.algo[a], a, error(h), spare=True) for a, h in heads]
        keys += [Key("dot", pal.status["lost"], f"R{i} range +{off:.0f} cm: too far, dropped", strong=True) if out
                 else Key("line", pal.status["echo"], f"R{i} range +{off:.0f} cm", strong=True),
                 Key("dot", pal.text_dim, f"R{i} range as read"),
                 Key("cross", pal.algo["CEP"], "CEP changes subset")]
        if sc.truth is not None:
            keys.append(Key("crosshair", pal.truth, "true position"))
        avoid = receiver_points(view, sc) + [view.px(*h) for _, h in heads if np.all(np.isfinite(h))]
        box   = draw_legend(p, view, keys, self.HINT, avoid)
        ends  = np.vstack([path[np.all(np.isfinite(path), axis=1)] for path in sw.paths.values()])
        centre = sc.truth if sc.truth is not None else sw.paths["CEP"][0]
        def paint(q, v):
            self._sweep_marks(q, v, k, t)
            draw_truth(q, v, sc.truth)

        if len(ends):
            magnifier(p, view, centre, 1.15 * float(np.hypot(*(ends - centre).T).max()) + 0.5, paint, avoid, [box])

    # ─── Explanation ─────────────────────────────────────────────────────────

    def html(self, k: int) -> str:
        if self.story == "sweep":
            return self._sweep_html(k)
        kind = self.steps[k]
        if kind == "filter":
            return filter_html(self.sc)
        if kind == "summary":
            return self._summary_html()
        return self._algo_html(kind)

    def _residual_rows(self, pos, receivers) -> list:
        pal, sc = theme.current, self.sc
        rows = []
        for i in receivers:
            rng = float(np.hypot(*(np.asarray(pos) - sc.coords[i])))
            st  = sc.status[i]
            err = sc.error(i)
            rows.append([f"R{i}", span(st, pal.status.get(st, pal.text_dim)),
                         "—" if err is None else num(err, True), num(sc.distances[i]), num(rng),
                         num(rng - sc.distances[i], True)])
        return rows

    def _algo_html(self, a: str) -> str:
        pal, info = theme.current, S.ALGORITHMS[a]
        e = self.fixes.get(a)
        head = (f"<h3>{span(a, pal.algo[a])} · {info.label}</h3>"
                f"<p class='dim'>{info.source}: {info.summary}</p>")
        if e is None or e.pos is None:
            return head + f"<p class='bad'>No fix: {self._note(a)}.</p>"
        err = self._error(a)
        return (head
                + f"<p>Fix <b>{pt(e.pos)}</b>"
                + (f", error <b>{num(err)} cm</b>" if err is not None else "")
                + f"<br>uses {names(e.used)}" + (f" · {e.note}" if e.note else "") + "</p>"
                + "<p class='dim'>Residuals at this fix, r = |p − R| − d (the reading's own error is its "
                  "distance minus the true range):</p>"
                + html_table(["", "heard", "reading err", "d", "|p − R|", "r"],
                             self._residual_rows(e.pos, self.inputs(a)))
                + f"<p class='muted'>{ALGO_HOW[a]}{self._mode_note(a)}</p>")

    def _mode_note(self, a: str) -> str:
        """With Bounds only, which readings a simulator algorithm sees that the library dropped."""
        extra = [i for i in self.inputs(a) if i not in self.sc.sane]
        if not extra:
            return ""
        return (f" Filter mode Bounds only: it also sees {names(extra)}, which the library's jump check "
                "dropped, and leaves them to its own outlier handling.")

    def _summary_html(self) -> str:
        pal, sc = theme.current, self.sc
        order = self.ranking()
        rows  = []
        for n, a in enumerate(order):
            e = self.fixes.get(a)
            rows.append([f"{n + 1}", span(a, pal.algo[a]), num(self._error(a)),
                         names(e.used) if e is not None and e.pos is not None else "—",
                         e.note if e is not None else self._note(a)])
        dropped = [i for i in range(len(sc.coords)) if i not in sc.sane]
        bad     = [i for i in sc.sane if (err := sc.error(i)) is not None and abs(err) > 3.0]
        jumped  = [i for i in self.inputs("RANSAC") if i not in sc.sane]
        if not sc.readings:
            lesson = "These are replayed ticks: how far each reading was off is not known."
        elif bad:
            lesson = (f"{names(bad)} {'reads' if len(bad) == 1 else 'read'} more than 3 cm wrong but "
                      "passed the filter. The algorithms that use every kept receiver (OLS, LM, BIAS) share "
                      "that error out; CEP and RANSAC can leave it out; ROBUST weighs it down.")
        else:
            lesson = ("Every kept reading is within 3 cm of the truth, so the algorithms agree"
                      + (f": the filter has already dropped {names(dropped)}." if dropped else
                         "; set a receiver to Blocked or +offset to see them part."))
        if jumped:
            lesson += (f" Filter mode Bounds only: CEP4, ROBUST, BIAS and RANSAC also see {names(jumped)}, which "
                       "the library's jump check dropped; OLS, LM and CEP use only the receivers it kept.")
        return ("<h3>Ranking by error</h3>"
                + html_table(["#", "", "error cm", "uses", "note"], rows, "rlrll")
                + f"<p>{lesson}</p>")

    def _sweep_html(self, k: int) -> str:
        pal, sc, sw = theme.current, self.sc, self.sweep
        i, off = sw.receiver, float(sw.offsets[k])
        rows = []
        for a in STATIC_ALGOS:
            path = sw.paths.get(a)
            if path is None:
                continue
            pos = path[k]
            ok  = np.all(np.isfinite(pos))
            err = num(float(np.hypot(*(pos - sc.truth)))) if ok and sc.truth is not None else "—"
            rows.append([span(a, pal.algo[a]), pt(pos) if ok else "no fix", err,
                         names(sw.cep_subsets[k]) if a == "CEP" else ""])
        sw_rows = "".join(f"<br>{'<b>' if sk <= k else ''}{text}{'</b>' if sk <= k else ''}"
                          for sk, _, text in self.switches) or "<br>none: CEP keeps the same subset"
        top, cut = sw.top_band, sw.dropped
        if sw.in_band[k]:
            band = (f"0 &lt; d<sub>{i}</sub> ≤ {num(top)}: within the top band, so the bounds check keeps it"
                    + (f" (until +{sw.offsets[cut]:.0f} cm)." if cut is not None else "."))
        else:
            band = span(f"d<sub>{i}</sub> &gt; {num(top)}: past the top band, so the library's bounds check "
                        f"drops R{i}", pal.status["lost"]) + f" (from +{sw.offsets[cut]:.0f} cm)."
        if sw.kept and len(sw.kept[k]) < 3:
            band += (f" That leaves the library only {names(sw.kept[k])}: with fewer than 3 receivers its OLS, "
                     "LM and CEP cannot solve.")
        return (f"<h3>R{i} reads +{off:.1f} cm long</h3>"
                f"<p><span class='eq'>d<sub>{i}</sub> = {num(sc.distances[i])} + {off:.1f} = "
                f"{num(sc.distances[i] + off)} cm</span>"
                + (f" <span class='muted'>(true range {num(sc.readings[i].true_cm)})</span>" if sc.readings else "")
                + f"<br>{band}</p>" + html_table(["", "fix", "error cm", "CEP subset"], rows)
                + f"<p>Where the library's CEP changes subset:{sw_rows}</p>"
                "<p class='muted'>At every offset the library's bounds check is applied again. Its jump check is "
                "not: the readings are taken to have been like this for a while, as on every frame after the "
                "first once a receiver is blocked (the jump check only catches the frame where the reading "
                "changes).</p>")


# ─── Tab 3: LM optimiser ─────────────────────────────────────────────────────

RECEIVERS_TIP = "Receivers LM fits: every one the filter kept, or any three of them"
LM_HEADERS    = ["#", "kind", "x", "y", "RMS cm", "|δ| cm"]


def rms_at(coords, d, pos) -> float:
    """RMS range residual (cm) of a fix: what LM minimises (as ½ Σ r²)."""
    return float(np.sqrt(np.mean((np.hypot(*(np.asarray(pos) - coords).T) - d) ** 2)))


class LmTab(LabTab):
    """The library's Levenberg-Marquardt (``_nlls``) from one initial guess, or
    from many at once, on the cost surface it walks down."""

    STEP_SECONDS = 1.0
    MANY_SECONDS = 0.8
    HINT         = "click: start LM there"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initial    = None      # where the user clicked (None: the library's OLS seed)
        self.idx        = []
        self.seed       = None
        self.seed_fix   = None      # where LM ends from the OLS seed (the library's own start)
        self.trace      = None
        self.points     = []        # the steps: trace.points, without Jacobian probes when hidden
        self.before     = []        # the iterate each point was evaluated from
        self.many       = []        # [(start, LMTrace)]
        self.minima     = []        # [(position, rms, starts)], lowest first
        self.basin      = []        # minimum each start ends in
        self.global_min = None
        self.contours   = []        # [(level, QPainterPath in world cm, label point or None)]
        self.steps      = []
        self.line_triple = None

        self.receivers = QComboBox()
        # Short, so the controls row never squeezes the Scenario dock; the popup shows the names in full
        self.receivers.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.receivers.setMinimumContentsLength(12)
        self.receivers.setToolTip(RECEIVERS_TIP)
        self.receivers.currentIndexChanged.connect(self._on_receivers)
        self.b_line = QPushButton("3 in a line")
        self.b_line.setCheckable(True)
        self.b_line.setToolTip("Fit only three receivers that stand in a line: their distances cannot tell the "
                               "transmitter from its mirror image, so the surface gets a second minimum")
        self.b_line.toggled.connect(self._on_line)
        self.b_many = QPushButton("Many starts")
        self.b_many.setCheckable(True)
        self.b_many.setToolTip(f"Run LM from a {STARTS_SIDE} × {STARTS_SIDE} grid of initial guesses at once; "
                               "each is coloured by the minimum it ends in (its basin of attraction)")
        self.b_many.toggled.connect(lambda _on: self._load_steps(play=True))
        self.c_probes = QCheckBox("Jacobian probes")
        self.c_probes.setChecked(True)
        self.c_probes.setToolTip("Include the evaluations the optimiser makes to estimate the Jacobian by "
                                 "finite differences")
        self.c_probes.toggled.connect(lambda _on: self._load_steps())
        self.b_seed = QPushButton("Start at OLS seed")
        self.b_seed.setToolTip("Start LM from the library's OLS seed again, as the library does")
        self.b_seed.clicked.connect(self.reset_start)
        self.canvas.clicked.connect(self.set_start)
        self.table = make_table(LM_HEADERS)
        self.table.setToolTip("Every point LM evaluated (or, with many starts, where each start ended); "
                              "click a row to jump to it")
        self.table.cellClicked.connect(lambda r, _c: None if self.b_many.isChecked() else self.player.seek(r))

        pick = QWidget()                 # the label stays with its combo when the row wraps
        pl   = QHBoxLayout(pick)
        pl.setContentsMargins(0, 0, 0, 0)
        rl = QLabel("Receivers")
        rl.setProperty("role", "dim")
        pl.addWidget(rl)
        pl.addWidget(self.receivers)
        # Wraps where the window is narrow, rather than squeezing the Scenario dock
        self.controls = FlowRow([pick, self.b_line, self.b_many, self.c_probes, self.b_seed])
        right = splitter(Qt.Orientation.Vertical, [self.text, self.table], [480, 230])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.controls)
        lay.addWidget(splitter(Qt.Orientation.Horizontal, [self.canvas, right], [640, 430]), 1)
        lay.addWidget(self.bar)

    # ─── Computing ───────────────────────────────────────────────────────────

    def compute(self) -> None:
        self._fill_receivers()
        self._solve()

    def _fill_receivers(self) -> None:
        sc, keep = self.sc, self.receivers.currentData()
        self.receivers.blockSignals(True)
        self.receivers.clear()
        self.receivers.addItem(f"all kept ({names(sc.sane)})", None)
        lines = []
        for tri in combinations(sc.sane, 3):
            line = collinear(sc.coords[list(tri)])
            self.receivers.addItem(names(tri) + ("  · in a line" if line else ""), tri)
            if line:
                lines.append(tri)
        at = self.receivers.findData(keep) if keep is not None else 0
        self.receivers.setCurrentIndex(max(at, 0))
        self.receivers.setToolTip(f"{RECEIVERS_TIP}. Now: {self.receivers.currentText()}")
        ref = sc.truth if sc.truth is not None else sc.coords[sc.sane].mean(axis=0) if sc.sane else np.zeros(2)
        self.line_triple = min(lines, key=lambda t: self._line_distance(t, ref)) if lines else None
        self.receivers.blockSignals(False)
        self._sync_line_button()

    def _line_distance(self, tri, point) -> float:
        a, b = self.sc.coords[tri[0]], self.sc.coords[tri[-1]]
        u = (b - a) / max(float(np.hypot(*(b - a))), 1e-9)
        v = np.asarray(point) - a
        return abs(float(u[0] * v[1] - u[1] * v[0]))

    def _sync_line_button(self) -> None:
        self.b_line.blockSignals(True)
        self.b_line.setEnabled(self.line_triple is not None)
        self.b_line.setChecked(self.line_triple is not None and self.receivers.currentData() == self.line_triple)
        self.b_line.blockSignals(False)

    def _on_line(self, on: bool) -> None:
        self.receivers.setCurrentIndex(self.receivers.findData(self.line_triple) if on else 0)

    def _on_receivers(self, _i: int) -> None:
        self.receivers.setToolTip(f"{RECEIVERS_TIP}. Now: {self.receivers.currentText()}")
        self._sync_line_button()
        if self.sc is not None and not self._dirty:
            self._solve()

    def _solve(self) -> None:
        """Trace one start and the grid of starts, and draw the cost surface."""
        sc, lib = self.sc, self.lib
        self.idx = list(self.receivers.currentData() or sc.sane)
        if len(self.idx) < 3:
            self.trace, self.many, self.minima, self.contours, self.global_min = None, [], [], [], None
            self._load_steps()
            return
        d, idx = sc.distances, np.array(self.idx)
        c      = sc.coords[idx]
        self.seed     = lib._ols(d, idx)
        self.seed_fix = lib._nlls(d, self.seed, idx)[0]
        x0, y0, x1, y1 = sc.bounds(0.0)
        mx, my = 0.08 * (x1 - x0), 0.08 * (y1 - y0)
        starts = [np.array([x, y]) for y in np.linspace(y1 - my, y0 + my, STARTS_SIDE)
                  for x in np.linspace(x0 + mx, x1 - mx, STARTS_SIDE)]
        self.many  = [(s, lab.trace_lm(lib, d, self.idx, s)) for s in starts]
        self.trace = lab.trace_lm(lib, d, self.idx, self.initial)
        self.minima, self.basin = [], []
        for _, t in self.many:
            for m, (pos, rms, n) in enumerate(self.minima):
                if np.hypot(*(pos - t.final)) < SAME_MIN_CM:
                    self.minima[m] = (pos, rms, n + 1)
                    self.basin.append(m)
                    break
            else:
                self.minima.append((t.final, rms_at(c, d[idx], t.final), 1))
                self.basin.append(len(self.minima) - 1)
        order       = sorted(range(len(self.minima)), key=lambda m: (round(self.minima[m][1], 3), -self.minima[m][2]))
        rank        = {m: r for r, m in enumerate(order)}
        self.minima = [self.minima[m] for m in order]
        self.basin  = [rank[m] for m in self.basin]
        pts = np.vstack([sc.coords, [m[0] for m in self.minima], [self.trace.initial, self.trace.final]])
        b0, b1 = np.minimum(pts.min(axis=0), (x0, y0)) - 25, np.maximum(pts.max(axis=0), (x1, y1)) + 25
        bounds = (float(b0[0]), float(b0[1]), float(b1[0]), float(b1[1]))
        self.canvas.set_bounds(bounds)
        self._surface(bounds)
        self._load_steps()

    def _surface(self, bounds) -> None:
        """Contour lines of the RMS residual at log-spaced levels, and the global minimum."""
        sc, d = self.sc, self.sc.distances
        xs, ys, Z = lab.cost_grid(sc.coords, d, self.idx, bounds, GRID_N)
        j, i = np.unravel_index(int(np.argmin(Z)), Z.shape)
        pos, _ = self.lib._nlls(d, np.array([xs[i], ys[j]]), np.array(self.idx))
        best   = (pos, rms_at(sc.coords[self.idx], d[self.idx], pos))
        self.global_min = min([best] + [(m[0], m[1]) for m in self.minima], key=lambda m: m[1])
        top = float(np.percentile(Z, 97))
        lo  = max(float(Z.min()) * 1.3, top / 150.0, 0.3)
        g   = np.asarray(self.global_min[0])
        self.contours = []
        for n, level in enumerate(np.geomspace(lo, top, LEVELS) if top > lo else [top]):
            segs = lab.contour(xs, ys, Z, level)
            path = QPainterPath()
            for (ax, ay), (bx, by) in segs:
                path.moveTo(ax, ay)
                path.lineTo(bx, by)
            at = None
            if n % 3 == 1 and len(segs):
                mids = segs.mean(axis=1) - g
                ang  = np.abs((np.arctan2(mids[:, 1], mids[:, 0]) - math.radians(55) + np.pi) % (2 * np.pi) - np.pi)
                at   = segs.mean(axis=1)[int(np.argmin(ang))]
            self.contours.append((float(level), path, at))

    def _load_steps(self, play: bool = False) -> None:
        if self.trace is None:
            self.steps = ["none"]
            fill_table(self.table, [])
            self.restart(1)
            return
        if self.b_many.isChecked():
            n = max(len(t.path) for _, t in self.many)
            self.steps = list(range(n))
            fill_table(self.table, [[f"{s + 1}", f"{num(st[0])}, {num(st[1])}", num(t.final[0]), num(t.final[1]),
                                     num(self.minima[self.basin[s]][1]), "→ " + "ABCDEFGHIJ"[self.basin[s] % 10]]
                                    for s, (st, t) in enumerate(self.many)],
                       headers=["#", "start", "fix x", "fix y", "RMS cm", "minimum"])
            self.restart(n, self.MANY_SECONDS, play)
            return
        self.points = [q for q in self.trace.points if self.c_probes.isChecked() or q[2] != "jacobian"]
        self.before, cur, rows = [], None, []
        for n, (x, rms, kind) in enumerate(self.points):
            cur = (x, rms) if cur is None else cur
            self.before.append(cur)
            rows.append([f"{n + 1}", {"jacobian": "probe", "rejected": "reject"}.get(kind, kind),
                         num(x[0]), num(x[1]), num(rms),
                         "—" if kind == "start" else num(float(np.hypot(*(x - cur[0]))))])
            if kind in ("start", "step"):
                cur = (x, rms)
        fill_table(self.table, rows, headers=LM_HEADERS)
        self.steps = list(range(len(self.points)))
        self.restart(len(self.steps), self.STEP_SECONDS, play)

    def set_start(self, x: float, y: float) -> None:
        """Start LM at (x, y) (a canvas click) and play it."""
        if self.sc is None or len(self.idx) < 3:
            return
        self.initial = np.array([x, y], dtype=float)
        self._retrace()

    def reset_start(self) -> None:
        self.initial = None
        if self.sc is not None and len(self.idx) >= 3:
            self._retrace()

    def _retrace(self) -> None:
        self.trace = lab.trace_lm(self.lib, self.sc.distances, self.idx, self.initial)
        self.b_many.blockSignals(True)
        self.b_many.setChecked(False)
        self.b_many.blockSignals(False)
        self._load_steps(play=True)

    # ─── Steps ───────────────────────────────────────────────────────────────

    def describe(self, k: int) -> str:
        if self.steps[0] == "none":
            return "LM needs 3 receivers"
        if self.b_many.isChecked():
            if k < len(self.steps) - 1:
                return f"iteration {k}: {self.moving(k)} of {len(self.many)} starts still moving"
            found = len(self.minima)
            return f"{found} minim{'um' if found == 1 else 'a'} found by {len(self.many)} starts"
        x, rms, kind = self.points[k]
        what = {"start": "initial guess", "step": "accepted step", "rejected": "rejected trial",
                "jacobian": "Jacobian probe"}[kind]
        return f"evaluation {k + 1} of {len(self.points)}: {what} at {pt(x)}, RMS {num(rms)} cm"

    def on_step(self, k: int) -> None:
        if self.b_many.isChecked():
            show_rows(self.table, self.table.rowCount(), None)
        else:
            show_rows(self.table, k, k)

    # ─── Canvas ──────────────────────────────────────────────────────────────

    def paint_scene(self, p: QPainter, view: draw.View, k: int, t: float) -> None:
        pal, sc = theme.current, self.sc
        draw_world(p, view, sc.world)
        if self.steps[0] == "none":
            draw_receivers(p, view, sc, lit=set(self.idx))
            draw_truth(p, view, sc.truth)
            banner(p, view, "LM needs 3 receivers", pal.status["lost"])
            return
        g = view.px(*self.global_min[0])
        self._paint_contours(p, view, receiver_points(view, sc) + [g] + [view.px(*m[0]) for m in self.minima])
        p.save()
        p.setOpacity(0.5)
        draw_ranges(p, view, sc, self.idx, dropped_too=False)
        p.restore()
        draw_receivers(p, view, sc, lit=set(self.idx))
        draw.marker(p, g, "ring", pal.text, 14, 1.2)
        many  = self.b_many.isChecked()
        marks = self._paint_many(p, view, k, t) if many else self._paint_single(p, view, k, t)
        draw_truth(p, view, sc.truth)
        avoid = receiver_points(view, sc) + [view.px(*m) for m in marks] + [g]
        box   = draw_legend(p, view, self._many_keys() if many else self._single_keys(k), self.HINT, avoid)
        if not many:
            self._paint_cost(p, view, k, avoid, [box])

    def _paint_contours(self, p, view, avoid) -> None:
        """The cost surface: contour lines, brighter toward low cost, and a few
        level labels, left out where they would sit on one of the ``avoid``
        points (pixels) or on another label."""
        c = theme.current.algo["LM"]
        n = max(len(self.contours) - 1, 1)
        p.save()
        p.setTransform(QTransform(view.scale, 0, 0, -view.scale, view.width / 2 - view.cx * view.scale,
                                  view.height / 2 + view.cy * view.scale))
        for m, (_, path, _) in enumerate(self.contours):
            draw.stroke(p, path, c, 1.0, 0.8 - 0.62 * m / n, glow=theme.current.glow and m < 3)
        p.restore()
        fm, taken = QFontMetricsF(theme.mono_font(7)), []
        for m, (level, _, at) in enumerate(self.contours):
            if at is None:
                continue
            text = f"{level:.0f} cm" if level >= 10 else f"{level:.1f} cm"
            q    = view.px(*at)
            w    = fm.horizontalAdvance(text) + 10
            rect = QRectF(q.x() - w / 2, q.y() - 8, w, 16)
            near = rect.adjusted(-12, -12, 12, 12)
            if any(near.contains(a) for a in avoid) or any(rect.intersects(r) for r in taken):
                continue
            taken.append(rect)
            draw.label(p, q, text, c, 7, "center", alpha=0.9 - 0.5 * m / n)

    def _single_keys(self, k: int) -> list:
        pal = theme.current
        c   = pal.algo["LM"]
        keys = [Key("line", c, "contours of RMS residual"), Key("plus", pal.algo["OLS"], "OLS seed")]
        if self.initial is not None:
            keys.append(Key("faint:circle", c, "your start"))
        keys += [Key("circle", c, "iterate", f"RMS {num(self.iterate_rms(k))} cm", strong=True),
                 Key("arrow", c, "accepted step"), Key("cross", pal.status["lost"], "rejected trial")]
        if self.c_probes.isChecked():
            keys.append(Key("tick", pal.text_dim, "Jacobian probe"))
        keys.append(Key("ring", pal.text, "global minimum", f"RMS {num(self.global_min[1])} cm"))
        if self.sc.truth is not None:
            keys.append(Key("crosshair", pal.truth, "true position"))
        return keys

    def _many_keys(self) -> list:
        pal  = theme.current
        keys = [Key("ring", pal.algo[MINIMA_KEYS[m % len(MINIMA_KEYS)]], f"minimum {'ABCDEFGHIJ'[m % 10]}",
                    f"{count} start{'s' * (count != 1)}", strong=True) for m, (_, _, count) in enumerate(self.minima)]
        keys += [Key("faint:circle", pal.text_dim, "start, coloured by its minimum"),
                 Key("line", pal.algo["LM"], "contours of RMS residual")]
        if self.sc.truth is not None:
            keys.append(Key("crosshair", pal.truth, "true position"))
        return keys

    def moving(self, k: int) -> int:
        """How many of the many starts take another step after iteration ``k``."""
        return sum(1 for _, t in self.many if len(t.path) > k + 1)

    def iterate_rms(self, k: int) -> float:
        """RMS residual at the iterate once evaluation ``k`` is done."""
        x, rms, kind = self.points[k]
        return rms if kind in ("start", "step") else self.before[k][1]

    def _paint_single(self, p, view, k: int, t: float) -> list:
        pal = theme.current
        c   = pal.algo["LM"]
        draw.marker(p, view.px(*self.seed), "plus", pal.algo["OLS"], 11, 1.4)
        head = None
        for n in range(k + 1):
            x, _, kind = self.points[n]
            cur  = self.before[n][0]
            g    = ease(t) if n == k else 1.0
            a, b = view.px(*cur), view.px(*(cur + (x - cur) * g))
            if kind == "start":
                draw.marker(p, view.px(*x), "circle", c, 8, 1.2, 0.6)
                head = x
            elif kind == "step":
                draw.arrow(p, a, b, c, 1.7 if n == k else 1.2, 7.0, 1.0 if n == k else 0.75)
                head = cur + (x - cur) * g
            elif kind == "rejected":
                p.save()
                p.setPen(draw.pen(pal.status["lost"], 1.0, 0.7, draw.DOT))
                p.drawLine(a, b)
                p.restore()
                draw.marker(p, b, "cross", pal.status["lost"], 9, 1.4, g)
                if n == k:
                    draw_offscreen(p, view, x, pal.status["lost"], "rejected trial")
            else:
                u = x - cur
                L = float(np.hypot(*u)) or 1.0
                p.setPen(draw.pen(pal.text_dim, 2.2 if n == k else 1.0, 0.9 if n == k else 0.5))
                p.drawLine(a, QPointF(a.x() + u[0] / L * 9, a.y() - u[1] / L * 9))
        if head is None:
            return []
        draw.marker(p, view.px(*head), "circle", c, 12, 1.6, fill=0.35)
        draw_offscreen(p, view, head, c, "iterate")
        return [head, self.seed]

    def _paint_many(self, p, view, k: int, t: float) -> list:
        pal  = theme.current
        last = k == len(self.steps) - 1
        for s, (start, tr) in enumerate(self.many):
            c = pal.algo[MINIMA_KEYS[self.basin[s] % len(MINIMA_KEYS)]]
            P = np.asarray(tr.path)
            if k == 0 or k >= len(P):
                head = P[min(k, len(P) - 1)]
            else:
                head = P[k - 1] + (P[k] - P[k - 1]) * ease(t)
            moving = k < len(P)
            draw.polyline(p, view, np.vstack([P[:min(k, len(P))], head[None]]), c, 1.0, 0.6 if moving else 0.3)
            draw.marker(p, view.px(*start), "circle", c, 5, 1.0, 0.7)
            draw.marker(p, view.px(*head), "circle", c, 7, 1.3, fill=0.5)
        for m, (pos, _, _) in enumerate(self.minima):
            c = pal.algo[MINIMA_KEYS[m % len(MINIMA_KEYS)]]
            q = view.px(*pos)
            draw.marker(p, q, "ring", c, 18, 1.5, 1.0 if last else 0.5)
            side_label(p, view, q, "ABCDEFGHIJ"[m % 10], c, 9)
            draw_offscreen(p, view, pos, c, f"minimum {'ABCDEFGHIJ'[m % 10]}")
        return [m[0] for m in self.minima]

    def _paint_cost(self, p, view, k: int, avoid, busy) -> None:
        """Inset: the RMS residual at every evaluation so far (log scale)."""
        pal = theme.current
        if view.width < 480 or view.height < 320 or len(self.points) < 2:
            return
        rect = free_spot(view, 216, 116, avoid, busy, "br")
        inset_frame(p, rect)
        plot = rect.adjusted(40, 22, -10, -14)
        rms  = np.array([r for _, r, _ in self.points], dtype=float)
        hi   = float(rms.max())
        lo   = max(float(rms[rms > 0].min()) if (rms > 0).any() else hi, hi * 1e-4, 1e-3)
        lo   = min(lo, hi / 10)

        def X(n):
            return plot.left() + n / (len(self.points) - 1) * plot.width()

        def Y(v):
            return plot.bottom() - (math.log10(max(v, lo)) - math.log10(lo)) / math.log10(hi / lo) * plot.height()

        for v in (lo, hi):
            p.setPen(draw.pen(pal.grid_minor, 1.0))
            p.drawLine(QPointF(plot.left(), Y(v)), QPointF(plot.right(), Y(v)))
            draw.label(p, QPointF(plot.left() - 4, Y(v)), num(v), pal.text_muted, 6.5, "right", halo=False)
        c = pal.algo["LM"]
        chain = [QPointF(X(n), Y(r)) for n, (_, r, kind) in enumerate(self.points[:k + 1]) if kind in ("start", "step")]
        draw.stroke(p, draw.polyline_path(chain), c, 1.3)
        for n, (_, r, kind) in enumerate(self.points[:k + 1]):
            q = QPointF(X(n), Y(r))
            if kind == "rejected":
                draw.marker(p, q, "cross", pal.status["lost"], 6, 1.1)
            elif kind == "jacobian":
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(color(pal.text_dim, 0.7))
                p.drawEllipse(q, 1.3, 1.3)
            else:
                draw.marker(p, q, "circle", c, 4.5, 1.1, fill=0.6)
        p.setPen(draw.pen(pal.accent, 1.0, 0.6))
        p.drawLine(QPointF(X(k), plot.top()), QPointF(X(k), plot.bottom()))
        inset_title(p, rect, "RMS residual (cm) per evaluation")

    # ─── Explanation ─────────────────────────────────────────────────────────

    def html(self, k: int) -> str:
        if self.steps[0] == "none":
            return ("<h3>LM optimiser</h3><p class='bad'>Fewer than 3 receivers: the library does not run "
                    "LM (it holds its last fix). Switch receivers back on in the Scenario panel.</p>")
        return self._many_html(k) if self.b_many.isChecked() else self._single_html(k)

    def _footer(self, run: bool = True) -> str:
        """The library call behind it all, and (``run``) how this run ended."""
        tr  = self.trace
        out = ("<p class='muted'>The library's <span class='dim'>_nlls</span> calls "
               "<span class='eq'>scipy.optimize.least_squares(residuals, initial, method=\"lm\")</span>: "
               "MINPACK's Levenberg-Marquardt, with J estimated by forward differences of r.")
        if run:
            out += (f" This run: {len(tr.points)} calls of the residual function (scipy reports nfev = "
                    f"{tr.nfev}), ending with “{html.escape(tr.message.replace('`', '').rstrip('.'))}”, "
                    f"fix {pt(tr.final)}.")
        return out + "</p>"

    def _single_html(self, k: int) -> str:
        pal, sc = theme.current, self.sc
        x, rms, kind = self.points[k]
        p0, rms0 = self.before[k]
        c, d = sc.coords[self.idx], sc.distances[self.idx]
        rng  = np.hypot(*(p0 - c).T)
        r    = rng - d
        J    = (p0 - c) / np.maximum(rng, 1e-9)[:, None]
        A, g = J.T @ J, J.T @ r
        m    = len(self.idx)

        def cost(rms_cm: float) -> float:           # F = ½ Σ r², from the RMS
            return 0.5 * m * rms_cm * rms_cm

        rows = [[f"R{i}", num(d[n]), num(rng[n]), span(num(r[n], True), pal.status["echo"]),
                 num(J[n, 0], True), num(J[n, 1], True)] for n, i in enumerate(self.idx)]
        head = {"start": "initial guess", "step": "accepted step", "rejected": "rejected trial",
                "jacobian": "Jacobian probe"}[kind]
        out = (f"<h3>Evaluation {k + 1} of {len(self.points)}: {head}</h3>"
               "<p><span class='eq'>r<sub>i</sub>(p) = |p − R<sub>i</sub>| − d<sub>i</sub></span> &nbsp; "
               "minimise <span class='eq'>F(p) = ½ Σ r<sub>i</sub>²</span></p>"
               f"<p class='dim'>At the current iterate p = {pt(p0)}: F = {num(cost(rms0))}, "
               f"RMS {num(rms0)} cm</p>"
               + html_table(["", "d", "|p − R|", "r", "∂r/∂x", "∂r/∂y"], rows)
               + matrix("JᵀJ", A) + matrix("Jᵀr", g)
               + "<p>LM update <span class='eq'>(JᵀJ + λ·D) δ = −Jᵀr</span>, "
               + ("D = diag(JᵀJ)" if LM_JAC_SCALING else "D = I")
               + ": small λ is a Gauss-Newton step, large λ a short step down the gradient.</p>")
        delta = x - p0
        if kind == "start":
            where = ("the library's OLS seed, as <span class='dim'>_multilaterate</span> and "
                     "<span class='dim'>_compute_cep</span> start" if self.initial is None else "where you clicked")
            out += f"<p>Initial guess <b>{pt(x)}</b>: {where}.</p>"
        elif kind == "step":
            try:
                gn = -np.linalg.solve(A, g)
            except np.linalg.LinAlgError:
                gn = None
            # MINPACK does not report λ: find the one that turns this δ into the update's solution
            D   = np.diag(A) if LM_JAC_SCALING else np.ones(2)
            Dd  = D * delta
            lam = -float(Dd @ (A @ delta + g)) / max(float(Dd @ Dd), 1e-18)
            fits = math.isfinite(lam) and lam > 0 and bool(
                np.hypot(*(np.linalg.solve(A + lam * np.diag(D), -g) - delta)) < 0.05 * np.hypot(*delta))
            out += (f"<p>δ taken = {pt(delta)}, |δ| = {num(float(np.hypot(*delta)))} cm"
                    + (f"<br>Gauss-Newton (λ = 0) would give δ = {pt(gn)}" if gn is not None else "")
                    + (f"<br>λ ≈ {num(lam)} <span class='muted'>(the λ that gives the step MINPACK took)</span>"
                       if fits else "")
                    + "</p><p>" + span("Accepted", pal.status["ok"])
                    + f": F {num(cost(rms0))} → {num(cost(rms))} (RMS {num(rms0)} → {num(rms)} cm). "
                      "When the drop matches the model's prediction, MINPACK lets λ fall, trusting the "
                      "quadratic model more next time.</p>")
        elif kind == "rejected":
            out += (f"<p>Trial p + δ = {pt(x)}, |δ| = {num(float(np.hypot(*delta)))} cm: F would rise "
                    f"{num(cost(rms0))} → {num(cost(rms))}. " + span("Rejected", pal.status["lost"])
                    + ": MINPACK raises λ and tries a shorter step, turned toward −Jᵀr.</p>")
        else:
            h     = float(np.hypot(*delta))
            moved = np.abs(delta) > 1e-12 * (1.0 + np.abs(p0))
            if moved.sum() == 1:            # a forward-difference probe moves one coordinate
                axis = "x" if moved[0] else "y"
                out += (f"<p>The optimiser nudges {axis} by h = {num(h)} cm (≈ √ε·|p|) and evaluates r again: one "
                        f"column of J by forward differences, <span class='eq'>∂r/∂{axis} ≈ "
                        f"(r(p + h·e<sub>{axis}</sub>) − r(p)) / h</span>.</p>")
            else:
                out += (f"<p>A move of only {num(h)} cm in both x and y: one of MINPACK's last, tiny trial steps "
                        "as it converges, or a probe round one. engine.lab cannot tell a move this small from a "
                        "probe, so it is listed with them.</p>")
        return out + self._footer()

    def _many_html(self, k: int) -> str:
        pal, sc = theme.current, self.sc
        rows = [[span("●", pal.algo[MINIMA_KEYS[m % len(MINIMA_KEYS)]]) + " " + "ABCDEFGHIJ"[m % 10], pt(pos),
                 num(rms), f"{count}"] for m, (pos, rms, count) in enumerate(self.minima)]
        moving = self.moving(k)
        seed_min = next((m for m, (pos, _, _) in enumerate(self.minima)
                         if np.hypot(*(pos - self.seed_fix)) < SAME_MIN_CM), None)
        out = (f"<h3>Many starts: {len(self.many)} initial guesses</h3>"
               f"<p>Each start runs the library's <span class='dim'>_nlls</span> on {names(self.idx)}. "
               "LM only ever walks downhill, so it ends in the minimum whose <b>basin of attraction</b> "
               f"holds the start. Iteration {k}: {moving} start{'s' * (moving != 1)} still moving.</p>"
               + html_table(["minimum", "position", "RMS cm", "starts"], rows))
        if collinear(sc.coords[self.idx]):
            out += (f"<p>{names(self.idx)} stand on one line, so their distances cannot tell the transmitter "
                    "from its mirror image across that line: the surface has two equally deep minima, "
                    "mirror images of each other.</p>")
        elif len(self.minima) > 1:
            out += ("<p>The extra minima are local: the range circles nearly meet there, and every small "
                    "move makes the fit worse, so LM stops.</p>")
        if seed_min is not None:
            out += (f"<p>The library starts from the OLS seed {pt(self.seed)}, which lies in the basin of "
                    f"<b>{'ABCDEFGHIJ'[seed_min % 10]}</b>.</p>")
        return out + self._footer(run=False)


# ─── Scenario editor ─────────────────────────────────────────────────────────

class ScenarioPanel(QWidget):
    """Where the scenario came from, an override per receiver, and the live
    status of each receiver's reading."""

    changed       = pyqtSignal()
    loadRequested = pyqtSignal()

    def __init__(self, count: int, parent=None):
        super().__init__(parent)
        self._sc = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        self.source = QLabel("—")
        self.source.setWordWrap(True)
        lay.addWidget(self.source)
        row = QHBoxLayout()
        self.b_load = QPushButton("Load latest frame")
        self.b_load.setToolTip("Take the simulator's latest frame as the scenario: its distances, the receivers "
                               "its filter kept and the true position")
        self.b_load.clicked.connect(self.loadRequested)
        self.b_clean = QPushButton("All clean")
        self.b_clean.setToolTip("Set every receiver back to Clean, with no offset")
        self.b_clean.clicked.connect(lambda: self.set_overrides(["clean"] * count, [0] * count, emit=True))
        row.addWidget(self.b_load)
        row.addWidget(self.b_clean)
        row.addStretch(1)
        lay.addLayout(row)
        hint = QLabel("Click the CEP or Blocking canvas to place the transmitter.")
        hint.setProperty("role", "muted")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)
        self.modes, self.sliders, self.values, self.statuses = [], [], [], []
        for i in range(count):
            name = QLabel(f"R{i}")
            name.setProperty("role", "heading")
            mode = QComboBox()
            for key, text in OVERRIDES.items():
                mode.addItem(text, key)
            mode.setToolTip(f"What happens to R{i}'s reading. Clean: as measured. Blocked: the direct sound and "
                            "sound bent round a corner are removed, so it hears a reflection, another "
                            "receiver's burst (with Crosstalk on), or nothing (the HC-SR04 times out). +offset: the slider's cm added. Off: 0 ticks.")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, MAX_OFFSET_CM)
            slider.setEnabled(False)
            slider.setToolTip(f"Centimetres added to R{i}'s reading (with +offset)")
            value = QLabel("+0 cm")
            value.setProperty("role", "dim")
            value.setFont(theme.mono_font(8))
            value.setFixedWidth(48)
            status = QLabel("")
            status.setFont(theme.ui_font(8.5))
            status.setWordWrap(True)
            grid.addWidget(name, 2 * i, 0)
            grid.addWidget(mode, 2 * i, 1)
            grid.addWidget(slider, 2 * i, 2)
            grid.addWidget(value, 2 * i, 3)
            grid.addWidget(status, 2 * i + 1, 1, 1, 3)
            mode.currentIndexChanged.connect(lambda _x, i=i: self._on_mode(i))
            slider.valueChanged.connect(lambda v, i=i: self._on_offset(i, v))
            self.modes.append(mode)
            self.sliders.append(slider)
            self.values.append(value)
            self.statuses.append(status)
        grid.setColumnStretch(2, 1)
        lay.addLayout(grid)
        self.just_changed = QCheckBox("Previous frame was clean")
        self.just_changed.setToolTip("Check each reading's jump against a frame where every receiver was clean, "
                                     "as on the first frame after a receiver is blocked. Unchecked: the readings "
                                     "have been like this for a while, so every jump is 0.")
        self.just_changed.toggled.connect(lambda _on: self.changed.emit())
        lay.addWidget(self.just_changed)
        lay.addStretch(1)

    def overrides(self) -> tuple[list, np.ndarray]:
        """Each receiver's override and the cm added to its reading."""
        modes = [m.currentData() for m in self.modes]
        return modes, np.array([s.value() if m == "offset" else 0.0 for m, s in zip(modes, self.sliders)])

    def set_overrides(self, modes, offsets=None, emit: bool = False) -> None:
        for i, m in enumerate(modes):
            self.modes[i].blockSignals(True)
            self.sliders[i].blockSignals(True)
            self.modes[i].setCurrentIndex(self.modes[i].findData(m))
            if offsets is not None:
                self.sliders[i].setValue(int(round(offsets[i])))
            self._sync_row(i)
            self.modes[i].blockSignals(False)
            self.sliders[i].blockSignals(False)
        if emit:
            self.changed.emit()

    def _sync_row(self, i: int) -> None:
        on = self.modes[i].currentData() == "offset"
        self.sliders[i].setEnabled(on)
        self.values[i].setText(f"+{self.sliders[i].value()} cm" if on else "")

    def _on_mode(self, i: int) -> None:
        self._sync_row(i)
        self.changed.emit()

    def _on_offset(self, i: int, _v: int) -> None:
        self._sync_row(i)
        self.changed.emit()

    def show_scenario(self, sc: Scenario) -> None:
        self._sc = sc
        where = "no true position (a log without truth)" if sc.truth is None else f"transmitter at {pt(sc.truth)} cm"
        self.source.setText(f"<b>{'Lab scenario' if sc.source == 'lab' else 'Snapshot of ' + sc.source}</b>"
                            f"<br>{where}")
        self._render_status()

    def _render_status(self) -> None:
        pal, sc = theme.current, self._sc
        if sc is None:
            return
        for i, label in enumerate(self.statuses):
            st, row = sc.status[i], sc.filter.rows[i]
            c   = pal.status.get(st, pal.text_dim)
            err = sc.error(i)
            keep = span("kept", pal.status["ok"]) if row.kept else span(verdict(row), pal.status["lost"])
            label.setText(f"{span('●', c)} {span(st, c)} · {num(row.distance)} cm"
                          + (f" ({num(err, True)})" if err is not None else "") + f" · {keep}")
            label.setToolTip(f"{STATUS.get(st, st)}; the library's distance"
                             + (" minus the true range in brackets" if err is not None else ""))

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in (QEvent.Type.StyleChange, QEvent.Type.PaletteChange):
            self._render_status()


# ─── Window ──────────────────────────────────────────────────────────────────

STORY_KEYS = {"Space": "toggle", "Left": "prev", "Right": "next", "Home": "first", "End": "last"}


def wants_key(widget, key) -> bool:
    """Does the focused ``widget`` (a button, slider, combo or tab bar) use this story key itself?"""
    if isinstance(widget, QComboBox):                 # Space opens it, Home and End choose (it ignores Left, Right)
        return key in (Qt.Key.Key_Space, Qt.Key.Key_Home, Qt.Key.Key_End)
    if isinstance(widget, QAbstractButton):           # Space presses it, toggles a check box
        return key == Qt.Key.Key_Space
    if isinstance(widget, QTabBar):                   # the arrows change the tab (it ignores Home, End)
        return key in (Qt.Key.Key_Left, Qt.Key.Key_Right)
    return key != Qt.Key.Key_Space                    # sliders move with the arrows, Home and End


class KeysToFocus(QObject):
    """Event filter on the lab's controls: a story key the focused control uses
    goes to that control, not to the window's shortcut.  Space toggles a focused
    check box rather than playing, Right moves a focused slider rather than
    stepping.  Elsewhere (the canvas, the explanation) the keys drive the story."""

    KEYS = {Qt.Key.Key_Space, Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Home, Qt.Key.Key_End}

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.ShortcutOverride and event.key() in self.KEYS:
            mods = event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
                                        | Qt.KeyboardModifier.MetaModifier)
            if not mods:
                # Accepted: the control gets the key press and the shortcut does not fire.  Set either
                # way, so this has the last word over an application-wide filter that ran first.
                event.setAccepted(wants_key(obj, event.key()))
        return False


class LabWindow(QMainWindow):
    """The algorithm lab: a Scenario dock beside one tab per algorithm story."""

    def __init__(self, sim, lock=None, parent=None):
        super().__init__(parent)
        self.sim      = sim
        self.lock     = lock if lock is not None else contextlib.nullcontext()
        self.lib      = UltraGPSPositionLib(sim.config_path)   # private: the simulator's filter memory is never touched
        self.scenario = None
        with self.lock:
            self.tx = np.array(sim.true, dtype=float)
        self.setWindowTitle("UltraGPS · algorithm lab")
        room = self.screen().availableGeometry()
        self.resize(min(1400, room.width() * 9 // 10), min(900, room.height() * 9 // 10))

        self.panel = ScenarioPanel(len(sim.receivers))
        scroll = QScrollArea()
        scroll.setWidget(self.panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.dock = QDockWidget("Scenario", self)
        self.dock.setObjectName("lab_scenario")
        self.dock.setWidget(scroll)
        self.dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                              | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                              | QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.dock)
        self.resizeDocks([self.dock], [270], Qt.Orientation.Horizontal)

        self.cep_tab      = CepTab()
        self.blocking_tab = BlockingTab()
        self.lm_tab       = LmTab()
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        for tab, title, tip in (
                (self.cep_tab, "CEP subsets", "Every receiver subset the library's CEP search scores, one at a time"),
                (self.blocking_tab, "Blocking", "Each algorithm's fix in turn, and a sweep of one bad reading"),
                (self.lm_tab, "LM optimiser", "How Levenberg-Marquardt walks down the cost surface from a guess")):
            self.tabs.setTabToolTip(self.tabs.addTab(tab, title), tip)
        corner = QWidget()
        cl = QHBoxLayout(corner)
        cl.setContentsMargins(0, 0, 6, 0)
        for action, text, tip in ((None, "Load latest frame", "Take the simulator's latest frame as the scenario"),
                                  (self.dock.toggleViewAction(), "Scenario panel", "Show or hide the Scenario panel")):
            b = QToolButton()
            if action is None:
                b.setText(text)
                b.clicked.connect(self.load_latest)
            else:
                action.setText(text)
                b.setDefaultAction(action)
            b.setToolTip(tip)
            cl.addWidget(b)
        self.tabs.setCornerWidget(corner, Qt.Corner.TopRightCorner)
        self.setCentralWidget(self.tabs)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self.rebuild)
        self.panel.changed.connect(self._debounce.start)
        self.panel.loadRequested.connect(self.load_latest)
        self.cep_tab.canvas.clicked.connect(self.place_transmitter)
        self.blocking_tab.canvas.clicked.connect(self.place_transmitter)
        for key, name in STORY_KEYS.items():
            a = QAction(self)
            a.setShortcut(QKeySequence(key))
            a.triggered.connect(lambda _=False, n=name: getattr(self.tabs.currentWidget().player, n)())
            self.addAction(a)
        self._keys = KeysToFocus(self)
        # Only on the controls made here, and the tab bar: a Python filter on the scroll bars Qt makes
        # for its scroll areas (they never take the focus anyway) crashes PyQt's clean-up at exit.
        for w in self.findChildren((QAbstractButton, QAbstractSlider, QComboBox)):
            if sip.ispycreated(w):
                w.installEventFilter(self._keys)
        self.tabs.tabBar().installEventFilter(self._keys)
        for tab in self.tab_list:
            tab.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)     # a click there hands the keys back to the story
        self.tabs.currentChanged.connect(self._focus_canvas)
        # Never narrower than the panel needs: the tabs give way first
        scroll.setMinimumWidth(self.panel.minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width())
        self._overrides_from_sim()
        self.rebuild()
        self.cep_tab.canvas.setFocus()

    @property
    def tab_list(self) -> list:
        return [self.cep_tab, self.blocking_tab, self.lm_tab]

    def _focus_canvas(self, _index: int) -> None:
        """A new tab: the story keys go to it (unless the tab bar is being driven by keys)."""
        if not self.tabs.tabBar().hasFocus():
            self.tabs.currentWidget().canvas.setFocus()

    # ─── Scenarios ───────────────────────────────────────────────────────────

    def load_frame(self, frame) -> None:
        """Take a simulator frame as the scenario (its distances, kept receivers and truth)."""
        self._overrides_from_sim()
        self.tx = None if frame.true is None else np.array(frame.true, dtype=float)
        receivers, world, _, _, settings, limits = self._sim_parts()
        lib  = self._sync_lib(limits)
        d     = np.array(frame.solution.distances, dtype=float)
        ticks = (list(frame.ticks) + [0] * len(d))[:len(d)]     # a replayed log line may be short or long
        top   = lib.max_distance if lib.max_distance is not None else lib._max_receiver_dist
        sane  = [int(i) for i in frame.solution.sane]
        rows  = [lab.FilterRow(i, int(t), lib._offsets[i]["slope"], lib._offsets[i]["intercept"], float(d[i]),
                               bool(0 < d[i] <= top), None, None, i in sane) for i, t in enumerate(ticks)]
        cep   = frame.estimates.get("CEP")
        held  = np.array(cep.pos, dtype=float) if cep is not None and not cep.ok and cep.pos is not None else None
        self._set_scenario(Scenario(
            f"frame {frame.index}", None if self.tx is None else self.tx.copy(), ticks, d,
            lab.FilterTrace(rows, float(top), float(lib.max_differential), sane), list(frame.readings),
            [r.state for r in receivers], np.zeros(len(d)), lib._receiver_coords.copy(),
            [r.facing_deg for r in receivers], world, settings, held))

    def load_latest(self) -> None:
        if self.sim.last is None:
            self.statusBar().showMessage("The simulator has no frame yet.", 4000)
        else:
            self.load_frame(self.sim.last)

    def place_transmitter(self, x: float, y: float) -> None:
        """Put the lab transmitter at (x, y) and measure the scenario there."""
        self.tx = np.array([x, y], dtype=float)
        self.rebuild()

    def rebuild(self) -> None:
        """Measure a lab scenario at the transmitter with the panel's overrides."""
        if self.tx is None:
            self.statusBar().showMessage("Click the CEP or Blocking canvas to place the transmitter first.", 4000)
            return
        receivers, world, env, heading, settings, limits = self._sim_parts()
        lib = self._sync_lib(limits)
        modes, offsets = self.panel.overrides()
        env = env.copy(amp_jitter_db=0.0, tick_dither=False, noise_rate_hz=0.0)

        def ping(states):
            for r, s in zip(receivers, states):
                r.state = s
            return Hardware(receivers, world, env, Faults(noise_cm=0.0), seed=0).measure(self.tx, heading)

        states   = [STATE[m] for m in modes]
        readings = ping(states)
        ticks    = [r.tick + offsets[i] / lib._offsets[i]["slope"] for i, r in enumerate(readings)]
        if self.panel.just_changed.isChecked():
            previous = lib._ticks_to_distances([r.tick for r in ping(["on"] * len(receivers))])
            note     = "the frame before, when every receiver was clean"
        else:
            previous = lib._ticks_to_distances(ticks)
            note     = "the same readings a frame earlier (steady state)"
        lib._filter_receivers(previous)
        trace = lab.trace_filter(lib, ticks)
        d     = np.array([row.distance for row in trace.rows])
        self._set_scenario(Scenario("lab", self.tx.copy(), ticks, d, trace, readings, states, offsets,
                                    lib._receiver_coords.copy(), [r.facing_deg for r in receivers], world,
                                    settings, None, note))

    def _set_scenario(self, sc: Scenario) -> None:
        self.scenario = sc
        self.panel.show_scenario(sc)
        for tab in self.tab_list:
            tab.set_scenario(sc, self.lib)

    def _sim_parts(self):
        """Copies of what the lab needs from the simulator, taken under its lock."""
        with self.lock:
            receivers = [replace(r, position=np.array(r.position, dtype=float)) for r in self.sim.receivers]
            world     = copy.deepcopy(self.sim.world)
            env       = self.sim.env
            heading   = float(self.sim.motion.heading)
            settings  = copy.copy(self.sim.solvers.settings)
            limits    = (self.sim.solvers.lib.max_distance, self.sim.solvers.lib.max_differential)
        return receivers, world, env, heading, settings, limits

    def _sync_lib(self, limits) -> UltraGPSPositionLib:
        """The private library with the current calibration, the simulator's filter limits and no memory."""
        self.lib.reload()
        self.lib.max_distance, self.lib.max_differential = limits
        return self.lib

    def _overrides_from_sim(self) -> None:
        mode = {"on": "clean", "blocked": "blocked", "off": "off"}
        self.panel.set_overrides([mode.get(r.state, "clean") for r in self.sim.receivers],
                                 np.zeros(len(self.sim.receivers)))

    def follow_settings(self) -> None:
        """Take up the simulator's solver settings if one the lab uses changed (an EKF
        setting alone does not restart the stories or drop a sweep)."""
        with self.lock:
            settings = copy.copy(self.sim.solvers.settings)
        old = self.scenario.settings if self.scenario is not None else None
        if old is not None and any(getattr(settings, f) != getattr(old, f) for f in LAB_SETTINGS):
            self._set_scenario(replace(self.scenario, settings=settings))

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            self.follow_settings()

    def hideEvent(self, event) -> None:
        for tab in self.tab_list:
            tab.player.pause()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:
        self.blocking_tab.stop()
        super().closeEvent(event)
