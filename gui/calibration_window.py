"""
Calibration window — calibrating each receiver's tick → distance conversion
on the simulated sensors, in four steps, top left to bottom right:

1. Points: choose 2 to 4 calibration points by searching for the best set
   (``engine.cal_points``), picking them on the arena, or typing them.  The
   score is the calibration error the set causes over the arena.
2. Run: the calibration library collects pings at each point in turn, with
   the current physics, faults and receiver states, fits each receiver's line
   and writes it to config.toml.
3. Offsets: each receiver's slope and intercept next to its true response,
   with the distance error they cause; edit, save, reload, or use the truth.
4. Response: one receiver's tick → distance line, true and calibrated, the
   modal tick it read at each calibration point, and the error between the
   lines, so a slope or intercept error (or a point heard as an echo) shows.
"""

from __future__ import annotations

import html
import math
from contextlib import nullcontext

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
                             QLabel, QMainWindow, QProgressBar, QPushButton, QScrollArea, QSpinBox,
                             QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from engine import cal_points
from engine.calibration import read_offsets
from gui import draw, theme
from gui.charts import nice_step
from gui.theme import color

ERROR_AT  = (100.0, 300.0)        # distances the offsets table reports the error at, cm
GOOD_CM   = 1.0                   # |error| up to this is drawn green, up to FAIR_CM amber, beyond red
FAIR_CM   = 3.0
PLOT_MAX  = 500.0                 # the response plot covers 0 .. this many cm
ROW_PX    = 24                    # table row height
SCORE_TIP = ("Arena score of the points: the RMS distance error their calibration causes over the "
             "whole arena and every receiver, per cm of error in a single reading "
             "(sqrt(1/n + (D − d̄)²/Sxx), see engine/cal_points.py). Lower is better. It depends "
             "only on geometry: a receiver wants its points at very different distances from it, "
             "centred on the distances it will measure.")


def error_colour(err: float) -> str:
    """Status colour for a distance error: green, amber or red."""
    s = theme.current.status
    if not math.isfinite(err):
        return s["lost"]
    return s["ok"] if abs(err) <= GOOD_CM else s["near"] if abs(err) <= FAIR_CM else s["lost"]


def calibration_error(true: tuple, cal: tuple, d: float) -> float:
    """Distance error (cm) at true distance ``d``: the tick the sensor gives
    there, converted back with the calibrated (slope, intercept)."""
    tick = (d - true[1]) / true[0]
    return cal[0] * tick + cal[1] - d


# ─── Background search ───────────────────────────────────────────────────────

class SearchJob(QThread):
    """``engine.cal_points.search`` for one point count, off the GUI thread."""

    found = pyqtSignal(object)            # (n, 2) array of points

    def __init__(self, coords, n: int, parent=None):
        super().__init__(parent)
        self.coords, self.n = np.array(coords, dtype=float), n

    def run(self) -> None:
        self.found.emit(cal_points.search(self.coords, self.n)[self.n])


# ─── Canvases ────────────────────────────────────────────────────────────────

class PointsView(QWidget):
    """Small plan of the arena: receivers, walls, obstacles, the points in
    use (dim) and the points being edited (accent)."""

    def __init__(self, sim, parent=None):
        super().__init__(parent)
        self.sim     = sim
        self.edited: list = []
        self.in_use: list = []
        self.view    = draw.View()
        self.setMinimumSize(200, 150)
        self.setToolTip("Where the calibration points are: bright squares are the points listed "
                        "above, dim squares the ones in config.toml when they differ")

    def set_points(self, edited, in_use) -> None:
        self.edited = [tuple(map(float, q)) for q in edited]
        self.in_use = [tuple(map(float, q)) for q in in_use]
        self.update()

    def paintEvent(self, _event) -> None:
        pal, v, world = theme.current, self.view, self.sim.world
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), color(pal.bg))
        pts = np.array([np.asarray(q, dtype=float) for q in world.walls] + list(self.sim.coords)
                       + [np.asarray(q) for q in self.edited])
        v.resize(self.width(), self.height())
        v.fit((*pts.min(axis=0), *pts.max(axis=0)), margin_px=18)
        draw.grid(p, v, labels=False)
        draw.wall_outline(p, v, world.walls, pal.wall)
        for o in world.obstacles:
            draw.hatched_polygon(p, v, o.points, pal.obstacle, 5.0)
        for r in self.sim.receivers:
            draw.receiver_glyph(p, v, r.position, r.facing_deg, pal.receiver, None, r.state, 6.0)
        for q in self.in_use:
            if q not in self.edited:
                draw.marker(p, v.px(*q), "square", pal.text_muted, 7.0, 1.0)
        for k, q in enumerate(self.edited):
            c = v.px(*q)
            draw.marker(p, c, "square", pal.accent, 8.0, 1.3, fill=0.25)
            draw.label(p, QPointF(c.x() + 8, c.y() - 8), f"P{k + 1}", pal.accent, 7.5, bold=True)


class ResponsePlot(QWidget):
    """One receiver's tick → distance: the true response, the calibrated
    conversion and the modal tick read at each calibration point (top), and
    the distance error between the two lines (bottom), where a slope error
    shows as a tilt and an intercept error as an offset."""

    LEFT, RIGHT, TOP, BOTTOM, GAP = 46, 14, 42, 24, 16       # px; the title and legend sit above TOP

    def __init__(self, parent=None):
        super().__init__(parent)
        self.receiver = 0
        self.true     = (1.0, 0.0)
        self.cal      = (1.0, 0.0)
        self.points: list = []            # [(label, tick, known distance, note)]; a note marks a doubtful point
        self.setMinimumSize(300, 220)

    def set_data(self, receiver: int, true: tuple, cal: tuple, points: list) -> None:
        self.receiver, self.true, self.cal, self.points = receiver, true, cal, points
        self.update()

    def paintEvent(self, _event) -> None:
        pal = theme.current
        p   = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), color(pal.bg))
        (a, b), (ac, bc) = self.true, self.cal
        values = [a, b, ac, bc] + [v for _, t, d, _ in self.points for v in (t, d)]
        if a <= 0 or not all(map(math.isfinite, values)):
            draw.label(p, QPointF(self.width() / 2, self.height() / 2),
                       "nothing to plot: the true slope must be positive", pal.status["lost"], 9, "center")
            return
        ticks  = [t for _, t, _, _ in self.points]          # a point read wrong can lie far off
        t0, t1 = min([-b / a] + ticks), max([(PLOT_MAX - b) / a] + ticks)
        inner  = self.height() - self.TOP - self.BOTTOM - self.GAP
        top    = QRectF(self.LEFT, self.TOP, self.width() - self.LEFT - self.RIGHT, inner * 0.62)
        bottom = QRectF(self.LEFT, top.bottom() + self.GAP, top.width(), inner * 0.38)
        res    = [d - (a * t + b) for _, t, d, _ in self.points]

        def err(t):
            return (ac * t + bc) - (a * t + b)

        emax = max([abs(err(t0)), abs(err(t1)), 2.0] + [abs(r) for r in res]) * 1.15

        def X(t):
            return top.left() + (t - t0) / (t1 - t0) * top.width()

        def Yd(d):
            return top.bottom() - d / PLOT_MAX * top.height()

        def Ye(e):
            return bottom.center().y() - e / emax * bottom.height() / 2

        def line(f, y):
            return draw.polyline_path([QPointF(X(t), y(f(t))) for t in (t0, t1)])

        self._axes(p, top, bottom, t0, t1, X, Yd, Ye, emax)
        p.save()
        p.setClipRect(top)
        draw.stroke(p, line(lambda t: a * t + b, Yd), pal.truth, 1.2)
        draw.stroke(p, line(lambda t: ac * t + bc, Yd), pal.accent, 1.4, dash=draw.DASH)
        p.restore()
        draw.stroke(p, line(lambda t: 0.0, Ye), pal.truth, 1.2)
        draw.stroke(p, line(err, Ye), pal.accent, 1.4, dash=draw.DASH)
        for k, d in enumerate(ERROR_AT):
            t, e = (d - b) / a, err((d - b) / a)
            guide = draw.polyline_path([QPointF(X(t), top.top()), QPointF(X(t), bottom.bottom())])
            draw.stroke(p, guide, pal.text_muted, 1.0, 0.7, draw.DOT, glow=False)
            draw.marker(p, QPointF(X(t), Ye(e)), "circle", error_colour(e), 6.0, 1.3)
            self._note(p, QPointF(X(t), bottom.top() + 8 + 12 * k), f"{e:+.1f} cm at {d:.0f}", error_colour(e),
                       bold=False)
        for (lab, t, d, note), r in zip(self.points, res):
            c = pal.status["echo"] if note else pal.text
            for q in (QPointF(X(t), Yd(d)), QPointF(X(t), Ye(r))):
                draw.marker(p, q, "square", c, 7.0, 1.2, fill=0.3)
            self._note(p, QPointF(X(t), Yd(d) - 9), f"{lab} {note}".strip(), c)
        self._legend(p)

    def _note(self, p, at: QPointF, text: str, c: str, bold: bool = True) -> None:
        """A label just right of ``at``, or just left of it where it would run off the plot."""
        p.setFont(theme.mono_font(7.5, bold))
        if at.x() + 7 + p.fontMetrics().horizontalAdvance(text) > self.width() - 4:
            draw.label(p, QPointF(at.x() - 7, at.y()), text, c, 7.5, "right", bold)
        else:
            draw.label(p, QPointF(at.x() + 7, at.y()), text, c, 7.5, bold=bold)

    def _axes(self, p, top, bottom, t0, t1, X, Yd, Ye, emax) -> None:
        pal = theme.current
        draw.label(p, QPointF(top.left(), 12), f"R{self.receiver}  tick → distance, cm", pal.text, 8.5,
                   bold=True, halo=False)

        def rule(a, b):
            p.setPen(draw.pen(pal.grid_minor, 1.0))
            p.drawLine(a, b)

        for d in np.arange(0, PLOT_MAX + 1e-9, nice_step(PLOT_MAX, 5)):
            rule(QPointF(top.left(), Yd(d)), QPointF(top.right(), Yd(d)))
            draw.label(p, QPointF(top.left() - 5, Yd(d)), f"{d:g}", pal.text_muted, 7.5, "right", halo=False)
        st = nice_step(2 * emax, 4)
        for e in np.arange(-math.floor(emax / st) * st, emax, st):
            rule(QPointF(bottom.left(), Ye(e)), QPointF(bottom.right(), Ye(e)))
            draw.label(p, QPointF(bottom.left() - 5, Ye(e)), f"{e:+g}" if e else "0", pal.text_muted, 7.5,
                       "right", halo=False)
        st = nice_step(t1 - t0, 6)
        for t in np.arange(math.ceil(t0 / st) * st, t1 + 1e-9, st):
            rule(QPointF(X(t), top.top()), QPointF(X(t), top.bottom()))
            rule(QPointF(X(t), bottom.top()), QPointF(X(t), bottom.bottom()))
            draw.label(p, QPointF(X(t), bottom.bottom() + 10), f"{t:g}", pal.text_muted, 7.5, "center",
                       halo=False)
        for rect in (top, bottom):
            p.setPen(draw.pen(pal.grid_major, 1.0))
            p.drawLine(rect.bottomLeft(), rect.bottomRight())
            p.drawLine(rect.bottomLeft(), rect.topLeft())
        draw.label(p, QPointF(bottom.left() + 4, bottom.bottom() - 8), "error, cm", pal.text_muted, 7.5)

    def _legend(self, p) -> None:
        """One row under the title, clear of the data."""
        pal   = theme.current
        items = [("true response", pal.truth, None), ("calibrated", pal.accent, draw.DASH)]
        if self.points:
            items.append(("modal tick at a point", pal.text, "square"))
        x, y = float(self.LEFT), 28.0
        for text, c, style in items:
            if style == "square":
                draw.marker(p, QPointF(x + 9, y), "square", c, 6.0, 1.1, fill=0.3)
            else:
                draw.stroke(p, draw.polyline_path([QPointF(x, y), QPointF(x + 18, y)]), c, 1.3, dash=style)
            x = draw.label(p, QPointF(x + 24, y), text, pal.text_dim, 7.5, halo=False).right() + 16


# ─── Window ──────────────────────────────────────────────────────────────────

class CalibrationWindow(QMainWindow):
    """Calibration points, simulated calibration runs and the offsets they produce."""

    pickRequested = pyqtSignal(int)       # pick this many points on the arena, then call set_points
    pointsChanged = pyqtSignal(object)    # the calibration points now in config.toml

    def __init__(self, sim, calibrator, lock=None, parent=None):
        super().__init__(parent)
        self.sim        = sim
        self.calibrator = calibrator
        self.lock       = lock if lock is not None else nullcontext()
        self.search_job: SearchJob | None = None
        self._read_config()                       # what config.toml says: _in_use, _saved, _reads
        self._points    = [list(q) for q in self._in_use]
        self._offsets   = {i: dict(o) for i, o in self._saved.items()}
        self._run_points: list | None = None      # the points of the last run started here
        self._seen      = (None, None, None)      # (state, version, palette) last shown
        self._filling   = False                   # tables being filled by code, not edited
        self.setWindowTitle("Calibration")
        self.resize(1180, 800)

        left = QWidget()
        lv   = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 4, 0)
        lv.addWidget(self._build_points_box(), 1)
        lv.addWidget(self._build_run_box())
        scroll = QScrollArea()
        scroll.setWidget(left)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(320)

        right = QWidget()
        rv    = QVBoxLayout(right)
        rv.setContentsMargins(4, 0, 0, 0)
        rv.addWidget(self._build_offsets_box())
        rv.addWidget(self._build_plot_box(), 1)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(scroll)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 820])
        central = QWidget()
        cv = QVBoxLayout(central)
        cv.setContentsMargins(8, 8, 8, 8)
        cv.addWidget(split)
        self.setCentralWidget(central)

        self.timer = QTimer(self)               # follows the calibrator while the window is shown
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)
        self.poll()

    # ─── Layout ──────────────────────────────────────────────────────────────

    def _build_points_box(self) -> QGroupBox:
        box = QGroupBox("1  Calibration points")
        self.count = QSpinBox()
        self.count.setRange(2, 4)
        self.count.setValue(min(max(len(self._points), 2), 4))
        self.count.setSuffix(" points")
        self.count.setToolTip("How many calibration points: more points average out reading errors "
                              "but take longer to collect")
        self.search_btn = QPushButton("Search best points")
        self.search_btn.setToolTip("Find the set of this many points with the lowest arena score "
                                   "(engine/cal_points.py)")
        self.pick_btn = QPushButton("Pick on arena")
        self.pick_btn.setToolTip("Click this many points on the main window's arena")
        self.points_table = _table(0, ["x cm", "y cm"], "The calibration points; double-click a value "
                                                          "to edit it")       # columns: label, x, y
        self.score = QLabel()
        self.score.setToolTip(SCORE_TIP)
        self.preview = PointsView(self.sim)
        self.use_btn = QPushButton("Use these points")
        self.use_btn.setToolTip("Write the points to config.toml (cal_points and every receiver's "
                                "cal_distances) and show them on the arena")
        self.revert_btn = QPushButton("Revert")
        self.revert_btn.setToolTip("Go back to the points in config.toml")

        g = QGridLayout(box)
        g.addWidget(self.count, 0, 0)
        g.addWidget(self.points_table, 1, 0, 1, 2)
        g.addWidget(self.search_btn, 2, 0)
        g.addWidget(self.pick_btn, 2, 1)
        g.addWidget(self.score, 3, 0, 1, 2)
        g.addWidget(self.preview, 4, 0, 1, 2)
        g.addWidget(self.use_btn, 5, 0)
        g.addWidget(self.revert_btn, 5, 1)
        g.setRowStretch(4, 1)

        self.count.valueChanged.connect(self._on_count)
        self.search_btn.clicked.connect(self.search)
        self.pick_btn.clicked.connect(lambda: self.pickRequested.emit(self.count.value()))
        self.points_table.itemChanged.connect(self._on_point_edited)
        self.use_btn.clicked.connect(self.use_points)
        self.revert_btn.clicked.connect(lambda: self.set_points(self._in_use))
        return box

    def _build_run_box(self) -> QGroupBox:
        box = QGroupBox("2  Calibration run")
        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.setToolTip("Run the calibration library at the points above (writing them "
                                      "to config.toml first if you changed them), then save and apply "
                                      "the offsets")
        self.state = QLabel()
        self.state.setFont(theme.mono_font(9, bold=True))
        self.run_bar = QProgressBar()
        self.run_bar.setRange(0, 1000)
        self.run_bar.setToolTip("Readings collected: a point is done once every receiver has read its "
                                "most frequent tick calibration_reads times")
        self.run_detail = QLabel()
        self.run_detail.setFont(theme.mono_font(8.5))
        self.run_detail.setProperty("role", "dim")
        self.run_detail.setWordWrap(True)
        note = QLabel("The calibration pings the simulated sensors with the current physics, faults "
                      "and receiver states: noise and range walk shift the modal ticks, and a "
                      "receiver without line of sight to a point calibrates on an echo. The "
                      "simulation waits while it runs.")
        note.setWordWrap(True)
        note.setProperty("role", "muted")
        g = QGridLayout(box)
        g.addWidget(self.calibrate_btn, 0, 0)
        g.addWidget(self.state, 0, 1)
        g.addWidget(self.run_bar, 1, 0, 1, 2)
        g.addWidget(self.run_detail, 2, 0, 1, 2)
        g.addWidget(note, 3, 0, 1, 2)
        g.setColumnStretch(1, 1)
        self.calibrate_btn.clicked.connect(self.calibrate)
        return box

    def _build_offsets_box(self) -> QGroupBox:
        box   = QGroupBox("3  Offsets   distance = tick × slope + intercept")
        heads = ["slope", "intercept", "true\nslope", "true\nintercept"] + \
                [f"error cm\nat {d:.0f}" for d in ERROR_AT]
        self.offsets_table = _table(len(self.sim.receivers), heads,
                                    "Click a row to plot that receiver's response below")
        tips = ["Calibrated slope in config.toml (cm per tick); double-click to edit",
                "Calibrated intercept in config.toml (cm); double-click to edit",
                "The simulated sensor's real slope (sensors.toml)",
                "The simulated sensor's real intercept (sensors.toml)"] + \
               [f"Distance error the calibration causes at a true {d:.0f} cm: the tick the sensor "
                f"gives there, converted with the calibrated slope and intercept, minus {d:.0f} cm. "
                f"Green within {GOOD_CM:g} cm, amber within {FAIR_CM:g} cm, red beyond" for d in ERROR_AT]
        for k, tip in enumerate(tips):
            self.offsets_table.horizontalHeaderItem(k + 1).setToolTip(tip)
        self.save_btn = QPushButton("Save && apply")
        self.save_btn.setToolTip("Write the slopes and intercepts above to config.toml and reload "
                                 "them into the positioning library")
        self.reload_btn = QPushButton("Reload from file")
        self.reload_btn.setToolTip("Discard edits and show the offsets in config.toml")
        self.true_btn = QPushButton("Use true values")
        self.true_btn.setToolTip("Fill in each sensor's true response (a perfect calibration); "
                                 "press Save & apply to use it")
        self.offsets_note = QLabel()
        self.offsets_note.setProperty("role", "dim")
        row = QHBoxLayout()
        for w in (self.save_btn, self.reload_btn, self.true_btn):
            row.addWidget(w)
        row.addSpacing(8)
        row.addWidget(self.offsets_note, 1)
        v = QVBoxLayout(box)
        v.addWidget(self.offsets_table)
        v.addLayout(row)
        self.offsets_table.itemChanged.connect(self._on_offset_edited)
        self.offsets_table.cellClicked.connect(lambda row, _col: self.receiver.setCurrentIndex(row))
        self.save_btn.clicked.connect(self.save_offsets)
        self.reload_btn.clicked.connect(self.reload_offsets)
        self.true_btn.clicked.connect(self.use_true_values)
        return box

    def _build_plot_box(self) -> QGroupBox:
        box = QGroupBox("4  Response")
        self.receiver = QComboBox()
        for r in self.sim.receivers:
            self.receiver.addItem(f"R{r.id}", r.id)
        self.receiver.setToolTip("The receiver to plot (or click its row in the table)")
        self.plot = ResponsePlot()
        self.plot.setToolTip("Top: tick against distance, the sensor's true line (sensors.toml) and "
                             "the calibrated one (config.toml). Bottom: the distance error between "
                             "them; a slope error tilts it, an intercept error shifts it. Squares: the "
                             "modal tick read at each calibration point in the last run, against the "
                             "point's known distance.")
        head = QHBoxLayout()
        lab_ = QLabel("Receiver")
        lab_.setProperty("role", "dim")
        head.addWidget(lab_)
        head.addWidget(self.receiver)
        head.addStretch(1)
        v = QVBoxLayout(box)
        v.addLayout(head)
        v.addWidget(self.plot, 1)
        self.receiver.currentIndexChanged.connect(self._update_plot)
        return box

    # ─── 1. Points ───────────────────────────────────────────────────────────

    def points(self) -> list:
        """The points in the table (not necessarily in use yet)."""
        return [list(q) for q in self._points]

    def set_points(self, points) -> None:
        """Show ``points`` in the table, ready for "Use these points" (the first
        4; with fewer than 2, one is added in the middle of the receivers)."""
        self._resize_points([[float(x), float(y)] for x, y in points], min(max(len(points), 2), 4))

    def _on_count(self, n: int) -> None:
        """Drop points, or add new ones in the middle of the receivers to edit."""
        self._resize_points(self._points, n)

    def _resize_points(self, points: list, n: int) -> None:
        """Show the first ``n`` of ``points``, adding points in the middle of the receivers if short."""
        centre = (self.sim.believed.min(axis=0) + self.sim.believed.max(axis=0)) / 2
        self._points = [list(q) for q in points[:n]]
        while len(self._points) < n:
            self._points.append([float(centre[0]), float(centre[1])])
        self.count.blockSignals(True)
        self.count.setValue(n)
        self.count.blockSignals(False)
        self._fill_points()

    def _fill_points(self) -> None:
        pal, t = theme.current, self.points_table
        self._filling = True
        t.setRowCount(len(self._points))
        for k, (x, y) in enumerate(self._points):
            far = np.linalg.norm(self.sim.believed - np.array([x, y]), axis=1)
            out = far.max() > cal_points.MAX_RANGE_CM
            tip = (f"R{int(np.argmax(far))} is {far.max():.0f} cm away: beyond the HC-SR04's "
                   f"{cal_points.MAX_RANGE_CM:.0f} cm range" if out else "Double-click to edit")
            t.setItem(k, 0, _row_label(f"P{k + 1}"))
            for c, value in enumerate((x, y)):
                t.setItem(k, c + 1, _item(f"{value:.1f}", pal.status["range"] if out else pal.text, tip=tip))
        _fit_rows(t)
        self._filling = False
        self._update_score()

    def _on_point_edited(self, item: QTableWidgetItem) -> None:
        if self._filling or item.column() == 0:
            return
        value = _number(item.text())
        if value is not None:
            self._points[item.row()][item.column() - 1] = value
        self._fill_points()               # not a number: puts the old value back

    def _update_score(self) -> None:
        coords = self.sim.believed
        in_use = self._in_use
        same   = len(in_use) == len(self._points) and np.allclose(in_use, self._points)
        text   = f"Arena score {cal_points.arena_score(self._points, coords):.3f}"
        self.score.setText(text + ("   (in use)" if same else
                                   f"   in use {cal_points.arena_score(in_use, coords):.3f}   lower is better"))
        self.preview.set_points(self._points, in_use)
        self.use_btn.setEnabled(not same and not self.calibrator.busy)
        self.revert_btn.setEnabled(not same)

    def search(self) -> None:
        """Find the best set of ``count`` points on a background thread."""
        if self.search_job is not None:
            return
        self.search_btn.setEnabled(False)
        self.search_btn.setText("Searching...")
        self.search_job = SearchJob(self.sim.believed, self.count.value(), self)
        self.search_job.found.connect(self.set_points)
        self.search_job.finished.connect(self._search_done)
        self.search_job.start()

    def _search_done(self) -> None:
        self.search_job.deleteLater()
        self.search_job = None
        self.search_btn.setEnabled(True)
        self.search_btn.setText("Search best points")

    def use_points(self) -> None:
        """Write the table's points to config.toml and tell the arena."""
        try:
            self.calibrator.set_points(self._points)
        except RuntimeError as exc:
            self.statusBar().showMessage(str(exc), 4000)
            return
        self._read_config()
        self.pointsChanged.emit(self.points())
        self._update_score()
        self.statusBar().showMessage("Calibration points written to config.toml", 4000)

    # ─── 2. Run ──────────────────────────────────────────────────────────────

    def calibrate(self) -> None:
        """Start a calibration run at the table's points."""
        if self.calibrator.busy:
            return
        if self._points != self._in_use:
            self.use_points()
        self._run_points = [list(q) for q in self._in_use]
        with self.lock:
            self.calibrator.start()
        self.poll()

    def poll(self) -> None:
        """Follow the calibrator: progress while it runs, the points and offsets
        in config.toml once it is not running (see ``_read_config``)."""
        c     = self.calibrator
        state = c.state                          # once: the run's thread changes it
        busy  = state.startswith("run")
        if busy:
            self._show_progress(int(state.split()[1]))
            now = (state, self._seen[1], theme.current.name)   # new offsets show once it is done
        else:
            now = (state, c.version, theme.current.name)
            if now[:2] != self._seen[:2]:
                self._read_config()
            if now[1] != self._seen[1]:
                self._offsets = {i: dict(o) for i, o in self._saved.items()}
        if now == self._seen:
            return
        self._seen = now
        self._fill_points()
        self._fill_offsets()
        self._show_state(state)

    def _read_config(self) -> None:
        """Remember the points, offsets and reads per point in config.toml.
        Only while no run is going: the calibration library rewrites the file
        in place at the end of a run, and a read then can find half a file."""
        c = self.calibrator
        self._in_use = [list(map(float, q)) for q in c.points]
        self._saved  = read_offsets(c.config_path)
        self._reads  = max(c.cal.calibration_reads, 1)

    def _show_state(self, state: str) -> None:
        c, pal = self.calibrator, theme.current
        busy   = state.startswith("run")
        for w in (self.calibrate_btn, self.save_btn, self.reload_btn):
            w.setEnabled(not busy)
        text, col = {"idle":  ("not run yet", pal.text_dim),
                     "done":  ("done: offsets saved and applied", pal.status["ok"]),
                     "error": (f"error: {c.error}", pal.status["lost"])}.get(
                         state, (f"collecting at P{state.split()[-1]}", pal.accent))
        self.state.setText(_coloured(text, col))
        if state == "done":
            self.run_bar.setValue(1000)
            self.run_detail.setText(self._run_summary())
        elif not busy:
            self.run_bar.setValue(0)
            self.run_detail.setText("")

    def _run_summary(self) -> str:
        """Which receivers the last run left uncalibrated or got badly wrong."""
        pal, c  = theme.current, self.calibrator
        skipped = [f"R{r.id}" for r in self.sim.receivers if r.id not in c.offsets]
        bad     = [f"R{r.id}" for r in self.sim.receivers if r.id in c.offsets and max(
                   abs(calibration_error((r.slope, r.intercept), (c.offsets[r.id]["slope"],
                       c.offsets[r.id]["intercept"]), d)) for d in ERROR_AT) > FAIR_CM]
        lines = []
        if skipped:
            lines.append(_coloured(f"{', '.join(skipped)} kept the old offsets: the same tick at every "
                                   f"point (no signal?)", pal.status["lost"]))
        if bad:
            lines.append(_coloured(f"{', '.join(bad)} off by more than {FAIR_CM:g} cm: plot them below "
                                   f"to see which point read wrong", pal.status["near"]))
        return "<br>".join(lines) or _coloured(f"every receiver within {FAIR_CM:g} cm", pal.status["ok"])

    def _show_progress(self, k: int) -> None:
        """Progress of the run at point ``k`` (1-based)."""
        c    = self.calibrator
        pts  = self._run_points or self._in_use
        need = self._reads
        if k > len(pts):
            return                        # a run started elsewhere, at points not read yet
        got  = [min(c.cal.get_max_count(r.id, k), need) for r in self.sim.receivers]
        self.run_bar.setValue(int(1000 * (k - 1 + np.mean(got) / need) / len(pts)))
        x, y  = pts[k - 1]
        pings = sum(c.cal.get_histogram_data(self.sim.receivers[0].id, k).values())
        seen  = " · ".join(f"R{r.id} {g}" for r, g in zip(self.sim.receivers, got))
        self.run_detail.setText(f"P{k} of {len(pts)} at ({x:.0f}, {y:.0f}): {pings} pings<br>"
                                f"times each has read its modal tick, of {need}:<br>{seen}")
        self._update_plot()

    # ─── 3. Offsets ──────────────────────────────────────────────────────────

    def offsets(self) -> dict:
        """The slopes and intercepts in the table ``{id: {"slope", "intercept"}}``."""
        return {i: dict(o) for i, o in self._offsets.items()}

    def _fill_offsets(self) -> None:
        """Columns: receiver, slope, intercept (editable), true slope, true intercept, errors."""
        pal, t = theme.current, self.offsets_table
        self._filling = True
        for row, r in enumerate(self.sim.receivers):
            o    = self._offsets[r.id]
            true = (r.slope, r.intercept)
            cal  = (o["slope"], o["intercept"])
            cells = [(f"{cal[0]:.5f}", pal.text), (f"{cal[1]:.3f}", pal.text),
                     (f"{true[0]:.5f}", pal.text_dim), (f"{true[1]:.3f}", pal.text_dim)]
            cells += [(f"{e:+.2f}", error_colour(e)) for e in
                      (calibration_error(true, cal, d) for d in ERROR_AT)]
            t.setItem(row, 0, _row_label(f"R{r.id}"))
            for col, (text, c) in enumerate(cells, start=1):
                t.setItem(row, col, _item(text, c, editable=col <= 2))
        _fit_rows(t)
        self._filling = False
        edited = self._offsets != self._saved
        self.offsets_note.setText(_coloured("edited: not saved", pal.status["near"]) if edited else
                                  "as in config.toml")
        self._update_plot()

    def _on_offset_edited(self, item: QTableWidgetItem) -> None:
        if self._filling or item.column() not in (1, 2):
            return
        rid   = self.sim.receivers[item.row()].id
        value = _number(item.text())
        if value is not None:
            self._offsets[rid]["slope" if item.column() == 1 else "intercept"] = value
        self._fill_offsets()              # not a number: puts the old value back

    def save_offsets(self) -> None:
        """Write the table's offsets to config.toml and apply them."""
        try:
            with self.lock:
                self.calibrator.save(self.offsets())
        except RuntimeError as exc:
            self.statusBar().showMessage(str(exc), 4000)
            return
        self.statusBar().showMessage("Offsets saved to config.toml and applied", 4000)
        self.poll()

    def reload_offsets(self) -> None:
        """Discard edits: back to the offsets in config.toml."""
        if self.calibrator.busy:
            return
        self._read_config()
        self._offsets = {i: dict(o) for i, o in self._saved.items()}
        self._fill_offsets()

    def use_true_values(self) -> None:
        """Fill the table with each sensor's true response (not saved yet)."""
        self._offsets = {r.id: {"slope": r.slope, "intercept": r.intercept} for r in self.sim.receivers}
        self._fill_offsets()

    # ─── 4. Plot ─────────────────────────────────────────────────────────────

    def _update_plot(self) -> None:
        row = max(self.receiver.currentIndex(), 0)
        r   = self.sim.receivers[row]
        o   = self._offsets[r.id]
        self.plot.set_data(r.id, (r.slope, r.intercept), (o["slope"], o["intercept"]), self._run_ticks(row))

    def _run_ticks(self, row: int) -> list:
        """``[(label, modal tick, known distance, note)]`` of the receiver in
        ``row`` at each point of the last run started here, as far as it has
        got (later points still hold an earlier run's readings).  The note says
        why a point may have read wrong: the receiver is off or can't see it."""
        r, state = self.sim.receivers[row], self.calibrator.state
        points   = self._run_points or []
        reached  = int(state.split()[1]) if state.startswith("run") else len(points)
        out = []
        for k, q in enumerate(points[:reached]):
            tick = self.calibrator.cal.get_most_frequent(r.id, k + 1)
            if tick is not None:
                known = float(np.linalg.norm(self.sim.believed[row] - np.array(q)))
                note  = ("receiver off" if r.state == "off" else
                         "no line of sight" if r.state == "blocked" or self.sim.world.blocked(q, r.position)
                         else "")
                out.append((f"P{k + 1}", float(tick), known, note))
        return out

    def showEvent(self, event) -> None:
        self.timer.start()
        self.poll()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self.timer.stop()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:
        if self.search_job is not None:
            self.search_job.wait(5000)
        super().closeEvent(event)


# ─── Small helpers ───────────────────────────────────────────────────────────

def _table(rows: int, heads: list, tip: str) -> QTableWidget:
    """A compact table of numbers under ``heads``, after a column of row
    labels: no selection highlight, editing by double-click."""
    t = QTableWidget(rows, len(heads) + 1)
    t.setHorizontalHeaderLabels([""] + heads)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
    t.verticalHeader().hide()
    t.verticalHeader().setDefaultSectionSize(ROW_PX)
    t.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    t.setFont(theme.mono_font(9))
    t.setToolTip(tip)
    return t


def _fit_rows(t: QTableWidget) -> None:
    """Make the table exactly as tall as its rows."""
    t.setFixedHeight(t.horizontalHeader().sizeHint().height() + t.rowCount() * ROW_PX + 2 * t.frameWidth())


def _row_label(text: str) -> QTableWidgetItem:
    """A row's name in the first column."""
    item = _item(text, theme.current.text_dim, editable=False)
    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    return item


def _item(text: str, c: str, editable: bool = True, tip: str = "") -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    item.setForeground(color(c))
    if not editable:
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if tip:
        item.setToolTip(tip)
    return item


def _number(text: str) -> float | None:
    """``text`` as a finite number, or None."""
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _coloured(text: str, c: str) -> str:
    """``text`` as rich text in colour ``c`` (for a QLabel)."""
    return f'<span style="color: {c}">{html.escape(text)}</span>'
