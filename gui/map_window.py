"""
Map window — a map of the whole arena at once, to see where the positioning
system works and why: how many receivers hear the direct sound, where they
time an echo instead, how good the receivers' geometry is, how far off the
ranges are, and how accurate each algorithm is (``engine.maps``).

A map is computed in a child process from a copy of the simulation, so the
simulation keeps running (a little slower) and the map takes only a little
longer than with the simulation paused (on a thread beside them it took
several times as long).  It is drawn as coloured cells with a few contour lines under the
walls, obstacles and receivers.  Hovering shows the value under the cursor.
"Show on arena" shades the main arena with it, with the same contour levels.
While a new map is computed, or after one failed or was cancelled, the
previous map stays dimmed under a note saying so.
"""

from __future__ import annotations

import math
import multiprocessing
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPainterPath
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QMainWindow, QProgressBar, QPushButton,
                             QSpinBox, QVBoxLayout, QWidget)

from engine import lab, maps
from engine.solvers import ALGORITHMS, CONTROL_MAX_SUBSETS
from gui import draw, theme
from gui.arena import viridis
from gui.charts import nice_step
from gui.theme import color
from ultragps_position import load_config


@dataclass(frozen=True)
class MapKind:
    key: str
    label: str          # in the map chooser
    title: str          # over the map
    short: str          # over the arena's colour bar
    unit: str           # over the colour bar
    counts: bool        # a whole number of receivers: one colour per count
    step: float         # default grid step, cm
    explain: str        # what the map means and how to use it


KINDS = {k.key: k for k in (
    MapKind("coverage", "Coverage", "receivers that hear the direct sound", "coverage, receivers",
            "receivers", True, 10.0,
            "How many receivers can detect the transmitter's direct sound from each point, with the "
            "current beam patterns, range, obstacles and receiver states. A fix needs 3 receivers "
            "(the dashed line is where coverage drops below 3), and CEP4 and BIAS need 4. Dark patches "
            "behind obstacles are where the solvers must manage with fewer, or with wrong readings. "
            "Use it to place and aim receivers so the area you drive in has 4 or more everywhere."),
    MapKind("nlos", "First arrival not direct (NLOS)", "receivers that do not time the direct sound",
            "not direct, receivers", "receivers", True, 10.0,
            "How many receivers would time something other than the direct sound from each point: a "
            "reflection (reads long), sound bent round an obstacle (reads a little long), the direct "
            "sound and a short echo adding up out of phase (fade), another receiver's burst, noise, "
            "or nothing (the HC-SR04 times out). These are the outliers the CEP "
            "subset search, RANSAC and the robust fit exist to reject. Zero is ideal (the dashed line "
            "encloses it). Keep calibration points in the zero area: a receiver calibrated on an "
            "echo is wrong everywhere."),
    MapKind("hdop", "Geometry (HDOP)", "HDOP: cm of fix error per cm of range error", "HDOP", "HDOP",
            False, 10.0,
            "Horizontal dilution of precision of the receivers that hear the direct sound: how far "
            "a 1 cm range error moves the fix. Below about 1.5 the geometry is good; it grows outside "
            "the receivers' box and where only receivers on one side can hear. Multiply by the range "
            "noise to get the error noise alone causes. Use it to judge receiver placement "
            "independently of noise and multipath."),
    MapKind("bias", "Range bias", "mean range error |measured − true| with no noise", "range bias, cm",
            "cm", False, 10.0,
            "How far off the ranges are with all noise switched off: range walk (a weak signal, far "
            "away or off a receiver's axis, is detected a few 25 µs cycles late) plus echoes and bent "
            "sound where the line of sight is blocked. Calibration absorbs the delay where the "
            "calibration points are, so put calibration points where this is low and typical of "
            "where you drive."),
    MapKind("accuracy", "Accuracy (Monte Carlo)", "RMS position error", "RMS error, cm", "cm", False, 20.0,
            "The RMS position error of one algorithm over many simulated pings at each point, with "
            "the current physics, faults, receiver states and calibration. Each ping is solved on "
            "every receiver within the library's bounds, with no jump filter and no history. Cells "
            "where fewer than half the pings gave a fix are marked no fix. Compute it for each "
            "algorithm to see where each one copes and where it breaks down. The library's CEP "
            "rejects any fix outside the receivers' box (its _validate_position), so for CEP the "
            "margin round the box shows as no fix; the header gives the no-fix share inside the "
            "box separately."),
)}

FUNCTIONS = {"coverage": maps.coverage, "nlos": maps.first_arrival, "hdop": maps.hdop, "bias": maps.bias}


@dataclass
class MapData:
    kind: MapKind
    xs: np.ndarray
    ys: np.ndarray
    Z: np.ndarray
    vmin: float
    vmax: float
    title: str
    short: str                          # title on the arena
    detail: str                         # how it was computed
    seconds: float
    scene: maps.Scene | None = None     # what it was computed from
    levels: list = field(default_factory=list)    # contour levels
    key_level: float | None = None      # the contour that matters most (drawn dashed)

    @property
    def cell_rect(self) -> tuple[float, float, float, float]:
        hx = (self.xs[-1] - self.xs[0]) / max(len(self.xs) - 1, 1) / 2
        hy = (self.ys[-1] - self.ys[0]) / max(len(self.ys) - 1, 1) / 2
        return self.xs[0] - hx, self.ys[0] - hy, self.xs[-1] + hx, self.ys[-1] + hy

    @property
    def half_cell(self) -> tuple[float, float]:
        """Half a cell's width and height, cm."""
        x0, y0, x1, y1 = self.cell_rect
        return (x1 - x0) / len(self.xs) / 2, (y1 - y0) / len(self.ys) / 2

    def cell(self, x: float, y: float) -> tuple[int, int] | None:
        """Grid indices ``(i, j)`` of the cell holding world point (x, y), or None outside the grid."""
        x0, y0, x1, y1 = self.cell_rect
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return None
        return int(np.argmin(np.abs(self.xs - x))), int(np.argmin(np.abs(self.ys - y)))

    def value_at(self, x: float, y: float) -> float | None:
        """The cell value at world point (x, y), or None outside the grid."""
        ij = self.cell(x, y)
        return None if ij is None else float(self.Z[ij[1], ij[0]])

    def format(self, v: float) -> str:
        if math.isnan(v):
            return "the transmitter can't be here"
        if math.isinf(v):
            return "no fix"
        if self.kind.counts:
            return f"{v:.0f} receiver" + ("" if v == 1 else "s")
        if self.kind.key == "hdop":
            return f"HDOP {v:.2f}"
        return f"{v:.1f} cm"

    @property
    def saturated(self) -> bool:
        """Do some values lie above the top of the colour scale?"""
        fin = self.Z[np.isfinite(self.Z)]
        return bool(fin.size) and float(fin.max()) > self.vmax

    def summary(self) -> str:
        """One line of statistics over the places the transmitter can be."""
        Z = self.Z[~np.isnan(self.Z)]
        if not Z.size:
            return "no point of the grid is inside the walls"

        def share(mask) -> str:
            return f"{100.0 * np.count_nonzero(mask) / Z.size:.0f} %"

        if self.kind.key == "coverage":
            return f"3+ receivers on {share(Z >= 3)} of the area, 4+ on {share(Z >= 4)}"
        if self.kind.key == "nlos":
            return f"every receiver direct on {share(Z == 0)} of the area, worst {Z.max():.0f}"
        fin = Z[np.isfinite(Z)]
        if not fin.size:
            text = "no fix anywhere"
        elif self.kind.key == "hdop":
            text = f"median {np.median(fin):.2f}, worst {fin.max():.2f}"
        else:
            text = f"median {np.median(fin):.1f} cm, 95th percentile {np.percentile(fin, 95):.1f} cm"
        if fin.size == Z.size:
            return text
        text += f", no fix on {share(np.isinf(Z))}"
        if self.kind.key == "accuracy" and self.scene is not None:
            # The library's CEP refuses fixes outside the receivers' box: say how it does inside it
            (x0, y0), (x1, y1) = np.min(self.scene.believed, axis=0), np.max(self.scene.believed, axis=0)
            X, Y = np.meshgrid(self.xs, self.ys)
            box  = ~np.isnan(self.Z) & (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
            if box.any():
                text += (f" ({100.0 * np.count_nonzero(np.isinf(self.Z) & box) / np.count_nonzero(box):.0f} % "
                         f"inside the receivers' box)")
        return text


def make_map(kind: MapKind, xs, ys, Z, n_receivers: int, title: str, detail: str, seconds: float,
             scene: maps.Scene | None = None, short: str = "") -> MapData:
    """A map with the colour range and contour levels that suit it."""
    fin = Z[np.isfinite(Z)]
    if kind.counts:
        vmin, vmax = 0.0, float(n_receivers)
        levels     = [k + 0.5 for k in range(n_receivers)]
        key        = 2.5 if kind.key == "coverage" else 0.5
    else:
        # A few bad cells would squash every other colour together: saturate
        # at three times the median, so the good area keeps its detail
        top  = min(float(np.percentile(fin, 98)), 3.0 * float(np.median(fin))) if fin.size else 1.0
        vmin = (math.floor(2 * fin.min()) / 2 if fin.size else 0.0) if kind.key == "hdop" else 0.0
        st   = nice_step(max(top - vmin, 1.0), 4)       # a flat map still gets a readable scale
        vmax = vmin + st * max(math.ceil((top - vmin) / st), 1)
        levels = [float(v) for v in np.arange(vmin + st, vmax - st / 2, st)]
        key    = None
    return MapData(kind, np.asarray(xs), np.asarray(ys), np.asarray(Z), vmin, vmax, title,
                   short or kind.short, detail, seconds, scene, levels, key)


# ─── Background computation ──────────────────────────────────────────────────

class MapJob(QThread):
    """Runs ``fn(progress=..., cancel=...)`` in a child process
    (``maps.run_job``), relaying what it sends as signals.  Where no child
    process can be started, ``fn`` runs on this thread instead, and so does
    every later job."""

    progressed = pyqtSignal(float)
    computed   = pyqtSignal(object)        # (xs, ys, Z)
    failed     = pyqtSignal(str)

    no_processes = False                   # set once a child process could not be started

    def __init__(self, fn, parent=None, process: bool = True):
        super().__init__(parent)
        self.fn        = fn
        self.cancelled = False
        self.process   = process
        self._conn     = None              # this end of the pipe to the child, while one runs
        self._lock     = threading.Lock()  # so cancel() never writes to a pipe being closed
        self._stopping = False

    def cancel(self) -> None:
        """Stop within one ping; safe to call from any thread."""
        with self._lock:
            self.cancelled = True
            self._stop_child()

    def _stop_child(self) -> None:
        """With the lock held: ask the child, if one runs, to stop (once)."""
        if self._conn is not None and not self._stopping:
            self._stopping = True
            try:
                self._conn.send("stop")
            except OSError:               # it has already gone
                pass

    def run(self) -> None:
        try:
            if not (self.process and self._in_child()):
                self.computed.emit(self.fn(progress=self.progressed.emit, cancel=lambda: self.cancelled))
        except maps.Cancelled:
            self.failed.emit("cancelled")
        except Exception as exc:      # show it rather than lose it with the thread
            self.failed.emit(f"failed: {exc}")

    def _in_child(self) -> bool:
        """Compute in a child process; false if none could be started."""
        if MapJob.no_processes:
            return False
        ctx = multiprocessing.get_context("spawn")      # a fork would copy the Qt threads' state
        here = there = None
        try:
            here, there = ctx.Pipe()
            child = ctx.Process(target=maps.run_job, args=(self.fn, there), daemon=True)
            child.start()
        except Exception:             # no process support here (a sandbox, a frozen app)
            MapJob.no_processes = True
            for end in (here, there):
                if end is not None:
                    end.close()
            return False
        there.close()                 # the child has its own: when it exits, this end reads EOF
        try:
            with self._lock:
                self._conn = here
                if self.cancelled:        # cancelled while the child was starting
                    self._stop_child()
            while True:
                try:
                    kind, value = here.recv()
                except (EOFError, OSError):   # it exited (or was killed) without a word
                    child.join(5)
                    self.failed.emit(f"failed: the map process stopped (exit code {child.exitcode})")
                    return True
                if kind == "progress":
                    self.progressed.emit(value)
                    continue
                if kind == "done":
                    self.computed.emit(value)
                elif kind == "cancelled":
                    self.failed.emit("cancelled")
                else:
                    self.failed.emit(f"failed: {value}")
                return True
        finally:
            with self._lock:
                self._conn = None
                here.close()          # tells the child to stop, if it has not
            child.join(5)
            if child.is_alive():
                child.terminate()
                child.join()


# ─── Canvas ──────────────────────────────────────────────────────────────────

class MapCanvas(QWidget):
    """The map drawn in the vector style, with a colour bar and a hover readout."""

    BAR_W    = 96                           # px kept on the right for the colour bar
    HEADER_H = 46                           # px kept at the top for the title lines

    def __init__(self, sim, parent=None):
        super().__init__(parent)
        self.sim        = sim
        self.data: MapData | None = None
        self.cal_points: list = []
        self.message    = "Choose a map and press Compute"
        self.view       = draw.View()
        self._image     = None
        self._contours  = None
        self._hover     = None              # world (x, y) under the cursor
        self.setMouseTracking(True)
        self.setMinimumSize(320, 180)

    def set_map(self, data: MapData | None) -> None:
        """Show ``data`` (None: no map, just the scene and the message)."""
        self.data = data
        if data is not None:
            self.message = ""
            pixels = viridis(data.Z[::-1], data.vmin, data.vmax)
            h, w   = pixels.shape
            self._image = QImage(np.ascontiguousarray(pixels).tobytes(), w, h, 4 * w,
                                 QImage.Format.Format_ARGB32).copy()
            self._contours = self._contour_paths(data)
        self.update()

    def set_message(self, text: str) -> None:
        """What to say in the middle of the canvas: alone while there is no
        map, else over the map, dimmed, until the next map replaces it."""
        self.message = text
        self.update()

    @staticmethod
    def _contour_paths(data: MapData) -> list:
        """``[(level, segments)]``; no-fix cells count as above every level."""
        Z = np.where(np.isinf(data.Z), data.vmax * 2 + 1, data.Z)
        out = []
        for level in data.levels:
            segs = lab.contour(data.xs, data.ys, Z, level)
            out.append((level, segs[np.all(np.isfinite(segs), axis=(1, 2))]))
        return out

    @property
    def scene(self):
        """What to draw: the scene the map was computed from, or the live simulation."""
        return self.data.scene if self.data is not None and self.data.scene is not None else self.sim

    def _bounds(self) -> tuple[float, float, float, float]:
        sc  = self.scene
        pts = [np.asarray(r.position) for r in sc.receivers] + list(np.asarray(sc.believed))
        if len(sc.world.walls) >= 3:
            pts += [np.asarray(p, dtype=float) for p in sc.world.walls]
        if self.data is not None:
            x0, y0, x1, y1 = self.data.cell_rect
            pts += [np.array([x0, y0]), np.array([x1, y1])]
        pts = np.array(pts, dtype=float)
        (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
        return x0, y0, x1, y1

    # ─── Painting ────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        pal, v = theme.current, self.view
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), color(pal.bg))
        # Fit the map below the header lines, then centre the view in that space
        v.resize(max(self.width() - self.BAR_W, 10), max(self.height() - self.HEADER_H, 10))
        v.fit(self._bounds(), margin_px=30)
        v.resize(v.width, self.height())
        v.cy += self.HEADER_H / 2 / v.scale
        draw.grid(p, v)
        if self.data is not None:
            p.save()
            if self.message:              # computing a new map, or the last one failed: this one is old
                p.setOpacity(0.4)
            self._draw_cells(p)
            p.restore()
        self._draw_scene(p)
        if self.data is None:
            draw.label(p, QPointF(v.width / 2, self.height() / 2), self.message, pal.text_dim, 10,
                       "center")
            return
        if self.message:
            self._draw_notice(p)
        else:
            self._draw_header(p)
        self._draw_colour_bar(p)
        self._draw_hover(p)

    def _draw_notice(self, p: QPainter) -> None:
        """In place of the header while the map shown is old: the message, and which map it is."""
        pal, d = theme.current, self.data
        p.fillRect(QRectF(0, 0, self.view.width, self.HEADER_H + 4), color(pal.bg, 0.9))
        draw.label(p, QPointF(12, 14), self.message, pal.accent, 9.0, bold=True)
        draw.label(p, QPointF(12, 30), f"dimmed below: the previous map, {d.title}", pal.text_dim, 8.0)
        draw.label(p, QPointF(12, 44), d.summary(), pal.text_muted, 7.5)

    def _draw_cells(self, p: QPainter) -> None:
        """The coloured cells, the no-fix cells crossed out, and the contour lines."""
        pal, v, d = theme.current, self.view, self.data
        x0, y0, x1, y1 = d.cell_rect
        target = QRectF(v.px(x0, y1), v.px(x1, y0))
        p.save()
        p.setOpacity(p.opacity() * (0.78 if pal.glow else 0.85))
        p.drawImage(target, self._image)
        p.restore()
        # No fix: cells in the "lost" colour, crossed out
        lost = np.argwhere(np.isinf(d.Z))
        if len(lost):
            cells, crosses = QPainterPath(), QPainterPath()
            for j, i in lost:
                r = self._cell_px(i, j)
                cells.addRect(r)
                crosses.moveTo(r.topLeft())
                crosses.lineTo(r.bottomRight())
            p.fillPath(cells, color(pal.status["lost"], 0.22))
            p.setPen(draw.pen(pal.status["lost"], 0.8, 0.5))
            p.drawPath(crosses)
        for level, segs in self._contours:
            path = QPainterPath()
            for (ax, ay), (bx, by) in segs:
                path.moveTo(v.px(ax, ay))
                path.lineTo(v.px(bx, by))
            if level == d.key_level:
                draw.stroke(p, path, pal.text, 1.6, 0.95, draw.DASH)
            else:
                draw.stroke(p, path, pal.bg, 0.9, 0.55, glow=False)

    def _draw_scene(self, p: QPainter) -> None:
        """Walls, obstacles, calibration points and receivers, over the cells."""
        pal, v, world = theme.current, self.view, self.scene.world
        draw.wall_outline(p, v, world.walls, pal.wall)
        for o in world.obstacles:
            draw.hatched_polygon(p, v, o.points, pal.obstacle)
        for k, (x, y) in enumerate(self.cal_points):
            q = v.px(x, y)
            draw.marker(p, q, "square", pal.text_dim, 8.0, 1.2, fill=0.25)
            draw.label(p, QPointF(q.x() + 9, q.y() - 9), f"P{k + 1}", pal.text_dim, 7.5, bold=True)
        ring = {"on": None, "blocked": pal.status["lost"], "off": pal.status["off"]}
        for r in self.scene.receivers:
            c = v.px(*r.position)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color(pal.bg, 0.75))
            p.drawEllipse(c, 13.0, 13.0)
            draw.receiver_glyph(p, v, r.position, r.facing_deg, pal.receiver, ring[r.state], r.state, 8.0,
                                f"R{r.id}")

    def _draw_header(self, p: QPainter) -> None:
        """The map's title, statistics and how it was computed."""
        pal, d = theme.current, self.data
        if d is None:
            return
        p.fillRect(QRectF(0, 0, self.view.width, self.HEADER_H + 4), color(pal.bg, 0.8))
        draw.label(p, QPointF(12, 14), d.title.upper(), pal.text, 8.5, bold=True)
        draw.label(p, QPointF(12, 30), d.summary(), pal.text_dim, 8.0)
        draw.label(p, QPointF(12, 44), d.detail, pal.text_muted, 7.5)

    def _draw_colour_bar(self, p: QPainter) -> None:
        """Colour key on the right: one block per count, or a ramp with ticks;
        the dashed contour's level marked, and the no-fix swatch if any."""
        pal, d = theme.current, self.data
        left   = self.width() - self.BAR_W + 18
        top    = 56.0
        bar    = QRectF(left, top, 12, max(self.height() - top - 70, 40))
        n      = round(d.vmax)
        draw.label(p, QPointF(left - 4, top - 14), d.kind.unit, pal.text_dim, 7.5)

        def Y(value):
            if d.kind.counts:
                return bar.top() + (n + 0.5 - value) * bar.height() / (n + 1)
            return bar.bottom() - (value - d.vmin) / max(d.vmax - d.vmin, 1e-12) * bar.height()

        if d.kind.counts:
            h = bar.height() / (n + 1)
            for k in range(n + 1):
                block = QRectF(left, Y(k) - h / 2 + 1, 12, h - 2)
                p.fillRect(block, QColor.fromRgba(int(viridis([k], d.vmin, d.vmax)[0])))
                draw.label(p, QPointF(left + 20, Y(k)), str(k), pal.text_dim, 7.5, halo=False)
        else:
            ramp = viridis(np.linspace(d.vmax, d.vmin, 128)[:, None], d.vmin, d.vmax)
            p.drawImage(bar, QImage(np.ascontiguousarray(ramp).tobytes(), 1, 128, 4,
                                    QImage.Format.Format_ARGB32).copy())
            p.setPen(draw.pen(pal.border, 1.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(bar)
            st = nice_step(d.vmax - d.vmin, 4)
            for value in np.arange(math.ceil(d.vmin / st - 1e-9) * st, d.vmax + st / 2, st):
                p.setPen(draw.pen(pal.text_dim, 1.0))
                p.drawLine(QPointF(bar.right(), Y(value)), QPointF(bar.right() + 4, Y(value)))
                more = "≥" if value + st / 2 > d.vmax and d.saturated else ""
                draw.label(p, QPointF(left + 20, Y(value)), f"{more}{value:g}", pal.text_dim, 7.5, halo=False)
        if d.key_level is not None:
            path = QPainterPath(QPointF(left - 6, Y(d.key_level)))
            path.lineTo(left + 16, Y(d.key_level))
            draw.stroke(p, path, pal.text, 1.4, dash=draw.DASH)
        if np.isinf(d.Z).any():
            swatch = QRectF(left, bar.bottom() + 16, 12, 10)
            p.fillRect(swatch, color(pal.status["lost"], 0.35))
            p.setPen(draw.pen(pal.status["lost"], 0.8, 0.8))
            p.drawLine(swatch.topLeft(), swatch.bottomRight())
            draw.label(p, QPointF(left + 20, swatch.center().y()), "no fix", pal.text_dim, 7.5, halo=False)
        p.setPen(draw.pen(pal.border, 1.0))
        p.drawLine(QPointF(self.width() - self.BAR_W, 8), QPointF(self.width() - self.BAR_W, self.height() - 8))

    def _cell_px(self, i: int, j: int) -> QRectF:
        """Cell ``(i, j)`` of the map in widget pixels."""
        d, v   = self.data, self.view
        hx, hy = d.half_cell
        return QRectF(v.px(d.xs[i] - hx, d.ys[j] + hy), v.px(d.xs[i] + hx, d.ys[j] - hy))

    def _draw_hover(self, p: QPainter) -> None:
        """Outline the cell under the cursor and say its value beside it."""
        if self._hover is None:
            return
        pal, v, d = theme.current, self.view, self.data
        ij = d.cell(*self._hover)
        if ij is None:
            return
        i, j = ij
        cell = self._cell_px(i, j)
        path = QPainterPath()
        path.addRect(cell)
        draw.stroke(p, path, pal.accent, 1.4)
        rows = [f"x {d.xs[i]:.0f}  y {d.ys[j]:.0f} cm", d.format(float(d.Z[j, i]))]
        p.setFont(theme.mono_font(8.0))
        fm  = p.fontMetrics()
        w   = max(fm.horizontalAdvance(r) for r in rows) + 14
        c   = cell.center()
        box = QRectF(c.x() + 14, c.y() + 10, w, 34)
        if box.right() > v.width - 4:
            box.moveRight(c.x() - 14)
        if box.bottom() > self.height() - 22:
            box.moveBottom(c.y() - 10)
        p.setPen(draw.pen(pal.border, 1.0))
        p.setBrush(color(pal.panel_alt, 0.94))
        p.drawRoundedRect(box, 3, 3)
        draw.label(p, QPointF(box.left() + 7, box.top() + 10), rows[0], pal.text_dim, 8.0, halo=False)
        draw.label(p, QPointF(box.left() + 7, box.top() + 24), rows[1], pal.text, 8.0, bold=True, halo=False)

    # ─── Mouse ───────────────────────────────────────────────────────────────

    def hover_text(self) -> str:
        """The readout for the point under the cursor ("" when there is none)."""
        if self._hover is None or self.data is None:
            return ""
        x, y  = self._hover
        value = self.data.value_at(x, y)
        return "" if value is None else f"({x:.0f}, {y:.0f}) cm: {self.data.format(value)}"

    def mouseMoveEvent(self, event) -> None:
        q = event.position()
        self._hover = self.view.world(q.x(), q.y()) if q.x() < self.view.width else None
        self.update()

    def leaveEvent(self, _event) -> None:
        self._hover = None
        self.update()


# ─── Window ──────────────────────────────────────────────────────────────────

class MapWindow(QMainWindow):
    """Choose a map, compute it in the background, look at it, put it on the arena."""

    showOnArena = pyqtSignal(object)      # (xs, ys, Z, title, vmin, vmax, levels), or None to clear

    def __init__(self, sim, lock=None, parent=None):
        super().__init__(parent)
        self.sim     = sim
        self.lock    = lock if lock is not None else nullcontext()
        self.job: MapJob | None = None
        self._t0     = 0.0
        self.setWindowTitle("Maps")
        self.resize(1000, 760)

        self.kind = QComboBox()
        for k in KINDS.values():
            self.kind.addItem(k.label, k.key)
        self.kind.setToolTip("Which map to compute")
        self.algorithm = QComboBox()
        for key in maps.ALGORITHMS:
            self.algorithm.addItem(f"{key}  {ALGORITHMS[key].label}", key)
        self.algorithm.setCurrentIndex(maps.ALGORITHMS.index("LM"))
        self.algorithm.setToolTip("The algorithm whose accuracy is mapped (the EKF needs a track, "
                                  "so it has no snapshot map)")
        self.trials = QSpinBox()
        self.trials.setRange(1, 200)
        self.trials.setValue(10)
        self.trials.setSuffix(" pings")
        self.trials.setToolTip("Simulated pings per grid point for the accuracy map: more is smoother "
                               "but slower (time grows in proportion)")
        self.step = QSpinBox()
        self.step.setRange(5, 50)
        self.step.setSingleStep(5)
        self.step.setSuffix(" cm")
        self.step.setToolTip("Grid spacing: halving it makes four times as many points to compute")

        self.compute_btn = QPushButton("Compute")
        self.compute_btn.setToolTip("Compute the map on a copy of the simulation as it is now")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip("Stop the computation")
        self.cancel_btn.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(True)
        self.progress.setToolTip("Progress of the computation")
        self.show_btn = QPushButton("Show on arena")
        self.show_btn.setToolTip("Shade the main window's arena with this map, with the same contour lines")
        self.show_btn.setEnabled(False)
        self.clear_btn = QPushButton("Clear arena")
        self.clear_btn.setToolTip("Remove the map shading from the arena")

        self.canvas  = MapCanvas(sim)
        self.explain = QLabel()
        self.explain.setWordWrap(True)
        self.explain.setProperty("role", "dim")
        self.explain.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        row1 = QHBoxLayout()
        for text, w in (("Map", self.kind), ("Algorithm", self.algorithm), ("Pings", self.trials),
                        ("Grid", self.step)):
            lab_ = QLabel(text)
            lab_.setProperty("role", "dim")
            row1.addWidget(lab_)
            row1.addWidget(w)
            row1.addSpacing(8)
        row1.addStretch(1)
        row2 = QHBoxLayout()
        for w in (self.compute_btn, self.cancel_btn):
            row2.addWidget(w)
        row2.addWidget(self.progress, 1)
        for w in (self.show_btn, self.clear_btn):
            row2.addWidget(w)
        box = QVBoxLayout()
        box.setContentsMargins(10, 8, 10, 6)
        box.addLayout(row1)
        box.addLayout(row2)
        box.addWidget(self.canvas, 1)
        box.addWidget(self.explain)
        central = QWidget()
        central.setLayout(box)
        self.setCentralWidget(central)

        self.kind.currentIndexChanged.connect(self._on_kind)
        self.compute_btn.clicked.connect(self.compute)
        self.cancel_btn.clicked.connect(self.cancel)
        self.show_btn.clicked.connect(self.show_on_arena)
        self.clear_btn.clicked.connect(lambda: self.showOnArena.emit(None))
        self.step.valueChanged.connect(self._idle_text)
        self._on_kind()

    # ─── Controls ────────────────────────────────────────────────────────────

    @property
    def current_kind(self) -> MapKind:
        """The map chosen in the Map box."""
        return KINDS[self.kind.currentData()]

    def set_kind(self, key: str) -> None:
        """Choose a map: coverage, nlos, hdop, bias or accuracy."""
        self.kind.setCurrentIndex(list(KINDS).index(key))

    def _on_kind(self) -> None:
        k = self.current_kind
        self.algorithm.setEnabled(k.key == "accuracy")
        self.trials.setEnabled(k.key == "accuracy")
        self.step.setValue(int(k.step))
        self.explain.setText(k.explain)
        self._idle_text()

    def _idle_text(self) -> None:
        if self.job is None:
            xs, ys = maps.grid(self.sim.believed, self.step.value())
            self.progress.setValue(0)
            self.progress.setFormat(f"{len(xs) * len(ys)} points")

    def busy(self) -> bool:
        """Is a map being computed?"""
        return self.job is not None

    def compute(self) -> None:
        """Start computing the chosen map on a copy of the simulation."""
        if self.job is not None:
            return
        k, step = self.current_kind, float(self.step.value())
        with self.lock:
            sc = maps.scene(self.sim)
        if k.key == "accuracy":
            alg, n = self.algorithm.currentData(), self.trials.value()
            fn     = partial(maps.accuracy, sc, alg, n, step)
            title  = f"{alg} {k.title}"
            short  = f"{alg} {k.short}"
            detail = f"{alg}, {n} pings per point, {step:g} cm grid"
            if alg == "CEP":                # how far the library's subset search goes (Solver settings)
                detail += (", every subset" if getattr(sc.settings, "cep_subsets", "control") == "all" else
                           f", first {CONTROL_MAX_SUBSETS} subsets as UltraGPS-Control")
        else:
            fn     = partial(FUNCTIONS[k.key], sc, step)
            title  = k.title
            short  = k.short
            detail = f"{step:g} cm grid, no noise or jitter"
        self.canvas.cal_points = _cal_points(self.sim.config_path)
        self.job = MapJob(fn, self)
        self.job.progressed.connect(self._on_progress)
        self.job.computed.connect(lambda r: self._on_computed(k, r, (title, short, detail), sc))
        self.job.failed.connect(self._on_failed)
        self.job.finished.connect(self._on_finished)
        self._t0 = time.perf_counter()
        self.compute_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setValue(0)
        self.progress.setFormat("starting")
        self.canvas.set_message("Computing...")
        self.statusBar().showMessage(f"Computing {title}...")
        self.job.start()

    def cancel(self) -> None:
        """Stop the computation within one ping (the next grid point)."""
        if self.job is not None:
            self.job.cancel()

    def show_on_arena(self) -> None:
        """Send the map shown to the main window's arena."""
        d = self.canvas.data
        if d is not None:
            self.showOnArena.emit((d.xs, d.ys, d.Z, d.short, d.vmin, d.vmax, list(d.levels)))

    def wait(self, ms: int = 60000) -> bool:
        """Block until the computation in progress (if any) ends, for scripts
        and tests; false if it is still going after ``ms``.  Its result
        arrives with the next processed events."""
        return self.job is None or self.job.wait(ms)

    # ─── Job events ──────────────────────────────────────────────────────────

    def _on_progress(self, fraction: float) -> None:
        self.progress.setValue(int(fraction * 1000))
        spent = time.perf_counter() - self._t0
        left  = spent * (1 - fraction) / fraction if fraction > 0.02 else None
        self.progress.setFormat(f"{100 * fraction:.0f} %" + (f"  about {left:.0f} s left" if left else ""))

    def _on_computed(self, kind: MapKind, result, labels: tuple, scene: maps.Scene) -> None:
        xs, ys, Z = result
        title, short, detail = labels
        seconds   = time.perf_counter() - self._t0
        data = make_map(kind, xs, ys, Z, len(scene.receivers), title, detail, seconds, scene, short)
        self.canvas.set_map(data)
        self.show_btn.setEnabled(True)
        self.statusBar().showMessage(f"{title}: {Z.size} points in {seconds:.1f} s")

    def _on_failed(self, why: str) -> None:
        self.canvas.set_message(f"Computation {why}")
        self.statusBar().showMessage(f"Computation {why}")

    def _on_finished(self) -> None:
        self.job.deleteLater()
        self.job = None
        self.compute_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._idle_text()

    def closeEvent(self, event) -> None:
        if self.job is not None:
            self.job.cancel()
            self.job.wait(5000)
        super().closeEvent(event)


def _cal_points(config_path: str) -> list:
    """The calibration points in config.toml (none if it can't be read: a
    calibration run rewrites it in place at the end, and they only decorate the map)."""
    try:
        cfg = load_config(config_path)
    except Exception:
        return []
    pts = cfg.get("cal_points", [cfg.get("cal_point_1"), cfg.get("cal_point_2")])
    return [tuple(map(float, p)) for p in pts if p is not None]
