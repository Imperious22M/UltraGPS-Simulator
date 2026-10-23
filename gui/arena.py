"""
Arena view — the main map.  It shows the room as the simulated sound sees it,
and what every part of the system made of the latest ping:

* walls and obstacles, the receivers' beam patterns and field-of-view limits;
* each receiver's measured range as a ring round where the library *believes*
  the receiver is (dotted when the library's filter dropped it);
* the path the detected sound really took to each receiver: straight, bounced
  off a wall (an echo), bent round an obstacle corner or another receiver's
  burst, and a dotted red line where the line of sight is blocked;
* the ping itself, slowed right down: the wavefront spreading from the
  transmitter (shadowed by obstacles), echoes spreading from their image
  sources, and each receiver's ring flashing when the sound it detected arrives;
* every shown algorithm's estimate, trail and 2σ ellipse against the truth.

The mouse edits the scene according to ``mode``:

  move      click: send the transmitter there (``targetClicked``)
  obstacle  drag: add a rectangular obstacle; right-click one: delete it
  receiver  drag a receiver: move where it really is; wheel over it: turn it 5°
  calpick   click: pick a calibration point (``calPointPicked``)

In every mode middle/right-drag pans, the wheel zooms, double-click fits the
room and Esc returns to ``move``.  Edits change the simulation under the
runner's lock by assigning new objects, so a step never sees half a change.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRect, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFontMetricsF, QImage, QPainter, QPainterPath, QPixmap, QRegion, QTransform
from PyQt6.QtSvg import QSvgGenerator
from PyQt6.QtWidgets import QToolTip, QWidget

from engine.acoustics import speed_of_sound
from engine.hardware import STATUS
from engine.lab import contour
from engine.solvers import ALGORITHMS
from engine.world import Obstacle
from gui import draw, theme

LAYERS = ("grid", "heatmap", "walls", "beams", "rings", "rays", "waves", "trails",
          "estimates", "ellipses", "labels", "calpoints", "legend")
MODES  = ("move", "obstacle", "receiver", "calpick")
HINTS  = {
    "move":     "click: send the transmitter there · wheel: zoom · right-drag: pan · double-click: fit",
    "obstacle": "drag: draw an obstacle · right-click an obstacle: delete it · Esc: done",
    "receiver": "drag a receiver: move it · wheel over it: turn it 5° · Esc: done",
    "calpick":  "click: pick a calibration point · Esc: cancel",
}

PICK_PX            = 14.0     # how close the mouse must be to a receiver or marker (pixels)
DRAG_PX            = 4.0      # a press that moves less than this is a click
FIT_MARGIN_PX      = 44.0
HUD_TOP_PX         = 44.0     # the time and status lines; edge arrows stay below them
HUD_BOTTOM_PX      = 64.0     # the mode hint and scale bar
COLOUR_BAR_PX      = 40.0     # ... and above them the heatmap's colour bar
LEGEND_MIN_W       = 660      # narrower, and the legend would sit on the fitted room
BEAM_CM            = 45.0     # radius of the drawn beam lobes
MAX_RING_CM        = 2000.0   # a longer "distance" is a timeout: no ring
MAX_ELLIPSE_CM     = 1000.0   # a longer 2σ ellipse is just a line across the map: not drawn
MAX_HOVER_ARRIVALS = 8        # paths shown for the receiver under the mouse
WAVE_SECONDS       = 0.5      # real time a ping's wavefront takes to cross the room
FLASH_SECONDS      = 0.3      # how long a receiver flashes when its sound arrives
MAX_PINGS          = 6        # pings animated at once when frames come quickly
MAX_SPRITES        = 300      # cached marker and receiver pictures
FAR_CM             = 5000.0   # "infinitely" far
RAY_STEP_DEG       = 0.5      # angular resolution of the wavefronts' shadows
RAY_ANGLES         = np.arange(0.0, 360.0, RAY_STEP_DEG)

_VIRIDIS = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in (
    "#440154", "#482475", "#414487", "#355F8D", "#2A788E", "#21918C",
    "#22A884", "#44BF70", "#7AD151", "#BDDF26", "#FDE725")], dtype=float)


def viridis(values, vmin: float, vmax: float) -> np.ndarray:
    """ARGB32 pixels (uint32) of ``values`` on the viridis scale; nan is transparent."""
    values = np.asarray(values, dtype=float)
    t      = np.nan_to_num(np.clip((values - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0))
    knots  = np.linspace(0.0, 1.0, len(_VIRIDIS))
    r, g, b = (np.interp(t, knots, _VIRIDIS[:, k]).astype(np.uint32) for k in range(3))
    a      = np.where(np.isfinite(values), 255, 0).astype(np.uint32)
    return (a << 24) | (r << 16) | (g << 8) | b


def _image(pixels: np.ndarray) -> QImage:
    h, w = pixels.shape
    return QImage(np.ascontiguousarray(pixels).tobytes(), w, h, 4 * w, QImage.Format.Format_ARGB32).copy()


# ─── Geometry (world cm) ─────────────────────────────────────────────────────

def _edges(points):
    pts = [np.asarray(p, dtype=float) for p in points]
    return list(zip(pts, pts[1:] + pts[:1]))


def _surfaces(world) -> list:
    """The polygons sound meets: the walls (if there are any) and every obstacle."""
    return ([world.walls] if len(world.walls) >= 3 else []) + [o.points for o in world.obstacles]


def _segments(polygons) -> tuple[np.ndarray, np.ndarray]:
    """End points ``A``, ``B`` (m × 2) of every edge of ``polygons``."""
    pairs = [e for pts in polygons for e in _edges(pts)]
    return (np.array([a for a, _ in pairs], dtype=float).reshape(-1, 2),
            np.array([b for _, b in pairs], dtype=float).reshape(-1, 2))


def _ray_distances(origin, angles_deg, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Distance along each ray from ``origin`` to each segment ``A[k]``→``B[k]``,
    inf where the ray misses it: shape (rays, segments)."""
    t      = np.radians(np.asarray(angles_deg, dtype=float))[:, None]
    dx, dy = np.cos(t), np.sin(t)
    sx, sy = (B - A)[:, 0], (B - A)[:, 1]
    ax, ay = (A - origin)[:, 0], (A - origin)[:, 1]
    den    = dx * sy - dy * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = (ax * sy - ay * sx) / den
        u    = (ax * dy - ay * dx) / den
    hit = (np.abs(den) > 1e-12) & (dist > 1e-6) & (u >= -1e-9) & (u <= 1 + 1e-9)
    return np.where(hit, dist, np.inf)


def _ray_end(origin, angle_deg: float, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Where a ray from ``origin`` first meets a segment (or ``FAR_CM`` away)."""
    d = float(np.min(_ray_distances(origin, [angle_deg], A, B), initial=FAR_CM))
    t = math.radians(angle_deg)
    return np.asarray(origin, dtype=float) + d * np.array([math.cos(t), math.sin(t)])


def _segment_through(point, A: np.ndarray, B: np.ndarray) -> int | None:
    """Index of the segment ``point`` (a bounce point) lies on, or None."""
    if not len(A):
        return None
    q    = np.asarray(point, dtype=float)
    s    = B - A
    t    = np.clip(np.sum((q - A) * s, axis=1) / np.maximum(np.sum(s * s, axis=1), 1e-12), 0.0, 1.0)
    dist = np.hypot(*(A + t[:, None] * s - q).T)
    k    = int(np.argmin(dist))
    return k if dist[k] < 0.05 else None


def _mirror(p, a, b) -> np.ndarray:
    """``p`` reflected in the line through ``a`` and ``b`` (an image source)."""
    d = (b - a) / np.hypot(*(b - a))
    return 2.0 * (a + d * float((p - a) @ d)) - p


def _sigma_cm(cov) -> float:
    """Standard deviation (cm) along the worst direction of a 2×2 covariance."""
    cov = np.asarray(cov, dtype=float)
    if not np.all(np.isfinite(cov)):
        return math.inf
    return float(np.sqrt(max(np.linalg.eigvalsh((cov + cov.T) / 2).max(), 0.0)))


def _path_key(arrival) -> str:
    """Colour key of a sound path: ``ray`` (direct) or a receiver status."""
    if arrival.source != "TX":
        return "crosstalk"
    return {"reflect": "echo", "diffract": "bent"}.get(arrival.path.kind, "ray")


def _key_colour(key: str) -> str:
    pal = theme.current
    return {"ray": pal.ray, "wave": pal.wave}.get(key) or pal.status[key]


# ─── Animation and heatmap state ─────────────────────────────────────────────

@dataclass
class _Front:
    """A wavefront: a circle round ``centre``, seen along ray angle
    ``RAY_ANGLES[k]`` only while ``lo[k]`` ≤ radius ≤ ``hi[k]``, i.e. inside the
    room and out of obstacles' shadows."""
    centre: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    key: str                # colour key


@dataclass
class _Ping:
    """One ping's animation, worked out when its frame arrives (world cm)."""
    start: float            # time.perf_counter() when it started
    reach: float            # cm of travel the animation covers
    fronts: list            # [_Front]: the direct one, then each detected echo's
    packets: list           # [(points (k, 2), cm of travel at each point, colour key)] per detected path
    arrive: dict            # {receiver: cm of travel when its detected sound gets there}


@dataclass
class _Heatmap:
    image: QImage
    rect: tuple             # x0, y0, x1, y1 of the image's outer edges, world cm
    contours: QPainterPath  # world cm
    levels: np.ndarray
    bar: QImage             # 256 × 1 colour bar
    title: str
    vmin: float
    vmax: float


class ArenaView(QWidget):
    """The arena map (see the module docstring); fed frames by ``set_frame``."""

    targetClicked   = pyqtSignal(float, float)
    calPointPicked  = pyqtSignal(float, float)
    worldEdited     = pyqtSignal()
    receiversEdited = pyqtSignal()
    cursorMoved     = pyqtSignal(float, float)
    modeChanged     = pyqtSignal(str)

    wave_seconds = WAVE_SECONDS

    def __init__(self, sim, lock=None, parent=None):
        super().__init__(parent)
        self.sim         = sim
        self.lock        = lock if lock is not None else contextlib.nullcontext()
        self.view        = draw.View()
        self.frame       = sim.last
        self.shown       = set(ALGORITHMS)
        self.layers      = {name: name != "ellipses" for name in LAYERS}
        self.mode        = "move"
        self.hud_text    = ""
        self.cal_points  = []
        self.cal_pending = False
        self._heatmap: _Heatmap | None = None
        self._heatmap_version = 0
        self._pings: list[_Ping] = []
        self._timer      = QTimer(self)      # ~60 fps, only while a ping is animating
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._animate)
        self._auto_fit   = True
        self._version    = 0             # bumped by every change the still picture shows
        self._scenery    = self._scenery_key = None     # cached pixmaps and what they show
        self._still      = self._still_key = None
        self._legend_box = None          # where the legend was last drawn
        self._sprites    = {}            # {look: QPixmap}, see _stamp
        self._live       = True          # painting the screen (not an export)
        self._press      = None          # (button, press position, last position)
        self._moved      = False         # the pressed mouse has moved beyond DRAG_PX: not a click
        self._band       = None          # obstacle being drawn: (corner, corner) in cm
        self._drag       = None          # (receiver, offset from the mouse in cm)
        self._hover_rx   = None
        self._hover_obstacle = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setMinimumSize(240, 180)
        self._set_cursor()

    # ─── Public API ──────────────────────────────────────────────────────────

    def set_frame(self, frame) -> None:
        """Show ``frame`` (a ``Frame`` from the simulation) and animate its ping."""
        self.frame = frame
        if (frame is not None and frame.true is not None and frame.readings
                and self.layers["waves"] and self.isVisible()):
            self._pings = (self._pings + [self._make_ping(frame)])[-MAX_PINGS:]
            if not self._timer.isActive():
                self._timer.start()
        self._changed()

    def set_algorithms(self, keys) -> None:
        """Draw only these algorithms' estimates, trails, ellipses and legend rows."""
        self.shown = set(keys)
        self._changed()

    def set_layer(self, name: str, on: bool) -> None:
        """Show or hide one of ``LAYERS``."""
        if name not in self.layers:
            raise KeyError(f"no layer {name!r}; layers are {', '.join(LAYERS)}")
        self.layers[name] = bool(on)
        if name == "waves" and not on:
            self._pings.clear()
            self._timer.stop()
        self._changed()

    def set_mode(self, mode: str) -> None:
        """What the mouse does: one of ``MODES`` (see the module docstring)."""
        if mode not in MODES:
            raise ValueError(f"no mode {mode!r}; modes are {', '.join(MODES)}")
        self._end_press()
        if mode != self.mode:
            self.mode = mode
            self._set_cursor()
            self.modeChanged.emit(mode)
        self._changed()

    def set_hud(self, text: str) -> None:
        """Status text shown top right under the time, e.g. "PAUSED"."""
        self.hud_text = text
        self._changed()

    def set_cal_points(self, points, pending: bool = False) -> None:
        """Calibration points to mark P1..Pn; ``pending`` while still being chosen."""
        self.cal_points  = [(float(x), float(y)) for x, y in points]
        self.cal_pending = pending
        self._changed()

    def set_heatmap(self, xs, ys, Z, title: str = "", vmin: float | None = None,
                    vmax: float | None = None) -> None:
        """Shade the arena with ``Z[j, i]`` at ``(xs[i], ys[j])`` (both ascending)."""
        xs, ys, Z = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), np.asarray(Z, dtype=float)
        finite    = Z[np.isfinite(Z)]
        if vmin is None:
            vmin = float(finite.min()) if finite.size else 0.0
        if vmax is None:
            vmax = float(finite.max()) if finite.size else 1.0
        vmax      = max(vmax, vmin + 1e-9)
        hx        = (xs[-1] - xs[0]) / max(len(xs) - 1, 1) / 2
        hy        = (ys[-1] - ys[0]) / max(len(ys) - 1, 1) / 2
        levels    = np.linspace(vmin, vmax, 6)[1:-1]
        lines     = QPainterPath()
        for level in levels:
            for (ax, ay), (bx, by) in contour(xs, ys, Z, level):
                lines.moveTo(ax, ay)
                lines.lineTo(bx, by)
        self._heatmap = _Heatmap(_image(viridis(Z[::-1], vmin, vmax)),
                                 (xs[0] - hx, ys[0] - hy, xs[-1] + hx, ys[-1] + hy), lines, levels,
                                 _image(viridis(np.linspace(vmin, vmax, 256)[None, :], vmin, vmax)),
                                 title, vmin, vmax)
        self._heatmap_version += 1
        self._changed()

    def clear_heatmap(self) -> None:
        """Remove the heatmap and its colour bar."""
        self._heatmap = None
        self._heatmap_version += 1
        self._changed()

    def fit(self) -> None:
        """Zoom to the whole room, and keep it fitted as the widget resizes."""
        self.view.resize(self.width(), self.height())
        self.view.fit(self._bounds(), FIT_MARGIN_PX)
        self._auto_fit = True
        self._changed()

    def render_to(self, device) -> None:
        """Paint the arena onto any paint device (QImage, QPixmap, ...) at its
        size, showing the same part of the room as the widget does."""
        w, h = device.width(), device.height()
        with self._view_for(w, h):
            p = QPainter(device)
            try:
                self.render_scene(p, w, h)
            finally:
                p.end()

    def export_svg(self, path: str) -> None:
        """Save the arena as it is drawn now (without the ping animation) as SVG."""
        w, h = self.width(), self.height()
        gen  = QSvgGenerator(QSvgGenerator.SvgVersion.Svg11)    # SVG Tiny drops the hatching's clip
        gen.setFileName(path)
        gen.setSize(QSize(w, h))
        gen.setViewBox(QRect(0, 0, w, h))
        gen.setResolution(self.logicalDpiX())                   # text the size it is on screen
        gen.setTitle("UltraGPS arena")
        with self._view_for(w, h):
            p = QPainter(gen)
            try:
                self.render_scene(p, w, h, live=False)
            finally:
                p.end()

    @contextlib.contextmanager
    def _view_for(self, w: int, h: int):
        """Size the view to ``w`` × ``h`` for the duration, showing what the
        widget shows (or the whole room, if it is fitted or never shown)."""
        v     = self.view
        saved = (v.scale, v.cx, v.cy, v.width, v.height)
        if self._auto_fit or v.width <= 1:
            v.resize(w, h)
            v.fit(self._bounds(), FIT_MARGIN_PX)
        else:
            v.scale *= min(w / v.width, h / v.height)
            v.resize(w, h)
        try:
            yield
        finally:
            v.scale, v.cx, v.cy, v.width, v.height = saved

    def tooltip_at(self, x: float, y: float) -> str:
        """What the hover tooltip says at widget pixel (x, y): details of the
        receiver or estimate under the mouse, or "" for nothing."""
        i = self.receiver_at(x, y)
        if i is not None:
            return self._receiver_tip(i)
        return "\n\n".join(self._estimate_tip(a) for a in self._estimates_at(x, y))

    def receiver_at(self, x: float, y: float) -> int | None:
        """The receiver whose true position is within ``PICK_PX`` of pixel (x, y)."""
        best = None
        for i, r in enumerate(self.sim.receivers):
            q = self.view.px(*r.position)
            d = math.hypot(q.x() - x, q.y() - y)
            if d <= PICK_PX and (best is None or d < best[0]):
                best = (d, i)
        return None if best is None else best[1]

    def obstacle_at(self, x: float, y: float) -> int | None:
        """Index of the (topmost) obstacle under pixel (x, y)."""
        p = self.view.world(x, y)
        for k in reversed(range(len(self.sim.world.obstacles))):
            if self.sim.world.obstacles[k].contains(p):
                return k
        return None

    # ─── Painting ────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        try:
            self.render_scene(p, self.width(), self.height())
        finally:
            p.end()

    def render_scene(self, p: QPainter, w: int, h: int, live: bool = True) -> None:
        """Paint the arena into ``p`` (``w`` × ``h`` pixels).  On screen
        (``live``) the still picture comes from a cached pixmap and only the
        ping animation is drawn afresh, 60 times a second.  An export draws
        every line as a vector, without the animation, hover highlights or the
        obstacle being drawn."""
        self._live = live
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not live:
            self._draw_still(p, w, h)
            return
        p.drawPixmap(0, 0, self._still_pixmap(w, h))
        if self._pings:
            now = time.perf_counter()
            if self._legend_box is not None:        # the animation passes behind the legend
                p.setClipRegion(QRegion(0, 0, w, h).subtracted(QRegion(self._legend_box.toAlignedRect())))
            self._draw_flashes(p, self._flashes(now))
            self._draw_waves(p, now)

    def _draw_still(self, p: QPainter, w: int, h: int) -> None:
        """Everything but the ping animation, bottom layer first."""
        frame, layers = self.frame, self.layers
        if self._live:
            p.drawPixmap(0, 0, self._scenery_pixmap(w, h))
        else:
            self._draw_scenery(p, w, h)
        if frame is not None:
            if layers["beams"] and frame.true is not None:
                self._draw_tx_beam(p, frame)
            if layers["rings"]:
                self._draw_rings(p, frame)
            if layers["rays"] and frame.true is not None:
                self._draw_rays(p, frame)
            if layers["trails"]:
                self._draw_trails(p)
            if layers["estimates"] or layers["ellipses"]:
                self._draw_estimates(p, frame)
            self._draw_truth(p, frame)
        self._draw_receivers(p, frame)
        if layers["calpoints"]:
            self._draw_cal_points(p)
        if self._live:
            self._draw_editing(p)
        self._draw_hud(p, frame, w, h)

    def _changed(self) -> None:
        """Repaint, redrawing the still picture: something it shows has changed."""
        self._version += 1
        self.update()

    def _pixmap(self, w: int, h: int, paint) -> QPixmap:
        """A ``w`` × ``h`` pixmap at the screen's pixel ratio, painted by ``paint(painter, w, h)``."""
        dpr = self.devicePixelRatioF()
        pm  = QPixmap(max(1, round(w * dpr)), max(1, round(h * dpr)))
        pm.setDevicePixelRatio(dpr)
        q = QPainter(pm)
        q.setRenderHint(QPainter.RenderHint.Antialiasing)
        paint(q, w, h)
        q.end()
        return pm

    def _still_pixmap(self, w: int, h: int) -> QPixmap:
        """The still picture, redrawn when the scenery, the frame, anything set
        through the API, the mouse's highlights or the transmitter's target change."""
        key = (self._version, self._scenery_state(w, h), self._target())
        if key != self._still_key:
            self._still, self._still_key = self._pixmap(w, h, self._draw_still), key
        return self._still

    def _scenery_pixmap(self, w: int, h: int) -> QPixmap:
        """Everything that changes only when the view, the room or the receivers
        do (background, grid, heatmap, walls, obstacles, receiver beams)."""
        key = self._scenery_state(w, h)
        if key != self._scenery_key:
            self._scenery, self._scenery_key = self._pixmap(w, h, self._draw_scenery), key
        return self._scenery

    def _scenery_state(self, w: int, h: int) -> tuple:
        """All the scenery depends on, including what the panels change in the sim."""
        v, world, env = self.view, self.sim.world, self.sim.env
        return (w, h, self.devicePixelRatioF(), v.scale, v.cx, v.cy, theme.current.name, self._heatmap_version,
                tuple(self.layers[k] for k in ("grid", "heatmap", "walls", "beams", "labels")),
                tuple(map(tuple, world.walls)),
                tuple((tuple(map(tuple, o.points)), o.name) for o in world.obstacles),
                tuple((tuple(r.position), r.facing_deg, r.state) for r in self.sim.receivers),
                (env.beam_patterns, env.rx_beam_deg, env.fov_cutoff, env.fov_half_deg))

    def _stamp(self, p: QPainter, at: QPointF, radius: int, look, paint) -> None:
        """Draw what ``paint(painter)`` draws round pixel ``at``.  On screen it
        comes from a sprite cached under ``look``, so the glowing strokes of a
        marker or receiver are drawn once rather than at 60 frames a second."""
        if not self._live:
            paint(p)
            return
        ax, ay = round(at.x()), round(at.y())
        dpr    = self.devicePixelRatioF()
        look   = (look, theme.current.name, dpr)
        sprite = self._sprites.get(look)
        if sprite is None:
            sprite = QPixmap(round(2 * radius * dpr), round(2 * radius * dpr))
            sprite.setDevicePixelRatio(dpr)
            sprite.fill(Qt.GlobalColor.transparent)
            q = QPainter(sprite)
            q.setRenderHint(QPainter.RenderHint.Antialiasing)
            q.translate(radius - ax, radius - ay)
            paint(q)
            q.end()
            if len(self._sprites) >= MAX_SPRITES:
                self._sprites.clear()
            self._sprites[look] = sprite
        p.drawPixmap(ax - radius, ay - radius, sprite)

    def _marker(self, p: QPainter, at: QPointF, shape: str, c: str, size: float = 11.0,
                width: float = 1.4, alpha: float = 1.0, fill: float = 0.0) -> None:
        """``draw.marker`` through the sprite cache."""
        self._stamp(p, at, int(size) + 8, ("marker", shape, c, size, width, alpha, fill),
                    lambda q: draw.marker(q, at, shape, c, size, width, alpha, fill))

    def _draw_scenery(self, p: QPainter, w: int, h: int) -> None:
        pal, v, world = theme.current, self.view, self.sim.world
        p.fillRect(QRectF(0, 0, w, h), theme.color(pal.bg))
        if self.layers["grid"]:
            draw.grid(p, v)
        hm = self._heatmap
        if hm is not None and self.layers["heatmap"]:
            x0, y0, x1, y1 = hm.rect
            p.save()
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.setOpacity(0.55)
            p.drawImage(QRectF(v.px(x0, y1), v.px(x1, y0)), hm.image)
            p.setOpacity(1.0)
            p.setTransform(self._world_transform(), True)
            draw.stroke(p, hm.contours, pal.text_dim, 0.8, glow=False)
            p.restore()
        if self.layers["walls"]:
            draw.wall_outline(p, v, world.walls, pal.wall)
            for o in world.obstacles:
                draw.hatched_polygon(p, v, o.points, pal.obstacle)
                if o.name and self.layers["labels"]:
                    pts = np.asarray(o.points, dtype=float)
                    top = v.px(pts[:, 0].mean(), pts[:, 1].max())
                    draw.label(p, QPointF(top.x(), top.y() - 9), o.name, pal.obstacle, 7.5, "center")
        if self.layers["beams"]:
            self._draw_receiver_beams(p)

    def _world_transform(self) -> QTransform:
        v = self.view
        return QTransform(v.scale, 0, 0, -v.scale, v.width / 2 - v.cx * v.scale, v.height / 2 + v.cy * v.scale)

    def _draw_receiver_beams(self, p: QPainter) -> None:
        """Receiver beam lobes and the dashed field-of-view limits out to the walls."""
        pal, v, env = theme.current, self.view, self.sim.env
        walls = self.sim.world.walls
        A, B  = _segments([walls] if len(walls) >= 3 else [])
        for r in self.sim.receivers:
            if r.state == "off":
                continue
            if env.beam_patterns:
                draw.beam_lobe(p, v, r.position, r.facing_deg, env.rx_beam_deg, BEAM_CM, pal.receiver, 0.2)
            if env.fov_cutoff:
                for side in (-1, 1):
                    end = _ray_end(r.position, r.facing_deg + side * env.fov_half_deg, A, B)
                    draw.polyline(p, v, [r.position, end], pal.status["angle"], 1.0, 0.5, draw.DASH, glow=False)

    def _draw_tx_beam(self, p: QPainter, frame) -> None:
        facing = self.sim.hardware.tx_facing(frame.heading)
        if facing is not None:
            draw.beam_lobe(p, self.view, frame.true, facing, self.sim.env.tx_beam_deg, BEAM_CM,
                           theme.current.truth, 0.3)

    def _draw_rings(self, p: QPainter, frame) -> None:
        """Each receiver's measured range, round where the library believes it
        is: dotted in a warning colour when the library's filter dropped it."""
        pal = theme.current
        for i in range(len(frame.solution.distances)):
            ring = self._ring(frame, i)
            if ring is None:
                continue
            centre, radius, kept = ring
            if kept:
                draw.circle(p, self.view, centre, radius, pal.receiver, 1.0, 0.3, glow=False)
            else:
                draw.circle(p, self.view, centre, radius, pal.status["near"], 1.0, 0.7, draw.DOT, glow=False)

    def _ring(self, frame, i: int):
        """``(centre, radius, kept by the filter)`` of receiver ``i``'s range
        ring, or None when it is off or its distance is no range (≤ 0 or absurd)."""
        d = frame.solution.distances[i]
        if i >= len(self.sim.receivers) or self.sim.receivers[i].state == "off" or not 0 < d <= MAX_RING_CM:
            return None
        return self.sim.believed[i], d, i in frame.solution.sane

    def _draw_flashes(self, p: QPainter, flashes: dict) -> None:
        """Receivers the ping's sound is reaching now: the range ring brightens
        and a halo in the status colour spreads round the receiver."""
        frame = self.frame
        if frame is None:
            return
        for i, f in flashes.items():
            ring = self._ring(frame, i) if self.layers["rings"] else None
            if ring is not None and ring[2]:
                self._draw_circle(p, ring[0], ring[1], theme.current.receiver, f)
            colour = self._status_colour(frame, i)
            if colour is not None:
                self._draw_circle(p, self.sim.receivers[i].position, (15.0 + 12.0 * (1.0 - f)) / self.view.scale,
                                  colour, f)

    def _draw_rays(self, p: QPainter, frame) -> None:
        """The path each receiver's detected sound took; for the hovered
        receiver, every path that reached it, detectable or not."""
        pal, v = theme.current, self.view
        hover  = self._hover_rx if self._live else None
        for i, rd in enumerate(frame.readings):
            rx = self.sim.receivers[i]
            if rx.state == "off":
                continue
            if not rd.los:
                draw.polyline(p, v, [frame.true, rx.position], pal.status["lost"], 1.0, 0.5,
                              draw.DOT, glow=False)
            if i == hover:
                self._draw_arrivals(p, rd)
            a = rd.detected
            if a is None:
                continue
            key = _path_key(a)
            c   = _key_colour(key)
            if key == "ray":
                draw.polyline(p, v, a.path.points, c, 1.0, 0.3, glow=False)
                continue
            draw.polyline(p, v, a.path.points, c, 1.3, 0.85)
            for q in a.path.points[1:-1]:
                shape = "diamond" if a.path.kind == "reflect" else "circle"
                self._marker(p, v.px(*q), shape, c, 7.0, 1.2, fill=0.6)

    def _draw_arrivals(self, p: QPainter, rd) -> None:
        """Every path that reached one receiver, labelled with its strength over
        the detection threshold: dashed if it could be detected, dotted if too weak."""
        v = self.view
        for a in sorted(rd.arrivals, key=lambda a: -a.ratio_db)[:MAX_HOVER_ARRIVALS]:
            c   = _key_colour(_path_key(a))
            pts = np.asarray(a.path.points, dtype=float)
            if a is not rd.detected:
                draw.polyline(p, v, pts, c, 1.0, 0.75, draw.DASH if a.cycle is not None else draw.DOT, glow=False)
            at = pts[-2] if len(pts) > 2 else pts.mean(axis=0)
            draw.label(p, v.px(*at) + QPointF(0, -10), f"{a.ratio_db:+.0f} dB", c, 7.5, "center")

    def _draw_waves(self, p: QPainter, now: float) -> None:
        """Each ping ``s`` cm into its travel: the wavefronts that can be seen,
        fading as they spread, and a short streak running along each detected path."""
        v = self.view
        for ping in self._pings:
            f = (now - ping.start) / self.wave_seconds
            if not 0.0 <= f <= 1.0:
                continue
            s    = f * ping.reach
            fade = (1.0 - f) ** 1.3
            for front in ping.fronts:
                seen = (front.lo <= s) & (s <= front.hi)
                if s > 0 and seen.any():
                    self._draw_circle(p, front.centre, s, _key_colour(front.key), 0.9 * fade, seen)
            back = s - 0.12 * ping.reach
            for pts, cum, key in ping.packets:
                if s >= cum[-1]:
                    continue
                along = [max(back, cum[0])] + [L for L in cum if back < L < s] + [s]
                xy    = np.column_stack([np.interp(along, cum, pts[:, 0]), np.interp(along, cum, pts[:, 1])])
                c     = _key_colour(key)
                draw.polyline(p, v, xy, c, 2.0, glow=False)
                self._marker(p, v.px(*xy[-1]), "circle", c, 5.0, 1.2, fill=1.0)

    def _draw_circle(self, p: QPainter, centre, radius_cm: float, c: str, alpha: float,
                     seen: np.ndarray | None = None, glow: bool = True) -> None:
        """A large circle, or only its arcs where ``seen`` (per ray of
        ``RAY_ANGLES``), as 1 px lines with fainter ones either side for the
        glow.  ``draw.stroke`` would glow with translucent pens wider than 1 px,
        which Qt draws about ten times slower on a path this long.  A faint
        line gets no glow: it would not show, and costs as much as a bright one's."""
        r0    = radius_cm * self.view.scale
        c0    = self.view.px(*centre)
        runs  = []
        if seen is not None and not seen.all():
            edges = np.flatnonzero(np.diff(np.r_[0, seen.astype(np.int8), 0]))
            runs  = list(zip(edges[::2], edges[1::2]))
        glow  = glow and theme.current.glow and alpha > 0.4
        bands = [(0.0, 1.0)] + ([(1.2, 0.35), (2.4, 0.12)] if glow else [])
        p.setBrush(Qt.BrushStyle.NoBrush)
        for dr, a in bands:
            path = QPainterPath()
            for r in {r0 - dr, r0 + dr}:
                if r <= 0:
                    continue
                rect = QRectF(c0.x() - r, c0.y() - r, 2 * r, 2 * r)
                if not runs:
                    path.addEllipse(rect)
                for first, end in runs:
                    start = (first - 0.5) * RAY_STEP_DEG
                    path.arcMoveTo(rect, start)
                    path.arcTo(rect, start, (end - first) * RAY_STEP_DEG)
            p.setPen(draw.pen(c, 1.0, alpha * a))
            p.drawPath(path)

    def _draw_trails(self, p: QPainter) -> None:
        for a in theme.ALGO_ORDER:
            if a in self.shown:
                pts = list(self.sim.trails[a])      # a copy: the runner appends to it
                if len(pts) > 1:
                    draw.polyline(p, self.view, pts, theme.current.algo[a], 1.0, 0.3, glow=False)

    def _draw_estimates(self, p: QPainter, frame) -> None:
        """Each shown algorithm's fix (hollow and dim when it has none this
        frame), its 2σ ellipse, and an arrow at the edge for a fix off screen."""
        pal = theme.current
        for a, e in self._shown_estimates(frame):
            c = pal.algo[a]
            if self.layers["ellipses"] and e.cov is not None and _sigma_cm(e.cov) * 2 < MAX_ELLIPSE_CM:
                draw.ellipse_cov(p, self.view, e.pos, e.cov, c, 2.0, 1.0, 0.5 if e.ok else 0.2)
            if not self.layers["estimates"]:
                continue
            q, tip = self._estimate_spot(e.pos)
            if tip is None:
                self._marker(p, q, theme.ALGO_SHAPES[a], c, 11.0, 1.4, 1.0 if e.ok else 0.4, 0.25 if e.ok else 0.0)
            else:
                draw.arrow(p, q + (tip - q) * 0.4, tip, c, 1.4, 6.0, 0.8)
                self._marker(p, q, theme.ALGO_SHAPES[a], c, 9.0, 1.2, 0.8)

    def _shown_estimates(self, frame):
        """(algorithm, Estimate) of each shown algorithm with a fix to draw.
        A solver that diverged to nan or inf is left out: it has no place on the map."""
        for a in theme.ALGO_ORDER:
            e = frame.estimates.get(a)
            if a in self.shown and e is not None and e.pos is not None and np.all(np.isfinite(e.pos)):
                yield a, e

    def _estimate_spot(self, pos) -> tuple[QPointF, QPointF | None]:
        """Where the marker of an estimate at ``pos`` goes, and for one off
        screen the tip of the arrow at the edge that points to it (else None)."""
        v      = self.view
        bottom = HUD_BOTTOM_PX + (COLOUR_BAR_PX if self._heatmap is not None and self.layers["heatmap"] else 0)
        inner  = QRectF(16, HUD_TOP_PX, max(v.width - 32, 1), max(v.height - HUD_TOP_PX - bottom, 1))
        q      = v.px(*pos)
        if inner.contains(q):
            return q, None
        tip = QPointF(min(max(q.x(), inner.left()), inner.right()), min(max(q.y(), inner.top()), inner.bottom()))
        d   = q - tip
        return tip - d * (24.0 / max(math.hypot(d.x(), d.y()), 1e-9)), tip

    def _draw_truth(self, p: QPainter, frame) -> None:
        """The true transmitter, its heading and where it is driving to."""
        if frame.true is None:
            return
        pal, v = theme.current, self.view
        target = self._target()
        if target is not None and np.hypot(*(np.asarray(target) - frame.true)) > 1.0:
            draw.polyline(p, v, [frame.true, target], pal.truth, 1.0, 0.4, draw.DASH, glow=False)
            path = QPainterPath()
            path.addEllipse(v.px(*target), 7.0, 7.0)
            draw.stroke(p, path, pal.truth, 1.0, 0.6, draw.DASH)
        c = v.px(*frame.true)
        t = math.radians(frame.heading)
        tick = QPainterPath(QPointF(c.x() + 11 * math.cos(t), c.y() - 11 * math.sin(t)))
        tick.lineTo(c.x() + 19 * math.cos(t), c.y() - 19 * math.sin(t))
        draw.stroke(p, tick, pal.truth, 1.4)
        self._stamp(p, c, 16, "truth", lambda q: draw.crosshair(q, c, pal.truth, 10.0))

    def _target(self) -> tuple | None:
        """Where the transmitter is driving to, unless it is only the moving
        point a circle or figure 8 chases (not worth drawing)."""
        m = self.sim.motion
        return None if m.mode in ("circle", "figure8") else tuple(map(float, m.target))

    def _draw_receivers(self, p: QPainter, frame) -> None:
        """Each receiver where it really is, ringed in its status colour (thick
        when the library's CEP used it), with a ghost where the library thinks it is."""
        pal, v = theme.current, self.view
        cep    = frame.estimates.get("CEP") if frame is not None else None
        used   = {int(i) for i in cep.used} if cep is not None else set()
        for i, r in enumerate(self.sim.receivers):
            believed = self.sim.believed[i]
            if np.hypot(*(believed - r.position)) > 0.5:
                draw.polyline(p, v, [believed, r.position], pal.text_muted, 1.0, 0.8, draw.DOT, glow=False)
                self._marker(p, v.px(*believed), "circle", pal.text_muted, 9.0, 1.0)
                self._marker(p, v.px(*believed), "plus", pal.text_muted, 5.0, 1.0)
            ring = self._status_colour(frame, i)
            c    = v.px(*r.position)
            if self._live and i == self._hover_rx and self.mode == "receiver":
                halo = QPainterPath()
                halo.addEllipse(c, 21.0, 21.0)
                draw.stroke(p, halo, pal.accent, 1.0, 0.8, draw.DASH)
            name = f"R{r.id}" if self.layers["labels"] else ""
            look = ("receiver", round(r.facing_deg, 1), ring, r.state, i in used, name)
            self._stamp(p, c, 40, look, lambda q: draw.receiver_glyph(
                q, v, r.position, r.facing_deg, pal.receiver, ring, r.state, 9.0, name, i in used))

    def _status_colour(self, frame, i: int) -> str | None:
        """Colour of receiver ``i``'s status (None before the first frame)."""
        pal, r = theme.current, self.sim.receivers[i]
        status = frame.status if frame is not None else []
        st     = "off" if r.state == "off" else status[i] if i < len(status) else None
        return pal.status.get(st, pal.text_dim) if st else None

    def _draw_cal_points(self, p: QPainter) -> None:
        pal = theme.current
        c   = pal.status["echo"] if self.cal_pending or self.mode == "calpick" else pal.text_dim
        for k, (x, y) in enumerate(self.cal_points):
            q = self.view.px(x, y)
            self._marker(p, q, "square", c, 9.0, 1.3, fill=0.2)
            draw.label(p, QPointF(q.x() + 10, q.y() - 10), f"P{k + 1}", c, 8.0, bold=True)

    def _draw_editing(self, p: QPainter) -> None:
        """The obstacle being drawn, or the one a right-click would delete."""
        pal, v = theme.current, self.view
        if self._band is not None:
            (x0, y0), (x1, y1) = self._band
            pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            draw.polyline(p, v, pts, pal.obstacle, 1.2, 0.9, draw.DASH, closed=True)
            draw.label(p, v.px(max(x0, x1), min(y0, y1)) + QPointF(6, 10),
                       f"{abs(x1 - x0):.0f} × {abs(y1 - y0):.0f} cm", pal.obstacle, 8.0)
        k = self._hover_obstacle
        if self.mode == "obstacle" and self._band is None and k is not None and k < len(self.sim.world.obstacles):
            pts = self.sim.world.obstacles[k].points
            draw.polyline(p, v, pts, pal.status["lost"], 1.6, 0.9, draw.DASH, closed=True)
            pts = np.asarray(pts, dtype=float)
            at  = v.px(pts[:, 0].mean(), pts[:, 1].min())
            draw.label(p, at + QPointF(0, 12), "right-click: delete", pal.status["lost"], 7.5, "center")

    # ─── HUD ─────────────────────────────────────────────────────────────────

    def _draw_hud(self, p: QPainter, frame, w: int, h: int) -> None:
        """Legend top left (when there is room), time and status top right,
        mode hint bottom left, scale (and the heatmap's colour bar) bottom right."""
        pal  = theme.current
        self._legend_box = self._draw_legend(p, frame, w, h) if self.layers["legend"] else None
        free = self._legend_box.right() + 16 if self._legend_box is not None else 12.0
        y    = 16.0
        if frame is not None:
            draw.label(p, QPointF(w - 12, y), f"t {frame.time:.2f} s   frame {frame.index}",
                       pal.text_dim, 8.5, "right")
            y += 17
            if any(r.status == "radio" for r in frame.readings):
                draw.label(p, QPointF(w - 12, y), "transmitter missed its radio trigger",
                           pal.status["radio"], 8.0, "right")
                y += 17
        if self.hud_text:
            text = QFontMetricsF(theme.mono_font(9.0, True)).elidedText(
                self.hud_text, Qt.TextElideMode.ElideRight, w - 12 - free)
            draw.label(p, QPointF(w - 12, y), text, pal.accent, 9.0, "right", bold=True)
        bar_w = self._scale_bar_px()
        r     = draw.label(p, QPointF(12, h - 28), self.mode.upper(), pal.accent, 8.0, bold=True)
        hint  = QFontMetricsF(theme.mono_font(8.0)).elidedText(
            HINTS[self.mode], Qt.TextElideMode.ElideRight, w - r.right() - bar_w - 50)
        draw.label(p, QPointF(r.right() + 10, h - 28), hint, pal.text_dim, 8.0)
        draw.scale_bar(p, self.view, QPointF(w - 14 - bar_w, h - 26))
        if self._heatmap is not None and self.layers["heatmap"]:
            self._draw_colour_bar(p, w - 14, h - HUD_BOTTOM_PX - 12)

    def _scale_bar_px(self) -> float:
        """Width ``draw.scale_bar`` will give its bar (it picks a round length ≥ 60 px)."""
        for length in (10, 25, 50, 100, 200, 500):
            if length * self.view.scale >= 60:
                break
        return length * self.view.scale

    def _draw_legend(self, p: QPainter, frame, w: int, h: int) -> QRectF | None:
        """The shown algorithms' markers and current errors, if the widget has
        room for them; returns the legend's box (None if not drawn)."""
        pal  = theme.current
        keys = [a for a in theme.ALGO_ORDER if a in self.shown]
        row, x0, y0, width = 17.0, 40.0, 12.0, 150.0     # x0 clears the grid's y labels
        box = QRectF(x0, y0, width, 12 + row * (len(keys) + 1))
        if not keys or w < LEGEND_MIN_W or box.bottom() > h - HUD_BOTTOM_PX - COLOUR_BAR_PX:
            return None
        p.setPen(draw.pen(pal.border, 1.0))
        p.setBrush(theme.color(pal.bg, 0.85))
        p.drawRoundedRect(box, 3, 3)
        draw.label(p, QPointF(x0 + 10, y0 + 14), "ESTIMATE", pal.text_muted, 7.5, halo=False)
        draw.label(p, QPointF(x0 + width - 10, y0 + 14), "ERROR", pal.text_muted, 7.5, "right", halo=False)
        for k, a in enumerate(keys):
            y   = y0 + 14 + row * (k + 1)
            e   = frame.estimates.get(a) if frame is not None else None
            ok  = e is not None and e.ok
            err = frame.errors.get(a, float("nan")) if frame is not None else float("nan")
            c   = pal.algo[a]
            self._marker(p, QPointF(x0 + 16, y), theme.ALGO_SHAPES[a], c, 9.0, 1.3, 1.0 if ok else 0.45,
                         0.25 if ok else 0.0)
            draw.label(p, QPointF(x0 + 30, y), a, pal.text if ok else pal.text_muted, 8.0, halo=False)
            draw.label(p, QPointF(x0 + width - 10, y), "—" if err != err else f"{err:6.1f} cm",
                       pal.text_dim if ok else pal.text_muted, 8.0, "right", halo=False)
        return box

    def _draw_colour_bar(self, p: QPainter, right: float, top: float) -> None:
        pal, hm = theme.current, self._heatmap
        bar = QRectF(right - 150, top, 150, 8)
        p.drawImage(bar, hm.bar)
        p.setPen(draw.pen(pal.text_dim, 1.0, 0.8))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(bar)
        for level in hm.levels:
            x = bar.left() + bar.width() * (level - hm.vmin) / (hm.vmax - hm.vmin)
            p.drawLine(QPointF(x, bar.bottom()), QPointF(x, bar.bottom() + 3))
        draw.label(p, QPointF(bar.left(), top - 9), hm.title, pal.text_dim, 7.5)
        draw.label(p, QPointF(bar.left(), bar.bottom() + 10), f"{hm.vmin:.3g}", pal.text_dim, 7.5, halo=False)
        draw.label(p, QPointF(bar.right(), bar.bottom() + 10), f"{hm.vmax:.3g}", pal.text_dim, 7.5, "right",
                   halo=False)

    # ─── Ping animation ──────────────────────────────────────────────────────

    def _make_ping(self, frame) -> _Ping:
        """Work out once what the ping animation needs: how far each wavefront
        can be seen along each ray, and each detected path."""
        env    = self.sim.env
        origin = np.asarray(frame.true, dtype=float)
        A, B   = _segments(_surfaces(self.sim.world))
        # A receiver's own burst (crosstalk) leaves when the receivers are
        # triggered, this many cm of travel before the transmitter fires
        lead   = env.burst_lead_ms * speed_of_sound(env.temperature_c, env.humidity_pct) / 10.0
        fronts, packets, arrive, echoed = [], [], {}, set()
        if not any(rd.status == "radio" for rd in frame.readings):     # the transmitter fired
            hits = _ray_distances(origin, RAY_ANGLES, A, B)
            fronts.append(_Front(origin, np.zeros(len(RAY_ANGLES)), hits.min(axis=1, initial=np.inf), "wave"))
        for i, rd in enumerate(frame.readings):
            a = rd.detected
            if a is None:
                continue
            pts = np.asarray(a.path.points, dtype=float)
            cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))] - (lead if a.source != "TX" else 0.0)
            packets.append((pts, cum, _path_key(a)))
            arrive[i] = max(float(cum[-1]), 0.0)
            k = _segment_through(pts[1], A, B) if a.source == "TX" and a.path.kind == "reflect" \
                and len(pts) == 3 else None
            if k is not None and k not in echoed:
                # A one-bounce echo spreads from the image source behind segment k:
                # seen from where it crosses k to the next thing it hits
                image  = _mirror(origin, A[k], B[k])
                hits   = _ray_distances(image, RAY_ANGLES, A, B)
                lo     = hits[:, k]
                beyond = np.where(hits > lo[:, None] + 1e-6, hits, np.inf)
                beyond[:, k] = np.inf
                fronts.append(_Front(image, lo, beyond.min(axis=1), "echo"))
                echoed.add(k)
        corners = np.asarray(self.sim.world.walls, dtype=float).reshape(-1, 2)
        reach   = max([float(np.max(np.hypot(*(corners - origin).T))) if len(corners) else 300.0]
                      + list(arrive.values()))
        return _Ping(time.perf_counter(), reach, fronts, packets, arrive)

    def _flashes(self, now: float) -> dict:
        """{receiver: 0..1} how brightly each receiver flashes now, as its sound arrives."""
        out = {}
        for ping in self._pings:
            for i, L in ping.arrive.items():
                age = now - (ping.start + self.wave_seconds * L / ping.reach)
                if 0.0 <= age < FLASH_SECONDS:
                    out[i] = max(out.get(i, 0.0), 1.0 - age / FLASH_SECONDS)
        return out

    def _animate(self) -> None:
        """Timer tick: forget finished pings, stop when none are left, repaint."""
        now = time.perf_counter()
        self._pings = [g for g in self._pings if now - g.start < self.wave_seconds + FLASH_SECONDS]
        if not self._pings:
            self._timer.stop()
        self.update()

    # ─── Tooltips ────────────────────────────────────────────────────────────

    def _receiver_tip(self, i: int) -> str:
        """What receiver ``i`` heard this frame and what the library made of it."""
        r, frame = self.sim.receivers[i], self.frame
        lines    = [f"R{r.id}   state: {r.state}   facing {r.facing_deg:.0f}°"]
        if frame is not None and i < len(frame.ticks):
            d  = frame.solution.distances[i]
            st = frame.status[i]
            lines += [f"{st}: {STATUS.get(st, 'replayed from a tick log')}",
                      f"tick          {frame.ticks[i]}",
                      f"distance      {d:.1f} cm (library, from the tick)"]
            rd = frame.readings[i] if i < len(frame.readings) else None
            if rd is not None and rd.measured_cm is not None:
                lines.append(f"timed         {rd.measured_cm:.1f} cm (hardware, from the echo time)")
            if rd is not None:
                lines.append(f"true          {rd.true_cm:.1f} cm (library {d - rd.true_cm:+.1f})")
            if rd is not None and rd.detected is not None:
                a = rd.detected
                lines += [f"strength      {a.ratio_db:+.1f} dB over threshold",
                          f"path          {a.path.kind}, {a.path.length:.1f} cm"
                          + ("" if a.source == "TX" else f", from {a.source}'s own burst")]
            elif rd is not None and rd.arrivals:
                best = max(rd.arrivals, key=lambda x: x.ratio_db)
                lines.append(f"strongest     {best.ratio_db:+.1f} dB ({best.path.kind}): too weak or unheard")
            if rd is not None and not rd.los:
                lines.append("line of sight blocked")
            lines.append("kept by the library's filter" if i in frame.solution.sane
                         else "dropped by the library's filter")
            cep = frame.estimates.get("CEP")
            if cep is not None and i in {int(k) for k in cep.used}:
                lines.append("used by the library's CEP fix")
        off = float(np.hypot(*(self.sim.believed[i] - r.position)))
        if off > 0.5:
            lines.append(f"{off:.1f} cm from where config.toml says (survey error)")
        return "\n".join(lines)

    def _estimates_at(self, x: float, y: float) -> list[str]:
        """The shown algorithms whose marker is under pixel (x, y)."""
        frame = self.frame
        if frame is None or not self.layers["estimates"]:
            return []
        out = []
        for a, e in self._shown_estimates(frame):
            q, _ = self._estimate_spot(e.pos)
            if math.hypot(q.x() - x, q.y() - y) <= PICK_PX * 0.7:
                out.append(a)
        return out

    def _estimate_tip(self, a: str) -> str:
        """Algorithm ``a``'s fix this frame, its error and the receivers it used."""
        e, info = self.frame.estimates[a], ALGORITHMS[a]
        err     = self.frame.errors.get(a, float("nan"))
        lines   = [f"{a}   {info.label} ({info.source})",
                   f"at ({e.pos[0]:.1f}, {e.pos[1]:.1f})" + ("" if err != err else f"   error {err:.1f} cm")]
        if e.used:
            lines.append("receivers " + " ".join(f"R{int(i)}" for i in e.used))
        if e.note:
            lines.append(e.note)
        if not e.ok:
            lines.append("no fix this frame")
        return "\n".join(lines)

    def event(self, e) -> bool:
        if e.type() == QEvent.Type.ToolTip:
            text = self.tooltip_at(e.pos().x(), e.pos().y())
            if text:
                QToolTip.showText(e.globalPos(), text, self)
            else:
                QToolTip.hideText()
                e.ignore()
            return True
        return super().event(e)

    # ─── Mouse and keyboard ──────────────────────────────────────────────────

    def resizeEvent(self, event) -> None:
        self.view.resize(self.width(), self.height())
        if self._auto_fit:
            self.view.fit(self._bounds(), FIT_MARGIN_PX)
        super().resizeEvent(event)

    def mousePressEvent(self, e) -> None:
        self.setFocus()
        if self._press is not None:         # a second button while one is held: ignore it
            return
        pos = e.position()
        self._press, self._moved = (e.button(), pos, pos), False
        if e.button() != Qt.MouseButton.LeftButton:
            return
        w = self.view.world(pos.x(), pos.y())
        if self.mode == "obstacle":
            self._band = (w, w)
        elif self.mode == "receiver":
            i = self.receiver_at(pos.x(), pos.y())
            if i is not None:
                self._drag = (i, self.sim.receivers[i].position - np.asarray(w))
                self._set_cursor()
        self._changed()

    def mouseMoveEvent(self, e) -> None:
        pos = e.position()
        wx, wy = self.view.world(pos.x(), pos.y())
        self.cursorMoved.emit(wx, wy)
        if self._press is None:
            self._hover(pos)
            return
        button, start, last = self._press
        if not self._moved and math.hypot(pos.x() - start.x(), pos.y() - start.y()) <= DRAG_PX:
            return
        self._moved = True
        self._press = (button, start, pos)
        if button in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._auto_fit = False
            self.view.pan(pos.x() - last.x(), pos.y() - last.y())
        elif self._band is not None:
            self._band = (self._band[0], (wx, wy))
        elif self._drag is not None:
            i, offset = self._drag
            with self.lock:
                self.sim.receivers[i].position = np.array([wx, wy]) + offset
        self._changed()

    def mouseReleaseEvent(self, e) -> None:
        if self._press is None or e.button() != self._press[0]:
            return
        pos    = e.position()
        wx, wy = self.view.world(pos.x(), pos.y())
        click  = not self._moved
        if e.button() == Qt.MouseButton.LeftButton:
            if click and self.mode == "move":
                self.targetClicked.emit(wx, wy)
            elif click and self.mode == "calpick":
                self.calPointPicked.emit(wx, wy)
            elif self._band is not None and not click:
                self._add_obstacle(self._band[0], (wx, wy))
        elif e.button() == Qt.MouseButton.RightButton and click and self.mode == "obstacle":
            k = self.obstacle_at(pos.x(), pos.y())
            if k is not None:
                self._delete_obstacle(k)
        self._end_press()
        self._hover(pos)
        self._changed()

    def mouseDoubleClickEvent(self, e) -> None:
        self._end_press()
        self.fit()

    def _end_press(self) -> None:
        """Forget the mouse press in progress; a receiver it dragged stays where it is."""
        if self._drag is not None and self._moved:
            self.receiversEdited.emit()
        self._press, self._moved, self._band, self._drag = None, False, None, None
        self._set_cursor()

    def wheelEvent(self, e) -> None:
        steps = e.angleDelta().y() / 120.0
        if not steps:
            return
        pos = e.position()
        i   = self.receiver_at(pos.x(), pos.y()) if self.mode == "receiver" else None
        if i is not None:
            r = self.sim.receivers[i]
            with self.lock:
                r.facing_deg = (r.facing_deg + 5.0 * steps + 180.0) % 360.0 - 180.0
            self.receiversEdited.emit()
        else:
            self._auto_fit = False
            self.view.zoom_at(1.15 ** steps, pos.x(), pos.y())
        self._changed()

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key.Key_Escape:
            self.set_mode("move")
        else:
            super().keyPressEvent(e)

    def leaveEvent(self, e) -> None:
        self._hover_rx = self._hover_obstacle = None
        self._changed()

    def _hover(self, pos: QPointF) -> None:
        rx = self.receiver_at(pos.x(), pos.y())
        ob = self.obstacle_at(pos.x(), pos.y()) if self.mode == "obstacle" else None
        if (rx, ob) != (self._hover_rx, self._hover_obstacle):
            self._hover_rx, self._hover_obstacle = rx, ob
            self._set_cursor()
            self._changed()

    def _set_cursor(self) -> None:
        shape = Qt.CursorShape.CrossCursor
        if self.mode == "receiver":
            shape = (Qt.CursorShape.ClosedHandCursor if self._drag is not None else
                     Qt.CursorShape.OpenHandCursor if self._hover_rx is not None else Qt.CursorShape.ArrowCursor)
        elif self.mode == "move":
            shape = Qt.CursorShape.ArrowCursor
        self.setCursor(shape)

    # ─── Editing the simulation ──────────────────────────────────────────────

    def _add_obstacle(self, a, b) -> None:
        (x0, y0), (x1, y1) = a, b
        if abs(x1 - x0) < 2.0 or abs(y1 - y0) < 2.0:
            return
        world = self.sim.world
        box   = Obstacle.rect(float(round(x0)), float(round(y0)), float(round(x1)), float(round(y1)),
                              name=f"box {len(world.obstacles) + 1}")
        with self.lock:
            world.obstacles = [*world.obstacles, box]
        self.worldEdited.emit()

    def _delete_obstacle(self, k: int) -> None:
        world = self.sim.world
        with self.lock:
            world.obstacles = [o for j, o in enumerate(world.obstacles) if j != k]
        self._hover_obstacle = None
        self.worldEdited.emit()

    def _bounds(self) -> tuple[float, float, float, float]:
        """The room, every receiver (true and believed) and the calibration points."""
        pts = [np.asarray(self.sim.world.walls, dtype=float).reshape(-1, 2), self.sim.coords,
               np.asarray(self.sim.believed, dtype=float), np.asarray(self.cal_points, dtype=float).reshape(-1, 2)]
        pts = np.vstack(pts)
        (x0, y0), (x1, y1) = pts.min(axis=0) - 10.0, pts.max(axis=0) + 10.0
        return float(x0), float(y0), float(x1), float(y1)
