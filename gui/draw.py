"""
Drawing kit — the vector-style primitives every canvas uses (arena, lab,
charts), so they all look alike: thin round-capped lines with an optional
soft glow, outlined markers, hatched obstacles, monospace labels.

``View`` maps world centimetres (y up) to widget pixels (y down) and handles
zoom and pan.  Everything else takes a ``QPainter`` plus either pixel or world
coordinates (functions taking a ``view`` work in world cm).
"""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QPainter, QPainterPath, QPen, QPolygonF

from engine.acoustics import directivity
from gui import theme
from gui.theme import color


class View:
    """World (cm, y up) ↔ widget pixels (y down)."""

    def __init__(self):
        self.scale  = 1.0       # pixels per cm
        self.cx     = 0.0       # world point at the widget centre
        self.cy     = 0.0
        self.width  = 1
        self.height = 1

    def resize(self, w: int, h: int) -> None:
        self.width, self.height = max(w, 1), max(h, 1)

    def fit(self, bounds, margin_px: float = 40.0) -> None:
        """Zoom so ``bounds`` (x0, y0, x1, y1) fills the widget."""
        x0, y0, x1, y1 = bounds
        self.cx, self.cy = (x0 + x1) / 2, (y0 + y1) / 2
        sx = (self.width - 2 * margin_px) / max(x1 - x0, 1e-9)
        sy = (self.height - 2 * margin_px) / max(y1 - y0, 1e-9)
        self.scale = max(min(sx, sy), 1e-3)

    def px(self, x: float, y: float) -> QPointF:
        return QPointF(self.width / 2 + (x - self.cx) * self.scale,
                       self.height / 2 - (y - self.cy) * self.scale)

    def world(self, px: float, py: float) -> tuple[float, float]:
        return (self.cx + (px - self.width / 2) / self.scale,
                self.cy - (py - self.height / 2) / self.scale)

    def visible(self) -> tuple[float, float, float, float]:
        x0, y1 = self.world(0, 0)
        x1, y0 = self.world(self.width, self.height)
        return x0, y0, x1, y1

    def zoom_at(self, factor: float, px: float, py: float) -> None:
        wx, wy = self.world(px, py)
        self.scale = min(max(self.scale * factor, 0.05), 200.0)
        nx, ny = self.world(px, py)
        self.cx += wx - nx
        self.cy += wy - ny

    def pan(self, dx_px: float, dy_px: float) -> None:
        self.cx -= dx_px / self.scale
        self.cy += dy_px / self.scale


# ─── Pens and lines ──────────────────────────────────────────────────────────

def pen(c, width: float = 1.0, alpha: float = 1.0, dash=None) -> QPen:
    p = QPen(color(c, alpha) if isinstance(c, str) else c, width)
    p.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setCosmetic(True)
    if dash:
        p.setDashPattern(dash)
    return p


DASH  = [4, 4]
DOT   = [1, 3]
LONG  = [8, 4]


def stroke(p: QPainter, path: QPainterPath, c: str, width: float = 1.2, alpha: float = 1.0,
           dash=None, glow: bool | None = None) -> None:
    """Draw ``path`` as a vector line, with a soft glow on dark palettes."""
    glow = theme.current.glow if glow is None else glow
    p.setBrush(Qt.BrushStyle.NoBrush)
    if glow and alpha > 0.2:
        p.setPen(pen(c, width + 5, alpha * 0.07, dash))
        p.drawPath(path)
        p.setPen(pen(c, width + 2.5, alpha * 0.16, dash))
        p.drawPath(path)
    p.setPen(pen(c, width, alpha, dash))
    p.drawPath(path)


def polyline_path(points_px) -> QPainterPath:
    path, first = QPainterPath(), True
    for q in points_px:
        if first:
            path.moveTo(q)
            first = False
        else:
            path.lineTo(q)
    return path


def polyline(p: QPainter, view: View, pts, c: str, width: float = 1.2, alpha: float = 1.0,
             dash=None, glow=None, closed: bool = False) -> None:
    """Line through world points (nan points break the line)."""
    path, pen_down = QPainterPath(), False
    for x, y in np.asarray(pts, dtype=float).reshape(-1, 2):
        if not (math.isfinite(x) and math.isfinite(y)):
            pen_down = False
            continue
        q = view.px(x, y)
        if pen_down:
            path.lineTo(q)
        else:
            path.moveTo(q)
            pen_down = True
    if closed:
        path.closeSubpath()
    stroke(p, path, c, width, alpha, dash, glow)


def circle(p: QPainter, view: View, centre, radius_cm: float, c: str, width: float = 1.0,
           alpha: float = 1.0, dash=None, glow=None) -> None:
    path = QPainterPath()
    r = radius_cm * view.scale
    path.addEllipse(view.px(*centre), r, r)
    stroke(p, path, c, width, alpha, dash, glow)


def ellipse_cov(p: QPainter, view: View, centre, cov, c: str, k: float = 2.0, width: float = 1.0,
                alpha: float = 0.8, dash=None) -> None:
    """``k``-sigma ellipse of a 2×2 covariance (cm²) around ``centre``."""
    if cov is None or centre is None:
        return
    cov = np.asarray(cov, dtype=float)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh((cov + cov.T) / 2)
    vals = np.sqrt(np.clip(vals, 0, None)) * k
    if vals.max() > 5000:
        return
    t = np.linspace(0, 2 * np.pi, 49)
    pts = (vecs @ (vals[:, None] * np.vstack([np.cos(t), np.sin(t)]))).T + np.asarray(centre)
    polyline(p, view, pts, c, width, alpha, dash, closed=True)


def arrow(p: QPainter, a: QPointF, b: QPointF, c: str, width: float = 1.2, head: float = 7.0,
          alpha: float = 1.0, dash=None) -> None:
    path = QPainterPath(a)
    path.lineTo(b)
    stroke(p, path, c, width, alpha, dash)
    d = b - a
    L = math.hypot(d.x(), d.y())
    if L < 1e-6:
        return
    ux, uy = d.x() / L, d.y() / L
    left  = QPointF(b.x() - head * (ux * 0.9 - uy * 0.45), b.y() - head * (uy * 0.9 + ux * 0.45))
    right = QPointF(b.x() - head * (ux * 0.9 + uy * 0.45), b.y() - head * (uy * 0.9 - ux * 0.45))
    hp = QPainterPath(left)
    hp.lineTo(b)
    hp.lineTo(right)
    stroke(p, hp, c, width, alpha)


# ─── Markers and text ────────────────────────────────────────────────────────

def marker_path(centre: QPointF, shape: str, size: float) -> QPainterPath:
    x, y, s = centre.x(), centre.y(), size / 2
    path = QPainterPath()
    if shape in ("circle", "ring"):
        path.addEllipse(centre, s, s)
        if shape == "ring":
            path.addEllipse(centre, s * 0.35, s * 0.35)
    elif shape == "square":
        path.addRect(QRectF(x - s * 0.85, y - s * 0.85, s * 1.7, s * 1.7))
    elif shape == "diamond":
        path.addPolygon(QPolygonF([QPointF(x, y - s * 1.15), QPointF(x + s * 1.15, y),
                                   QPointF(x, y + s * 1.15), QPointF(x - s * 1.15, y)]))
        path.closeSubpath()
    elif shape in ("triangle", "triangle_down"):
        k = 1 if shape == "triangle" else -1
        path.addPolygon(QPolygonF([QPointF(x, y - k * s * 1.2), QPointF(x + s * 1.1, y + k * s * 0.8),
                                   QPointF(x - s * 1.1, y + k * s * 0.8)]))
        path.closeSubpath()
    elif shape == "hexagon":
        path.addPolygon(QPolygonF([QPointF(x + s * math.cos(a), y + s * math.sin(a))
                                   for a in np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6]))
        path.closeSubpath()
    elif shape == "star":
        pts = []
        for k in range(10):
            r = s * (1.25 if k % 2 == 0 else 0.5)
            a = -np.pi / 2 + k * np.pi / 5
            pts.append(QPointF(x + r * math.cos(a), y + r * math.sin(a)))
        path.addPolygon(QPolygonF(pts))
        path.closeSubpath()
    elif shape == "cross":
        path.moveTo(x - s, y - s), path.lineTo(x + s, y + s)
        path.moveTo(x - s, y + s), path.lineTo(x + s, y - s)
    elif shape == "plus":
        path.moveTo(x - s, y), path.lineTo(x + s, y)
        path.moveTo(x, y - s), path.lineTo(x, y + s)
    else:
        path.addEllipse(centre, s, s)
    return path


def marker(p: QPainter, centre: QPointF, shape: str, c: str, size: float = 10.0,
           width: float = 1.4, alpha: float = 1.0, fill: float = 0.0) -> None:
    path = marker_path(centre, shape, size)
    if fill > 0:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(color(c, alpha * fill)))
        p.drawPath(path)
    stroke(p, path, c, width, alpha)


def crosshair(p: QPainter, centre: QPointF, c: str, size: float = 9.0, width: float = 1.4,
              gap: float = 3.0) -> None:
    x, y = centre.x(), centre.y()
    path = QPainterPath()
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        path.moveTo(x + dx * gap, y + dy * gap)
        path.lineTo(x + dx * size, y + dy * size)
    path.addEllipse(centre, size * 0.55, size * 0.55)
    stroke(p, path, c, width)


def label(p: QPainter, at: QPointF, text: str, c: str, size: float = 8.5, align: str = "left",
          bold: bool = False, halo: bool = True, alpha: float = 1.0) -> QRectF:
    """Monospace text anchored at ``at`` (left/center/right of its baseline middle)."""
    f = theme.mono_font(size, bold)
    p.setFont(f)
    fm = p.fontMetrics()
    w, h = fm.horizontalAdvance(text), fm.height()
    x = at.x() - (w / 2 if align == "center" else w if align == "right" else 0)
    rect = QRectF(x, at.y() - h / 2, w, h)
    if halo:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color(theme.current.bg, 0.72 * alpha))
        p.drawRoundedRect(rect.adjusted(-3, -1, 3, 1), 2, 2)
    p.setPen(pen(c, 1, alpha))
    p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
    return rect


# ─── Scenery ─────────────────────────────────────────────────────────────────

def grid(p: QPainter, view: View, minor: float = 25.0, major: float = 100.0, labels: bool = True) -> None:
    pal = theme.current
    x0, y0, x1, y1 = view.visible()
    while minor * view.scale < 6:
        minor *= 2
    while major * view.scale < 40:
        major *= 2
    for step, c in ((minor, pal.grid_minor), (major, pal.grid_major)):
        p.setPen(pen(c, 1.0))
        for x in np.arange(math.floor(x0 / step) * step, x1 + step, step):
            a, b = view.px(x, y0), view.px(x, y1)
            p.drawLine(a, b)
        for y in np.arange(math.floor(y0 / step) * step, y1 + step, step):
            p.drawLine(view.px(x0, y), view.px(x1, y))
    # Axes through the origin
    p.setPen(pen(pal.grid_major, 1.2, 1.0))
    p.drawLine(view.px(0, y0), view.px(0, y1))
    p.drawLine(view.px(x0, 0), view.px(x1, 0))
    if labels:
        for x in np.arange(math.floor(x0 / major) * major, x1 + major, major):
            label(p, QPointF(view.px(x, 0).x() + 3, view.height - 10), f"{x:.0f}", pal.text_muted, 7.5, halo=False)
        for y in np.arange(math.floor(y0 / major) * major, y1 + major, major):
            label(p, QPointF(4, view.px(0, y).y() - 7), f"{y:.0f}", pal.text_muted, 7.5, halo=False)


def scale_bar(p: QPainter, view: View, at: QPointF) -> None:
    pal = theme.current
    for length in (10, 25, 50, 100, 200, 500):
        if length * view.scale >= 60:
            break
    w = length * view.scale
    path = QPainterPath(QPointF(at.x(), at.y() - 4))
    path.lineTo(at.x(), at.y())
    path.lineTo(at.x() + w, at.y())
    path.lineTo(at.x() + w, at.y() - 4)
    stroke(p, path, pal.text_dim, 1.0, glow=False)
    label(p, QPointF(at.x() + w / 2, at.y() - 10), f"{length} cm", pal.text_dim, 7.5, "center", halo=False)


def _inside_pieces(a: QPointF, b: QPointF, poly: list) -> list:
    """The parts of the line a-b that lie inside the polygon (even-odd rule),
    as parameter pairs along a-b.  ``a`` and ``b`` must lie outside it."""
    dx, dy = b.x() - a.x(), b.y() - a.y()
    ts = []
    for p0, p1 in zip(poly, poly[1:] + poly[:1]):
        ex, ey = p1.x() - p0.x(), p1.y() - p0.y()
        den    = dx * ey - dy * ex
        if abs(den) < 1e-12:
            continue
        wx, wy = p0.x() - a.x(), p0.y() - a.y()
        u = (wx * dy - wy * dx) / den
        if 0.0 <= u < 1.0:
            ts.append((wx * ey - wy * ex) / den)
    ts.sort()
    return list(zip(ts[0::2], ts[1::2]))


def hatched_polygon(p: QPainter, view: View, pts, c: str, spacing_px: float = 7.0,
                    width: float = 1.2, alpha: float = 1.0) -> None:
    """Outlined polygon filled with diagonal hatching, like a technical drawing.

    The hatch lines are cut to the outline here rather than with a clip path,
    so an exported SVG shows them the same in every viewer."""
    corners = [view.px(x, y) for x, y in pts]
    path    = QPainterPath()
    path.addPolygon(QPolygonF(corners))
    path.closeSubpath()
    r = path.boundingRect().adjusted(-1, -1, 1, 1)
    p.setPen(pen(c, 0.8, alpha * 0.45))
    k = -r.height()
    while k < r.width():
        a = QPointF(r.left() + k, r.bottom())
        b = QPointF(r.left() + k + r.height(), r.top())
        for t0, t1 in _inside_pieces(a, b, corners):
            p.drawLine(a + (b - a) * t0, a + (b - a) * t1)
        k += spacing_px
    stroke(p, path, c, width, alpha)


def wall_outline(p: QPainter, view: View, pts, c: str) -> None:
    """Room walls: a solid line with short ticks on the outside."""
    if len(pts) < 3:
        return
    polyline(p, view, list(pts) + [pts[0]], c, 1.6)
    pts = np.asarray(pts, dtype=float)
    centre = pts.mean(axis=0)
    p.setPen(pen(c, 0.8, 0.5))
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        d = b - a
        L = float(np.hypot(*d))
        if L < 1e-6:
            continue
        n = np.array([d[1], -d[0]]) / L
        if np.dot(n, (a + b) / 2 - centre) < 0:
            n = -n
        step = 12.0 / view.scale
        for t in np.arange(step / 2, L, step):
            q = a + d / L * t
            p.drawLine(view.px(*q), view.px(*(q + n * 6.0 / view.scale + d / L * 4.0 / view.scale)))


def beam_lobe(p: QPainter, view: View, pos, facing_deg: float, half_angle: float, radius_cm: float,
              c: str, alpha: float = 0.7, width: float = 1.0) -> None:
    """Polar plot of a transducer's beam pattern (amplitude) around ``pos``."""
    angles = np.linspace(-90, 90, 73)
    pts = []
    for a in angles:
        r = radius_cm * directivity(a, half_angle)
        t = math.radians(facing_deg + a)
        pts.append((pos[0] + r * math.cos(t), pos[1] + r * math.sin(t)))
    pts = [tuple(pos)] + pts + [tuple(pos)]
    polyline(p, view, pts, c, width, alpha)


def receiver_glyph(p: QPainter, view: View, pos, facing_deg: float, body: str, ring: str | None,
                   state: str = "on", size: float = 9.0, label_text: str = "", used: bool = False) -> None:
    """A receiver: a circle with its transducer face drawn as an arc toward
    ``facing_deg``, an outer ring in its status colour, and a label."""
    pal = theme.current
    c = view.px(*pos)
    t = math.radians(facing_deg)
    alpha = 0.45 if state == "off" else 1.0
    path = QPainterPath()
    path.addEllipse(c, size * 0.55, size * 0.55)
    fx, fy = math.cos(t), -math.sin(t)
    face = QPainterPath()
    face.arcMoveTo(QRectF(c.x() - size, c.y() - size, 2 * size, 2 * size), facing_deg - 50)
    face.arcTo(QRectF(c.x() - size, c.y() - size, 2 * size, 2 * size), facing_deg - 50, 100)
    face.moveTo(c.x() + fx * size * 0.55, c.y() + fy * size * 0.55)
    face.lineTo(c.x() + fx * size * 1.45, c.y() + fy * size * 1.45)
    stroke(p, path, body, 1.4, alpha)
    stroke(p, face, body, 1.4, alpha)
    if ring:
        r = QPainterPath()
        r.addEllipse(c, size * 1.65, size * 1.65)
        stroke(p, r, ring, 2.4 if used else 1.2, alpha, None if state != "blocked" else DASH)
    if state == "off":
        x = marker_path(c, "cross", size * 1.4)
        stroke(p, x, pal.status["off"], 1.4)
    if label_text:
        label(p, QPointF(c.x() - fx * size * 2.6, c.y() - fy * size * 2.6), label_text, body, 8.5,
              "center", bold=True)
