"""
Motion — where the transmitter goes.

Modes: ``target`` (drive to a clicked point and stop), ``circle``, ``figure8``,
``lawnmower`` (back and forth across the arena) and ``random`` (random
waypoints inside the arena).  The transmitter drives at ``speed`` cm/s and
faces the way it is going (its heading), which matters when its own beam
pattern is on.

It never drives into or through an obstacle: ``clip_move`` tests the whole
step, not just where it ends, and stops it just short of the first edge.  One
that is already inside an obstacle (drawn over it, or put there) may drive out;
``push_out`` moves a point inside an obstacle to just outside it.
"""

from __future__ import annotations

import math

import numpy as np

MODES = ("target", "circle", "figure8", "lawnmower", "random")


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def clip_move(p, q, obstacles, margin: float = 0.5) -> tuple[np.ndarray, bool]:
    """How far the move ``p``→``q`` can go: ``(q, False)``, or a point ``margin``
    cm short of the first obstacle edge it would touch and ``True``.  Obstacles
    that already contain ``p`` are ignored, so it can always leave one."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    r    = q - p
    L    = float(np.hypot(*r))
    t    = math.inf
    for o in obstacles:
        if o.contains(p):
            continue
        a   = np.asarray(o.points, dtype=float)
        s   = np.roll(a, -1, axis=0) - a
        den = _cross(r, s)
        ok  = np.abs(den) > 1e-12
        den = np.where(ok, den, 1.0)
        ap  = a - p
        tt  = _cross(ap, s) / den
        uu  = _cross(ap, r) / den
        hit = ok & (tt >= 0) & (tt <= 1) & (uu >= 0) & (uu <= 1)
        if hit.any():
            t = min(t, float(tt[hit].min()))
        elif o.contains(q):              # rounding: it ends inside without crossing an edge
            t = 0.0
    if t == math.inf:
        return q, False
    return p + r * max(0.0, t - margin / max(L, 1e-12)), True


def push_out(p, obstacles, margin: float = 1.0) -> np.ndarray:
    """``p``, or if it is inside an obstacle, the nearest point on that
    obstacle's edge moved ``margin`` cm outwards."""
    p = np.asarray(p, dtype=float).copy()
    for _ in range(len(obstacles)):         # overlapping obstacles: out of one may be into another
        o = next((o for o in obstacles if o.contains(p)), None)
        if o is None:
            break
        a    = np.asarray(o.points, dtype=float)
        s    = np.roll(a, -1, axis=0) - a
        ll   = np.maximum(np.sum(s * s, axis=1), 1e-12)
        u    = np.clip(np.sum((p - a) * s, axis=1) / ll, 0.0, 1.0)
        near = a + s * u[:, None]
        k    = int(np.argmin(np.hypot(*(near - p).T)))
        n    = np.array([s[k, 1], -s[k, 0]]) / np.sqrt(ll[k])
        if np.dot(n, a[k] - a.mean(axis=0)) < 0:     # point it away from the inside
            n = -n
        p = near[k] + n * margin
    return p


class Motion:
    def __init__(self, bounds: tuple[float, float, float, float], seed: int | None = None):
        self.bounds  = bounds          # x0, y0, x1, y1 the paths stay inside
        self.mode    = "target"
        self.speed   = 50.0
        self.radius  = 100.0
        self.phase   = 0.0
        self.target  = np.zeros(2)
        self.heading = 90.0
        self.rng     = np.random.default_rng(seed)
        self._lane   = 0

    def set_mode(self, mode: str) -> None:
        self.mode, self.phase, self._lane = mode, 0.0, 0

    def go_to(self, xy) -> None:
        self.mode   = "target"
        self.target = np.asarray(xy, dtype=float)

    def blocked(self) -> None:
        """The way ahead runs into an obstacle.  The transmitter stops at it;
        a waypoint mode picks its next waypoint instead."""
        if self.mode in ("lawnmower", "random"):
            self.phase = 0.0

    def _path_point(self, s: float) -> np.ndarray:
        x0, y0, x1, y1 = self.bounds
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if self.mode == "circle":
            a = s / self.radius
            return np.array([cx + self.radius * math.cos(a), cy + self.radius * math.sin(a)])
        if self.mode == "figure8":
            a = s / (1.6 * self.radius)
            return np.array([cx + self.radius * 0.9 * math.sin(a),
                             cy + self.radius * 1.4 * math.sin(a) * math.cos(a) * 1.4])
        raise ValueError(self.mode)

    def _lawnmower_points(self, spacing: float = 40.0) -> list:
        """Back and forth across the arena in lanes ``spacing`` apart, then back up."""
        x0, y0, x1, y1 = self.bounds
        lanes = np.linspace(y0, y1, max(2, int(round((y1 - y0) / spacing)) + 1))
        pts = []
        for k, y in enumerate(lanes):
            ends = (x0, x1) if k % 2 == 0 else (x1, x0)
            pts += [np.array([ends[0], y]), np.array([ends[1], y])]
        return pts + pts[-2:0:-1]

    def step(self, pos: np.ndarray, dt: float) -> np.ndarray:
        """Advance ``dt`` seconds from ``pos``; returns the new position."""
        move = self.speed * dt
        if self.mode in ("circle", "figure8"):
            self.phase += move
            self.target = self._path_point(self.phase)
        elif self.mode == "lawnmower":
            if np.linalg.norm(self.target - pos) < 1.0 or self.phase == 0.0:
                self.phase  = 1.0
                pts         = self._lawnmower_points()
                self.target = pts[self._lane % len(pts)]
                self._lane += 1
        elif self.mode == "random":
            if np.linalg.norm(self.target - pos) < 1.0 or self.phase == 0.0:
                self.phase = 1.0
                x0, y0, x1, y1 = self.bounds
                self.target = np.array([self.rng.uniform(x0, x1), self.rng.uniform(y0, y1)])

        # On a moving path point the transmitter may go twice as fast until it
        # catches up; at the same speed it would chase the point for ever.
        reach = 2 * move if self.mode in ("circle", "figure8") else move
        delta = self.target - pos
        dist  = float(np.hypot(*delta))
        if dist > 1e-9:
            self.heading = math.degrees(math.atan2(delta[1], delta[0]))
        if dist <= reach:
            return self.target.copy()
        return pos + delta / dist * reach
