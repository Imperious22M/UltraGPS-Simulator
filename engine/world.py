"""
World geometry — the room walls and obstacles sound travels between, and the
paths it can take from a source to a receiver:

* ``direct``    straight line, if nothing blocks it;
* ``reflect``   bouncing off walls or obstacles (image-source method, up to
                ``max_order`` bounces; each bounce keeps ``reflectivity`` of the
                amplitude);
* ``diffract``  bending round one corner of an obstacle that blocks the
                straight line, with the knife-edge loss from ``acoustics``.

All in 2D, centimetres.  Obstacles are convex polygons; walls are one closed
polygon around the arena (or none, for an open field).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine.acoustics import knife_edge_db

EPS = 1e-7


@dataclass
class Obstacle:
    points: list                    # convex polygon [(x, y), ...]
    reflectivity: float = 0.7       # amplitude kept on each bounce
    name: str = ""

    @staticmethod
    def rect(x0: float, y0: float, x1: float, y1: float, **kw) -> "Obstacle":
        xa, xb, ya, yb = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        return Obstacle([(xa, ya), (xb, ya), (xb, yb), (xa, yb)], **kw)

    def contains(self, p) -> bool:
        pts = np.asarray(self.points, dtype=float)
        e   = np.roll(pts, -1, axis=0) - pts
        rel = np.asarray(p, dtype=float) - pts
        s   = e[:, 0] * rel[:, 1] - e[:, 1] * rel[:, 0]
        return bool(np.all(s >= 0) or np.all(s <= 0))


@dataclass
class Path:
    points: np.ndarray              # (k, 2): source, bounce/corner points, receiver
    kind: str                       # "direct", "reflect" or "diffract"
    gain: float = 1.0               # reflection coefficients x diffraction loss (amplitude)

    @property
    def legs(self) -> list[tuple[np.ndarray, float]]:
        """(unit direction of travel, length) of each straight leg."""
        out = []
        for a, b in zip(self.points[:-1], self.points[1:]):
            d = b - a
            n = float(np.hypot(*d))
            out.append((d / n if n > 0 else np.zeros(2), n))
        return out

    @property
    def length(self) -> float:
        return float(np.sum(np.hypot(*np.diff(self.points, axis=0).T)))

    @property
    def depart(self) -> np.ndarray:
        return self.legs[0][0]

    @property
    def arrive(self) -> np.ndarray:
        return self.legs[-1][0]


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _mirror(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (b - a) / np.hypot(*(b - a))
    return 2.0 * (a + d * np.dot(p - a, d)) - p


@dataclass
class World:
    walls: list = field(default_factory=list)       # closed polygon [(x, y), ...]; [] = open field
    wall_reflectivity: float = 0.8
    obstacles: list = field(default_factory=list)   # [Obstacle]

    def __post_init__(self):
        self._key = None

    # ─── Segments ────────────────────────────────────────────────────────────

    def _segments(self):
        """Arrays ``A``, ``B`` (segment end points), ``refl`` and ``owner``
        (-1 for a wall, else the obstacle index), rebuilt when the geometry changes."""
        key = (tuple(map(tuple, self.walls)), self.wall_reflectivity,
               tuple((tuple(map(tuple, o.points)), o.reflectivity) for o in self.obstacles))
        if key != self._key:
            A, B, refl, owner = [], [], [], []
            polys = ([(self.walls, self.wall_reflectivity, -1)] if len(self.walls) >= 3 else [])
            polys += [(o.points, o.reflectivity, i) for i, o in enumerate(self.obstacles)]
            for pts, r, who in polys:
                pts = [np.asarray(p, dtype=float) for p in pts]
                for a, b in zip(pts, pts[1:] + pts[:1]):
                    if np.hypot(*(b - a)) > EPS:
                        A.append(a), B.append(b), refl.append(r), owner.append(who)
            self._A     = np.array(A, dtype=float).reshape(-1, 2)
            self._B     = np.array(B, dtype=float).reshape(-1, 2)
            self._refl  = np.array(refl, dtype=float)
            self._owner = np.array(owner, dtype=int)
            self._key   = key
        return self._A, self._B, self._refl, self._owner

    @property
    def segment_count(self) -> int:
        return len(self._segments()[0])

    def blocked(self, p, q, skip: tuple = ()) -> bool:
        """Does the straight line p→q cross a wall or obstacle edge (other than
        segments in ``skip``)?  Touching an end point does not count."""
        A, B, _, _ = self._segments()
        if not len(A):
            return False
        p, q  = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
        r, s  = q - p, B - A
        den   = _cross(r, s)
        ok    = np.abs(den) > 1e-12
        den   = np.where(ok, den, 1.0)
        ap    = A - p
        t     = _cross(ap, s) / den
        u     = _cross(ap, r) / den
        hit   = ok & (t > EPS) & (t < 1 - EPS) & (u > EPS) & (u < 1 - EPS)
        if skip:
            hit[list(skip)] = False
        return bool(hit.any())

    def inside(self, p) -> bool:
        """Is ``p`` inside the walls (always true for an open field)?"""
        if len(self.walls) < 3:
            return True
        x, y, pts, c = float(p[0]), float(p[1]), self.walls, False
        for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                c = not c
        return c

    # ─── Paths ───────────────────────────────────────────────────────────────

    def paths(self, src, dst, max_order: int = 1, diffraction: bool = True,
              direct: bool = True, wavelength: float = 0.8575) -> list[Path]:
        """Every path from ``src`` to ``dst``: direct (or, when blocked, round an
        obstacle corner) and reflections with up to ``max_order`` bounces.
        ``direct=False`` leaves out the direct and diffracted paths, as when the
        line of sight is blocked by something not in the world."""
        src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
        out = []
        if direct:
            if not self.blocked(src, dst):
                out.append(Path(np.array([src, dst]), "direct"))
            elif diffraction:
                out += self._diffracted(src, dst, wavelength)
        if max_order >= 1:
            out += self._reflected(src, dst, max_order)
        return out

    def _reflected(self, src, dst, max_order) -> list[Path]:
        A, B, refl, _ = self._segments()
        n, out = len(A), []
        for i in range(n):
            img = _mirror(src, A[i], B[i])
            p = self._hit(img, dst, i)
            if p is not None and not self.blocked(src, p, (i,)) and not self.blocked(p, dst, (i,)):
                out.append(Path(np.array([src, p, dst]), "reflect", refl[i]))
            if max_order < 2:
                continue
            for j in range(n):
                if j == i:
                    continue
                img2 = _mirror(img, A[j], B[j])
                p2 = self._hit(img2, dst, j)
                if p2 is None:
                    continue
                p1 = self._hit(img, p2, i)
                if p1 is None:
                    continue
                if (self.blocked(src, p1, (i,)) or self.blocked(p1, p2, (i, j))
                        or self.blocked(p2, dst, (j,))):
                    continue
                out.append(Path(np.array([src, p1, p2, dst]), "reflect", refl[i] * refl[j]))
        return out

    def _hit(self, img, dst, i):
        """Where the line img→dst crosses segment ``i`` strictly between its ends
        (the bounce point), or ``None``."""
        A, B, _, _ = self._segments()
        r, s = dst - img, B[i] - A[i]
        den = _cross(r, s)
        if abs(den) < 1e-12:
            return None
        ap = A[i] - img
        t, u = _cross(ap, s) / den, _cross(ap, r) / den
        if EPS < t < 1 - EPS and EPS < u < 1 - EPS:
            return img + t * r
        return None

    def _diffracted(self, src, dst, wavelength) -> list[Path]:
        """For each obstacle, the shortest path round it on either side: from
        ``src`` to a corner it can see, along the obstacle's edges, to a corner
        ``dst`` can see.  The loss is the sum of each corner's knife-edge loss
        (Epstein-Peterson)."""
        _, _, _, owner = self._segments()
        out = []
        for k, o in enumerate(self.obstacles):
            pts  = [np.asarray(p, dtype=float) for p in o.points]
            n    = len(pts)
            own  = tuple(np.where(owner == k)[0])
            see  = [not self.blocked(src, v) for v in pts]
            seen = [not self.blocked(v, dst) for v in pts]
            for step in (1, -1):
                best = None
                for i in range(n):
                    if not see[i]:
                        continue
                    walk = [i]
                    for _ in range(n):
                        j = walk[-1]
                        if seen[j]:
                            cand = np.array([src] + [pts[m] for m in walk] + [dst])
                            L = float(np.sum(np.hypot(*np.diff(cand, axis=0).T)))
                            if best is None or L < best[0]:
                                best = (L, cand)
                        walk.append((j + step) % n)
                if best is None:
                    continue
                cand = best[1]
                if any(self.blocked(a, b, own) for a, b in zip(cand[1:-2], cand[2:-1])):
                    continue
                loss = 0.0
                for a, v, b in zip(cand[:-2], cand[1:-1], cand[2:]):
                    line = b - a
                    L    = float(np.hypot(*line))
                    d1   = float(np.dot(v - a, line / L))
                    h    = abs(float(_cross(line / L, v - a)))
                    loss += knife_edge_db(h, max(d1, 1e-3), max(L - d1, 1e-3), wavelength)
                if not any(len(p.points) == len(cand) and np.allclose(p.points, cand) for p in out):
                    out.append(Path(cand, "diffract", 10.0 ** (-loss / 20.0)))
        return out


def default_walls(coords, margin: float = 25.0) -> list:
    """A rectangle ``margin`` cm outside the receivers."""
    c = np.asarray(coords, dtype=float)
    x0, y0 = (float(v) for v in c.min(axis=0) - margin)
    x1, y1 = (float(v) for v in c.max(axis=0) + margin)
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
