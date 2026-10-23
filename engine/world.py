"""
World geometry — the room walls and obstacles sound travels between, and the
paths it can take from a source to a receiver:

* ``direct``    straight line, if nothing blocks it; passing within the first
                Fresnel zone of an obstacle corner costs up to 6 dB (the
                knife-edge loss with the corner just clear of the line);
* ``reflect``   bouncing off walls or obstacles (image-source method, up to
                ``max_order`` bounces; each bounce keeps ``reflectivity`` of the
                amplitude);
* ``diffract``  bending round one corner of an obstacle that blocks the
                straight line, with the knife-edge loss from ``acoustics``.

All in 2D, centimetres.  Obstacles are convex polygons; walls are one closed
polygon around the arena (or none, for an open field).  Obstacles that touch
each other or a wall, along an edge or at a single corner, leave no gap: sound
can neither run along the seam nor slip through, or bend round, the point
where they touch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from engine.acoustics import knife_edge_db

EPS   = 1e-7
SIDE  = 0.01            # cm either side of a leg where a seam is probed for solid
V_MIN = -0.78           # knife-edge parameter below which a corner costs nothing


@dataclass
class Obstacle:
    points: list                    # convex polygon [(x, y), ...]
    reflectivity: float = 0.7       # amplitude kept on each bounce
    name: str = ""

    @staticmethod
    def rect(x0: float, y0: float, x1: float, y1: float, **kw) -> "Obstacle":
        xa, xb, ya, yb = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        return Obstacle([(xa, ya), (xb, ya), (xb, yb), (xa, yb)], **kw)

    @property
    def area(self) -> float:
        """cm² (shoelace formula)."""
        pts = np.asarray(self.points, dtype=float).reshape(-1, 2)
        return 0.5 * abs(float(np.sum(_cross(pts, np.roll(pts, -1, axis=0)))))

    def contains(self, p, strict: bool = False) -> bool:
        """Is ``p`` inside, or on the edge unless ``strict``?  An obstacle with
        no area contains nothing."""
        pts = np.asarray(self.points, dtype=float).reshape(-1, 2)
        nxt = np.concatenate((pts[1:], pts[:1]))
        if abs(float(_cross(pts, nxt).sum())) < 1e-9:
            return False
        s = _cross(nxt - pts, np.asarray(p, dtype=float) - pts)
        if strict:
            return bool(np.all(s > 1e-9) or np.all(s < -1e-9))
        return bool(np.all(s >= 0) or np.all(s <= 0))


@dataclass
class Path:
    points: np.ndarray              # (k, 2): source, bounce/corner points, receiver
    kind: str                       # "direct", "reflect" or "diffract"
    gain: float = 1.0               # reflection coefficients x diffraction loss (amplitude)

    @cached_property
    def legs(self) -> list[tuple[np.ndarray, float]]:
        """(unit direction of travel, length) of each straight leg."""
        out = []
        for a, b in zip(self.points[:-1], self.points[1:]):
            d = b - a
            n = float(np.hypot(*d))
            out.append((d / n if n > 0 else np.zeros(2), n))
        return out

    @cached_property
    def length(self) -> float:
        return float(sum(n for _, n in self.legs))

    @property
    def depart(self) -> np.ndarray:
        return self.legs[0][0]

    @property
    def arrive(self) -> np.ndarray:
        return self.legs[-1][0]


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


@dataclass
class World:
    walls: list = field(default_factory=list)       # closed polygon [(x, y), ...]; [] = open field
    wall_reflectivity: float = 0.8
    obstacles: list = field(default_factory=list)   # [Obstacle]

    def __post_init__(self):
        self._key = self._edges = None

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
            self._S     = self._B - self._A
            self._U     = self._S / np.hypot(*self._S.T).reshape(-1, 1)
            self._refl  = np.array(refl, dtype=float)
            self._owner = np.array(owner, dtype=int)
            self._key   = key
            self._edges = None
        return self._A, self._B, self._refl, self._owner

    @property
    def segment_count(self) -> int:
        return len(self._segments()[0])

    def blocked(self, p, q, skip: tuple = ()) -> bool:
        """Does the straight line p→q cross a wall or obstacle edge (other than
        segments in ``skip``)?  Touching an end point does not count, nor does
        running along an edge or through a corner, unless the line squeezes
        between two solids there (along a seam between touching obstacles or
        an obstacle and a wall, through a point where two solids touch, or
        corner to corner across an obstacle)."""
        p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
        mask = None
        if skip:
            mask = np.zeros((1, self.segment_count), dtype=bool)
            mask[0, list(skip)] = True
        return bool(self.blocked_legs(p[None], q[None], mask)[0])

    def blocked_legs(self, P, Q, skip=None) -> np.ndarray:
        """``blocked`` for many legs at once: ``P``, ``Q`` (m, 2), and ``skip``
        an (m, segments) mask of the segments each leg ignores, or ``None``."""
        A, _, _, _ = self._segments()
        P, Q = np.asarray(P, dtype=float).reshape(-1, 2), np.asarray(Q, dtype=float).reshape(-1, 2)
        if not len(A) or not len(P):
            return np.zeros(len(P), dtype=bool)
        r   = (Q - P)[:, None, :]
        s   = self._S
        ap  = A - P[:, None, :]
        den = _cross(r, s)
        cr  = _cross(ap, r)
        par = np.abs(den) <= 1e-12
        den = np.where(par, 1.0, den)
        t   = _cross(ap, s) / den
        u   = cr / den
        cut = ~par & (t > EPS) & (t < 1 - EPS) & (u > EPS) & (u < 1 - EPS)
        if skip is not None:
            cut &= ~skip
        out = cut.any(axis=1)
        # A corner on the line, or an edge along it: it may only graze, or the
        # line may squeeze between two solids there
        L  = np.hypot(r[..., 0], r[..., 1])
        on = ~out[:, None] & (np.abs(cr) <= 1e-6 * L) & (L > EPS)
        if skip is not None:
            on &= ~skip
        if on.any():
            rr    = L * L
            ta    = np.sum(ap * r, axis=2) / np.where(rr > 0, rr, 1.0)
            tb    = ta + np.sum(s * r, axis=2) / np.where(rr > 0, rr, 1.0)
            inner = (ta > EPS) & (ta < 1 - EPS)
            along = par & (np.minimum(ta, tb) < 1 - EPS) & (np.maximum(ta, tb) > EPS)
            for k in np.flatnonzero((on & (inner | along)).any(axis=1)):
                out[k] = self._squeezed(P[k], Q[k], ta[k, on[k]])
        return out

    def _squeezed(self, p, q, ts) -> bool:
        """Split p→q at ``ts`` (where corners lie on it) and test each piece's
        middle, and each corner, a hair either side: solid on both sides means
        the line runs between two solids (along a seam), through a point where
        two touch, or through one (corner to corner)."""
        d     = q - p
        side  = np.array([-d[1], d[0]]) / np.hypot(*d) * SIDE
        cuts  = np.unique(np.clip(np.r_[0.0, ts, 1.0], 0.0, 1.0))
        mids  = ((cuts[:-1] + cuts[1:]) / 2)[np.diff(cuts) > EPS]
        for t in np.r_[mids, cuts[(cuts > EPS) & (cuts < 1 - EPS)]]:
            m = p + d * t
            if self.solid(m + side) and self.solid(m - side):
                return True
        return False

    def solid(self, p) -> bool:
        """Is ``p`` inside an obstacle or outside the walls?"""
        return not self.inside(p) or any(o.contains(p) for o in self.obstacles)

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
                out.append(Path(np.array([src, dst]), "direct", self._clearance_gain(src, dst, wavelength)))
            elif diffraction:
                out += self._diffracted(src, dst, wavelength)
        if max_order >= 1:
            out += self._reflected(src, dst, max_order)
        return out

    def _clearance_gain(self, src, dst, wavelength) -> float:
        """Amplitude a clear straight path keeps when it passes within the first
        Fresnel zone of an obstacle corner: the knife-edge loss of the corner
        with negative clearance (0 dB well clear, 6 dB grazing it), so the level
        does not jump at the edge of a shadow."""
        A, _, _, owner = self._segments()
        C = A[owner >= 0]
        if not len(C):
            return 1.0
        line = dst - src
        L    = float(np.hypot(*line))
        if L < EPS:
            return 1.0
        rel  = C - src
        d1   = rel @ (line / L)
        near = (d1 > EPS) & (d1 < L - EPS)
        if not near.any():
            return 1.0
        h, d1 = np.abs(_cross(line / L, rel[near])), d1[near]
        v     = -h * np.sqrt(2.0 / wavelength * L / (d1 * (L - d1)))
        k     = int(np.argmax(v))
        if v[k] <= V_MIN:
            return 1.0
        return 10.0 ** (-knife_edge_db(-h[k], d1[k], L - d1[k], wavelength) / 20.0)

    def _reflected(self, src, dst, max_order) -> list[Path]:
        """Image-source reflections, every candidate bounce tested at once."""
        A, _, refl, _ = self._segments()
        n = len(A)
        if not n:
            return []
        seg  = np.arange(n)
        img  = _mirror_all(src[None], A, self._U)                    # src seen in each segment
        p1, ok = self._hits(img, np.broadcast_to(dst, (n, 2)), seg)
        k    = np.flatnonzero(ok)
        good = k[~self._any_blocked([(np.broadcast_to(src, (len(k), 2)), p1[k], k, None),
                                     (p1[k], np.broadcast_to(dst, (len(k), 2)), k, None)], n)]
        first = {int(i): Path(np.array([src, p1[i], dst]), "reflect", refl[i]) for i in good}
        second: dict = {}
        if max_order >= 2 and n > 1:
            I, J  = np.nonzero(~np.eye(n, dtype=bool))                 # every ordered pair i ≠ j
            img2  = _mirror_all(img[I], A[J], self._U[J])
            b2, ok = self._hits(img2, np.broadcast_to(dst, (len(I), 2)), J)
            I, J, b2 = I[ok], J[ok], b2[ok]
            b1, ok = self._hits(img[I], b2, I)
            I, J, b1, b2 = I[ok], J[ok], b1[ok], b2[ok]
            bad = self._any_blocked([(np.broadcast_to(src, (len(I), 2)), b1, I, None),
                                     (b1, b2, I, J), (b2, np.broadcast_to(dst, (len(I), 2)), J, None)], n)
            for i, j, q1, q2 in zip(I[~bad], J[~bad], b1[~bad], b2[~bad]):
                second.setdefault(int(i), []).append(
                    Path(np.array([src, q1, q2, dst]), "reflect", refl[i] * refl[j]))
        out = []
        for i in range(n):
            if i in first:
                out.append(first[i])
            out += second.get(i, [])
        return out

    def _any_blocked(self, legs, n) -> np.ndarray:
        """For paths made of the given legs ``[(P, Q, skip_i, skip_j)]`` (one row
        per path, each leg ignoring the segments it bounces off), is any leg blocked?"""
        rows = len(legs[0][0])
        if not rows:
            return np.zeros(0, dtype=bool)
        P = np.concatenate([np.asarray(p, dtype=float) for p, _, _, _ in legs])
        Q = np.concatenate([np.asarray(q, dtype=float) for _, q, _, _ in legs])
        skip = np.zeros((len(P), n), dtype=bool)
        for k, (_, _, si, sj) in enumerate(legs):
            at = np.arange(rows) + k * rows
            skip[at, si] = True
            if sj is not None:
                skip[at, sj] = True
        return self.blocked_legs(P, Q, skip).reshape(len(legs), rows).any(axis=0)

    def _hits(self, img, dst, seg):
        """Where each line ``img[k]``→``dst[k]`` crosses segment ``seg[k]``
        strictly between its ends (the bounce point), and whether it does."""
        A, B, _, _ = self._segments()
        r, s = dst - img, B[seg] - A[seg]
        den  = _cross(r, s)
        ok   = np.abs(den) >= 1e-12
        den  = np.where(ok, den, 1.0)
        ap   = A[seg] - img
        t, u = _cross(ap, s) / den, _cross(ap, r) / den
        ok  &= (t > EPS) & (t < 1 - EPS) & (u > EPS) & (u < 1 - EPS)
        return img + t[:, None] * r, ok

    def _diffracted(self, src, dst, wavelength) -> list[Path]:
        """For each obstacle, the shortest path round it on either side: from
        ``src`` to a corner it can see, along the obstacle's open edges, to a
        corner ``dst`` can see, never through a corner where it touches another
        solid.  The loss is the sum of each corner's knife-edge loss
        (Epstein-Peterson)."""
        corners = [np.asarray(o.points, dtype=float).reshape(-1, 2) for o in self.obstacles]
        if not corners:
            return []
        walkable = self._walkable()
        allc = np.concatenate(corners)
        m    = len(allc)
        see  = ~self.blocked_legs(np.broadcast_to(src, (m, 2)), allc)
        seen = ~self.blocked_legs(allc, np.broadcast_to(dst, (m, 2)))
        out, at = [], 0
        for k, pts in enumerate(corners):
            n, (pinched, edge_open) = len(pts), walkable[k]
            sk, sn = see[at:at + n] & ~pinched, seen[at:at + n]
            at += n
            for step in (1, -1):
                best = None
                for i in np.flatnonzero(sk):
                    walk = [i]
                    for _ in range(n):
                        j = walk[-1]
                        if sn[j]:
                            cand = np.array([src] + [pts[q] for q in walk] + [dst])
                            L = float(np.sum(np.hypot(*np.diff(cand, axis=0).T)))
                            if best is None or L < best[0]:
                                best = (L, cand)
                        nxt = (j + step) % n
                        if pinched[nxt] or not edge_open[j if step == 1 else nxt]:
                            break
                        walk.append(nxt)
                if best is None:
                    continue
                cand = best[1]
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

    def _walkable(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Per obstacle, ``(pinched, edge_open)``: which corners touch another
        obstacle or a wall, and which edges (corner k to k+1) nothing else lies
        along or across.  Sound bending round the obstacle can use neither a
        pinched corner nor a closed edge.  Kept until the geometry changes."""
        A, _, _, owner = self._segments()
        if self._edges is None:
            corners = [np.asarray(o.points, dtype=float).reshape(-1, 2) for o in self.obstacles]
            self._edges = []
            if corners:
                allc  = np.concatenate(corners)
                whose = np.repeat(np.arange(len(corners)), [len(c) for c in corners])
                lo    = np.array([c.min(axis=0) for c in corners]) - 1e-9
                hi    = np.array([c.max(axis=0) for c in corners]) + 1e-9
                near  = np.all((allc[:, None] >= lo) & (allc[:, None] <= hi), axis=2)   # in their boxes
                near[np.arange(len(allc)), whose] = False
                pinched = np.array([any(self.obstacles[j].contains(v) for j in np.flatnonzero(row))
                                    for v, row in zip(allc, near)], dtype=bool)
                wa, ws = A[owner < 0], self._S[owner < 0]
                if len(wa):
                    rel = allc[:, None] - wa[None]
                    t   = np.clip(np.sum(rel * ws, axis=2) / np.sum(ws * ws, axis=1), 0.0, 1.0)
                    pinched |= np.linalg.norm(rel - ws * t[..., None], axis=2).min(axis=1) < 1e-6
                nxt  = np.concatenate([np.roll(c, -1, axis=0) for c in corners])
                shut = self.blocked_legs(allc, nxt, owner[None, :] == whose[:, None])
                at   = np.cumsum([0] + [len(c) for c in corners])
                self._edges = [(pinched[a:b], ~shut[a:b]) for a, b in zip(at[:-1], at[1:])]
        return self._edges


def _mirror_all(P: np.ndarray, A: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Each point ``P[k]`` mirrored in the line through ``A[k]`` along unit ``U[k]``."""
    along = np.sum((P - A) * U, axis=1)[:, None]
    return 2.0 * (A + U * along) - P


def default_walls(coords, margin: float = 25.0) -> list:
    """A rectangle ``margin`` cm outside the receivers."""
    c = np.asarray(coords, dtype=float)
    x0, y0 = (float(v) for v in c.min(axis=0) - margin)
    x1, y1 = (float(v) for v in c.max(axis=0) + margin)
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
