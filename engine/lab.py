"""
Algorithm lab — records, step by step, what the positioning library does with
one frame, so the lab window can replay it.  Every number comes from the
library's own functions, called on a private ``UltraGPSPositionLib`` instance;
the library code is not changed, only watched:

* ``trace_filter``  the sanity filter: ticks → distances, bounds, jump check
                    (``_ticks_to_distances``, ``_filter_receivers``)
* ``trace_cep``     every subset ``_find_best_subset`` evaluates, in its order,
                    with the seed, fit, residuals, weights, covariance and CEP
                    ``_compute_cep`` produced, then ``_validate_position``
* ``trace_lm``      every point the library's ``_nlls`` (scipy's MINPACK
                    Levenberg-Marquardt) evaluates on its way from an initial
                    guess to the fix
* ``sweep``         how each algorithm's fix moves as one receiver's reading
                    goes wrong (e.g. blocked, reading a reflection)

Watching works by wrapping the instance's methods, and the module's
``least_squares`` for the LM trace, for the duration of one call.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

import ultragps_position.core as core
from engine import solvers as S
from ultragps_position import UltraGPSPositionLib


# ─── Sanity filter ───────────────────────────────────────────────────────────

@dataclass
class FilterRow:
    receiver: int
    tick: int
    slope: float
    intercept: float
    distance: float
    in_bounds: bool        # 0 < distance <= top band
    previous: float | None # last frame's distance (None on the first frame)
    jump: float | None
    kept: bool


@dataclass
class FilterTrace:
    rows: list
    top_band: float
    max_differential: float
    sane: list


def trace_filter(lib: UltraGPSPositionLib, ticks) -> FilterTrace:
    """Run the library's filter on ``ticks`` and explain each receiver's verdict."""
    prev = None if lib._last_distances is None else lib._last_distances.copy()
    top  = lib.max_distance if lib.max_distance is not None else lib._max_receiver_dist
    d    = lib._ticks_to_distances(ticks)
    sane = [int(i) for i in lib._filter_receivers(d)]
    rows = []
    for i, dist in enumerate(d):
        off  = lib._offsets[i]
        jump = None if prev is None else abs(dist - prev[i])
        rows.append(FilterRow(i, int(ticks[i]), off["slope"], off["intercept"], float(dist),
                              bool(0 < dist <= top), None if prev is None else float(prev[i]),
                              jump, i in sane))
    return FilterTrace(rows, float(top), float(lib.max_differential), sane)


# ─── CEP subset search ───────────────────────────────────────────────────────

@dataclass
class SubsetStep:
    subset: list
    seed: np.ndarray        # _ols
    pos: np.ndarray         # _nlls
    residuals: np.ndarray   # |p − rᵢ| − dᵢ for the subset
    weights: np.ndarray     # 1 / (|residual| + 0.1), the library's weighting
    cov: np.ndarray         # (JᵀWJ)⁻¹
    cep: float              # 0.59 · sqrt(σx² + σy²) · √2
    best: bool              # best so far when it was evaluated


@dataclass
class CepTrace:
    distances: np.ndarray
    sane: list
    steps: list
    best: int | None        # index into steps of the winner
    position: np.ndarray | None   # after _validate_position
    rejected: bool          # winner outside the receivers' box
    box: tuple


@contextmanager
def _watch(lib, name, wrapper):
    """Shadow ``lib.name`` with ``wrapper(original, *args)`` for the duration."""
    orig = getattr(lib, name)
    setattr(lib, name, lambda *a, **k: wrapper(orig, *a, **k))
    try:
        yield
    finally:
        delattr(lib, name)


def trace_cep(lib: UltraGPSPositionLib, distances, sane: list, max_subsets: int | None = None) -> CepTrace:
    """Run ``_find_best_subset`` and record every subset it scores."""
    n           = lib.receiver_count
    max_subsets = max_subsets or 2 ** n
    d           = tuple(float(v) for v in distances)
    steps, fits = [], []

    def nlls(orig, dist, initial, idx):
        pos, res = orig(dist, initial, idx)
        fits.append((np.array(initial, dtype=float), res))
        return pos, res

    def compute(orig, dist, idx):
        fits.clear()
        pos, cep, cov = orig(dist, idx)
        seed, res = fits[-1]
        best = cep < min((s.cep for s in steps), default=float("inf"))   # the library's own test
        steps.append(SubsetStep(list(idx), seed, np.array(pos, dtype=float), np.array(res.fun),
                                1.0 / (np.abs(res.fun) + 0.1), np.array(cov), float(cep), best))
        return pos, cep, cov

    with _watch(lib, "_nlls", nlls), _watch(lib, "_compute_cep", compute):
        best_pos, best_cep, best_idx, _ = lib._find_best_subset(d, list(sane), max_subsets)
    pos, invalid = lib._validate_position(best_pos)
    c   = lib._receiver_coords
    box = (float(c[:, 0].min()), float(c[:, 1].min()), float(c[:, 0].max()), float(c[:, 1].max()))
    best = next((k for k, s in enumerate(steps) if s.subset == best_idx), None)
    return CepTrace(np.array(d), list(sane), steps, best,
                    None if pos is None else np.array(pos, dtype=float), bool(invalid), box)


# ─── Levenberg-Marquardt path ────────────────────────────────────────────────

@dataclass
class LMTrace:
    initial: np.ndarray
    final: np.ndarray
    points: list            # every evaluation [(xy, rms residual, kind)]: "start", "step", "rejected", "jacobian"
    path: list              # start + accepted steps
    message: str
    nfev: int


_lsq_lock     = threading.Lock()
_lsq_original = core.least_squares


def trace_lm(lib: UltraGPSPositionLib, distances, indices, initial=None) -> LMTrace:
    """Run the library's ``_nlls`` from ``initial`` (default: its OLS seed) and
    record each point the optimiser evaluates."""
    d    = np.asarray(distances, dtype=float)
    idx  = np.asarray(indices)
    x0   = lib._ols(d, idx) if initial is None else np.asarray(initial, dtype=float)
    seen = []
    me   = threading.get_ident()

    def recording(fun, x0_, *a, **k):
        if threading.get_ident() != me:            # another thread: leave it alone
            return _lsq_original(fun, x0_, *a, **k)
        def f(x):
            r = fun(x)
            seen.append((np.array(x, dtype=float), float(np.sqrt(np.mean(np.square(r))))))
            return r
        return _lsq_original(f, x0_, *a, **k)

    with _lsq_lock:
        core.least_squares = recording
        try:
            pos, result = lib._nlls(d, x0, idx)
        finally:
            core.least_squares = _lsq_original

    points, path = [], []
    cur, cur_cost = None, None
    for x, cost in seen:
        if cur is None:
            cur, cur_cost = x, cost
            points.append((x, cost, "start"))
            path.append(x)
        elif np.hypot(*(x - cur)) < 1e-6 * (1.0 + np.hypot(*cur)):    # a finite-difference probe
            points.append((x, cost, "jacobian"))
        elif cost <= cur_cost:
            cur, cur_cost = x, cost
            points.append((x, cost, "step"))
            path.append(x)
        else:
            points.append((x, cost, "rejected"))
    final = np.array(pos, dtype=float)
    if not path or np.hypot(*(path[-1] - final)) > 1e-6:
        path.append(final)
    return LMTrace(x0, final, points, path, str(result.message), int(result.nfev))


def cost_grid(coords, distances, indices, bounds, n: int = 140):
    """RMS range residual (cm) of a fix at each point of a grid over ``bounds``:
    the surface LM walks down.  Returns ``(xs, ys, Z)`` with ``Z[j, i]`` at (xs[i], ys[j])."""
    x0, y0, x1, y1 = bounds
    xs  = np.linspace(x0, x1, n)
    ys  = np.linspace(y0, y1, int(n * (y1 - y0) / max(x1 - x0, 1e-9)) or n)
    X, Y = np.meshgrid(xs, ys)
    c   = np.asarray(coords, dtype=float)[list(indices)]
    dd  = np.asarray(distances, dtype=float)[list(indices)]
    r   = np.hypot(X[..., None] - c[:, 0], Y[..., None] - c[:, 1]) - dd
    return xs, ys, np.sqrt(np.mean(r ** 2, axis=-1))


# Marching-squares edge pairs per case (corners: 0 bottom-left, 1 bottom-right,
# 2 top-right, 3 top-left; edges: 0 bottom, 1 right, 2 top, 3 left)
_CASES = {1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)], 5: [(3, 2), (0, 1)], 6: [(0, 2)],
          7: [(3, 2)], 8: [(2, 3)], 9: [(2, 0)], 10: [(2, 1), (3, 0)], 11: [(2, 1)], 12: [(1, 3)],
          13: [(1, 0)], 14: [(0, 3)]}


def contour(xs, ys, Z, level: float) -> np.ndarray:
    """Line segments ``(k, 2, 2)`` of the ``level`` iso-line of ``Z`` (marching squares)."""
    a, b, c, d = Z[:-1, :-1], Z[:-1, 1:], Z[1:, 1:], Z[1:, :-1]
    case = ((a > level) * 1 + (b > level) * 2 + (c > level) * 4 + (d > level) * 8)
    X0, X1 = xs[:-1][None, :], xs[1:][None, :]
    Y0, Y1 = ys[:-1][:, None], ys[1:][:, None]

    def lerp(p, q, v0, v1):
        t = np.clip((level - v0) / np.where(v1 - v0 == 0, 1e-12, v1 - v0), 0, 1)
        return p + t * (q - p)

    with np.errstate(invalid="ignore"):
        edges = [
            (lerp(X0, X1, a, b), np.broadcast_to(Y0, a.shape)),        # bottom
            (np.broadcast_to(X1, a.shape), lerp(Y0, Y1, b, c)),        # right
            (lerp(X0, X1, d, c), np.broadcast_to(Y1, a.shape)),        # top
            (np.broadcast_to(X0, a.shape), lerp(Y0, Y1, a, d)),        # left
        ]
    segs = []
    for k, pairs in _CASES.items():
        m = case == k
        if not m.any():
            continue
        for e0, e1 in pairs:
            p = np.stack([edges[e0][0][m], edges[e0][1][m]], axis=1)
            q = np.stack([edges[e1][0][m], edges[e1][1][m]], axis=1)
            segs.append(np.stack([p, q], axis=1))
    return np.concatenate(segs) if segs else np.empty((0, 2, 2))


# ─── One receiver going wrong ────────────────────────────────────────────────

@dataclass
class Sweep:
    receiver: int
    offsets: np.ndarray                 # cm added to its reading
    paths: dict                         # {algorithm: (k, 2) fixes, nan where none}
    cep_subsets: list                   # subset the library's CEP chose at each offset
    cep_rejected: list                  # library CEP fix outside the box


def static_solve(lib: UltraGPSPositionLib, d, idx, settings: S.Settings | None = None) -> dict:
    """Every snapshot algorithm on distances ``d`` from receivers ``idx``, with
    no memory between calls (the EKF is left out: it needs a history)."""
    settings = settings or S.Settings()
    d   = np.asarray(d, dtype=float)
    idx = list(idx)
    out = {}
    if len(idx) >= 3:
        out["OLS"] = S.Estimate(lib._ols(d, np.array(idx)), True, idx)
        pos, r = lib._multilaterate(tuple(d), np.array(idx))
        out["LM"] = S.Estimate(np.array(pos), bool(r.success), idx)
        best_pos, cep, best_idx, cov = lib._find_best_subset(tuple(d), idx, 2 ** lib.receiver_count)
        c = lib._receiver_coords
        inside = (best_pos is not None and c[:, 0].min() <= best_pos[0] <= c[:, 0].max()
                  and c[:, 1].min() <= best_pos[1] <= c[:, 1].max())
        out["CEP"] = S.Estimate(np.array(best_pos), bool(inside), list(best_idx or []), cov,
                                f"CEP {cep:.1f} cm" + ("" if inside else ", outside the box"))
    out["CEP4"]   = S.solve_cep4(lib, d, idx)
    out["ROBUST"] = S.solve_robust(lib, d, idx, settings.robust_scale_cm)
    out["BIAS"]   = S.solve_bias(lib, d, idx)
    out["RANSAC"] = S.solve_ransac(lib, d, idx, settings.ransac_tol_cm)
    return out


def sweep(lib: UltraGPSPositionLib, distances, idx, receiver: int, max_offset: float = 150.0,
          steps: int = 61, settings: S.Settings | None = None) -> Sweep:
    """Add 0..``max_offset`` cm to one receiver's reading (as when it hears a
    reflection instead of the blocked direct sound) and solve at each step."""
    offsets = np.linspace(0.0, max_offset, steps)
    paths, subsets, rejected = {}, [], []
    for k, off in enumerate(offsets):
        d = np.array(distances, dtype=float)
        d[receiver] += off
        res = static_solve(lib, d, idx, settings)
        for a, e in res.items():
            paths.setdefault(a, np.full((steps, 2), np.nan))
            if e.pos is not None:
                paths[a][k] = e.pos
        cep = res.get("CEP")
        subsets.append(cep.used if cep else [])
        rejected.append(bool(cep and not cep.ok))
    return Sweep(receiver, offsets, paths, subsets, rejected)
