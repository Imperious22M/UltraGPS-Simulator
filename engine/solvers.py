"""
Solvers — every positioning algorithm, run on the same frame of ticks.

From the real ``ultragps_position`` library, unchanged, called as
UltraGPS-Control's position window calls it (``get_position_full``):

  OLS     the linear least-squares seed (``_ols``)
  LM      OLS seed refined by Levenberg-Marquardt
  CEP     the receiver subset with the smallest Circular Error Probable.  With
          ``Settings.cep_subsets = "control"`` (the default) the search stops
          after 15 subsets, the library default UltraGPS-Control uses: all the
          kept receivers, then the first 14 three-receiver subsets.  With
          ``"all"`` every subset of 3 or more kept receivers is tried (42 with
          six).  The library's matrix (JᵀWJ)⁻¹ is a ranking score, not a
          position covariance: it is kept as ``Estimate.score_cov``, and
          ``Estimate.cov`` is the winning subset's geometry scaled by the noise
          LM sees over every kept receiver.

Simulator-only algorithms, for comparison.  They are built from the library's
own ``_ols`` / ``_nlls`` and range equation ``|p − rᵢ| = dᵢ`` where they can be,
and never modify it:

  CEP4    CEP over subsets of 4 or more receivers, textbook CEP score, no
          bounding-box rejection
  ROBUST  one least-squares fit over all receivers with a soft-L1 loss, so
          a bad reading pulls the fix far less than in plain LM
  BIAS    LM that also solves for a range offset common to every receiver
          (x, y, b): absorbs radio/trigger timing error; needs 4 receivers
  RANSAC  fit every 3-receiver subset, keep the one most other receivers agree
          with (within ``ransac_tol_cm``), refit on those
  EKF     extended Kalman filter tracking position and velocity; uses each
          range on its own, rejects ranges far from the prediction, and keeps
          tracking with fewer than 3 receivers

All of them see the receivers the library's filter kept (bounds + jump).
With ``Settings.filter_mode = "bounds"``, the simulator algorithms use every
receiver within bounds instead, so their own outlier handling does the work.
The EKF always uses bounds only: its gate replaces the jump filter.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares

from ultragps_position import UltraGPSPositionLib

_REPO          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(_REPO, "config", "config.toml")


@dataclass(frozen=True)
class AlgoInfo:
    key: str
    label: str
    source: str             # "library" or "simulator"
    summary: str


ALGORITHMS = {a.key: a for a in (
    AlgoInfo("OLS",    "Linear least squares",       "library",   "Linearised ranges, one matrix solve. The seed for LM and CEP."),
    AlgoInfo("LM",     "Levenberg-Marquardt",        "library",   "OLS seed refined by non-linear least squares over every kept receiver."),
    AlgoInfo("CEP",    "CEP best subset",            "library",   "Subsets of 3+ kept receivers (the first 15, as UltraGPS-Control, or all); smallest Circular Error Probable wins; must lie in the receivers' box."),
    AlgoInfo("CEP4",   "CEP, 4+ receivers",          "simulator", "Subsets of 4+ only, textbook CEP from s²(JᵀJ)⁻¹, no bounding-box rejection."),
    AlgoInfo("ROBUST", "Robust LM (soft-L1)",        "simulator", "One fit with a loss that grows linearly for large residuals, so outliers count for less."),
    AlgoInfo("BIAS",   "LM + common offset",         "simulator", "Solves x, y and a range offset shared by every receiver (trigger timing). Needs 4."),
    AlgoInfo("RANSAC", "RANSAC consensus",           "simulator", "Every 3-receiver fit votes; the one most receivers agree with is refitted on them."),
    AlgoInfo("EKF",    "Kalman tracker",             "simulator", "Tracks position and velocity; gates each range against the prediction; coasts through gaps."),
)}
LIBRARY_ALGOS = [k for k, a in ALGORITHMS.items() if a.source == "library"]


@dataclass
class Estimate:
    pos: np.ndarray | None          # None: nothing to show
    ok: bool                        # solved this frame (False: failed or holding a previous fix)
    used: list = field(default_factory=list)
    cov: np.ndarray | None = None   # 2×2 position covariance, cm²
    note: str = ""
    score_cov: np.ndarray | None = None   # CEP only: the library's (JᵀWJ)⁻¹, a ranking score, not a covariance


# The library's own default, which UltraGPS-Control's get_position_full call uses
CONTROL_MAX_SUBSETS = inspect.signature(UltraGPSPositionLib.get_position_full).parameters["max_subsets"].default


@dataclass
class Settings:
    filter_mode: str      = "library"   # receivers the simulator algorithms use: "library" or "bounds"
    cep_subsets: str      = "control"   # library CEP search: "control" (first 15, as UltraGPS-Control) or "all"
    robust_scale_cm: float = 5.0        # soft-L1 knee
    ransac_tol_cm: float  = 8.0         # a receiver agrees with a fit within this
    ekf_accel: float      = 150.0       # process noise: random acceleration, cm/s²
    ekf_sigma_cm: float   = 2.0         # range noise the EKF assumes
    ekf_gate: float       = 3.5         # reject a range more than this many σ from the prediction


@dataclass
class Solution:
    distances: list                 # every receiver, from the library's calibration
    sane: list                      # kept by the library filter (bounds + jump)
    inbounds: list                  # within the library's bounds only
    estimates: dict                 # {algorithm: Estimate}


# ─── Building blocks (stateless) ─────────────────────────────────────────────

def cep_max_subsets(settings: Settings, n_receivers: int) -> int:
    """``max_subsets`` for the library's CEP search under ``settings.cep_subsets``."""
    return 2 ** n_receivers if settings.cep_subsets == "all" else CONTROL_MAX_SUBSETS


def ranges_from(coords: np.ndarray, pos) -> np.ndarray:
    return np.linalg.norm(coords - np.asarray(pos, dtype=float), axis=1)


def noise_var(residuals: np.ndarray | None, unknowns: int = 2, floor_cm: float = 1.0) -> float | None:
    """s² of a fit with ``unknowns`` parameters, at least ``floor_cm``²
    (None without more residuals than unknowns)."""
    if residuals is None or len(residuals) <= unknowns:
        return None
    return max(float(np.sum(np.square(residuals))) / (len(residuals) - unknowns), floor_cm ** 2)


def covariance(coords: np.ndarray, pos, residuals: np.ndarray | None, floor_cm: float = 1.0,
               offset: bool = False, s2: float | None = None) -> np.ndarray | None:
    """x, y block of s²(JᵀJ)⁻¹ of a range fit at ``pos``, with s² from the
    residuals unless given.  ``offset``: the fit also solved for a range
    offset shared by every receiver, a third unknown."""
    if s2 is None:
        s2 = noise_var(residuals, 3 if offset else 2, floor_cm)
    if pos is None or s2 is None:
        return None
    diff = np.asarray(pos, dtype=float) - coords
    J    = diff / np.maximum(np.linalg.norm(diff, axis=1), 1e-9)[:, None]
    if offset:
        J = np.c_[J, np.ones(len(J))]
    try:
        return s2 * np.linalg.inv(J.T @ J)[:2, :2]
    except np.linalg.LinAlgError:
        return None


def cep_from_cov(cov: np.ndarray | None) -> float:
    if cov is None:
        return float("inf")
    return 0.59 * (np.sqrt(abs(cov[0, 0])) + np.sqrt(abs(cov[1, 1])))


def lm_fit(coords: np.ndarray, d: np.ndarray, x0) -> tuple[np.ndarray, np.ndarray]:
    """Levenberg-Marquardt on the library's range equation ``|p − rᵢ| − dᵢ``,
    with an analytic Jacobian: lands on the same minimum as the library's
    ``_nlls`` (scipy/MINPACK) about 30 times faster, for the subset searches."""
    x    = np.asarray(x0, dtype=float).copy()
    r    = np.hypot(*(x - coords).T) - d
    cost = float(r @ r)
    lam  = 1e-3
    for _ in range(100):
        diff = x - coords
        J    = diff / np.maximum(np.hypot(*diff.T), 1e-12)[:, None]
        A, g = J.T @ J, J.T @ r
        while True:
            step = -np.linalg.solve(A + lam * np.diag(np.diag(A)) + 1e-12 * np.eye(2), g)
            xn   = x + step
            rn   = np.hypot(*(xn - coords).T) - d
            cn   = float(rn @ rn)
            if cn <= cost:
                x, r, cost, lam = xn, rn, cn, max(lam / 10.0, 1e-12)
                break
            lam *= 10.0
            if lam > 1e12:
                return x, r
        if np.hypot(*step) < 1e-9 * (1.0 + np.hypot(*x)):
            break
    return x, r


def fit_lm(lib, d: np.ndarray, idx) -> tuple[np.ndarray, np.ndarray]:
    """The library's OLS seed, then LM on receivers ``idx``: (position, residuals)."""
    idx = np.asarray(idx)
    return lm_fit(lib._receiver_coords[idx], d[idx], lib._ols(d, idx))


def solve_cep4(lib, d, idx) -> Estimate:
    best = None
    for k in range(4, len(idx) + 1):
        for sub in combinations(idx, k):
            pos, res = fit_lm(lib, d, sub)
            cov = covariance(lib._receiver_coords[list(sub)], pos, res, floor_cm=0.0)
            cep = cep_from_cov(cov)
            if best is None or cep < best[0]:
                best = (cep, pos, list(sub), cov)
    if best is None:
        return Estimate(None, False, note="needs 4 receivers")
    return Estimate(best[1], True, best[2], best[3], f"CEP {best[0]:.1f} cm")


def solve_robust(lib, d, idx, scale: float) -> Estimate:
    if len(idx) < 3:
        return Estimate(None, False, note="needs 3 receivers")
    idx    = list(idx)
    coords = lib._receiver_coords[idx]
    dd     = d[idx]
    r      = least_squares(lambda p: np.linalg.norm(coords - p, axis=1) - dd,
                           lib._ols(d, np.array(idx)), loss="soft_l1", f_scale=scale)
    res    = r.fun
    used   = [i for i, e in zip(idx, res) if abs(e) < 3 * scale]
    keep   = np.abs(res) < 3 * scale
    cov    = covariance(coords[keep], r.x, res[keep]) if keep.sum() >= 3 else None
    out    = [f"R{i}" for i, e in zip(idx, res) if abs(e) >= 3 * scale]
    return Estimate(r.x, bool(r.success), used, cov,
                    "down-weighted " + ", ".join(out) if out else "all consistent")


MAX_OFFSET_CM = 100.0     # a common range offset larger than this means the ranges disagree, not a timing error


def solve_bias(lib, d, idx) -> Estimate:
    if len(idx) < 4:
        return Estimate(None, False, note="needs 4 receivers")
    idx    = list(idx)
    coords = lib._receiver_coords[idx]
    dd     = d[idx]
    x0     = np.r_[lib._ols(d, np.array(idx)), 0.0]
    r      = least_squares(lambda v: np.linalg.norm(coords - v[:2], axis=1) + v[2] - dd, x0, method="lm")
    if not np.all(np.isfinite(r.x)) or abs(r.x[2]) > MAX_OFFSET_CM:
        # Ranges that disagree badly (an echo, a blocked path) can be "explained"
        # by a huge common offset and a position far away: not a fix.
        return Estimate(None, False, idx, note=f"offset {r.x[2]:+.0f} cm: not plausible")
    cov    = covariance(coords, r.x[:2], r.fun, offset=True) if len(idx) > 4 else None
    return Estimate(r.x[:2], bool(r.success), idx, cov, f"offset {r.x[2]:+.1f} cm")


def solve_ransac(lib, d, idx, tol: float) -> Estimate:
    if len(idx) < 3:
        return Estimate(None, False, note="needs 3 receivers")
    idx    = list(idx)
    coords = lib._receiver_coords[idx]
    best   = None
    for sub in combinations(idx, 3):
        pos, _ = fit_lm(lib, d, sub)
        res    = np.abs(ranges_from(coords, pos) - d[idx])
        inl    = [i for i, e in zip(idx, res) if e < tol]
        score  = (len(inl), -float(np.sqrt(np.mean(res[res < tol] ** 2))) if inl else 0.0)
        if best is None or score > best[0]:
            best = (score, inl)
    inliers = best[1]
    if len(inliers) < 3:
        return Estimate(None, False, note="no 3 receivers agree")
    pos, res = fit_lm(lib, d, inliers)
    return Estimate(pos, True, inliers, covariance(lib._receiver_coords[inliers], pos, res),
                    f"{len(inliers)} of {len(idx)} agree")


class Tracker:
    """Constant-velocity extended Kalman filter on raw ranges."""

    def __init__(self, coords: np.ndarray, settings: Settings):
        self.coords   = coords
        self.settings = settings
        self.reset()

    def reset(self) -> None:
        self.x, self.P, self.misses = None, None, 0

    def step(self, lib, d: np.ndarray, idx: list, dt: float) -> Estimate:
        s = self.settings
        if self.x is None:
            if len(idx) < 3:
                return Estimate(None, False, note="waiting for 3 receivers")
            pos, _ = fit_lm(lib, d, idx)
            self.x = np.r_[pos, 0.0, 0.0]
            self.P = np.diag([25.0, 25.0, 100.0 ** 2, 100.0 ** 2])
            return Estimate(pos.copy(), True, list(idx), self.P[:2, :2].copy(), "started")

        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * s.ekf_accel ** 2

        R, used, gated = s.ekf_sigma_cm ** 2, [], []
        for i in idx:
            diff = self.x[:2] - self.coords[i]
            h    = float(np.hypot(*diff))
            if h < 1e-6:
                continue
            H = np.r_[diff / h, 0.0, 0.0]
            y = float(d[i] - h)
            S = float(H @ self.P @ H + R)
            if y * y > s.ekf_gate ** 2 * S:
                gated.append(i)
                continue
            K      = self.P @ H / S
            self.x = self.x + K * y
            self.P = (np.eye(4) - np.outer(K, H)) @ self.P
            used.append(i)

        # Lost lock: every range keeps disagreeing, so start again from a fresh fix
        self.misses = self.misses + 1 if (gated and not used) else 0
        if self.misses >= 5 and len(idx) >= 3:
            self.reset()
            return self.step(lib, d, idx, dt)

        note = "gated " + ", ".join(f"R{i}" for i in gated) if gated else (
            "coasting" if not used else f"{len(used)} ranges")
        return Estimate(self.x[:2].copy(), bool(used), used, self.P[:2, :2].copy(), note)


# ─── Per-frame driver ────────────────────────────────────────────────────────

class Solvers:
    def __init__(self, config_path: str = DEFAULT_CONFIG, max_differential: float = 50.0,
                 max_distance: float | None = None, settings: Settings | None = None):
        self.lib      = UltraGPSPositionLib(config_path, max_differential=max_differential,
                                            max_distance=max_distance)
        self.settings = settings or Settings()
        self.tracker  = Tracker(self.lib._receiver_coords, self.settings)
        self.held: dict = {}      # last good fix of each simulator algorithm

    @property
    def max_subsets(self) -> int:
        """The library CEP's ``max_subsets`` under the current settings."""
        return cep_max_subsets(self.settings, self.lib.receiver_count)

    @property
    def coords(self) -> np.ndarray:
        return self.lib._receiver_coords

    @property
    def top_band(self) -> float:
        lib = self.lib
        return lib.max_distance if lib.max_distance is not None else lib._max_receiver_dist

    def reset(self) -> None:
        self.lib.reset_state()
        self.tracker.reset()
        self.held.clear()

    def reload(self) -> None:
        """Re-read the calibration in config.toml."""
        self.lib.reload()
        self.tracker.coords = self.lib._receiver_coords
        self.reset()

    def solve(self, ticks: list, dt: float = 0.1, only: set | None = None) -> Solution:
        """Run every algorithm (or just those in ``only``) on one frame."""
        lib  = self.lib
        r    = lib.get_position_full(ticks, max_subsets=self.max_subsets)
        d    = np.array(r["distances"], dtype=float)
        sane = [int(i) for i in r["sane_indices"]]
        inb  = [int(i) for i in np.where((d > 0) & (d <= self.top_band))[0]]
        idx  = sane if self.settings.filter_mode == "library" else inb
        s    = self.settings
        want = (lambda k: True) if only is None else (lambda k: k in only)

        est = {}
        if len(sane) >= 3:
            ols = lib._ols(d, np.array(sane))
            est["OLS"] = Estimate(ols, True, sane, covariance(self.coords[sane], ols,
                                  ranges_from(self.coords[sane], ols) - d[sane]))
        else:
            est["OLS"] = Estimate(None, False, note="needs 3 receivers")
        lm     = r["lm_position"]
        lm_res = ranges_from(self.coords[sane], lm) - d[sane] if lm is not None and len(sane) >= 3 else None
        est["LM"] = Estimate(lm, bool(r["lm_success"]), sane if r["lm_success"] else [],
                             covariance(self.coords[sane], lm, lm_res),
                             f"rms {r['lm_rms']:.1f} cm" if r["lm_rms"] is not None else
                             ("holding last fix" if lm is not None else "needs 3 receivers"))
        est["CEP"] = self._cep(r, lm_res)

        if want("CEP4"):
            est["CEP4"] = solve_cep4(lib, d, idx)
        if want("ROBUST"):
            est["ROBUST"] = solve_robust(lib, d, idx, s.robust_scale_cm)
        if want("BIAS"):
            est["BIAS"] = solve_bias(lib, d, idx)
        if want("RANSAC"):
            est["RANSAC"] = solve_ransac(lib, d, idx, s.ransac_tol_cm)
        if want("EKF"):
            est["EKF"] = self.tracker.step(lib, d, inb, dt)

        # Like the library's solvers, the others hold their last good fix when they can't solve
        for k, e in est.items():
            if k in LIBRARY_ALGOS:
                continue
            if e.ok and e.pos is not None:
                self.held[k] = np.asarray(e.pos, dtype=float).copy()
            elif e.pos is None and k in self.held:
                est[k] = Estimate(self.held[k], False, [], None, e.note + ": holding last fix")
        return Solution(list(d), sane, inb, est)

    def _cep(self, r: dict, lm_res: np.ndarray | None) -> Estimate:
        """The library's CEP fix.  Its ellipse: the winning subset's geometry,
        with the noise level s² from LM's residuals over every kept receiver (a
        3-receiver subset fits almost exactly, so its own residuals say nothing).
        When the CEP dropped a bad reading, LM's residuals still hold it, so the
        ellipse grows: the readings disagree and a 3-receiver pick can be wrong."""
        if r["cep_success"]:
            best = list(r["best_indices"] or [])
            cov  = covariance(self.coords[best], r["cep_position"], None, s2=noise_var(lm_res))
            return Estimate(r["cep_position"], True, best, cov, f"CEP {r['cep']:.1f} cm", r["cov"])
        if len(r["sane_indices"]) < 3:
            note = "holding last fix" if r["cep_position"] is not None else "needs 3 receivers"
        elif self.lib._cep_last_good_pos is None:
            note = "outside the box: rejected (no earlier fix to hold)"
        else:
            note = "outside the box: holding last fix"
        return Estimate(r["cep_position"], False, [], None, note)
