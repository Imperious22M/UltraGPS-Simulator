"""
Calibration-point search — finds the set of 2, 3 or 4 calibration points that
gives the most accurate per-receiver slope and intercept.

    python3 -m engine.cal_points [--max-points 4] [--margin 30] [--noise 1] [--trials 200] [--apply]

Why the choice matters
----------------------
Calibration fits ``distance = slope * tick + intercept`` for each receiver from
one modal tick per calibration point.  Each modal tick is off by a little
(tick rounding plus noise).  With points at distances ``d_j`` from a receiver,
the least-squares fit's error is

    slope error     ∝ 1 / sqrt(Sxx)                     Sxx = Σ (d_j - d̄)²
    intercept error ∝ sqrt(1/n + d̄² / Sxx)
    distance error at D ∝ sqrt(1/n + (D - d̄)² / Sxx)

so a receiver wants its calibration points at very different distances from
it (large Sxx), centred near the distances it will actually measure.  Six
receivers pull in different directions, so the search scores a point set by
the RMS distance error it would cause over the whole arena, for all receivers:

    score = sqrt(mean over receivers and arena positions of 1/n + (D - d̄)² / Sxx)

i.e. the calibration-induced distance error per unit of per-reading error.
That single number combines slope and intercept error in the units that
matter (cm of distance).  Lower is better.  It depends only on geometry, so
the ranking holds for any noise level.

Search
------
Candidates are a grid over the receivers' bounding box, shrunk by ``margin``
cm on every side, keeping only points within the HC-SR04's 400 cm of every
receiver (otherwise a receiver can't be calibrated there).  The margin keeps
points away from the receiver walls: a point level with a wall of receivers is
far off their beams, where readings are weak and late.  Given the room, only
points inside the walls and outside the obstacles are candidates, and a set
costs ``BLOCKED_PENALTY`` extra for every point a receiver that is on can't
see: a receiver calibrated on an echo is wrong everywhere, so a set in every
receiver's line of sight wins whenever there is one.  No two points of a set
are closer than ``MIN_SEPARATION_CM``: the same spot read twice repeats its
range walk and adds nothing.  2 points: every pair.  3 and 4 points: greedy
from the best smaller set, then exchange passes (swap one point for any
candidate while that improves the score), then a fine-grid refinement.

Check
-----
The best sets are then calibrated many times on the simulated sensors, with
the same "collect until the modal tick is seen ``calibration_reads`` times"
procedure as the calibration library, fitted by least squares (the library's
own fit), and compared with the sensors' true response.  The largest best set
is also run once through the calibration library as a cross-check.

``--apply`` writes the best ``--max-points`` set into config.toml
(``cal_points`` and each receiver's ``cal_distances``).  The simulator's
"Search points" button does the same.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from itertools import combinations

import numpy as np

from engine.calibration import effective_true
from engine.config import DEFAULT_CONFIG, DEFAULT_SENSORS
from engine.hardware import Faults
from engine.simulation import Simulation
from ultragps_position import load_config

MAX_RANGE_CM      = 400.0
MIN_SEPARATION_CM = 20.0          # points closer than this are one point read twice
BLOCKED_PENALTY   = 1000.0        # per point a receiver can't see: worse than any clear set


# ─── Geometry ────────────────────────────────────────────────────────────────

def grid(coords: np.ndarray, step: float) -> np.ndarray:
    """Points on a ``step`` grid over the receivers' bounding box."""
    (x0, y0), (x1, y1) = coords.min(axis=0), coords.max(axis=0)
    xs = np.arange(x0, x1 + 1e-9, step)
    ys = np.arange(y0, y1 + 1e-9, step)
    return np.array([(x, y) for x in xs for y in ys])


def candidates(coords: np.ndarray, step: float, margin: float = 0.0, world=None) -> np.ndarray:
    """Grid points ``margin`` inside the receivers' box and in range of every
    receiver; with ``world``, only those inside the walls and outside the obstacles."""
    pts = grid(coords, step)
    (x0, y0), (x1, y1) = coords.min(axis=0) + margin, coords.max(axis=0) - margin
    pts = pts[(pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)]
    far = np.linalg.norm(pts[:, None, :] - coords[None, :, :], axis=2).max(axis=1)
    pts = pts[far <= MAX_RANGE_CM].reshape(-1, 2)
    if world is not None:
        pts = pts[np.array([world.inside(p) and not any(o.contains(p) for o in world.obstacles)
                            for p in pts], dtype=bool).reshape(-1)]
    return pts


def unseen(points, world, receivers) -> np.ndarray:
    """(n, R) bool: receiver r is on but has no line of sight to point n.
    Receivers that are off or blocked can't be helped, so they never count."""
    P   = np.asarray(points, dtype=float).reshape(-1, 2)
    out = np.zeros((len(P), len(receivers)), dtype=bool)
    for k, r in enumerate(receivers):
        if r.state == "on" and len(P):
            out[:, k] = world.blocked_legs(P, np.broadcast_to(np.asarray(r.position, dtype=float), P.shape))
    return out


def too_close(points) -> list[tuple[int, int, float]]:
    """``(i, j, distance)`` of every two points closer than ``MIN_SEPARATION_CM``."""
    P = np.asarray(points, dtype=float).reshape(-1, 2)
    return [(i, j, float(np.hypot(*(P[i] - P[j])))) for i, j in combinations(range(len(P)), 2)
            if np.hypot(*(P[i] - P[j])) < MIN_SEPARATION_CM - 1e-6]


def _crowded(pool: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    """Which ``pool`` points are closer than ``MIN_SEPARATION_CM`` to one of ``chosen``."""
    if not len(chosen):
        return np.zeros(len(pool), dtype=bool)
    d = np.linalg.norm(pool[:, None, :] - np.asarray(chosen)[None, :, :], axis=2)
    return (d < MIN_SEPARATION_CM - 1e-6).any(axis=1)


def score(dists: np.ndarray, eval_d: np.ndarray) -> np.ndarray:
    """Score point sets.

    dists:  (..., n, R) distance from each of n calibration points to each receiver
    eval_d: (E, R) distance from each arena position to each receiver
    Returns (...) RMS of sqrt(1/n + (D - d̄)²/Sxx) over arena positions and receivers.
    """
    n    = dists.shape[-2]
    dbar = dists.mean(axis=-2)                                   # (..., R)
    sxx  = ((dists - dbar[..., None, :]) ** 2).sum(axis=-2)      # (..., R)
    with np.errstate(divide="ignore"):
        inv = np.where(sxx > 1e-9, 1.0 / sxx, np.inf)
    # mean over E of (D - d̄)² = var(D) + (mean(D) - d̄)²
    m, v = eval_d.mean(axis=0), eval_d.var(axis=0)               # (R,)
    per_rx = 1.0 / n + (v + (m - dbar) ** 2) * inv               # (..., R)
    return np.sqrt(per_rx.mean(axis=-1))


def receiver_errors(points: np.ndarray, coords: np.ndarray) -> list[tuple[float, float]]:
    """Per receiver: (relative slope error, intercept error), per unit of
    per-reading distance error."""
    d    = np.linalg.norm(points[:, None, :] - coords[None, :, :], axis=2)
    dbar = d.mean(axis=0)
    sxx  = ((d - dbar) ** 2).sum(axis=0)
    return [(1 / np.sqrt(s), np.sqrt(1 / len(points) + db ** 2 / s)) for s, db in zip(sxx, dbar)]


def search(coords: np.ndarray, max_points: int, margin: float = 30.0, step: float = 20.0,
           fine: float = 5.0, world=None, receivers=None) -> dict[int, np.ndarray]:
    """Best point set for each size 2..max_points.  With ``world``, only points
    the transmitter can be at; with ``receivers`` too (where they really are,
    and their states), points out of a receiver's line of sight are avoided.
    Raises ValueError when not enough points qualify."""
    cand = candidates(coords, step, margin, world)
    if len(cand) < max(max_points, 2):
        found = ("no point qualifies" if not len(cand) else
                 f"only {len(cand)} point{' qualifies' if len(cand) == 1 else 's qualify'} for {max_points}")
        where = ", inside the walls and outside the obstacles" if world is not None else ""
        raise ValueError(f"{found}: a point must be within {MAX_RANGE_CM:.0f} cm of every receiver and "
                         f"{margin:g} cm inside their box{where}. Are the receivers too far apart, "
                         f"or in a line?")
    eval_d = np.linalg.norm(grid(coords, 20.0)[:, None, :] - coords[None, :, :], axis=2)

    def pool(pts: np.ndarray) -> tuple:
        """The points, their distances to the receivers and their line-of-sight penalty."""
        d   = np.linalg.norm(pts[:, None, :] - coords[None, :, :], axis=2)                 # (M, R)
        pen = (BLOCKED_PENALTY * unseen(pts, world, receivers).sum(axis=1)
               if world is not None and receivers is not None else np.zeros(len(pts)))
        return pts, d, pen

    def cost(p: tuple, idx: np.ndarray) -> np.ndarray:
        return score(p[1][idx], eval_d) + p[2][idx].sum(axis=-1)

    def exchange(idx: list[int], p: tuple) -> list[int]:
        idx = list(idx)
        cur = float(cost(p, np.array(idx)))
        improved = True
        while improved:
            improved = False
            for slot in range(len(idx)):
                trial = np.repeat(np.array(idx)[None, :], len(p[0]), axis=0)
                trial[:, slot] = np.arange(len(p[0]))
                sc = cost(p, trial)
                sc[_crowded(p[0], p[0][[k for n, k in enumerate(idx) if n != slot]])] = np.inf
                j  = int(np.argmin(sc))
                if sc[j] < cur - 1e-12:
                    idx[slot], cur, improved = j, float(sc[j]), True
        return idx

    # 2 points: every pair (distinct grid points are at least a step apart)
    coarse = pool(cand)
    pairs  = np.array(list(combinations(range(len(cand)), 2)))
    s      = np.concatenate([cost(coarse, pairs[i:i + 20000]) for i in range(0, len(pairs), 20000)])
    best   = {2: list(pairs[int(np.argmin(s))])}

    for n in range(3, max_points + 1):
        base  = best[n - 1]
        trial = np.array([base + [j] for j in range(len(cand))])
        sc    = cost(coarse, trial)
        sc[_crowded(cand, cand[base])] = np.inf
        best[n] = exchange(base + [int(np.argmin(sc))], coarse)
    best[2] = exchange(best[2], coarse)

    # Fine refinement: let each point move on a finer grid
    fpool = pool(candidates(coords, fine, margin, world))
    out   = {}
    for n, idx in best.items():
        near   = [int(np.argmin(np.linalg.norm(fpool[0] - q, axis=1))) for q in cand[idx]]
        out[n] = fpool[0][exchange(near, fpool)]
    return out


def arena_score(points, coords) -> float:
    eval_d = np.linalg.norm(grid(coords, 20.0)[:, None, :] - coords[None, :, :], axis=2)
    d      = np.linalg.norm(np.asarray(points)[:, None, :] - coords[None, :, :], axis=2)
    return float(score(d, eval_d))


# ─── Simulated check ─────────────────────────────────────────────────────────

def true_response(sim: Simulation) -> tuple[np.ndarray, np.ndarray]:
    """Each receiver's true (slope, intercept) from sensors.toml, in the current air."""
    a, b = zip(*(effective_true(r, sim.env) for r in sim.receivers))
    return np.array(a), np.array(b)


def simulate(points, sim: Simulation, min_reads: int, trials: int) -> dict:
    """Calibrate ``trials`` times at ``points`` on the simulated sensors.
    Returns RMS slope error (%), intercept error (cm) and resulting distance
    error over the arena (cm), across receivers and trials."""
    pts    = np.asarray(points, dtype=float)
    a, b   = true_response(sim)
    R      = len(a)
    known  = np.linalg.norm(pts[:, None, :] - sim.coords[None, :, :], axis=2)   # (n, R)
    eval_d = np.linalg.norm(grid(sim.coords, 20.0)[:, None, :] - sim.coords[None, :, :], axis=2)
    se, ie, de = [], [], []
    for _ in range(trials):
        modal = np.zeros((len(pts), R))
        for j, p in enumerate(pts):
            hist = [dict() for _ in range(R)]
            while min(max(h.values(), default=0) for h in hist) < min_reads:
                ticks = [r.tick for r in sim.hardware.measure(p)]
                for i, t in enumerate(ticks):
                    hist[i][t] = hist[i].get(t, 0) + 1
            modal[j] = [max(h, key=h.__getitem__) for h in hist]
        for i in range(R):
            if np.ptp(modal[:, i]) == 0:
                continue
            ah, bh = np.polyfit(modal[:, i], known[:, i], 1)
            se.append((ah - a[i]) / a[i] * 100)
            ie.append(bh - b[i])
            t = (eval_d[:, i] - b[i]) / a[i]
            de.extend(ah * t + bh - eval_d[:, i])
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
    return {"slope_pct": rms(se), "intercept_cm": rms(ie), "distance_cm": rms(de)}


def library_check(points, config_path: str, noise: float, seed: int,
                  sensors_path: str = DEFAULT_SENSORS) -> dict:
    """Run the calibration library once at a point set; return errors."""
    from engine.calibration import Calibrator, set_cal_points
    tmp = os.path.join(tempfile.mkdtemp(), "config.toml")
    set_cal_points(config_path, points, tmp)
    sim = Simulation(tmp, sensors_path, seed=seed)
    sim.faults = Faults(noise_cm=noise)
    off = Calibrator(sim, tmp).run()
    a, b = true_response(sim)
    return {i: ((o["slope"] - a[i]) / a[i] * 100, o["intercept"] - b[i]) for i, o in off.items()}


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Find calibration points that minimise calibration error.")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--sensors", default=DEFAULT_SENSORS)
    p.add_argument("--max-points", type=int, default=4, choices=(2, 3, 4))
    p.add_argument("--margin", type=float, default=30.0,
                   help="keep points this far (cm) inside the receivers' box (default 30)")
    p.add_argument("--noise", type=float, default=1.0, help="range noise σ (cm) for the check")
    p.add_argument("--trials", type=int, default=200, help="simulated calibrations per set")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--apply", action="store_true",
                   help="write the best --max-points set into --config")
    p.add_argument("--plot", default="", help="save a plot here (needs matplotlib)")
    args = p.parse_args()

    sim = Simulation(args.config, args.sensors, seed=args.seed)
    sim.faults = Faults(noise_cm=args.noise)
    cfg = load_config(args.config)
    min_reads = int(cfg.get("calibration_reads", 5))
    current = cfg.get("cal_points", [cfg["cal_point_1"], cfg["cal_point_2"]])
    sets = {"current": np.array(current, dtype=float)}
    try:
        found = search(sim.coords, args.max_points, args.margin, world=sim.world, receivers=sim.receivers)
    except ValueError as exc:
        raise SystemExit(f"cal_points: {exc}")
    for n, pts in found.items():
        sets[f"best {n}"] = pts

    print(f"Simulated check: noise σ = {args.noise} cm, {args.trials} calibrations per set, "
          f"modal tick from {min_reads} reads\n")
    print(f"{'set':<9} {'score':>6} {'slope err':>10} {'intcpt err':>11} {'dist err':>9}   points (cm)")
    for name, pts in sets.items():
        r = simulate(pts, sim, min_reads, args.trials)
        where = "  ".join(f"({x:.0f}, {y:.0f})" for x, y in pts)
        print(f"{name:<9} {arena_score(pts, sim.coords):6.3f} {r['slope_pct']:9.2f}% "
              f"{r['intercept_cm']:9.2f}cm {r['distance_cm']:7.2f}cm   {where}")

    print("\nPer receiver, relative to the current set (slope err ×, intercept err ×):")
    cur = receiver_errors(sets["current"], sim.coords)
    for name, pts in sets.items():
        if name == "current":
            continue
        e = receiver_errors(pts, sim.coords)
        print(f"  {name}: " + "  ".join(f"R{i} {s / cs:.2f}/{c / cc:.2f}"
                                        for i, ((s, c), (cs, cc)) in enumerate(zip(e, cur))))

    best = f"best {args.max_points}"
    lib = library_check(sets[best], args.config, args.noise, args.seed, args.sensors)
    print(f"\n{best} through the calibration library (one run): "
          + "  ".join(f"R{i} {s:+.1f}%/{c:+.1f}cm" for i, (s, c) in lib.items()))

    if args.apply:
        from engine.calibration import set_cal_points
        set_cal_points(args.config, sets[best])
        print(f"\nWrote the {best} set to {args.config}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 8))
        c = sim.coords
        ax.scatter(c[:, 0], c[:, 1], marker="^", s=150, c="tab:green", edgecolors="k", zorder=3)
        for i, (x, y) in enumerate(c):
            ax.annotate(f"R{i}", (x, y), xytext=(0, 10), textcoords="offset points", ha="center")
        style = {"current": ("s", "gray"), "best 2": ("o", "tab:blue"),
                 "best 3": ("D", "tab:orange"), "best 4": ("*", "tab:red")}
        for name, pts in sets.items():
            m, col = style[name]
            ax.scatter(pts[:, 0], pts[:, 1], marker=m, s=110, facecolors="none" if m != "*" else col,
                       edgecolors=col, linewidths=2, label=name, zorder=4)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4)
        ax.set_title("Calibration points")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=100)
        print(f"Plot saved to {args.plot}")


if __name__ == "__main__":
    main()
