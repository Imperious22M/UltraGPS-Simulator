"""
Maps — one number for every point of the arena, to see where the positioning
system works and why.  Pure numpy over the engine, no GUI code.

    coverage       receivers whose direct sound, on its own, is strong enough
                   to detect before the module or the base station gives up
    first_arrival  receivers that would time something other than the direct
                   sound: a reflection, sound bent round an obstacle, another
                   receiver's burst, the direct sound shifted or cancelled by
                   an overlapping echo (fade), or nothing at all
    hdop           horizontal dilution of precision of the covered receivers:
                   cm of position error per cm of range error
    bias           mean range error |measured − true| with no noise: range
                   walk where the signal is weak, echoes and bent sound where
                   the line of sight is blocked
    accuracy       Monte-Carlo RMS position error of one algorithm

Each returns ``(xs, ys, Z)`` with ``Z[j, i]`` at ``(xs[i], ys[j])``, on a grid
covering the box of the receivers (where config.toml says they are) plus
``MARGIN_CM``.  ``nan`` marks a point the transmitter can't be at (inside an
obstacle or outside the walls), ``inf`` a point with no fix.

The first four are deterministic: amplitude turbulence, timing jitter, radio
loss, ambient noise, near-receiver noise and injected faults are switched off,
so they show the physics alone (each receiver keeps its module's gain and aim
spread, which do not change from ping to ping).  ``accuracy`` pings each point
``trials`` times with all of it on, as the simulation does.  A transmitter set
to face its heading faces the simulation's current heading everywhere.

Runtime on one core with 6 receivers and the default room, 10 cm grid (about
1300 points): ``coverage`` and ``hdop`` about 1 s, ``first_arrival`` and
``bias`` about 5 s.  ``accuracy`` costs points × trials × (about 2 ms per ping
+ the algorithm): at 20 cm and 10 trials, about 10 s for OLS or LM, 30 s for
CEP4 or RANSAC and 1-2 minutes for the library's CEP.  Everything runs on the
calling thread: ``progress(fraction)`` is called after each point and
``cancel()`` checked before each point and each of ``accuracy``'s pings; when
it returns true the computation stops by raising ``Cancelled``.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from engine import acoustics as ac
from engine import solvers as S
from engine.hardware import Environment, Faults, Hardware
from engine.world import World
from ultragps_position import UltraGPSPositionLib

MARGIN_CM = 30.0                                    # grid reaches this far outside the receivers
DIRECT    = ("ok", "walk", "near")                  # reading statuses that timed the direct sound
HEARD     = DIRECT + ("echo", "bent", "crosstalk", "fade")   # ... that may have timed some sound
ALGORITHMS = ("OLS", "LM", "CEP", "CEP4", "ROBUST", "BIAS", "RANSAC")   # snapshot algorithms (no EKF)


class Cancelled(Exception):
    """Raised inside a map computation once ``cancel()`` returns true."""


@dataclass
class Scene:
    """What the maps read from a simulation, copied so a map can be computed
    on another thread while the simulation carries on."""
    receivers: list
    world: World
    env: Environment
    faults: Faults
    heading: float
    believed: np.ndarray
    config_path: str
    max_distance: float | None      # the library's top-band reject (None: its default)
    settings: S.Settings


def scene(sim) -> Scene:
    """Copy of everything the maps need from ``sim`` (hold the runner's lock
    while taking it).  A ``Scene`` is returned as it is."""
    if isinstance(sim, Scene):
        return sim
    return Scene(copy.deepcopy(sim.receivers), copy.deepcopy(sim.world), sim.env.copy(),
                 dataclasses.replace(sim.faults), float(sim.motion.heading), np.array(sim.believed),
                 sim.config_path, sim.solvers.lib.max_distance, dataclasses.replace(sim.solvers.settings))


def grid(believed, step: float = 10.0, margin: float = MARGIN_CM) -> tuple[np.ndarray, np.ndarray]:
    """Grid lines ``step`` cm apart over the receivers' box plus ``margin``, centred on it."""
    lo = np.min(believed, axis=0) - margin
    hi = np.max(believed, axis=0) + margin
    axes = []
    for a, b in zip(lo, hi):
        n = math.floor((b - a) / step + 1e-9) + 1
        axes.append(a + ((b - a) - (n - 1) * step) / 2 + step * np.arange(n))
    return axes[0], axes[1]


# ─── The maps ────────────────────────────────────────────────────────────────

def coverage(sim, step: float = 10.0, progress=None, cancel=None):
    """Number of receivers whose direct sound from each point is detectable."""
    sc = scene(sim)
    hw = _steady_hardware(sc)
    tx = hw.tx_facing(sc.heading)
    return _map(sc, step, lambda p: float(sum(_hears_direct(hw, rx, p, tx) for rx in sc.receivers)),
                progress, cancel)


def first_arrival(sim, step: float = 10.0, progress=None, cancel=None):
    """Number of receivers (switched on) that would not time the direct sound
    from each point: they read a reflection, bent sound, another receiver's
    burst, or nothing."""
    sc = scene(sim)
    hw = _steady_hardware(sc)

    def count(p):
        readings = hw.measure(p, sc.heading)
        return float(sum(r.status not in DIRECT for r, rx in zip(readings, sc.receivers) if rx.state != "off"))
    return _map(sc, step, count, progress, cancel)


def hdop(sim, step: float = 10.0, progress=None, cancel=None):
    """``sqrt(trace((JᵀJ)⁻¹))`` over the receivers covering each point, where
    J's rows are the unit vectors from those receivers; ``inf`` with fewer than 3."""
    sc = scene(sim)
    hw = _steady_hardware(sc)
    tx = hw.tx_facing(sc.heading)

    def dop(p):
        pos = np.array([rx.position for rx in sc.receivers if _hears_direct(hw, rx, p, tx)])
        if len(pos) < 3:
            return math.inf
        diff = p - pos
        J    = diff / np.maximum(np.hypot(*diff.T), 1e-9)[:, None]
        try:
            return float(np.sqrt(np.trace(np.linalg.inv(J.T @ J))))
        except np.linalg.LinAlgError:
            return math.inf
    return _map(sc, step, dop, progress, cancel)


def bias(sim, step: float = 10.0, progress=None, cancel=None):
    """Mean |measured − true| range (cm) over the receivers that detect some
    sound from each point (a reading lost to the timeout is left out: the
    library's bounds filter drops it); ``inf`` when none does."""
    sc = scene(sim)
    hw = _steady_hardware(sc)

    def mean_error(p):
        errs = [abs(r.measured_cm - r.true_cm) for r in hw.measure(p, sc.heading)
                if r.status in HEARD and r.detected is not None]
        return float(np.mean(errs)) if errs else math.inf
    return _map(sc, step, mean_error, progress, cancel)


def accuracy(sim, algorithm: str = "LM", trials: int = 10, step: float = 20.0, progress=None,
             cancel=None, seed: int = 0):
    """RMS position error (cm) of ``algorithm`` over ``trials`` pings at each
    point, with the current physics, faults, receiver states and calibration.

    Each ping goes through a private ``Hardware`` (seeded with ``seed``) and a
    private ``UltraGPSPositionLib``, and is solved by ``solve`` on every
    receiver within the library's bounds, with no jump filter and no history.
    ``inf`` where fewer than half the pings give a fix."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"no snapshot algorithm {algorithm!r}; choose one of {ALGORITHMS}")
    if trials < 1:
        raise ValueError("accuracy needs at least one ping per point")
    sc  = scene(sim)
    hw  = Hardware(sc.receivers, _remember_paths(copy.deepcopy(sc.world)), sc.env, sc.faults, seed)
    lib = UltraGPSPositionLib(sc.config_path, max_distance=sc.max_distance)
    top = lib.max_distance if lib.max_distance is not None else lib._max_receiver_dist

    def rms(p):
        errs = []
        for _ in range(trials):
            if cancel is not None and cancel():        # a CEP point can take seconds
                raise Cancelled()
            ticks = [r.tick for r in hw.measure(p, sc.heading)]
            d     = np.array(lib._ticks_to_distances(ticks), dtype=float)
            idx   = [i for i, v in enumerate(d) if 0 < v <= top]
            est   = solve(lib, d, idx, algorithm, sc.settings)
            if est.ok and est.pos is not None:
                errs.append(float(np.hypot(*(np.asarray(est.pos, dtype=float) - p))))
        if 2 * len(errs) < trials:
            return math.inf
        return float(np.sqrt(np.mean(np.square(errs))))
    return _map(sc, step, rms, progress, cancel)


def solve(lib: UltraGPSPositionLib, d: np.ndarray, idx: list, algorithm: str,
          settings: S.Settings | None = None) -> S.Estimate:
    """One snapshot algorithm on distances ``d`` from receivers ``idx``, as
    ``lab.static_solve`` runs it.  ``static_solve`` runs all seven (about 55 ms,
    most of it the library's CEP); a map needs thousands of fixes of one.  The
    library's CEP tries as many subsets as ``settings.cep_subsets`` allows."""
    s = settings or S.Settings()
    if algorithm == "CEP4":
        return S.solve_cep4(lib, d, idx)
    if algorithm == "ROBUST":
        return S.solve_robust(lib, d, idx, s.robust_scale_cm)
    if algorithm == "BIAS":
        return S.solve_bias(lib, d, idx)
    if algorithm == "RANSAC":
        return S.solve_ransac(lib, d, idx, s.ransac_tol_cm)
    if len(idx) < 3:
        return S.Estimate(None, False, note="needs 3 receivers")
    if algorithm == "OLS":
        return S.Estimate(lib._ols(d, np.array(idx)), True, list(idx))
    if algorithm == "LM":
        pos, r = lib._multilaterate(tuple(d), np.array(idx))
        return S.Estimate(np.array(pos), bool(r.success), list(idx))
    if algorithm == "CEP":
        pos, _, used, score = lib._find_best_subset(tuple(d), list(idx),
                                                    S.cep_max_subsets(s, lib.receiver_count))
        _, outside = lib._validate_position(pos)
        return S.Estimate(None if pos is None else np.array(pos), not outside, list(used or []),
                          score_cov=score)
    raise ValueError(f"no snapshot algorithm {algorithm!r}")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _map(sc: Scene, step: float, value, progress, cancel):
    """``value(point)`` at every grid point the transmitter can be at."""
    xs, ys = grid(sc.believed, step)
    Z      = np.full((len(ys), len(xs)), np.nan)
    total  = Z.size
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            if cancel is not None and cancel():
                raise Cancelled()
            p = np.array([x, y])
            if sc.world.inside(p) and not any(o.contains(p) for o in sc.world.obstacles):
                Z[j, i] = value(p)
            if progress is not None:
                progress((j * len(xs) + i + 1) / total)
    return xs, ys, Z


def _steady_hardware(sc: Scene) -> Hardware:
    """Hardware with every random effect off, so each point reads the same every ping."""
    env = sc.env.copy(amp_jitter_db=0.0, sync_jitter_us=0.0, radio_loss=0.0, tick_dither=False,
                      near_noise=False, noise_rate_hz=0.0)
    return Hardware(sc.receivers, sc.world, env, Faults(noise_cm=0.0), seed=0)


def _hears_direct(hw: Hardware, rx, p, tx_facing) -> bool:
    """Is ``rx``'s direct sound from ``p`` detectable on its own, while the
    module listens and the base station samples?  Off, blocked or a wall or
    obstacle in the way: no."""
    e = hw.env
    if rx.state == "off":
        return False
    direct = hw.world.paths(p, rx.position, max_order=0, diffraction=False, direct=rx.state == "on",
                            wavelength=ac.wavelength_cm(e.temperature_c, e.humidity_pct))
    if not direct:
        return False
    path = direct[0]
    if e.fov_cutoff and not hw._in_fov(path, rx):
        return False
    arrive = e.burst_lead_ms + e.sync_offset_us / 1000.0 + hw._travel_ms(path)
    if arrive < e.rx_ready_ms:                                # here before the receiver listens
        return False
    if not e.threshold:
        cycle = 0 if path.length <= e.max_range_cm else None
    else:
        cycle = ac.detect_cycle(10.0 ** (hw.rx_strength_db(path, tx_facing, rx) / 20.0), e.rise_cycles)
    give_up = min(e.rx_ready_ms + e.timeout_ms, e.window_ms)
    return cycle is not None and arrive + cycle * ac.CYCLE_MS <= give_up


def _remember_paths(world: World) -> World:
    """Make ``world.paths`` remember its answers, so the repeated pings of one
    point trace each sound path once (it is half the cost of a ping)."""
    trace, memo = world.paths, {}

    def paths(src, dst, *args, **kw):
        key = (tuple(np.round(src, 6)), tuple(np.round(dst, 6)), args, tuple(sorted(kw.items())))
        if key not in memo:
            if len(memo) > 256:
                memo.clear()
            memo[key] = trace(src, dst, *args, **kw)
        return memo[key]
    world.paths = paths
    return world
