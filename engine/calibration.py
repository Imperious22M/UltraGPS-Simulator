"""
Calibration — runs the real ``ultragps_calibration`` library against the
simulated sensors, and reads / writes the offsets it produces.

The library drives hardware through a client object whose ``pulse()`` returns
one tick per receiver.  ``SimClient`` stands in for ``UltraGPSClient``: its
``pulse()`` pings the simulated hardware with the transmitter parked on the
calibration point, with the current physics, faults and receiver states.

The library collects readings at each calibration point in turn (2 to 4 of
them, ``cal_points`` in config.toml), takes each receiver's most frequent tick
at each point, fits a least-squares line through (tick, known distance) —
for two points exactly the control program's

    slope     = (dist_1 - dist_2) / (tick_1 - tick_2)
    intercept = dist_1 - slope * tick_1

— and writes the offsets to config.toml.  The positioning library is then
reloaded so the new offsets take effect.  ``set_cal_points`` changes the
points (and every receiver's ``cal_distances``) in config.toml.

``Calibrator`` drives the library's step-by-step API (``start_run`` per
point, with ``on_error``), as a GUI would, rather than the blocking
``run_full_calibration``, which waits for ever when a run ends without
completing: a run can be cancelled, gives up after ``TIMEOUT_S``, and reports
a client error instead of leaving the simulation held.

The sensors time echoes with the speed of sound of the calibration air
(``cal_temperature_c``), so in other air their true response scales by
c(now) / c(cal): ``effective_true``.
"""

from __future__ import annotations

import threading
import time
from contextlib import nullcontext

import numpy as np

from engine import acoustics as ac
from ultragps_calibration import UltraGPSCalibration
from ultragps_calibration import calibration as cal_lib
from ultragps_position import load_config


class SimClient:
    """Minimal stand-in for ``UltraGPSClient``: the calibration library only
    calls ``pulse()``."""

    def __init__(self, sim, lock=None):
        self.sim      = sim
        self.lock     = lock or nullcontext()
        self.position = np.zeros(2)

    def pulse(self) -> list[int]:
        with self.lock:
            return [r.tick for r in self.sim.hardware.measure(self.position, self.sim.motion.heading)]


def read_offsets(config_path: str) -> dict[int, dict[str, float]]:
    """``{receiver_id: {"slope": .., "intercept": ..}}`` from config.toml."""
    return {
        int(r["id"]): {"slope": float(r["offset"]["slope"]),
                       "intercept": float(r["offset"]["intercept"])}
        for r in load_config(config_path)["receivers"]
    }


def effective_true(receiver, env) -> tuple[float, float]:
    """Receiver's true ``(slope, intercept)`` in the current air: the tick it
    gives at a true distance d is ``(d · c_cal / c − intercept) / slope``,
    because it converts time with the calibration air's speed of sound, so
    d = k · (slope · tick + intercept) with k = c(now) / c(cal)."""
    k = (ac.speed_of_sound(env.temperature_c, env.humidity_pct)
         / ac.speed_of_sound(env.cal_temperature_c, env.cal_humidity_pct))
    return receiver.slope * k, receiver.intercept * k


def set_cal_points(config_path: str, points, out_path: str | None = None) -> None:
    """Store calibration points in config.toml: ``cal_points``, the first two
    as ``cal_point_1`` / ``cal_point_2``, and each receiver's ``cal_distances``
    (its distance to every point).  Written with the calibration library's
    own config writer, to ``out_path`` if given."""
    cfg = load_config(config_path)
    pts = [[float(x), float(y)] for x, y in points]
    cfg["cal_points"] = pts
    cfg["cal_point_1"], cfg["cal_point_2"] = pts[0], pts[1]
    for r in cfg["receivers"]:
        pos = np.array(r["position"], dtype=float)
        r["cal_distances"] = [float(np.linalg.norm(pos - p)) for p in pts]
    cal_lib._write_config(out_path or config_path, cfg)


class Calibrator:
    """One simulated calibration at 2 to 4 points, run in the background.

    ``state`` is ``idle``, ``run 1`` .. ``run 4`` (collecting at that point),
    ``done`` or ``error`` (``error`` says why; ``cancelled`` after ``cancel``);
    ``reads`` is the number of pulses collected in the current run.  ``lock``
    is the simulation runner's lock: the calibration runs on its own thread and
    must not move the transmitter or swap the library in the middle of a frame.
    """

    TIMEOUT_S = 60.0              # a whole calibration gives up after this long

    def __init__(self, sim, config_path: str, lock=None):
        self.sim         = sim
        self.config_path = config_path
        self.lock        = lock or nullcontext()
        self.client      = SimClient(sim, lock)
        self.cal         = UltraGPSCalibration(config_path, self.client)
        self.state       = "idle"
        self.reads       = 0
        self.offsets: dict = {}
        self.error       = ""
        self.version     = 0          # bumped whenever the offsets in use change
        self.timeout_s   = self.TIMEOUT_S
        self._cancel     = threading.Event()
        self._parked     = None       # where the last point put the transmitter

    @property
    def busy(self) -> bool:
        return self.state.startswith("run")

    def start(self, min_reads: int | None = None) -> None:
        if self.busy:
            return
        self._cancel.clear()
        self.state = "run 1"
        threading.Thread(target=self._run, args=(min_reads,), daemon=True).start()

    def cancel(self) -> None:
        """Stop a run at once; config.toml keeps the offsets it had."""
        if self.busy:
            self._cancel.set()
            self.cal.stop_run()

    @property
    def points(self) -> list[list[float]]:
        return self.cal.cal_points

    def set_points(self, points) -> None:
        if self.busy:
            raise RuntimeError("calibration is running")
        set_cal_points(self.config_path, points)

    def run(self, min_reads: int | None = None) -> dict:
        """Run every calibration point (blocking), save and apply offsets."""
        self._cancel.clear()
        return self._run(min_reads)

    def _run(self, min_reads: int | None) -> dict:
        drive, self.error, self._parked = None, "", None
        try:
            points    = self.points[:self.cal.MAX_POINTS]
            min_reads = self.cal.calibration_reads if min_reads is None else min_reads
            deadline  = time.monotonic() + self.timeout_s
            with self.lock:
                drive = self._drive()
            for k, point in enumerate(points, start=1):
                self._park(point, f"run {k}")
                self._collect(k, min_reads, deadline)
            offsets = self.cal.calculate_and_save_offsets()
            with self.lock:
                self.sim.reload_calibration()
                self._resume(drive)
            self.offsets = offsets
            self.version += 1
            self.state = "done"
        except Exception as exc:   # surface it in the GUI rather than kill the thread
            self.cal.stop_run()
            with self.lock:
                self._resume(drive)
            self.error, self.state = str(exc) or type(exc).__name__, "error"
        return self.offsets

    def _collect(self, k: int, min_reads: int, deadline: float) -> None:
        """Collect at point ``k`` through the library's ``start_run``, until
        every receiver has read its modal tick ``min_reads`` times; raises on a
        client error, a cancel or the timeout."""
        cal, done, errors = self.cal, threading.Event(), []
        cal.clear_run_data(k)
        cal.start_run(k, min_reads,
                      on_reading=lambda run, n: setattr(self, "reads", n),
                      on_complete=lambda run: done.set(),
                      on_error=lambda run, msg: (errors.append(msg), done.set()))
        try:
            while not done.wait(0.05):
                if self._cancel.is_set():
                    break
                if time.monotonic() > deadline:
                    slow = [f"R{i}" for i in range(cal.RECEIVER_COUNT) if cal.get_max_count(i, k) < min_reads]
                    raise RuntimeError(f"timed out after {self.timeout_s:g} s at P{k}" + (
                        f": {', '.join(slow)} never read the same tick {min_reads} times" if slow else ""))
        finally:
            cal.stop_run()
            worker = getattr(cal, "_cal_thread", None)    # let it finish its pulse before the next run
            if worker is not None and worker is not threading.current_thread():
                worker.join(2.0)
        if self._cancel.is_set():
            raise RuntimeError("cancelled")
        if errors:
            raise RuntimeError(f"P{k}: {errors[0]}")

    def _drive(self) -> tuple:
        """The transmitter's position and drive, to resume after the run."""
        m = self.sim.motion
        return (self.sim.true.copy(), m.mode, np.array(m.target, dtype=float), m.phase, m._lane, m.heading)

    def _resume(self, drive) -> None:
        """Put the transmitter back where it was and resume its drive (circle,
        lawnmower ...), unless it was moved or given a new drive meanwhile."""
        m, at = self.sim.motion, self._parked
        if drive is None or at is None or m.mode != "target" or not (
                np.array_equal(m.target, at) and np.array_equal(self.sim.true, at)):
            return
        self.sim.true = drive[0]
        m.mode, m.target, m.phase, m._lane, m.heading = drive[1:]

    def _park(self, point, state: str) -> None:
        """Put the transmitter on a calibration point."""
        self.client.position = np.array(point, dtype=float)
        with self.lock:
            self.sim.place(self.client.position)
            self._parked = self.sim.true.copy()
        self.state, self.reads = state, 0

    def save(self, offsets: dict[int, dict[str, float]]) -> None:
        """Write hand-edited offsets through the library and apply them."""
        if self.busy:
            raise RuntimeError("calibration is running")
        self.cal.save_offsets(offsets)
        with self.lock:
            self.sim.reload_calibration()
        self.version += 1
