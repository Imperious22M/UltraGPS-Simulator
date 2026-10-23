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
"""

from __future__ import annotations

import threading

import numpy as np

from ultragps_calibration import UltraGPSCalibration
from ultragps_calibration import calibration as cal_lib
from ultragps_position import load_config


class SimClient:
    """Minimal stand-in for ``UltraGPSClient``: the calibration library only
    calls ``pulse()``."""

    def __init__(self, sim):
        self.sim      = sim
        self.position = np.zeros(2)

    def pulse(self) -> list[int]:
        return [r.tick for r in self.sim.hardware.measure(self.position, self.sim.motion.heading)]


def read_offsets(config_path: str) -> dict[int, dict[str, float]]:
    """``{receiver_id: {"slope": .., "intercept": ..}}`` from config.toml."""
    return {
        int(r["id"]): {"slope": float(r["offset"]["slope"]),
                       "intercept": float(r["offset"]["intercept"])}
        for r in load_config(config_path)["receivers"]
    }


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
    """One simulated two-point calibration, run in the background.

    ``state`` is ``idle``, ``run 1``, ``run 2``, ``done`` or ``error``;
    ``reads`` is the number of pulses collected in the current run.
    """

    def __init__(self, sim, config_path: str):
        self.sim         = sim
        self.config_path = config_path
        self.client      = SimClient(sim)
        self.cal         = UltraGPSCalibration(config_path, self.client)
        self.state       = "idle"
        self.reads       = 0
        self.offsets: dict = {}
        self.error       = ""
        self.version     = 0          # bumped whenever the offsets in use change

    @property
    def busy(self) -> bool:
        return self.state.startswith("run")

    def start(self, min_reads: int | None = None) -> None:
        if self.busy:
            return
        self.state = "run 1"
        threading.Thread(target=self.run, args=(min_reads,), daemon=True).start()

    @property
    def points(self) -> list[list[float]]:
        return self.cal.cal_points

    def set_points(self, points) -> None:
        if self.busy:
            raise RuntimeError("calibration is running")
        set_cal_points(self.config_path, points)

    def run(self, min_reads: int | None = None) -> dict:
        """Run every calibration point (blocking), save and apply offsets."""
        try:
            points = self.points
            nxt    = iter(range(1, len(points)))
            self._park(points[0], "run 1")

            def between() -> bool:
                k = next(nxt)
                self._park(points[k], f"run {k + 1}")
                return True

            self.offsets = self.cal.run_full_calibration(
                min_reads=min_reads,
                on_between_runs=between,
                on_reading=lambda run, n: setattr(self, "reads", n),
            )
            self.sim.reload_calibration()
            self.version += 1
            self.state = "done"
        except Exception as exc:   # surface it in the GUI rather than kill the thread
            self.error, self.state = str(exc), "error"
        return self.offsets

    def _park(self, point, state: str) -> None:
        """Put the transmitter on a calibration point."""
        self.client.position = np.array(point, dtype=float)
        self.sim.place(self.client.position)
        self.state, self.reads = state, 0

    def save(self, offsets: dict[int, dict[str, float]]) -> None:
        """Write hand-edited offsets through the library and apply them."""
        if self.busy:
            raise RuntimeError("calibration is running")
        self.cal.save_offsets(offsets)
        self.sim.reload_calibration()
        self.version += 1
