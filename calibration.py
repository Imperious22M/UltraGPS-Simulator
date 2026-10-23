"""
Calibration — runs the real ``ultragps_calibration`` library against the
simulated sensors, and reads / writes the offsets it produces.

The library drives hardware through a client object whose ``pulse()`` returns
one tick per receiver.  ``SimClient`` stands in for ``UltraGPSClient``: its
``pulse()`` measures the simulated sensors at the calibration point the
transmitter is parked on, with the current faults and receivers switched off.

The library collects readings at ``cal_point_1``, then ``cal_point_2``, takes
each receiver's most frequent tick at each point, fits

    slope     = (dist_1 - dist_2) / (tick_1 - tick_2)
    intercept = dist_1 - slope * tick_1

and writes them to config.toml, exactly as the control program does.  The
positioning library is then reloaded so the new offsets take effect.
"""

from __future__ import annotations

import threading

import numpy as np

from ultragps_calibration import UltraGPSCalibration
from ultragps_position import load_config


class SimClient:
    """Minimal stand-in for ``UltraGPSClient``: the calibration library only
    calls ``pulse()``."""

    def __init__(self, sim):
        self.sim      = sim
        self.position = np.zeros(2)

    def pulse(self) -> list[int]:
        ticks, _ = self.sim.sensor.measure(self.position, self.sim.enabled, self.sim.faults)
        return ticks


def read_offsets(config_path: str) -> dict[int, dict[str, float]]:
    """``{receiver_id: {"slope": .., "intercept": ..}}`` from config.toml."""
    return {
        int(r["id"]): {"slope": float(r["offset"]["slope"]),
                       "intercept": float(r["offset"]["intercept"])}
        for r in load_config(config_path)["receivers"]
    }


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

    def run(self, min_reads: int | None = None) -> dict:
        """Run both calibration points (blocking), save and apply offsets."""
        try:
            self._park(self.cal.cal_point_1, "run 1")
            self.offsets = self.cal.run_full_calibration(
                min_reads=min_reads,
                on_between_runs=lambda: self._park(self.cal.cal_point_2, "run 2") or True,
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
        self.sim.true = self.sim.target = self.client.position.copy()
        self.state, self.reads = state, 0

    def save(self, offsets: dict[int, dict[str, float]]) -> None:
        """Write hand-edited offsets through the library and apply them."""
        if self.busy:
            raise RuntimeError("calibration is running")
        self.cal.save_offsets(offsets)
        self.sim.reload_calibration()
        self.version += 1
