"""
Simulation — one object holding the whole state of a run.

Each ``step()`` moves the transmitter toward its target, measures ticks with
the sensor model, solves with every algorithm, and records the error of each.
No GUI code lives here, so it can be driven from a test or a script.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from sensor_model import Faults, SensorModel
from solvers import ALGORITHMS, DEFAULT_CONFIG, DEFAULT_SENSORS, Solvers, load_sensors


class Simulation:
    def __init__(
        self,
        config_path: str = DEFAULT_CONFIG,
        sensors_path: str = DEFAULT_SENSORS,
        max_differential: float = 50.0,
        max_distance: float | None = None,
        history: int = 100,
        seed: int | None = None,
    ):
        # The sensors use the simulated hardware's true response; the solvers
        # use the calibration in config_path, which may differ from it.
        coords, slopes, intercepts = load_sensors(config_path, sensors_path)
        self.sensor  = SensorModel(coords, slopes, intercepts, seed=seed)
        self.solvers = Solvers(config_path, max_differential, max_distance)
        self.coords  = coords
        self.faults  = Faults()
        self.enabled = [True] * len(coords)
        self.speed   = 50.0           # cm/s toward the target
        self.true    = np.zeros(2)
        self.target  = np.zeros(2)
        self.errors  = {a: deque(maxlen=history) for a in ALGORITHMS}
        self.trails  = {a: deque(maxlen=30) for a in ALGORITHMS}
        self.fails   = {a: 0 for a in ALGORITHMS}
        self.frames  = 0
        self.time    = 0.0            # simulated seconds
        self.last    = None

    def reset(self) -> None:
        self.solvers.reset()
        for a in ALGORITHMS:
            self.errors[a].clear()
            self.trails[a].clear()
            self.fails[a] = 0
        self.frames = 0
        self.time   = 0.0

    def reload_calibration(self) -> None:
        """Re-read the offsets in config.toml into the positioning library."""
        self.solvers.lib.reload()
        self.reset()

    def step(self, dt: float = 0.1) -> dict:
        # Move toward the target at a fixed speed
        delta = self.target - self.true
        dist  = float(np.linalg.norm(delta))
        move  = self.speed * dt
        self.true = self.target.copy() if dist <= move else self.true + delta / dist * move

        ticks, status = self.sensor.measure(self.true, self.enabled, self.faults)
        result = self.solvers.solve(ticks)
        self.frames += 1
        self.time   += dt

        for a in ALGORITHMS:
            pos = result["positions"][a]
            if not result["success"][a]:
                self.fails[a] += 1
            if pos is not None:
                self.errors[a].append(float(np.linalg.norm(np.asarray(pos) - self.true)))
                self.trails[a].append(np.asarray(pos, dtype=float))

        self.last = {**result, "ticks": ticks, "status": status, "true": self.true.copy()}
        return self.last

    def stats(self, algorithm: str) -> tuple[float, float, int]:
        """``(current error, RMS error over history, failed frames)`` in cm."""
        e = self.errors[algorithm]
        if not e:
            return float("nan"), float("nan"), self.fails[algorithm]
        return e[-1], float(np.sqrt(np.mean(np.square(e)))), self.fails[algorithm]
