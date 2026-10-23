"""
Simulation — the whole state of a run, with no GUI code, so a test or a
script can drive it the same way the windows do.

Each ``step()`` moves the transmitter, pings the simulated hardware, runs every
solver on the ticks and scores each estimate against the true position.  A
tick log (from the real base station or a recording) can be replayed in place
of the simulated hardware.
"""

from __future__ import annotations

import csv
import time as _time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from engine.config import (DEFAULT_CONFIG, DEFAULT_SENSORS, DEFAULT_WORLD, believed_positions,
                           load_receivers, load_world)
from engine.hardware import Environment, Faults, Hardware
from engine.motion import Motion
from engine.solvers import ALGORITHMS, Solution, Solvers
from ultragps_position import UltraGPSPositionLib


@dataclass
class Frame:
    index: int
    time: float                      # simulated seconds
    true: np.ndarray | None          # None when replaying a log without truth
    target: np.ndarray
    heading: float
    ticks: list
    solution: Solution
    readings: list = field(default_factory=list)   # [Reading] (empty when replaying)
    errors: dict = field(default_factory=dict)     # {algorithm: cm or nan}
    compute_ms: float = 0.0

    @property
    def estimates(self) -> dict:
        return self.solution.estimates

    @property
    def status(self) -> list:
        return [r.status for r in self.readings] if self.readings else ["log"] * len(self.ticks)


class TickLog:
    """Ticks to replay: the base station's output ("N: 1, 2, ...", "C: ...")
    one ping per line, or a CSV recorded by the simulator (with true positions)."""

    def __init__(self, path: str):
        self.path, self.rows, self.pos = path, [], 0
        with open(path, newline="") as fh:
            text = fh.read()
        first = text.lstrip().split("\n", 1)[0]
        if first.startswith("time,"):
            for row in csv.DictReader(text.splitlines()):
                ticks = [int(float(row[k])) for k in row if k.startswith("tick")]
                true  = (np.array([float(row["true_x"]), float(row["true_y"])])
                         if row.get("true_x") not in (None, "") else None)
                self.rows.append((ticks, true))
        else:
            for line in text.splitlines():
                ticks = UltraGPSPositionLib.parse_message(line)
                if ticks:
                    self.rows.append((ticks, None))
        if not self.rows:
            raise ValueError(f"no ticks found in {path}")

    def next(self):
        row = self.rows[self.pos % len(self.rows)]
        self.pos += 1
        return row


class Simulation:
    def __init__(self, config_path: str = DEFAULT_CONFIG, sensors_path: str = DEFAULT_SENSORS,
                 world_path: str | None = DEFAULT_WORLD, max_differential: float = 50.0,
                 max_distance: float | None = None, history: int = 300, seed: int | None = None,
                 env: Environment | None = None, faults: Faults | None = None):
        self.config_path  = config_path
        self.sensors_path = sensors_path
        self.world_path   = world_path
        self.believed     = believed_positions(config_path)
        receivers         = load_receivers(config_path, sensors_path)
        self.world        = load_world(world_path, self.believed)
        self.hardware     = Hardware(receivers, self.world, env or Environment(), faults or Faults(), seed)
        self.solvers      = Solvers(config_path, max_differential, max_distance)
        (x0, y0), (x1, y1) = self.believed.min(axis=0), self.believed.max(axis=0)
        self.motion       = Motion((x0 + 10, y0 + 10, x1 - 10, y1 - 10), seed)
        self.true         = np.zeros(2)
        self.history      = history
        self.errors       = {a: deque(maxlen=history) for a in ALGORITHMS}
        self.trails       = {a: deque(maxlen=40) for a in ALGORITHMS}
        self.fails        = {a: 0 for a in ALGORITHMS}
        self.times        = deque(maxlen=history)
        self.frames       = 0
        self.time         = 0.0
        self.last: Frame | None = None
        self.recording: list | None = None
        self.replay: TickLog | None = None

    # ─── Shortcuts ───────────────────────────────────────────────────────────

    @property
    def env(self) -> Environment:
        return self.hardware.env

    @env.setter
    def env(self, value: Environment) -> None:
        self.hardware.env = value

    @property
    def faults(self) -> Faults:
        return self.hardware.faults

    @faults.setter
    def faults(self, value: Faults) -> None:
        self.hardware.faults = value

    @property
    def receivers(self) -> list:
        return self.hardware.receivers

    @property
    def coords(self) -> np.ndarray:
        """Where the receivers really are."""
        return self.hardware.coords

    @property
    def target(self) -> np.ndarray:
        return self.motion.target

    @target.setter
    def target(self, xy) -> None:
        self.motion.go_to(xy)

    @property
    def speed(self) -> float:
        return self.motion.speed

    @speed.setter
    def speed(self, v: float) -> None:
        self.motion.speed = v

    def set_receiver_state(self, i: int, state: str) -> None:
        """``on``, ``blocked`` (no line of sight) or ``off``."""
        self.receivers[i].state = state

    def place(self, xy) -> None:
        """Put the transmitter at ``xy`` and keep it there."""
        self.true = np.asarray(xy, dtype=float).copy()
        self.motion.go_to(self.true)

    def survey_error(self, sigma_cm: float, seed: int | None = None) -> None:
        """Move each receiver's true position off where config.toml says it is,
        by a random amount with standard deviation ``sigma_cm`` (0 puts them back)."""
        rng = np.random.default_rng(seed)
        for r, b in zip(self.receivers, self.believed):
            r.position = b + (rng.normal(0.0, sigma_cm, 2) if sigma_cm > 0 else 0.0)

    # ─── Running ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear statistics and every solver's memory."""
        self.solvers.reset()
        for a in ALGORITHMS:
            self.errors[a].clear()
            self.trails[a].clear()
            self.fails[a] = 0
        self.times.clear()
        self.frames = 0
        self.time   = 0.0

    def reload_calibration(self) -> None:
        """Re-read the offsets in config.toml into the positioning library."""
        self.solvers.reload()
        self.reset()

    def step(self, dt: float = 0.1) -> Frame:
        t0 = _time.perf_counter()
        readings, true = [], None
        if self.replay is not None:
            ticks, true = self.replay.next()
            if true is not None:
                self.true = true
        else:
            ahead = self.motion.step(self.true, dt)
            if any(o.contains(ahead) for o in self.world.obstacles):
                self.motion.blocked()          # it can't drive through an obstacle: stop at it
            else:
                self.true = ahead
            true      = self.true.copy()
            readings  = self.hardware.measure(true, self.motion.heading)
            ticks     = [r.tick for r in readings]

        sol = self.solvers.solve(ticks, dt)
        self.frames += 1
        self.time   += dt
        self.times.append(self.time)

        errors = {}
        for a, e in sol.estimates.items():
            if not e.ok:
                self.fails[a] += 1
            err = float("nan")
            if e.pos is not None:
                self.trails[a].append(np.asarray(e.pos, dtype=float))
                if true is not None:
                    err = float(np.linalg.norm(np.asarray(e.pos) - true))
            self.errors[a].append(err)
            errors[a] = err

        self.last = Frame(self.frames, self.time, true, self.motion.target.copy(), self.motion.heading,
                          list(ticks), sol, readings, errors, (_time.perf_counter() - t0) * 1000.0)
        if self.recording is not None:
            self.recording.append(self.last)
        return self.last

    def stats(self, algorithm: str) -> tuple[float, float, float, int]:
        """``(current error, RMS, 95th percentile, failed frames)`` in cm over the history."""
        e = np.array([v for v in self.errors[algorithm] if v == v])
        if not len(e):
            return float("nan"), float("nan"), float("nan"), self.fails[algorithm]
        last = self.errors[algorithm][-1]
        return last, float(np.sqrt(np.mean(e ** 2))), float(np.percentile(e, 95)), self.fails[algorithm]

    # ─── Recording and replay ────────────────────────────────────────────────

    def start_recording(self) -> None:
        self.recording = []

    def stop_recording(self) -> list:
        frames, self.recording = self.recording or [], None
        return frames

    @staticmethod
    def save_csv(frames: list, path: str) -> None:
        """One row per frame: truth, ticks, distances and every estimate."""
        if not frames:
            return
        n     = len(frames[0].ticks)
        algos = list(frames[0].estimates)
        head  = (["time", "true_x", "true_y"] + [f"tick{i}" for i in range(n)]
                 + [f"dist{i}" for i in range(n)] + [f"status{i}" for i in range(n)]
                 + [f"{a}_{c}" for a in algos for c in ("x", "y", "err")])
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(head)
            for f in frames:
                tx = ["", ""] if f.true is None else [f"{f.true[0]:.2f}", f"{f.true[1]:.2f}"]
                row = [f"{f.time:.3f}"] + tx + list(f.ticks)
                row += [f"{d:.2f}" for d in f.solution.distances] + list(f.status)
                for a in algos:
                    e = f.estimates[a]
                    row += (["", "", ""] if e.pos is None else
                            [f"{e.pos[0]:.2f}", f"{e.pos[1]:.2f}", f"{f.errors.get(a, float('nan')):.2f}"])
                w.writerow(row)

    @staticmethod
    def save_ticks(frames: list, path: str) -> None:
        """The ticks as the base station prints them in continuous mode ("C: ...")."""
        with open(path, "w") as fh:
            for f in frames:
                fh.write("C: " + ", ".join(str(t) for t in f.ticks) + "\n")

    def load_replay(self, path: str) -> None:
        self.replay = TickLog(path)
        self.reset()

    def stop_replay(self) -> None:
        self.replay = None
        self.reset()
