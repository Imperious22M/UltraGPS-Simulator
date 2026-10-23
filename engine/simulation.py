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
import re
import time as _time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from engine.config import (DEFAULT_CONFIG, DEFAULT_SENSORS, DEFAULT_WORLD, believed_positions,
                           load_receivers, load_world)
from engine.hardware import Environment, Faults, Hardware
from engine.motion import Motion, clip_move, push_out
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


def _xy(p) -> tuple | None:
    return None if p is None else (float(p[0]), float(p[1]))


def _fit(values, n: int, fill) -> list:
    """``values`` cut or padded with ``fill`` to length ``n``."""
    return list(values)[:n] + [fill] * (n - len(values))


@dataclass(slots=True)
class Record:
    """What a recording keeps of a frame: what ``save_csv`` and ``save_ticks``
    write, without the sound paths (a whole Frame is 20 to 45 kB)."""
    time: float
    true: tuple | None
    ticks: tuple
    distances: tuple
    status: tuple
    fixes: dict                      # {algorithm: (x, y) or None}
    errors: dict                     # {algorithm: cm or nan}

    @staticmethod
    def of(frame: Frame) -> Record:
        return Record(frame.time, _xy(frame.true), tuple(frame.ticks),
                      tuple(float(d) for d in frame.solution.distances), tuple(frame.status),
                      {a: _xy(e.pos) for a, e in frame.estimates.items()}, dict(frame.errors))


# A serial monitor's timestamp ("12:00:01.123 -> ", "[12:00:01.123] "), and the protocol prefix of a ping ("C: ")
_STAMP  = re.compile(r"\s*\[?(\d{1,2}):(\d{2}):(\d{2}(?:\.\d*)?)\]?\s*(?:-?>)?\s*")
_PREFIX = re.compile(r"\s*[A-Z]: ")
DT_RANGE = (0.01, 1.0)               # s: a replay's time step taken from the file's times


class TickLog:
    """Ticks to replay: the base station's output ("N: 1, 2, ...", "C: ...")
    one ping per line, as a serial monitor captures it (with or without its
    "12:00:01.123 -> " or "[12:00:01.123] " timestamps), or a CSV recorded by
    the simulator (with true positions).

    With ``n``, rows without exactly ``n`` values (a capture that starts or
    stops mid-line, another number of receivers) are left out and counted in
    ``skipped``.  After ``next()``, ``dt`` is the time since the previous row
    when the file has times, else None.  ``fixes`` holds each row's recorded
    ``{algorithm: (x, y)}`` (empty for a tick log)."""

    def __init__(self, path: str, n: int | None = None):
        self.path, self.n, self.pos, self.dt = path, n, 0, None
        self.rows, self.times, self.fixes, self.skipped = [], [], [], 0
        # A capture can start with line noise: undecodable bytes only spoil their own line
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
            lines = fh.read().strip().splitlines()
        if lines and lines[0].startswith("time,"):
            self._read_csv(lines)
        else:
            self._read_log(lines)
        if not self.rows:
            why = f" ({self.skipped} rows without {n} values)" if self.skipped and n else ""
            raise ValueError(f"no ticks found in {path}{why}")

    def _keep(self, ticks: list, true, t: float | None, fixes: dict | None = None) -> None:
        if self.n is not None and len(ticks) != self.n:
            self.skipped += 1
        else:
            self.rows.append((ticks, true))
            self.times.append(t)
            self.fixes.append(fixes or {})

    def _read_log(self, lines: list) -> None:
        for line in lines:
            t = None
            m = _STAMP.match(line)
            if m:
                t    = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
                line = line[m.end():]
            try:
                ticks = UltraGPSPositionLib.parse_message(line)
            except OverflowError:                 # "inf" in a damaged line
                ticks = []
            if ticks:
                self._keep(ticks, None, t)
            elif _PREFIX.match(line):             # a ping, but damaged
                self.skipped += 1

    def _read_csv(self, lines: list) -> None:
        head = lines[0].split(",")                # save_csv never quotes its header
        col  = {name: k for k, name in enumerate(head)}
        tick = [k for k, name in enumerate(head) if name.startswith("tick")]
        algo = [name[:-2] for name in head if name.endswith("_x") and name != "true_x" and name[:-2] + "_y" in col]
        for line in lines[1:]:
            try:
                # One line at a time: a stray quote or an over-long field spoils only its own row
                row = next(csv.reader([line]), [])
                if not any(row):
                    continue
                if len(row) != len(head):         # cut short, or a stray comma
                    raise ValueError(row)
                ticks  = [int(float(row[k])) for k in tick]
                tx, ty = (row[col["true_x"]], row[col["true_y"]]) if "true_x" in col else ("", "")
                true   = np.array([float(tx), float(ty)]) if tx != "" else None
                t      = float(row[col["time"]])
                fixes  = {a: (float(row[col[a + "_x"]]), float(row[col[a + "_y"]]))
                          for a in algo if row[col[a + "_x"]] != ""}
            except (ValueError, OverflowError, KeyError, csv.Error):
                self.skipped += 1
                continue
            self._keep(ticks, true if true is not None and np.all(np.isfinite(true)) else None, t, fixes)

    def next(self):
        """The next ``(ticks, true)``, from the start again after the last."""
        k = self.pos % len(self.rows)
        self.pos += 1
        t0, t1  = self.times[k - 1] if k else None, self.times[k]
        self.dt = (min(max(t1 - t0, DT_RANGE[0]), DT_RANGE[1])
                   if t0 is not None and t1 is not None and t1 > t0 else None)
        return self.rows[k]


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
        """Put the transmitter at ``xy`` and keep it there; a point inside an
        obstacle is moved to just outside it."""
        self.true = push_out(xy, self.world.obstacles)
        self.motion.go_to(self.true.copy())

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
            dt    = self.replay.dt or dt
            n     = len(self.receivers)
            ticks = list(ticks[:n]) + [0] * (n - len(ticks))     # 0: no reading
            if true is not None:
                self.true = np.array(true, dtype=float)
        else:
            self.true, hit = clip_move(self.true, self.motion.step(self.true, dt), self.world.obstacles)
            if hit:
                self.motion.blocked()          # it can't drive into or through an obstacle: stop at it
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
        rec = self.recording               # read once: the GUI may stop recording meanwhile
        if rec is not None:
            rec.append(Record.of(self.last))
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
        """Keep a ``Record`` of every frame from now on."""
        self.recording = []

    def stop_recording(self) -> list:
        """The records since ``start_recording``."""
        records, self.recording = self.recording or [], None
        return records

    @staticmethod
    def save_csv(records: list, path: str) -> None:
        """One row per frame (``Record`` or ``Frame``): truth, ticks, distances
        and every estimate."""
        recs = [r if isinstance(r, Record) else Record.of(r) for r in records]
        if not recs:
            return
        n     = max(len(r.ticks) for r in recs)
        algos = list(dict.fromkeys(a for r in recs for a in r.fixes))
        head  = (["time", "true_x", "true_y"] + [f"tick{i}" for i in range(n)]
                 + [f"dist{i}" for i in range(n)] + [f"status{i}" for i in range(n)]
                 + [f"{a}_{c}" for a in algos for c in ("x", "y", "err")])
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(head)
            for r in recs:
                tx = ["", ""] if r.true is None else [f"{r.true[0]:.2f}", f"{r.true[1]:.2f}"]
                row = ([f"{r.time:.3f}"] + tx + _fit(r.ticks, n, 0)
                       + _fit([f"{d:.2f}" for d in r.distances], n, "") + _fit(r.status, n, ""))
                for a in algos:
                    p = r.fixes.get(a)
                    row += (["", "", ""] if p is None else
                            [f"{p[0]:.2f}", f"{p[1]:.2f}", f"{r.errors.get(a, float('nan')):.2f}"])
                w.writerow(row)

    @staticmethod
    def save_ticks(records: list, path: str) -> None:
        """The ticks as the base station prints them in continuous mode ("C: ...")."""
        with open(path, "w") as fh:
            for r in records:
                fh.write("C: " + ", ".join(str(t) for t in r.ticks) + "\n")

    def load_replay(self, path: str) -> None:
        """Replay ``path`` in place of the simulated hardware.  Rows without
        one value per receiver are left out (``replay.skipped`` counts them)."""
        self.replay = TickLog(path, len(self.receivers))
        self.reset()

    def stop_replay(self) -> None:
        self.replay = None
        self.reset()
