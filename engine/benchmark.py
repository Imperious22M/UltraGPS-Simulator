"""
Benchmark — every algorithm through a set of standard conditions, as a table.

    python3 -m engine.benchmark [--frames 300] [--seed 1] [--markdown]
    python3 -m engine.benchmark --compare recording.csv

Each scenario drives the transmitter between random waypoints for ``frames``
frames (10 Hz) with a fixed seed, so runs are repeatable, and reports each
algorithm's RMS and 95th-percentile error (cm) and how often it failed to
produce a fresh fix.  CEP is the library's search as UltraGPS-Control runs it
(the first 15 subsets); the extra column CEP-all is the same library CEP on
the same ticks trying every subset (``Settings.cep_subsets = "all"``).  They
differ when R0 and R1 both read long, as each of those 15 holds R0 or R1.

``--compare`` replays a CSV recorded by the simulator with the calibration in
config.toml and the default solver settings, and reports how far each
algorithm's fixes moved from the recorded ones: a regression check after
changing either.  The first frames can differ for another reason: the
recording does not hold the solvers' memory from before it started.
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import numpy as np

from engine.config import DEFAULT_CONFIG
from engine.hardware import PRESETS, Environment, Faults
from engine.simulation import Simulation
from engine.solvers import ALGORITHMS, Solvers
from engine.world import Obstacle

CEP_ALL = "CEP-all"


def _box(sim):
    sim.world.obstacles = [Obstacle.rect(30, 30, 70, 70, name="box")]


def _cabinets(sim):
    # Each blocks a middle receiver's view of part of the arena, but not of the
    # wall behind it: R1 and R4 often hear that reflection first and read long
    sim.world.obstacles = [Obstacle.rect(-100, 20, -60, 60, name="cabinet"),
                           Obstacle.rect(60, -60, 100, -20, name="cabinet")]


def _r0_r1_out(sim):
    # Both really 30 cm further out than config.toml says, so both read long.
    # Every one of the 15 subsets UltraGPS-Control tries holds R0 or R1, so
    # its CEP cannot drop the two together; CEP-all can
    centre = sim.believed.mean(axis=0)
    for i in (0, 1):
        u = sim.believed[i] - centre
        sim.receivers[i].position = sim.believed[i] + 30.0 * u / np.linalg.norm(u)


def _r4_off(sim):
    sim.set_receiver_state(4, "off")


def _survey(sim):
    sim.survey_error(3.0, seed=7)


SCENARIOS = [
    ("Ideal (first simulator), noise 1 cm",  Environment.ideal(),                    Faults(1.0), None),
    ("Realistic room",                       Environment(),                          Faults(0.5), None),
    ("Realistic + 40 cm box in the arena",   Environment(),                          Faults(0.5), _box),
    ("Two cabinets: R1 and R4 hear echoes",  Environment(),                          Faults(0.5), _cabinets),
    ("Bad values 5 % (±50 %)",               Environment(),                          Faults(0.5, spike_rate=0.05), None),
    ("Dropouts 10 %, lost signal 5 %",       Environment(),                          Faults(0.5, dropout_rate=0.1, timeout_rate=0.05), None),
    ("Trigger jitter 100 µs",                Environment(sync_jitter_us=100.0),      Faults(0.5), None),
    ("Air at 30 °C (calibrated at 20 °C)",   Environment(temperature_c=30.0),        Faults(0.5), None),
    ("Receiver R4 off",                      Environment(),                          Faults(0.5), _r4_off),
    ("Receivers misplaced (σ 3 cm)",         Environment(),                          Faults(0.5), _survey),
    ("R0 and R1 mounted 30 cm further out",  Environment(),                          Faults(0.5), _r0_r1_out),
    ("Harsh preset",                         Environment(**PRESETS["Harsh"]),        Faults(1.0, spike_rate=0.02), None),
]


def _row(errors, fails: int, frames: int) -> tuple[float, float, float]:
    e = np.array([v for v in errors if v == v])
    return (float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan"),
            float(np.percentile(e, 95)) if len(e) else float("nan"),
            100.0 * fails / frames)


def run(env, faults, setup, frames: int, seed: int) -> dict:
    """``{algorithm: (RMS, 95th percentile, % failed)}``, with CEP-all."""
    sim = Simulation(seed=seed, env=env, faults=faults, history=frames)
    lib   = sim.solvers.lib
    every = Solvers(sim.config_path, lib.max_differential, lib.max_distance,
                    dataclasses.replace(sim.solvers.settings, cep_subsets="all"))
    if setup:
        setup(sim)
    sim.place(sim.believed.mean(axis=0))
    sim.motion.set_mode("random")
    extra, fails = [], 0
    for _ in range(frames):
        f = sim.step(0.1)
        e = every.solve(f.ticks, 0.1, only=set()).estimates["CEP"]
        extra.append(float("nan") if e.pos is None else float(np.hypot(*(np.asarray(e.pos) - f.true))))
        fails += not e.ok
    out = {a: _row(sim.errors[a], sim.fails[a], frames) for a in ALGORITHMS}
    out[CEP_ALL] = _row(extra, fails, frames)
    return out


def compare(path: str, config_path: str = DEFAULT_CONFIG) -> dict:
    """``{algorithm: (mean, largest)}`` distance (cm) between each recorded fix
    in the CSV at ``path`` and the fix the algorithm gives now on its ticks."""
    sim = Simulation(config_path)
    sim.load_replay(path)
    shift = {}
    for fixes in sim.replay.fixes:
        f = sim.step()
        for a, xy in fixes.items():
            e = f.estimates.get(a)
            d = float("nan") if e is None or e.pos is None else float(np.hypot(*(np.asarray(e.pos) - xy)))
            if d == d:
                shift.setdefault(a, []).append(d)
    return {a: (float(np.mean(v)), float(np.max(v))) for a, v in shift.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="Every algorithm under standard conditions.")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--markdown", action="store_true", help="print Markdown tables")
    p.add_argument("--compare", metavar="CSV", help="replay a recording and compare its fixes with today's")
    args = p.parse_args()

    if args.compare:
        try:
            shift = compare(args.compare)
        except (OSError, ValueError) as e:
            raise SystemExit(f"--compare: {e}")
        if not shift:
            raise SystemExit(f"no recorded fixes in {args.compare}: --compare needs a CSV recorded by the simulator")
        for a, (mean, top) in shift.items():
            print(f"  {a:<7} moved {mean:7.2f} cm on average, {top:7.2f} cm at most")
        return

    algos = list(ALGORITHMS) + [CEP_ALL]
    for name, env, faults, setup in SCENARIOS:
        t0 = time.perf_counter()
        res = run(env, faults, setup, args.frames, args.seed)
        best = min(algos, key=lambda a: res[a][0])
        if args.markdown:
            print(f"\n**{name}** — RMS / 95th percentile error (cm), failed frames\n")
            print("| " + " | ".join(algos) + " |")
            print("|" + "---|" * len(algos))
            print("| " + " | ".join(
                (f"**{r:.1f}** / {q:.1f}" if a == best else f"{r:.1f} / {q:.1f}") + (f", {f:.0f}%" if f >= 0.5 else "")
                for a in algos for r, q, f in [res[a]]) + " |")
        else:
            print(f"\n{name}  ({time.perf_counter() - t0:.0f} s)")
            for a in algos:
                r, q, f = res[a]
                print(f"  {a:<7} RMS {r:7.1f}  p95 {q:7.1f}  failed {f:5.1f}%" + ("   <- best" if a == best else ""))


if __name__ == "__main__":
    main()
