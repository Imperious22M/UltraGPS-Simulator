"""
Benchmark — every algorithm through a set of standard conditions, as a table.

    python3 -m engine.benchmark [--frames 300] [--seed 1] [--markdown]

Each scenario drives the transmitter between random waypoints for ``frames``
frames (10 Hz) with a fixed seed, so runs are repeatable, and reports each
algorithm's RMS and 95th-percentile error (cm) and how often it failed to
produce a fresh fix.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from engine.hardware import PRESETS, Environment, Faults
from engine.simulation import Simulation
from engine.solvers import ALGORITHMS
from engine.world import Obstacle


def _box(sim):
    sim.world.obstacles = [Obstacle.rect(30, 30, 70, 70, name="box")]


def _r4_off(sim):
    sim.set_receiver_state(4, "off")


def _survey(sim):
    sim.survey_error(3.0, seed=7)


SCENARIOS = [
    ("Ideal (first simulator), noise 1 cm",  Environment.ideal(),                    Faults(1.0), None),
    ("Realistic room",                       Environment(),                          Faults(0.5), None),
    ("Realistic + 40 cm box in the arena",   Environment(),                          Faults(0.5), _box),
    ("Bad values 5 % (±50 %)",               Environment(),                          Faults(0.5, spike_rate=0.05), None),
    ("Dropouts 10 %, lost signal 5 %",       Environment(),                          Faults(0.5, dropout_rate=0.1, timeout_rate=0.05), None),
    ("Trigger jitter 100 µs",                Environment(sync_jitter_us=100.0),      Faults(0.5), None),
    ("Air at 30 °C (calibrated at 20 °C)",   Environment(temperature_c=30.0),        Faults(0.5), None),
    ("Receiver R4 off",                      Environment(),                          Faults(0.5), _r4_off),
    ("Receivers misplaced (σ 3 cm)",         Environment(),                          Faults(0.5), _survey),
    ("Harsh preset",                         Environment(**PRESETS["Harsh"]),        Faults(1.0, spike_rate=0.02), None),
]


def run(env, faults, setup, frames: int, seed: int) -> dict:
    sim = Simulation(seed=seed, env=env, faults=faults, history=frames)
    if setup:
        setup(sim)
    sim.place(sim.believed.mean(axis=0))
    sim.motion.set_mode("random")
    for _ in range(frames):
        sim.step(0.1)
    out = {}
    for a in ALGORITHMS:
        e = np.array([v for v in sim.errors[a] if v == v])
        out[a] = (float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan"),
                  float(np.percentile(e, 95)) if len(e) else float("nan"),
                  100.0 * sim.fails[a] / frames)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Every algorithm under standard conditions.")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--markdown", action="store_true", help="print Markdown tables")
    args = p.parse_args()

    algos = list(ALGORITHMS)
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
