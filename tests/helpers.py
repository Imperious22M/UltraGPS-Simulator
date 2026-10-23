import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.config import DEFAULT_CONFIG  # noqa: E402
from engine.hardware import Environment, Faults  # noqa: E402
from engine.simulation import Simulation  # noqa: E402


def temp_config() -> str:
    """Copy of config.toml in a temp dir, so calibration tests don't edit the repo's."""
    path = os.path.join(tempfile.mkdtemp(), "config.toml")
    shutil.copy(DEFAULT_CONFIG, path)
    return path


def ideal_sim(config=DEFAULT_CONFIG, noise=0.0, seed=0, **env) -> Simulation:
    """Straight-line model with Gaussian noise only, as in the first simulator."""
    return Simulation(config, seed=seed, env=Environment.ideal().copy(**env), faults=Faults(noise_cm=noise))


def run(sim: Simulation, pos, frames: int = 30):
    sim.place(pos)
    for _ in range(frames):
        sim.step()
    return sim.last
