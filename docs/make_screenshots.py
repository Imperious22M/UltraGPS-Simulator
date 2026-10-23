"""
Regenerate the pictures in MANUAL.md:

    QT_QPA_PLATFORM=offscreen python3 docs/make_screenshots.py [docs/img]

Each picture is a fixed, seeded scene, so a change to the drawing code shows up
as a change in the picture.  Calibration runs against a temporary copy of
config.toml; nothing in config/ is touched.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from PyQt6.QtWidgets import QApplication  # noqa: E402

from engine.calibration import Calibrator  # noqa: E402
from engine.hardware import PRESETS, Environment  # noqa: E402
from engine.simulation import Simulation  # noqa: E402
from engine.world import Obstacle  # noqa: E402
from gui import theme  # noqa: E402
from tests.helpers import temp_config  # noqa: E402

app = QApplication.instance() or QApplication([])
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "docs", "img")
SIZE = (1500, 950)


def pump(until=lambda: True, seconds: float = 30.0) -> None:
    """Let Qt run until ``until()`` or ``seconds`` pass."""
    end = time.perf_counter() + seconds
    app.processEvents()
    while not until() and time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def save(widget, name: str) -> None:
    path = os.path.join(OUT, name)
    widget.grab().save(path)
    print("saved", os.path.relpath(path, REPO))


def realistic_sim(config=None, seed=3) -> Simulation:
    args = (config,) if config else ()
    return Simulation(*args, seed=seed, env=Environment(**PRESETS["Realistic"]))


def main_window(palette: str, name: str, setup=None, frames: int = 60, motion: str = "circle") -> None:
    from gui.main_window import MainWindow
    theme.apply(app, palette)
    cfg = temp_config()
    sim = realistic_sim(cfg)
    win = MainWindow(sim, cfg, restore=False)
    win.resize(*SIZE)
    win.show()
    win.runner.paused = True
    sim.motion.set_mode(motion)
    if setup:
        setup(sim, win)
    for _ in range(frames):
        win.on_frame(sim.step(win.runner.dt))
    pump(seconds=0.2)
    win.on_frame(sim.last)
    save(win, name)
    win.close()


def obstacle_scene(sim, win) -> None:
    """A pillar between the transmitter and R1 (R1 hears the sound bend round
    it and reads long) and a box in front of R2 (R2 hears nothing and times out)."""
    sim.world.obstacles = [Obstacle.rect(-80, -10, -60, 10, name="pillar"),
                           Obstacle.rect(-60, 60, -20, 100, name="box")]
    sim.place((45, 15))


def lab_window() -> None:
    from gui.lab_window import LabWindow
    theme.apply(app, "vector")
    sim = realistic_sim()
    sim.place((30, 50))
    for _ in range(5):
        sim.step()
    win = LabWindow(sim)
    win.resize(*SIZE)
    win.show()
    win.load_frame(sim.last)

    # CEP: R2 reads 25 cm long; the subsets that leave it out win
    win.place_transmitter(30, 50)
    win.panel.set_overrides(["clean", "clean", "offset", "clean", "clean", "clean"], [0, 0, 25, 0, 0, 0])
    win.rebuild()
    win.tabs.setCurrentWidget(win.cep_tab)
    t = win.cep_tab
    t.ensure_computed()
    t.player.pause()
    t.player.seek(14, 0.6)
    pump(seconds=0.2)
    save(win, "lab-cep.png")

    # Blocking: R3 blocked, transmitter near it
    win.tabs.setCurrentWidget(win.blocking_tab)
    t = win.blocking_tab
    win.place_transmitter(100, -100)
    win.panel.set_overrides(["clean", "clean", "clean", "blocked", "clean", "clean"], [0] * 6)
    win.rebuild()
    t.ensure_computed()
    t.player.pause()
    t.player.seek(t.player.count - 1)
    pump(seconds=0.2)
    save(win, "lab-blocking.png")

    # Sweep: R1's reading pushed from -120 to +120 cm
    win.place_transmitter(30, 50)
    win.panel.set_overrides(["clean"] * 6, [0] * 6)
    win.rebuild()
    t.ensure_computed()
    t.sweep_steps = 31
    t.rx.setCurrentIndex(t.rx.findData(1))
    t.max_off.setValue(120)
    t.run_sweep()
    pump(lambda: t.sweep is not None, 60)
    t.player.pause()
    t.player.seek(t.player.count - 1)
    pump(seconds=0.2)
    save(win, "lab-sweep.png")

    # LM: start far out at the top left
    win.tabs.setCurrentWidget(win.lm_tab)
    t = win.lm_tab
    t.ensure_computed()
    t.set_start(-150, 200)
    t.player.pause()
    t.player.seek(t.player.count - 1)
    pump(seconds=0.2)
    save(win, "lab-lm.png")
    win.close()


def calibration_window() -> None:
    from gui.calibration_window import CalibrationWindow
    theme.apply(app, "vector")
    cfg = temp_config()
    sim = realistic_sim(cfg, seed=1)
    cal = Calibrator(sim, cfg)
    win = CalibrationWindow(sim, cal)
    win.resize(*SIZE)
    win.show()
    win.count.setValue(3)
    win.search()
    pump(lambda: win.search_job is None, 120)
    win.calibrate()
    pump(lambda: not cal.busy, 120)
    pump(seconds=0.3)
    win.receiver.setCurrentIndex(1)
    pump(seconds=0.2)
    save(win, "calibration.png")
    win.close()


def map_window() -> None:
    from gui.map_window import MapWindow
    theme.apply(app, "vector")
    sim = realistic_sim(seed=0)
    sim.world.obstacles = [Obstacle.rect(-70, -40, -20, 30, name="box")]
    win = MapWindow(sim)
    win.resize(*SIZE)
    win.show()
    for kind, name in (("nlos", "maps-multipath.png"), ("accuracy", "maps-accuracy.png")):
        win.set_kind(kind)
        if kind == "accuracy":
            win.step.setValue(15)
            win.trials.setValue(4)
        win.compute()
        pump(lambda: not win.busy(), 600)
        save(win, name)
    win.close()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main_window("vector", "main-vector.png")
    main_window("paper", "main-paper.png")
    main_window("vector", "obstacle.png", obstacle_scene, frames=8, motion="target")
    lab_window()
    calibration_window()
    map_window()
