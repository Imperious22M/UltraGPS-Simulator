"""
UltraGPS positioning simulator.

    python3 app.py [--config config/config.toml] [--sensors config/sensors.toml]
                   [--world config/world.toml] [--preset Realistic] [--palette vector]

Simulates the UltraGPS receivers, transmitter and base station, feeds the
ticks to the real positioning library and shows every algorithm side by side.
See MANUAL.md (Help > Manual) for what everything does.

    python3 app.py --screenshot shot.png [--frames 40] [--size 1500x950]

renders the window after a number of frames and exits (no display needed with
QT_QPA_PLATFORM=offscreen).
"""

from __future__ import annotations

import argparse
import sys

from PyQt6.QtWidgets import QApplication

from engine.config import DEFAULT_CONFIG, DEFAULT_SENSORS, DEFAULT_WORLD
from engine.hardware import PRESETS, Environment
from engine.simulation import Simulation
from gui import theme


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip())
    p.add_argument("--config", default=DEFAULT_CONFIG, help="config.toml with receivers and calibration")
    p.add_argument("--sensors", default=DEFAULT_SENSORS, help="sensors.toml with the hardware's true response")
    p.add_argument("--world", default=DEFAULT_WORLD, help="world.toml with walls and obstacles")
    p.add_argument("--preset", default="Realistic", choices=list(PRESETS), help="physics preset")
    p.add_argument("--palette", default=None, choices=list(theme.PALETTES), help="vector (dark) or paper (light)")
    p.add_argument("--max-differential", type=float, default=50.0, help="library jump filter, cm per frame")
    p.add_argument("--max-distance", type=float, default=None,
                   help="library top-band reject, cm (default: widest receiver spacing)")
    p.add_argument("--seed", type=int, default=None, help="random seed for repeatable runs")
    p.add_argument("--screenshot", help="render the window to this PNG and exit")
    p.add_argument("--frames", type=int, default=40, help="frames to run before --screenshot")
    p.add_argument("--size", default="1500x950", help="window size for --screenshot")
    p.add_argument("--motion", default="circle", help="motion for --screenshot")
    args = p.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("UltraGPS simulator")
    theme.apply(app, args.palette or "vector")

    from gui.main_window import MainWindow   # after the palette, so widgets pick it up
    sim = Simulation(args.config, args.sensors, args.world, args.max_differential, args.max_distance,
                     seed=args.seed, env=Environment(**PRESETS[args.preset]))
    win = MainWindow(sim, args.config)
    if args.palette:
        win.set_palette(args.palette)

    if args.screenshot:
        w, h = (int(v) for v in args.size.lower().split("x"))
        win.resize(w, h)
        win.reset_layout()
        win.show()
        sim.motion.set_mode(args.motion)
        win.runner.paused = True
        for _ in range(args.frames):
            win.on_frame(sim.step(win.runner.dt))
        app.processEvents()
        win.arena.set_frame(sim.last)
        win.grab().save(args.screenshot)
        return

    win.show()
    win.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
