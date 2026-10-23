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

An exception nothing catches is printed and shown in the status bar; the
simulator carries on rather than quit.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from engine.config import DEFAULT_CONFIG, DEFAULT_SENSORS, DEFAULT_WORLD
from engine.hardware import PRESETS, Environment
from engine.simulation import Simulation
from gui import theme


class ErrorReporter(QObject):
    """Carries the traceback of an exception nothing caught to the GUI thread."""

    raised = pyqtSignal(str)


def install_excepthook(win) -> ErrorReporter:
    """Print any exception nothing caught and show it in ``win``'s status bar,
    instead of letting PyQt abort the application (its default for one raised
    in a slot or a thread).  The hook runs in the thread that raised, so it
    only emits a signal; the window shows the error in the GUI thread."""
    reporter = ErrorReporter(win)
    reporter.raised.connect(win.show_error)

    def hook(kind, value, tb) -> None:
        text = "".join(traceback.format_exception(kind, value, tb))
        sys.stderr.write(text)
        reporter.raised.emit(text)

    sys.excepthook = hook
    return reporter


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
    win = MainWindow(sim, args.config, restore=not args.screenshot)
    install_excepthook(win)
    if args.palette:
        win.set_palette(args.palette)

    if args.screenshot:
        w, h = (int(v) for v in args.size.lower().split("x"))
        win.resize(w, h)
        win.show()
        sim.motion.set_mode(args.motion)
        win.runner.paused = True              # the runner thread is not started: the frames are made here
        for _ in range(args.frames):
            win.on_frame(sim.step(win.runner.dt))
        # The last one as a running simulator shows it: no PAUSED, 1x real time
        win.runner.paused    = False
        win.runner.real_rate = 1.0 / win.runner.dt
        end = time.perf_counter() + 0.2       # past the panels' 0.1 s refresh limit
        while time.perf_counter() < end:
            app.processEvents()
            time.sleep(0.01)
        win.on_frame(sim.last)
        app.processEvents()
        win.grab().save(args.screenshot)
        win.remove_key_filter()               # the window stays open, and the application goes first
        return

    win.show()
    win.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
