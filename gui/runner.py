"""
Runner — steps the simulation on its own thread, so solving (the library's
CEP alone tries 42 subsets a frame) never freezes the windows.

``frameReady`` carries each new ``Frame`` to the GUI thread.  Anything that
changes the simulation in more than one assignment takes ``lock`` first, so it
never lands half-way through a step.
"""

from __future__ import annotations

import threading
import time

from PyQt6.QtCore import QThread, pyqtSignal


class SimRunner(QThread):
    frameReady = pyqtSignal(object)

    def __init__(self, sim, parent=None):
        super().__init__(parent)
        self.sim        = sim
        self.lock       = threading.RLock()
        self.dt         = 0.1          # simulated seconds per frame (the base station's 10 Hz)
        self.speed      = 1.0          # simulated seconds per real second
        self.paused     = False
        self.hold       = lambda: False   # e.g. calibrating: the hardware is busy elsewhere
        self._steps     = 0
        self._stop      = False
        self.real_rate  = 0.0          # frames per real second, measured

    def step_once(self) -> None:
        """Pause, then run exactly one frame."""
        self.paused = True
        self._steps += 1

    def stop(self) -> None:
        self._stop = True
        self.wait(3000)

    def run(self) -> None:
        due, last = time.perf_counter(), time.perf_counter()
        while not self._stop:
            if (self.paused and self._steps <= 0) or self.hold():
                time.sleep(0.01)
                due = time.perf_counter()
                continue
            with self.lock:
                frame = self.sim.step(self.dt)
            if self._steps > 0:
                self._steps -= 1
            self.frameReady.emit(frame)
            now = time.perf_counter()
            self.real_rate = 0.9 * self.real_rate + 0.1 / max(now - last, 1e-6)
            last = now
            due += self.dt / max(self.speed, 1e-3)
            wait = due - now
            if wait > 0:
                time.sleep(wait)
            else:
                due = now           # falling behind: run flat out
