"""
Runner — steps the simulation on its own thread, so solving (the library's
CEP alone can try 42 subsets a frame) never freezes the windows.

``frameReady`` carries a new ``Frame`` to the GUI thread, at most one at a
time: while the GUI is still busy with the last one, newer frames replace each
other and only the latest is sent once it calls ``frame_shown()``.  The error
chart and the trails still see every frame, through the simulation's history.

Anything that changes the simulation in more than one assignment either takes
``lock`` (and may wait for the step in progress) or hands the change to
``post(fn)``, which never waits: it runs ``fn`` at once if the simulation is
between steps, else queues it for the runner to run before its next step.

A step that raises does not end the thread: the runner pauses and emits
``error`` with the traceback.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from collections import deque

from PyQt6.QtCore import QThread, pyqtSignal


class SimRunner(QThread):
    frameReady = pyqtSignal(object)
    error      = pyqtSignal(str)       # traceback of a step or posted change that raised; now paused
    applied    = pyqtSignal()          # posted changes were applied while paused (no frame follows)

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
        self._posted    = deque()      # changes waiting for the step in progress to end
        self._latest    = None         # newest frame the GUI has not been sent yet
        self._taken     = threading.Event()     # the GUI is done with the last frame sent
        self._taken.set()
        self.real_rate  = 0.0          # frames per real second, measured

    def step_once(self) -> None:
        """Pause, then run exactly one frame."""
        self.paused = True
        self._steps += 1

    def stop(self) -> None:
        self._stop = True
        self.wait(3000)

    def post(self, fn) -> None:
        """Change the simulation with ``fn()`` between two steps, without
        waiting: at once if no step is running, else just before the next."""
        self._posted.append(fn)
        if self.lock.acquire(blocking=False):
            try:
                self._drain()
            finally:
                self.lock.release()

    def call(self, fn):
        """Run ``fn()`` under ``lock``, after every change posted before it, and
        return its result (waits for the step in progress to end)."""
        with self.lock:
            self._drain()
            return fn()

    def frame_shown(self) -> None:
        """The GUI has taken the last frame: the next one may be sent."""
        self._taken.set()

    def _drain(self) -> bool:
        """Run the posted changes, oldest first (the caller holds ``lock``)."""
        ran = False
        while self._posted:
            try:
                fn = self._posted.popleft()
            except IndexError:          # emptied by another thread meanwhile
                break
            fn()
            ran = True
        return ran

    def _offer(self) -> None:
        """Send the newest frame if the GUI is ready for it."""
        if self._latest is not None and self._taken.is_set():
            frame, self._latest = self._latest, None
            self._taken.clear()
            self.frameReady.emit(frame)

    def _failed(self) -> None:
        """A step or a posted change raised: report it and pause, but keep running."""
        text = traceback.format_exc()
        sys.stderr.write(text)
        self.paused, self._steps = True, 0
        self.error.emit(text)

    def run(self) -> None:
        due, last = time.perf_counter(), time.perf_counter()
        while not self._stop:
            self._offer()
            if (self.paused and self._steps <= 0) or self.hold():
                if self._posted:
                    try:
                        with self.lock:
                            if self._drain():
                                self.applied.emit()
                    except Exception:
                        self._failed()
                time.sleep(0.01)
                due = time.perf_counter()
                continue
            try:
                with self.lock:
                    self._drain()
                    self._latest = self.sim.step(self.dt)
            except Exception:
                self._failed()
                continue
            if self._steps > 0:
                self._steps -= 1
            self._offer()
            now = time.perf_counter()
            self.real_rate = 0.9 * self.real_rate + 0.1 / max(now - last, 1e-6)
            last = now
            due += self.dt / max(self.speed, 1e-3)
            wait = due - now
            if wait > 0:
                time.sleep(wait)
            else:
                # Falling behind: run flat out, but give the GUI thread the
                # interpreter until it takes the last frame (at most 20 ms).
                due = now
                self._taken.wait(0.02)
