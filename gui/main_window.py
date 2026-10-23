"""
Main window — the arena in the middle, every control in a dock you can move,
float, stack into tabs or close, and the tool windows (algorithm lab,
calibration, maps) one menu or shortcut away.  The layout is remembered
between runs; View > Reset layout puts it back.
"""

from __future__ import annotations

import os
import time

from PyQt6.QtCore import QByteArray, QSettings, Qt
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence
from PyQt6.QtWidgets import (QApplication, QDockWidget, QFileDialog, QLabel, QMainWindow,
                             QMessageBox, QTextBrowser, QToolBar, QWidget)

from engine.calibration import Calibrator
from engine.config import save_world
from engine.hardware import PRESETS, Environment
from engine.motion import MODES
from gui import theme
from gui.arena import ArenaView
from gui.calibration_window import CalibrationWindow
from gui.charts import EchoScope, ErrorChart
from gui.lab_window import LabWindow
from gui.map_window import MapWindow
from gui.panels import AlgorithmPanel, EnvironmentPanel, FaultsPanel, ReceiverPanel, ScenarioPanel
from gui.runner import SimRunner

REPO   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANUAL = os.path.join(REPO, "MANUAL.md")

LAYERS = (("grid", "Grid"), ("walls", "Walls and obstacles"), ("beams", "Beam patterns"),
          ("rings", "Measured ranges"), ("rays", "Sound paths"), ("waves", "Wavefronts"),
          ("trails", "Trails"), ("estimates", "Estimates"), ("ellipses", "Uncertainty ellipses"),
          ("labels", "Labels"), ("calpoints", "Calibration points"), ("legend", "Legend"),
          ("heatmap", "Map overlay"))

MODE_NAMES = {"target": "Click to go", "circle": "Circle", "figure8": "Figure eight",
              "lawnmower": "Lawnmower", "random": "Random waypoints"}

TOOL_HINTS = {"move": "Click to send the transmitter there",
              "obstacle": "Drag to draw an obstacle; right-click one to delete it",
              "receiver": "Drag a receiver to move it; scroll over it to turn it",
              "calpick": "Click the calibration points"}


class MainWindow(QMainWindow):
    def __init__(self, sim, config_path: str):
        super().__init__()
        self.sim         = sim
        self.config_path = config_path
        self.runner      = SimRunner(sim, self)
        self.calibrator  = Calibrator(sim, config_path)
        self.runner.hold = lambda: self.calibrator.busy
        self.lab = self.calib_win = self.map_win = self.manual = None
        self._picks: list | None = None
        self._last_panel_refresh = 0.0

        self.setWindowTitle("UltraGPS positioning simulator")
        self.resize(1500, 950)
        self.setDockOptions(QMainWindow.DockOption.AnimatedDocks | QMainWindow.DockOption.AllowTabbedDocks
                            | QMainWindow.DockOption.AllowNestedDocks)

        self.arena = ArenaView(sim, self.runner.lock)
        self.setCentralWidget(self.arena)
        self.arena.set_cal_points(self.calibrator.points)
        self._build_docks()
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_status_bar()
        self._connect()

        self._default_state = self.saveState()
        self._restore_layout()
        self.runner.frameReady.connect(self.on_frame)

    # ─── Layout ──────────────────────────────────────────────────────────────

    def _dock(self, title: str, name: str, widget: QWidget, area) -> QDockWidget:
        d = QDockWidget(title, self)
        d.setObjectName(name)
        d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                      | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                      | QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.addDockWidget(area, d)
        return d

    def _build_docks(self) -> None:
        L, R, B = (Qt.DockWidgetArea.LeftDockWidgetArea, Qt.DockWidgetArea.RightDockWidgetArea,
                   Qt.DockWidgetArea.BottomDockWidgetArea)
        s, r = self.sim, self.runner
        self.scenario  = ScenarioPanel(s, r)
        self.receivers = ReceiverPanel(s, r)
        self.algos     = AlgorithmPanel(s, r)
        self.envpanel  = EnvironmentPanel(s, r)
        self.faults    = FaultsPanel(s, r)
        self.scope     = EchoScope()
        self.chart     = ErrorChart()

        self.d_scenario  = self._dock("Scenario", "scenario", self.scenario, L)
        self.d_receivers = self._dock("Receivers", "receivers", self.receivers, L)
        self.d_algos     = self._dock("Algorithms", "algorithms", self.algos, R)
        self.d_env       = self._dock("Environment", "environment", self.envpanel, R)
        self.d_faults    = self._dock("Faults & filter", "faults", self.faults, R)
        self.d_scope     = self._dock("Echo scope", "scope", self.scope, B)
        self.d_chart     = self._dock("Error", "error", self.chart, B)
        self.tabifyDockWidget(self.d_env, self.d_faults)
        self.d_env.raise_()
        self.splitDockWidget(self.d_scope, self.d_chart, Qt.Orientation.Horizontal)
        self.resizeDocks([self.d_scenario, self.d_algos], [330, 360], Qt.Orientation.Horizontal)
        self.resizeDocks([self.d_scope], [250], Qt.Orientation.Vertical)

    def _action(self, text: str, slot=None, shortcut=None, tip: str = "", checkable: bool = False,
                checked: bool = False) -> QAction:
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if tip:
            a.setToolTip(f"{tip} ({a.shortcut().toString()})" if shortcut else tip)
            a.setStatusTip(tip)
        a.setCheckable(checkable)
        a.setChecked(checked)
        if slot is not None:
            (a.toggled if checkable else a.triggered).connect(slot)
        return a

    def _build_actions(self) -> None:
        A = self._action
        self.a_run    = A("Pause", self.toggle_run, "Space", "Run or pause the simulation")
        self.a_step   = A("Step", self.step, "Right", "Pause and advance exactly one frame")
        self.a_reset  = A("Reset stats", self.reset_stats, "R", "Clear error statistics and solver memory")
        self.a_fit    = A("Fit", self.arena.fit, "F", "Zoom the arena to fit the window")
        self.a_lab    = A("Algorithm lab", self.open_lab, "Ctrl+L",
                          "Step through what each algorithm does with the current frame")
        self.a_calib  = A("Calibration", self.open_calibration, "Ctrl+K",
                          "Calibration points, simulated calibration runs and offsets")
        self.a_maps   = A("Maps", self.open_maps, "Ctrl+M",
                          "Coverage, geometry, multipath and accuracy maps of the arena")
        self.a_record = A("Record", self.toggle_recording, "Ctrl+R",
                          "Record frames, then save them as CSV or as a base-station tick log",
                          checkable=True)

        self.rx_actions = [A(f"Switch R{i} on/off", lambda _=False, i=i: self.toggle_receiver(i), f"Ctrl+{i}",
                             f"Switch receiver R{i} on or off") for i in range(len(self.sim.receivers))]
        for a in self.rx_actions:
            self.addAction(a)

        self.tools = QActionGroup(self)
        self.tool_actions = {}
        for mode, text, key in (("move", "Move", "M"), ("obstacle", "Obstacles", "O"),
                                ("receiver", "Receivers", "E")):
            a = A(text, None, key, TOOL_HINTS[mode], checkable=True, checked=mode == "move")
            a.triggered.connect(lambda _=False, m=mode: self.arena.set_mode(m))
            self.tools.addAction(a)
            self.tool_actions[mode] = a

    def _build_menus(self) -> None:
        A, mb = self._action, self.menuBar()

        m = mb.addMenu("&File")
        m.addAction(A("Replay tick log...", self.open_replay, "Ctrl+O",
                      "Replay ticks from the base station or a recording instead of the simulated hardware"))
        self.a_stop_replay = A("Stop replay", self.stop_replay, tip="Go back to the simulated hardware")
        self.a_stop_replay.setEnabled(False)
        m.addAction(self.a_stop_replay)
        m.addAction(self.a_record)
        m.addSeparator()
        m.addAction(A("Save world...", self.save_world, "Ctrl+S", "Save the walls and obstacles to a world.toml"))
        m.addAction(A("Clear obstacles", self.clear_obstacles, tip="Remove every obstacle"))
        m.addSeparator()
        m.addAction(A("Export arena as SVG...", self.export_svg, tip="Save the arena as a vector drawing"))
        m.addAction(A("Save screenshot...", self.save_screenshot, "Ctrl+P", tip="Save the window as a PNG"))
        m.addSeparator()
        m.addAction(A("Quit", self.close, "Ctrl+Q"))

        m = mb.addMenu("&Simulation")
        for a in (self.a_run, self.a_step, self.a_reset):
            m.addAction(a)
        m.addSeparator()
        motion = m.addMenu("Motion")
        for mode in MODES:
            motion.addAction(A(MODE_NAMES[mode], lambda _=False, md=mode: self.set_motion(md)))
        presets = m.addMenu("Physics preset")
        for name in PRESETS:
            presets.addAction(A(name, lambda _=False, n=name: self.apply_preset(n)))
        m.addAction(A("Put transmitter in the centre", lambda: self.place(self.sim.believed.mean(axis=0)), "C"))

        m = mb.addMenu("&View")
        for d in (self.d_scenario, self.d_receivers, self.d_algos, self.d_env, self.d_faults,
                  self.d_scope, self.d_chart):
            m.addAction(d.toggleViewAction())
        m.addSeparator()
        layers = m.addMenu("Arena layers")
        for key, text in LAYERS:
            layers.addAction(A(text, lambda on, k=key: self.arena.set_layer(k, on), checkable=True,
                               checked=self.arena.layers.get(key, True)))
        pal = m.addMenu("Palette")
        group = QActionGroup(self)
        for name, text in (("vector", "Vector (dark)"), ("paper", "Paper (light)")):
            a = A(text, lambda _=False, n=name: self.set_palette(n), checkable=True,
                  checked=theme.current.name == name)
            group.addAction(a)
            pal.addAction(a)
        m.addSeparator()
        m.addAction(self.a_fit)
        m.addAction(A("Reset layout", self.reset_layout, tip="Put every dock back where it started"))

        m = mb.addMenu("&Tools")
        for a in (self.a_lab, self.a_calib, self.a_maps):
            m.addAction(a)
        m.addSeparator()
        for a in self.tools.actions():
            m.addAction(a)

        m = mb.addMenu("&Help")
        m.addAction(A("Manual", self.open_manual, "F1", "Open the simulator manual"))
        m.addAction(A("Keyboard shortcuts", self.show_shortcuts))

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main", self)
        tb.setObjectName("toolbar")
        tb.setMovable(False)
        self.addToolBar(tb)
        for a in (self.a_run, self.a_step, self.a_reset):
            tb.addAction(a)
        tb.addSeparator()
        for a in self.tools.actions():
            tb.addAction(a)
        tb.addAction(self.a_fit)
        tb.addSeparator()
        for a in (self.a_lab, self.a_calib, self.a_maps):
            tb.addAction(a)
        tb.addSeparator()
        tb.addAction(self.a_record)

    def _build_status_bar(self) -> None:
        sb = self.statusBar()
        self.s_hint   = QLabel(TOOL_HINTS["move"])
        self.s_cursor = QLabel("")
        self.s_time   = QLabel("")
        self.s_rate   = QLabel("")
        for w in (self.s_cursor, self.s_time, self.s_rate):
            w.setProperty("role", "dim")
            w.setFont(theme.mono_font(8.5))
        sb.addWidget(self.s_hint, 1)
        for w in (self.s_cursor, self.s_time, self.s_rate):
            sb.addPermanentWidget(w)

    def _connect(self) -> None:
        a = self.arena
        a.targetClicked.connect(lambda x, y: self.sim.motion.go_to((x, y)))
        a.cursorMoved.connect(lambda x, y: self.s_cursor.setText(f"x {x:7.1f}  y {y:7.1f} cm"))
        a.modeChanged.connect(self._on_mode_changed)
        a.calPointPicked.connect(self._on_cal_pick)
        a.receiversEdited.connect(lambda: self.sim.solvers.tracker.reset())
        self.algos.shownChanged.connect(self._on_shown)
        self._on_shown(set(theme.ALGO_ORDER))

    # ─── Frames ──────────────────────────────────────────────────────────────

    def on_frame(self, frame) -> None:
        """A new frame from the runner: the arena every frame, the rest at most 10 times a second."""
        self.arena.set_frame(frame)
        now = time.perf_counter()
        if now - self._last_panel_refresh < 0.1 and not self.runner.paused:
            return
        self._last_panel_refresh = now
        for panel in (self.scenario, self.receivers, self.algos, self.envpanel, self.faults):
            panel.refresh(frame)
        self.scope.set_frame(frame, self.sim.env)
        self.chart.set_data(self.sim, self.shown)
        rt = self.runner.real_rate * self.runner.dt
        self.s_time.setText(f"t {frame.time:7.1f} s   frame {frame.index}")
        self.s_rate.setText(f"{frame.compute_ms:5.1f} ms/frame   {rt:4.2f}x real time"
                            + ("   REPLAY" if self.sim.replay else "")
                            + ("   REC" if self.sim.recording is not None else ""))
        self.arena.set_hud(self._hud())

    def _hud(self) -> str:
        parts = []
        if self.runner.paused:
            parts.append("PAUSED")
        if self.calibrator.busy:
            parts.append(f"CALIBRATING {self.calibrator.state}")
        if self.sim.replay:
            parts.append("REPLAY " + os.path.basename(self.sim.replay.path))
        if self.sim.recording is not None:
            parts.append(f"REC {len(self.sim.recording)}")
        return "   ".join(parts)

    def _on_shown(self, shown: set) -> None:
        self.shown = set(shown)
        self.arena.set_algorithms(self.shown)

    def _on_mode_changed(self, mode: str) -> None:
        if mode in self.tool_actions:
            self.tool_actions[mode].setChecked(True)
        self.s_hint.setText(TOOL_HINTS.get(mode, ""))
        if mode != "calpick" and self._picks is not None:
            self._picks = None
            self.arena.set_cal_points(self.calibrator.points)

    # ─── Simulation control ──────────────────────────────────────────────────

    def toggle_run(self) -> None:
        self.runner.paused = not self.runner.paused
        self.a_run.setText("Run" if self.runner.paused else "Pause")
        self.arena.set_hud(self._hud())

    def step(self) -> None:
        self.runner.step_once()
        self.a_run.setText("Run")

    def toggle_receiver(self, i: int) -> None:
        r = self.sim.receivers[i]
        self.sim.set_receiver_state(i, "on" if r.state == "off" else "off")

    def reset_stats(self) -> None:
        with self.runner.lock:
            self.sim.reset()

    def set_motion(self, mode: str) -> None:
        self.sim.motion.set_mode(mode)
        self.scenario.refresh(self.sim.last)

    def apply_preset(self, name: str) -> None:
        self.sim.env = Environment(**PRESETS[name])
        self.envpanel.refresh(self.sim.last)

    def place(self, xy) -> None:
        with self.runner.lock:
            self.sim.place(xy)

    # ─── Tool windows ────────────────────────────────────────────────────────

    def open_lab(self) -> None:
        if self.lab is None:
            self.lab = LabWindow(self.sim, self.runner.lock, self)
        if self.sim.last is not None:
            self.lab.load_frame(self.sim.last)
        self._show(self.lab)

    def open_calibration(self) -> None:
        if self.calib_win is None:
            self.calib_win = CalibrationWindow(self.sim, self.calibrator, self.runner.lock, self)
            self.calib_win.pickRequested.connect(self._start_cal_pick)
            self.calib_win.pointsChanged.connect(lambda pts: self.arena.set_cal_points(pts))
        self._show(self.calib_win)

    def open_maps(self) -> None:
        if self.map_win is None:
            self.map_win = MapWindow(self.sim, self.runner.lock, self)
            self.map_win.showOnArena.connect(self._show_map)
        self._show(self.map_win)

    @staticmethod
    def _show(w) -> None:
        w.show()
        w.raise_()
        w.activateWindow()

    def _show_map(self, data) -> None:
        if data is None:
            self.arena.clear_heatmap()
        else:
            self.arena.set_heatmap(*data)

    def _start_cal_pick(self, n: int) -> None:
        self._picks, self._n_picks = [], n
        self.arena.set_cal_points([], pending=True)
        self.arena.set_mode("calpick")
        self.s_hint.setText(f"Click calibration point 1 of {n} (Esc cancels)")
        self.raise_()

    def _on_cal_pick(self, x: float, y: float) -> None:
        if self._picks is None:
            return
        self._picks.append((x, y))
        self.arena.set_cal_points(self._picks, pending=True)
        if len(self._picks) < self._n_picks:
            self.s_hint.setText(f"Click calibration point {len(self._picks) + 1} of {self._n_picks} (Esc cancels)")
            return
        picks, self._picks = self._picks, None
        self.arena.set_mode("move")
        self.arena.set_cal_points(picks)
        if self.calib_win is not None:
            self.calib_win.set_points(picks)
            self._show(self.calib_win)

    # ─── Files ───────────────────────────────────────────────────────────────

    def open_replay(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Replay ticks", REPO,
                                              "Tick logs and recordings (*.txt *.log *.csv);;All files (*)")
        if not path:
            return
        try:
            with self.runner.lock:
                self.sim.load_replay(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Replay", f"Could not read {path}:\n{exc}")
            return
        self.a_stop_replay.setEnabled(True)

    def stop_replay(self) -> None:
        with self.runner.lock:
            self.sim.stop_replay()
        self.a_stop_replay.setEnabled(False)

    def toggle_recording(self, on: bool) -> None:
        if on:
            self.sim.start_recording()
            return
        frames = self.sim.stop_recording()
        if not frames:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, f"Save {len(frames)} recorded frames", os.path.join(REPO, "recording.csv"),
            "CSV with truth and estimates (*.csv);;Base-station tick log (*.txt)")
        if not path:
            return
        if path.endswith(".txt"):
            self.sim.save_ticks(frames, path)
        else:
            self.sim.save_csv(frames, path)
        self.statusBar().showMessage(f"Saved {len(frames)} frames to {path}", 5000)

    def save_world(self) -> None:
        start = self.sim.world_path or os.path.join(REPO, "config", "world.toml")
        path, _ = QFileDialog.getSaveFileName(self, "Save world", start, "TOML (*.toml)")
        if path:
            save_world(path, self.sim.world)
            self.statusBar().showMessage(f"Saved the walls and obstacles to {path}", 5000)

    def clear_obstacles(self) -> None:
        with self.runner.lock:
            self.sim.world.obstacles = []

    def export_svg(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export arena", os.path.join(REPO, "arena.svg"),
                                              "SVG (*.svg)")
        if path:
            self.arena.export_svg(path)

    def save_screenshot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save screenshot", os.path.join(REPO, "screenshot.png"),
                                              "PNG (*.png)")
        if path:
            self.grab().save(path)

    # ─── Look and layout ─────────────────────────────────────────────────────

    def set_palette(self, name: str) -> None:
        theme.apply(QApplication.instance(), name)
        for w in QApplication.topLevelWidgets():
            w.update()
            for child in w.findChildren(QWidget):
                child.update()

    def _restore_layout(self) -> None:
        s = QSettings("UltraGPS", "Simulator")
        geo, state = s.value("geometry"), s.value("state")
        if isinstance(geo, QByteArray):
            self.restoreGeometry(geo)
        if isinstance(state, QByteArray):
            self.restoreState(state)
        name = s.value("palette", "")
        if name in theme.PALETTES and name != theme.current.name:
            self.set_palette(name)

    def reset_layout(self) -> None:
        self.restoreState(self._default_state)
        for d in self.findChildren(QDockWidget):
            if d.isFloating():
                d.setFloating(False)

    def open_manual(self) -> None:
        if self.manual is None:
            self.manual = QTextBrowser()
            self.manual.setWindowTitle("UltraGPS simulator manual")
            self.manual.setOpenExternalLinks(True)
            self.manual.setSearchPaths([REPO])
            self.manual.resize(900, 800)
            if os.path.exists(MANUAL):
                with open(MANUAL, encoding="utf-8") as fh:
                    self.manual.setMarkdown(fh.read())
            else:
                self.manual.setPlainText("MANUAL.md not found.")
        self._show(self.manual)

    def show_shortcuts(self) -> None:
        rows = [(a.text(), a.shortcut().toString()) for a in self.findChildren(QAction)
                if not a.shortcut().isEmpty()]
        text = "\n".join(f"{k:<14} {t}" for t, k in sorted(rows, key=lambda r: r[1]))
        box = QMessageBox(self)
        box.setWindowTitle("Keyboard shortcuts")
        box.setText(f"<pre>{text}</pre>")
        box.setModal(False)
        box.show()

    def start(self) -> None:
        self.runner.start()

    def closeEvent(self, event) -> None:
        s = QSettings("UltraGPS", "Simulator")
        s.setValue("geometry", self.saveGeometry())
        s.setValue("state", self.saveState())
        s.setValue("palette", theme.current.name)
        self.runner.stop()
        for w in (self.lab, self.calib_win, self.map_win, self.manual):
            if w is not None:
                w.close()
        super().closeEvent(event)
