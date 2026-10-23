"""
Main window — the arena in the middle, every control in a dock you can move,
float, stack into tabs or close, and the tool windows (algorithm lab,
calibration, maps) one menu or shortcut away.  The layout is remembered
between runs; View > Reset layout puts it back.

The simulation runs on the runner's thread.  Changes made here go to it
through ``runner.post`` (applied between two steps, never waiting for one) or,
when the answer is needed at once, ``runner.call``.  The display takes the
latest frame whenever it is ready for one, so it never falls behind.
"""

from __future__ import annotations

import atexit
import os
import re
import time
import weakref

from PyQt6.QtCore import QByteArray, QEvent, QObject, QSettings, Qt, QTimer, QUrl
from PyQt6.QtGui import (QAction, QActionGroup, QImage, QKeySequence, QPixmap, QTextCharFormat, QTextCursor,
                         QTextDocument)
from PyQt6.QtWidgets import (QAbstractButton, QAbstractItemView, QAbstractSlider, QApplication, QComboBox,
                             QDockWidget, QFileDialog, QLabel, QMainWindow, QMessageBox, QTabBar, QTextBrowser,
                             QToolBar, QWidget)

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

PALETTE_NAMES = (("vector", "Vector (dark)"), ("paper", "Paper (light)"))

TOOL_HINTS = {"move": "Click to send the transmitter there",
              "obstacle": "Drag to draw an obstacle; right-click one to delete it",
              "receiver": "Drag a receiver to move it; scroll over it to turn it",
              "calpick": "Click the calibration points"}

# Keys and gestures without a menu item, for Help > Keyboard shortcuts
ARENA_KEYS = (("Esc", "back to Move (cancels picking points)"),
              ("Wheel", "zoom; in Receivers mode, over a receiver: turn it 5°"),
              ("Right/middle drag", "pan"),
              ("Double-click", "fit the room"))
LAB_KEYS   = (("Space", "play or pause"), ("Left / Right", "step back / forward"),
              ("Home / End", "first / last step"))

DOCK_LAYOUT = (1500, 950)      # window size the starting dock sizes were chosen for

NAV_KEYS = {Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Home,
            Qt.Key.Key_End, Qt.Key.Key_PageUp, Qt.Key.Key_PageDown}


def wants_key(widget, key, text: str) -> bool:
    """Does the focused ``widget`` use this key (no Ctrl, Alt or Meta) itself?"""
    if isinstance(widget, QComboBox):                 # Space opens it, arrows and letters choose
        return key == Qt.Key.Key_Space or key in NAV_KEYS or (text.isprintable() and text.strip() != "")
    if isinstance(widget, QAbstractButton):           # Space presses it
        return key == Qt.Key.Key_Space
    if isinstance(widget, (QAbstractSlider, QAbstractItemView, QTabBar)):
        return key in NAV_KEYS
    return False


class KeysToFocus(QObject):
    """Application event filter: a key the focused control uses goes to that
    control, not to a one-key shortcut.  Space toggles a focused check box
    rather than pausing, Right moves a focused slider rather than stepping, a
    letter picks an item in a focused combo box.  Letters still reach the
    shortcuts from buttons and sliders, which have no use for them; text and
    spin boxes keep their keys anyway."""

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.ShortcutOverride and isinstance(obj, QWidget):
            mods = event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
                                        | Qt.KeyboardModifier.MetaModifier)
            if not mods and wants_key(obj, event.key(), event.text()):
                event.accept()             # the widget gets the key press; the shortcut does not fire
        return False


def _drop_key_filter(keys) -> None:
    """Take a ``KeysToFocus`` off the application, if both are still there."""
    app = QApplication.instance()
    if keys is None or app is None:
        return
    try:
        app.removeEventFilter(keys)
    except RuntimeError:                   # deleted with its window already
        pass


class MainWindow(QMainWindow):
    def __init__(self, sim, config_path: str, restore: bool = True):
        super().__init__()
        self.sim         = sim
        self.config_path = config_path
        self.runner      = SimRunner(sim, self)
        self.calibrator  = Calibrator(sim, config_path, self.runner.lock)
        self.runner.hold = lambda: self.calibrator.busy
        self.lab = self.calib_win = self.map_win = self.manual = self._shortcuts_box = None
        self._picks: list | None = None
        self._unsaved: list | None = None      # the last recording, until it is saved
        self._error = ""                       # why the runner stopped itself
        self._last_panel_refresh = 0.0

        self.setWindowTitle("UltraGPS positioning simulator")
        self.resize(*DOCK_LAYOUT)
        self.setDockOptions(QMainWindow.DockOption.AnimatedDocks | QMainWindow.DockOption.AllowTabbedDocks
                            | QMainWindow.DockOption.AllowNestedDocks)

        self.arena = ArenaView(sim, self.runner.lock, post=self.runner.post)
        self.setCentralWidget(self.arena)
        self.arena.set_cal_points(self.calibrator.points)
        self._build_docks()
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_status_bar()
        self._connect()

        self._keys = KeysToFocus(self)
        app = QApplication.instance()
        app.installEventFilter(self._keys)
        app.aboutToQuit.connect(self.remove_key_filter)
        keys = weakref.ref(self._keys)      # atexit runs last in, first out: before PyQt's clean-up
        atexit.register(lambda: _drop_key_filter(keys()))
        self._scene_timer = QTimer(self, singleShot=True, interval=30, timeout=self._show_scene)
        self._hud_timer   = QTimer(self, interval=250, timeout=self._tick_hud)    # e.g. calibrating: no frames
        self._hud_timer.start()
        if restore:
            self._restore_layout()
        self.runner.frameReady.connect(self.on_frame)
        self.runner.error.connect(self._on_runner_error)
        self.runner.applied.connect(self.scene_changed)

    # ─── Layout ──────────────────────────────────────────────────────────────

    def _dock(self, title: str, name: str, widget: QWidget) -> QDockWidget:
        d = QDockWidget(title, self)
        d.setObjectName(name)
        d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                      | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                      | QDockWidget.DockWidgetFeature.DockWidgetClosable)
        return d

    def _build_docks(self) -> None:
        s, r = self.sim, self.runner
        self.scenario  = ScenarioPanel(s, r)
        self.receivers = ReceiverPanel(s, r)
        self.algos     = AlgorithmPanel(s, r)
        self.envpanel  = EnvironmentPanel(s, r)
        self.faults    = FaultsPanel(s, r)
        self.scope     = EchoScope()
        self.chart     = ErrorChart()
        self.panels    = (self.scenario, self.receivers, self.algos, self.envpanel, self.faults)

        self.d_scenario  = self._dock("Scenario", "scenario", self.scenario)
        self.d_receivers = self._dock("Receivers", "receivers", self.receivers)
        self.d_algos     = self._dock("Algorithms", "algorithms", self.algos)
        self.d_env       = self._dock("Environment", "environment", self.envpanel)
        self.d_faults    = self._dock("Faults and filter", "faults", self.faults)
        self.d_scope     = self._dock("Echo scope", "scope", self.scope)
        self.d_chart     = self._dock("Error", "error", self.chart)
        self.docks       = (self.d_scenario, self.d_receivers, self.d_algos, self.d_env, self.d_faults,
                            self.d_scope, self.d_chart)
        self._place_docks()

    def _place_docks(self) -> None:
        """Put every dock where it starts, sized for the window as it is now:
        Scenario and Receivers left, Algorithms and Environment / Faults (as
        tabs) right, the echo scope and the error chart along the bottom."""
        L, R, B = (Qt.DockWidgetArea.LeftDockWidgetArea, Qt.DockWidgetArea.RightDockWidgetArea,
                   Qt.DockWidgetArea.BottomDockWidgetArea)
        H, V    = Qt.Orientation.Horizontal, Qt.Orientation.Vertical
        for d, area in zip(self.docks, (L, L, R, R, R, B, B)):
            self.addDockWidget(area, d)
            d.show()
        self.tabifyDockWidget(self.d_env, self.d_faults)
        self.d_env.raise_()
        self.splitDockWidget(self.d_scope, self.d_chart, H)
        w, h = self.width() / DOCK_LAYOUT[0], self.height() / DOCK_LAYOUT[1]
        self.resizeDocks([self.d_scenario, self.d_algos], [round(330 * w), round(360 * w)], H)
        self.resizeDocks([self.d_scope, self.d_chart], [round(855 * w), round(645 * w)], H)
        self.resizeDocks([self.d_scope], [round(250 * h)], V)
        self.resizeDocks([self.d_scenario, self.d_receivers], [round(300 * h), round(330 * h)], V)

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

    def _choices(self, menu, items, slot, tip: str) -> dict:
        """Exclusive, checkable menu items ``[(key, text)]`` calling ``slot(key)``
        when chosen: ``{key: action}``."""
        group, out = QActionGroup(self), {}
        for key, text in items:
            a = self._action(text, tip=tip.format(text), checkable=True)
            a.triggered.connect(lambda _=False, k=key: slot(k))
            group.addAction(a)
            menu.addAction(a)
            out[key] = a
        return out

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
        self.a_save_rec = A("Save last recording...", self.save_recording,
                            tip="Save the last recording, kept until it is saved (or a new one replaces it)")
        self.a_save_rec.setEnabled(False)

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
        m.addAction(self.a_save_rec)
        m.addSeparator()
        m.addAction(A("Save world...", self.save_world, "Ctrl+S", "Save the walls and obstacles to a world.toml"))
        m.addAction(A("Clear obstacles", self.clear_obstacles, tip="Remove every obstacle"))
        m.addSeparator()
        m.addAction(A("Export arena as SVG...", self.export_svg, tip="Save the arena as a vector drawing"))
        m.addAction(A("Save screenshot...", self.save_screenshot, "Ctrl+P", tip="Save the window as a PNG"))
        m.addSeparator()
        m.addAction(A("Quit", self.close, "Ctrl+Q", "Close the simulator"))

        m = mb.addMenu("&Simulation")
        for a in (self.a_run, self.a_step, self.a_reset):
            m.addAction(a)
        m.addSeparator()
        motion = m.addMenu("Motion")
        self.motion_actions = self._choices(motion, [(md, MODE_NAMES[md]) for md in MODES], self.set_motion,
                                            "Transmitter motion: {}")
        presets = m.addMenu("Physics preset")
        self.preset_actions = self._choices(presets, [(n, n) for n in PRESETS] + [("Custom", "Custom")],
                                            self.apply_preset, "Replace every Environment setting with the "
                                            "{} preset")
        custom = self.preset_actions["Custom"]
        custom.setEnabled(False)                      # only ticked, when the environment matches no preset
        custom.setToolTip("The environment has been edited: it matches no preset")
        custom.setStatusTip(custom.toolTip())
        for menu in (motion, presets):
            menu.aboutToShow.connect(self._check_menus)
        m.addAction(A("Put transmitter in the centre", lambda: self.place(self.sim.believed.mean(axis=0)), "C",
                      "Move the transmitter to the middle of the receivers and keep it there"))

        m = mb.addMenu("&View")
        for d in self.docks:
            m.addAction(d.toggleViewAction())
        m.addSeparator()
        layers = m.addMenu("Arena layers")
        for key, text in LAYERS:
            layers.addAction(A(text, lambda on, k=key: self.arena.set_layer(k, on), tip=f"Show or hide: {text}",
                               checkable=True, checked=self.arena.layers.get(key, True)))
        pal = m.addMenu("Palette")
        self.palette_actions = self._choices(pal, PALETTE_NAMES, self.set_palette,
                                             "Switch every window to the {} palette")
        self.palette_actions[theme.current.name].setChecked(True)
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
        m.addAction(A("Keyboard shortcuts", self.show_shortcuts, tip="List the keyboard shortcuts"))

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
        a.targetClicked.connect(self._go_to)
        a.cursorMoved.connect(lambda x, y: self.s_cursor.setText(f"x {x:7.1f}  y {y:7.1f} cm"))
        a.modeChanged.connect(self._on_mode_changed)
        a.calPointPicked.connect(self._on_cal_pick)
        a.receiversEdited.connect(self._on_receivers_edited)
        a.worldEdited.connect(self.scene_changed)
        for panel in self.panels:
            panel.edited.connect(self.scene_changed)
        self.algos.shownChanged.connect(self._on_shown)
        self._on_shown(set(theme.ALGO_ORDER))

    # ─── Frames ──────────────────────────────────────────────────────────────

    def on_frame(self, frame) -> None:
        """A new frame from the runner: the arena every frame, the rest at most 10 times a second."""
        self.runner.frame_shown()             # the runner may send the next while this one is drawn
        self.arena.set_frame(frame)
        now = time.perf_counter()
        if now - self._last_panel_refresh < 0.1 and not self.runner.paused:
            return
        self._last_panel_refresh = now
        self._refresh(frame)

    def _refresh(self, frame) -> None:
        """Show ``frame`` in the panels, the scope, the chart and the status bar."""
        for panel in self.panels:
            panel.refresh(frame)
        self.scope.set_frame(frame, self.sim.env)
        self.chart.set_data(self.sim, self.shown)
        if frame is not None:
            rt = self.runner.real_rate * self.runner.dt
            self.s_time.setText(f"t {frame.time:7.1f} s   frame {frame.index}")
            self.s_rate.setText(f"{frame.compute_ms:5.1f} ms/frame   {rt:4.2f}x real time"
                                + ("   REPLAY" if self.sim.replay else "")
                                + ("   REC" if self.sim.recording is not None else ""))
        self.arena.set_hud(self._hud())

    def scene_changed(self) -> None:
        """Something changed the simulation outside a frame: redraw the arena
        now and, while paused (no frames are coming), the panels and scope too."""
        self.arena.refresh()
        if self.runner.paused:
            self._scene_timer.start()          # once for a burst of edits, e.g. a slider dragged

    def _show_scene(self) -> None:
        if self.runner.paused:
            self._refresh(self.sim.last)

    def _hud(self) -> str:
        parts = []
        if self._error:
            parts.append(f"STOPPED: {self._error}")
        elif self.runner.paused:
            parts.append("PAUSED")
        if self.calibrator.busy:
            parts.append(f"CALIBRATING {self.calibrator.state} · {self.calibrator.reads} reads")
        replay = self.sim.replay                    # read once: the runner may stop it meanwhile
        if replay:
            parts.append("REPLAY " + os.path.basename(replay.path))
        if self.sim.recording is not None:
            parts.append(f"REC {len(self.sim.recording)}")
        return "   ".join(parts)

    def _tick_hud(self) -> None:
        """Keep the HUD current between frames (while calibrating none come)."""
        text = self._hud()
        if text != self.arena.hud_text:
            self.arena.set_hud(text)

    def _on_runner_error(self, text: str) -> None:
        """A step (or a change posted to the runner) raised: the runner has paused."""
        lines = text.strip().splitlines()
        self._error = lines[-1] if lines else "error"
        self.a_run.setText("Run")
        self.statusBar().showMessage(f"The simulation stopped: {self._error} (details on the console). "
                                     "Run tries again.")
        self.arena.set_hud(self._hud())

    def show_error(self, text: str) -> None:
        """Report an error nothing else caught (see app.py's exception hook)."""
        lines = text.strip().splitlines()
        self.statusBar().showMessage(f"Error: {lines[-1] if lines else 'unknown'} (details on the console)",
                                     15000)

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
        if not self.runner.paused:
            self._error = ""
        self.a_run.setText("Run" if self.runner.paused else "Pause")
        self.arena.set_hud(self._hud())
        if self.runner.paused and self.sim.last is not None:
            self._refresh(self.sim.last)            # the panels may have skipped the last frame

    def step(self) -> None:
        self.runner.step_once()
        self.a_run.setText("Run")

    def toggle_receiver(self, i: int) -> None:
        r = self.sim.receivers[i]
        self.sim.set_receiver_state(i, "on" if r.state == "off" else "off")
        self.scene_changed()

    def reset_stats(self) -> None:
        self.runner.post(self.sim.reset)
        self.scene_changed()

    def set_motion(self, mode: str) -> None:
        self.runner.post(lambda: self.sim.motion.set_mode(mode))
        self.scenario.refresh(self.sim.last)
        self.scene_changed()

    def apply_preset(self, name: str) -> None:
        self.sim.env = Environment(**PRESETS[name])
        self.envpanel.refresh(self.sim.last)
        self.scene_changed()

    def place(self, xy) -> None:
        """Put the transmitter at ``xy``; while paused, run one frame to show it there."""
        xy = (float(xy[0]), float(xy[1]))
        self.runner.post(lambda: self.sim.place(xy))
        if self.runner.paused:
            self.step()
        self.scene_changed()

    def _go_to(self, x: float, y: float) -> None:
        self.runner.post(lambda: self.sim.motion.go_to((x, y)))
        self.scene_changed()

    def _on_receivers_edited(self) -> None:
        self.runner.post(lambda: self.sim.solvers.tracker.reset())
        self.scene_changed()

    def _check_menus(self) -> None:
        """Tick the motion and the physics preset in use (as a menu opens)."""
        a = self.motion_actions.get(self.sim.motion.mode)
        if a is not None:
            a.setChecked(True)
        now = self.sim.env.as_dict()
        self.preset_actions[next((n for n, v in PRESETS.items() if v == now), "Custom")].setChecked(True)

    # ─── Tool windows ────────────────────────────────────────────────────────

    def open_lab(self) -> None:
        """Show the algorithm lab; a new one starts from the latest frame (its
        own Load latest frame button takes a later one)."""
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
            self.calib_win.installEventFilter(self)
        self._show(self.calib_win)

    def open_maps(self) -> None:
        if self.map_win is None:
            self.map_win = MapWindow(self.sim, self.runner.lock, self)
            self.map_win.showOnArena.connect(self._show_map)
        self._show(self.map_win)

    def eventFilter(self, obj, event) -> bool:
        if (obj is self.calib_win and event.type() == QEvent.Type.Hide and not event.spontaneous()
                and self._picks is None):
            # Closed (not just minimized): points picked or typed but never used are
            # gone with the window, so show config.toml's
            self.arena.set_cal_points(self.calibrator.points)
        return super().eventFilter(obj, event)

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
        self.arena.set_cal_points(picks, pending=True)        # not in config.toml until "Use these points"
        if self.calib_win is not None:
            self.calib_win.set_points(picks)
            self._show(self.calib_win)

    # ─── Files ───────────────────────────────────────────────────────────────

    def _ask_save(self, title: str, start: str, filters: str) -> tuple[str, str]:
        """Ask where to save: ``(path, its suffix in lower case)``, or ``("", "")``.
        A name typed without a suffix gets the chosen filter's."""
        path, chosen = QFileDialog.getSaveFileName(self, title, start, filters)
        if not path:
            return "", ""
        if not os.path.splitext(path)[1]:
            m = re.search(r"\*(\.\w+)", chosen or filters)
            if m:
                path += m.group(1)
        return path, os.path.splitext(path)[1].lower()

    def _cannot_save(self, what: str, path: str, why) -> None:
        QMessageBox.warning(self, f"Save {what}", f"Could not save the {what} to\n{path}\n\n{why}")

    def open_replay(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Replay ticks", REPO,
                                              "Tick logs and recordings (*.txt *.log *.csv);;All files (*)")
        if not path:
            return
        try:
            replay = self.runner.call(lambda: self.sim.load_replay(path) or self.sim.replay)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Replay", f"Could not read {path}:\n{exc}")
            return
        self.a_stop_replay.setEnabled(True)
        skipped = getattr(replay, "skipped", 0)
        self.statusBar().showMessage(
            f"Replaying {len(replay.rows)} pings from {os.path.basename(path)}"
            + (f"; {skipped} lines without {len(self.sim.receivers)} values left out" if skipped else ""), 10000)
        self.scene_changed()

    def stop_replay(self) -> None:
        self.runner.post(self.sim.stop_replay)
        self.a_stop_replay.setEnabled(False)
        self.scene_changed()

    def toggle_recording(self, on: bool) -> None:
        if on:
            if not self._settle_recording("before recording again"):
                self.a_record.blockSignals(True)
                self.a_record.setChecked(False)
                self.a_record.blockSignals(False)
                return
            self.runner.post(self.sim.start_recording)
            self.arena.set_hud(self._hud())
            return
        records = self.runner.call(self.sim.stop_recording)
        self.arena.set_hud(self._hud())
        if records:
            self._unsaved = records
            self.save_recording()

    def save_recording(self) -> None:
        """Ask where to save the last recording until it is saved; one the user
        does not save is kept for File > Save last recording."""
        records = self._unsaved
        while records:
            path, ext = self._ask_save(f"Save {len(records)} recorded frames", os.path.join(REPO, "recording.csv"),
                                       "CSV with truth and estimates (*.csv);;Base-station tick log (*.txt)")
            if not path:
                break
            try:
                if ext in (".txt", ".log"):
                    self.sim.save_ticks(records, path)
                else:
                    self.sim.save_csv(records, path)
            except OSError as exc:
                self._cannot_save("recording", path, exc)
                continue
            self._unsaved = None
            self.statusBar().showMessage(f"Saved {len(records)} frames to {path}", 5000)
            break
        self._show_unsaved()

    def _show_unsaved(self) -> None:
        """File > Save last recording, and a reminder, while one is kept unsaved."""
        self.a_save_rec.setEnabled(bool(self._unsaved))
        if self._unsaved:
            self.statusBar().showMessage(f"The recording ({len(self._unsaved)} frames) is not saved yet: "
                                         "File > Save last recording saves it.")

    def _settle_recording(self, when: str) -> bool:
        """The last recording is about to be dropped: offer to save it ``when``.
        False when the user would rather not go on (it stays kept)."""
        if not self._unsaved:
            return True
        B = QMessageBox.StandardButton
        answer = QMessageBox.question(
            self, "Unsaved recording", f"The last recording ({len(self._unsaved)} frames) has not been saved. "
            f"Save it {when}?", B.Save | B.Discard | B.Cancel, B.Save)
        if answer == B.Save:
            self.save_recording()
            return not self._unsaved
        if answer == B.Discard:
            self._unsaved = None
            self.a_save_rec.setEnabled(False)
            return True
        self._show_unsaved()               # e.g. quitting stopped a recording that is now only kept
        return False

    def save_world(self) -> None:
        start = self.sim.world_path or os.path.join(REPO, "config", "world.toml")
        path, _ = self._ask_save("Save world", start, "TOML (*.toml)")
        if not path:
            return
        try:
            save_world(path, self.sim.world)
        except OSError as exc:
            self._cannot_save("world", path, exc)
            return
        self.statusBar().showMessage(f"Saved the walls and obstacles to {path}", 5000)

    def clear_obstacles(self) -> None:
        world = self.sim.world
        self.runner.post(lambda: setattr(world, "obstacles", []))
        self.scene_changed()

    def export_svg(self) -> None:
        path, _ = self._ask_save("Export arena", os.path.join(REPO, "arena.svg"), "SVG (*.svg)")
        if path and not self.arena.export_svg(path):
            self._cannot_save("drawing", path, "The file could not be written.")
        elif path:
            self.statusBar().showMessage(f"Saved the arena to {path}", 5000)

    def save_screenshot(self) -> None:
        path, _ = self._ask_save("Save screenshot", os.path.join(REPO, "screenshot.png"), "PNG (*.png)")
        if path and not self.grab().save(path):
            self._cannot_save("screenshot", path, "The file could not be written (or its type is unknown).")
        elif path:
            self.statusBar().showMessage(f"Saved the window to {path}", 5000)

    # ─── Look and layout ─────────────────────────────────────────────────────

    def set_palette(self, name: str) -> None:
        if name != theme.current.name:
            theme.apply(QApplication.instance(), name)
            for w in QApplication.topLevelWidgets():
                w.update()
                for child in w.findChildren(QWidget):
                    child.update()
        a = self.palette_actions.get(name)
        if a is not None:
            a.setChecked(True)           # the menu follows a restored or --palette choice too

    def _restore_layout(self) -> None:
        s = QSettings("UltraGPS", "Simulator")
        geo, state = s.value("geometry"), s.value("state")
        if isinstance(geo, QByteArray):
            self.restoreGeometry(geo)
        if isinstance(state, QByteArray):
            self.restoreState(state)
        name = s.value("palette", "")
        if name in theme.PALETTES:
            self.set_palette(name)

    def reset_layout(self) -> None:
        """Every dock back where it started, sized for the window as it is now.
        Rebuilt rather than restored from a saved state, which leaves stray
        tab bars behind once the window has been shown."""
        for d in self.docks:
            d.setFloating(False)
            self.removeDockWidget(d)
        self._place_docks()

    def open_manual(self) -> None:
        if self.manual is None:
            self.manual = ManualView()
        self._show(self.manual)

    def show_shortcuts(self) -> None:
        if self._shortcuts_box is None:
            own  = [(a.shortcut().toString(), "Run / pause" if a is self.a_run else a.text())
                    for a in self.findChildren(QAction, "", Qt.FindChildOption.FindDirectChildrenOnly)
                    if not a.shortcut().isEmpty() and a.text()]
            rows = ([("", "Main window")] + sorted(own)
                    + [("", ""), ("", "Arena (mouse and keys)")] + list(ARENA_KEYS)
                    + [("", ""), ("", "Algorithm lab")] + list(LAB_KEYS))
            text = "\n".join(f"{k:<18} {t}" if k else t for k, t in rows)
            box  = QMessageBox(self)
            box.setWindowTitle("Keyboard shortcuts")
            box.setText(f"<pre>{text}</pre>")
            box.setModal(False)
            self._shortcuts_box = box
        self._show(self._shortcuts_box)

    def start(self) -> None:
        self.runner.start()

    def remove_key_filter(self) -> None:
        """Take the KeysToFocus filter off the application.  It has to go
        before the application does: PyQt's clean-up deletes the objects left
        while the filter is still called for their events, and crashes.
        Closing the window, the end of the event loop and the exit do it; a
        script that ends with none of these (app.py --screenshot) calls this."""
        _drop_key_filter(self._keys)

    def closeEvent(self, event) -> None:
        if self.sim.recording is not None:            # recording until the end: keep what it has
            self._unsaved = self.runner.call(self.sim.stop_recording) or self._unsaved
            self.a_record.blockSignals(True)
            self.a_record.setChecked(False)
            self.a_record.blockSignals(False)
        if not self._settle_recording("before quitting"):
            event.ignore()
            return
        self.remove_key_filter()
        s = QSettings("UltraGPS", "Simulator")
        s.setValue("geometry", self.saveGeometry())
        s.setValue("state", self.saveState())
        s.setValue("palette", theme.current.name)
        self._hud_timer.stop()
        self.runner.stop()
        for w in (self.lab, self.calib_win, self.map_win, self.manual):
            if w is not None:
                w.close()
        super().closeEvent(event)


class ManualView(QTextBrowser):
    """MANUAL.md as the user reads it on GitHub: headings are link targets for
    the contents, and the pictures shrink to the window's width."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("UltraGPS simulator manual")
        self.setOpenExternalLinks(True)
        self.setSearchPaths([REPO])
        self.resize(900, 800)
        self._palette  = ""
        self._reload_due = False
        self._load()

    def _load(self) -> None:
        """Read MANUAL.md.  The Markdown reader colours the links with the
        palette in use, so a palette change reads it again."""
        self._palette = theme.current.name
        if not os.path.exists(MANUAL):
            self.setPlainText("MANUAL.md not found.")
            return
        with open(MANUAL, encoding="utf-8") as fh:
            self.setMarkdown(fh.read())
        block = self.document().begin()
        while block.isValid():
            if block.blockFormat().headingLevel():
                cur = QTextCursor(block)
                cur.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
                fmt = QTextCharFormat()
                fmt.setAnchor(True)
                fmt.setAnchorNames([_slug(block.text())])
                cur.mergeCharFormat(fmt)
            block = block.next()
        self._fit_images()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if (event.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange) and not self._reload_due
                and theme.current.name != self._palette):
            self._reload_due = True                  # theme.apply sends several: read once
            QTimer.singleShot(0, self._reload)

    def _reload(self) -> None:
        self._reload_due = False
        at = self.cursorForPosition(self.viewport().rect().topLeft()).position()
        self._load()
        cur = QTextCursor(self.document())
        cur.setPosition(min(at, self.document().characterCount() - 1))
        self.setTextCursor(cur)                      # back where the reader was
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        self.ensureCursorVisible()

    def _fit_images(self) -> None:
        width = max(self.viewport().width() - 30, 200)
        doc   = self.document()
        block = doc.begin()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                frag = it.fragment()
                if frag.isValid() and frag.charFormat().isImageFormat():
                    fmt   = frag.charFormat().toImageFormat()
                    image = doc.resource(QTextDocument.ResourceType.ImageResource, QUrl(fmt.name()))
                    if isinstance(image, (QImage, QPixmap)) and image.width():
                        fmt.setWidth(min(width, image.width()))
                        fmt.setHeight(min(width, image.width()) * image.height() / image.width())
                        cur = QTextCursor(doc)
                        cur.setPosition(frag.position())
                        cur.setPosition(frag.position() + frag.length(), QTextCursor.MoveMode.KeepAnchor)
                        cur.setCharFormat(fmt)
                it += 1
            block = block.next()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_images()


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading: "3.1 The arena" -> "31-the-arena"."""
    text = "".join(ch for ch in heading.lower() if ch.isalnum() or ch in " -_")
    return text.replace(" ", "-")
