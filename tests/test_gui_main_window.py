"""Main window, driven headless on a temp copy of config.toml: the runner
thread (edits between steps, the latest frame only, errors that pause it),
keyboard shortcuts versus focused controls, the dock layout, palettes, menus,
saving and replaying files, and what shows while paused or calibrating."""

import hashlib
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp(prefix="ultragps_xdg_")
try:
    from PyQt6.QtCore import QEvent, QObject, QPointF, QSettings, Qt, QTimer
    from PyQt6.QtGui import QKeyEvent, QMouseEvent
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import (QAbstractButton, QApplication, QCheckBox, QFileDialog, QMessageBox,
                                 QTabBar)
except ImportError as exc:
    raise unittest.SkipTest(f"PyQt6 is not installed: {exc}")

import numpy as np

from tests.helpers import temp_config
from engine.config import DEFAULT_CONFIG
from engine.hardware import PRESETS, STATUS, Environment
from engine.simulation import Simulation
from engine.world import Obstacle
from gui import theme
from gui.runner import SimRunner

app = QApplication.instance() or QApplication([])


def config_hash() -> str:
    with open(DEFAULT_CONFIG, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


REPO_CONFIG_HASH = config_hash()
SETTINGS_DIR     = tempfile.mkdtemp(prefix="ultragps_settings_")
OUT              = tempfile.mkdtemp(prefix="ultragps_out_")


class Dialogs:
    """Stands in for every file dialog and message box: answers come from
    ``save`` / ``open`` (lists, used in turn, the last one repeated) and
    ``answer``; every message shown is kept in ``shown``."""

    def __init__(self):
        self.save, self.open, self.shown = [], [], []
        self.answer = QMessageBox.StandardButton.Cancel

    def _next(self, answers):
        if not answers:
            return ("", "")
        return answers.pop(0) if len(answers) > 1 else answers[0]

    def install(self, case: unittest.TestCase) -> None:
        saved = {(QFileDialog, n): getattr(QFileDialog, n) for n in ("getSaveFileName", "getOpenFileName")}
        saved.update({(QMessageBox, n): getattr(QMessageBox, n) for n in ("warning", "information", "question")})
        QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: self._next(self.save))
        QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: self._next(self.open))
        for name in ("warning", "information"):
            setattr(QMessageBox, name, staticmethod(
                lambda parent, title, text, *a, _n=name, **k: self.shown.append((_n, title, text))
                or QMessageBox.StandardButton.Ok))
        QMessageBox.question = staticmethod(
            lambda parent, title, text, *a, **k: self.shown.append(("question", title, text)) or self.answer)
        for (cls, name), fn in saved.items():
            case.addCleanup(setattr, cls, name, fn)


def setUpModule():
    QSettings.setPath(QSettings.Format.NativeFormat, QSettings.Scope.UserScope, SETTINGS_DIR)
    theme.apply(app, "vector")


def tearDownModule():
    theme.apply(app, "vector")
    assert config_hash() == REPO_CONFIG_HASH, "a test wrote the repo's config.toml"


def pump(seconds: float = 0.0) -> None:
    end = time.perf_counter() + seconds
    app.processEvents()
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)


def wait_for(done, timeout: float = 20.0) -> bool:
    end = time.perf_counter() + timeout
    while not done():
        if time.perf_counter() > end:
            return False
        app.processEvents()
        time.sleep(0.005)
    return True


def make_window(restore: bool = False, seed: int = 1, size=(1500, 950)):
    from gui.main_window import MainWindow
    cfg = temp_config()
    sim = Simulation(cfg, seed=seed, env=Environment(**PRESETS["Realistic"]))
    win = MainWindow(sim, cfg, restore=restore)
    win.resize(*size)
    return win, sim


def frames(win, sim, n: int = 1) -> None:
    for _ in range(n):
        win.on_frame(sim.step(win.runner.dt))
    app.processEvents()


def mouse(widget, kind, x: float, y: float, buttons=Qt.MouseButton.LeftButton) -> None:
    pos = QPointF(x, y)
    QApplication.sendEvent(widget, QMouseEvent(kind, pos, widget.mapToGlobal(pos), Qt.MouseButton.LeftButton,
                                               buttons, Qt.KeyboardModifier.NoModifier))


def dispose(win) -> None:
    win.close()
    win.runner.stop()
    for w in (win.lab, win.calib_win, win.map_win, win.manual):
        if w is not None:
            w.deleteLater()
    win.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    app.processEvents()


class WindowTest(unittest.TestCase):
    """A shown main window, paused, with a few frames; dialogs stubbed."""

    restore = False

    def setUp(self):
        theme.apply(app, "vector")
        self.dialogs = Dialogs()
        self.dialogs.install(self)
        self.win, self.sim = make_window(self.restore)
        self.win.show()
        self.win.activateWindow()
        self.win.runner.paused = True
        frames(self.win, self.sim, 3)

    def tearDown(self):
        self.dialogs.answer = QMessageBox.StandardButton.Discard     # nothing asks to stay open
        dispose(self.win)
        theme.apply(app, "vector")


# ─── The runner ──────────────────────────────────────────────────────────────

class FailingSim:
    """A simulation whose step raises once, on the ``fail_at``-th call."""

    def __init__(self, fail_at: int = 2):
        self.calls, self.fail_at = 0, fail_at

    def step(self, dt):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("solver blew up")
        return self.calls


class RunnerTest(unittest.TestCase):
    def test_post_never_waits_for_a_step(self):
        runner = SimRunner(FailingSim(fail_at=0))
        done   = []
        runner.post(lambda: done.append("free"))                    # no step running: at once
        self.assertEqual(done, ["free"])
        held, release = threading.Event(), threading.Event()
        def long_step():
            with runner.lock:
                held.set()
                release.wait(5)
        worker = threading.Thread(target=long_step)
        worker.start()
        held.wait(5)
        start = time.perf_counter()
        runner.post(lambda: done.append("first"))
        runner.post(lambda: done.append("second"))
        self.assertLess(time.perf_counter() - start, 0.05)          # queued, not waited for
        self.assertEqual(done, ["free"])
        release.set()
        worker.join()
        self.assertEqual(runner.call(lambda: done.append("call") or len(done)), 4)   # posts run first
        self.assertEqual(done, ["free", "first", "second", "call"])

    def test_a_failing_step_pauses_the_runner_instead_of_killing_it(self):
        runner, errors = SimRunner(FailingSim(fail_at=2)), []
        runner.error.connect(errors.append)
        runner.frameReady.connect(lambda _f: runner.frame_shown())
        runner.dt = 0.01
        stderr, sys.stderr = sys.stderr, open(os.devnull, "w")       # the traceback is printed
        try:
            runner.start()
            self.assertTrue(wait_for(lambda: errors, 5))
        finally:
            sys.stderr.close()
            sys.stderr = stderr
        self.assertIn("solver blew up", errors[0])
        self.assertTrue(runner.paused)
        self.assertTrue(runner.isRunning())
        calls = runner.sim.calls
        runner.paused = False                                        # runs again
        self.assertTrue(wait_for(lambda: runner.sim.calls > calls + 2, 5))
        runner.stop()
        self.assertFalse(runner.isRunning())

    def test_only_the_latest_frame_is_sent_while_the_gui_is_busy(self):
        runner, got = SimRunner(FailingSim(fail_at=0)), []
        runner.frameReady.connect(got.append)
        runner.dt, runner.speed = 0.01, 8.0
        runner.start()
        try:
            self.assertTrue(wait_for(lambda: runner.sim.calls >= 20, 5))
            self.assertEqual(got, [1])                               # nobody took the first one yet
            runner.frame_shown()
            self.assertTrue(wait_for(lambda: len(got) == 2, 5))
            self.assertGreaterEqual(got[1], 20)                      # the newest, not the second
            runner.paused = True
            pump(0.05)
            last = runner.sim.calls
            runner.frame_shown()
            self.assertTrue(wait_for(lambda: len(got) == 3, 5))
            self.assertEqual(got[2], last)                           # the last before the pause too
        finally:
            runner.stop()


class RealRunnerTest(WindowTest):
    def test_a_few_seconds_of_the_real_runner_with_edits(self):
        win, sim = self.win, self.sim
        errors   = []
        win.runner.error.connect(errors.append)
        win.runner.paused = False
        win.start()
        self.assertTrue(wait_for(lambda: sim.frames >= 8, 20))
        arena = win.arena
        arena.set_mode("receiver")                                  # drag R2 a little
        q = arena.view.px(*sim.receivers[2].position)
        mouse(arena, QEvent.Type.MouseButtonPress, q.x(), q.y())
        for dx in (6, 12, 18):
            mouse(arena, QEvent.Type.MouseMove, q.x() + dx, q.y())
            pump(0.02)
        mouse(arena, QEvent.Type.MouseButtonRelease, q.x() + 18, q.y(), Qt.MouseButton.NoButton)
        arena.set_mode("move")
        win.set_motion("random")
        win.apply_preset("Harsh")
        win.toggle_receiver(3)
        win.place((20.0, 30.0))
        with win.runner.lock:
            sim.world.obstacles = [Obstacle.rect(-30, -30, -10, -10)]
        win.clear_obstacles()
        win.reset_stats()
        win.a_record.trigger()
        pump(1.0)
        win.set_motion("circle")
        win.toggle_receiver(3)
        self.dialogs.save = [(os.path.join(OUT, "live.csv"), "")]
        win.a_record.trigger()
        start = sim.frames
        self.assertTrue(wait_for(lambda: sim.frames >= start + 8, 20))
        pump(0.3)
        self.assertEqual(errors, [])
        self.assertTrue(win.runner.isRunning())
        self.assertEqual(sim.motion.mode, "circle")
        self.assertEqual(sim.world.obstacles, [])
        self.assertEqual(sim.env, Environment(**PRESETS["Harsh"]))
        self.assertFalse(np.allclose(sim.receivers[2].position, sim.believed[2]))
        self.assertTrue(os.path.exists(os.path.join(OUT, "live.csv")))
        win.toggle_run()
        self.assertTrue(win.runner.paused)
        # Paused, the last frame comes next: no backlog of old ones (it once grew past 400)
        self.assertTrue(wait_for(lambda: arena.frame.index == sim.last.index, 5))

    def test_an_error_in_a_step_shows_and_pauses(self):
        win = self.win
        win.runner.error.emit("Traceback ...\nRuntimeError: solver blew up")
        win.runner.paused = True
        self.assertIn("STOPPED: RuntimeError: solver blew up", win.arena.hud_text)
        self.assertIn("solver blew up", win.statusBar().currentMessage())
        self.assertEqual(win.a_run.text(), "Run")
        win.toggle_run()
        self.assertNotIn("STOPPED", win.arena.hud_text)


class ExceptHookTest(WindowTest):
    def test_an_exception_in_a_slot_is_shown_not_fatal(self):
        import app as app_module
        hook = sys.excepthook
        self.addCleanup(setattr, sys, "excepthook", hook)
        app_module.install_excepthook(self.win)
        stderr, sys.stderr = sys.stderr, open(os.devnull, "w")
        try:
            QTimer.singleShot(0, lambda: 1 / 0)                     # PyQt would abort without the hook
            pump(0.1)
        finally:
            sys.stderr.close()
            sys.stderr = stderr
        self.assertIn("ZeroDivisionError", self.win.statusBar().currentMessage())


# ─── Keyboard ────────────────────────────────────────────────────────────────

class ShortcutTest(WindowTest):
    def setUp(self):
        super().setUp()
        self.fired = []
        for a in self.win.findChildren(type(self.win.a_run)):
            if not a.shortcut().isEmpty():
                a.triggered.connect(lambda _=False, a=a: self.fired.append(a))

    def press(self, widget, key, modifier=Qt.KeyboardModifier.NoModifier):
        """The actions a key press on ``widget`` (given the focus) triggers."""
        self.fired.clear()
        widget.setFocus()
        app.processEvents()
        QTest.keyClick(widget, key, modifier)
        app.processEvents()
        return list(self.fired)

    def test_focused_controls_keep_their_keys(self):
        win  = self.win
        beam = next(w for w in win.envpanel.findChildren(QCheckBox) if w.text() == "Beam patterns")
        before = beam.isChecked()
        self.assertEqual(self.press(beam, Qt.Key.Key_Space), [])
        self.assertNotEqual(beam.isChecked(), before)
        self.assertTrue(win.runner.paused)                          # still paused: Space did not run
        slider = win.envpanel.forms[0].editors["temperature_c"].slider
        v0 = slider.value()
        self.assertEqual(self.press(slider, Qt.Key.Key_Right), [])
        self.assertEqual(slider.value(), v0 + 1)
        combo = win.scenario.form.editors["mode"]
        self.assertEqual(self.press(combo, Qt.Key.Key_R), [])        # not Reset stats ...
        self.assertEqual(combo.currentData(), "random")              # ... but Random waypoints
        button = next(b for b in win.algos.findChildren(QAbstractButton) if b.text() == "Reset stats")
        self.assertEqual(self.press(button, Qt.Key.Key_Space), [])

    def test_shortcuts_still_work_elsewhere(self):
        win = self.win
        self.assertEqual(self.press(win.arena, Qt.Key.Key_Space), [win.a_run])
        self.assertFalse(win.runner.paused)
        slider = win.envpanel.forms[0].editors["temperature_c"].slider
        self.assertEqual(self.press(slider, Qt.Key.Key_R), [win.a_reset])           # sliders ignore letters
        combo = win.scenario.form.editors["mode"]
        self.assertEqual(self.press(combo, Qt.Key.Key_1, Qt.KeyboardModifier.ControlModifier),
                         [win.rx_actions[1]])                                        # Ctrl keys always

    def test_the_key_filter_comes_off_when_the_loop_ends_or_the_window_closes(self):
        # PyQt's clean-up at exit crashes in an application filter still installed
        win = self.win
        beam = next(w for w in win.envpanel.findChildren(QCheckBox) if w.text() == "Beam patterns")
        def kept() -> bool:
            """Does a focused check box keep Space (the filter is installed)?"""
            ev = QKeyEvent(QEvent.Type.ShortcutOverride, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " ")
            ev.ignore()
            QApplication.sendEvent(beam, ev)
            return ev.isAccepted()
        self.assertTrue(kept())
        app.aboutToQuit.emit()                                       # what the end of app.exec() sends
        self.assertFalse(kept())
        app.installEventFilter(win._keys)
        self.assertTrue(kept())
        win.close()
        self.assertFalse(kept())

    def test_shortcuts_box_lists_each_key_once_with_its_meaning(self):
        win = self.win
        win.open_lab()                                              # its unnamed actions must not show
        win.show_shortcuts()
        box = win._shortcuts_box
        win.show_shortcuts()
        self.assertIs(win._shortcuts_box, box)                      # one box, shown again
        lines = [ln for ln in box.text().removeprefix("<pre>").removesuffix("</pre>").splitlines() if ln]
        keys  = [ln.split()[0] for ln in lines if "  " in ln]
        self.assertEqual(keys.count("Right"), 1)
        self.assertEqual(keys.count("Space"), 2)                    # the main window and the lab
        self.assertTrue(all(len(ln.split(None, 1)) == 2 for ln in lines if "  " in ln))
        self.assertIn("Esc", box.text())
        self.assertIn("Algorithm lab", box.text())


# ─── Layout, palettes, menus ─────────────────────────────────────────────────

def visible_tab_bars(win) -> list:
    """Tab bars with tabs, shown inside the window."""
    rect = win.rect()
    return [[tb.tabText(i) for i in range(tb.count())] for tb in win.findChildren(QTabBar)
            if tb.isVisible() and tb.count() and rect.intersects(tb.geometry())]


class LayoutTest(WindowTest):
    def test_reset_layout_rebuilds_the_docks_without_stray_tab_bars(self):
        win = self.win
        self.assertEqual(visible_tab_bars(win), [["Environment", "Faults and filter"]])
        win.d_scope.setFloating(True)
        win.d_receivers.close()
        win.tabifyDockWidget(win.d_scenario, win.d_algos)
        pump(0.2)
        win.resize(1100, 700)
        pump(0.2)
        for _ in range(2):
            win.reset_layout()
            pump(0.3)
            self.assertEqual(visible_tab_bars(win), [["Environment", "Faults and filter"]])
        for d in win.docks:
            self.assertTrue(d.isVisible(), d.objectName())
            self.assertFalse(d.isFloating(), d.objectName())
        self.assertGreater(win.d_scenario.height(), 150)            # not squashed to two rows
        self.assertGreater(win.d_chart.width(), 300)                # not left at its minimum width
        self.assertTrue(win.tabifiedDockWidgets(win.d_env) == [win.d_faults])

    def test_dock_titles_and_close_buttons(self):
        win = self.win
        self.assertEqual(win.d_faults.windowTitle(), "Faults and filter")      # no '&' to escape
        for d in win.docks:
            close = d.findChild(QAbstractButton, "qt_dockwidget_closebutton")
            self.assertIsNotNone(close)
            self.assertFalse(close.icon().isNull(), d.objectName())
            self.assertGreaterEqual(close.height(), 10, d.objectName())


class RestoredPaletteTest(WindowTest):
    restore = True

    def setUp(self):
        QSettings("UltraGPS", "Simulator").setValue("palette", "paper")
        super().setUp()

    def tearDown(self):
        super().tearDown()
        QSettings("UltraGPS", "Simulator").clear()

    def test_the_palette_menu_shows_the_restored_palette_and_switches(self):
        win = self.win
        self.assertEqual(theme.current.name, "paper")
        acts = win.palette_actions
        self.assertTrue(acts["paper"].isChecked())
        self.assertFalse(acts["vector"].isChecked())
        calls = []
        original = theme.apply
        def counting(*a, **k):
            calls.append(a[1] if len(a) > 1 else k.get("name"))
            return original(*a, **k)
        theme.apply = counting
        self.addCleanup(setattr, theme, "apply", original)
        acts["vector"].trigger()
        self.assertEqual(theme.current.name, "vector")
        self.assertEqual(calls, ["vector"])                         # applied once, not twice
        acts["vector"].trigger()                                     # already in use: nothing to do
        self.assertEqual(calls, ["vector"])
        self.assertTrue(acts["vector"].isChecked())


class MenuTest(WindowTest):
    def test_motion_and_preset_menus_tick_what_is_in_use(self):
        win = self.win
        win.set_motion("lawnmower")
        win._check_menus()
        self.assertTrue(win.motion_actions["lawnmower"].isChecked())
        self.assertTrue(win.preset_actions["Realistic"].isChecked())
        win.sim.env = win.sim.env.copy(snr_db=11.0)
        win._check_menus()
        self.assertTrue(win.preset_actions["Custom"].isChecked())
        self.assertFalse(win.preset_actions["Custom"].isEnabled())
        win.preset_actions["Ideal"].trigger()
        self.assertEqual(win.sim.env, Environment(**PRESETS["Ideal"]))
        win.motion_actions["circle"].trigger()
        self.assertEqual(win.sim.motion.mode, "circle")

    def test_every_status_has_a_colour_in_every_palette(self):
        for name, pal in theme.PALETTES.items():
            for status in STATUS:
                self.assertIn(status, pal.status, f"{name}: {status}")
            self.assertEqual(pal.status["no such status"], pal.status.fallback)

    def test_the_manual_links_follow_the_palette(self):
        win = self.win
        win.open_manual()
        pump(0.1)
        def link_colours():
            out, block = set(), win.manual.document().begin()
            while block.isValid():
                it = block.begin()
                while not it.atEnd():
                    f = it.fragment()
                    if f.isValid() and f.charFormat().anchorHref():
                        out.add(f.charFormat().foreground().color().name().lower())
                    it += 1
                block = block.next()
            return out
        self.assertEqual(link_colours(), {theme.PALETTES["vector"].accent.lower()})
        win.manual.verticalScrollBar().setValue(win.manual.verticalScrollBar().maximum() // 2)
        win.set_palette("paper")
        pump(0.2)
        self.assertEqual(link_colours(), {theme.PALETTES["paper"].accent.lower()})
        self.assertGreater(win.manual.verticalScrollBar().value(), 0)   # the reader keeps their place


# ─── Files ───────────────────────────────────────────────────────────────────

class RecordingTest(WindowTest):
    def record(self, n: int = 5) -> None:
        self.win.a_record.trigger()
        self.assertIsNotNone(self.sim.recording)
        frames(self.win, self.sim, n)

    def test_a_failed_save_keeps_the_recording_and_offers_again(self):
        win, d = self.win, self.dialogs
        gone = os.path.join(OUT, "no such folder", "rec.csv")
        good = os.path.join(OUT, "rec_again.csv")
        self.record(5)
        d.save = [(gone, ""), ("", "")]                             # fails, then the user cancels
        win.a_record.trigger()
        self.assertEqual(d.shown[-1][0], "warning")
        self.assertEqual(len(win._unsaved), 5)
        self.assertTrue(win.a_save_rec.isEnabled())
        self.assertIn("not saved", win.statusBar().currentMessage())
        d.save = [(good, "")]
        win.a_save_rec.trigger()
        with open(good) as fh:
            self.assertEqual(len(fh.read().splitlines()), 6)
        self.assertIsNone(win._unsaved)
        self.assertFalse(win.a_save_rec.isEnabled())

    def test_a_cancelled_save_keeps_the_recording_until_it_is_settled(self):
        win, d = self.win, self.dialogs
        self.record(4)
        d.save = [("", "")]
        win.a_record.trigger()
        self.assertEqual(len(win._unsaved), 4)
        d.answer = QMessageBox.StandardButton.Cancel                # new recording? not yet
        win.a_record.trigger()
        self.assertFalse(win.a_record.isChecked())
        self.assertIsNone(self.sim.recording)
        self.assertEqual(len(win._unsaved), 4)
        d.answer = QMessageBox.StandardButton.Discard
        win.a_record.trigger()
        self.assertTrue(win.a_record.isChecked())
        self.assertIsNone(win._unsaved)

    def test_quitting_with_a_recording_asks_first(self):
        win, d = self.win, self.dialogs
        self.record(3)
        d.answer = QMessageBox.StandardButton.Cancel
        win.close()
        self.assertTrue(win.isVisible())                            # stayed open
        self.assertEqual(len(win._unsaved), 3)
        self.assertEqual(d.shown[-1][0], "question")
        self.assertFalse(win.a_record.isChecked())                  # quitting stopped it ...
        self.assertTrue(win.a_save_rec.isEnabled())                 # ... and it can be saved from the menu
        self.assertIn("not saved", win.statusBar().currentMessage())
        good = os.path.join(OUT, "kept_on_quit.csv")
        d.save = [(good, "")]
        win.a_save_rec.trigger()
        self.assertTrue(os.path.exists(good))
        self.assertFalse(win.a_save_rec.isEnabled())

    def test_the_chosen_filter_and_suffix_decide_the_format(self):
        win, d = self.win, self.dialogs
        tick = "Base-station tick log (*.txt)"
        for name, flt, saved, ticks in (("run1", tick, "run1.txt", True), ("RUN2.TXT", tick, "RUN2.TXT", True),
                                        ("run3", "", "run3.csv", False), ("run4.csv", tick, "run4.csv", False)):
            self.record(2)
            d.save = [(os.path.join(OUT, name), flt)]
            win.a_record.trigger()
            with open(os.path.join(OUT, saved)) as fh:
                first = fh.readline()
            self.assertEqual(first.startswith("C: "), ticks, name)
            self.assertEqual(first.startswith("time,"), not ticks, name)

    def test_screenshot_svg_and_world_saves_report_failures(self):
        win, d = self.win, self.dialogs
        d.save = [(os.path.join(OUT, "shot"), "PNG (*.png)")]
        win.save_screenshot()
        self.assertTrue(os.path.exists(os.path.join(OUT, "shot.png")))
        gone = os.path.join(OUT, "no such folder")
        for fn, name in ((win.save_screenshot, "a.png"), (win.export_svg, "a.svg"), (win.save_world, "w.toml")):
            d.save = [(os.path.join(gone, name), "")]
            d.shown.clear()
            fn()
            self.assertEqual([m[0] for m in d.shown], ["warning"], name)
        d.save = [(os.path.join(OUT, "world"), "TOML (*.toml)")]
        win.save_world()
        self.assertTrue(os.path.exists(os.path.join(OUT, "world.toml")))


class ReplayTest(WindowTest):
    def write(self, name: str, text: str) -> str:
        path = os.path.join(OUT, name)
        with open(path, "w") as fh:
            fh.write(text)
        return path

    def test_short_rows_are_left_out_and_reported(self):
        win, d = self.win, self.dialogs
        d.open = [(self.write("mixed.txt", "C: 1500, 1400, 1300\n" + "C: 461, 301, 273, 486, 348, 337\n" * 3
                              + "C: 1, 2, 3, 4, 5, 6, 7, 8\n"), "")]
        win.open_replay()
        self.assertTrue(win.a_stop_replay.isEnabled())
        self.assertIn("2 lines without 6 values left out", win.statusBar().currentMessage())
        frames(win, self.sim, 4)
        self.assertEqual(win.receivers.table.item(1, 3).text(), "461")
        win.stop_replay()
        self.assertIsNone(self.sim.replay)

    def test_unreadable_logs_are_refused_with_a_message(self):
        win, d = self.win, self.dialogs
        for name, text in (("garbage.txt", "hello\n"), ("all_short.txt", "C: 1, 2, 3\n")):
            d.open = [(self.write(name, text), "")]
            d.shown.clear()
            win.open_replay()
            self.assertEqual(d.shown[-1][0], "warning", name)
            self.assertIsNone(self.sim.replay)
        d.open = [(os.path.join(OUT, "missing.txt"), "")]
        win.open_replay()
        self.assertEqual(d.shown[-1][0], "warning")

    def test_the_hud_survives_the_runner_stopping_the_replay(self):
        win, sim = self.win, self.sim
        sim.load_replay(self.write("hud.txt", "C: 461, 301, 273, 486, 348, 337\n" * 3))
        class Stopping:
            """The simulation, its replay stopped (on the runner) right after a first look."""
            recording, looks = None, 0
            @property
            def replay(self):
                self.looks += 1
                return sim.replay if self.looks == 1 else None
        win.sim = Stopping()
        try:
            self.assertIn("REPLAY hud.txt", win._hud())
        finally:
            win.sim = sim


# ─── What shows between frames ───────────────────────────────────────────────

class PausedTest(WindowTest):
    def paints(self, fn) -> int:
        """Arena repaints after ``fn()``."""
        count = [0]
        class Counter(QObject):
            def eventFilter(self, obj, event):
                if event.type() == QEvent.Type.Paint:
                    count[0] += 1
                return False
        counter = Counter()
        self.win.arena.installEventFilter(counter)
        pump(0.05)
        count[0] = 0
        fn()
        pump(0.15)
        self.win.arena.removeEventFilter(counter)
        return count[0]

    def test_edits_while_paused_repaint_and_refresh_the_panels(self):
        win, sim = self.win, self.sim
        sim.world.obstacles = [Obstacle.rect(-20, -20, 20, 20)]
        self.assertGreater(self.paints(win.clear_obstacles), 0)
        self.assertEqual(sim.world.obstacles, [])
        self.assertGreater(self.paints(lambda: win.apply_preset("Ideal")), 0)
        self.assertGreater(self.paints(lambda: win.receivers.survey.editors["sigma"].slider.setValue(10)), 0)
        self.assertGreater(self.paints(lambda: win.rx_actions[3].trigger()), 0)
        self.assertEqual(win.receivers.groups[3].checkedId(), 2)     # the panel shows R3 off at once
        self.assertGreater(self.paints(lambda: win.set_motion("lawnmower")), 0)
        self.assertEqual(win.scenario.form.editors["mode"].currentData(), "lawnmower")

    def test_a_focused_spin_box_shows_a_change_made_elsewhere(self):
        win  = self.win
        edit = win.envpanel.forms[0].editors["temperature_c"]
        edit.spin.setFocus()
        edit.spin.lineEdit().selectAll()
        QTest.keyClicks(edit.spin, "30")
        self.assertTrue(edit.editing())                             # typing: left alone
        QTest.keyClick(edit.spin, Qt.Key.Key_Return)
        self.assertEqual(self.sim.env.temperature_c, 30.0)
        self.assertTrue(edit.spin.hasFocus())
        win.apply_preset("Harsh")
        self.assertEqual(edit.value(), 20.0)

    def test_the_hud_follows_a_calibration_without_frames(self):
        win, cal = self.win, self.win.calibrator
        cal.state, cal.reads = "run 2", 37
        win._tick_hud()
        self.assertIn("CALIBRATING run 2 · 37 reads", win.arena.hud_text)
        cal.reads = 38
        win._tick_hud()
        self.assertIn("38 reads", win.arena.hud_text)
        cal.state = "done"
        win._tick_hud()
        self.assertNotIn("CALIBRATING", win.arena.hud_text)

    def test_recording_and_replay_show_in_the_hud(self):
        win = self.win
        win.a_record.trigger()
        frames(win, self.sim, 2)
        self.assertIn("REC 2", win.arena.hud_text)
        self.dialogs.answer = QMessageBox.StandardButton.Discard
        self.dialogs.save = [("", "")]
        win.a_record.trigger()
        self.assertNotIn("REC", win.arena.hud_text)


class ToolWindowTest(WindowTest):
    def test_reopening_the_lab_keeps_its_scenario(self):
        win = self.win
        win.open_lab()
        lab = win.lab
        loads = []
        lab.load_frame = lambda frame: loads.append(frame)
        frames(win, self.sim, 2)
        lab.close()
        win.open_lab()
        win.open_lab()
        self.assertIs(win.lab, lab)
        self.assertEqual(loads, [])                                 # only a new lab loads a frame

    def test_picked_calibration_points_stay_pending_until_used(self):
        win = self.win
        win.open_calibration()
        in_use = [list(p) for p in win.calibrator.points]
        win._start_cal_pick(2)
        win._on_cal_pick(-60.0, -100.0)
        win._on_cal_pick(50.0, 120.0)
        self.assertEqual(win.arena.cal_points, [(-60.0, -100.0), (50.0, 120.0)])
        self.assertTrue(win.arena.cal_pending)                      # not in config.toml yet
        self.assertEqual([list(p) for p in win.calibrator.points], in_use)
        win.calib_win.showMinimized()                               # hidden, but not closed
        pump(0.05)
        win.calib_win.showNormal()
        pump(0.05)
        self.assertEqual(win.arena.cal_points, [(-60.0, -100.0), (50.0, 120.0)])
        self.assertTrue(win.arena.cal_pending)
        win.calib_win.close()
        pump(0.05)
        self.assertEqual(win.arena.cal_points, [tuple(p) for p in in_use])   # back to config.toml's
        self.assertFalse(win.arena.cal_pending)

    def test_maps_can_send_contour_levels(self):
        win = self.win
        xs, ys = np.linspace(-100, 100, 5), np.linspace(-80, 80, 4)
        Z = np.add.outer(ys, xs) / 50.0
        win._show_map((xs, ys, Z, "test", -4.0, 4.0, [-1.0, 0.0, 1.0]))
        self.assertEqual(list(win.arena._heatmap.levels), [-1.0, 0.0, 1.0])
        win._show_map((xs, ys, Z, "old style", -4.0, 4.0))
        self.assertEqual(len(win.arena._heatmap.levels), 4)
        win._show_map(None)
        self.assertIsNone(win.arena._heatmap)


# ─── The application, start to exit ──────────────────────────────────────────

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# app.py --screenshot, telling what the window shows as it is grabbed
SCREENSHOT = """
import sys
from PyQt6.QtWidgets import QMainWindow
import app
grab = QMainWindow.grab
def spy(self, *a):
    print("HUD", repr(self.arena.hud_text))
    print("RATE", self.s_rate.text())
    return grab(self, *a)
QMainWindow.grab = spy
sys.argv = ["app.py"] + sys.argv[1:]
app.main()
"""

# app.py run as usual, the lab and the manual opened, then quit without closing the window
RUN_AND_QUIT = """
import sys
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
import app
run = QApplication.exec
def exec_(*_):
    win = next(w for w in QApplication.topLevelWidgets() if type(w).__name__ == "MainWindow")
    QTimer.singleShot(300, win.open_lab)
    QTimer.singleShot(600, win.open_manual)
    QTimer.singleShot(1500, QApplication.quit)
    return run()
QApplication.exec = exec_
sys.argv = ["app.py"] + sys.argv[1:]
app.main()
"""

# A script that makes a window and ends without closing it or running the event loop
LEFT_OPEN = """
import sys
from PyQt6.QtWidgets import QApplication
app = QApplication([])
from gui import theme
theme.apply(app, "vector")
from engine.simulation import Simulation
from gui.main_window import MainWindow
win = MainWindow(Simulation(sys.argv[2], seed=1), sys.argv[2], restore=False)
win.show()
for _ in range(5):
    win.on_frame(win.sim.step(0.1))
app.processEvents()
"""


class AppExitTest(unittest.TestCase):
    """The app in a process of its own exits cleanly: PyQt's clean-up at exit
    once crashed on the key filter the main window puts on the application."""

    def launch(self, script: str, *args: str):
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen", XDG_CONFIG_HOME=tempfile.mkdtemp(prefix="ultragps_xdg_"),
                   PYTHONDONTWRITEBYTECODE="1")
        done = subprocess.run([sys.executable, "-c", script, "--config", temp_config(), *args], cwd=REPO, env=env,
                              capture_output=True, text=True, timeout=180)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        return done.stdout

    def test_a_screenshot_exits_cleanly_and_shows_a_running_simulator(self):
        shot = os.path.join(OUT, "app_shot.png")
        out  = self.launch(SCREENSHOT, "--screenshot", shot, "--frames", "8", "--size", "1200x800")
        self.assertTrue(os.path.getsize(shot) > 0)
        self.assertIn("HUD ''", out)                                # not PAUSED
        self.assertIn("1.00x real time", out)                       # not 0.00x

    def test_quitting_with_tool_windows_open_exits_cleanly(self):
        self.launch(RUN_AND_QUIT)

    def test_a_script_leaving_the_window_open_exits_cleanly(self):
        self.launch(LEFT_OPEN)


if __name__ == "__main__":
    unittest.main()
