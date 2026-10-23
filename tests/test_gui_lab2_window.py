"""The algorithm lab window in use: it solves a frame as the simulator did
under every solver setting, a sweep can be cancelled by closing the window,
the story keys leave focused controls alone, the step tables follow the story,
the CEP tab shows where the library's subset search stops, and the window
stays narrow enough to sit beside the main window."""

import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import QEvent, QEventLoop, QObject, QPoint, Qt
    from PyQt6.QtGui import QFontMetrics
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QComboBox, QTabBar
except ImportError:                                  # pragma: no cover - depends on the machine
    raise unittest.SkipTest("PyQt6 is not installed")

import numpy as np

from engine import lab
from engine.simulation import Simulation
from gui import theme
from gui.lab_window import STATIC_ALGOS, LabWindow, cep_switches, names
from tests.helpers import temp_config
from ultragps_position import UltraGPSPositionLib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app  = None


def setUpModule():
    global app
    app = QApplication.instance() or QApplication([])
    theme.apply(app, "vector")


def tearDownModule():
    theme.apply(app, "vector")


def pump(seconds: float = 0.05) -> None:
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)


def wait_until(cond, seconds: float = 20.0) -> bool:
    t0 = time.time()
    while not cond() and time.time() - t0 < seconds:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
    return cond()


def make_window(pos=(30.0, 50.0), size=(1200, 800), show: bool = False, **settings):
    sim = Simulation(temp_config(), seed=1)
    for k, v in settings.items():
        setattr(sim.solvers.settings, k, v)
    sim.place(pos)
    sim.step()
    win = LabWindow(sim)
    win.resize(*size)
    if show:
        win.show()
        win.activateWindow()
        pump(0.2)
    return sim, win


def lab_scenario(win, pos=(30.0, 50.0), modes=None, offsets=None):
    n = len(win.sim.receivers)
    win.place_transmitter(*pos)
    win.panel.set_overrides(modes or ["clean"] * n, offsets if offsets is not None else [0] * n)
    win.rebuild()
    for tab in win.tab_list:
        tab.player.pause()
    return win.scenario


def jumped_frame(filter_mode: str):
    """A replayed frame in which R1 reads 60 cm long: within the bounds, but the
    library's jump check drops it."""
    path = os.path.join(tempfile.mkdtemp(), "ticks.txt")
    sim  = Simulation(temp_config(), seed=1)
    sim.solvers.settings.filter_mode = filter_mode
    lib  = UltraGPSPositionLib(sim.config_path)
    d    = np.hypot(*(lib._receiver_coords - (100.0, -100.0)).T) + np.random.default_rng(2).normal(0.0, 1.0, 6)
    rows = []
    for extra in (0.0, 0.0, 60.0):
        dd = d.copy()
        dd[1] += extra
        rows.append(", ".join(str(round((v - lib._offsets[i]["intercept"]) / lib._offsets[i]["slope"]))
                              for i, v in enumerate(dd)))
    with open(path, "w") as fh:
        fh.write("".join(f"C: {r}\n" for r in rows))
    sim.load_replay(path)
    frame = [sim.step() for _ in rows][-1]
    return sim, frame


class SettingsTest(unittest.TestCase):
    def test_blocking_solves_the_frame_as_the_simulator_did_in_both_filter_modes(self):
        for mode in ("library", "bounds"):
            sim, f = jumped_frame(mode)
            self.assertNotIn(1, f.solution.sane)
            self.assertIn(1, f.solution.inbounds)
            win = LabWindow(sim)
            win.load_frame(f)
            tab = win.blocking_tab
            tab.ensure_computed()
            for a in STATIC_ALGOS:
                e = f.estimates[a]
                self.assertTrue(e.ok, f"{mode} {a}: {e.note}")
                np.testing.assert_allclose(tab.fixes[a].pos, e.pos, atol=1e-6, err_msg=f"{mode} {a}")
                self.assertEqual(sorted(int(i) for i in tab.fixes[a].used), sorted(int(i) for i in e.used),
                                 f"{mode} {a}")
            np.testing.assert_allclose(tab.fixes["CEP"].cov, f.estimates["CEP"].cov)
            sweepable = [tab.rx.itemData(i) for i in range(tab.rx.count())]
            self.assertEqual(1 in sweepable, mode == "bounds")    # Bounds only: R1 can be swept, for the others
            tab.player.last()
            self.assertEqual("R1, which the library's jump check dropped" in tab.text.toPlainText(),
                             mode == "bounds")
            win.close()

    def test_the_lab_takes_up_changed_solver_settings(self):
        sim, win = make_window()
        lab_scenario(win)
        win.cep_tab.ensure_computed()
        self.assertEqual(win.cep_tab.player.count, 1 + 15 + 1)
        sim.solvers.settings.cep_subsets = "all"
        win.follow_settings()                        # as when the window is activated again
        win.cep_tab.ensure_computed()
        self.assertEqual(win.scenario.settings.cep_subsets, "all")
        self.assertEqual(win.cep_tab.player.count, 1 + 42 + 1)
        win.cep_tab.player.last()
        self.assertIn("UltraGPS-Control (max_subsets = 15", win.cep_tab.text.toPlainText())
        win.close()

    def test_an_ekf_setting_alone_leaves_the_lab_as_it_is(self):
        sim, win = make_window()
        lab_scenario(win, (100.0, -100.0))
        tab = win.blocking_tab
        tab.ensure_computed()
        tab.sweep_steps = 5
        tab.run_sweep()
        self.assertTrue(wait_until(lambda: tab.sweep is not None))
        tab.player.pause()
        sc = win.scenario
        sim.solvers.settings.ekf_gate = 5.0          # the lab never uses it
        win.follow_settings()
        tab.ensure_computed()
        self.assertIs(win.scenario, sc)
        self.assertIsNotNone(tab.sweep)
        self.assertEqual(tab.story, "sweep")
        self.assertTrue(tab.b_sweep.isEnabled())
        sim.solvers.settings.ransac_tol_cm = 12.0    # one it does: the scenario is solved again
        win.follow_settings()
        tab.ensure_computed()
        self.assertEqual(win.scenario.settings.ransac_tol_cm, 12.0)
        self.assertIsNone(tab.sweep)
        win.close()


class CepLimitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim, cls.win = make_window(show=True)
        lab_scenario(cls.win, modes=["clean", "clean", "offset", "clean", "clean", "clean"],
                     offsets=[0, 0, 25, 0, 0, 0])
        cls.tab = cls.win.cep_tab
        cls.tab.ensure_computed()
        cls.tab.player.pause()

    @classmethod
    def tearDownClass(cls):
        cls.win.close()

    def test_the_validation_says_where_the_search_stops_and_what_lies_past_it(self):
        tr = self.tab.trace
        self.tab.player.last()
        text = self.tab.text.toPlainText()
        self.assertIn("first 15 of the 42 subsets", text)
        self.assertIn("never scores a 4-receiver or 5-receiver subset", text)
        if tr.best_all == tr.best:
            self.assertIn("would pick the same winner", text)
        else:
            self.assertIn(f"would pick {names(tr.steps[tr.best_all].subset)} instead", text)
        self.assertEqual((self.tab.chart.searched, len(self.tab.chart.steps)), (15, 42))

    def test_a_bar_past_the_search_is_not_a_step(self):
        chart = self.tab.chart
        self.tab.player.seek(3)
        xs, _ = chart._columns()
        QTest.mouseClick(chart, Qt.MouseButton.LeftButton, pos=QPoint(int(xs[30]), chart.height() // 2))
        self.assertEqual(self.tab.player.index, 3)
        QTest.mouseClick(chart, Qt.MouseButton.LeftButton, pos=QPoint(int(xs[5]), chart.height() // 2))
        self.assertEqual(self.tab.player.index, 6)

    def test_a_snapshot_without_a_last_good_fix_says_so(self):
        sim = Simulation(temp_config(), seed=1)
        sim.place((150.0, 0.0))                      # between the receivers' box and the walls, from the start
        f = sim.step()
        self.assertFalse(f.estimates["CEP"].ok)
        win = LabWindow(sim)
        win.load_frame(f)
        win.cep_tab.ensure_computed()
        win.cep_tab.player.last()
        text = win.cep_tab.text.toPlainText()
        self.assertTrue(win.cep_tab.trace.rejected)
        self.assertNotIn("returns its last good CEP fix", text)
        self.assertIn("no last good CEP fix in that frame", text)
        win.close()


class CepTextTest(unittest.TestCase):
    def test_the_texts_count_the_subsets_the_library_scores(self):
        sim, win = make_window()
        tab, three = win.cep_tab, ["clean", "off", "clean", "off", "off", "clean"]
        for modes, setting, at_filter, at_validation in (
                (None, "control", "The library's CEP scores the first 15 of the 42 subsets of 3 or more of them",
                 "The library scores the first 15 of the 42 subsets"),
                (None, "all", "The library's CEP scores every subset of 3 or more of them, all 42",
                 "The library scores all 42 subsets"),
                (three, "control", "The library's CEP has one subset to score: all three.",
                 "The library scores the only subset, all three kept receivers")):
            sim.solvers.settings.cep_subsets = setting
            lab_scenario(win, modes=modes)
            tab.ensure_computed()
            tab.player.pause()
            tab.player.seek(0)
            self.assertIn(at_filter, tab.text.toPlainText(), setting)
            self.assertNotIn("is a candidate", tab.text.toPlainText())
            tab.player.last()
            self.assertIn(at_validation, tab.text.toPlainText(), setting)
            self.assertNotIn("all 1 subsets", tab.text.toPlainText())
        win.close()


class TableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim, cls.win = make_window(show=True, cep_subsets="all")
        lab_scenario(cls.win, modes=["clean", "clean", "offset", "clean", "clean", "clean"],
                     offsets=[0, 0, 25, 0, 0, 0])

    @classmethod
    def tearDownClass(cls):
        cls.win.close()

    def assertRowInView(self, t, row):
        rect = t.visualItemRect(t.item(row, 0))
        self.assertTrue(rect.top() >= 0 and rect.bottom() <= t.viewport().height(),
                        f"row {row} at {rect.top()}..{rect.bottom()} of {t.viewport().height()}")
        self.assertEqual([i.row() for i in t.selectionModel().selectedRows()], [row])

    def test_the_cep_table_follows_jumps(self):
        tab = self.win.cep_tab
        self.win.tabs.setCurrentWidget(tab)
        tab.ensure_computed()
        tab.player.pause()
        self.assertGreater(tab.table.rowCount(), 30)
        tab.player.seek(23)
        pump()
        self.assertRowInView(tab.table, 22)
        tab.player.last()
        pump()
        self.assertRowInView(tab.table, tab.trace.best)

    def test_the_lm_table_follows_jumps(self):
        tab = self.win.lm_tab
        self.win.tabs.setCurrentWidget(tab)
        tab.ensure_computed()
        tab.player.last()
        pump()
        self.assertRowInView(tab.table, tab.player.count - 1)

    def test_columns_fit_every_row_from_the_first_step(self):
        sim, win = make_window(show=True)
        win.panel.set_overrides(["clean", "clean", "offset", "clean", "clean", "clean"], [0, 0, 25, 0, 0, 0])
        win.place_transmitter(30.0, 50.0)            # computed while shown, with every row hidden at first
        tab = win.cep_tab
        tab.player.pause()
        pump()
        tab.player.seek(1)
        pump()
        t, fm = tab.table, QFontMetrics(tab.table.font())
        for c in range(t.columnCount() - 1):
            need = max(fm.horizontalAdvance(t.item(r, c).text()) for r in range(t.rowCount()))
            self.assertGreaterEqual(t.columnWidth(c), need, f"column {c}")
        win.close()


class SweepTest(unittest.TestCase):
    def test_closing_the_window_cancels_a_sweep(self):
        sim, win = make_window(show=True)
        lab_scenario(win)
        tab = win.blocking_tab
        win.tabs.setCurrentWidget(tab)
        tab.ensure_computed()
        tab.max_off.setValue(300)                    # 61 offsets: seconds of work
        tab.run_sweep()
        worker = tab._worker
        win.close()
        self.assertTrue(worker.cancel.is_set())
        self.assertIsNone(tab._worker)
        pump(0.3)                                    # its result, if it sent one, is dropped
        self.assertIsNone(tab.sweep)
        self.assertEqual(tab.story, "fixes")
        self.assertFalse(any(t.player.playing for t in win.tab_list))
        win.show()
        pump()
        self.assertTrue(tab.run_button.isEnabled())
        self.assertEqual(tab.run_button.text(), "Run sweep")
        self.assertFalse(tab.busy.isVisible())
        self.assertNotIn("sweeping", tab.bar.full_text)
        win.close()

    def test_a_sweep_that_ends_out_of_sight_is_not_played(self):
        sim, win = make_window(show=True)
        lab_scenario(win)
        tab = win.blocking_tab
        win.tabs.setCurrentWidget(tab)
        tab.ensure_computed()
        tab.sweep_steps = 5
        tab.run_sweep()
        win.tabs.setCurrentWidget(win.cep_tab)
        self.assertTrue(wait_until(lambda: tab.sweep is not None))
        self.assertEqual(tab.story, "sweep")
        self.assertFalse(tab.player.playing)
        win.close()

    def test_the_sweep_shows_where_the_bounds_check_drops_the_reading(self):
        sim, win = make_window(show=True)
        lab_scenario(win, (100.0, -100.0))
        tab = win.blocking_tab
        win.tabs.setCurrentWidget(tab)
        tab.ensure_computed()
        tab.sweep_steps = 7
        tab.rx.setCurrentIndex(tab.rx.findData(2))
        tab.max_off.setValue(150)
        tab.run_sweep()
        self.assertTrue(wait_until(lambda: tab.sweep is not None))
        tab.player.pause()
        k = tab.sweep.dropped
        self.assertIsNotNone(k)
        self.assertIn("past the top band: the bounds check drops it", tab.describe(k))
        self.assertNotIn("past the top band", tab.describe(k - 1))
        tab.player.seek(k)
        text = tab.text.toPlainText()
        self.assertIn("past the top band, so the library's bounds check drops R2", text)
        self.assertIn("bounds check is applied again", text)
        self.assertFalse(tab.charts.grab().isNull())
        win.close()

    def test_a_sweep_that_leaves_fewer_than_three_receivers_says_cep_has_no_fix(self):
        sim, win = make_window()
        lab_scenario(win, (100.0, -150.0), modes=["clean", "off", "clean", "off", "off", "clean"])
        tab = win.blocking_tab
        tab.ensure_computed()
        tab.sweep_steps = 7
        tab.rx.setCurrentIndex(tab.rx.findData(2))
        tab.max_off.setValue(150)
        tab.run_sweep()
        self.assertTrue(wait_until(lambda: tab.sweep is not None))
        tab.player.pause()
        sw, k = tab.sweep, tab.sweep.dropped
        self.assertEqual((sw.kept[k - 1], sw.kept[k]), ([0, 2, 5], [0, 5]))
        at = f"+{sw.offsets[k]:.0f} cm"
        self.assertEqual([text for _, _, text in tab.switches],
                         [f"CEP has no fix at {at}: fewer than 3 receivers kept"])
        for j in (k, k + 1):
            self.assertIn("· CEP has no fix (fewer than 3 receivers kept)", tab.describe(j))
            self.assertNotIn("CEP uses none", tab.describe(j))
        tab.player.seek(k)
        self.assertIn("That leaves the library only R0 R5: with fewer than 3 receivers", tab.text.toPlainText())
        for j in range(len(sw.offsets)):            # every step draws with no fix to mark
            tab.player.seek(j)
            self.assertFalse(tab.canvas.grab().isNull())
        win.close()

    def test_losing_the_cep_fix_is_not_a_subset_choice(self):
        offsets = np.array([0.0, 25.0, 50.0, 75.0])
        lost = lab.Sweep(2, offsets, {}, [[0, 2, 5], [0, 2, 5], [], []], [True, True, False, False],
                         [True, True, False, False], 452.0, [[0, 2, 5], [0, 2, 5], [0, 5], [0, 5]])
        self.assertEqual(cep_switches(lost), [(2, 50.0, "CEP has no fix at +50 cm: fewer than 3 receivers kept")])
        back = lab.Sweep(2, offsets, {}, [[], [], [0, 1, 2], [0, 1, 2]], [True, True, True, True],
                         [True] * 4, 452.0, [[0, 1, 2]] * 4)
        self.assertEqual(cep_switches(back), [(2, 50.0, "CEP has a fix again at +50 cm: R0 R1 R2")])
        moved = lab.Sweep(2, offsets, {}, [[0, 1, 2, 3]] * 2 + [[0, 1, 3]] * 2, [False] * 3 + [True],
                          [True] * 4, 452.0, [[0, 1, 2, 3]] * 4)
        self.assertEqual([t for _, _, t in cep_switches(moved)],
                         ["CEP drops R2 at +50 cm", "CEP fix leaves the box at +75 cm"])


class KeysTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim, cls.win = make_window(show=True)
        cls.player = cls.win.cep_tab.player

    @classmethod
    def tearDownClass(cls):
        cls.win.close()

    def key(self, key) -> None:
        QTest.keyClick(self.win.windowHandle(), key)    # the way a real key press arrives
        pump()

    def setUp(self):
        self.win.activateWindow()                       # a test's own window may have taken it over
        pump()
        self.win.tabs.setCurrentWidget(self.win.cep_tab)
        self.player.pause()
        self.player.seek(0)

    def test_at_start_the_keys_step_the_story(self):
        sim, win = make_window(show=True)            # no click anywhere yet
        player = win.cep_tab.player
        player.pause()
        player.seek(0)
        self.assertIs(QApplication.focusWidget(), win.cep_tab.canvas)
        for key, index in ((Qt.Key.Key_Right, 1), (Qt.Key.Key_End, player.count - 1), (Qt.Key.Key_Home, 0)):
            QTest.keyClick(win.windowHandle(), key)
            pump()
            self.assertEqual(player.index, index, key)
        win.close()

    def test_a_focused_slider_keeps_the_arrows(self):
        panel = self.win.panel
        panel.set_overrides(["clean", "clean", "offset", "clean", "clean", "clean"], [0, 0, 25, 0, 0, 0])
        slider = panel.sliders[2]
        slider.setFocus()
        pump()
        self.key(Qt.Key.Key_Right)
        self.assertEqual(slider.value(), 26)
        self.assertEqual(self.player.index, 0)
        panel.set_overrides(["clean"] * 6, [0] * 6, emit=True)
        pump(0.3)

    def test_a_focused_check_box_keeps_space(self):
        box = self.win.panel.just_changed
        box.setFocus()
        pump()
        was = box.isChecked()
        self.key(Qt.Key.Key_Space)
        self.assertNotEqual(box.isChecked(), was)
        self.assertFalse(self.player.playing)
        box.setChecked(was)
        pump(0.3)

    def test_a_focused_tab_bar_keeps_only_the_arrows(self):
        bar = self.win.tabs.tabBar()
        bar.setFocus()
        pump()
        self.player.seek(3)
        self.key(Qt.Key.Key_End)                     # a tab bar has no use for Home and End
        self.assertEqual(self.player.index, self.player.count - 1)
        self.key(Qt.Key.Key_Home)
        self.assertEqual(self.player.index, 0)
        self.key(Qt.Key.Key_Right)                   # but the arrows change the tab
        self.assertIs(self.win.tabs.currentWidget(), self.win.blocking_tab)
        self.assertEqual(self.player.index, 0)
        self.assertIs(QApplication.focusWidget(), bar)
        self.win.blocking_tab.player.pause()

    def test_the_lab_has_the_last_word_over_an_application_wide_filter(self):
        class ClaimForCombos(QObject):               # as the main window's filter, on the whole application
            def eventFilter(self, obj, event):
                if event.type() == QEvent.Type.ShortcutOverride and isinstance(obj, (QComboBox, QTabBar)):
                    event.accept()
                return False

        claim = ClaimForCombos()
        app.installEventFilter(claim)
        try:
            combo = self.win.panel.modes[0]
            combo.setFocus()
            pump()
            self.key(Qt.Key.Key_Right)               # a combo has no use for Right: it steps the story
            self.assertEqual(self.player.index, 1)
            self.assertEqual(combo.currentData(), "clean")
        finally:
            app.removeEventFilter(claim)

    def test_a_click_on_the_canvas_hands_the_keys_back(self):
        self.win.panel.sliders[0].setFocus()
        canvas = self.win.blocking_tab.canvas
        self.win.tabs.setCurrentWidget(self.win.blocking_tab)
        pump()
        self.assertIs(QApplication.focusWidget(), canvas)    # a new tab: its story has the keys
        self.win.panel.sliders[0].setFocus()
        QTest.mousePress(canvas, Qt.MouseButton.RightButton, pos=QPoint(40, 40))
        QTest.mouseRelease(canvas, Qt.MouseButton.RightButton, pos=QPoint(40, 40))
        player = self.win.blocking_tab.player
        player.pause()
        player.seek(0)
        self.key(Qt.Key.Key_Right)
        self.assertEqual(player.index, 1)


class LayoutTest(unittest.TestCase):
    def test_the_scenario_dock_keeps_its_width_through_a_palette_change(self):
        sim, win = make_window(size=(900, 600), show=True)
        win.tabs.setCurrentWidget(win.lm_tab)
        win.lm_tab.ensure_computed()
        try:
            for name in ("paper", "vector"):
                theme.apply(app, name)
                pump(0.2)
                need = win.panel.minimumSizeHint().width()
                self.assertGreaterEqual(win.dock.widget().width(), need, name)
                self.assertLess(win.lm_tab.receivers.minimumSizeHint().width(), 180)
                self.assertIn("all kept", win.lm_tab.receivers.toolTip())   # the full text, where it is cut short
        finally:
            theme.apply(app, "vector")
            for tab in win.tab_list:
                tab.player.pause()
            win.close()

    def test_the_lm_controls_wrap_rather_than_widen_the_window(self):
        sim, win = make_window(size=(1400, 800), show=True)
        lm = win.lm_tab
        win.tabs.setCurrentWidget(lm)
        lm.ensure_computed()
        controls = [lm.receivers, lm.b_line, lm.b_many, lm.c_probes, lm.b_seed]

        def box(w):
            return w.geometry().translated(w.parentWidget().mapTo(lm, QPoint(0, 0)))

        try:
            pump(0.2)
            self.assertEqual(len({box(w).center().y() // 8 for w in controls}), 1)    # one line when wide
            self.assertLessEqual(lm.minimumSizeHint().width(), win.cep_tab.minimumSizeHint().width())
            self.assertLess(win.minimumSizeHint().width(), 800)
            win.resize(win.minimumSizeHint().width(), 600)
            pump(0.2)
            self.assertGreater(box(lm.b_seed).top(), box(lm.receivers).bottom())       # wrapped
            for w in controls:
                self.assertLessEqual(box(w).right(), lm.width(), w)
            self.assertGreaterEqual(box(lm.canvas).top(), max(box(w).bottom() for w in controls))
            self.assertGreaterEqual(win.dock.widget().width(), win.panel.minimumSizeHint().width())
        finally:
            for tab in win.tab_list:
                tab.player.pause()
            win.close()


class ExitTest(unittest.TestCase):
    def test_python_exits_cleanly_after_a_lab_window_is_deleted(self):
        script = textwrap.dedent("""
            import gc, time
            from PyQt6.QtCore import QEventLoop, QTimer
            from PyQt6.QtWidgets import QApplication
            app = QApplication([])
            from engine.simulation import Simulation
            from gui import theme
            from gui.lab_window import LabWindow
            from tests.helpers import temp_config
            theme.apply(app, "vector")

            def pump(seconds=0.2):
                t0 = time.time()
                while time.time() - t0 < seconds:
                    app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)

            sim = Simulation(temp_config(), seed=1)
            sim.place((30.0, 50.0))
            sim.step()
            win = LabWindow(sim)
            win.resize(800, 600)
            win.show()
            pump()
            win.tabs.setCurrentWidget(win.lm_tab)
            pump()
            for tab in win.tab_list:
                tab.player.pause()
            win.close()
            win.deleteLater()
            pump()
            del win
            gc.collect()
            pump()
            again = LabWindow(sim)          # still open at exit
            again.show()
            pump()
            QTimer.singleShot(200, app.quit)
            app.exec()
        """)
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen", XDG_CONFIG_HOME=tempfile.mkdtemp())
        run = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True, timeout=180)
        self.assertEqual(run.returncode, 0, run.stderr.decode(errors="replace")[-1500:])


if __name__ == "__main__":
    unittest.main()
