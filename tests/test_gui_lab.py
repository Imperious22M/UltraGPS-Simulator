"""The algorithm lab window: its scenarios leave the simulator alone, each
story follows the library step by step, and every tab draws in both palettes."""

import os
import tempfile
import time
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import QEventLoop, QPoint, Qt
    from PyQt6.QtGui import QColor, QImage, QPainter
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import (QAbstractButton, QAbstractSlider, QAbstractSpinBox, QApplication, QComboBox,
                                 QDockWidget, QScrollBar)
except ImportError:                                  # pragma: no cover - depends on the machine
    raise unittest.SkipTest("PyQt6 is not installed")

import numpy as np

from engine.simulation import Simulation
from gui import draw, theme
from gui.lab_window import STATIC_ALGOS, LabWindow, Player, cep_switches, collinear, num
from ultragps_position import UltraGPSPositionLib

app = None


def setUpModule():
    global app
    app = QApplication.instance() or QApplication([])
    theme.apply(app, "vector")


def tearDownModule():
    theme.apply(app, "vector")


def make_window(pos=(30.0, 50.0), frames: int = 3):
    """A simulator that has run a few frames, and a lab window on it."""
    sim = Simulation(seed=1)
    sim.place(pos)
    for _ in range(frames):
        sim.step()
    win = LabWindow(sim)
    win.resize(1200, 800)
    return sim, win


def lab_scenario(win, pos=(30.0, 50.0), modes=None, offsets=None):
    """Measure a lab scenario at ``pos`` with the given overrides."""
    n = len(win.sim.receivers)
    win.place_transmitter(*pos)
    win.panel.set_overrides(modes or ["clean"] * n, offsets if offsets is not None else [0] * n)
    win.rebuild()
    return win.scenario


def wait_until(cond, seconds: float = 20.0) -> bool:
    t0 = time.time()
    while not cond() and time.time() - t0 < seconds:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
    return cond()


def ink(widget) -> float:
    """Share of a widget's pixels that differ from the palette background."""
    image = widget.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)
    ptr = image.constBits()
    ptr.setsize(image.sizeInBytes())
    px = np.frombuffer(ptr, np.uint8).reshape(image.height(), image.bytesPerLine() // 4, 4)[:, :image.width(), :3]
    bg = QColor(theme.current.bg)
    return float(np.mean(np.any(np.abs(px.astype(int) - (bg.blue(), bg.green(), bg.red())) > 8, axis=2)))


class ScenarioTest(unittest.TestCase):
    def test_the_simulators_library_is_never_touched(self):
        sim = Simulation(seed=1)
        lib = sim.solvers.lib
        lib.max_distance, lib.max_differential = 420.0, 33.0
        sim.place((30.0, 50.0))
        for _ in range(3):
            sim.step()
        before = (lib._last_distances.copy(), lib._cep_last_good_pos.copy())
        win = LabWindow(sim)
        win.load_frame(sim.last)
        for tab in win.tab_list:
            tab.ensure_computed()
        lab_scenario(win, (-60.0, 80.0), ["clean", "blocked", "offset", "clean", "off", "clean"], [0, 0, 40, 0, 0, 0])
        for tab in win.tab_list:
            tab.ensure_computed()
        self.assertIsNot(win.lib, lib)
        np.testing.assert_array_equal(lib._last_distances, before[0])
        np.testing.assert_array_equal(lib._cep_last_good_pos, before[1])
        self.assertEqual((win.lib.max_distance, win.lib.max_differential), (420.0, 33.0))
        win.close()

    def test_an_offset_adds_exactly_its_centimetres(self):
        _, win = make_window()
        clean = lab_scenario(win).distances.copy()
        moved = lab_scenario(win, modes=["clean", "clean", "offset", "clean", "clean", "clean"],
                             offsets=[0, 0, 25, 0, 0, 0]).distances
        self.assertAlmostEqual(moved[2] - clean[2], 25.0, places=6)
        np.testing.assert_allclose(np.delete(moved, 2), np.delete(clean, 2))
        win.close()

    def test_lab_readings_repeat_exactly(self):
        _, win = make_window()
        a = lab_scenario(win, (80.0, -40.0)).distances.copy()
        b = lab_scenario(win, (80.0, -40.0)).distances
        np.testing.assert_array_equal(a, b)
        win.close()

    def test_off_reads_nothing_and_blocked_hears_no_direct_sound(self):
        _, win = make_window()
        sc = lab_scenario(win, (100.0, -100.0), ["clean", "clean", "clean", "blocked", "off", "clean"])
        self.assertEqual(sc.ticks[4], 0)
        self.assertNotIn(4, sc.sane)
        self.assertIn(sc.status[3], ("echo", "bent", "lost"))
        self.assertEqual(sc.states[3], "blocked")
        self.assertIn("blocked", win.panel.statuses[3].text() + win.panel.modes[3].currentText().lower())
        win.close()

    def test_a_frame_snapshot_keeps_the_frames_numbers(self):
        sim, win = make_window()
        win.load_frame(sim.last)
        sc = win.scenario
        self.assertTrue(sc.source.startswith("frame"))
        np.testing.assert_allclose(sc.distances, sim.last.solution.distances)
        self.assertEqual(sc.sane, [int(i) for i in sim.last.solution.sane])
        np.testing.assert_allclose(sc.truth, sim.last.true)
        win.close()

    def test_every_control_has_a_tooltip(self):
        _, win = make_window()
        win.show()
        for tab in win.tab_list:
            win.tabs.setCurrentWidget(tab)
            for kind in (QAbstractButton, QAbstractSlider, QAbstractSpinBox, QComboBox):
                for w in tab.findChildren(kind) + win.panel.findChildren(kind):
                    if isinstance(w, QScrollBar) or not w.isVisibleTo(win) or isinstance(w.parent(), QDockWidget):
                        continue
                    if type(w).__name__ == "QTableCornerButton":
                        continue
                    self.assertTrue(w.toolTip(), f"{type(w).__name__} {getattr(w, 'text', lambda: '')()!r} "
                                                 "has no tooltip")
        win.close()


class CepTabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim, cls.win = make_window()
        lab_scenario(cls.win, modes=["clean", "clean", "offset", "clean", "clean", "clean"],
                     offsets=[0, 0, 25, 0, 0, 0])
        cls.tab = cls.win.cep_tab
        cls.tab.ensure_computed()

    @classmethod
    def tearDownClass(cls):
        cls.win.close()

    def test_steps_are_the_filter_the_librarys_subsets_and_validation(self):
        tr = self.tab.trace
        self.assertEqual(self.win.scenario.settings.cep_subsets, "control")
        self.assertEqual(len(tr.steps), 42)            # every subset of 3+ of 6 receivers is scored here ...
        self.assertEqual(tr.searched, 15)              # ... but the library, as UltraGPS-Control calls it, stops at 15
        self.assertEqual(self.tab.player.count, 1 + 15 + 1)
        self.assertEqual(self.tab.table.rowCount(), 15)

    def test_the_winner_is_the_librarys(self):
        sc  = self.win.scenario
        lib = UltraGPSPositionLib(self.sim.config_path)
        pos, cep, idx, _ = lib._find_best_subset(tuple(sc.distances), sc.sane, 15)
        best = self.tab.trace.steps[self.tab.trace.best]
        self.assertEqual(best.subset, idx)
        self.assertAlmostEqual(best.cep, cep)
        np.testing.assert_allclose(self.tab.trace.position, pos)
        self.assertNotIn(2, idx)                       # the reading 25 cm long is left out
        _, _, every, _ = lib._find_best_subset(tuple(sc.distances), sc.sane, 2 ** 6)
        self.assertEqual(self.tab.trace.steps[self.tab.trace.best_all].subset, every)

    def test_chart_and_table_jump_to_a_subset(self):
        self.tab.chart.picked.emit(5)
        self.assertEqual(self.tab.player.index, 6)
        self.tab.table.cellClicked.emit(3, 0)
        self.assertEqual(self.tab.player.index, 4)

    def test_rows_appear_as_the_story_goes(self):
        self.tab.player.seek(5)
        hidden = [self.tab.table.isRowHidden(r) for r in range(self.tab.table.rowCount())]
        self.assertEqual(hidden.index(True), 5)
        self.assertTrue(all(hidden[5:]))
        self.tab.player.last()
        self.assertFalse(any(self.tab.table.isRowHidden(r) for r in range(self.tab.table.rowCount())))

    def test_the_explanation_shows_the_equations_with_numbers(self):
        self.tab.player.seek(8)
        s    = self.tab.trace.steps[7]
        text = self.tab.text.toPlainText()
        for part in ("0.59", "JᵀWJ", num(s.cep), "1 / (|ri| + 0.1)"):
            self.assertIn(part, text)
        self.assertIn(f"subset 8 of {self.tab.trace.searched}", self.tab.bar.full_text)


class BlockingTabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim, cls.win = make_window()
        lab_scenario(cls.win, (100.0, -100.0), ["clean", "clean", "clean", "blocked", "clean", "clean"])
        cls.tab = cls.win.blocking_tab
        cls.tab.ensure_computed()

    @classmethod
    def tearDownClass(cls):
        cls.win.close()

    def test_steps_are_the_filter_each_algorithm_and_a_ranking(self):
        self.assertEqual(self.tab.steps, ["filter", *STATIC_ALGOS, "summary"])
        self.assertEqual(self.tab.player.count, 9)
        self.assertTrue(all(self.tab.fixes[a].pos is not None for a in STATIC_ALGOS))

    def test_cep_fix_matches_the_cep_story(self):
        self.win.cep_tab.ensure_computed()
        np.testing.assert_allclose(self.tab.fixes["CEP"].pos, self.win.cep_tab.trace.position)

    def test_ranking_names_the_best_and_worst(self):
        self.tab.player.last()
        errors = {a: self.tab._error(a) for a in STATIC_ALGOS}
        text = self.tab.bar.full_text
        self.assertIn(f"best {min(errors, key=errors.get)}", text)
        self.assertIn(f"worst {max(errors, key=errors.get)}", text)

    def test_a_sweep_runs_in_a_thread_and_is_replayed(self):
        sim, win = make_window()
        lab_scenario(win)
        tab = win.blocking_tab
        tab.ensure_computed()
        tab.sweep_steps = 7
        tab.rx.setCurrentIndex(tab.rx.findData(0))       # R0 is in CEP's winner at offset 0
        tab.max_off.setValue(60)
        tab.run_sweep()
        self.assertIsNotNone(tab._worker)
        self.assertFalse(tab.run_button.isEnabled())
        self.assertTrue(wait_until(lambda: tab.sweep is not None))
        tab.player.pause()
        self.assertEqual(len(tab.sweep.offsets), 7)
        self.assertEqual(tab.story, "sweep")
        self.assertEqual(tab.player.count, 7)
        self.assertTrue(tab.b_sweep.isEnabled() and tab.run_button.isEnabled())
        switches = cep_switches(tab.sweep)
        self.assertTrue(switches)                      # exact readings: CEP leaves R0 out at once
        self.assertRegex(switches[0][2], r"^CEP drops R0 .*at \+\d+ cm$")
        win.close()

    def test_a_sweep_of_an_old_scenario_is_dropped(self):
        sim, win = make_window()
        lab_scenario(win)
        tab = win.blocking_tab
        tab.ensure_computed()
        tab.sweep_steps = 5
        tab.run_sweep()
        lab_scenario(win, (0.0, 0.0))
        tab.ensure_computed()
        self.assertTrue(wait_until(lambda: tab._worker is None))
        self.assertIsNone(tab.sweep)
        self.assertEqual(tab.story, "fixes")
        win.close()


class LmTabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim, cls.win = make_window()
        lab_scenario(cls.win)
        cls.tab = cls.win.lm_tab
        cls.tab.ensure_computed()

    @classmethod
    def tearDownClass(cls):
        cls.win.close()

    def setUp(self):
        self.tab.b_many.setChecked(False)
        self.tab.receivers.setCurrentIndex(0)
        self.tab.reset_start()
        self.tab.player.pause()

    def test_lm_from_the_seed_ends_where_the_library_does(self):
        sc, lib = self.win.scenario, self.win.lib
        idx  = np.array(sc.sane)
        seed = lib._ols(sc.distances, idx)
        np.testing.assert_allclose(self.tab.trace.initial, seed)
        np.testing.assert_allclose(self.tab.trace.final, lib._nlls(sc.distances, seed, idx)[0], atol=1e-6)

    def test_a_click_sets_the_initial_guess(self):
        self.tab.set_start(-150.0, 200.0)
        self.tab.player.pause()
        np.testing.assert_allclose(self.tab.trace.initial, (-150.0, 200.0))
        x, _, kind = self.tab.points[0]
        self.assertEqual(kind, "start")
        np.testing.assert_allclose(x, (-150.0, 200.0))
        self.assertLess(float(np.hypot(*(self.tab.trace.final - self.win.scenario.truth))), 2.0)
        self.assertEqual(self.tab.table.rowCount(), len(self.tab.points))

    def test_probes_can_be_hidden(self):
        self.tab.c_probes.setChecked(False)
        self.assertNotIn("jacobian", [kind for _, _, kind in self.tab.points])
        self.tab.c_probes.setChecked(True)
        self.assertIn("jacobian", [kind for _, _, kind in self.tab.points])

    def test_explanation_shows_the_lm_update(self):
        self.tab.player.seek(min(3, self.tab.player.count - 1))
        text = self.tab.text.toPlainText()
        for part in ("JᵀJ", "least_squares", 'method="lm"', "MINPACK"):
            self.assertIn(part, text)

    def test_many_starts_find_one_minimum_with_every_receiver(self):
        self.tab.b_many.setChecked(True)
        self.tab.player.pause()
        self.assertEqual(len(self.tab.many), 36)
        self.assertEqual(len(self.tab.minima), 1)
        self.assertEqual(self.tab.minima[0][2], 36)

    def test_three_in_a_line_have_a_mirror_minimum(self):
        self.assertIsNotNone(self.tab.line_triple)
        self.tab.b_line.setChecked(True)
        self.tab.b_many.setChecked(True)
        self.tab.player.pause()
        self.assertEqual(self.tab.idx, list(self.tab.line_triple))
        self.assertTrue(collinear(self.win.scenario.coords[self.tab.idx]))
        (a, rms_a, _), (b, rms_b, _) = self.tab.minima[:2]
        self.assertLess(max(rms_a, rms_b), 0.5)
        self.assertGreater(float(np.hypot(*(a - b))), 50.0)
        self.assertIn("mirror", self.tab.text.toPlainText())
        self.tab.b_line.setChecked(False)
        self.assertEqual(self.tab.idx, self.win.scenario.sane)


class EdgeCaseTest(unittest.TestCase):
    def paint_every_step(self, win):
        """Describe, explain and paint every step of every tab straight into an
        image, so an error fails the test instead of aborting Qt."""
        image = QImage(480, 360, QImage.Format.Format_RGB32)
        view  = draw.View()
        view.resize(image.width(), image.height())
        view.fit(win.scenario.bounds())
        for tab in win.tab_list:
            tab.ensure_computed()
            for k in range(tab.player.count):
                tab.player.seek(k)
                self.assertTrue(tab.describe(k) and tab.html(k))
                p = QPainter(image)
                tab.paint_scene(p, view, k, 0.5)
                p.end()

    def test_fewer_than_three_receivers(self):
        _, win = make_window()
        lab_scenario(win, modes=["clean", "clean", "off", "off", "off", "off"])
        self.assertEqual(win.scenario.sane, [0, 1])
        self.paint_every_step(win)
        self.assertEqual(win.cep_tab.player.count, 2)              # filter and validation only
        self.assertIsNone(win.cep_tab.trace.best)
        self.assertEqual(win.lm_tab.steps, ["none"])
        self.assertTrue(all(win.blocking_tab.fixes[a].pos is None for a in ("CEP4", "ROBUST", "BIAS", "RANSAC")))
        self.assertFalse(win.blocking_tab.run_button.isEnabled())
        win.close()

    def test_a_replayed_frame_without_truth_or_readings(self):
        sim, win = make_window()
        path = os.path.join(tempfile.mkdtemp(), "ticks.txt")
        with open(path, "w") as fh:
            fh.write("C: " + ", ".join(str(t) for t in sim.last.ticks) + "\n")
        sim.load_replay(path)
        win.load_frame(sim.step())
        sc = win.scenario
        self.assertIsNone(sc.truth)
        self.assertEqual(sc.status, ["log"] * len(sim.receivers))
        self.assertIsNone(sc.error(0))
        self.paint_every_step(win)
        win.lm_tab.b_many.setChecked(True)
        self.paint_every_step(win)
        win.close()

    def test_a_double_click_fits_without_a_click(self):
        _, win = make_window()
        win.show()
        canvas = win.cep_tab.canvas
        before = win.tx.copy()
        QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 60))
        QTest.mouseDClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 60))
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 60))
        wait_until(lambda: False, 1.5 * QApplication.doubleClickInterval() / 1000)
        np.testing.assert_array_equal(win.tx, before)
        where = canvas.view.world(60, 60)
        QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 60))
        self.assertTrue(wait_until(lambda: not np.array_equal(win.tx, before), 2.0))
        np.testing.assert_allclose(win.tx, where)
        win.close()


class PlayerTest(unittest.TestCase):
    def test_steps_and_animation(self):
        p = Player()
        p.load(5, 1.0)
        self.assertEqual((p.index, p.phase), (0, 1.0))
        p.next()
        self.assertEqual(p.index, 1)
        self.assertEqual(p.phase, 0.0)
        p.advance(0.2)
        self.assertGreater(p.phase, 0.0)
        p.advance(10.0)
        self.assertEqual(p.phase, 1.0)
        p.prev()
        self.assertEqual((p.index, p.phase), (0, 1.0))
        p.last()
        self.assertEqual(p.index, 4)
        p.play()                                       # from the end it starts again
        self.assertTrue(p.playing)
        self.assertEqual(p.index, 0)
        for _ in range(200):
            p.advance(0.1)
        self.assertEqual(p.index, 4)
        self.assertFalse(p.playing)
        p.first()
        self.assertEqual(p.index, 0)


class PaintTest(unittest.TestCase):
    def test_every_tab_draws_in_both_palettes(self):
        sim, win = make_window()
        win.show()
        lab_scenario(win, modes=["clean", "clean", "offset", "clean", "clean", "clean"], offsets=[0, 0, 25, 0, 0, 0])
        try:
            for name in ("paper", "vector"):
                theme.apply(app, name)
                for tab in win.tab_list:
                    win.tabs.setCurrentWidget(tab)
                    tab.ensure_computed()
                    tab.player.pause()
                    for k, phase in ((tab.player.count // 2, 0.5), (tab.player.count - 1, 1.0)):
                        tab.player.seek(k, phase)
                        app.processEvents()
                        self.assertGreater(ink(tab.canvas), 0.03, f"{name} {type(tab).__name__} step {k}")
                    self.assertFalse(win.grab().isNull())
                app.processEvents()
                html = win.cep_tab.text.toHtml().lower()
                self.assertIn(QColor(theme.current.text).name(), html)   # re-rendered in this palette
        finally:
            theme.apply(app, "vector")
            win.close()


if __name__ == "__main__":
    unittest.main()
