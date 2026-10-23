"""Calibration and map windows: points, a calibration run, editing offsets,
the response plot, and computing, cancelling and showing maps.  Every
calibration writes to a temp copy of config.toml, never the repo's."""

import hashlib
import os
import time
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
try:
    from PyQt6.QtWidgets import QApplication
except ImportError as exc:
    raise unittest.SkipTest(f"PyQt6 is not installed: {exc}")

import numpy as np

from tests.helpers import ideal_sim, temp_config
from engine import cal_points, maps
from engine.calibration import Calibrator, read_offsets
from engine.config import DEFAULT_CONFIG
from engine.world import Obstacle
from gui import theme
from gui.calibration_window import CalibrationWindow, ResponsePlot, calibration_error
from gui.map_window import KINDS, MapWindow, make_map

app = QApplication.instance() or QApplication([])


def config_hash() -> str:
    with open(DEFAULT_CONFIG, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


REPO_CONFIG_HASH = config_hash()


def tearDownModule():
    theme.apply(app, "vector")
    assert config_hash() == REPO_CONFIG_HASH, "a test wrote the repo's config.toml"


class CountingLock:
    """Stands in for the runner's lock and counts how often it is taken."""

    def __init__(self):
        self.entered = 0

    def __enter__(self):
        self.entered += 1

    def __exit__(self, *exc):
        return False


def wait_for(done, timeout: float = 20.0) -> bool:
    """Process events until ``done()`` or the timeout."""
    end = time.perf_counter() + timeout
    while not done():
        if time.perf_counter() > end:
            return False
        app.processEvents()
        time.sleep(0.005)
    app.processEvents()
    return True


def not_blank(widget) -> bool:
    img = widget.grab().toImage()
    return len({img.pixel(x, y) for x in range(0, img.width(), 23) for y in range(0, img.height(), 23)}) > 5


class CalibrationWindowTest(unittest.TestCase):
    def setUp(self):
        theme.apply(app, "vector")
        self.config = temp_config()
        self.sim    = ideal_sim(self.config)
        self.cal    = Calibrator(self.sim, self.config)
        self.lock   = CountingLock()
        self.win    = CalibrationWindow(self.sim, self.cal, self.lock)
        self.win.resize(1100, 760)
        self.win.show()

    def tearDown(self):
        wait_for(lambda: not self.cal.busy and self.win.search_job is None)
        self.win.close()

    def test_shows_the_points_and_offsets_in_config(self):
        self.assertEqual(self.win.points(), [list(map(float, p)) for p in self.cal.points])
        self.assertEqual(self.win.points_table.rowCount(), len(self.cal.points))
        self.assertEqual(self.win.offsets(), read_offsets(self.config))
        score = cal_points.arena_score(self.cal.points, self.sim.believed)
        self.assertIn(f"{score:.3f}", self.win.score.text())
        self.assertTrue(not_blank(self.win))

    def test_set_points_then_use_them(self):
        pts, sent = [[-90.0, -100.0], [80.0, 120.0], [90.0, -140.0]], []
        self.win.pointsChanged.connect(sent.append)
        self.win.set_points(pts)
        self.assertEqual(self.win.count.value(), 3)
        self.assertEqual(self.win.points_table.rowCount(), 3)
        self.assertNotEqual(self.cal.points, pts)                 # not written yet
        self.win.use_btn.click()
        self.assertEqual(self.cal.points, pts)
        self.assertEqual(sent, [pts])
        self.assertFalse(self.win.use_btn.isEnabled())

    def test_count_and_editing_points(self):
        self.win.count.setValue(4)
        self.assertEqual(self.win.points_table.rowCount(), 4)
        self.win.points_table.item(3, 1).setText("55")
        self.assertEqual(self.win.points()[3][0], 55.0)
        self.win.points_table.item(3, 2).setText("not a number")
        self.assertEqual(self.win.points_table.item(3, 2).text(), f"{self.win.points()[3][1]:.1f}")
        self.win.count.setValue(2)
        self.assertEqual(len(self.win.points()), 2)

    def test_rejects_values_that_are_not_finite_numbers(self):
        before = self.win.offsets()[0]["slope"]
        for text in ("nan", "inf", "-inf"):
            self.win.offsets_table.item(0, 1).setText(text)
            self.win.points_table.item(0, 1).setText(text)
        self.assertEqual(self.win.offsets()[0]["slope"], before)
        self.assertTrue(all(np.isfinite(self.win.points()).flat))
        self.assertTrue(not_blank(self.win.plot))

    def test_set_points_keeps_two_to_four(self):
        self.win.set_points([[10.0, 20.0]])
        self.assertEqual(len(self.win.points()), 2)
        self.assertEqual(self.win.count.value(), 2)
        self.win.set_points([[float(k), 0.0] for k in range(6)])
        self.assertEqual(len(self.win.points()), 4)
        self.assertEqual(self.win.count.value(), 4)

    def test_pick_asks_the_main_window_for_points(self):
        asked = []
        self.win.pickRequested.connect(asked.append)
        self.win.count.setValue(3)
        self.win.pick_btn.click()
        self.assertEqual(asked, [3])

    def test_search_runs_in_the_background(self):
        self.win.count.setValue(2)
        self.win.search_btn.click()
        self.assertTrue(wait_for(lambda: self.win.search_job is None))
        best = cal_points.search(self.sim.believed, 2)[2]
        self.assertTrue(np.allclose(self.win.points(), best))
        self.assertTrue(self.win.search_btn.isEnabled())

    def test_calibrate_applies_offsets_and_shows_the_run(self):
        version = self.cal.version
        self.win.set_points([[-90.0, -100.0], [80.0, 120.0]])
        self.win.calibrate_btn.click()
        self.assertEqual(self.lock.entered, 1)
        self.assertEqual(self.cal.points, [[-90.0, -100.0], [80.0, 120.0]])   # edited points written first
        self.assertTrue(wait_for(lambda: self.cal.state == "done"))
        self.win.poll()
        self.assertEqual(self.cal.version, version + 1)
        self.assertIn("done", self.win.state.text())
        self.assertEqual(self.win.offsets(), read_offsets(self.config))
        for r in self.sim.receivers:                  # ideal sensors: within a tick of the truth
            o = self.win.offsets()[r.id]
            err = calibration_error((r.slope, r.intercept), (o["slope"], o["intercept"]), 200.0)
            self.assertLess(abs(err), 2.0)
        self.assertEqual([p[0] for p in self.win.plot.points], ["P1", "P2"])
        self.assertEqual([p[3] for p in self.win.plot.points], ["", ""])         # line of sight everywhere
        self.cal.state = "run 1"                  # a new run at P1: P2 still holds the old readings
        self.win.poll()
        self.assertEqual([p[0] for p in self.win.plot.points], ["P1"])
        self.cal.state = "done"

    def test_a_receiver_switched_off_keeps_its_old_offsets(self):
        self.sim.set_receiver_state(3, "off")
        before = read_offsets(self.config)[3]
        self.win.receiver.setCurrentIndex(3)
        self.win.calibrate_btn.click()
        self.assertTrue(wait_for(lambda: self.cal.state == "done"))
        self.win.poll()
        self.assertEqual(read_offsets(self.config)[3], before)
        self.assertIn("R3 kept the old offsets", self.win.run_detail.text())
        self.assertEqual({p[3] for p in self.win.plot.points}, {"receiver off"})

    def test_controls_wait_while_calibrating(self):
        pulse = self.cal.client.pulse
        self.cal.client.pulse = lambda: (time.sleep(0.003), pulse())[1]
        self.win.calibrate_btn.click()
        self.win.poll()
        self.assertTrue(self.cal.busy)
        self.assertFalse(self.win.calibrate_btn.isEnabled())
        self.assertFalse(self.win.save_btn.isEnabled())
        self.assertIn("collecting", self.win.state.text())
        self.assertTrue(wait_for(lambda: not self.cal.busy))
        self.win.poll()
        self.assertTrue(self.win.calibrate_btn.isEnabled())

    def test_never_reads_the_config_while_a_run_rewrites_it(self):
        with open(self.config) as fh:
            whole = fh.read()
        self.cal.state = "run 1"                          # as if a run were going ...
        with open(self.config, "w") as fh:                # ... and writing the file in place
            fh.write(whole[:len(whole) // 3])
        self.win.poll()
        self.assertFalse(self.win.reload_btn.isEnabled())
        self.win.reload_offsets()
        self.win.offsets_table.item(1, 1).setText("0.9")
        self.win.set_points([[0.0, 0.0], [50.0, 50.0]])
        theme.apply(app, "paper")
        self.win.poll()
        with open(self.config, "w") as fh:                # the run finishes and saves
            fh.write(whole)
        self.cal.version += 1
        self.cal.state = "done"
        self.win.poll()
        self.assertTrue(self.win.reload_btn.isEnabled())
        self.assertEqual(self.win.offsets(), read_offsets(self.config))

    def test_edit_save_reload_and_true_values(self):
        t = self.win.offsets_table
        t.item(2, 1).setText("0.75")
        self.assertEqual(self.win.offsets()[2]["slope"], 0.75)
        self.assertIn("not saved", self.win.offsets_note.text())
        t.item(2, 2).setText("bad")
        self.assertEqual(t.item(2, 2).text(), f"{self.win.offsets()[2]['intercept']:.3f}")
        self.win.reload_btn.click()
        self.assertEqual(self.win.offsets(), read_offsets(self.config))
        t.item(4, 2).setText("-180")
        version = self.cal.version
        self.win.save_btn.click()
        self.assertEqual(self.cal.version, version + 1)
        self.assertEqual(read_offsets(self.config)[4]["intercept"], -180.0)
        self.assertGreaterEqual(self.lock.entered, 1)
        self.assertIn("as in config.toml", self.win.offsets_note.text())
        self.win.true_btn.click()
        for r in self.sim.receivers:
            self.assertEqual(self.win.offsets()[r.id], {"slope": r.slope, "intercept": r.intercept})
            self.assertEqual(t.item(r.id, 5).text(), "+0.00")

    def test_follows_offsets_saved_elsewhere(self):
        offsets = read_offsets(self.config)
        offsets[3]["slope"] = 0.8
        self.cal.save(offsets)                                      # e.g. by another window
        self.win.poll()
        self.assertEqual(self.win.offsets()[3]["slope"], 0.8)
        self.assertEqual(self.win.offsets_table.item(3, 1).text(), "0.80000")

    def test_polls_only_while_shown(self):
        self.assertTrue(self.win.timer.isActive())
        self.win.hide()
        self.assertFalse(self.win.timer.isActive())
        self.win.show()
        self.assertTrue(self.win.timer.isActive())

    def test_error_columns_and_plot_follow_the_edits(self):
        r = self.sim.receivers[1]
        self.win.offsets_table.item(1, 2).setText(f"{r.intercept + 2.0}")       # 2 cm intercept error
        self.assertEqual(self.win.offsets_table.item(1, 5).text(), "+2.00")
        self.win.offsets_table.cellClicked.emit(1, 0)
        self.assertEqual(self.win.plot.receiver, 1)
        self.assertAlmostEqual(self.win.plot.cal[1], r.intercept + 2.0)

    def test_both_palettes(self):
        for name in ("paper", "vector"):
            theme.apply(app, name)
            self.win.poll()
            self.assertTrue(not_blank(self.win.plot))
            self.assertTrue(not_blank(self.win.preview))


class ResponsePlotTest(unittest.TestCase):
    def test_calibration_error(self):
        self.assertAlmostEqual(calibration_error((0.9, -130.0), (0.9, -130.0), 250.0), 0.0)
        self.assertAlmostEqual(calibration_error((1.0, -100.0), (1.01, -100.0), 300.0), 4.0)

    def test_draws_points_far_off_the_line_and_bad_slopes(self):
        plot = ResponsePlot()
        plot.resize(500, 320)
        plot.set_data(0, (0.96, -139.0), (0.18, 102.0),
                      [("P1", 250.0, 100.0, ""), ("P2", 1500.0, 270.0, "no line of sight")])
        self.assertTrue(not_blank(plot))
        plot.set_data(0, (0.96, -139.0), (-0.55, 313.0), [("P1", 250.0, 100.0, "")])     # a bad run
        self.assertTrue(not_blank(plot))
        plot.set_data(0, (0.0, -139.0), (0.96, -139.0), [])          # says so instead of plotting
        self.assertFalse(plot.grab().isNull())


class MapWindowTest(unittest.TestCase):
    def setUp(self):
        theme.apply(app, "vector")
        self.sim  = ideal_sim()
        self.sim.world.obstacles = [Obstacle.rect(-60, -20, -30, 20)]
        self.lock = CountingLock()
        self.win  = MapWindow(self.sim, self.lock)
        self.win.resize(900, 640)
        self.win.show()

    def tearDown(self):
        self.win.cancel()
        self.win.wait()
        self.win.close()

    def compute(self, kind: str, step: int = 30):
        self.win.set_kind(kind)
        self.win.step.setValue(step)
        self.win.compute()
        self.assertTrue(wait_for(lambda: not self.win.busy()))
        return self.win.canvas.data

    def test_coverage_map(self):
        data = self.compute("coverage")
        xs, ys = maps.grid(self.sim.believed, 30)
        self.assertEqual(data.Z.shape, (len(ys), len(xs)))
        self.assertEqual((data.vmin, data.vmax), (0.0, 6.0))
        self.assertEqual(self.lock.entered, 1)                       # the copy is taken under the lock
        self.assertTrue(self.win.show_btn.isEnabled())
        self.assertIn("receivers", data.summary())
        self.assertTrue(not_blank(self.win.canvas))

    def test_hover_readout(self):
        self.compute("hdop")
        c = self.win.canvas
        c._hover = (0.0, 0.0)
        self.assertIn("HDOP", c.hover_text())
        c._hover = (-45.0, 0.0)                                      # inside the obstacle
        self.assertIn("can't be here", c.hover_text())
        c._hover = (5000.0, 0.0)
        self.assertEqual(c.hover_text(), "")
        self.assertTrue(not_blank(c))

    def test_show_on_and_clear_the_arena(self):
        sent = []
        self.win.showOnArena.connect(sent.append)
        data = self.compute("bias", 40)
        self.win.show_btn.click()
        self.win.clear_btn.click()
        _, _, Z, title, vmin, vmax, levels = sent[0]
        self.assertIs(Z, data.Z)
        self.assertIn("cm", title)
        self.assertLess(vmin, vmax)
        self.assertEqual(levels, data.levels)                       # the arena draws the same contours
        self.assertIsNone(sent[1])

    def test_wait_says_when_the_computation_has_ended(self):
        self.win.set_kind("hdop")
        self.win.step.setValue(40)
        self.win.compute()
        self.assertTrue(self.win.wait(20000))
        self.assertTrue(wait_for(lambda: self.win.canvas.data is not None))

    def test_accuracy_map_and_controls(self):
        self.win.set_kind("accuracy")
        self.assertTrue(self.win.algorithm.isEnabled() and self.win.trials.isEnabled())
        self.assertEqual(self.win.step.value(), KINDS["accuracy"].step)
        self.win.trials.setValue(2)
        data = self.compute("accuracy", 50)
        self.assertTrue(np.isfinite(data.Z).any())
        self.assertTrue(data.title.startswith("LM"))
        self.win.set_kind("coverage")
        self.assertFalse(self.win.algorithm.isEnabled())

    def test_cancel(self):
        self.win.set_kind("accuracy")
        self.win.algorithm.setCurrentIndex(maps.ALGORITHMS.index("CEP"))
        self.win.trials.setValue(50)
        self.win.compute()
        self.assertTrue(self.win.busy() and self.win.cancel_btn.isEnabled())
        self.win.cancel_btn.click()
        self.assertTrue(wait_for(lambda: not self.win.busy()))
        self.assertIsNone(self.win.canvas.data)
        self.assertIn("cancelled", self.win.canvas.message)
        self.assertTrue(self.win.compute_btn.isEnabled())

    def test_colour_ranges(self):
        Z = np.array([[1.0, 2.0, np.inf], [np.nan, 3.0, 40.0]])
        for kind in KINDS.values():
            data = make_map(kind, np.arange(3.0), np.arange(2.0), Z, 6, kind.title, "", 0.1)
            self.assertLess(data.vmin, data.vmax, kind.key)
            self.assertTrue(all(data.vmin < v < data.vmax for v in data.levels), kind.key)
            self.assertIsInstance(data.summary(), str)
        flat = make_map(KINDS["bias"], np.arange(3.0), np.arange(2.0), np.zeros((2, 3)), 6, "", "", 0.1)
        self.assertGreaterEqual(flat.vmax - flat.vmin, 0.25)          # not a scale of 0.0001 cm steps

    def test_both_palettes(self):
        self.compute("coverage", 40)
        for name in ("paper", "vector"):
            theme.apply(app, name)
            self.assertTrue(not_blank(self.win.canvas))


if __name__ == "__main__":
    unittest.main()
