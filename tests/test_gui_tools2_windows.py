"""Calibration and map windows: a search that respects the room and reports
failures instead of aborting, points that can't be calibrated on refused,
line-of-sight warnings, true values in the current air, cancelling a run, an
error message that wraps, merged plot labels; and the maps' contour levels on
the arena, the CEP no-fix note, the dimmed previous map and the child process
maps are computed in.  Every calibration writes to a temp copy of
config.toml."""

import multiprocessing
import os
import tempfile
import time
import unittest
from functools import partial
from unittest import mock

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("XDG_CONFIG_HOME", tempfile.mkdtemp())
try:
    from PyQt6.QtWidgets import QApplication, QScrollArea
except ImportError as exc:
    raise unittest.SkipTest(f"PyQt6 is not installed: {exc}")

import numpy as np

from tests.helpers import ideal_sim, temp_config
from tests.test_gui_tools2_engine import scaled_config
from engine import cal_points, maps
from engine.calibration import Calibrator, effective_true, read_offsets
from engine.simulation import Simulation
from engine.world import Obstacle
from gui import theme
from gui.calibration_window import CalibrationWindow, ResponsePlot, _merged_labels, calibration_error
from gui.map_window import KINDS, MapJob, MapWindow, make_map

app = QApplication.instance() or QApplication([])


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
    def make(self, config=None, sim=None):
        theme.apply(app, "vector")
        self.config = config or temp_config()
        self.sim    = sim or ideal_sim(self.config)
        self.cal    = Calibrator(self.sim, self.config)
        self.lock   = CountingLock()
        self.win    = CalibrationWindow(self.sim, self.cal, self.lock)
        self.win.resize(1100, 760)
        self.win.show()
        self.addCleanup(self.close)
        return self.win

    def close(self):
        self.cal.cancel()
        wait_for(lambda: not self.cal.busy and self.win.search_job is None)
        self.win.close()

    def type_points(self, points):
        win = self.win
        win.count.setValue(len(points))
        for row, (x, y) in enumerate(points):
            win.points_table.item(row, 1).setText(repr(float(x)))
            win.points_table.item(row, 2).setText(repr(float(y)))

    # ─── Search ──────────────────────────────────────────────────────────────

    def test_search_avoids_obstacles_and_hidden_receivers(self):
        win = self.make()
        self.sim.world.obstacles = [Obstacle.rect(-70, -40, -20, 30, name="box")]
        win.count.setValue(3)
        win.search_btn.click()
        self.assertEqual(self.lock.entered, 1)                       # the room is copied under the lock
        self.assertTrue(wait_for(lambda: win.search_job is None))
        pts = np.array(win.points())
        self.assertEqual(len(pts), 3)
        self.assertFalse(cal_points.unseen(pts, self.sim.world, self.sim.receivers).any())
        self.assertFalse(any(o.contains(q) for q in pts for o in self.sim.world.obstacles))
        self.assertTrue(win.calibrate_btn.isEnabled() and win.use_btn.isEnabled())

    def test_search_failure_is_shown_not_fatal(self):
        config = scaled_config(2.4)
        win    = self.make(config, Simulation(config, seed=0))
        win.search_btn.click()
        self.assertTrue(wait_for(lambda: win.search_job is None))
        self.assertIn("no point qualifies", win.score.text())
        self.assertIn("no point qualifies", win.statusBar().currentMessage())
        self.assertTrue(win.search_btn.isEnabled())
        self.assertEqual(win.search_btn.text(), "Search best points")

    # ─── Points that can't be used ───────────────────────────────────────────

    def test_refuses_points_that_cannot_be_calibrated_on(self):
        win = self.make()
        self.sim.world.obstacles = [Obstacle.rect(-110, -40, -90, 40)]
        in_use = self.cal.points
        cases  = {"0.4 cm apart": ([(20.0, 30.0), (20.4, 30.2)], 1),         # (points, a row marked bad)
                  "inside an obstacle": ([(-100.0, 0.0), (78.0, 153.0)], 0),
                  "outside the walls": ([(-82.0, -152.0), (400.0, 0.0)], 1),
                  "0.0 cm apart": ([(10.0, 10.0), (10.0, 10.0)], 0)}
        for why, (pts, row) in cases.items():
            self.type_points(pts)
            self.assertIn(why, win.score.text(), why)
            self.assertFalse(win.calibrate_btn.isEnabled(), why)
            self.assertFalse(win.use_btn.isEnabled(), why)
            self.assertEqual(win.points_table.item(row, 1).foreground().color().name().lower(),
                             theme.current.status["lost"].lower(), why)
            self.assertFalse(win.use_points(), why)                    # refused however it is called
            win.calibrate()
            self.assertEqual(self.cal.state, "idle", why)
            self.assertEqual(self.cal.points, in_use, why)
            self.assertIn("Can't", win.statusBar().currentMessage(), why)

    def test_points_in_config_that_cannot_be_used_are_not_calibrated(self):
        self.config = temp_config()
        Calibrator(ideal_sim(self.config), self.config).set_points([(10.0, 10.0), (10.0, 10.0)])
        win = self.make(self.config)
        self.assertFalse(win.calibrate_btn.isEnabled())
        win.calibrate()
        self.assertEqual(self.cal.state, "idle")

    def test_marks_receivers_that_cannot_see_a_point(self):
        win = self.make()
        self.sim.world.obstacles = [Obstacle.rect(-110, -40, -90, 40)]
        win.set_points([(-82.0, -152.0), (78.0, 153.0), (93.0, -147.0)])
        text = win.score.text()
        self.assertIn("No line of sight", text)
        self.assertIn("P2–R1", text)
        self.assertIn("R1 can't see it", win.points_table.item(1, 1).toolTip())
        for name in ("paper", "vector"):                               # amber like the warning lines,
            theme.apply(app, name)                                     # never the red of a refused row
            win.poll()
            amber = theme.current.status["near"]
            self.assertEqual(win.points_table.item(1, 1).foreground().color().name().lower(), amber.lower())
            self.assertIn(amber, win.score.text())
            self.assertNotEqual(amber.lower(), theme.current.status["lost"].lower())
        self.assertTrue(win.calibrate_btn.isEnabled())                 # a warning: it may be meant
        self.assertTrue(not_blank(win.preview))

    def test_follows_obstacles_drawn_elsewhere(self):
        win = self.make()
        win.set_points([(-82.0, -152.0), (78.0, 153.0)])
        self.assertNotIn("line of sight", win.score.text())
        self.sim.world.obstacles = [Obstacle.rect(-110, -40, -90, 40)]     # e.g. drawn on the arena
        win.poll()
        self.assertIn("No line of sight", win.score.text())

    def test_typed_points_far_off_are_refused(self):
        win = self.make()
        before = win.points()[1][0]
        win.points_table.item(1, 1).setText("1e300")
        self.assertEqual(win.points()[1][0], before)
        self.assertTrue(not_blank(win.preview))

    def test_added_points_are_not_one_point_twice(self):
        win = self.make()
        win.count.setValue(4)
        self.assertEqual(cal_points.too_close(win.points()), [])

    def test_revert_tells_the_arena(self):
        win  = self.make()
        sent = []
        win.pointsChanged.connect(sent.append)
        win.set_points([(-90.0, -100.0), (80.0, 120.0)])
        win.revert_btn.click()
        self.assertEqual(win.points(), self.cal.points)
        self.assertEqual(sent, [self.cal.points])

    # ─── Run ─────────────────────────────────────────────────────────────────

    def test_cancel_a_run(self):
        win = self.make()
        before = read_offsets(self.config)
        pulse  = self.cal.client.pulse
        self.cal.client.pulse = lambda: (time.sleep(0.005), pulse())[1]
        self.assertFalse(win.cancel_btn.isEnabled())
        win.calibrate_btn.click()
        win.poll()
        self.assertTrue(win.cancel_btn.isEnabled())
        win.cancel_btn.click()
        self.assertTrue(wait_for(lambda: not self.cal.busy, 5))
        win.poll()
        self.assertIn("cancelled", win.state.text())
        self.assertEqual(read_offsets(self.config), before)
        self.assertTrue(win.calibrate_btn.isEnabled())
        self.assertFalse(win.cancel_btn.isEnabled())

    def test_a_long_error_wraps_instead_of_widening_the_panel(self):
        win = self.make()
        app.processEvents()
        scroll = win.findChild(QScrollArea)
        width  = scroll.widget().width()
        self.cal.state = "error"
        self.cal.error = ("[Errno 13] Permission denied: '/home/someone/projects/UltraGPS-Simulator/"
                          "config/a/rather/long/path/to/the/config.toml' " * 2)
        win.poll()
        for _ in range(5):
            app.processEvents()
        self.assertLessEqual(scroll.widget().width(), max(width, scroll.viewport().width()))
        self.assertIn("Permission denied", win.state.toolTip())
        self.cal.state = "idle"

    # ─── True values in the current air ──────────────────────────────────────

    def test_true_values_follow_the_air(self):
        win = self.make()
        t   = win.offsets_table
        win.receiver.setCurrentIndex(2)
        self.assertEqual(t.horizontalHeaderItem(3).text(), "true\nslope")
        self.sim.env = self.sim.env.copy(temperature_c=35.0)
        win.poll()                                                   # the Environment panel changed it
        self.assertIn("35 °C", t.horizontalHeaderItem(3).text())
        self.assertIn("35 °C", win.true_btn.toolTip())
        r    = self.sim.receivers[2]
        a, b = effective_true(r, self.sim.env)
        self.assertEqual(t.item(2, 3).text(), f"{a:.5f}")
        self.assertGreater(a, r.slope)
        # The calibration in config.toml was right at 20 °C: at 35 °C it reads short
        o   = read_offsets(self.config)[r.id]
        err = calibration_error((a, b), (o["slope"], o["intercept"]), 100.0)
        self.assertEqual(t.item(2, 5).text(), f"{err:+.2f}")
        self.assertLess(float(t.item(2, 5).text()), -2.0)
        win.true_btn.click()
        self.assertEqual(win.offsets()[r.id], {"slope": a, "intercept": b})
        self.assertEqual(t.item(2, 5).text(), "+0.00")
        self.assertEqual(win.plot.true, (a, b))
        self.assertIn("35", win.plot.air)

    def test_calibrating_in_warm_air_is_reported_good(self):
        win = self.make()
        self.sim.env = self.sim.env.copy(temperature_c=35.0)
        win.set_points([(-97.0, -141.94), (93.0, 98.06), (93.0, -146.94), (-97.0, 98.06)])
        win.calibrate_btn.click()
        self.assertTrue(wait_for(lambda: self.cal.state == "done"))
        win.poll()
        self.assertIn("every receiver within", win.run_detail.text())
        for row in range(len(self.sim.receivers)):
            self.assertLess(abs(float(win.offsets_table.item(row, 6).text())), 1.0)


class ResponsePlotLabelTest(unittest.TestCase):
    def test_merged_labels(self):
        self.assertEqual(_merged_labels([("P2", "no line of sight"), ("P3", "no line of sight")]),
                         "P2, P3 no line of sight")
        self.assertEqual(_merged_labels([("P1", ""), ("P2", "receiver off")]), "P1 · P2 receiver off")
        self.assertEqual(_merged_labels([("P1", "")]), "P1")

    def test_points_on_top_of_each_other_get_one_label(self):
        plot = ResponsePlot()
        plot.resize(500, 320)
        plot.set_data(1, (0.93, -136.5), (0.09, 131.2),
                      [("P1", 319.0, 158.5, ""), ("P2", 1509.0, 255.8, "no line of sight"),
                       ("P3", 1510.0, 264.6, "no line of sight")], air="35 °C")
        self.assertTrue(not_blank(plot))


class MapWindowTest(unittest.TestCase):
    def setUp(self):
        theme.apply(app, "vector")
        self.sim  = ideal_sim()
        self.sim.world.obstacles = [Obstacle.rect(-60, -20, -30, 20)]
        self.win  = MapWindow(self.sim)
        self.win.resize(900, 640)
        self.win.show()

    def tearDown(self):
        self.win.cancel()
        self.win.wait()
        self.win.close()

    def compute(self, kind: str, step: int = 40, **kw):
        self.win.set_kind(kind)
        if "algorithm" in kw:
            self.win.algorithm.setCurrentIndex(maps.ALGORITHMS.index(kw["algorithm"]))
        self.win.trials.setValue(kw.get("trials", 2))
        self.win.step.setValue(step)
        self.win.compute()
        self.assertTrue(wait_for(lambda: not self.win.busy(), 120))
        return self.win.canvas.data

    def test_coverage_levels_reach_the_arena(self):
        sent = []
        self.win.showOnArena.connect(sent.append)
        data = self.compute("coverage")
        self.win.show_btn.click()
        *_, levels = sent[0]
        self.assertEqual(len(sent[0]), 7)
        self.assertIn(2.5, levels)                                    # "below 3 receivers"
        self.assertEqual(levels, data.levels)

    def test_cep_no_fix_outside_the_box_is_explained(self):
        self.assertIn("_validate_position", KINDS["accuracy"].explain)
        data = self.compute("accuracy", 50, algorithm="CEP")
        text = data.summary()
        self.assertIn("inside the receivers' box", text)
        self.assertIn("(0 % inside", text)                            # all its no-fix cells are the margin
        self.assertIn("first 15 subsets", data.detail)                # the Solver settings' CEP search

    def test_summary_without_a_scene(self):
        Z = np.array([[1.0, np.inf], [2.0, 3.0]])
        data = make_map(KINDS["accuracy"], np.arange(2.0), np.arange(2.0), Z, 6, "", "", 0.1)
        self.assertTrue(data.summary().endswith("no fix on 25 %"))

    def test_the_previous_map_is_dimmed_while_computing(self):
        old = self.compute("coverage")
        self.win.set_kind("hdop")
        self.win.step.setValue(30)
        self.win.compute()
        self.assertIs(self.win.canvas.data, old)
        self.assertIn("Computing", self.win.canvas.message)
        self.assertTrue(not_blank(self.win.canvas))
        self.assertTrue(wait_for(lambda: not self.win.busy(), 120))
        self.assertIsNot(self.win.canvas.data, old)
        self.assertEqual(self.win.canvas.message, "")

    def test_a_cancelled_computation_keeps_the_previous_map(self):
        old = self.compute("coverage")
        self.win.set_kind("accuracy")
        self.win.algorithm.setCurrentIndex(maps.ALGORITHMS.index("CEP"))
        self.win.trials.setValue(50)
        self.win.compute()
        self.win.cancel_btn.click()
        self.assertTrue(wait_for(lambda: not self.win.busy(), 60))
        self.assertIs(self.win.canvas.data, old)
        self.assertIn("cancelled", self.win.canvas.message)
        for name in ("paper", "vector"):
            theme.apply(app, name)
            self.assertTrue(not_blank(self.win.canvas))


class MapJobTest(unittest.TestCase):
    def run_job(self, fn, process: bool = True):
        job  = MapJob(fn, process=process)
        seen = {"progress": [], "done": [], "failed": []}
        job.progressed.connect(seen["progress"].append)
        job.computed.connect(seen["done"].append)
        job.failed.connect(seen["failed"].append)
        job.start()
        self.assertTrue(wait_for(job.isFinished, 60))
        return seen

    def test_a_child_process_gives_the_same_map_as_this_thread(self):
        fn    = partial(maps.coverage, maps.scene(ideal_sim()), 40.0)
        child = self.run_job(fn)
        here  = self.run_job(fn, process=False)
        self.assertEqual((child["failed"], len(child["done"])), ([], 1))
        np.testing.assert_array_equal(child["done"][0][2], here["done"][0][2])
        self.assertEqual(child["progress"][-1], 1.0)
        self.assertEqual(child["progress"], sorted(child["progress"]))

    def test_a_failure_in_the_child_is_reported(self):
        seen = self.run_job(partial(maps.coverage, None, 40.0))
        self.assertEqual(seen["done"], [])
        self.assertTrue(seen["failed"][0].startswith("failed:"), seen["failed"])

    def test_without_child_processes_the_map_is_computed_here(self):
        spawn = type(multiprocessing.get_context("spawn"))
        fn    = partial(maps.coverage, maps.scene(ideal_sim()), 40.0)
        with mock.patch.object(MapJob, "no_processes", False), \
             mock.patch.object(spawn, "Process", side_effect=OSError("no processes here")) as start:
            first  = self.run_job(fn)
            second = self.run_job(fn)
            self.assertTrue(MapJob.no_processes)
        for seen in (first, second):
            self.assertEqual((seen["failed"], len(seen["done"])), ([], 1))
        self.assertEqual(start.call_count, 1)                     # the next map does not try again

    def test_cancelling_stops_the_child(self):
        job  = MapJob(partial(maps.accuracy, maps.scene(ideal_sim()), "CEP", 200, 10.0))
        seen = []
        job.failed.connect(seen.append)
        job.start()
        self.assertTrue(wait_for(lambda: job._conn is not None, 30))
        job.cancel()
        self.assertTrue(wait_for(job.isFinished, 20))
        self.assertEqual(seen, ["cancelled"])


class RunJobTest(unittest.TestCase):
    """The child's side, run here on one end of a pipe."""

    def test_the_child_stops_when_the_other_end_closes(self):
        here, there = multiprocessing.Pipe()
        here.close()                                              # as when the simulator is killed
        maps.run_job(partial(maps.coverage, maps.scene(ideal_sim()), 40.0), there)
        self.assertTrue(there.closed)

    def test_progress_then_one_result(self):
        here, there = multiprocessing.Pipe()
        maps.run_job(partial(maps.coverage, maps.scene(ideal_sim()), 40.0), there)
        sent = []
        try:
            while True:
                sent.append(here.recv())
        except EOFError:                                          # run_job closed its end
            pass
        kinds = [k for k, _ in sent]
        self.assertEqual(kinds[-1], "done")
        self.assertEqual(set(kinds[:-1]), {"progress"})
        self.assertLessEqual(len(kinds), 102)                     # every 1 % at most

if __name__ == "__main__":
    unittest.main()
