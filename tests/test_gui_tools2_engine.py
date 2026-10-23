"""Calibration engine: the true response in the current air, a calibration
run that can be cancelled, times out and reports client errors instead of
hanging, resuming the drive afterwards, and a point search that respects the
room (walls, obstacles, line of sight) and never repeats a point.  Every
calibration writes to a temp copy of config.toml."""

import os
import re
import sys
import threading
import time
import unittest
from unittest import mock

import numpy as np

from tests.helpers import ideal_sim, temp_config
from engine import cal_points
from engine.calibration import Calibrator, effective_true, read_offsets
from engine.hardware import Environment, Faults
from engine.simulation import Simulation
from engine.world import Obstacle


def scaled_config(scale: float) -> str:
    """A temp config.toml with every receiver position scaled by ``scale``."""
    path = temp_config()
    with open(path) as fh:
        text = fh.read()
    text = re.sub(r"position = \[([-\d.e]+), ([-\d.e]+)\]",
                  lambda m: f"position = [{float(m.group(1)) * scale}, {float(m.group(2)) * scale}]", text)
    with open(path, "w") as fh:
        fh.write(text)
    return path


def run_with_deadline(cal: Calibrator, seconds: float = 20.0) -> None:
    """``cal.run()`` on a thread; fail rather than hang if it never returns."""
    t = threading.Thread(target=cal.run, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise AssertionError("the calibration run never returned")


class EffectiveTrueTest(unittest.TestCase):
    def test_same_air_is_sensors_toml(self):
        sim = ideal_sim()
        for r in sim.receivers:
            self.assertEqual(effective_true(r, sim.env), (r.slope, r.intercept))

    def test_matches_the_ticks_in_warmer_air(self):
        sim = ideal_sim(temperature_c=35.0, humidity_pct=80.0)
        for d in (80.0, 150.0, 300.0):
            p = sim.receivers[0].position + np.array([0.0, d])
            for r, reading in zip(sim.receivers[:1], sim.hardware.measure(p)):
                a, b = effective_true(r, sim.env)
                self.assertAlmostEqual(a * reading.tick + b, d, delta=a)                # within a tick
                self.assertGreater(abs(r.slope * reading.tick + r.intercept - d), 2.0)  # sensors.toml's is off


class CalibratorTest(unittest.TestCase):
    def setUp(self):
        self.config = temp_config()
        self.sim    = ideal_sim(self.config)
        self.cal    = Calibrator(self.sim, self.config)
        self.before = read_offsets(self.config)

    def test_resumes_the_drive(self):
        self.sim.motion.set_mode("circle")
        for _ in range(10):
            self.sim.step()
        where, phase = self.sim.true.copy(), self.sim.motion.phase
        self.cal.run()
        self.assertEqual(self.cal.state, "done", self.cal.error)
        self.assertEqual(self.sim.motion.mode, "circle")
        self.assertEqual(self.sim.motion.phase, phase)
        np.testing.assert_array_equal(self.sim.true, where)
        self.sim.step()
        self.assertLess(np.hypot(*(self.sim.true - where)), 20.0)          # carries on from where it was

    def test_keeps_a_drive_changed_during_the_run(self):
        self.sim.motion.set_mode("circle")
        pulse = self.cal.client.pulse

        def pulse_and_steer():
            if self.sim.motion.mode == "target":      # e.g. a click on the arena while it runs
                self.sim.motion.set_mode("lawnmower")
            return pulse()
        self.cal.client.pulse = pulse_and_steer
        self.cal.run()
        self.assertEqual(self.cal.state, "done", self.cal.error)
        self.assertEqual(self.sim.motion.mode, "lawnmower")

    def test_cancel(self):
        pulse = self.cal.client.pulse
        self.cal.client.pulse = lambda: (time.sleep(0.005), pulse())[1]
        self.cal.start()
        self.assertTrue(self.cal.busy)
        time.sleep(0.05)
        t0 = time.perf_counter()
        self.cal.cancel()
        while self.cal.busy and time.perf_counter() - t0 < 5:
            time.sleep(0.005)
        self.assertLess(time.perf_counter() - t0, 3.0)
        self.assertEqual((self.cal.state, self.cal.error), ("error", "cancelled"))
        self.assertEqual(read_offsets(self.config), self.before)
        self.cal.client.pulse = pulse
        self.cal.run()                                   # a later run is not cancelled
        self.assertEqual(self.cal.state, "done", self.cal.error)

    def test_cancel_when_idle_does_nothing(self):
        self.cal.cancel()
        self.assertEqual(self.cal.state, "idle")
        self.cal.run()
        self.assertEqual(self.cal.state, "done", self.cal.error)

    def test_a_client_error_ends_the_run(self):
        def broken():
            raise OSError("serial port gone")
        self.cal.client.pulse = broken
        with mock.patch("builtins.print"):
            run_with_deadline(self.cal)
        self.assertEqual(self.cal.state, "error")
        self.assertIn("serial port gone", self.cal.error)
        self.assertFalse(self.cal.busy)
        self.assertEqual(read_offsets(self.config), self.before)

    def test_wrong_number_of_ticks_ends_the_run(self):
        self.cal.client.pulse = lambda: [100] * 5
        with mock.patch("builtins.print"):
            run_with_deadline(self.cal)
        self.assertEqual(self.cal.state, "error")
        self.assertIn("P1", self.cal.error)

    def test_times_out(self):
        ticks = iter(range(10 ** 9))
        self.cal.client.pulse = lambda: [next(ticks)] * 6        # never the same tick twice
        self.cal.timeout_s = 0.3
        run_with_deadline(self.cal)
        self.assertEqual(self.cal.state, "error")
        self.assertIn("timed out after 0.3 s at P1", self.cal.error)
        self.assertIn("R0", self.cal.error)
        self.assertEqual(self.sim.motion.mode, "target")

    def test_four_points_still_calibrate(self):
        self.cal.set_points([(-97, -142), (93, 98), (93, -147), (-97, 98)])
        offsets = self.cal.run()
        self.assertEqual(self.cal.state, "done", self.cal.error)
        for r in self.sim.receivers:
            self.assertLess(abs(offsets[r.id]["slope"] / r.slope - 1), 0.02)


class PointSearchTest(unittest.TestCase):
    def setUp(self):
        self.sim = Simulation(temp_config(), seed=4, env=Environment(), faults=Faults(noise_cm=1.0))
        self.sim.world.obstacles = [Obstacle.rect(-70, -40, -20, 30, name="box")]

    def test_default_room_gives_the_same_sets(self):
        sim  = ideal_sim()
        bare = cal_points.search(sim.believed, 4)
        room = cal_points.search(sim.believed, 4, world=sim.world, receivers=sim.receivers)
        for n in (2, 3, 4):
            np.testing.assert_allclose(room[n], bare[n])

    def test_respects_obstacles_and_line_of_sight(self):
        sim = self.sim
        for n, pts in cal_points.search(sim.believed, 4, world=sim.world, receivers=sim.receivers).items():
            self.assertEqual(len(pts), n)
            self.assertFalse(cal_points.unseen(pts, sim.world, sim.receivers).any(), n)
            self.assertFalse(any(o.contains(q) for q in pts for o in sim.world.obstacles), n)
            self.assertTrue(all(sim.world.inside(q) for q in pts), n)
            self.assertEqual(cal_points.too_close(pts), [], n)

    def test_line_of_sight_search_calibrates_well(self):
        sim = self.sim
        pts = cal_points.search(sim.believed, 3, world=sim.world, receivers=sim.receivers)[3]
        cfg = sim.config_path
        cal = Calibrator(sim, cfg)
        cal.set_points(pts)
        with mock.patch("builtins.print"):
            cal.run()
        for r in sim.receivers:
            o    = read_offsets(cfg)[r.id]
            a, b = effective_true(r, sim.env)
            for d in (100.0, 300.0):
                tick = (d - b) / a
                self.assertLess(abs(o["slope"] * tick + o["intercept"] - d), 15.0, f"R{r.id}")   # not an echo

    def test_a_receiver_nobody_can_see_does_not_stop_the_search(self):
        sim = self.sim
        sim.world.obstacles = [Obstacle.rect(-120, -250, -110, 250)]        # a screen in front of R0..R2
        pts = cal_points.search(sim.believed, 3, world=sim.world, receivers=sim.receivers)[3]
        self.assertEqual(len(pts), 3)
        self.assertEqual(cal_points.too_close(pts), [])
        hidden = cal_points.unseen(pts, sim.world, sim.receivers)
        self.assertFalse(hidden[:, 3:].any())                        # the others still see every point

    def test_receivers_off_or_blocked_never_count(self):
        sim = self.sim
        q   = np.array([[-100.0, 0.0], [0.0, 0.0]])                          # R1 can't see behind the box
        self.assertTrue(cal_points.unseen(q, sim.world, sim.receivers)[:, 1].any())
        for state in ("off", "blocked"):
            sim.set_receiver_state(1, state)
            self.assertFalse(cal_points.unseen(q, sim.world, sim.receivers)[:, 1].any(), state)

    def test_points_are_never_repeated(self):
        sim = ideal_sim()
        for scale in (1.0, 1.5):
            for n, pts in cal_points.search(sim.believed * scale, 4).items():
                self.assertEqual(len({tuple(q) for q in pts}), n, scale)
                self.assertEqual(cal_points.too_close(pts), [], scale)

    def test_too_few_candidates_is_a_clear_error(self):
        sim = ideal_sim()
        for coords in (sim.believed * 2.4, np.array([[x, 0.0] for x in (-150, -90, -30, 30, 90, 150)])):
            with self.assertRaisesRegex(ValueError, "within 400 cm of every receiver"):
                cal_points.search(coords, 2)
        with self.assertRaisesRegex(ValueError, "only 3 points qualify for 4"):
            cal_points.search(sim.believed * 1.7, 4)

    def test_too_close(self):
        self.assertEqual(cal_points.too_close([[0, 0], [20, 0], [0, 19]]), [(0, 2, 19.0)])

    def test_command_line_says_why_it_cannot_search(self):
        argv = ["cal_points", "--config", scaled_config(2.4), "--trials", "1"]
        with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as caught:
            cal_points.main()
        self.assertIn("no point qualifies", str(caught.exception.code))
        self.assertTrue(os.path.exists(argv[2]))


if __name__ == "__main__":
    unittest.main()
