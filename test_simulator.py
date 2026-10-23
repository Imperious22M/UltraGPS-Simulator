"""Headless checks for the simulator:  python3 -m unittest test_simulator"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibration import Calibrator, read_offsets, set_cal_points  # noqa: E402
from sensor_model import Faults  # noqa: E402
from simulation import Simulation  # noqa: E402
from solvers import DEFAULT_CONFIG, DEFAULT_SENSORS  # noqa: E402
from ultragps_position import load_config  # noqa: E402


def temp_config() -> str:
    """Copy of config.toml in a temp dir, so calibration tests don't edit the repo's."""
    path = os.path.join(tempfile.mkdtemp(), "config.toml")
    shutil.copy(DEFAULT_CONFIG, path)
    return path


def run(sim: Simulation, pos, frames: int = 30) -> dict:
    sim.true = sim.target = np.array(pos, dtype=float)
    for _ in range(frames):
        sim.step()
    return sim.last


class SimulatorTest(unittest.TestCase):
    def test_ticks_round_trip_through_library_calibration(self):
        sim = Simulation(seed=0)
        sim.faults = Faults(noise_cm=0.0)
        frame = run(sim, [20, -40], frames=1)
        true = sim.sensor.true_distances(frame["true"])
        # Integer ticks quantise distance by at most half a slope
        np.testing.assert_allclose(frame["distances"], true, atol=0.6)

    def test_clean_signal_every_algorithm_is_accurate(self):
        sim = Simulation(seed=0)
        sim.faults = Faults(noise_cm=0.0)
        run(sim, [30, 50])
        for a in ("OLS", "LM", "CEP", "CEP4"):
            self.assertLess(sim.stats(a)[0], 2.0, a)

    def test_receiver_off_reports_zero_ticks(self):
        sim = Simulation(seed=0)
        sim.enabled[2] = False
        frame = run(sim, [0, 0], frames=1)
        self.assertEqual(frame["ticks"][2], 0)
        self.assertEqual(frame["status"][2], "off")

    def test_cep_drops_a_dead_receiver(self):
        sim = Simulation(seed=0)
        sim.enabled[0] = False
        frame = run(sim, [30, 50])
        self.assertNotIn(0, frame["cep_subset"])
        self.assertLess(sim.stats("CEP")[0], 5.0)

    def test_cep_tries_every_subset_of_three_or_more(self):
        sim = Simulation(seed=0)
        calls = []
        real = sim.solvers.lib._compute_cep
        sim.solvers.lib._compute_cep = lambda d, idx: calls.append(tuple(idx)) or real(d, idx)
        run(sim, [30, 50], frames=1)
        # 6 receivers: C(6,3) + C(6,4) + C(6,5) + C(6,6) = 20 + 15 + 6 + 1
        self.assertEqual(len(set(calls)), 42)

    def test_cep_reports_no_subset_when_holding_last_fix(self):
        sim = Simulation(seed=0)
        good = run(sim, [30, 50], frames=5)
        for i in (1, 2, 3, 4):
            sim.enabled[i] = False
        held = sim.step()
        self.assertFalse(held["success"]["CEP"])
        self.assertEqual(held["cep_subset"], [])
        np.testing.assert_array_equal(held["positions"]["CEP"], good["positions"]["CEP"])

    def test_cep4_uses_four_or_more_and_keeps_fixes_outside_the_box(self):
        sim = Simulation(seed=0)
        frame = run(sim, [140, 0])          # x beyond the receivers at ±127
        self.assertGreaterEqual(len(frame["cep4_subset"]), 4)
        self.assertLess(sim.stats("CEP4")[0], 5.0)
        self.assertFalse(frame["success"]["CEP"])   # library CEP holds instead

    def test_cep4_holds_with_fewer_than_four_receivers(self):
        sim = Simulation(seed=0)
        good = run(sim, [30, 50], frames=5)
        for i in (1, 2, 3):
            sim.enabled[i] = False
        held = sim.step()
        self.assertEqual(held["cep4_subset"], [])
        np.testing.assert_array_equal(held["positions"]["CEP4"], good["positions"]["CEP4"])

    def test_dead_receiver_is_filtered_out(self):
        sim = Simulation(seed=0)
        run(sim, [30, 50], frames=5)
        sim.enabled[5] = False
        for _ in range(3):
            frame = sim.step()
        self.assertNotIn(5, frame["sane"])
        self.assertLess(sim.stats("LM")[0], 5.0)

    def test_lost_signal_reports_high_value_rejected_by_top_band(self):
        sim = Simulation(seed=0)
        sim.faults = Faults(noise_cm=0.0, timeout_rate=1.0)
        frame = run(sim, [30, 50], frames=1)
        self.assertTrue(all(st == "timeout" for st in frame["status"]))
        self.assertTrue(all(d > 1000 for d in frame["distances"]))
        self.assertEqual(frame["sane"], [])

    def test_top_band_reject_is_adjustable(self):
        sim = Simulation(seed=0)
        sim.faults = Faults(noise_cm=0.0)
        sim.solvers.lib.max_distance = 200.0
        frame = run(sim, [30, 50], frames=1)
        far = [i for i, d in enumerate(frame["distances"]) if d > 200]
        self.assertTrue(far)
        self.assertFalse(set(far) & set(frame["sane"]))

    def test_bad_value_stays_within_spike_range(self):
        sim = Simulation(seed=0)
        sim.faults = Faults(noise_cm=0.0, spike_rate=1.0, spike_pct=10.0)
        for _ in range(20):
            frame = run(sim, [30, 50], frames=1)
            true = sim.sensor.true_distances(frame["true"])
            ratio = np.array(frame["distances"]) / true
            self.assertTrue(np.all(np.abs(ratio - 1) <= 0.10 + 0.01))

    def test_calibration_routine_recovers_sensor_response(self):
        cfg = temp_config()
        sim = Simulation(cfg, seed=0)
        sim.faults = Faults(noise_cm=0.5)
        cal = Calibrator(sim, cfg)
        offsets = cal.run()
        self.assertEqual(cal.state, "done")
        self.assertEqual(set(offsets), set(range(6)))
        saved = read_offsets(cfg)
        for i in range(6):
            self.assertEqual(saved[i], offsets[i])
            self.assertEqual(sim.solvers.lib._offsets[i], offsets[i])   # applied
            # Fitted line agrees with the true response at the calibration points
            for d in cal.cal.get_cal_distances(i):
                tick = (d - sim.sensor.intercepts[i]) / sim.sensor.slopes[i]
                fitted = tick * offsets[i]["slope"] + offsets[i]["intercept"]
                self.assertLess(abs(fitted - d), 3.0)

    def test_edited_offsets_are_saved_and_used(self):
        cfg = temp_config()
        sim = Simulation(cfg, seed=0)
        sim.faults = Faults(noise_cm=0.0)
        cal = Calibrator(sim, cfg)
        offsets = read_offsets(cfg)
        offsets[0]["intercept"] += 40.0          # mis-calibrate R0 by 40 cm
        cal.save(offsets)
        self.assertEqual(read_offsets(cfg)[0]["intercept"], offsets[0]["intercept"])
        frame = run(sim, [30, 50], frames=1)
        true = sim.sensor.true_distances(frame["true"])
        self.assertAlmostEqual(frame["distances"][0] - true[0], 40.0, delta=1.0)

    def test_sensor_truth_comes_from_sensors_toml(self):
        sim = Simulation(DEFAULT_CONFIG, DEFAULT_SENSORS)
        truth = sorted(load_config(DEFAULT_SENSORS)["receivers"], key=lambda r: r["id"])
        np.testing.assert_allclose(sim.sensor.slopes, [r["slope"] for r in truth])

    def test_calibration_point_search_beats_current_points(self):
        import cal_points
        sim = Simulation(seed=0)
        sets = cal_points.search(sim.coords, max_points=4)
        cfg = load_config(DEFAULT_CONFIG)
        current = cal_points.arena_score([cfg["cal_point_1"], cfg["cal_point_2"]], sim.coords)
        scores = [cal_points.arena_score(sets[n], sim.coords) for n in (2, 3, 4)]
        self.assertLess(scores[0], current)
        self.assertEqual(scores, sorted(scores, reverse=True))   # more points, lower error
        for n in (2, 3, 4):
            self.assertEqual(len(sets[n]), n)

    def test_four_point_calibration(self):
        cfg = temp_config()
        points = [(-97, -142), (93, 98), (93, -147), (-97, 98)]
        set_cal_points(cfg, points)
        saved = load_config(cfg)
        self.assertEqual(saved["cal_points"], [list(map(float, p)) for p in points])
        self.assertEqual(len(saved["receivers"][0]["cal_distances"]), 4)
        sim = Simulation(cfg, seed=0)
        sim.faults = Faults(noise_cm=0.5)
        cal = Calibrator(sim, cfg)
        offsets = cal.run()
        self.assertEqual(cal.state, "done")
        for i in range(6):
            self.assertLess(abs(offsets[i]["slope"] / sim.sensor.slopes[i] - 1), 0.02)

    def test_two_point_fit_matches_original_formula(self):
        cfg = temp_config()
        sim = Simulation(cfg, seed=0)
        cal = Calibrator(sim, cfg).cal
        for run, ticks in ((1, [300] * 6), (2, [200] * 6)):
            for i in range(6):
                cal.add_reading(i, ticks[i], run)
        off = cal.calculate_offsets()
        for i in range(6):
            d1, d2 = cal.get_cal_distances(i)
            slope = (d1 - d2) / (300 - 200)
            self.assertAlmostEqual(off[i]["slope"], slope)
            self.assertAlmostEqual(off[i]["intercept"], d1 - slope * 300)


if __name__ == "__main__":
    unittest.main()
