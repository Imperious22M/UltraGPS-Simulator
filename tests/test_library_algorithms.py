"""The real positioning and calibration libraries, driven by the simulator
(ported from the first simulator's tests)."""

import hashlib
import os
import unittest
from itertools import combinations

import numpy as np

from tests.helpers import ideal_sim, run, temp_config
from engine.calibration import Calibrator, read_offsets, set_cal_points
from engine.config import DEFAULT_CONFIG, DEFAULT_SENSORS
from engine.hardware import Faults
from engine.simulation import Simulation
from engine import solvers as S
from engine.solvers import ALGORITHMS
from ultragps_position import load_config

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The library copies as they were before this branch; nothing here may change them.
LIBRARY_SHA256 = {
    "ultragps_position/__init__.py":       "43c86ef7d7c6ab4d374dec7a49d6a5de58befd6888a2be96d55e4f8058a94ef5",
    "ultragps_position/core.py":           "1b2e264bc4ff738cc6396ec1ce7beae100376aa360344e6e6854deeca77e82e4",
    "ultragps_calibration/__init__.py":    "28efebf00443ad50fdd535aae527c727293659bff0359a546642924d0f5c1b8a",
    "ultragps_calibration/calibration.py": "24a3679384de8ea4953664914b16bd0556890167683c800390f10d8f1d463b48",
}


class LibraryTest(unittest.TestCase):
    def test_libraries_are_unchanged(self):
        for rel, sha in LIBRARY_SHA256.items():
            with open(os.path.join(REPO, rel), "rb") as fh:
                self.assertEqual(hashlib.sha256(fh.read()).hexdigest(), sha, rel)

    def test_ticks_round_trip_through_library_calibration(self):
        sim = ideal_sim()
        frame = run(sim, [20, -40], frames=1)
        true = [r.true_cm for r in frame.readings]
        # Integer ticks quantise distance by at most half a slope
        np.testing.assert_allclose(frame.solution.distances, true, atol=0.6)

    def test_clean_signal_every_algorithm_is_accurate(self):
        sim = ideal_sim()
        run(sim, [30, 50])
        for a in ALGORITHMS:
            self.assertLess(sim.stats(a)[0], 2.0, a)

    def test_receiver_off_reports_zero_ticks(self):
        sim = ideal_sim()
        sim.set_receiver_state(2, "off")
        frame = run(sim, [0, 0], frames=1)
        self.assertEqual(frame.ticks[2], 0)
        self.assertEqual(frame.status[2], "off")

    def test_cep_drops_a_dead_receiver(self):
        sim = ideal_sim(noise=1.0)
        sim.set_receiver_state(0, "off")
        frame = run(sim, [30, 50])
        self.assertNotIn(0, frame.estimates["CEP"].used)
        self.assertLess(sim.stats("CEP")[0], 5.0)

    def _cep_subsets_tried(self, setting: str) -> list:
        sim = ideal_sim(noise=1.0)
        sim.solvers.settings.cep_subsets = setting
        calls = []
        real = sim.solvers.lib._compute_cep
        sim.solvers.lib._compute_cep = lambda d, idx: calls.append(tuple(idx)) or real(d, idx)
        run(sim, [30, 50], frames=1)
        return calls

    def test_cep_tries_the_subsets_ultragps_control_tries(self):
        self.assertEqual(S.CONTROL_MAX_SUBSETS, 15)       # get_position_full's default
        calls = self._cep_subsets_tried("control")
        # All six, then the first 14 three-receiver subsets: every one holds R0 or R1
        self.assertEqual(calls, [tuple(range(6))] + list(combinations(range(6), 3))[:14])
        self.assertNotIn((2, 4, 5), calls)

    def test_cep_can_try_every_subset_of_three_or_more(self):
        calls = self._cep_subsets_tried("all")
        # 6 receivers: C(6,3) + C(6,4) + C(6,5) + C(6,6) = 20 + 15 + 6 + 1
        self.assertEqual(len(set(calls)), 42)
        self.assertEqual(S.cep_max_subsets(S.Settings(cep_subsets="all"), 6), 64)
        self.assertEqual(S.cep_max_subsets(S.Settings(), 6), 15)

    def test_control_cep_cannot_drop_r0_and_r1_together(self):
        # Both read long (echoes): every subset UltraGPS-Control tries holds one of them
        truth = np.array([30.0, 50.0])
        sim = ideal_sim()
        lib = sim.solvers.lib
        d = np.hypot(*(lib._receiver_coords - truth).T)
        d[0] += 40.0
        d[1] += 25.0
        ticks = [round((v - lib._offsets[i]["intercept"]) / lib._offsets[i]["slope"]) for i, v in enumerate(d)]
        err = {}
        for setting in ("control", "all"):
            sim.solvers.settings.cep_subsets = setting
            sim.solvers.reset()
            cep = sim.solvers.solve(ticks).estimates["CEP"]
            err[setting] = float(np.hypot(*(cep.pos - truth)))
            if setting == "all":
                self.assertFalse({0, 1} & set(cep.used))
        self.assertLess(err["all"], 1.0)
        self.assertGreater(err["control"], 10.0)

    def test_cep_reports_no_subset_when_holding_last_fix(self):
        sim = ideal_sim(noise=1.0)
        good = run(sim, [30, 50], frames=5)
        for i in (1, 2, 3, 4):
            sim.set_receiver_state(i, "off")
        held = sim.step()
        self.assertFalse(held.estimates["CEP"].ok)
        self.assertEqual(held.estimates["CEP"].used, [])
        np.testing.assert_array_equal(held.estimates["CEP"].pos, good.estimates["CEP"].pos)

    def test_cep4_uses_four_or_more_and_keeps_fixes_outside_the_box(self):
        sim = ideal_sim(noise=1.0)
        frame = run(sim, [140, 0])          # x beyond the receivers at ±127
        self.assertGreaterEqual(len(frame.estimates["CEP4"].used), 4)
        self.assertLess(sim.stats("CEP4")[0], 5.0)
        self.assertFalse(frame.estimates["CEP"].ok)   # library CEP holds instead

    def test_simulator_algorithms_hold_their_last_fix(self):
        sim = ideal_sim(noise=1.0)
        good = run(sim, [30, 50], frames=5)
        for i in (1, 2, 3):
            sim.set_receiver_state(i, "off")
        held = sim.step()
        for a in ("CEP4", "BIAS"):
            self.assertFalse(held.estimates[a].ok, a)
            np.testing.assert_array_equal(held.estimates[a].pos, good.estimates[a].pos)

    def test_dead_receiver_is_filtered_out(self):
        sim = ideal_sim(noise=1.0)
        run(sim, [30, 50], frames=5)
        sim.set_receiver_state(5, "off")
        for _ in range(3):
            frame = sim.step()
        self.assertNotIn(5, frame.solution.sane)
        self.assertLess(sim.stats("LM")[0], 5.0)

    def test_lost_signal_reports_high_value_rejected_by_top_band(self):
        sim = ideal_sim()
        sim.faults = Faults(noise_cm=0.0, timeout_rate=1.0)
        frame = run(sim, [30, 50], frames=1)
        self.assertTrue(all(st == "timeout" for st in frame.status))
        self.assertTrue(all(d > 1000 for d in frame.solution.distances))
        self.assertEqual(frame.solution.sane, [])

    def test_top_band_reject_is_adjustable(self):
        sim = ideal_sim()
        sim.solvers.lib.max_distance = 200.0
        frame = run(sim, [30, 50], frames=1)
        far = [i for i, d in enumerate(frame.solution.distances) if d > 200]
        self.assertTrue(far)
        self.assertFalse(set(far) & set(frame.solution.sane))

    def test_bad_value_stays_within_spike_range(self):
        sim = ideal_sim()
        sim.faults = Faults(noise_cm=0.0, spike_rate=1.0, spike_pct=10.0)
        for _ in range(20):
            frame = run(sim, [30, 50], frames=1)
            true = np.array([r.true_cm for r in frame.readings])
            ratio = np.array(frame.solution.distances) / true
            self.assertTrue(np.all(np.abs(ratio - 1) <= 0.10 + 0.01))

    def test_calibration_routine_recovers_sensor_response(self):
        cfg = temp_config()
        sim = ideal_sim(cfg, noise=0.5)
        cal = Calibrator(sim, cfg)
        offsets = cal.run()
        self.assertEqual(cal.state, "done")
        self.assertEqual(set(offsets), set(range(6)))
        saved = read_offsets(cfg)
        for i in range(6):
            self.assertEqual(saved[i], offsets[i])
            self.assertEqual(sim.solvers.lib._offsets[i], offsets[i])   # applied
            rx = sim.receivers[i]
            for d in cal.cal.get_cal_distances(i):
                tick = (d - rx.intercept) / rx.slope
                fitted = tick * offsets[i]["slope"] + offsets[i]["intercept"]
                self.assertLess(abs(fitted - d), 3.0)

    def test_calibration_runs_with_realistic_physics(self):
        cfg = temp_config()
        sim = Simulation(cfg, seed=0)
        cal = Calibrator(sim, cfg)
        cal.run()
        self.assertEqual(cal.state, "done", cal.error)

    def test_edited_offsets_are_saved_and_used(self):
        cfg = temp_config()
        sim = ideal_sim(cfg)
        cal = Calibrator(sim, cfg)
        offsets = read_offsets(cfg)
        offsets[0]["intercept"] += 40.0          # mis-calibrate R0 by 40 cm
        cal.save(offsets)
        self.assertEqual(read_offsets(cfg)[0]["intercept"], offsets[0]["intercept"])
        frame = run(sim, [30, 50], frames=1)
        self.assertAlmostEqual(frame.solution.distances[0] - frame.readings[0].true_cm, 40.0, delta=1.0)

    def test_sensor_truth_comes_from_sensors_toml(self):
        sim = Simulation(DEFAULT_CONFIG, DEFAULT_SENSORS)
        truth = sorted(load_config(DEFAULT_SENSORS)["receivers"], key=lambda r: r["id"])
        np.testing.assert_allclose([r.slope for r in sim.receivers], [r["slope"] for r in truth])
        np.testing.assert_allclose([r.facing_deg for r in sim.receivers], [r["facing_deg"] for r in truth])

    def test_calibration_point_search_beats_current_points(self):
        from engine import cal_points
        sim = ideal_sim()
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
        sim = ideal_sim(cfg, noise=0.5)
        offsets = Calibrator(sim, cfg).run()
        for i in range(6):
            self.assertLess(abs(offsets[i]["slope"] / sim.receivers[i].slope - 1), 0.02)

    def test_two_point_fit_matches_original_formula(self):
        cfg = temp_config()
        sim = ideal_sim(cfg)
        cal = Calibrator(sim, cfg).cal
        for run_no, ticks in ((1, [300] * 6), (2, [200] * 6)):
            for i in range(6):
                cal.add_reading(i, ticks[i], run_no)
        off = cal.calculate_offsets()
        for i in range(6):
            d1, d2 = cal.get_cal_distances(i)
            slope = (d1 - d2) / (300 - 200)
            self.assertAlmostEqual(off[i]["slope"], slope)
            self.assertAlmostEqual(off[i]["intercept"], d1 - slope * 300)

    def test_field_of_view_cutoff_drops_receivers_outside_15_degrees(self):
        sim = ideal_sim(fov_cutoff=True)
        # R1 at (-127, 0) faces +x: (0, 0) is dead ahead, (0, 100) is ~38° off
        self.assertEqual(run(sim, [0, 0], frames=1).status[1], "ok")
        off = run(sim, [0, 100], frames=1)
        self.assertEqual(off.status[1], "angle")
        self.assertEqual(off.ticks[1], 0)
        sim.env = sim.env.copy(fov_cutoff=False)
        self.assertEqual(run(sim, [0, 100], frames=1).status[1], "ok")

    def test_near_receiver_noise(self):
        sim = ideal_sim(near_noise=True)
        errs = []
        for _ in range(200):
            f = run(sim, [-117, 0], frames=1)          # 10 cm from R1 (< 16 cm)
            self.assertEqual(f.status[1], "near")
            errs.append(f.solution.distances[1] - 10.0)
        self.assertGreater(np.std(errs), 3.0)
        self.assertEqual(run(sim, [0, 0], frames=1).status[1], "ok")   # far: no extra noise


if __name__ == "__main__":
    unittest.main()
