"""Simulator-only algorithms, the lab traces, recording and replay."""

import os
import tempfile
import unittest

import numpy as np

from tests.helpers import ideal_sim, run
from engine import lab
from engine import solvers as S
from engine.config import DEFAULT_CONFIG
from engine.hardware import Faults
from engine.motion import Motion
from engine.world import Obstacle
from ultragps_position import UltraGPSPositionLib


def lib_and_ranges(pos=(30.0, 50.0)):
    lib = UltraGPSPositionLib(DEFAULT_CONFIG)
    return lib, np.hypot(*(lib._receiver_coords - np.array(pos)).T)


class SolverTest(unittest.TestCase):
    def test_fast_lm_lands_where_the_library_does(self):
        lib, d = lib_and_ranges()
        rng = np.random.default_rng(0)
        for _ in range(50):
            dd = d + rng.normal(0, 3, 6)
            for idx in ([0, 1, 2, 3, 4, 5], [0, 2, 4], [1, 2, 3, 5]):
                fast = S.fit_lm(lib, dd, idx)[0]
                ref = lib._nlls(dd, lib._ols(dd, np.array(idx)), np.array(idx))[0]
                self.assertLess(np.hypot(*(fast - ref)), 0.05)

    def test_robust_and_ransac_resist_one_bad_reading(self):
        lib, d = lib_and_ranges()
        d[2] += 60.0
        idx = list(range(6))
        lm = lib._multilaterate(tuple(d), np.array(idx))[0]
        robust = S.solve_robust(lib, d, idx, 5.0)
        ransac = S.solve_ransac(lib, d, idx, 8.0)
        truth = np.array([30.0, 50.0])
        self.assertGreater(np.hypot(*(lm - truth)), 10.0)
        self.assertLess(np.hypot(*(robust.pos - truth)), 3.0)
        self.assertLess(np.hypot(*(ransac.pos - truth)), 0.5)
        self.assertNotIn(2, ransac.used)

    def test_bias_solver_recovers_a_trigger_offset(self):
        lib, d = lib_and_ranges()
        est = S.solve_bias(lib, d + 12.0, list(range(6)))
        np.testing.assert_allclose(est.pos, [30.0, 50.0], atol=1e-3)
        self.assertIn("+12.0", est.note)

    def test_bias_solver_refuses_an_implausible_offset(self):
        lib, d = lib_and_ranges()
        d[[0, 3]] += 250.0                       # two echoes: "explained" by a huge offset
        est = S.solve_bias(lib, d, list(range(6)))
        if est.ok:
            self.assertLess(abs(float(est.note.split()[1])), S.MAX_OFFSET_CM)
        d2 = d.copy()
        d2[:] = 1e6                              # nonsense in every range
        self.assertFalse(S.solve_bias(lib, d2, list(range(6))).ok)

    def test_bias_solver_beats_lm_under_trigger_jitter(self):
        sim = ideal_sim(noise=0.5, sync_jitter_us=150.0)
        run(sim, [30, 50], frames=200)
        self.assertLess(sim.stats("BIAS")[1], sim.stats("LM")[1] / 2)

    def test_kalman_tracks_and_coasts_through_gaps(self):
        sim = ideal_sim(noise=1.0)
        sim.motion.set_mode("circle")
        for _ in range(100):
            sim.step()
        self.assertLess(sim.stats("EKF")[1], 3.0)
        for i in (0, 1, 2, 3):
            sim.set_receiver_state(i, "off")
        for _ in range(3):
            f = sim.step()
        self.assertLess(f.errors["EKF"], 10.0)          # still tracking with 2 receivers
        self.assertFalse(f.estimates["LM"].ok)

    def test_kalman_gates_a_bad_reading(self):
        sim = ideal_sim(noise=0.5)
        run(sim, [30, 50], frames=20)
        sim.faults = Faults(noise_cm=0.5, spike_rate=1.0, spike_pct=40)
        for i in range(1, 6):
            sim.receivers[i].state = "on"
        f = sim.step()
        self.assertTrue(f.estimates["EKF"].note.startswith("gated"))

    def test_bounds_filter_mode_lets_simulator_algorithms_see_more(self):
        sim = ideal_sim(noise=0.0)
        sim.solvers.settings.filter_mode = "bounds"
        run(sim, [30, 50], frames=5)
        sim.faults = Faults(noise_cm=0.0, spike_rate=0.0)
        f = sim.step()
        self.assertEqual(sorted(f.estimates["RANSAC"].used), f.solution.inbounds)


class LabTest(unittest.TestCase):
    def test_cep_trace_replays_the_library_search(self):
        lib, d = lib_and_ranges()
        d[2] += 40
        tr = lab.trace_cep(lib, d, list(range(6)))
        self.assertEqual(len(tr.steps), 42)
        self.assertEqual(tr.steps[0].subset, list(range(6)))           # library tries all first
        best_pos, best_cep, best_idx, _ = lib._find_best_subset(tuple(d), list(range(6)), 64)
        self.assertEqual(tr.steps[tr.best].subset, best_idx)
        self.assertAlmostEqual(tr.steps[tr.best].cep, best_cep)
        np.testing.assert_allclose(tr.position, best_pos)
        s = tr.steps[5]
        np.testing.assert_allclose(s.weights, 1 / (np.abs(s.residuals) + 0.1))
        self.assertFalse(hasattr(lib, "__dict__") and "_compute_cep" in lib.__dict__)   # unwrapped again

    def test_cep_trace_never_calls_an_unusable_subset_best(self):
        lib, d = lib_and_ranges()
        trace = lab.trace_cep(lib, d, list(range(6)))
        for k, step in enumerate(trace.steps):
            if step.best:
                self.assertTrue(np.isfinite(step.cep))
                self.assertTrue(all(step.cep < s.cep for s in trace.steps[:k]))
        self.assertTrue(trace.steps[trace.best].best)

    def test_lm_trace_ends_where_the_library_does(self):
        lib, d = lib_and_ranges()
        start = np.array([-100.0, 150.0])
        tr = lab.trace_lm(lib, d, list(range(6)), start)
        ref = lib._nlls(d, start, np.arange(6))[0]
        np.testing.assert_allclose(tr.final, ref)
        np.testing.assert_allclose(tr.path[0], start)
        np.testing.assert_allclose(tr.path[-1], ref)
        costs = [c for _, c, k in tr.points if k in ("start", "step")]
        self.assertEqual(costs, sorted(costs, reverse=True))           # every step goes downhill

    def test_filter_trace_matches_the_library(self):
        lib, d = lib_and_ranges()
        ticks = [round((dist - lib._offsets[i]["intercept"]) / lib._offsets[i]["slope"]) for i, dist in enumerate(d)]
        ticks[3] = 0
        tr = lab.trace_filter(lib, ticks)
        self.assertEqual(tr.sane, [0, 1, 2, 4, 5])
        self.assertFalse(tr.rows[3].in_bounds)
        ticks2 = list(ticks)
        ticks2[0] += 100
        tr2 = lab.trace_filter(lib, ticks2)
        self.assertFalse(tr2.rows[0].kept)
        self.assertGreater(tr2.rows[0].jump, tr2.max_differential)

    def test_sweep_shows_cep_dropping_the_bad_receiver(self):
        lib, d = lib_and_ranges()
        sw = lab.sweep(lib, d, list(range(6)), 2, 120.0, 7)
        self.assertNotIn(2, sw.cep_subsets[-1])
        self.assertLess(np.hypot(*(sw.paths["RANSAC"][-1] - [30, 50])), 0.5)
        self.assertGreater(np.hypot(*(sw.paths["LM"][-1] - [30, 50])), 10.0)

    def test_contour_of_a_cone_is_a_circle(self):
        xs = ys = np.linspace(-10, 10, 81)
        X, Y = np.meshgrid(xs, ys)
        segs = lab.contour(xs, ys, np.hypot(X, Y), 5.0)
        r = np.hypot(segs[..., 0], segs[..., 1])
        self.assertTrue(len(segs) > 20)
        np.testing.assert_allclose(r, 5.0, atol=0.05)


class RecordingTest(unittest.TestCase):
    def test_record_and_replay(self):
        sim = ideal_sim(noise=1.0)
        sim.motion.set_mode("circle")
        sim.start_recording()
        for _ in range(10):
            sim.step()
        frames = sim.stop_recording()
        tmp = tempfile.mkdtemp()
        csv_path, log_path = os.path.join(tmp, "run.csv"), os.path.join(tmp, "ticks.txt")
        sim.save_csv(frames, csv_path)
        sim.save_ticks(frames, log_path)
        with open(log_path) as fh:
            self.assertTrue(fh.readline().startswith("C: "))
        for path, truth in ((csv_path, True), (log_path, False)):
            sim.load_replay(path)
            replayed = [sim.step() for _ in range(10)]
            self.assertEqual([f.ticks for f in replayed], [f.ticks for f in frames])
            self.assertEqual(replayed[0].true is not None, truth)
        sim.stop_replay()


class MotionTest(unittest.TestCase):
    def test_circle_mode_catches_its_path_and_stays_on_it(self):
        m = Motion((-100.0, -100.0, 100.0, 100.0))
        m.set_mode("circle")
        pos = np.array([0.0, 0.0])
        for _ in range(60):
            pos = m.step(pos, 0.1)
        for _ in range(40):
            pos = m.step(pos, 0.1)
            self.assertAlmostEqual(float(np.hypot(*pos)), m.radius, delta=1e-6)

    def test_transmitter_does_not_drive_through_obstacles(self):
        sim = ideal_sim()
        box = Obstacle.rect(-40, -40, 40, 40)
        sim.world.obstacles = [box]
        sim.place((-100, 0))
        sim.motion.go_to((100, 0))                 # straight through the box
        for _ in range(60):
            sim.step()
            self.assertFalse(box.contains(sim.true))
        self.assertLess(sim.true[0], -39)            # stopped at its edge
        sim.motion.set_mode("random")
        for _ in range(200):
            sim.step()
            self.assertFalse(box.contains(sim.true))


if __name__ == "__main__":
    unittest.main()
