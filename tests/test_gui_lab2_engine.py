"""engine.lab against the library it watches: the sweep applies the bounds
check at every offset and can be cancelled, the snapshot solve follows the
solver settings as the simulator does, and the CEP trace knows where the
library's subset search stops."""

import threading
import unittest

import numpy as np

from engine import lab
from engine import solvers as S
from tests.helpers import temp_config
from ultragps_position import UltraGPSPositionLib

CONFIG = temp_config()


def ranges(lib, pos) -> np.ndarray:
    return np.hypot(*(lib._receiver_coords - np.asarray(pos, dtype=float)).T)


def ticks_for(lib, d) -> list:
    """The (fractional) ticks that calibrate to distances ``d``."""
    return [(v - lib._offsets[i]["intercept"]) / lib._offsets[i]["slope"] for i, v in enumerate(d)]


def steady(d, max_subsets: int) -> dict:
    """A fresh library that has seen these readings once already, then solves
    them: no jumps, as the sweep assumes."""
    lib   = UltraGPSPositionLib(CONFIG)
    ticks = ticks_for(lib, d)
    lib.get_position_full(ticks, max_subsets=max_subsets)
    return lib.get_position_full(ticks, max_subsets=max_subsets)


class CountdownEvent(threading.Event):
    """An event that reports itself set after ``n`` checks."""

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def is_set(self) -> bool:
        self.n -= 1
        return self.n < 0


class SweepTest(unittest.TestCase):
    def test_a_reading_past_the_top_band_is_dropped_as_the_library_drops_it(self):
        lib = UltraGPSPositionLib(CONFIG)
        d0  = ranges(lib, (100.0, -100.0))          # R2 reads about 366 cm; the top band is about 452
        sw  = lab.sweep(lib, d0, list(range(6)), 2, 150.0, 7)
        top = lab.top_band(lib)
        self.assertEqual(sw.top_band, top)
        self.assertEqual(sw.in_band, [bool(d0[2] + o <= top) for o in sw.offsets])
        self.assertIsNotNone(sw.dropped)
        self.assertGreater(d0[2] + sw.offsets[sw.dropped], top)
        self.assertLessEqual(d0[2] + sw.offsets[sw.dropped - 1], top)
        for k, off in enumerate(sw.offsets):
            d = d0.copy()
            d[2] += off
            r = steady(d, S.CONTROL_MAX_SUBSETS)
            np.testing.assert_allclose(sw.paths["LM"][k], r["lm_position"], atol=1e-6)
            np.testing.assert_allclose(sw.paths["OLS"][k], lib._ols(d, np.array(r["sane_indices"])), atol=1e-6)
            np.testing.assert_allclose(sw.paths["CEP"][k], r["cep_position"], atol=1e-6)
            self.assertEqual([int(i) for i in sw.cep_subsets[k]], [int(i) for i in r["best_indices"]])
            self.assertEqual(2 in r["sane_indices"], sw.in_band[k])
            self.assertEqual(sw.kept[k], [int(i) for i in r["sane_indices"]])
        self.assertLess(float(np.hypot(*(sw.paths["LM"][-1] - (100.0, -100.0)))), 0.01)   # fits the other five

    def test_with_three_kept_the_drop_leaves_the_library_no_fix(self):
        lib = UltraGPSPositionLib(CONFIG)
        d0  = ranges(lib, (100.0, -150.0))          # R2 reads about 406 cm
        sw  = lab.sweep(lib, d0, [0, 2, 5], 2, 150.0, 7)
        k   = sw.dropped
        self.assertEqual(sw.kept[:k + 1], [[0, 2, 5]] * k + [[0, 5]])
        self.assertEqual(sw.cep_subsets[k - 1], [0, 2, 5])
        self.assertEqual(sw.cep_subsets[k:], [[]] * (len(sw.offsets) - k))
        self.assertTrue(np.all(np.isnan(sw.paths["LM"][k:])))
        self.assertFalse(any(sw.cep_rejected[k:]))

    def test_a_sweep_follows_the_cep_subsets_setting(self):
        lib = UltraGPSPositionLib(CONFIG)
        d0  = ranges(lib, (30.0, 50.0)) + np.random.default_rng(4).normal(0.0, 1.0, 6)
        for mode, m in (("control", 15), ("all", 64)):
            sw = lab.sweep(lib, d0, list(range(6)), 2, 40.0, 3, S.Settings(cep_subsets=mode))
            for k, off in enumerate(sw.offsets):
                d = d0.copy()
                d[2] += off
                _, _, idx, _ = lib._find_best_subset(tuple(d), list(range(6)), m)
                self.assertEqual(sw.cep_subsets[k], idx, f"{mode} +{off}")

    def test_a_cancelled_sweep_stops_between_offsets(self):
        lib, done = UltraGPSPositionLib(CONFIG), threading.Event()
        done.set()
        self.assertIsNone(lab.sweep(lib, ranges(lib, (30.0, 50.0)), list(range(6)), 2, 150.0, 61, cancel=done))
        later = CountdownEvent(2)
        self.assertIsNone(lab.sweep(lib, ranges(lib, (30.0, 50.0)), list(range(6)), 2, 150.0, 61, cancel=later))
        self.assertEqual(later.n, -1)                # two offsets solved, stopped at the third


class StaticSolveTest(unittest.TestCase):
    def test_it_matches_the_simulators_solvers_on_a_first_frame(self):
        for mode in ("control", "all"):
            sol   = S.Solvers(CONFIG, settings=S.Settings(cep_subsets=mode))
            d     = ranges(sol.lib, (60.0, -80.0)) + np.random.default_rng(7).normal(0.0, 1.5, 6)
            frame = sol.solve(ticks_for(sol.lib, d))
            lib   = UltraGPSPositionLib(CONFIG)
            fixes = lab.static_solve(lib, frame.distances, frame.sane, sol.settings)
            for a in ("OLS", "LM", "CEP", "CEP4", "ROBUST", "BIAS", "RANSAC"):
                np.testing.assert_allclose(fixes[a].pos, frame.estimates[a].pos, atol=1e-6, err_msg=f"{mode} {a}")
            cep, ref = fixes["CEP"], frame.estimates["CEP"]
            self.assertEqual([int(i) for i in cep.used], [int(i) for i in ref.used])
            np.testing.assert_allclose(cep.cov, ref.cov)                # a position covariance, cm²
            np.testing.assert_allclose(cep.score_cov, ref.score_cov)    # the library's ranking matrix

    def test_bounds_only_gives_the_simulator_algorithms_every_reading_in_bounds(self):
        lib  = UltraGPSPositionLib(CONFIG)
        d    = ranges(lib, (30.0, 50.0))
        d[2] += 60.0                                 # in bounds, but the jump check dropped it
        kept = [0, 1, 3, 4, 5]
        for mode, given in (("library", kept), ("bounds", list(range(6)))):
            fixes = lab.static_solve(lib, d, kept, S.Settings(filter_mode=mode))
            self.assertEqual(fixes["OLS"].used, kept)                    # the library's own keep the kept ones
            self.assertEqual(fixes["LM"].used, kept)
            np.testing.assert_allclose(fixes["ROBUST"].pos, S.solve_robust(lib, d, given, 5.0).pos)
            np.testing.assert_allclose(fixes["BIAS"].pos, S.solve_bias(lib, d, given).pos)
            np.testing.assert_allclose(fixes["CEP4"].pos, S.solve_cep4(lib, d, given).pos)
            self.assertEqual(sorted(fixes["RANSAC"].used), sorted(S.solve_ransac(lib, d, given, 8.0).used))
            self.assertTrue(fixes["RANSAC"].note.endswith(f"of {len(given)} agree"), fixes["RANSAC"].note)


class CepTraceTest(unittest.TestCase):
    def test_the_search_limit_picks_the_librarys_winner(self):
        lib = UltraGPSPositionLib(CONFIG)
        rng = np.random.default_rng(11)
        for _ in range(8):
            d = ranges(lib, rng.uniform((-120.0, -180.0), (120.0, 180.0))) + rng.normal(0.0, 1.0, 6)
            for sane in (list(range(6)), [0, 1, 2, 4, 5]):
                _, _, every, _ = lib._find_best_subset(tuple(d), sane, 64)
                for m in (15, 5):
                    tr = lab.trace_cep(lib, d, sane, m)
                    pos, cep, idx, _ = lib._find_best_subset(tuple(d), sane, m)
                    self.assertEqual(tr.steps[tr.best].subset, idx)
                    self.assertAlmostEqual(tr.steps[tr.best].cep, cep)
                    np.testing.assert_allclose(tr.position, lib._validate_position(pos)[0])
                    self.assertEqual(tr.steps[tr.best_all].subset, every)

    def test_it_scores_every_subset_and_marks_how_far_the_library_goes(self):
        lib = UltraGPSPositionLib(CONFIG)
        d   = ranges(lib, (30.0, 50.0))
        tr  = lab.trace_cep(lib, d, list(range(6)), 15)
        self.assertEqual((tr.searched, len(tr.steps)), (15, 42))
        self.assertEqual([len(s.subset) for s in tr.steps[:15]], [6] + [3] * 14)   # no 4- or 5-receiver subset
        self.assertEqual(lab.trace_cep(lib, d, list(range(6))).searched, 42)         # no limit: all of them
        self.assertEqual(lab.trace_cep(lib, d, [0, 1, 2, 4, 5], 15).searched, 15)    # 15 of 16
        self.assertEqual(lab.searched_count(4, 15), 5)                              # fewer than the limit
        self.assertEqual(lab.trace_cep(lib, d, [0, 1], 15).searched, 0)


if __name__ == "__main__":
    unittest.main()
