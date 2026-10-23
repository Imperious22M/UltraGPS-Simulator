"""Arena maps: coverage, first arrival, HDOP, range bias and Monte-Carlo
accuracy, on small grids so the whole module runs in a few seconds."""

import math
import unittest

import numpy as np

from tests.helpers import ideal_sim
from engine import maps
from engine import solvers as S
from engine.hardware import Environment
from engine.simulation import Simulation
from engine.world import Obstacle


def cell(xs, ys, Z, x, y) -> float:
    """The value of the grid cell nearest (x, y)."""
    return float(Z[int(np.argmin(np.abs(ys - y))), int(np.argmin(np.abs(xs - x)))])


def inside_box(sim, xs, ys):
    """Mask of the cells inside the receivers' box."""
    (x0, y0), (x1, y1) = sim.believed.min(axis=0), sim.believed.max(axis=0)
    X, Y = np.meshgrid(xs, ys)
    return (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)


class GridTest(unittest.TestCase):
    def test_grid_covers_the_receivers_box_plus_the_margin(self):
        sim = Simulation(seed=0)
        xs, ys = maps.grid(sim.believed, 20.0)
        (x0, y0), (x1, y1) = sim.believed.min(axis=0), sim.believed.max(axis=0)
        self.assertTrue(np.allclose(np.diff(xs), 20.0) and np.allclose(np.diff(ys), 20.0))
        self.assertLessEqual(xs[0], x0 - maps.MARGIN_CM + 20.0)
        self.assertGreaterEqual(xs[-1], x1 + maps.MARGIN_CM - 20.0)
        self.assertAlmostEqual((xs[0] + xs[-1]) / 2, (x0 + x1) / 2)     # centred on the box
        self.assertAlmostEqual((ys[0] + ys[-1]) / 2, (y0 + y1) / 2)

    def test_a_scene_is_a_copy(self):
        sim = Simulation(seed=0)
        sc  = maps.scene(sim)
        sim.world.obstacles = [Obstacle.rect(-10, -10, 10, 10)]
        sim.receivers[0].state = "off"
        self.assertEqual(sc.world.obstacles, [])
        self.assertEqual(sc.receivers[0].state, "on")
        self.assertIs(maps.scene(sc), sc)


class CoverageTest(unittest.TestCase):
    def test_coverage_drops_behind_an_obstacle(self):
        sim = Simulation(seed=0)
        xs, ys, free = maps.coverage(sim, 20.0)
        sim.world.obstacles = [Obstacle.rect(-100, -30, -80, 30)]          # in front of R1 at (-127, 0)
        _, _, shadow = maps.coverage(sim, 20.0)
        self.assertLess(cell(xs, ys, shadow, 50, 0), cell(xs, ys, free, 50, 0))
        both = np.isfinite(shadow) & np.isfinite(free)
        self.assertTrue(np.all(shadow[both] <= free[both]))               # an obstacle never adds coverage
        self.assertTrue(math.isnan(cell(xs, ys, shadow, -90, 0)))          # the transmitter can't be inside it

    def test_coverage_counts_only_receivers_that_are_on(self):
        sim = Simulation(seed=0)
        _, _, before = maps.coverage(sim, 40.0)
        sim.set_receiver_state(0, "blocked")
        sim.set_receiver_state(3, "off")
        _, _, after = maps.coverage(sim, 40.0)
        self.assertLessEqual(np.nanmax(after), len(sim.receivers) - 2)
        self.assertTrue(np.all(after <= before))

    def test_maps_are_deterministic_and_leave_the_simulation_alone(self):
        sim   = Simulation(seed=0)
        state = sim.hardware.rng.bit_generator.state
        a = maps.coverage(sim, 40.0)[2]
        b = maps.coverage(sim, 40.0)[2]
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(sim.hardware.rng.bit_generator.state, state)


    def test_coverage_and_first_arrival_agree_in_an_ideal_room(self):
        # Fired at once and listening from 3 ms: a receiver closer than 3 ms of
        # sound (about 1 m) misses the direct sound, in both models.
        sim = ideal_sim(max_range_cm=1000.0, burst_lead_ms=0.0, rx_ready_ms=3.0)
        sim.set_receiver_state(2, "off")
        _, _, covered = maps.coverage(sim, 40.0)
        _, _, missed  = maps.first_arrival(sim, 40.0)
        inside = np.isfinite(covered)
        self.assertTrue(np.array_equal(covered[inside] + missed[inside], np.full(inside.sum(), 5.0)))
        self.assertLess(np.nanmin(covered), 5.0)


class FirstArrivalTest(unittest.TestCase):
    def test_no_wrong_arrivals_in_an_ideal_empty_room(self):
        _, _, Z = maps.first_arrival(ideal_sim(max_range_cm=1000.0), 40.0)
        self.assertEqual(np.nanmax(Z), 0.0)

    def test_out_of_range_counts_as_not_direct(self):
        sim = ideal_sim()                                                  # 400 cm range
        xs, ys, Z = maps.first_arrival(sim, 40.0)
        far = np.count_nonzero(np.linalg.norm(sim.coords - [xs[-1], ys[-1]], axis=1) > sim.env.max_range_cm)
        self.assertGreater(far, 0)
        self.assertEqual(cell(xs, ys, Z, xs[-1], ys[-1]), float(far))

    def test_an_obstacle_makes_receivers_hear_something_else(self):
        sim = Simulation(seed=0)
        sim.world.obstacles = [Obstacle.rect(-100, -30, -80, 30)]
        xs, ys, Z = maps.first_arrival(sim, 40.0)
        self.assertGreaterEqual(cell(xs, ys, Z, 50, 0), 1.0)


class HdopTest(unittest.TestCase):
    def test_hdop_is_worse_outside_the_receivers(self):
        sim = ideal_sim()
        xs, ys, Z = maps.hdop(sim, 20.0)
        box = inside_box(sim, xs, ys)
        self.assertTrue(np.all(np.isfinite(Z)))
        self.assertGreater(np.median(Z[~box]), np.median(Z[box]))
        self.assertLess(cell(xs, ys, Z, 0, 0), 1.0)                      # six receivers round the centre

    def test_hdop_is_infinite_with_fewer_than_three_receivers(self):
        sim = ideal_sim()
        for i in range(4):
            sim.set_receiver_state(i, "off")
        _, _, Z = maps.hdop(sim, 40.0)
        self.assertTrue(np.all(np.isinf(Z)))


class BiasTest(unittest.TestCase):
    def test_no_bias_without_physics(self):
        _, _, Z = maps.bias(ideal_sim(), 40.0)
        self.assertLess(np.nanmax(Z), 1e-6)

    def test_range_walk_shows_where_the_signal_is_weak(self):
        sim = Simulation(seed=0, env=Environment(reflections=False, diffraction=False))
        _, _, Z = maps.bias(sim, 40.0)
        self.assertGreater(np.nanmax(Z[np.isfinite(Z)]), 0.5)


class AccuracyTest(unittest.TestCase):
    def test_accuracy_is_finite_and_small_in_the_middle(self):
        sim = Simulation(seed=0)
        xs, ys, Z = maps.accuracy(sim, "LM", trials=3, step=40.0)
        middle = cell(xs, ys, Z, 0, 0)
        self.assertTrue(math.isfinite(middle))
        self.assertLess(middle, 10.0)

    def test_progress_and_cancel(self):
        sim, seen = ideal_sim(noise=1.0), []
        maps.accuracy(sim, "OLS", trials=1, step=50.0, progress=seen.append)
        self.assertTrue(all(b >= a for a, b in zip(seen, seen[1:])))
        self.assertAlmostEqual(seen[-1], 1.0)
        calls = []
        with self.assertRaises(maps.Cancelled):
            maps.accuracy(sim, "OLS", trials=1, step=50.0, cancel=lambda: calls.append(1) or len(calls) > 3)

    def test_cancel_stops_within_a_point(self):
        # A CEP point at 200 pings takes seconds: cancel must not wait for it
        sim, seen, calls = ideal_sim(noise=1.0), [], []
        with self.assertRaises(maps.Cancelled):
            maps.accuracy(sim, "LM", trials=50, step=50.0, progress=seen.append,
                          cancel=lambda: calls.append(1) or len(calls) > 3)
        self.assertEqual(seen, [])                                        # not even the first point finished
        self.assertEqual(len(calls), 4)

    def test_no_fix_without_enough_receivers(self):
        sim = ideal_sim()
        for i in range(4):
            sim.set_receiver_state(i, "off")
        _, _, Z = maps.accuracy(sim, "LM", trials=2, step=60.0)
        self.assertTrue(np.all(np.isinf(Z)))

    def test_unknown_algorithm_or_no_pings(self):
        with self.assertRaises(ValueError):
            maps.accuracy(ideal_sim(), "EKF", trials=1, step=60.0)
        with self.assertRaises(ValueError):
            maps.accuracy(ideal_sim(), "LM", trials=0, step=60.0)

    def test_every_snapshot_algorithm_solves_perfect_ranges(self):
        sim = ideal_sim()
        lib = sim.solvers.lib
        p   = np.array([20.0, -35.0])
        d   = np.linalg.norm(lib._receiver_coords - p, axis=1)
        for alg in maps.ALGORITHMS:
            est = maps.solve(lib, d, list(range(6)), alg)
            self.assertTrue(est.ok, alg)
            self.assertLess(np.hypot(*(est.pos - p)), 0.5, alg)
        self.assertFalse(maps.solve(lib, d, [0, 1], "LM").ok)

    def test_library_cep_searches_as_the_settings_say(self):
        sim, asked = ideal_sim(), []
        lib    = sim.solvers.lib
        search = lib._find_best_subset
        lib._find_best_subset = lambda d, idx, n: asked.append(n) or search(d, idx, n)
        d   = np.linalg.norm(lib._receiver_coords - [10.0, 20.0], axis=1)
        est = maps.solve(lib, d, list(range(6)), "CEP")
        maps.solve(lib, d, list(range(6)), "CEP", S.Settings(cep_subsets="all"))
        self.assertEqual(asked, [S.CONTROL_MAX_SUBSETS, 2 ** 6])
        self.assertIsNone(est.cov)                                        # the library's matrix is a score
        self.assertIsNotNone(est.score_cov)


if __name__ == "__main__":
    unittest.main()
