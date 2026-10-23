"""Room geometry edge cases: seams between touching obstacles, lines through
corners, degenerate obstacles, the Fresnel-zone loss of clear paths, and the
vectorised image-source search against a plain reference."""

import math
import os
import tempfile
import unittest
import warnings

import numpy as np

from engine.config import load_world
from engine.hardware import Environment, Faults
from engine.simulation import Simulation
from engine.world import EPS, Obstacle, World, _cross

ROOM = [(-177, -237), (177, -237), (177, 237), (-177, 237)]


def best(paths):
    """The shortest path (what a receiver would detect first)."""
    return min(paths, key=lambda p: p.length)


class SeamTest(unittest.TestCase):
    def test_two_touching_boxes_block_like_one(self):
        one = World(ROOM, 0.6, [Obstacle.rect(90, -40, 100, 40)])
        two = World(ROOM, 0.6, [Obstacle.rect(90, -40, 100, 0), Obstacle.rect(90, 0, 100, 40)])
        for w in (one, two):
            self.assertTrue(w.blocked([40, 5], [127, 0]))
            self.assertTrue(w.blocked([90, 0], [127, 0]))                   # along the seam, then out
        self.assertTrue(two.blocked([95, 0], [98, 0]))                      # within the seam itself
        lengths = [sorted(round(p.length, 6) for p in w.paths([40, 5], [127, 0], 0)) for w in (one, two)]
        self.assertEqual(lengths[0], lengths[1])                            # round the ends, not through
        self.assertGreater(min(lengths[1]), 110.0)

    def test_a_box_flush_with_a_wall_leaves_no_gap(self):
        w = World(ROOM, 0.6, [Obstacle.rect(-177, -10, 40, 10)])
        self.assertTrue(w.blocked([-177, -10], [-177, 10]))                 # along the wall face
        paths = w.paths([-150, -50], [-150, 50], 0)
        self.assertTrue(paths)
        for p in paths:                                                     # only round the free end
            self.assertEqual(p.kind, "diffract")
            self.assertTrue(np.allclose(p.points[1:-1, 0], 40.0))

    def test_boxes_touching_at_one_corner_block_like_one(self):
        two = World(ROOM, 0.6, [Obstacle.rect(80, 0, 100, 20), Obstacle.rect(100, -20, 120, 0)])
        self.assertTrue(two.blocked([90, -10], [110, 10]))                  # exactly through the touch point
        self.assertTrue(two.blocked([60, 0], [140, 0]))                     # along both faces, through it
        paths = two.paths([40, -5], [127, 0], 1)
        self.assertTrue(paths)
        for p in paths:                                                     # only the long ways round
            self.assertFalse(any(np.allclose(v, [100, 0]) for v in p.points))
            self.assertGreater(p.length, 100.0)
        side = World(ROOM, 0.6, [Obstacle.rect(80, 0, 100, 20), Obstacle.rect(100, 0, 120, 20)])
        self.assertFalse(side.blocked([60, 0], [140, 0]))                   # grazing two boxes in a row

    def test_the_receiver_behind_them_hears_what_it_would_behind_one_box(self):
        def r4(obstacles):
            sim = Simulation(seed=0, env=Environment(amp_jitter_db=0.0, tick_dither=False),
                             faults=Faults(noise_cm=0.0))
            sim.world.obstacles[:] = obstacles
            return sim.hardware.measure([40.0, -5.0])[4]                    # R4 at (127, 0)
        two = r4([Obstacle.rect(80, 0, 100, 20), Obstacle.rect(100, -20, 120, 0)])
        one = r4([Obstacle.rect(80, -20, 120, 20)])
        self.assertEqual((two.status, two.echo_ms), (one.status, one.echo_ms))
        self.assertEqual(two.status, "lost")                                # was "bent", via (100, 0)

    def test_a_staircase_of_boxes_has_no_gaps(self):
        stairs = [Obstacle.rect(10 * k, 10 * k, 10 * k + 10, 10 * k + 10) for k in range(4)]
        w = World(ROOM, 0.6, stairs)
        for k in range(1, 4):                                               # across each joint
            self.assertTrue(w.blocked([10 * k - 8, 10 * k + 8], [10 * k + 8, 10 * k - 8]))
        paths = w.paths([2, 38], [38, 2], 0)
        self.assertTrue(paths)
        for p in paths:                                                     # round an end of the stair
            self.assertFalse(any(np.allclose(v, [10 * k, 10 * k]) for v in p.points for k in range(1, 4)))

    def test_a_corner_touching_a_wall_leaves_no_gap(self):
        w = World(ROOM, 0.6, [Obstacle([(-177, 0), (-167, 10), (-157, 0), (-167, -10)])])
        paths = w.paths([-175, 30], [-175, -30], 0)
        self.assertTrue(paths)
        for p in paths:                                                     # round its free corner only
            self.assertTrue(np.allclose(p.points[1:-1], [[-157, 0]]))
        self.assertTrue(w.blocked([-177, 30], [-177, -30]))                 # along the wall through it

    def test_corner_to_corner_through_an_obstacle_is_blocked(self):
        w = World(ROOM, 0.6, [Obstacle.rect(-10, -10, 10, 10)])
        self.assertTrue(w.blocked([-50, -50], [50, 50]))
        self.assertFalse(any(p.kind == "direct" for p in w.paths([-50, -50], [50, 50], 0)))

    def test_grazing_a_face_or_a_corner_stays_clear(self):
        w = World(ROOM, 0.6, [Obstacle.rect(-10, -10, 10, 10)])
        self.assertFalse(w.blocked([-50, 10], [50, 10]))                    # along the top face
        self.assertFalse(w.blocked([-50, -30], [30, 50]))                   # through one corner only
        self.assertFalse(w.blocked([-177, 0], [-177, 100]))                 # along a wall
        self.assertTrue(w.blocked([-50, 9.99], [50, 9.99]))

    def test_skipped_segments_are_ignored(self):
        w = World(ROOM, 0.6, [Obstacle.rect(-10, -10, 10, 10)])
        own = tuple(i for i in range(w.segment_count) if w._owner[i] == 0)
        self.assertTrue(w.blocked([-50, 0], [50, 0]))
        self.assertFalse(w.blocked([-50, 0], [50, 0], own))
        self.assertFalse(w.blocked([-10, 10], [10, 10], own))               # walking round its own edge


class DegenerateObstacleTest(unittest.TestCase):
    def test_an_obstacle_with_no_area_contains_nothing(self):
        point = Obstacle([(5.0, 5.0)] * 3)
        line  = Obstacle([(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)])
        for o in (point, line):
            self.assertEqual(o.area, 0.0)
            for p in ([1000, -1000], [5, 5], [5, 0], [0, 50]):
                self.assertFalse(o.contains(p))
        box = Obstacle.rect(0, 0, 10, 10)
        self.assertEqual(box.area, 100.0)
        self.assertTrue(box.contains([10, 5]))                               # on the edge
        self.assertFalse(box.contains([10, 5], strict=True))
        self.assertTrue(box.contains([5, 5], strict=True))

    def test_load_world_skips_them_with_a_warning(self):
        path = os.path.join(tempfile.mkdtemp(), "world.toml")
        with open(path, "w") as fh:
            fh.write('wall_reflectivity = 0.6\nwalls = [[-100, -100], [100, -100], [100, 100], [-100, 100]]\n'
                     '[[obstacles]]\nname = "dot"\npoints = [[5, 5], [5, 5], [5, 5]]\n'
                     '[[obstacles]]\nname = "line"\npoints = [[0, 0], [10, 0], [20, 0]]\n'
                     '[[obstacles]]\nname = "box"\npoints = [[0, 0], [10, 0], [10, 10], [0, 10]]\n')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            w = load_world(path, np.zeros((3, 2)))
        self.assertEqual([o.name for o in w.obstacles], ["box"])
        self.assertEqual(len(caught), 2)
        self.assertIn("dot", str(caught[0].message))


class FresnelZoneTest(unittest.TestCase):
    def setUp(self):
        self.w = World([(-300, -300), (300, -300), (300, 300), (-300, 300)], 0.6,
                       [Obstacle.rect(-5, -100, 5, 0)])                      # a panel, top corner at (0, 0)

    def test_no_jump_at_the_edge_of_a_shadow(self):
        dst = np.array([100.0, 0.0])
        below = best(self.w.paths([-100, -0.01], dst, 0))
        above = best(self.w.paths([-100, 0.01], dst, 0))
        self.assertEqual((below.kind, above.kind), ("diffract", "direct"))
        self.assertAlmostEqual(20 * math.log10(below.gain), 20 * math.log10(above.gain), delta=0.2)
        self.assertAlmostEqual(20 * math.log10(above.gain), -6.0, delta=0.1)

    def test_loss_fades_out_within_the_first_fresnel_zone(self):
        dst  = np.array([100.0, 0.0])
        gain = [best(self.w.paths([-100, y], dst, 0)).gain for y in (0.01, 1.0, 3.0, 10.0)]
        self.assertEqual(gain, sorted(gain))
        self.assertEqual(gain[-1], 1.0)                                      # 10 cm clear: no loss
        self.assertEqual(best(World().paths([0, 0], [100, 0], 0)).gain, 1.0)


def reference_reflections(w: World, src, dst, order: int) -> set:
    """The image-source search written plainly, one candidate at a time."""
    A, B, refl, _ = w._segments()

    def mirror(p, a, b):
        d = (b - a) / np.hypot(*(b - a))
        return 2.0 * (a + d * np.dot(p - a, d)) - p

    def hit(img, to, i):
        r, s = to - img, B[i] - A[i]
        den = _cross(r, s)
        if abs(den) < 1e-12:
            return None
        ap = A[i] - img
        t, u = _cross(ap, s) / den, _cross(ap, r) / den
        return img + t * r if EPS < t < 1 - EPS and EPS < u < 1 - EPS else None

    out, src, dst = set(), np.asarray(src, float), np.asarray(dst, float)
    for i in range(len(A)):
        img = mirror(src, A[i], B[i])
        p = hit(img, dst, i)
        if p is not None and not w.blocked(src, p, (i,)) and not w.blocked(p, dst, (i,)):
            out.add((i, None, round(refl[i], 9)))
        for j in range(len(A)) if order >= 2 else ():
            if j == i:
                continue
            p2 = hit(mirror(img, A[j], B[j]), dst, j)
            p1 = None if p2 is None else hit(img, p2, i)
            if p1 is None or w.blocked(src, p1, (i,)) or w.blocked(p1, p2, (i, j)) or w.blocked(p2, dst, (j,)):
                continue
            out.add((i, j, round(refl[i] * refl[j], 9)))
    return out


def which(w: World, path) -> tuple:
    """(first segment, second segment or None, gain) a reflected path bounces off."""
    A, B, _, _ = w._segments()

    def on(q):
        d = np.abs(_cross(B - A, q - A)) / np.hypot(*(B - A).T)
        return int(np.argmin(d))
    pts = path.points
    return (on(pts[1]), on(pts[2]) if len(pts) == 4 else None, round(path.gain, 9))


class ReflectionSearchTest(unittest.TestCase):
    def test_vectorised_search_finds_the_same_paths(self):
        rng = np.random.default_rng(3)
        for trial in range(12):
            obs = []
            for _ in range(trial % 4):
                x, y = rng.uniform(-140, 120), rng.uniform(-200, 180)
                obs.append(Obstacle.rect(x, y, x + rng.uniform(5, 40), y + rng.uniform(5, 40),
                                         reflectivity=rng.uniform(0.3, 0.9)))
            w = World(ROOM, 0.6, obs)
            src, dst = rng.uniform([-170, -230], [170, 230], (2, 2))
            if w.solid(src) or w.solid(dst):
                continue
            for order in (1, 2):
                got = {which(w, p) for p in w.paths(src, dst, order) if p.kind == "reflect"}
                self.assertEqual(got, reference_reflections(w, src, dst, order), (trial, order))


if __name__ == "__main__":
    unittest.main()
