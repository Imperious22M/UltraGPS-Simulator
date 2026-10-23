"""Acoustics, room geometry and the hardware timing model."""

import math
import unittest

import numpy as np

from tests.helpers import ideal_sim, run
from engine import acoustics as ac
from engine.hardware import Environment, Faults
from engine.simulation import Simulation
from engine.world import Obstacle, World


class AcousticsTest(unittest.TestCase):
    def test_absorption_matches_iso_9613(self):
        self.assertAlmostEqual(ac.absorption_db_per_m(20, 70, 1000.0), 0.00498, places=4)   # ISO table
        self.assertAlmostEqual(ac.absorption_db_per_m(20, 50), 1.32, delta=0.05)          # 40 kHz in a room

    def test_speed_of_sound(self):
        self.assertAlmostEqual(ac.speed_of_sound(20), 343.4, delta=0.1)
        self.assertGreater(ac.speed_of_sound(20, 100), ac.speed_of_sound(20, 0))

    def test_humidity_matters_more_in_warm_air(self):
        moist = {t: ac.speed_of_sound(t, 100) - ac.speed_of_sound(t, 0) for t in (0, 20, 30, 45)}
        self.assertAlmostEqual(moist[20], 1.27, delta=0.1)               # saturated vs dry at 20 °C
        self.assertAlmostEqual(moist[30], 2.4, delta=0.2)
        self.assertAlmostEqual(moist[45], 5.6, delta=0.3)
        self.assertEqual(list(moist.values()), sorted(moist.values()))

    def test_beam_is_minus_6_db_at_its_half_angle_and_falls_off(self):
        self.assertAlmostEqual(ac.directivity(0, 27.5), 1.0)
        self.assertAlmostEqual(ac.directivity(27.5, 27.5), 0.5, places=3)
        self.assertEqual(ac.directivity(180, 27.5), ac.BACK_LEVEL)
        angles = np.arange(0.0, 180.0001, 0.1)
        for half in (5.0, 10.0, 15.0, 27.5, 45.0, 80.0):
            db = 20 * np.log10([ac.directivity(a, half) for a in angles])
            self.assertTrue(np.all(np.diff(db) <= 1e-9), half)           # no nulls, no upward steps
            self.assertLess(np.max(np.abs(np.diff(db))), 1.0, half)      # and no cliff (was 14 dB at 90°)
            self.assertAlmostEqual(ac.directivity(120, half), ac.BACK_LEVEL)

    def test_weak_bursts_are_detected_later_or_not_at_all(self):
        cycles = [ac.detect_cycle(r, 6.0) for r in (100, 10, 3, 1.5)]
        self.assertEqual(cycles, sorted(cycles))
        self.assertGreater(cycles[-1], cycles[0])
        self.assertIsNone(ac.detect_cycle(1.2, 6.0))                     # never crosses in 8 cycles
        self.assertIsNone(ac.detect_cycle(0.9, 6.0))
        self.assertIsNone(ac.detect_cycle(float("nan"), 6.0))

    def test_overlapping_bursts_add_as_phasors(self):
        C = ac.CYCLE_MS
        for ratio, start, rise in ((1.5, 10.0, 6.0), (40.0, 3.3, 2.0), (1.37, 7.0, 6.0), (0.9, 5.0, 6.0)):
            k   = ac.detect_cycle(ratio, rise)
            hit = ac.first_detection([start], [ratio], rise)
            if k is None:
                self.assertIsNone(hit)
            else:
                self.assertAlmostEqual(hit[0], start + k * C, places=9)     # alone: exactly detect_cycle
        self.assertIsNone(ac.detect_cycle(0.9, 6.0))
        both = ac.first_detection([10.0, 10.0], [0.9, 0.9], 6.0)           # in phase: twice as strong
        self.assertAlmostEqual(both[0], 10.0 + ac.detect_cycle(1.8, 6.0) * C)
        self.assertIsNone(ac.first_detection([10.0, 10.0 + 0.5 * C], [0.9, 0.9], 6.0))   # out of phase
        self.assertIsNone(ac.first_detection([10.0, 10.0 + 2.5 * C], [2.0, 1.8], 6.0))   # an echo cancels it
        far = ac.first_detection([10.0, 12.0], [2.0, 1.8], 6.0)            # no overlap: the first alone
        self.assertAlmostEqual(far[0], 10.0 + ac.detect_cycle(2.0, 6.0) * C)
        self.assertEqual(far[1], 0)


class WorldTest(unittest.TestCase):
    def setUp(self):
        self.room = World([(-100, -100), (100, -100), (100, 100), (-100, 100)], 0.8)

    def test_reflection_follows_the_image_source(self):
        src, dst = np.array([-50.0, 0.0]), np.array([50.0, 0.0])
        refl = [p for p in self.room.paths(src, dst, 1) if p.kind == "reflect"]
        self.assertEqual(len(refl), 4)
        top = min(refl, key=lambda p: abs(p.points[1][1] - 100))
        self.assertAlmostEqual(top.length, math.hypot(100, 200), places=6)   # image at (-50, 200)
        self.assertAlmostEqual(top.gain, 0.8)

    def test_second_order_bounces(self):
        two = [p for p in self.room.paths([-50, 0], [50, 0], 2) if len(p.points) == 4]
        self.assertTrue(two)
        self.assertTrue(all(abs(p.gain - 0.64) < 1e-9 for p in two))

    def test_obstacle_blocks_and_sound_bends_round_it(self):
        self.room.obstacles.append(Obstacle.rect(-10, -10, 10, 10))
        paths = self.room.paths([-50, 0], [50, 0], 0)
        self.assertFalse(any(p.kind == "direct" for p in paths))
        bent = [p for p in paths if p.kind == "diffract"]
        self.assertEqual(len(bent), 2)                                   # over and under
        for p in bent:
            self.assertGreater(p.length, 100.0)
            self.assertLess(p.gain, 0.1)                                 # tens of dB of loss

    def test_walls_block_a_source_outside(self):
        self.assertTrue(self.room.blocked([0, 0], [0, 150]))
        self.assertFalse(self.room.inside([0, 150]))


class HardwareTest(unittest.TestCase):
    def test_realistic_readings_are_close_in_the_open(self):
        sim = Simulation(seed=0, faults=Faults(noise_cm=0.0))
        frame = run(sim, [0, 0], frames=1)
        for r in frame.readings:
            self.assertIn(r.status, ("ok", "walk"))
            self.assertLess(abs(r.measured_cm - r.true_cm), 3.0)

    def test_range_walk_grows_off_axis(self):
        sim = Simulation(seed=0, faults=Faults(noise_cm=0.0), env=Environment(amp_jitter_db=0, tick_dither=False))
        rx = sim.receivers[1]                                            # at (-127, 0), facing +x
        ahead = sim.hardware.measure([60, 0])[1]
        aside = sim.hardware.measure([-127 + 187 * math.cos(math.radians(70)), 187 * math.sin(math.radians(70))])[1]
        self.assertAlmostEqual(ahead.true_cm, aside.true_cm, places=6)
        self.assertGreater(aside.measured_cm - aside.true_cm, ahead.measured_cm - ahead.true_cm)
        self.assertEqual(rx.facing_deg, 0.0)

    def test_obstacle_makes_a_receiver_read_a_reflection(self):
        sim = Simulation(seed=0, faults=Faults(noise_cm=0.0))
        sim.world.obstacles.append(Obstacle.rect(-100, -20, -80, 20))   # in front of R1
        r = run(sim, [0, 0], frames=1).readings[1]
        self.assertFalse(r.los)
        self.assertIn(r.status, ("echo", "bent", "lost"))
        if r.status == "echo":
            self.assertGreater(r.measured_cm, r.true_cm + 20)

    def test_blocked_receiver_has_no_direct_path(self):
        sim = Simulation(seed=0, faults=Faults(noise_cm=0.0))
        sim.set_receiver_state(1, "blocked")
        r = run(sim, [0, 0], frames=1).readings[1]
        self.assertFalse(any(a.path.kind == "direct" for a in r.arrivals))
        self.assertNotIn(r.status, ("ok", "walk"))

    def test_lost_radio_trigger_times_out_every_receiver(self):
        sim = Simulation(seed=0, env=Environment(radio_loss=1.0))
        frame = run(sim, [0, 0], frames=1)
        self.assertEqual(set(frame.status), {"radio"})
        self.assertTrue(all(d > 1000 for d in frame.solution.distances))

    def test_timeout_longer_than_the_window_reads_zero(self):
        sim = Simulation(seed=0, env=Environment(radio_loss=1.0, timeout_ms=200.0))
        frame = run(sim, [0, 0], frames=1)
        self.assertEqual(set(frame.status), {"missed"})
        self.assertEqual(set(frame.ticks), {0})

    def test_trigger_timing_offset_shifts_every_range_equally(self):
        sim = ideal_sim(sync_offset_us=100.0)  # 100 µs late
        frame = run(sim, [30, 50], frames=1)
        shift = np.array(frame.solution.distances) - [r.true_cm for r in frame.readings]
        np.testing.assert_allclose(shift, 3.43, atol=0.6)                # 100 µs of sound

    def test_temperature_scales_ranges(self):
        sim = ideal_sim(temperature_c=40.0)
        r = run(sim, [30, 50], frames=1).readings[0]
        expect = r.true_cm * ac.speed_of_sound(20, 50) / ac.speed_of_sound(40, 50)
        self.assertAlmostEqual(r.measured_cm, expect, places=6)

    def test_crosstalk_reads_short(self):
        sim = Simulation(seed=0, faults=Faults(noise_cm=0.0), env=Environment(crosstalk=True))
        frame = run(sim, [100, 150], frames=1)
        cross = [r for r in frame.readings if r.status == "crosstalk"]
        self.assertTrue(cross)
        self.assertTrue(any(r.measured_cm < r.true_cm for r in cross))

    def test_receivers_cannot_hear_before_they_listen(self):
        # Transmitter fires before the receivers are ready: a close receiver misses the direct sound
        sim = Simulation(seed=0, env=Environment(burst_lead_ms=1.0, rx_ready_ms=3.0))
        r = run(sim, [-117, 0], frames=1).readings[1]
        self.assertNotIn(r.status, ("ok", "walk"))


if __name__ == "__main__":
    unittest.main()
