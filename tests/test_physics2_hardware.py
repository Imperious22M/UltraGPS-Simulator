"""The hardware model's timing limits, injected timeouts, crosstalk, enclosed
or undefined transmitters, interference between overlapping arrivals, ambient
noise and the spread between modules."""

import time
import unittest
from dataclasses import replace

import numpy as np

from tests.helpers import run
from engine import acoustics as ac
from engine import maps
from engine.hardware import PRESETS, STATUS, Environment, Faults, Hardware, Receiver
from engine.simulation import Simulation
from engine.world import Obstacle, World

STEADY = dict(amp_jitter_db=0.0, tick_dither=False)


def steady_sim(**env) -> Simulation:
    return Simulation(seed=0, env=Environment(**STEADY).copy(**env), faults=Faults(noise_cm=0.0))


def wall_hardware(**env) -> Hardware:
    """One receiver 10 cm off the bottom wall of a hard room, looking along it,
    so the wall's echo arrives a cycle or two after the direct sound."""
    room = World([(-200, -200), (200, -200), (200, 200), (-200, 200)], 0.9)
    rx   = [Receiver(0, np.array([-150.0, -190.0]), 0.0, 1.0, 0.0)]
    e    = Environment(snr_db=14.0, gain_spread_db=0.0, aim_spread_deg=0.0, **STEADY).copy(**env)
    return Hardware(rx, room, e, Faults(noise_cm=0.0), seed=0)


class TimingLimitTest(unittest.TestCase):
    def test_nothing_counts_after_the_module_gives_up(self):
        sim = steady_sim(timeout_ms=10.0)
        e   = sim.env
        rs  = sim.hardware.measure([117, 177], 90.0)
        for r in rs:
            self.assertLessEqual(r.echo_ms, e.rx_ready_ms + e.timeout_ms + 1e-9)
        far = rs[0]                                                    # 438 cm: sound needs ~17 ms
        self.assertEqual(far.status, "lost")
        self.assertIsNone(far.detected)
        self.assertAlmostEqual(far.echo_ms, 13.0)
        self.assertEqual(rs[5].status, "ok")                           # 14 cm away: in time

    def test_nothing_counts_after_the_base_station_stops_sampling(self):
        sim = steady_sim(window_ms=10.0)
        rs  = sim.hardware.measure([117, 177], 90.0)
        late = [r for r in rs if r.status == "missed"]
        self.assertGreaterEqual(len(late), 3)
        for r in late:
            self.assertEqual((r.tick, r.echo_ms, r.measured_cm, r.detected), (0, None, None, None))
        for r in rs:
            self.assertTrue(r.echo_ms is None or r.echo_ms <= 10.0)

    def test_coverage_map_applies_the_same_limits(self):
        sim = steady_sim(window_ms=10.0, reflections=False, diffraction=False, interference=False)
        _, _, Z    = maps.coverage(sim, 60.0)
        _, _, nlos = maps.first_arrival(sim, 60.0)
        inside = np.isfinite(Z)
        self.assertTrue(np.array_equal(Z[inside] + nlos[inside], np.full(inside.sum(), 6.0)))
        self.assertLess(np.nanmin(Z), 6.0)


class InjectedTimeoutTest(unittest.TestCase):
    def test_it_reads_like_a_real_timeout(self):
        real = run(Simulation(seed=0, env=Environment(radio_loss=1.0), faults=Faults(noise_cm=0.0)), [0, 0], 1)
        fake = run(Simulation(seed=0, faults=Faults(noise_cm=0.0, timeout_rate=1.0)), [0, 0], 1)
        e    = Environment()
        for r, f in zip(real.readings, fake.readings):
            self.assertEqual(f.status, "timeout")
            self.assertAlmostEqual(f.measured_cm, r.measured_cm, places=6)
            self.assertAlmostEqual(f.echo_ms, e.rx_ready_ms + e.timeout_ms)
            self.assertIsNone(f.detected)

    def test_beyond_the_window_it_reads_zero(self):
        frame = run(Simulation(seed=0, env=Environment(timeout_ms=200.0),
                               faults=Faults(noise_cm=0.0, timeout_rate=1.0)), [0, 0], 1)
        self.assertEqual(set(frame.status), {"missed"})
        self.assertEqual(set(frame.ticks), {0})


class CrosstalkTest(unittest.TestCase):
    def test_reflections_off_leaves_only_each_receivers_own_echo(self):
        sim = steady_sim(crosstalk=True, reflections=False)
        arr = sim.hardware.arrivals([0.0, 0.0])[1]
        others = {a.path.kind for a in arr if a.source not in ("TX", "R1")}
        self.assertTrue(others)
        self.assertNotIn("reflect", others)
        self.assertEqual({a.path.kind for a in arr if a.source == "R1"}, {"reflect"})

    def test_a_mis_aimed_receiver_sends_its_burst_along_its_real_axis(self):
        def ratio(aim):
            rx = [Receiver(0, np.array([0.0, 0.0]), 0.0, 1.0, 0.0, spread=(0.0, 0.0)),
                  Receiver(1, np.array([100.0, 0.0]), 180.0, 1.0, 0.0, spread=(0.0, aim))]
            hw = Hardware(rx, World(), Environment(crosstalk=True, aim_spread_deg=20.0, **STEADY),
                          Faults(noise_cm=0.0), seed=0)
            return [a.ratio_db for a in hw.arrivals([50.0, 50.0])[0] if a.source == "R1"]
        straight, askew = ratio(0.0), ratio(1.0)                            # R1 facing R0, then 20° off
        self.assertEqual((len(straight), len(askew)), (1, 1))
        drop = -20 * np.log10(ac.directivity(20.0, Environment().rx_beam_deg))
        self.assertGreater(drop, 2.0)
        self.assertAlmostEqual(straight[0] - askew[0], drop, places=6)

    def test_cached_paths_give_the_same_readings_and_follow_the_room(self):
        def pings(hw, fresh):
            out = []
            for p in ([0, 0], [40, -60], [-80, 120]):
                if fresh:
                    hw._xt_key = None                                    # trace everything again
                out.append([(r.status, r.tick, len(r.arrivals)) for r in hw.measure(p)])
            return out
        a, b = (Simulation(seed=4, env=Environment(crosstalk=True, bounces=2)) for _ in range(2))
        self.assertEqual(pings(a.hardware, False), pings(b.hardware, True))

        heard = lambda sim: {a.source for a in sim.hardware.arrivals([0, 0])[1] if a.path.kind == "direct"}
        sim = steady_sim(crosstalk=True)
        self.assertIn("R4", heard(sim))
        sim.world.obstacles.append(Obstacle.rect(-20, -40, 20, 40))       # between R1 and R4
        self.assertNotIn("R4", heard(sim))

    def test_crosstalk_with_obstacles_is_fast(self):
        sim = Simulation(seed=0, env=Environment(**PRESETS["Harsh"]).copy(crosstalk=True))
        for k in range(3):
            x = -100 + 25 * k
            sim.world.obstacles.append(Obstacle.rect(x, -150 + 30 * k, x + 15, -130 + 30 * k))
        sim.hardware.measure([20, 60])
        t = time.perf_counter()
        for p in ([20, 60], [-30, 10], [60, -90]):
            sim.hardware.measure(p)
        self.assertLess((time.perf_counter() - t) / 3, 0.15)              # was 0.4 s a ping


class SilentTransmitterTest(unittest.TestCase):
    def test_a_transmitter_with_no_position_is_lost_not_a_crash(self):
        sim = steady_sim()
        rs  = sim.hardware.measure([float("nan"), 0.0])
        self.assertEqual({r.status for r in rs}, {"lost"})
        self.assertFalse(any(r.los for r in rs))
        self.assertFalse(any(a.source == "TX" for r in rs for a in r.arrivals))

    def test_a_transmitter_inside_an_obstacle_is_enclosed(self):
        sim = steady_sim()
        sim.world.obstacles.append(Obstacle.rect(-30, -30, 30, 30))
        rs = sim.hardware.measure([0.0, 0.0])
        self.assertEqual({r.status for r in rs}, {"lost"})
        self.assertFalse(any(r.los for r in rs))
        edge = sim.hardware.measure([30.0, 0.0])                           # on its face: heard
        self.assertIn(edge[4].status, ("ok", "walk"))


class InterferenceTest(unittest.TestCase):
    def sweep(self, hw):
        return [hw.measure([x, -190.0])[0] for x in np.arange(-50.0, 190.0, 2.0)]

    def test_an_echo_close_behind_the_direct_sound_makes_it_fade(self):
        on, off = self.sweep(wall_hardware()), self.sweep(wall_hardware(interference=False))
        fades = [(a, b) for a, b in zip(on, off) if a.status == "fade"]
        self.assertGreater(len(fades), 10)
        self.assertNotIn("fade", {r.status for r in off})
        self.assertTrue(any(a.measured_cm != b.measured_cm for a, b in fades))
        self.assertTrue(any(a.detected is None and b.detected is not None for a, b in fades))   # cancelled
        self.assertTrue(any(a.detected is not None and b.detected is None for a, b in fades))   # reinforced
        for a, b in zip(on, off):
            self.assertEqual(len(a.arrivals), len(b.arrivals))            # every arrival is still listed
            if a.detected is not None:
                self.assertLess(abs(a.detected.detect_ms - a.echo_ms), ac.CYCLE_MS / 2 + 1e-9)

    def test_it_is_deterministic_and_off_when_ideal(self):
        a, b = self.sweep(wall_hardware()), self.sweep(wall_hardware())
        self.assertEqual([(r.status, r.echo_ms) for r in a], [(r.status, r.echo_ms) for r in b])
        self.assertFalse(Environment.ideal().interference)
        self.assertTrue(PRESETS["Realistic"]["interference"] and PRESETS["Harsh"]["interference"])
        strong = self.sweep(wall_hardware(snr_db=30.0))                    # detected before the echo matters
        self.assertNotIn("fade", {r.status for r in strong})

    def test_statuses_explain_themselves(self):
        for status in ("fade", "noise"):
            self.assertIn(status, STATUS)


class SideStream:
    """A generator that hands ambient-noise draws to a second stream, so the
    main stream sees a run that never draws noise."""
    def __init__(self, seed: int):
        self.main, self.side = np.random.default_rng(seed), np.random.default_rng(seed + 1)

    def exponential(self, *args, **kw):
        return self.side.exponential(*args, **kw)

    def __getattr__(self, name):
        return getattr(self.main, name)


class AmbientNoiseTest(unittest.TestCase):
    def test_noise_trips_the_comparator_early(self):
        sim = steady_sim(noise_rate_hz=5000.0)                             # a trigger every 0.2 ms
        quiet = steady_sim()
        for p in ([0, 0], [80, 100], [-60, -150]):
            for r, q in zip(sim.hardware.measure(p), quiet.hardware.measure(p)):
                if r.status == "noise":
                    self.assertIsNone(r.detected)
                    self.assertLess(r.echo_ms, q.echo_ms)
                    self.assertGreaterEqual(r.echo_ms, sim.env.rx_ready_ms)
        statuses = [r.status for _ in range(5) for r in sim.hardware.measure([0, 0])]
        self.assertGreater(statuses.count("noise"), 20)

    def test_with_no_noise_the_random_stream_is_untouched(self):
        def pings(rate, rng):
            sim = steady_sim(noise_rate_hz=rate, amp_jitter_db=1.0, tick_dither=True, radio_loss=0.1)
            sim.hardware.rng = rng
            out = [[(r.status, r.tick, r.echo_ms) for r in sim.hardware.measure(p)]
                   for p in ([0, 0], [30, -40], [-60, 90], [120, 150])]
            return out, rng.random()
        quiet = pings(0.0, np.random.default_rng(7))
        never = pings(1e-12, SideStream(7))                                 # noise drawn elsewhere, never first
        self.assertEqual(quiet, never)

    def test_noise_comes_from_the_hardware_rng_only(self):
        def pings(global_seed):
            np.random.seed(global_seed)
            state = np.random.get_state()[1].copy()
            sim   = steady_sim(noise_rate_hz=200.0)
            out   = [r.echo_ms for _ in range(20) for r in sim.hardware.measure([0, 0])]
            self.assertTrue(np.array_equal(np.random.get_state()[1], state))   # the global generator untouched
            return out, [r.status for r in sim.hardware.measure([0, 0])]
        a, b = pings(1), pings(2)
        self.assertEqual(a, b)
        self.assertIn("noise", a[1])

    def test_off_by_default_and_in_the_deterministic_maps(self):
        self.assertEqual((PRESETS["Realistic"]["noise_rate_hz"], PRESETS["Ideal"]["noise_rate_hz"]), (0.0, 0.0))
        self.assertGreater(PRESETS["Harsh"]["noise_rate_hz"], 0.0)
        quiet = maps.first_arrival(steady_sim(), 60.0)[2]
        noisy = maps.first_arrival(steady_sim(noise_rate_hz=5000.0), 60.0)[2]
        self.assertTrue(np.array_equal(quiet, noisy, equal_nan=True))


class ModuleSpreadTest(unittest.TestCase):
    def test_each_module_is_drawn_once_and_kept(self):
        sim = Simulation(seed=0)
        hw  = sim.hardware
        before = [hw.module(i) for i in range(6)]
        self.assertTrue(any(g != 0.0 for g, _ in before) and any(a != 0.0 for _, a in before))
        run(sim, [0, 0], 3)
        self.assertEqual([hw.module(i) for i in range(6)], before)
        self.assertEqual([Simulation(seed=0).hardware.module(i) for i in range(6)], before)
        self.assertNotEqual([Simulation(seed=1).hardware.module(i) for i in range(6)], before)
        # Copies (the lab's, the maps') keep the same modules
        lab = Hardware([replace(r) for r in sim.receivers], sim.world, sim.env, seed=99)
        self.assertEqual([lab.module(i) for i in range(6)], before)
        steady = maps._steady_hardware(maps.scene(sim))
        self.assertEqual([steady.module(i) for i in range(6)], before)
        # The spread scales with its σ and is off when ideal
        sim.env = sim.env.copy(gain_spread_db=2 * sim.env.gain_spread_db)
        self.assertAlmostEqual(hw.module(0)[0], 2 * before[0][0])
        sim.env = Environment.ideal()
        self.assertEqual({hw.module(i) for i in range(6)}, {(0.0, 0.0)})

    def test_gain_and_aim_change_what_a_receiver_hears(self):
        def strength(spread, **env):
            rx = Receiver(0, np.array([0.0, 0.0]), 0.0, 1.0, 0.0, spread=spread)
            hw = Hardware([rx], World(), Environment(**STEADY).copy(**env), Faults(noise_cm=0.0), seed=0)
            return hw.arrivals([150.0, 0.0])[0][0].ratio_db
        base = strength((0.0, 0.0))
        self.assertAlmostEqual(strength((1.0, 0.0), gain_spread_db=4.0), base + 4.0)
        self.assertLess(strength((0.0, 1.0), aim_spread_deg=20.0), base - 3.0)   # aimed 20° off it
        rx = Receiver(0, np.array([0.0, 0.0]), 0.0, 1.0, 0.0, spread=(0.0, 1.0))
        hw = Hardware([rx], World(), Environment(fov_cutoff=True, aim_spread_deg=20.0, **STEADY),
                      Faults(noise_cm=0.0), seed=0)
        self.assertEqual(hw.measure([150.0, 0.0])[0].status, "angle")           # its real axis decides

    def test_coverage_uses_each_modules_sensitivity(self):
        sim = steady_sim(reflections=False, diffraction=False)
        _, _, before = maps.coverage(sim, 60.0)
        sim.receivers[2].spread = (-40.0, 0.0)                              # 60 dB deaf at σ 1.5 dB
        _, _, after = maps.coverage(sim, 60.0)
        self.assertLessEqual(np.nanmax(after), 5.0)
        self.assertTrue(np.all(after[np.isfinite(after)] <= before[np.isfinite(before)]))


if __name__ == "__main__":
    unittest.main()
