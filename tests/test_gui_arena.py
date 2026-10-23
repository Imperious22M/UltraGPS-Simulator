"""Arena view: drawing in both palettes, layers, the ping animation, tooltips,
export, and editing the scene with the mouse (under the runner's lock, or
handed to its ``post``)."""

import dataclasses
import os
import tempfile
import time
import types
import unittest
from unittest import mock

os.environ["QT_QPA_PLATFORM"] = "offscreen"
try:
    from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
    from PyQt6.QtGui import QImage, QMouseEvent, QPainter, QWheelEvent
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication
except ImportError as exc:
    raise unittest.SkipTest(f"PyQt6 is not installed: {exc}")

import numpy as np

from tests.helpers import ideal_sim  # noqa: F401  (puts the repo on sys.path)
from engine import lab
from engine.hardware import STATUS
from engine.simulation import Simulation
from engine.solvers import Estimate
from engine.world import Obstacle
from gui import draw, theme
from gui.arena import COLOUR_BAR_PX, HUD_BOTTOM_PX, LAYERS, ArenaView

app = QApplication.instance() or QApplication([])

LEFT, RIGHT, NONE = Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton, Qt.MouseButton.NoButton


class CountingLock:
    """Stands in for the runner's lock and counts how often it is taken."""

    def __init__(self):
        self.entered = 0

    def __enter__(self):
        self.entered += 1

    def __exit__(self, *exc):
        return False


class Posted:
    """Stands in for ``SimRunner.post``: keeps the changes until ``run()``,
    as while a step is in progress."""

    def __init__(self):
        self.fns = []

    def __call__(self, fn):
        self.fns.append(fn)

    def run(self):
        fns, self.fns = self.fns, []
        for fn in fns:
            fn()


def mouse(widget, kind, x, y, button=LEFT, buttons=None):
    pos = QPointF(x, y)
    if buttons is None:
        buttons = NONE if kind == QEvent.Type.MouseButtonRelease else button
    QApplication.sendEvent(widget, QMouseEvent(kind, pos, widget.mapToGlobal(pos), button, buttons,
                                               Qt.KeyboardModifier.NoModifier))


def drag(widget, a, b, button=LEFT):
    mouse(widget, QEvent.Type.MouseButtonPress, *a, button)
    for t in (0.25, 0.5, 1.0):
        mouse(widget, QEvent.Type.MouseMove, a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), NONE, button)
    mouse(widget, QEvent.Type.MouseButtonRelease, *b, button)


def click(widget, x, y, button=LEFT):
    drag(widget, (x, y), (x, y), button)


def wheel(widget, x, y, notches=1):
    pos = QPointF(x, y)
    QApplication.sendEvent(widget, QWheelEvent(pos, widget.mapToGlobal(pos), QPoint(0, 0), QPoint(0, 120 * notches),
                                               NONE, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase,
                                               False))


def pixels(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB32)
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    return np.frombuffer(ptr, np.uint32).reshape(img.height(), -1)[:, :img.width()] & 0xFFFFFF


def echo_sim() -> Simulation:
    """R1 hears the far wall's echo: a crate blocks its line of sight and the
    field-of-view limit keeps it from hearing the sound bend round the crate."""
    sim = Simulation(seed=3)
    sim.env = sim.env.copy(fov_cutoff=True)
    sim.world.obstacles = [Obstacle.rect(-112, 4, -98, 36, name="crate")]
    sim.survey_error(5.0, seed=1)
    sim.place((82, 76))
    return sim


class ArenaDrawingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        theme.apply(app, "vector")
        cls.sim    = echo_sim()
        cls.frames = [cls.sim.step(0.1) for _ in range(3)]

    def setUp(self):
        theme.apply(app, "vector")
        self.arena = ArenaView(self.sim)
        self.arena.resize(900, 600)
        self.arena.show()
        app.processEvents()
        for f in self.frames:
            self.arena.set_frame(f)

    def tearDown(self):
        self.arena.close()
        self.arena.deleteLater()
        theme.apply(app, "vector")

    def test_scenario_reads_an_echo(self):
        self.assertEqual(self.frames[-1].readings[1].status, "echo")
        self.assertFalse(self.frames[-1].readings[1].los)

    def test_draws_in_both_palettes(self):
        for name in ("vector", "paper"):
            theme.apply(app, name)
            img = QImage(900, 600, QImage.Format.Format_RGB32)
            self.arena.render_to(img)
            px = pixels(img)
            values, counts = np.unique(px, return_counts=True)
            bg = int(theme.current.bg[1:], 16)
            self.assertEqual(values[np.argmax(counts)], bg, name)          # mostly background
            self.assertGreater(len(values), 200, name)                      # with plenty drawn on it

    def test_every_layer_can_be_switched_off(self):
        for name in LAYERS:
            self.arena.set_layer(name, False)
        img = QImage(900, 600, QImage.Format.Format_RGB32)
        self.arena.render_to(img)
        for name in LAYERS:
            self.arena.set_layer(name, True)
        with self.assertRaises(KeyError):
            self.arena.set_layer("nonsense", True)

    def test_a_ping_animates_then_the_timer_stops(self):
        self.arena.wave_seconds = 0.05
        self.arena.set_frame(self.frames[-1])
        self.assertTrue(self.arena._timer.isActive())
        deadline = time.perf_counter() + 2.0
        while self.arena._timer.isActive() and time.perf_counter() < deadline:
            QTest.qWait(20)
        self.assertFalse(self.arena._timer.isActive())
        self.assertEqual(self.arena._pings, [])

    def test_waves_off_stops_the_animation(self):
        self.arena.set_frame(self.frames[-1])
        self.arena.set_layer("waves", False)
        self.assertFalse(self.arena._timer.isActive())
        self.arena.set_frame(self.frames[-1])
        self.assertFalse(self.arena._timer.isActive())

    def test_echo_spreads_from_the_image_source_behind_the_wall(self):
        ping   = self.arena._make_ping(self.frames[-1])
        echoes = [f for f in ping.fronts if f.key == "echo"]
        bounce = self.frames[-1].readings[1].detected.path.points[1]
        self.assertAlmostEqual(bounce[0], 177.0, places=6)                 # off the right-hand wall
        tx = self.frames[-1].true
        self.assertTrue(any(np.allclose(f.centre, (2 * 177.0 - tx[0], tx[1])) for f in echoes))
        direct = ping.fronts[0]
        self.assertEqual(direct.key, "wave")
        self.assertLess(np.max(direct.hi[np.isfinite(direct.hi)]), 600.0)  # stops at the walls
        self.assertEqual(set(ping.arrive), {i for i, r in enumerate(self.frames[-1].readings) if r.detected})

    def test_tooltips_explain_receivers_and_estimates(self):
        v   = self.arena.view
        q   = v.px(*self.sim.receivers[1].position)
        tip = self.arena.tooltip_at(q.x(), q.y())
        self.assertIn("R1", tip)
        self.assertIn(STATUS["echo"], tip)
        self.assertIn("survey error", tip)
        q   = v.px(*self.frames[-1].estimates["ROBUST"].pos)
        self.assertIn("Robust LM", self.arena.tooltip_at(q.x(), q.y()))
        ols = self.frames[-1].estimates["OLS"].pos                          # 5 m off: an arrow at the edge
        self.assertFalse(self.arena.rect().contains(v.px(*ols).toPoint()))
        q, tip = self.arena._estimate_spot(ols)
        self.assertIsNotNone(tip)
        self.assertIn("OLS", self.arena.tooltip_at(q.x(), q.y()))
        self.assertEqual(self.arena.tooltip_at(3, 3), "")

    def test_heatmap_and_svg_export(self):
        f = self.frames[-1]
        c = self.sim.believed
        xs, ys, Z = lab.cost_grid(c, f.solution.distances, f.solution.sane,
                                  (*(c.min(axis=0) - 50), *(c.max(axis=0) + 50)), 60)
        self.arena.set_heatmap(xs, ys, Z, "RMS residual (cm)", 0.0, float(np.percentile(Z, 95)))
        img = QImage(900, 600, QImage.Format.Format_RGB32)
        self.arena.render_to(img)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "arena.svg")
            self.arena.export_svg(path)
            with open(path, encoding="utf-8") as fh:
                svg = fh.read()
        self.assertIn("<svg", svg)
        self.assertIn("RMS residual", svg)
        self.arena.clear_heatmap()
        self.assertIsNone(self.arena._heatmap)

    def test_heatmap_contours_at_the_levels_given(self):
        xs, ys = np.linspace(0, 100, 11), np.linspace(0, 50, 6)
        Z      = np.tile(xs, (len(ys), 1))
        self.arena.set_heatmap(xs, ys, Z, "rising", 0.0, 100.0, levels=[-5.0, 25.0, 50.0, 150.0])
        np.testing.assert_array_equal(self.arena._heatmap.levels, [25.0, 50.0])     # outside the range: left out
        self.assertFalse(self.arena._heatmap.contours.isEmpty())
        self.arena.set_heatmap(xs, ys, Z, "rising", 0.0, 100.0)
        np.testing.assert_allclose(self.arena._heatmap.levels, [20.0, 40.0, 60.0, 80.0])
        self.arena.set_heatmap(xs, ys, Z, "rising", 0.0, 100.0, levels=[])
        self.assertTrue(self.arena._heatmap.contours.isEmpty())
        self.arena.render_to(QImage(900, 600, QImage.Format.Format_RGB32))

    def test_render_to_works_before_the_widget_is_shown(self):
        hidden = ArenaView(self.sim)
        hidden.set_frame(self.frames[-1])
        img = QImage(400, 300, QImage.Format.Format_RGB32)
        hidden.render_to(img)
        self.assertGreater(len(np.unique(pixels(img))), 50)

    def test_paint_is_quick(self):
        self.arena.resize(1400, 900)
        app.processEvents()
        full, tick = [], []
        for f in self.frames * 4:
            t = time.perf_counter()
            self.arena.set_frame(f)                                     # a new frame: everything redrawn
            self.arena.repaint()
            full.append(time.perf_counter() - t)
            t = time.perf_counter()
            self.arena.repaint()                                        # an animation tick
            tick.append(time.perf_counter() - t)
        # ~15 and ~5 ms here; generous for slow machines
        self.assertLess(np.median(full), 0.08)
        self.assertLess(np.median(tick), 0.04)

    def test_animation_ticks_reuse_the_still_picture(self):
        self.arena.set_frame(self.frames[-1])
        self.arena.repaint()
        still = self.arena._still
        self.arena._animate()
        self.arena.repaint()
        self.assertIs(self.arena._still, still)
        self.arena.set_hud("PAUSED")
        self.arena.repaint()
        self.assertIsNot(self.arena._still, still)
        still = self.arena._still
        theme.apply(app, "paper")                                       # a palette switch redraws it too
        self.arena.repaint()
        self.assertIsNot(self.arena._still, still)

    def test_a_diverged_estimate_is_left_out(self):
        f         = self.frames[-1]
        estimates = dict(f.estimates, LM=Estimate(np.array([np.nan, np.nan]), False),
                         CEP=Estimate(np.array([np.inf, 0.0]), True, cov=np.full((2, 2), np.nan)))
        self.arena.set_frame(dataclasses.replace(f, solution=dataclasses.replace(f.solution, estimates=estimates)))
        self.arena.set_layer("ellipses", True)
        self.arena.render_to(QImage(900, 600, QImage.Format.Format_RGB32))  # a paint error would abort the app
        drawn = [a for a, _ in self.arena._shown_estimates(self.arena.frame)]
        fixed = [a for a in theme.ALGO_ORDER if f.estimates[a].pos is not None]
        self.assertEqual(drawn, [a for a in fixed if a not in ("LM", "CEP")])

    def test_a_replayed_frame_has_ticks_but_no_readings_or_truth(self):
        f = dataclasses.replace(self.frames[-1], readings=[], true=None,
                                errors={a: float("nan") for a in self.frames[-1].errors})
        pings = len(self.arena._pings)
        self.arena.set_frame(f)
        self.assertEqual(len(self.arena._pings), pings)                  # no new ping: nothing to animate
        self.arena.render_to(QImage(900, 600, QImage.Format.Format_RGB32))
        q   = self.arena.view.px(*self.sim.receivers[2].position)
        tip = self.arena.tooltip_at(q.x(), q.y())
        self.assertIn(f"tick          {f.ticks[2]}", tip)
        self.assertIn("replayed", tip)

    def test_hud_makes_room_in_small_widgets(self):
        self.arena.render_to(QImage(900, 600, QImage.Format.Format_RGB32))
        self.assertIsNotNone(self.arena._legend_box)
        self.arena.render_to(QImage(500, 360, QImage.Format.Format_RGB32))
        self.assertIsNone(self.arena._legend_box)                     # it would cover the room
        c = self.sim.believed
        self.arena.set_heatmap(c[:, 0], c[:, 1], np.ones((6, 6)))
        _, tip = self.arena._estimate_spot((500.0, -900.0))           # off the bottom right
        self.assertLessEqual(tip.y(), self.arena.height() - HUD_BOTTOM_PX - COLOUR_BAR_PX)

    def test_crosstalk_sets_off_before_the_transmitter(self):
        sim = Simulation(seed=5)
        sim.env = sim.env.copy(crosstalk=True)
        sim.set_receiver_state(0, "off")
        sim.place((30, -40))
        frame = [sim.step(0.1) for _ in range(3)][-1]
        heard = {i: r.detected for i, r in enumerate(frame.readings) if r.detected and r.detected.source != "TX"}
        self.assertTrue(heard)
        ping = ArenaView(sim)._make_ping(frame)
        for i, a in heard.items():                                      # it reads short: it arrives early
            self.assertLess(ping.arrive[i], a.path.length - 100.0)


class ArenaEditingTest(unittest.TestCase):
    def setUp(self):
        theme.apply(app, "vector")
        self.sim   = Simulation(seed=0)
        self.lock  = CountingLock()
        self.arena = ArenaView(self.sim, self.lock)
        self.arena.resize(800, 600)
        self.arena.show()
        app.processEvents()
        self.seen = []
        a = self.arena
        a.targetClicked.connect(lambda x, y: self.seen.append(("target", x, y)))
        a.calPointPicked.connect(lambda x, y: self.seen.append(("calpoint", x, y)))
        a.worldEdited.connect(lambda: self.seen.append(("world",)))
        a.receiversEdited.connect(lambda: self.seen.append(("receivers",)))
        a.modeChanged.connect(lambda m: self.seen.append(("mode", m)))

    def tearDown(self):
        self.arena.close()
        self.arena.deleteLater()

    def test_click_sends_the_transmitter(self):
        click(self.arena, 400, 300)
        x, y = self.arena.view.world(400, 300)
        self.assertEqual(self.seen, [("target", x, y)])

    def test_drag_draws_an_obstacle_and_right_click_deletes_it(self):
        self.arena.set_mode("obstacle")
        before = self.sim.world.obstacles
        drag(self.arena, (380, 280), (440, 320))
        obstacles = self.sim.world.obstacles
        self.assertIsNot(obstacles, before)                                 # a new list, not appended to
        self.assertEqual(len(obstacles), len(before) + 1)
        (x0, y1), (x1, y0) = self.arena.view.world(380, 280), self.arena.view.world(440, 320)
        pts = np.asarray(obstacles[-1].points)
        np.testing.assert_allclose(pts.min(axis=0), np.round([x0, y0]))
        np.testing.assert_allclose(pts.max(axis=0), np.round([x1, y1]))
        click(self.arena, 410, 300, RIGHT)
        self.assertEqual(len(self.sim.world.obstacles), len(before))
        self.assertEqual([s for s in self.seen if s[0] == "world"], [("world",), ("world",)])
        self.assertGreaterEqual(self.lock.entered, 2)

    def test_a_tiny_drag_adds_no_obstacle(self):
        self.arena.set_mode("obstacle")
        drag(self.arena, (380, 280), (381, 281))
        self.assertEqual(self.sim.world.obstacles, [])
        self.assertNotIn(("world",), self.seen)

    def test_dragging_a_receiver_moves_where_it_really_is(self):
        self.arena.set_mode("receiver")
        r        = self.sim.receivers[0]
        believed = self.sim.believed.copy()
        start    = r.position.copy()
        q        = self.arena.view.px(*start)
        drag(self.arena, (q.x(), q.y()), (q.x() + 40, q.y()))
        np.testing.assert_allclose(r.position, start + [40 / self.arena.view.scale, 0], atol=1e-6)
        np.testing.assert_array_equal(self.sim.believed, believed)        # the library's idea is unchanged
        self.assertIn(("receivers",), self.seen)
        self.assertGreater(self.lock.entered, 0)

    def test_clicking_a_receiver_leaves_it_alone(self):
        self.arena.set_mode("receiver")
        r     = self.sim.receivers[0]
        start = r.position.copy()
        q     = self.arena.view.px(*start)
        click(self.arena, q.x(), q.y())
        np.testing.assert_array_equal(r.position, start)
        self.assertNotIn(("receivers",), self.seen)

    def test_escape_mid_drag_keeps_the_move_and_reports_it(self):
        self.arena.set_mode("receiver")
        r     = self.sim.receivers[0]
        start = r.position.copy()
        q     = self.arena.view.px(*start)
        mouse(self.arena, QEvent.Type.MouseButtonPress, q.x(), q.y())
        mouse(self.arena, QEvent.Type.MouseMove, q.x() + 30, q.y(), NONE, LEFT)
        self.arena.setFocus()
        QTest.keyClick(self.arena, Qt.Key.Key_Escape)
        self.assertEqual(self.arena.mode, "move")
        self.assertFalse(np.allclose(r.position, start))
        self.assertEqual(self.seen.count(("receivers",)), 1)
        mouse(self.arena, QEvent.Type.MouseButtonRelease, q.x() + 30, q.y())
        self.assertEqual(self.seen.count(("receivers",)), 1)

    def test_wheel_turns_a_receiver_in_receiver_mode_and_zooms_otherwise(self):
        r = self.sim.receivers[1]
        q = self.arena.view.px(*r.position)
        self.arena.set_mode("receiver")
        facing = r.facing_deg
        wheel(self.arena, q.x(), q.y())
        self.assertAlmostEqual(r.facing_deg, facing + 5.0)
        scale = self.arena.view.scale
        wheel(self.arena, 400, 300)
        self.assertGreater(self.arena.view.scale, scale)

    def test_calpick_click_picks_a_point(self):
        self.arena.set_mode("calpick")
        click(self.arena, 300, 200)
        x, y = self.arena.view.world(300, 200)
        self.assertIn(("calpoint", x, y), self.seen)
        self.assertNotIn("target", [s[0] for s in self.seen])

    def test_escape_returns_to_move_mode(self):
        self.arena.set_mode("obstacle")
        self.arena.setFocus()
        QTest.keyClick(self.arena, Qt.Key.Key_Escape)
        self.assertEqual(self.arena.mode, "move")
        self.assertEqual([s for s in self.seen if s[0] == "mode"], [("mode", "obstacle"), ("mode", "move")])
        with self.assertRaises(ValueError):
            self.arena.set_mode("teleport")

    def test_right_drag_pans_and_double_click_fits(self):
        v = self.arena.view
        fitted = (v.cx, v.cy, v.scale)
        drag(self.arena, (400, 300), (460, 300), RIGHT)
        self.assertAlmostEqual(v.cx, fitted[0] - 60 / v.scale)
        self.assertEqual(self.seen, [])                                     # panning is not a click
        mouse(self.arena, QEvent.Type.MouseButtonDblClick, 400, 300)
        self.assertEqual((v.cx, v.cy, v.scale), fitted)

    def test_cursor_reports_world_coordinates(self):
        moved = []
        self.arena.cursorMoved.connect(lambda x, y: moved.append((x, y)))
        mouse(self.arena, QEvent.Type.MouseMove, 123, 234, NONE, NONE)
        self.assertEqual(moved, [self.arena.view.world(123, 234)])


class ArenaPhysicsTest(unittest.TestCase):
    """What the arena shows of the hardware model: module aim, noise, statuses it does not know."""

    def setUp(self):
        theme.apply(app, "vector")
        self.sim = Simulation(seed=2)
        self.sim.env = self.sim.env.copy(beam_patterns=True, fov_cutoff=True, aim_spread_deg=12.0)
        self.arena = ArenaView(self.sim)
        self.arena.resize(900, 600)

    def tearDown(self):
        self.arena.deleteLater()

    def test_beams_point_along_each_modules_real_axis(self):
        img = QImage(900, 600, QImage.Format.Format_RGB32)
        p   = QPainter(img)
        try:
            with mock.patch.object(draw, "beam_lobe") as lobe:
                self.arena._draw_receiver_beams(p)
        finally:
            p.end()
        hw   = self.sim.hardware
        axes = [r.facing_deg + hw.module(i)[1] for i, r in enumerate(self.sim.receivers) if r.state != "off"]
        np.testing.assert_allclose([c.args[3] for c in lobe.call_args_list], axes)
        self.assertTrue(any(abs(hw.module(i)[1]) > 1.0 for i in range(len(self.sim.receivers))))
        i   = max(range(len(self.sim.receivers)), key=lambda k: abs(hw.module(k)[1]))
        q   = self.arena.view.px(*self.sim.receivers[i].position)
        self.assertIn("beam axis", self.arena.tooltip_at(q.x(), q.y()))

    def test_changing_the_aim_spread_redraws_the_beams(self):
        before = self.arena._scenery_state(900, 600)
        self.sim.env = self.sim.env.copy(aim_spread_deg=0.0)
        self.assertNotEqual(self.arena._scenery_state(900, 600), before)
        self.assertEqual(self.arena._aim(0), 0.0)
        q = self.arena.view.px(*self.sim.receivers[0].position)
        self.assertNotIn("beam axis", self.arena.tooltip_at(q.x(), q.y()))

    def test_hardware_without_modules_aims_along_the_facing(self):
        with mock.patch.object(self.sim, "hardware", types.SimpleNamespace()):
            self.assertEqual(self.arena._aim(1), 0.0)

    def test_noise_and_unknown_statuses_are_drawn_and_explained(self):
        self.sim.env = self.sim.env.copy(noise_rate_hz=400.0)
        self.sim.place((0, 0))
        for _ in range(40):
            frame = self.sim.step(0.1)
            noisy = [i for i, r in enumerate(frame.readings) if r.status == "noise"]
            if noisy:
                break
        self.assertTrue(noisy, "no reading tripped by noise")
        self.arena.set_frame(frame)
        self.arena.render_to(QImage(900, 600, QImage.Format.Format_RGB32))
        q   = self.arena.view.px(*self.sim.receivers[noisy[0]].position)
        tip = self.arena.tooltip_at(q.x(), q.y())
        self.assertIn(STATUS["noise"], tip)
        self.assertIn("tripped the comparator at", tip)
        odd = dataclasses.replace(frame, readings=[dataclasses.replace(r, status="humming") for r in frame.readings])
        self.arena.set_frame(odd)
        self.arena.render_to(QImage(900, 600, QImage.Format.Format_RGB32))   # no KeyError for a new status
        self.assertEqual(self.arena._status_colour(odd, 0), theme.current.text_dim)
        self.assertIn("humming: no description", self.arena.tooltip_at(q.x(), q.y()))


class ArenaPostTest(unittest.TestCase):
    """With the runner's ``post``, edits never wait for the lock: they apply
    when the runner gets to them, between two steps."""

    def setUp(self):
        theme.apply(app, "vector")
        self.sim    = Simulation(seed=0)
        self.lock   = CountingLock()
        self.posted = Posted()
        self.arena  = ArenaView(self.sim, self.lock, post=self.posted)
        self.arena.resize(800, 600)
        self.arena.show()
        app.processEvents()
        self.seen = []
        self.arena.worldEdited.connect(lambda: self.seen.append("world"))
        self.arena.receiversEdited.connect(lambda: self.seen.append("receivers"))

    def tearDown(self):
        self.arena.close()
        self.arena.deleteLater()

    def test_dragging_and_turning_a_receiver_are_posted(self):
        self.arena.set_mode("receiver")
        r     = self.sim.receivers[0]
        start = r.position.copy()
        q     = self.arena.view.px(*start)
        drag(self.arena, (q.x(), q.y()), (q.x() + 40, q.y()))
        np.testing.assert_array_equal(r.position, start)                  # not yet: a step is "running"
        self.posted.run()
        np.testing.assert_allclose(r.position, start + [40 / self.arena.view.scale, 0], atol=1e-6)
        facing = r.facing_deg
        q      = self.arena.view.px(*r.position)
        wheel(self.arena, q.x(), q.y())
        self.assertEqual(r.facing_deg, facing)
        self.posted.run()
        self.assertAlmostEqual(r.facing_deg, facing + 5.0)
        self.assertEqual(self.seen, ["receivers", "receivers"])
        self.assertEqual(self.lock.entered, 0)

    def test_adding_and_deleting_obstacles_are_posted(self):
        self.arena.set_mode("obstacle")
        drag(self.arena, (380, 280), (440, 320))
        self.assertEqual(self.sim.world.obstacles, [])
        self.posted.run()
        (first,) = self.sim.world.obstacles
        drag(self.arena, (520, 100), (580, 150))                           # a second box, still pending ...
        click(self.arena, 410, 300, RIGHT)                                 # ... when the first is deleted
        self.assertEqual(self.sim.world.obstacles, [first])
        self.posted.run()
        (second,) = self.sim.world.obstacles
        self.assertIsNot(second, first)
        self.assertEqual(second.name, "box 2")
        self.assertEqual(self.seen, ["world", "world", "world"])
        self.assertEqual(self.lock.entered, 0)


class HatchTest(unittest.TestCase):
    def test_hatch_lines_are_cut_to_the_outline(self):
        from gui.draw import _inside_pieces
        square = [QPointF(0, 0), QPointF(10, 0), QPointF(10, 10), QPointF(0, 10)]
        self.assertEqual(_inside_pieces(QPointF(-5, 5), QPointF(15, 5), square), [(0.25, 0.75)])
        notch = [QPointF(0, 0), QPointF(10, 0), QPointF(10, 10), QPointF(6, 10), QPointF(6, 4),
                 QPointF(4, 4), QPointF(4, 10), QPointF(0, 10)]
        pieces = _inside_pieces(QPointF(-5, 7), QPointF(15, 7), notch)
        self.assertEqual([(round(a, 3), round(b, 3)) for a, b in pieces], [(0.25, 0.45), (0.55, 0.75)])

    def test_a_huge_zoom_hatches_only_what_is_on_the_page(self):
        QApplication.instance() or QApplication([])
        img = QImage(200, 150, QImage.Format.Format_ARGB32)
        img.fill(0)
        v = draw.View()
        v.resize(200, 150)
        v.scale = 1e12                                   # a 1 m box is 10¹² pixels wide
        p = QPainter(img)
        t0 = time.perf_counter()
        draw.hatched_polygon(p, v, [(-50, -50), (50, -50), (50, 50), (-50, 50)], "#ff0000")
        p.end()
        self.assertLess(time.perf_counter() - t0, 1.0)
        self.assertTrue(any(img.pixel(x, 75) for x in range(200)))   # the page is inside the box: hatched


class PaletteSwitchTest(unittest.TestCase):
    def test_no_garbage_collection_runs_while_qt_walks_the_widgets(self):
        # A collection deleting a closed window half way through Qt's walk crashed it
        import gc
        from PyQt6.QtCore import QObject
        from PyQt6.QtWidgets import QLabel
        app = QApplication.instance() or QApplication([])
        seen = []

        class Watch(QObject):
            def eventFilter(self, obj, ev):
                if ev.type() in (QEvent.Type.PaletteChange, QEvent.Type.StyleChange, QEvent.Type.FontChange):
                    seen.append(gc.isenabled())
                return False

        label = QLabel("x")
        label.show()
        watch = Watch()
        app.installEventFilter(watch)
        try:
            theme.apply(app, "paper")
            theme.apply(app, "vector")
        finally:
            app.removeEventFilter(watch)
            label.close()
        self.assertTrue(seen)
        self.assertNotIn(True, seen)
        self.assertTrue(gc.isenabled())


if __name__ == "__main__":
    unittest.main()
