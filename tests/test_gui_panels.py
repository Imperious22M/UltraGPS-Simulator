"""GUI charts and control panels: they draw every kind of frame, and each
control writes straight into the simulation."""

import dataclasses
import os
import threading
import time
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QWheelEvent
    from PyQt6.QtTest import QSignalSpy, QTest
    from PyQt6.QtWidgets import (QAbstractButton, QAbstractSlider, QAbstractSpinBox, QApplication,
                                 QComboBox, QScrollBar)
except ImportError:                                  # pragma: no cover - depends on the machine
    raise unittest.SkipTest("PyQt6 is not installed")

import numpy as np

from engine.hardware import PRESETS, Environment, Faults
from engine.simulation import Simulation
from engine.world import Obstacle
from gui import theme
from gui.charts import EchoScope, ErrorChart, nice_step
from gui.panels import (ENV_GROUPS, SOLVER_FIELDS, AlgorithmPanel, EnvironmentPanel, FaultsPanel,
                        ReceiverPanel, ScenarioPanel)
from gui.runner import SimRunner
from tests.helpers import ideal_sim

app = None


def setUpModule():
    global app
    app = QApplication.instance() or QApplication([])
    theme.apply(app, "vector")


def make_sim(frames: int = 3, faults=None, **env):
    """A runner (not started) on a simulation whose transmitter stands where an
    obstacle hides it from R1, so R1 hears a reflection first."""
    sim = Simulation(seed=3, faults=faults)
    sim.world.obstacles = [Obstacle.rect(10, 32, 39, 87)]
    if env:
        sim.env = sim.env.copy(**env)
    sim.place((58, 72))
    for _ in range(frames):
        sim.step()
    return sim, SimRunner(sim), sim.last


def painted(widget, w: int = 800, h: int = 300):
    widget.resize(w, h)
    image = widget.grab()
    assert not image.isNull()
    return image


class ChartTest(unittest.TestCase):
    def test_nice_steps_are_1_2_5(self):
        self.assertEqual(nice_step(25.0, 5), 5.0)
        self.assertEqual(nice_step(7.0, 5), 2.0)
        self.assertAlmostEqual(nice_step(0.3, 3), 0.1)

    def test_error_chart_draws_both_scales_and_toggles_on_click(self):
        sim, _, _ = make_sim(30)
        chart = ErrorChart()
        chart.set_data(sim, {"LM", "CEP"})
        self.assertEqual(set(chart._series), {"LM", "CEP"})
        self.assertEqual(len(chart._t), 30)
        painted(chart)
        chart.show()
        QTest.mouseMove(chart, chart.rect().center())          # hover read-out
        painted(chart)
        QTest.mouseClick(chart, Qt.MouseButton.LeftButton, pos=chart._toggle.center().toPoint())
        self.assertTrue(chart.log_scale)
        painted(chart)
        chart.hide()

    def test_error_chart_spans_the_history_the_simulation_keeps(self):
        sim, _, _ = make_sim(0)
        sim.times = type(sim.times)([0.02 * k for k in range(1, 301)], maxlen=300)
        sim.errors["LM"] = type(sim.errors["LM"])([1.0] * 300, maxlen=300)
        chart = ErrorChart()
        chart.set_data(sim, {"LM"})
        t, series, t0, span = chart._visible()
        self.assertAlmostEqual(span, 6.0)                 # 300 frames of 0.02 s, not 30 s
        self.assertEqual(len(t), 300)
        self.assertAlmostEqual(t0 + span, 6.0)            # the latest frame at the right edge

    def test_error_chart_without_truth_or_data(self):
        sim, _, _ = make_sim(5)
        for k in sim.errors:
            sim.errors[k] = type(sim.errors[k])([float("nan")] * len(sim.times), maxlen=300)
        chart = ErrorChart()
        chart.set_data(sim, {"LM"})
        painted(chart)
        chart.set_data(sim, set())
        painted(chart)

    def test_end_labels_stay_clear_of_the_lin_log_switch(self):
        sim, _, _ = make_sim(0)
        n = 30
        sim.times = type(sim.times)([0.1 * k for k in range(1, n + 1)], maxlen=300)
        for k in theme.ALGO_ORDER:                       # every line ends at the bottom ...
            sim.errors[k] = type(sim.errors[k])([0.5] * n, maxlen=300)
        sim.errors["CEP"] = type(sim.errors["CEP"])([0.5] * (n - 1) + [500.0], maxlen=300)   # ... but one
        chart = ErrorChart()
        chart.set_data(sim, set(theme.ALGO_ORDER))
        for h in (200, 140):
            painted(chart, 600, h)
            plot = chart._plot_rect()
            t, series, t0, span = chart._visible()
            lo, hi, _ = chart._y_range(np.concatenate(list(series.values())))
            labels = chart._end_labels(plot, series, lambda v: plot.bottom() - (v - lo) / (hi - lo) * plot.height())
            ys = sorted(e[1] for e in labels)
            self.assertGreaterEqual(ys[0], plot.top() + 4 - 1e-9, h)      # below the switch's row
            self.assertLessEqual(ys[-1], plot.bottom() + 4 + 1e-9, h)
            self.assertGreaterEqual(min(np.diff(ys)), ErrorChart.MIN_LABEL_GAP - 1e-9, h)
            self.assertIn("CEP ↑", [e[2] for e in labels])

    def test_scope_draws_noise_interference_and_unknown_statuses(self):
        sim, _, frame = make_sim(1, noise_rate_hz=400.0)
        noisy = [r for r in frame.readings if r.status == "noise"]
        self.assertTrue(noisy)
        self.assertTrue(all(r.detected is None and r.echo_ms is not None for r in noisy))
        scope = EchoScope()
        scope.set_frame(frame, sim.env)
        painted(scope, 900, 300)
        self.assertTrue(any("ambient ultrasonic noise" in tip for _, tip in scope.hits))
        sim, _, frame = make_sim(1)
        i, r = next((i, r) for i, r in enumerate(frame.readings) if r.detected is not None)
        copy = dataclasses.replace(r.detected, cycle=(r.detected.cycle or 0) + 2)   # as interference reports it
        late = r.echo_ms + 2 * 0.025
        readings = list(frame.readings)
        readings[i] = dataclasses.replace(r, status="fade", detected=copy, echo_ms=late)
        readings[(i + 1) % len(readings)] = dataclasses.replace(readings[(i + 1) % len(readings)],
                                                                status="no such status")
        scope.set_frame(dataclasses.replace(frame, readings=readings), sim.env)
        painted(scope, 900, 300)
        self.assertTrue(any("interference" in tip for _, tip in scope.hits))
        panel = ReceiverPanel(sim, SimRunner(sim))
        panel.refresh(dataclasses.replace(frame, readings=readings))
        self.assertTrue(panel.table.item(2 * i, 0).text().startswith(f"R{i}  fade"))

    def test_scope_shows_why_r1_reads_long(self):
        sim, _, frame = make_sim()
        self.assertEqual(frame.readings[1].status, "echo")
        scope = EchoScope()
        scope.set_frame(frame, sim.env)
        painted(scope, 1000, 320)
        tips = [tip for _, tip in scope.hits]
        self.assertTrue(any("reflection" in t and "dropped the echo line" in t for t in tips))
        self.assertTrue(any("timed out" in t for t in tips))           # R0 lost its signal
        self.assertTrue(any("true distance" in t for t in tips))
        rect, tip = next((r, t) for r, t in scope.hits if "dropped the echo line" in t)
        self.assertEqual(scope.tooltip_at(rect.center()), tip)
        self.assertEqual(scope.tooltip_at(QPointF(-50, -50)), "")

    def test_scope_shows_crosstalk_switched_off_receivers_and_replays(self):
        sim, _, _ = make_sim(crosstalk=True)
        sim.set_receiver_state(4, "off")
        frame = sim.step()
        scope = EchoScope()
        scope.set_frame(frame, sim.env)
        painted(scope, 460, 240)
        self.assertTrue(any("own burst" in tip for _, tip in scope.hits))
        scope.set_frame(dataclasses.replace(frame, readings=[]), sim.env)   # a replayed tick log
        painted(scope)
        self.assertEqual(scope.hits, [])
        scope.set_frame(None, None)
        painted(scope)


    def test_scope_tells_injected_faults_and_radio_loss_apart(self):
        sim, _, frame = make_sim(faults=Faults(noise_cm=0.0, dropout_rate=1.0))
        scope = EchoScope()
        scope.set_frame(frame, sim.env)
        painted(scope, 900, 300)
        tips = [tip for _, tip in scope.hits]
        self.assertTrue(all(r.status == "dropout" for r in frame.readings))
        self.assertFalse(any("dropped the echo line" in t for t in tips))   # the line never fell
        self.assertTrue(any("injected fault (dropout)" in t for t in tips))
        sim, _, frame = make_sim(radio_loss=1.0)
        scope.set_frame(frame, sim.env)
        painted(scope, 900, 300)
        self.assertTrue(any("missed its radio trigger" in tip for _, tip in scope.hits))

    def test_scope_with_every_receiver_off_and_tiny(self):
        sim, _, _ = make_sim()
        for i in range(len(sim.receivers)):
            sim.set_receiver_state(i, "off")
        scope = EchoScope()
        scope.set_frame(sim.step(), sim.env)
        painted(scope, 320, 170)
        self.assertEqual(sum("switched off" in tip for _, tip in scope.hits), len(sim.receivers))


class PanelTest(unittest.TestCase):
    def test_every_environment_field_has_a_control(self):
        attrs = [f.attr for _, group in ENV_GROUPS for f in group]
        self.assertEqual(sorted(attrs), sorted(f.name for f in dataclasses.fields(Environment)))

    def test_every_control_has_a_tooltip(self):
        sim, runner, frame = make_sim()
        for cls in (ScenarioPanel, ReceiverPanel, AlgorithmPanel, EnvironmentPanel, FaultsPanel):
            panel = cls(sim, runner)
            panel.refresh(frame)
            for kind in (QAbstractButton, QAbstractSlider, QAbstractSpinBox, QComboBox):
                for w in panel.findChildren(kind):
                    if isinstance(w, QScrollBar) or not w.isVisibleTo(panel):
                        continue
                    self.assertTrue(w.toolTip(), f"{cls.__name__}: {type(w).__name__} has no tooltip")

    def test_environment_controls_swap_in_a_new_environment(self):
        sim, runner, _ = make_sim(1)
        panel = EnvironmentPanel(sim, runner)
        editors = {a: w for form in panel.forms for a, w in form.editors.items()}
        old = sim.env
        editors["temperature_c"].spin.setValue(30.0)
        self.assertIsNot(sim.env, old)
        self.assertEqual((old.temperature_c, sim.env.temperature_c), (20.0, 30.0))
        self.assertIn("ranges read -1.", panel.derived.text())        # warmer air: ranges read short
        editors["reflections"].setChecked(False)
        self.assertFalse(sim.env.reflections)
        self.assertFalse(editors["bounces"].isEnabled())
        self.assertFalse(editors["tx_fixed_deg"].isEnabled())
        editors["tx_mode"].setCurrentIndex(editors["tx_mode"].findData("fixed"))
        self.assertEqual(sim.env.tx_mode, "fixed")
        self.assertTrue(editors["tx_fixed_deg"].isEnabled())
        editors["radio_loss"].spin.setValue(5.0)                     # shown in %, stored as a rate
        self.assertAlmostEqual(sim.env.radio_loss, 0.05)
        self.assertEqual(panel.preset.currentText(), "Custom")

    def test_presets_replace_the_environment_and_the_controls_follow(self):
        sim, runner, _ = make_sim(1)
        panel = EnvironmentPanel(sim, runner)
        panel.preset.activated.emit(panel.preset.findText("Ideal"))
        self.assertEqual(sim.env, Environment(**PRESETS["Ideal"]))
        editors = {a: w for form in panel.forms for a, w in form.editors.items()}
        self.assertFalse(editors["threshold"].isChecked())
        self.assertEqual(panel.preset.currentText(), "Ideal")
        sim.env = sim.env.copy(snr_db=12.0)                            # changed elsewhere
        panel.refresh(None)
        self.assertEqual(editors["snr_db"].value(), 12.0)

    def test_faults_and_library_filter(self):
        sim, runner, frame = make_sim(1)
        panel = FaultsPanel(sim, runner)
        panel.refresh(frame)
        self.assertAlmostEqual(panel.filter.editors["max_distance"].value(), sim.solvers.top_band,
                               delta=2.5)
        panel.faults.editors["dropout_rate"].spin.setValue(10.0)
        self.assertAlmostEqual(sim.faults.dropout_rate, 0.10)
        panel.filter.editors["max_distance"].spin.setValue(300.0)
        panel.filter.editors["max_differential"].spin.setValue(20.0)
        self.assertEqual((sim.solvers.lib.max_distance, sim.solvers.lib.max_differential), (300.0, 20.0))

    def test_receiver_panel_shows_a_replayed_row_of_the_wrong_length(self):
        sim, runner, frame = make_sim()
        panel = ReceiverPanel(sim, runner)
        for ticks, r5 in (([1500, 1400, 1300], "0"), (list(range(1, 9)), "6")):
            sim.last = dataclasses.replace(frame, ticks=ticks, readings=[], true=None)
            panel.refresh(sim.last)
            self.assertEqual(panel.table.item(2 * 5 + 1, 3).text(), r5)    # R5: "0" when the row is short
            panel._restyle()                                               # a palette change redraws it

    def test_panels_report_edits_and_post_them_without_waiting(self):
        sim, runner, frame = make_sim(1)
        receivers, scenario = ReceiverPanel(sim, runner), ScenarioPanel(sim, runner)
        seen = []
        for panel in (receivers, scenario):
            panel.edited.connect(lambda p=panel: seen.append(type(p).__name__))
        held, done = threading.Event(), threading.Event()
        def long_step():
            with runner.lock:
                held.set()
                done.wait(5)
        worker = threading.Thread(target=long_step)
        worker.start()
        try:
            held.wait(5)
            start = time.perf_counter()
            receivers.survey.editors["sigma"].spin.setValue(4.0)
            mode = scenario.form.editors["mode"]
            mode.setCurrentIndex(mode.findData("lawnmower"))
            self.assertLess(time.perf_counter() - start, 0.5)       # the GUI did not wait for the step
            np.testing.assert_allclose(sim.coords, sim.believed)      # not applied mid-step
        finally:
            done.set()
            worker.join()
        runner.call(lambda: None)                                     # the runner's next step applies them
        self.assertGreater(np.abs(sim.coords - sim.believed).max(), 0.0)
        self.assertEqual(sim.motion.mode, "lawnmower")
        self.assertEqual(seen, ["ReceiverPanel", "ScenarioPanel"])

    def test_receiver_panel_switches_states_and_shows_readings(self):
        sim, runner, frame = make_sim()
        panel = ReceiverPanel(sim, runner)
        panel.refresh(frame)
        panel.groups[2].button(1).click()
        self.assertEqual(sim.receivers[2].state, "blocked")
        sim.set_receiver_state(3, "off")                               # changed elsewhere
        panel.refresh(frame)
        self.assertEqual(panel.groups[3].checkedId(), 2)
        t = panel.table
        self.assertEqual(t.item(3, 3).text(), str(frame.ticks[1]))          # R1's numbers ...
        self.assertTrue(t.item(2, 0).text().startswith("R1  echo"))         # ... under its status
        panel.survey.editors["sigma"].spin.setValue(5.0)
        self.assertGreater(np.abs(sim.coords - sim.believed).max(), 0.0)
        panel.survey.editors["sigma"].spin.setValue(0.0)
        np.testing.assert_allclose(sim.coords, sim.believed)

    def test_algorithm_panel(self):
        sim, runner, frame = make_sim(5)
        panel = AlgorithmPanel(sim, runner)
        panel.refresh(frame)
        spy = QSignalSpy(panel.shownChanged)
        panel.checks["EKF"].setChecked(False)
        self.assertEqual(spy[-1][0], set(theme.ALGO_ORDER) - {"EKF"})
        self.assertEqual(panel.shown, set(theme.ALGO_ORDER) - {"EKF"})
        panel.settings.editors["robust_scale_cm"].spin.setValue(9.0)
        self.assertEqual(sim.solvers.settings.robust_scale_cm, 9.0)
        panel.settings.editors["filter_mode"].setCurrentIndex(1)
        self.assertEqual(sim.solvers.settings.filter_mode, "bounds")
        subsets = panel.settings.editors["cep_subsets"]
        self.assertEqual(subsets.currentData(), "control")            # as UltraGPS-Control calls it
        subsets.setCurrentIndex(subsets.findData("all"))
        self.assertEqual(sim.solvers.settings.cep_subsets, "all")
        next(b for b in panel.findChildren(QAbstractButton) if b.text() == "Reset stats").click()
        self.assertEqual((sim.frames, len(sim.times)), (0, 0))

    def test_cep_subsets_tip_says_which_subsets_the_library_tries(self):
        sim = ideal_sim(noise=1.0)
        sizes, real = [], sim.solvers.lib._compute_cep
        sim.solvers.lib._compute_cep = lambda d, idx: sizes.append(len(idx)) or real(d, idx)
        sim.place((30, 50))
        sim.step()
        tip = next(f.tip for f in SOLVER_FIELDS if f.attr == "cep_subsets")
        self.assertEqual(sorted(set(sizes)), [3, 6])                  # all six, then triples only
        self.assertIn(f"all six plus {sizes.count(3)} of the 20 triples", tip)
        self.assertIn("no 4- or 5-receiver subset", tip)
        self.assertIn(f"{42 - len(sizes)} of the 42", tip)
        self.assertNotIn("smaller subsets", tip)

    def test_scenario_panel(self):
        sim, runner, frame = make_sim(1)
        panel = ScenarioPanel(sim, runner)
        mode = panel.form.editors["mode"]
        mode.setCurrentIndex(mode.findData("circle"))
        self.assertEqual(sim.motion.mode, "circle")
        self.assertTrue(panel.form.editors["radius"].isEnabled())
        panel.form.editors["dt"].spin.setValue(0.05)
        scale = panel.form.editors["time_scale"]
        scale.setCurrentIndex(scale.findData(4.0))
        self.assertEqual((runner.dt, runner.speed), (0.05, 4.0))
        panel.refresh(frame)
        self.assertEqual(panel.values["x"].text(), f"{frame.true[0]:.1f} cm")
        sim.place((0, 0))                                             # a click on the arena
        panel.refresh(frame)
        self.assertEqual(mode.currentData(), "target")
        self.assertFalse(panel.form.editors["radius"].isEnabled())

    def test_tables_follow_a_palette_switch_without_a_new_frame(self):
        sim, runner, frame = make_sim()
        panels = [ReceiverPanel(sim, runner), AlgorithmPanel(sim, runner)]
        for panel in panels:
            panel.refresh(frame)
        try:
            theme.apply(app, "paper")
            app.processEvents()                                         # the deferred re-colour
            heading = panels[0].table.item(2, 0)                        # R1, status "echo"
            self.assertEqual(heading.foreground().color().name().lower(),
                             theme.current.status["echo"].lower())
            rms = panels[1].table.item(0, 2)
            self.assertEqual(rms.foreground().color().name().lower(), theme.current.text.lower())
        finally:
            theme.apply(app, "vector")

    def test_mouse_wheel_scrolls_past_controls_without_focus(self):
        sim, runner, _ = make_sim(1)
        panel = EnvironmentPanel(sim, runner)
        slider = {a: w for form in panel.forms for a, w in form.editors.items()}["snr_db"].slider
        before = slider.value()
        wheel = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, -120),
                            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                            Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(slider, wheel)
        self.assertEqual(slider.value(), before)
        self.assertEqual(sim.env.snr_db, 30.0)

    def test_refresh_never_waits_for_the_runner(self):
        sim, runner, frame = make_sim(5)
        panels = [cls(sim, runner) for cls in
                  (ScenarioPanel, ReceiverPanel, AlgorithmPanel, EnvironmentPanel, FaultsPanel)]
        held, done = threading.Event(), threading.Event()
        def hold_lock():                                                # a long simulation step
            with runner.lock:
                held.set()
                done.wait(5)
        worker = threading.Thread(target=hold_lock)
        worker.start()
        try:
            held.wait(5)
            start = time.perf_counter()
            for panel in panels:
                panel.refresh(frame)
            self.assertLess(time.perf_counter() - start, 1.0)
        finally:
            done.set()
            worker.join()

    def test_panels_show_a_replayed_frame_without_readings_or_truth(self):
        sim, runner, frame = make_sim()
        replay = dataclasses.replace(frame, readings=[], true=None,
                                     errors={k: float("nan") for k in frame.errors})
        receivers, scenario = ReceiverPanel(sim, runner), ScenarioPanel(sim, runner)
        receivers.refresh(replay)
        scenario.refresh(replay)
        AlgorithmPanel(sim, runner).refresh(replay)
        self.assertTrue(receivers.table.item(0, 0).text().startswith("R0  log"))
        self.assertEqual(receivers.table.item(1, 5).text(), "—")         # no true distance
        self.assertEqual(scenario.values["x"].text(), "—")

    def test_everything_redraws_in_the_paper_palette(self):
        sim, runner, frame = make_sim()
        try:
            theme.apply(app, "paper")
            for cls in (ScenarioPanel, ReceiverPanel, AlgorithmPanel, EnvironmentPanel, FaultsPanel):
                panel = cls(sim, runner)
                panel.refresh(frame)
                painted(panel, 330, 600)
            scope = EchoScope()
            scope.set_frame(frame, sim.env)
            painted(scope)
        finally:
            theme.apply(app, "vector")


if __name__ == "__main__":
    unittest.main()
