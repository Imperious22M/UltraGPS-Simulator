"""
UltraGPS positioning simulator — interactive window.

    python3 app.py [--config config/config.toml] [--sensors config/sensors.toml]
                   [--max-differential 50] [--max-distance 400]

Click the arena to send the transmitter somewhere, toggle receivers on and off,
and drag the sliders to add noise and faults.  Every algorithm in the real
positioning library solves each frame side by side, with its error against the
true position shown live.  "Cal points" picks how many calibration points (2-4)
to use, "Search points" finds the best ones, "Manual selection" lets you click
them on the arena, "Calibrate" runs the calibration library's routine on the
simulated sensors at those points, and "Modify calibration values" opens a
window to edit each receiver's slope and intercept.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox

import cal_points
from calibration import Calibrator, read_offsets
from simulation import Simulation
from solvers import ALGORITHMS, DEFAULT_CONFIG, DEFAULT_SENSORS

DT          = 0.1   # default s per frame — the base station's default 10 Hz
COLORS      = {"OLS": "tab:orange", "LM": "tab:blue", "CEP": "tab:green", "CEP4": "tab:purple"}
LABELS      = {"OLS": "Linear LS (OLS)", "LM": "OLS + Levenberg-Marquardt", "CEP": "CEP best subset",
               "CEP4": "CEP, 4+ receivers, no box"}
STATUS_FILL = {"ok": "tab:green", "off": "lightgray", "dropout": "tab:orange",
               "range": "tab:orange", "timeout": "tab:brown", "spike": "tab:red"}
PATH_RADIUS = 100.0


class CalibrationEditor:
    """Separate window: one row per receiver with editable slope and intercept.
    "Save & apply" writes them to config.toml through the calibration library
    and reloads the positioning library."""

    def __init__(self, calibrator: Calibrator):
        self.cal    = calibrator
        self.fig    = plt.figure("Modify calibration values", figsize=(7, 5.5))
        n           = len(calibrator.sim.coords)
        self.fig.text(0.05, 0.93, "distance_cm = tick × slope + intercept", weight="bold")
        for x, title in ((0.18, "slope"), (0.43, "intercept (cm)"), (0.70, "sensor truth (sensors.toml)")):
            self.fig.text(x, 0.86, title, fontsize=9)
        self.slope, self.icpt = [], []
        sensor = calibrator.sim.sensor
        for i in range(n):
            y = 0.78 - i * 0.09
            self.fig.text(0.05, y + 0.01, f"R{i}", weight="bold")
            self.slope.append(TextBox(self.fig.add_axes([0.18, y, 0.20, 0.055]), ""))
            self.icpt.append(TextBox(self.fig.add_axes([0.43, y, 0.20, 0.055]), ""))
            self.fig.text(0.70, y + 0.015, f"{sensor.slopes[i]:.4f}   {sensor.intercepts[i]:8.2f}",
                          family="monospace", fontsize=9, color="gray")
        self.b_save   = Button(self.fig.add_axes([0.18, 0.10, 0.20, 0.06]), "Save & apply")
        self.b_reload = Button(self.fig.add_axes([0.43, 0.10, 0.20, 0.06]), "Reload from file")
        self.b_save.on_clicked(self._on_save)
        self.b_reload.on_clicked(lambda _: self.load())
        self.status = self.fig.text(0.05, 0.03, "", fontsize=9)
        self.seen   = -1
        self.refresh()
        self.fig.show()

    def refresh(self) -> None:
        """Reload the fields if the offsets changed since this window last saw them."""
        if self.seen != self.cal.version:
            self.load()

    def load(self) -> None:
        offsets = read_offsets(self.cal.config_path)
        for i, (sb, ib) in enumerate(zip(self.slope, self.icpt)):
            sb.set_val(f"{offsets[i]['slope']:.6f}")
            ib.set_val(f"{offsets[i]['intercept']:.4f}")
        self.seen = self.cal.version
        self.status.set_text(f"Loaded from {self.cal.config_path}")
        self.fig.canvas.draw_idle()

    def _on_save(self, _) -> None:
        try:
            offsets = {i: {"slope": float(sb.text), "intercept": float(ib.text)}
                       for i, (sb, ib) in enumerate(zip(self.slope, self.icpt))}
            self.cal.save(offsets)
            self.seen = self.cal.version
            self.status.set_text("Saved to config.toml and applied")
        except (ValueError, RuntimeError) as exc:
            self.status.set_text(f"Not saved: {exc}")
        self.fig.canvas.draw_idle()


class App:
    def __init__(self, sim: Simulation, config_path: str = DEFAULT_CONFIG):
        self.sim     = sim
        self.calib   = Calibrator(sim, config_path)
        self.editor  = None
        self.n_points = len(self.calib.points)   # how many points Search / Manual pick
        self.picking  = None                      # points clicked so far in manual selection
        self.paused  = False
        self.on_path = False
        self.angle   = 0.0
        self.dt      = DT
        self.shown   = {a: True for a in ALGORITHMS}

        self.fig = plt.figure("UltraGPS positioning simulator", figsize=(13, 8))
        self._build_arena()
        self._build_controls()
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.anim = FuncAnimation(self.fig, self.update, interval=int(DT * 1000),
                                  cache_frame_data=False)

    # ─── Layout ──────────────────────────────────────────────────────────────

    def _build_arena(self) -> None:
        c  = self.sim.coords
        ax = self.ax = self.fig.add_axes([0.05, 0.13, 0.55, 0.82])
        ax.set_aspect("equal")
        ax.set_xlim(c[:, 0].min() - 60, c[:, 0].max() + 60)
        ax.set_ylim(c[:, 1].min() - 40, c[:, 1].max() + 40)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_title("Click to move the transmitter")
        ax.grid(alpha=0.3)

        self.rings = [plt.Circle(xy, 0, fill=False, lw=0.8, alpha=0.5) for xy in c]
        for r in self.rings:
            ax.add_patch(r)
        self.towers = ax.scatter(c[:, 0], c[:, 1], s=180, marker="^",
                                 c=["tab:green"] * len(c), edgecolors="k", zorder=5)
        for i, (x, y) in enumerate(c):
            ax.annotate(f"R{i}", (x, y), xytext=(0, 12), textcoords="offset points",
                        ha="center", fontsize=9)

        self.cal_artists = []
        self._draw_cal_points(self.calib.points)

        self.trail = {a: ax.plot([], [], "-", color=COLORS[a], alpha=0.4, lw=1)[0]
                      for a in ALGORITHMS}
        self.est   = {a: ax.plot([], [], "o", color=COLORS[a], ms=9, label=LABELS[a])[0]
                      for a in ALGORITHMS}
        self.target_m, = ax.plot([], [], "o", mfc="none", mec="gray", ms=14, label="Target")
        self.true_m,   = ax.plot([], [], "kx", ms=12, mew=2.5, label="True position", zorder=6)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=8)

    def _build_controls(self) -> None:
        f, n = self.fig, len(self.sim.coords)

        f.text(0.64, 0.95, "Receivers", weight="bold")
        self.tower_cb = CheckButtons(f.add_axes([0.64, 0.74, 0.14, 0.20]),
                                     [f"R{i}" for i in range(n)], [True] * n)
        self.tower_cb.on_clicked(self._on_tower)

        f.text(0.80, 0.95, "Algorithms", weight="bold")
        self.algo_cb = CheckButtons(f.add_axes([0.80, 0.78, 0.18, 0.16]),
                                    list(ALGORITHMS), [True] * len(ALGORITHMS))
        self.algo_cb.on_clicked(self._on_algo)

        fl = self.sim.faults
        def slider(y, label, lo, hi, init, fmt):
            s = Slider(f.add_axes([0.72, y, 0.20, 0.025]), label, lo, hi, valinit=init, valfmt=fmt)
            s.on_changed(lambda _: self._on_slider())
            return s
        lib = self.sim.solvers.lib
        top = lib.max_distance if lib.max_distance is not None else lib._max_receiver_dist
        rows = iter(0.70 - 0.028 * i for i in range(10))
        self.s_noise   = slider(next(rows), "Noise σ (cm)",         0, 20,   fl.noise_cm,           "%.1f")
        self.s_drop    = slider(next(rows), "Dropouts (%)",         0, 50,   fl.dropout_rate * 100, "%.0f")
        self.s_timeout = slider(next(rows), "Lost signal (%)",      0, 50,   fl.timeout_rate * 100, "%.0f")
        self.s_spike   = slider(next(rows), "Bad values (%)",       0, 50,   fl.spike_rate * 100,   "%.0f")
        self.s_spkpct  = slider(next(rows), "Bad value range (±%)", 0, 100,  fl.spike_pct,          "%.0f")
        self.s_temp    = slider(next(rows), "Air temp (°C)",        0, 40,   fl.temperature_c,      "%.0f")
        self.s_speed   = slider(next(rows), "Speed (cm/s)",         5, 200,  self.sim.speed,        "%.0f")
        self.s_top     = slider(next(rows), "Top-band reject (cm)", 50, 1500, top,                  "%.0f")
        self.s_dt      = slider(next(rows), "Timestep (s)",      0.02, 1.0,  self.dt,               "%.2f")

        y = 0.425
        self.b_pause = Button(f.add_axes([0.64, y, 0.08, 0.04]), "Pause")
        self.b_step  = Button(f.add_axes([0.73, y, 0.08, 0.04]), "Step")
        self.b_path  = Button(f.add_axes([0.82, y, 0.08, 0.04]), "Circle path")
        self.b_reset = Button(f.add_axes([0.91, y, 0.08, 0.04]), "Reset stats")
        self.b_pause.on_clicked(self._on_pause)
        self.b_step.on_clicked(self._on_step)
        self.b_path.on_clicked(self._on_path)
        self.b_reset.on_clicked(lambda _: self.sim.reset())

        y = 0.38
        self.b_npts   = Button(f.add_axes([0.64, y, 0.10, 0.04]), f"Cal points: {self.n_points}")
        self.b_search = Button(f.add_axes([0.75, y, 0.11, 0.04]), "Search points")
        self.b_manual = Button(f.add_axes([0.87, y, 0.12, 0.04]), "Manual selection")
        self.b_npts.on_clicked(self._on_npoints)
        self.b_search.on_clicked(self._on_search)
        self.b_manual.on_clicked(self._on_manual)

        y = 0.335
        self.b_calib = Button(f.add_axes([0.64, y, 0.17, 0.04]), "Calibrate")
        self.b_edit  = Button(f.add_axes([0.82, y, 0.17, 0.04]), "Modify calibration values")
        self.b_calib.on_clicked(self._on_calibrate)
        self.b_edit.on_clicked(self._on_edit)

        self.info = f.text(0.64, 0.32, "", family="monospace", fontsize=8, va="top", linespacing=1.05)

    def _draw_cal_points(self, points, pending: bool = False) -> None:
        """Show calibration points P1..Pn on the arena (orange while picking)."""
        for a in self.cal_artists:
            a.remove()
        color = "tab:orange" if pending else "dimgray"
        self.cal_artists = []
        for k, (x, y) in enumerate(points, start=1):
            self.cal_artists += self.ax.plot(x, y, "s", mfc="none", mec=color, ms=10, mew=1.5)
            self.cal_artists.append(self.ax.annotate(
                f"P{k}", (x, y), xytext=(7, -12), textcoords="offset points", fontsize=8, color=color))

    # ─── Callbacks ───────────────────────────────────────────────────────────

    def _on_click(self, event) -> None:
        if event.inaxes is self.ax and event.button == 1 and self.picking is not None:
            self.picking.append((event.xdata, event.ydata))
            if len(self.picking) < self.n_points:
                self._draw_cal_points(self.picking, pending=True)
                self.ax.set_title(f"Manual selection: click calibration point "
                                  f"{len(self.picking) + 1} of {self.n_points}")
            else:
                self._use_points(self.picking, "Manual selection")
                self.picking = None
                self.b_manual.label.set_text("Manual selection")
            self.fig.canvas.draw_idle()
        elif event.inaxes is self.ax and event.button == 1:
            self.on_path = False
            self.sim.target = np.array([event.xdata, event.ydata])

    def _on_tower(self, label) -> None:
        self.sim.enabled[int(label[1:])] ^= True

    def _on_algo(self, label) -> None:
        self.shown[label] = not self.shown[label]

    def _on_slider(self) -> None:
        fl = self.sim.faults
        fl.noise_cm      = self.s_noise.val
        fl.dropout_rate  = self.s_drop.val / 100
        fl.timeout_rate  = self.s_timeout.val / 100
        fl.spike_rate    = self.s_spike.val / 100
        fl.spike_pct     = self.s_spkpct.val
        fl.temperature_c = self.s_temp.val
        self.sim.speed   = self.s_speed.val
        self.sim.solvers.lib.max_distance = self.s_top.val
        self.dt = self.s_dt.val
        if hasattr(self, "anim") and self.anim.event_source is not None:
            self.anim.event_source.interval = int(self.dt * 1000)

    def _on_pause(self, _) -> None:
        self.paused = not self.paused
        self.b_pause.label.set_text("Resume" if self.paused else "Pause")

    def _on_step(self, _) -> None:
        if not self.paused:
            self._on_pause(None)
        self.advance()
        self.fig.canvas.draw_idle()

    def _on_npoints(self, _) -> None:
        self.n_points = 2 if self.n_points == 4 else self.n_points + 1
        self.b_npts.label.set_text(f"Cal points: {self.n_points}")
        self.ax.set_title(f"{self.n_points} calibration points: use Search points or "
                          f"Manual selection to place them")
        self.fig.canvas.draw_idle()

    def _on_search(self, _) -> None:
        if self.calib.busy:
            return
        self.ax.set_title(f"Searching for the best {self.n_points} calibration points...")
        self.fig.canvas.draw()
        best = cal_points.search(self.sim.coords, self.n_points)[self.n_points]
        self._use_points(best, f"Search: best {self.n_points} points")

    def _on_manual(self, _) -> None:
        if self.calib.busy:
            return
        if self.picking is None:
            self.picking = []
            self.b_manual.label.set_text("Cancel selection")
            self.ax.set_title(f"Manual selection: click calibration point 1 of {self.n_points}")
        else:
            self.picking = None
            self.b_manual.label.set_text("Manual selection")
            self._draw_cal_points(self.calib.points)
            self.ax.set_title("Click to move the transmitter")
        self.fig.canvas.draw_idle()

    def _use_points(self, points, how: str) -> None:
        """Save the calibration points to config.toml and show them."""
        self.calib.set_points(points)
        self._draw_cal_points(self.calib.points)
        score = cal_points.arena_score(np.asarray(points, dtype=float), self.sim.coords)
        self.ax.set_title(f"{how}, saved to config.toml (score {score:.2f}, lower is better). "
                          f"Press Calibrate to use them.", fontsize=9)
        self.fig.canvas.draw_idle()

    def _on_calibrate(self, _) -> None:
        self.on_path = False
        self.calib.start()

    def _on_edit(self, _) -> None:
        if self.editor is None or not plt.fignum_exists(self.editor.fig.number):
            self.editor = CalibrationEditor(self.calib)
        else:
            self.editor.load()

    def _on_path(self, _) -> None:
        self.on_path = not self.on_path

    # ─── Frame ───────────────────────────────────────────────────────────────

    def update(self, _=None) -> None:
        if self.editor is not None and plt.fignum_exists(self.editor.fig.number):
            self.editor.refresh()                         # pick up a new calibration
        if self.calib.busy:                               # sensors belong to the calibration
            self.true_m.set_data([self.sim.true[0]], [self.sim.true[1]])
            self.target_m.set_data([self.sim.true[0]], [self.sim.true[1]])
            self.info.set_text(f"calibration  {self._calib_text()}")
        elif not self.paused:
            self.advance()

    def advance(self) -> None:
        """Run one simulation frame of ``self.dt`` seconds and redraw."""
        if self.on_path:
            self.angle += self.sim.speed * self.dt / PATH_RADIUS
            self.sim.target = PATH_RADIUS * np.array([np.cos(self.angle), np.sin(self.angle)])

        f = self.sim.step(self.dt)

        self.true_m.set_data([f["true"][0]], [f["true"][1]])
        self.target_m.set_data([self.sim.target[0]], [self.sim.target[1]])

        # Receivers: colour = what the sensor did, thick edge = used by CEP
        self.towers.set_facecolors([STATUS_FILL[s] for s in f["status"]])
        self.towers.set_linewidths([3 if i in f["cep_subset"] else 1
                                    for i in range(len(f["status"]))])

        # Measured-range circles: solid if the library kept the reading
        for i, ring in enumerate(self.rings):
            ring.set_radius(max(f["distances"][i], 0))
            kept = i in f["sane"]
            ring.set_linestyle("-" if kept else ":")
            ring.set_edgecolor("tab:gray" if kept else "tab:red")
            ring.set_visible(f["status"][i] != "off")

        for a in ALGORITHMS:
            pos   = f["positions"][a]
            trail = np.array(self.sim.trails[a]) if self.sim.trails[a] else np.empty((0, 2))
            self.est[a].set_visible(self.shown[a] and pos is not None)
            self.trail[a].set_visible(self.shown[a])
            if pos is not None:
                self.est[a].set_data([pos[0]], [pos[1]])
            self.trail[a].set_data(trail[:, 0], trail[:, 1])

        self.info.set_text(self._info_text(f))

    def _calib_text(self) -> str:
        c = self.calib
        if c.busy:
            k = c.state[-1]
            return f"{c.state} at P{k}, {c.reads} reads"
        return {"idle": "offsets from config.toml", "done": "done, saved to config.toml",
                "error": f"error: {c.error}"}[c.state]

    def _info_text(self, f: dict) -> str:
        lines = [f"{'error (cm)':<10} {'now':>7} {'RMS':>7} {'fails':>6}"]
        for a in ALGORITHMS:
            now, rms, fails = self.sim.stats(a)
            lines.append(f"{a:<10} {now:7.1f} {rms:7.1f} {fails:6d}")
        lines += [
            "",
            f"frames     {self.sim.frames}   t = {self.sim.time:.1f} s",
            f"calib.     {self._calib_text()}",
            f"kept       {', '.join(f'R{i}' for i in f['sane']) or 'none'}",
            f"CEP used   {', '.join(f'R{i}' for i in f['cep_subset']) or 'none, holding last fix'}",
            f"CEP4 used  {', '.join(f'R{i}' for i in f['cep4_subset']) or 'none, holding last fix'}",
            "",
            "rx  status   ticks   dist(cm)",
        ]
        for i, (s, t, d) in enumerate(zip(f["status"], f["ticks"], f["distances"])):
            lines.append(f"R{i}  {s:<8} {t:6d}  {d:8.1f}")
        return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default=DEFAULT_CONFIG, help="config.toml with receivers + calibration")
    p.add_argument("--sensors", default=DEFAULT_SENSORS,
                   help="sensors.toml with the simulated hardware's true tick response")
    p.add_argument("--max-differential", type=float, default=50.0,
                   help="library jump filter, cm per frame (default 50)")
    p.add_argument("--max-distance", type=float, default=None,
                   help="library top-band reject, cm (default: widest receiver spacing)")
    p.add_argument("--seed", type=int, default=None, help="random seed for repeatable runs")
    args = p.parse_args()

    sim = Simulation(args.config, args.sensors, args.max_differential, args.max_distance, seed=args.seed)
    app = App(sim, args.config)  # noqa: F841 (keeps the animation alive)
    plt.show()


if __name__ == "__main__":
    main()
