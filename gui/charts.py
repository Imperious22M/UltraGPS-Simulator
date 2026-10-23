"""
Charts — two time plots drawn in the vector style.

``ErrorChart`` is a strip chart of how far each shown algorithm's estimate is
from the true position (cm) over the last 30 simulated seconds.

``EchoScope`` is a logic analyser of one ping.  One row per receiver: every
sound arrival (straight, bounced off a wall or obstacle, bent round a corner,
or another receiver's own burst) is a stem as tall as its strength over the
detection threshold, and under it the receiver's echo line: high once the
receiver listens, dropping at the first arrival strong enough to trip its
comparator, or at the HC-SR04's timeout when none is.  The moment the line
drops is the reading, so the scope shows *why* a receiver read what it did.
"""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QPainter, QPainterPath
from PyQt6.QtWidgets import QMenu, QToolTip, QWidget

from engine import acoustics as ac
from engine.hardware import STATUS, Hardware
from gui import draw, theme
from gui.theme import color


def nice_step(span: float, count: int = 5) -> float:
    """A 1-2-5 step that cuts ``span`` into about ``count`` intervals."""
    raw = max(span, 1e-9) / max(count, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 5.0):
        if raw <= m * mag * 1.0001:
            return m * mag
    return 10.0 * mag


def fmt_cm(v: float) -> str:
    """A distance in cm in at most five characters ("—" for no value)."""
    if v is None or not math.isfinite(v):
        return "—"
    a = abs(v)
    if a < 100:
        return f"{v:.2f}" if a < 10 else f"{v:.1f}"
    return f"{v:.0f}" if a < 1e4 else f"{v / 1e3:.0f}k" if a < 1e6 else f"{v / 1e6:.1f}M"


def _message(p: QPainter, rect: QRectF, text: str) -> None:
    draw.label(p, rect.center(), text, theme.current.text_muted, 9, "center", halo=False)


def _tooltip_event(widget: QWidget, event, text: str) -> bool:
    """Show ``text`` (or hide the tooltip if empty) for a ``QEvent.ToolTip``."""
    if text:
        QToolTip.showText(event.globalPos(), text, widget)
    else:
        QToolTip.hideText()
        event.ignore()
    return True


# ─── Error chart ─────────────────────────────────────────────────────────────

class ErrorChart(QWidget):
    """Strip chart of each shown algorithm's error (cm) over the last ``window_s`` seconds."""

    LEFT, RIGHT, TOP, BOTTOM = 44, 70, 22, 20     # margins round the plot, px
    LOG_TOP = 1e4                                 # cm: the log axis stops here (100 m is just "lost")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_s  = 30.0
        self.log_scale = False
        self._t        = np.empty(0)
        self._series   = {}            # {algorithm: errors in cm, nan = no fix}, aligned with _t
        self._capacity = 0             # frames the simulation keeps (its history length)
        self._hover    = None          # cursor x (px) while over the plot
        self._toggle   = QRectF()      # the "lin / log" switch, set when painted
        self.setMouseTracking(True)
        self.setMinimumSize(240, 120)

    def set_data(self, sim, shown) -> None:
        """Copy ``sim.times`` and ``sim.errors`` of the algorithms in ``shown``.

        Needs no lock: each ``list(deque)`` is one atomic copy, and trimming to
        a common length keeps the arrays aligned (a copy taken while the runner
        appends can be a frame out for one redraw)."""
        try:
            t      = list(sim.times)
            series = {k: list(sim.errors[k]) for k in theme.ALGO_ORDER if k in shown}
        except RuntimeError:           # the runner appended mid-copy: keep the old data
            return
        n              = min([len(t)] + [len(v) for v in series.values()])
        self._t        = np.array(t[len(t) - n:], dtype=float)
        self._series   = {k: np.array(v[len(v) - n:], dtype=float) for k, v in series.items()}
        self._capacity = sim.times.maxlen or 0
        self.update()

    def set_log(self, on: bool) -> None:
        self.log_scale = bool(on)
        self.update()

    # ─── Geometry ────────────────────────────────────────────────────────────

    def _plot_rect(self) -> QRectF:
        return QRectF(self.LEFT, self.TOP, max(self.width() - self.LEFT - self.RIGHT, 10),
                      max(self.height() - self.TOP - self.BOTTOM, 10))

    def _visible(self):
        """What the time axis shows: ``(t, {algorithm: errors}, t0, span)``, the
        latest frame at the right edge.  The axis spans ``window_s``, or less
        when the simulation keeps a shorter history (small timesteps) or has
        run for less time than that."""
        t = self._t
        if not len(t):
            return t, {}, 0.0, self.window_s
        dt   = float(t[-1] - t[-2]) if len(t) > 1 else 0.0
        span = min(self.window_s, self._capacity * dt) if self._capacity and dt > 0 else self.window_s
        span = min(span, max(float(t[-1] - t[0]) + dt, 5.0))   # a short run fills the plot, not a corner
        t0   = float(t[-1]) - span
        keep = t >= t0 - 1e-9
        return t[keep], {k: v[keep] for k, v in self._series.items()}, t0, span

    def _y_range(self, values: np.ndarray) -> tuple[float, float, list]:
        """Axis limits and tick values for the finite ``values``."""
        if self.log_scale:
            pos = values[values > 0]
            lo  = 10.0 ** math.floor(math.log10(max(float(pos.min()), 0.01))) if len(pos) else 0.1
            hi  = 10.0 ** math.ceil(math.log10(max(float(pos.max()) if len(pos) else 1.0, lo * 10)))
            hi  = min(hi, max(self.LOG_TOP, lo * 10))
            ticks = [10.0 ** e for e in range(round(math.log10(lo)), round(math.log10(hi)) + 1)]
            return lo, hi, ticks
        top, p95 = float(values.max()), float(np.percentile(values, 95))
        if top > 4 * p95:              # a few spikes: let them run off the top
            top = 1.3 * p95
        step = nice_step(max(top, 0.5), 4)
        hi   = step * max(math.ceil(top / step * 0.9999), 1)
        return 0.0, hi, list(np.arange(0.0, hi + step / 2, step))

    # ─── Painting ────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        pal = theme.current
        p   = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), color(pal.bg))
        plot = self._plot_rect()
        draw.label(p, QPointF(self.LEFT - 6, 11), "cm", pal.text_muted, 7.5, "right", halo=False)
        self._draw_toggle(p)
        title = "error from the true position"
        p.setFont(theme.mono_font(8))
        if self.LEFT + 2 + p.fontMetrics().horizontalAdvance(title) > self._toggle.left() - 6:
            title = "error"                            # narrow: keep clear of the lin / log switch
        draw.label(p, QPointF(self.LEFT + 2, 11), title, pal.text_dim, 8, halo=False)

        t, series, t0, span = self._visible()
        finite = np.concatenate([v[np.isfinite(v)] for v in series.values()]) if series else np.empty(0)
        if not len(t) or not series:
            _message(p, plot, "no frames yet" if series or not len(t) else "no algorithm shown")
            return
        if not len(finite):
            _message(p, plot, "no true position: no errors")
            return

        lo, hi, ticks = self._y_range(finite)
        if self.log_scale:
            llo, lhi = math.log10(lo), math.log10(hi)
            def Y(v):
                return plot.bottom() - (math.log10(max(v, lo * 0.5)) - llo) / (lhi - llo) * plot.height()
        else:
            def Y(v):
                return plot.bottom() - (v - lo) / (hi - lo) * plot.height()
        def X(tt):
            return plot.left() + (tt - t0) / span * plot.width()

        self._draw_axes(p, plot, ticks, Y, X, t0, span)
        p.save()
        p.setClipRect(plot.adjusted(-3, -3, 3, 3))
        for k, v in series.items():
            self._draw_series(p, t, v, X, Y, pal.algo[k])
        p.restore()
        self._draw_end_labels(p, plot, series, Y)
        if self._hover is not None and plot.left() <= self._hover <= plot.right():
            self._draw_readout(p, plot, t, series, X, Y, t0, span)

    def _draw_toggle(self, p: QPainter) -> None:
        pal = theme.current
        x   = self.width() - 10
        r1  = draw.label(p, QPointF(x, 11), "log", pal.accent if self.log_scale else pal.text_muted,
                         8, "right", bold=self.log_scale, halo=False)
        r0  = draw.label(p, QPointF(r1.left() - 18, 11), "lin",
                         pal.text_muted if self.log_scale else pal.accent, 8, "right",
                         bold=not self.log_scale, halo=False)
        draw.label(p, QPointF(r1.left() - 9, 11), "/", pal.text_muted, 8, "center", halo=False)
        self._toggle = r0.united(r1).adjusted(-4, -3, 4, 3)

    def _draw_axes(self, p, plot, ticks, Y, X, t0, span) -> None:
        pal = theme.current
        p.setPen(draw.pen(pal.grid_minor, 1.0))
        for v in ticks:
            y = Y(v)
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            text = f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"
            draw.label(p, QPointF(plot.left() - 6, y), text, pal.text_muted, 7.5, "right", halo=False)
        step = nice_step(span, 6)
        for tt in np.arange(max(math.ceil(t0 / step), 0) * step, t0 + span + 1e-9, step):
            x = X(tt)
            p.setPen(draw.pen(pal.grid_minor, 1.0))
            p.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            draw.label(p, QPointF(x, plot.bottom() + 10), f"{tt:g} s", pal.text_muted, 7.5, "center",
                       halo=False)
        p.setPen(draw.pen(pal.grid_major, 1.0))
        p.drawLine(plot.bottomLeft(), plot.bottomRight())

    @staticmethod
    def _draw_series(p, t, v, X, Y, c) -> None:
        """One line, broken where there is no fix (nan); lone points as dots."""
        path, run = QPainterPath(), []
        def close_run():
            if len(run) == 1:
                path.addEllipse(run[0], 1.3, 1.3)
            run.clear()
        for tt, e in zip(t, v):
            if not math.isfinite(e):
                close_run()
                continue
            q = QPointF(X(tt), Y(e))
            if run:
                path.lineTo(q)
            else:
                path.moveTo(q)
            run.append(q)
        close_run()
        draw.stroke(p, path, c, 1.3)

    def _draw_end_labels(self, p, plot, series, Y) -> None:
        """Each algorithm's key at the right end of its line, pushed apart so none
        overlap; "↑" marks a line that ends above the top of the axis."""
        pal, ends = theme.current, []
        for k, v in series.items():
            idx = np.where(np.isfinite(v))[0]
            if len(idx):
                last = float(v[idx[-1]])
                y    = Y(last)
                text = f"{k} ↑" if y < plot.top() - 0.5 else k
                y    = min(max(y, plot.top()), plot.bottom())
                ends.append([y, y, text, k, idx[-1] == len(v) - 1])
        ends.sort()
        gap = 11.0
        for i in range(1, len(ends)):
            ends[i][1] = max(ends[i][1], ends[i - 1][1] + gap)
        overflow = ends[-1][1] - (plot.bottom() + 4) if ends else 0
        if overflow > 0:
            for e in ends:
                e[1] -= overflow
            for i in range(len(ends) - 2, -1, -1):
                ends[i][1] = min(ends[i][1], ends[i + 1][1] - gap)
        for y_line, y_lab, text, k, live in ends:
            a = 1.0 if live else 0.45
            p.setPen(draw.pen(pal.algo[k], 1.0, 0.5 * a))
            p.drawLine(QPointF(plot.right() + 1, y_line), QPointF(plot.right() + 6, y_lab))
            draw.marker(p, QPointF(plot.right() + 12, y_lab), theme.ALGO_SHAPES[k], pal.algo[k], 6.5,
                        1.1, a)
            draw.label(p, QPointF(plot.right() + 19, y_lab), text, pal.algo[k], 7.5, halo=False, alpha=a)

    def _draw_readout(self, p, plot, t, series, X, Y, t0, span) -> None:
        """Cursor line at the nearest frame and a box with every value there."""
        pal  = theme.current
        t_at = t0 + (self._hover - plot.left()) / plot.width() * span
        i    = int(np.clip(np.searchsorted(t, t_at), 0, len(t) - 1))
        if i > 0 and abs(t[i - 1] - t_at) < abs(t[i] - t_at):
            i -= 1
        x = X(t[i])
        path = QPainterPath(QPointF(x, plot.top()))
        path.lineTo(x, plot.bottom())
        draw.stroke(p, path, pal.text_dim, 1.0, 0.8, draw.DASH, glow=False)
        rows = [(f"t {t[i]:.1f} s", pal.text_dim)]
        for k, v in series.items():
            rows.append((f"{k:<6} {fmt_cm(v[i]):>6}", pal.algo[k]))
            if math.isfinite(v[i]) and plot.top() <= Y(v[i]) <= plot.bottom():
                draw.marker(p, QPointF(x, Y(v[i])), "circle", pal.algo[k], 5.0, 1.2)
        p.setFont(theme.mono_font(7.5))
        fm    = p.fontMetrics()
        w     = max(fm.horizontalAdvance(r[0]) for r in rows) + 12
        h     = len(rows) * 12 + 6
        left  = x + 10 if x + 10 + w <= self.width() - 2 else x - 10 - w
        box   = QRectF(left, max(min(plot.top() + 2, self.height() - h - 2), 2), w, h)
        p.setPen(draw.pen(pal.border, 1.0))
        p.setBrush(color(pal.panel_alt, 0.92))
        p.drawRoundedRect(box, 3, 3)
        for j, (text, c) in enumerate(rows):
            draw.label(p, QPointF(box.left() + 6, box.top() + 9 + 12 * j), text, c, 7.5, halo=False)

    # ─── Mouse ───────────────────────────────────────────────────────────────

    def mouseMoveEvent(self, event) -> None:
        self._hover = event.position().x()
        self.update()

    def leaveEvent(self, _event) -> None:
        self._hover = None
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._toggle.contains(event.position()):
            self.set_log(not self.log_scale)

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        for text, log in (("Linear scale", False), ("Log scale", True)):
            act = menu.addAction(text)
            act.setCheckable(True)
            act.setChecked(self.log_scale == log)
            act.triggered.connect(lambda _=False, log=log: self.set_log(log))
        menu.popup(event.globalPos())

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            on_toggle = self._toggle.contains(QPointF(event.pos()))
            return _tooltip_event(self, event, "Click to switch the error axis between linear and "
                                  "logarithmic (log shows small and huge errors together)."
                                  if on_toggle else "")
        return super().event(event)


# ─── Echo scope ──────────────────────────────────────────────────────────────

KIND_STATUS = {"direct": "ok", "reflect": "echo", "diffract": "bent"}   # colour of each path kind
KIND_TEXT   = {"direct": "direct", "reflect": "reflection", "diffract": "bent round a corner"}
TIMEOUTS    = ("lost", "radio", "timeout")
LEGEND      = (                # (text, colour key, tooltip)
    ("direct", "ok", "Sound that came straight from the transmitter."),
    ("reflection", "echo", "Sound that bounced off a wall or obstacle. It travelled further, so a "
                           "receiver that detects it first reads long."),
    ("bent", "bent", "Sound that bent round an obstacle corner (diffraction): reads a little long."),
    ("crosstalk", "crosstalk", "Another receiver's own burst (unmodified HC-SR04s). It left at the "
                               "trigger, before the transmitter fired, so it reads short."),
    ("too weak", "weak", "Hollow: never crosses the threshold within the 8-cycle burst (or is "
                         "outside the field-of-view limit), so it cannot be detected."),
    ("threshold", "threshold", "Dotted line: the comparator's detection threshold (0 dB). Each stem "
                               "is as tall as its arrival's strength over it, from -20 to +40 dB."),
    ("true", "truth", "Where an ideal receiver's echo line would drop for the true distance."),
)


def arrival_colour(a) -> str:
    """Status colour of an arrival's path kind (another receiver's burst: crosstalk)."""
    return theme.current.status["crosstalk" if a.source != "TX" else KIND_STATUS[a.path.kind]]


class EchoScope(QWidget):
    """Logic-analyser view of one ping: per receiver every arrival and the echo line."""

    MAX_MS       = 25.0           # the time axis never runs longer: later events are pointed to
    DB_LO, DB_HI = -20.0, 40.0    # stem height scale; 0 dB is the detection threshold
    RIGHT        = 34             # margin right of the plot: timeout arrows, later-arrival counts
    TWO_LINES    = 28             # px: rows this tall show the error under the row label

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame  = None
        self._env    = None
        self._ref_ms = 0.0         # detection delay counted as calibration (see Hardware.ref_cycle)
        self.hits    = []          # [(QRectF, tooltip)] from the last paint, for hover tooltips
        self.setMinimumSize(320, 170)

    def set_frame(self, frame, env) -> None:
        """Show ``frame``'s readings, timed with environment ``env``."""
        if env is not None and env is not self._env:       # a changed env is a new object
            self._ref_ms = Hardware([], env=env).ref_cycle() * ac.CYCLE_MS
        self._frame, self._env = frame, env
        self.update()

    def tooltip_at(self, pos: QPointF) -> str:
        """Tooltip for whatever was drawn at ``pos`` (the nearest stem wins)."""
        best, text = 1e9, ""
        for rect, tip in self.hits:
            if rect.contains(pos):
                d = abs(rect.center().x() - pos.x())
                if d < best:
                    best, text = d, tip
        return text

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            return _tooltip_event(self, event, self.tooltip_at(QPointF(event.pos())))
        return super().event(event)

    # ─── Layout ──────────────────────────────────────────────────────────────

    def _time_end(self, readings, env) -> float:
        """End of the time axis (ms): every arrival and drop, rounded up, at most ``MAX_MS``."""
        last = max(env.burst_lead_ms, env.rx_ready_ms) + 1.0
        for r in readings:
            for a in r.arrivals:
                last = max(last, a.time_ms)
            if r.echo_ms is not None:
                last = max(last, r.echo_ms)
        return min(self.MAX_MS, 5.0 * math.ceil((last + 0.3) / 5.0))

    def _ideal_ms(self, true_cm: float) -> float:
        """When an ideal receiver's echo line would drop for the true distance."""
        e = self._env
        c = ac.speed_of_sound(e.cal_temperature_c, e.cal_humidity_pct) / 10.0     # cm per ms
        return e.burst_lead_ms + self._ref_ms + true_cm / c

    def _events(self, readings) -> list:
        """The receivers starting to listen and the transmitter firing, earliest
        first: ``[(t ms, text, colour, dash, tooltip)]``."""
        pal, env = theme.current, self._env
        fire = env.burst_lead_ms + env.sync_offset_us / 1000.0
        if any(r.status == "radio" for r in readings):
            tx = (fire, "TX trigger lost", pal.status["radio"], draw.DASH,
                  "The transmitter missed its radio trigger this ping (the radio has no retries): "
                  "it never fired, so every receiver times out")
        else:
            jitter = (f", give or take {env.sync_jitter_us:.0f} µs of jitter"
                      if env.sync_jitter_us > 0 else "")
            tx = (fire, "TX fires", pal.accent, draw.DASH,
                  f"The transmitter fires at {fire:.2f} ms, on its radio trigger{jitter}")
        rx = (env.rx_ready_ms, "listening", pal.text_dim, draw.DOT,
              f"The receivers listen from {env.rx_ready_ms:.2f} ms: their echo lines rise. "
              "Anything arriving earlier is not heard")
        return sorted([rx, tx], key=lambda e: e[0])

    @staticmethod
    def _event_labels(p: QPainter, events, X, left: float) -> list:
        """Left edge of each event's label: the first left of its line when it
        fits, the others right of theirs, never overlapping the one before."""
        p.setFont(theme.mono_font(7.5))
        fm, out, end = p.fontMetrics(), [], left
        for i, (t, text, *_rest) in enumerate(events):
            x, w = X(t), fm.horizontalAdvance(text)
            x0   = x - 4 - w if i == 0 and x - 4 - w >= left else max(x + 4, end + 8)
            out.append(x0)
            end = x0 + w
        return out

    # ─── Painting ────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        pal = theme.current
        p   = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), color(pal.bg))
        self.hits = []
        f, env = self._frame, self._env
        if f is None or env is None:
            _message(p, QRectF(self.rect()), "waiting for the first ping")
            return
        if not f.readings:
            _message(p, QRectF(self.rect()), "replayed ticks: no sound arrivals to show")
            return

        p.setFont(theme.mono_font(8))
        two     = (self.height() - 54) / len(f.readings) >= self.TWO_LINES
        label_w = p.fontMetrics().horizontalAdvance("R0  crosstalk" if two else "R0  crosstalk  +999.9") + 16
        right   = max(self.width() - self.RIGHT, label_w + 40)
        t_end   = self._time_end(f.readings, env)
        def X(t):
            return label_w + t / t_end * (right - label_w)

        # The legend shares the top line with the event labels when both fit
        events  = self._events(f.readings)
        lefts   = self._event_labels(p, events, X, label_w)
        legend  = self._legend_lines(p, self.width() - 20)
        p.setFont(theme.mono_font(7.5))
        ends    = [x0 + p.fontMetrics().horizontalAdvance(e[1]) for x0, e in zip(lefts, events)]
        shared  = len(legend) == 1 and max(ends) + 12 <= self.width() - 10 - legend[0][0]
        top     = 20 if shared else 34 + 14 * (len(legend) - 1)
        plot    = QRectF(label_w, top, right - label_w, max(self.height() - top - 20, 20))

        self._draw_time_axis(p, plot, t_end, X)
        self._draw_legend(p, legend)
        row_h = plot.height() / len(f.readings)
        for i, r in enumerate(f.readings):
            self._draw_row(p, i, r, QRectF(0, plot.top() + i * row_h, self.width(), row_h), plot,
                           t_end, X, two)
        self._draw_events(p, plot, events, lefts, X)

    @staticmethod
    def _legend_lines(p: QPainter, max_w: float) -> list:
        """``LEGEND`` packed into lines no wider than ``max_w``: ``[[width, [item, ...]], ...]``."""
        p.setFont(theme.mono_font(7.5))
        fm, lines = p.fontMetrics(), [[0.0, []]]
        for item in LEGEND:
            w = fm.horizontalAdvance(item[0]) + 22
            if lines[-1][1] and lines[-1][0] + w > max_w:
                lines.append([0.0, []])
            lines[-1][0] += w
            lines[-1][1].append(item)
        return lines

    def _draw_legend(self, p: QPainter, lines: list) -> None:
        """What the stem colours, hollow stems, dotted line and triangle mean,
        right-aligned at the top."""
        pal = theme.current
        for k, (width, items) in enumerate(lines):
            x, y = self.width() - 10 - width + 11, 10 + 14 * k
            for text, key, tip in items:
                if key == "truth":
                    self._truth_mark(p, x + 4, y + 3)
                elif key == "threshold":
                    path = QPainterPath(QPointF(x - 2, y + 1))
                    path.lineTo(x + 8, y + 1)
                    draw.stroke(p, path, pal.text_muted, 1.0, 0.8, draw.DOT, glow=False)
                else:
                    c = pal.text_dim if key == "weak" else pal.status[key]
                    path = QPainterPath(QPointF(x + 4, y + 5))
                    path.lineTo(x + 4, y - 3)
                    draw.stroke(p, path, c, 1.4, glow=False)
                    draw.marker(p, QPointF(x + 4, y - 4), "circle", c, 4.5, 1.1,
                                fill=0.0 if key == "weak" else 1.0)
                r = draw.label(p, QPointF(x + 11, y), text, pal.text_dim, 7.5, halo=False)
                self.hits.append((QRectF(x - 2, r.top(), r.right() - x + 4, r.height()), tip))
                x = r.right() + 11

    @staticmethod
    def _truth_mark(p: QPainter, x: float, y: float) -> None:
        """Small triangle pointing down at ``(x, y)``: where the line should drop."""
        tri = QPainterPath(QPointF(x - 3, y - 4))
        tri.lineTo(x + 3, y - 4)
        tri.lineTo(x, y)
        tri.closeSubpath()
        draw.stroke(p, tri, theme.current.truth, 1.0, 0.9, glow=False)

    def _draw_time_axis(self, p, plot, t_end, X) -> None:
        pal  = theme.current
        step = nice_step(t_end, 5)
        if step * plot.width() / t_end > 60:
            p.setPen(draw.pen(pal.grid_minor, 1.0, 0.6))
            for t in np.arange(step / 5, t_end, step / 5):
                p.drawLine(QPointF(X(t), plot.top()), QPointF(X(t), plot.bottom()))
        for t in np.arange(0.0, t_end + 1e-9, step):
            x = X(t)
            p.setPen(draw.pen(pal.grid_major, 1.0))
            p.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            draw.label(p, QPointF(x, plot.bottom() + 10), f"{t:g}", pal.text_muted, 7.5, "center",
                       halo=False)
        draw.label(p, QPointF(plot.left() - 10, plot.bottom() + 10), "ms after trigger",
                   pal.text_muted, 7, "right", halo=False)

    def _draw_events(self, p, plot, events, lefts, X) -> None:
        """Vertical markers for the receivers starting to listen and the transmitter firing."""
        for (t, text, c, dash, tip), x0 in zip(events, lefts):
            x    = X(t)
            path = QPainterPath(QPointF(x, plot.top() - 4))
            path.lineTo(x, plot.bottom())
            draw.stroke(p, path, c, 1.0, 0.8, dash, glow=False)
            rect = draw.label(p, QPointF(x0, plot.top() - 9), text, c, 7.5, halo=False)
            self.hits.append((QRectF(x - 3, plot.top() - 16, 6, 16), tip))
            self.hits.append((rect, tip))

    def _draw_row(self, p, i, r, row: QRectF, plot: QRectF, t_end, X, two: bool) -> None:
        pal   = theme.current
        env   = self._env
        sc    = pal.status.get(r.status, pal.text)
        top   = row.top() + 4
        base  = row.top() + row.height() * 0.64          # stems stand on this line
        hi    = row.top() + row.height() * 0.74          # echo line high ...
        lo    = row.bottom() - 4                          # ... and low
        def Ydb(db):
            db = min(max(db, self.DB_LO), self.DB_HI)
            return base - (db - self.DB_LO) / (self.DB_HI - self.DB_LO) * (base - top)

        p.setPen(draw.pen(pal.grid_minor, 1.0))
        p.drawLine(QPointF(8, row.bottom()), QPointF(plot.right(), row.bottom()))
        self._draw_row_label(p, i, r, QRectF(0, row.top(), plot.left() - 4, row.height()), sc, two)
        if r.status == "off":
            draw.label(p, QPointF(plot.center().x(), row.center().y()), "switched off",
                       pal.text_muted, 8, "center", halo=False)
            return

        # Detection threshold (0 dB)
        thr = QPainterPath(QPointF(plot.left(), Ydb(0)))
        thr.lineTo(plot.right(), Ydb(0))
        draw.stroke(p, thr, pal.text_muted, 1.0, 0.55, draw.DOT, glow=False)

        # The echo line drops where the comparator fires (an injected timeout
        # reads as if it dropped at the module's timeout)
        drop       = env.rx_ready_ms + env.timeout_ms if r.status == "timeout" else r.echo_ms
        falls      = drop is not None and drop <= t_end
        dropped_by = r.detected if falls and r.status != "timeout" else None

        # Arrivals
        later = []
        for a in r.arrivals:
            if a.time_ms > t_end:
                later.append(a)
                continue
            self._draw_stem(p, r, a, X, base, Ydb(a.ratio_db), hi, a is dropped_by)
        if later:
            rect = draw.label(p, QPointF(plot.right() + 4, top + 5), f"+{len(later)}›",
                              pal.text_muted, 7, halo=False)
            self.hits.append((rect.adjusted(-2, -2, 2, 2), f"R{i}: {len(later)} later arrival(s), "
                               "after the end of the axis:\n" + "\n".join(
                                   self._arrival_line(a) for a in later)))

        # Echo line: low, high once listening, low again when it drops
        ready = X(env.rx_ready_ms)
        pts   = [QPointF(plot.left(), lo), QPointF(ready, lo), QPointF(ready, hi)]
        pts  += ([QPointF(X(drop), hi), QPointF(X(drop), lo), QPointF(plot.right(), lo)] if falls
                 else [QPointF(plot.right() - 2, hi)])
        draw.stroke(p, draw.polyline_path(pts), sc, 1.3)
        if not falls:
            draw.arrow(p, QPointF(plot.right() - 12, hi), QPointF(plot.right() + 8, hi), sc, 1.3, 5)
        if r.status in TIMEOUTS or not falls:
            text = (f"timeout {drop:.0f} ms" if r.status in TIMEOUTS else
                    "injected dropout: reads 0" if r.status == "dropout" else "never drops: reads 0")
            draw.label(p, QPointF(X(drop) - 4 if falls else plot.right() - 14, (hi + lo) / 2), text,
                       sc, 7, "right")

        # Where an ideal receiver would have dropped its line
        if math.isfinite(r.true_cm):
            t = self._ideal_ms(r.true_cm)
            if t <= t_end:
                x = X(t)
                self._truth_mark(p, x, hi - 2)
                self.hits.append((QRectF(x - 4, hi - 8, 8, lo - hi + 8),
                                   f"R{i}: true distance {r.true_cm:.1f} cm; an ideal receiver's "
                                   f"echo line would drop here, at {t:.2f} ms"))

    def _draw_row_label(self, p, i, r, cell: QRectF, sc: str, two: bool) -> None:
        """"R2  echo" and the reading's error (measured − true): under it in a
        tall row, right-aligned beside it in a short one."""
        pal = theme.current
        y1  = cell.center().y() - (6 if two else 0)
        r0  = draw.label(p, QPointF(10, y1), f"R{i}", pal.text, 8, bold=True, halo=False)
        draw.label(p, QPointF(r0.right() + 7, y1), r.status, sc, 8, halo=False)
        if r.measured_cm is not None:
            err  = r.measured_cm - r.true_cm
            text = (f"{err:+.1f}" if abs(err) < 1000 else f"{err:+.0f}") + (" cm" if two else "")
        else:
            text = "reads 0" if r.status != "off" else ""
        if two:
            draw.label(p, QPointF(10, y1 + 12), text, pal.text_dim, 7.5, halo=False)
        else:
            draw.label(p, QPointF(cell.right() - 6, y1), text, pal.text_dim, 7.5, "right", halo=False)
        meas = "—" if r.measured_cm is None else f"{r.measured_cm:.1f} cm"
        self.hits.append((cell,
                           f"R{i} {r.status}: {STATUS.get(r.status, '')}\n"
                           f"measured {meas}, true {r.true_cm:.1f} cm, tick {r.tick}"))

    def _draw_stem(self, p, r, a, X, base, y, hi, dropped: bool) -> None:
        """One arrival: a stem as tall as its strength, filled if it could be
        detected, bold if the comparator fired on it.  ``dropped``: it is what
        dropped the echo line, so a dotted link leads to where the line falls."""
        env      = self._env
        c        = arrival_colour(a)
        x        = X(a.time_ms)
        detected = a is r.detected
        heard    = a.time_ms >= env.rx_ready_ms
        alpha    = 1.0 if heard else 0.4
        stem     = QPainterPath(QPointF(x, base))
        stem.lineTo(x, y)
        draw.stroke(p, stem, c, 2.4 if detected else 1.1, alpha, glow=True if detected else None)
        draw.marker(p, QPointF(x, y), "circle", c, 6.0 if detected else 4.5, 1.2, alpha,
                    fill=0.0 if a.cycle is None else 1.0)
        if dropped:
            fall = QPointF(X(a.detect_ms), hi)
            link = QPainterPath(QPointF(x, base))
            link.lineTo(fall)
            draw.stroke(p, link, c, 1.0, 0.8, draw.DOT, glow=False)
            draw.marker(p, fall, "diamond", c, 6.0, 1.2, fill=0.35)
        self.hits.append((QRectF(x - 4, min(y, base) - 5, 8, abs(base - y) + 10),
                           self._stem_tip(r, a, dropped, heard)))

    @staticmethod
    def _arrival_line(a) -> str:
        src = "transmitter" if a.source == "TX" else f"{a.source}'s own burst"
        return (f"{src}, {KIND_TEXT[a.path.kind]}, {a.path.length:.1f} cm, "
                f"{a.time_ms:.2f} ms, {a.ratio_db:+.1f} dB")

    def _stem_tip(self, r, a, dropped: bool, heard: bool) -> str:
        env   = self._env
        src   = "transmitter" if a.source == "TX" else f"{a.source}'s own burst (crosstalk)"
        kind  = KIND_TEXT[a.path.kind]
        if a.path.kind == "reflect":
            n    = len(a.path.points) - 2
            kind = f"reflection, {n} bounce{'s' if n > 1 else ''}"
        lines = [f"from the {src}", f"path: {kind}, {a.path.length:.1f} cm",
                 f"arrives {a.time_ms:.3f} ms", f"strength {a.ratio_db:+.1f} dB over the threshold"]
        strong = not env.threshold or ac.detect_cycle(10 ** (a.ratio_db / 20), env.rise_cycles)
        if a.cycle is None:
            lines.append("outside the field-of-view limit: ignored" if strong else
                         "too weak: never crosses the threshold within the 8-cycle burst")
        elif not env.threshold:
            lines.append("ideal timing (threshold detection off): detected at once")
        else:
            lines.append(f"crosses the threshold on cycle {a.cycle} "
                         f"(+{a.cycle * ac.CYCLE_MS * 1000:.0f} µs)")
        if dropped:
            lines.append("→ this arrival dropped the echo line")
        elif a is r.detected:
            lines.append(f"→ detected, but an injected fault ({r.status}) replaced the reading")
        elif not heard and a.cycle is not None:
            lines.append(f"not heard: arrived before the receiver listens ({env.rx_ready_ms:.1f} ms)")
        elif a.cycle is not None and r.detected is not None:
            lines.append("not used: the echo line had already dropped")
        return "\n".join(lines)
