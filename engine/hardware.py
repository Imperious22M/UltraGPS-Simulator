"""
Hardware model — turns the transmitter's true position into the tick counts
the base station would print, following one ping through the real system:

  1. The base station fires all six receiver HC-SR04s together (t = 0).  Each
     raises its echo line once it is listening (``rx_ready_ms``).
  2. The transmitter fires ``burst_lead_ms`` later, on its radio trigger, give
     or take the radio/PLL timing error (``sync_offset_us``, ``sync_jitter_us``).
     The radio has no retries, so the trigger is sometimes lost (``radio_loss``).
  3. Sound reaches each receiver along every path the world allows: direct,
     bounced off walls and obstacles, or bent round an obstacle corner.  Each
     arrival's strength comes from both beam patterns, spreading, air absorption,
     reflection and diffraction losses, and a little turbulence (``amp_jitter_db``).
  4. A receiver's comparator fires on the first cycle at which an arrival's
     burst crosses its threshold, so its echo line drops at the first arrival
     strong enough to detect, a few cycles late if weak (range walk).  Nothing
     detectable: the HC-SR04 times out (``timeout_ms``) and drops the line anyway.
  5. The base station records the time each echo line drops, as a count of its
     sampling loop.  A line that never drops within the sampling window
     (``window_ms``), or a pin with nothing connected, reads 0.
  6. Injected faults (``Faults``) are applied on top: extra noise, dropouts,
     lost signal and bad values, for testing the solvers.

Times become ticks through each receiver's true response from sensors.toml,
``distance_cm = tick × slope + intercept``, where distance is the time since
the transmitter fired at the calibration speed of sound.  Detection takes a
cycle or so even for a strong signal; that part (the delay on axis at
``ref_distance_cm``) is treated as part of the calibration, so only range walk
beyond it shows up as error.

``Environment.ideal()`` turns all of this off except the faults, giving the
straight-line model of earlier versions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace

import numpy as np

from engine import acoustics as ac
from engine.world import Path, World


@dataclass
class Environment:
    # Air
    temperature_c: float      = 20.0
    humidity_pct: float       = 50.0
    cal_temperature_c: float  = 20.0    # conditions the calibration was taken in
    cal_humidity_pct: float   = 50.0
    wind_mps: float           = 0.0     # steady air flow (fan, HVAC)
    wind_dir_deg: float       = 0.0     # direction it blows toward
    # Transducers
    beam_patterns: bool       = True    # piston beam patterns (off: omnidirectional)
    rx_beam_deg: float        = 27.5    # receiver −6 dB half-angle
    tx_mode: str              = "omni"  # transmitter: "omni", "heading" (faces where it drives), "fixed"
    tx_beam_deg: float        = 27.5
    tx_fixed_deg: float       = 90.0
    fov_cutoff: bool          = False   # hard field-of-view limit: nothing heard outside ±fov_half_deg
    fov_half_deg: float       = 15.0
    # Detection
    threshold: bool           = True    # threshold detection with range walk (off: ideal timing)
    snr_db: float             = 30.0    # burst amplitude over the threshold, on axis at 1 m
    rise_cycles: float        = 6.0     # transducer ring-up time constant, cycles of 40 kHz
    amp_jitter_db: float      = 1.0     # ping-to-ping amplitude fluctuation (air turbulence)
    ref_distance_cm: float    = 150.0   # detection delay at this distance counts as calibration
    near_noise: bool          = False   # extra noise within near_cm of a receiver
    near_cm: float            = 16.0    # 4 % of the HC-SR04's 400 cm range
    near_noise_cm: float      = 5.0
    max_range_cm: float       = 400.0   # with threshold detection off: no echo beyond this
    # Propagation
    reflections: bool         = True    # sound bounces off walls and obstacles
    bounces: int              = 1       # most bounces per path (1 or 2)
    diffraction: bool         = True    # sound bends round obstacle corners
    # Timing and electronics
    burst_lead_ms: float      = 4.0     # transmitter fires this long after the receivers
    rx_ready_ms: float        = 3.0     # receivers start listening this long after their trigger
    timeout_ms: float         = 38.0    # HC-SR04 gives up and drops its echo line
    window_ms: float          = 80.0    # base station stops sampling (80 % of the 100 ms frame)
    sync_offset_us: float     = 0.0     # transmitter fires late by this much
    sync_jitter_us: float     = 0.0     # ... plus a random amount, the same for every receiver
    radio_loss: float         = 0.0     # chance the transmitter misses its radio trigger
    tick_dither: bool         = True    # sampling loop starts at a random phase: ±1 tick
    crosstalk: bool           = False   # receivers' own bursts are heard (unmodified modules)

    @staticmethod
    def ideal() -> "Environment":
        """Straight-line ranges with no physics, as in the first simulator."""
        return Environment(beam_patterns=False, threshold=False, amp_jitter_db=0.0,
                           reflections=False, diffraction=False, tick_dither=False)

    def copy(self, **changes) -> "Environment":
        return replace(self, **changes)

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


PRESETS = {
    "Ideal":     dict(Environment.ideal().as_dict()),
    "Realistic": dict(Environment().as_dict()),
    "Harsh":     dict(Environment(snr_db=24.0, amp_jitter_db=2.0, bounces=2, sync_jitter_us=30.0,
                                  radio_loss=0.02, wind_mps=0.5).as_dict()),
}


@dataclass
class Faults:
    """Synthetic faults injected after the physics, for testing the solvers."""
    noise_cm: float     = 1.0    # Gaussian noise added to every range
    dropout_rate: float = 0.0    # reading missed: 0 ticks
    timeout_rate: float = 0.0    # signal lost: the HC-SR04 timeout's high value
    spike_rate: float   = 0.0    # wrong value
    spike_pct: float    = 50.0   # ... the range off by a random amount within ± this %


@dataclass
class Receiver:
    id: int
    position: np.ndarray        # where it really is (config.toml has where the library thinks)
    facing_deg: float
    slope: float                # true response: distance_cm = tick × slope + intercept
    intercept: float
    state: str = "on"           # "on", "blocked" (no line of sight) or "off"


@dataclass
class Arrival:
    source: str                 # "TX", or "R3" for receiver 3's own burst (crosstalk)
    path: Path
    time_ms: float              # after the receivers' trigger
    ratio_db: float             # strength over the detection threshold
    cycle: int | None           # cycle the comparator fires on (None: too weak)

    @property
    def detect_ms(self) -> float | None:
        return None if self.cycle is None else self.time_ms + self.cycle * ac.CYCLE_MS


@dataclass
class Reading:
    tick: int
    status: str                 # see STATUS
    true_cm: float              # straight-line distance
    measured_cm: float | None   # what the hardware timed, as a distance
    echo_ms: float | None       # when the echo line dropped (None: not in the window)
    los: bool                   # line of sight clear
    detected: Arrival | None = None
    arrivals: list = field(default_factory=list)


STATUS = {
    "ok":        "direct sound detected",
    "walk":      "direct sound, detected late (weak signal)",
    "echo":      "a reflection was detected first: reads long",
    "bent":      "sound bent round an obstacle corner: reads a little long",
    "crosstalk": "another receiver's burst was detected: reads short",
    "lost":      "nothing detected: HC-SR04 timed out (high value)",
    "missed":    "echo line never dropped in the sampling window (0)",
    "radio":     "transmitter missed its radio trigger",
    "angle":     "outside the field-of-view limit (0)",
    "range":     "beyond the 400 cm range (0)",
    "off":       "receiver switched off (0)",
    "near":      "very close: extra noise",
    "dropout":   "injected: reading missed (0)",
    "timeout":   "injected: signal lost (high value)",
    "spike":     "injected: wrong value",
}


class Hardware:
    def __init__(self, receivers: list[Receiver], world: World | None = None,
                 env: Environment | None = None, faults: Faults | None = None,
                 seed: int | None = None):
        self.receivers = receivers
        self.world     = world or World()
        self.env       = env or Environment()
        self.faults    = faults or Faults()
        self.rng       = np.random.default_rng(seed)

    # ─── Helpers ─────────────────────────────────────────────────────────────

    @property
    def coords(self) -> np.ndarray:
        return np.array([r.position for r in self.receivers], dtype=float)

    def tx_facing(self, heading_deg: float) -> float | None:
        e = self.env
        if not e.beam_patterns or e.tx_mode == "omni":
            return None
        return heading_deg if e.tx_mode == "heading" else e.tx_fixed_deg

    def strength_db(self, path: Path, tx_facing: float | None, rx_facing: float,
                    tx_beam: float | None = None) -> float:
        """Arrival strength over the detection threshold (dB), without turbulence."""
        e = self.env
        a = 10.0 ** (e.snr_db / 20.0) * path.gain
        a *= ac.spreading(path.length, e.temperature_c, e.humidity_pct)
        if e.beam_patterns:
            if tx_facing is not None:
                a *= ac.directivity(_angle(path.depart) - tx_facing, tx_beam or e.tx_beam_deg)
            a *= ac.directivity(_angle(-path.arrive) - rx_facing, e.rx_beam_deg)
        return 20.0 * math.log10(max(a, 1e-12))

    def ref_cycle(self) -> int:
        """Detection cycle of an on-axis burst at ``ref_distance_cm`` in calibration conditions."""
        e = self.env
        if not e.threshold:
            return 0
        a = 10.0 ** (e.snr_db / 20.0) * ac.spreading(e.ref_distance_cm, e.cal_temperature_c,
                                                     e.cal_humidity_pct)
        return ac.detect_cycle(a, e.rise_cycles) or ac.BURST_CYCLES

    def _travel_ms(self, path: Path) -> float:
        e = self.env
        c = ac.speed_of_sound(e.temperature_c, e.humidity_pct)
        w = e.wind_mps * np.array([math.cos(math.radians(e.wind_dir_deg)),
                                   math.sin(math.radians(e.wind_dir_deg))])
        return sum(L / 100.0 / (c + float(np.dot(w, u))) for u, L in path.legs) * 1000.0

    def _in_fov(self, path: Path, rx: Receiver) -> bool:
        off = (_angle(-path.arrive) - rx.facing_deg + 180.0) % 360.0 - 180.0
        return abs(off) <= self.env.fov_half_deg

    def _arrival(self, source: str, path: Path, start_ms: float, tx_facing, rx: Receiver,
                 tx_beam=None) -> Arrival:
        e   = self.env
        sdb = self.strength_db(path, tx_facing, rx.facing_deg, tx_beam)
        if e.amp_jitter_db > 0:
            sdb += self.rng.normal(0.0, e.amp_jitter_db)
        cyc = ac.detect_cycle(10.0 ** (sdb / 20.0), e.rise_cycles) if e.threshold else 0
        if e.fov_cutoff and not self._in_fov(path, rx):
            cyc = None
        return Arrival(source, path, start_ms + self._travel_ms(path), sdb, cyc)

    # ─── Measurement ─────────────────────────────────────────────────────────

    def arrivals(self, tx_pos, heading_deg: float = 0.0, fired_ms: float | None = None) -> list[list[Arrival]]:
        """Every arrival at every receiver that is on, earliest first."""
        e        = self.env
        tx_pos   = np.asarray(tx_pos, dtype=float)
        tx_face  = self.tx_facing(heading_deg)
        fired_ms = e.burst_lead_ms if fired_ms is None else fired_ms
        order    = e.bounces if e.reflections else 0
        wl       = ac.wavelength_cm(e.temperature_c, e.humidity_pct)
        out      = []
        for rx in self.receivers:
            arr = []
            if rx.state != "off":
                if math.isfinite(fired_ms):
                    for p in self.world.paths(tx_pos, rx.position, order, e.diffraction,
                                              direct=rx.state != "blocked", wavelength=wl):
                        arr.append(self._arrival("TX", p, fired_ms, tx_face, rx))
                if e.crosstalk:
                    for other in self.receivers:
                        if other.state == "off":
                            continue
                        same = other is rx
                        for p in self.world.paths(other.position, rx.position, max(order, 1),
                                                  e.diffraction, direct=not same, wavelength=wl):
                            a = self._arrival(f"R{other.id}", p, 0.0, other.facing_deg, rx,
                                              e.rx_beam_deg)
                            arr.append(a)
            arr.sort(key=lambda a: a.time_ms)
            out.append(arr)
        return out

    def measure(self, tx_pos, heading_deg: float = 0.0) -> list[Reading]:
        """One ping: a ``Reading`` per receiver."""
        e, f, rng = self.env, self.faults, self.rng
        tx_pos    = np.asarray(tx_pos, dtype=float)
        c_cal     = ac.speed_of_sound(e.cal_temperature_c, e.cal_humidity_pct) / 10.0   # cm per ms
        ref_ms    = self.ref_cycle() * ac.CYCLE_MS
        radio_ok  = rng.random() >= e.radio_loss
        fired_ms  = e.burst_lead_ms + (e.sync_offset_us + rng.normal(0.0, e.sync_jitter_us)
                                       if e.sync_jitter_us > 0 else e.sync_offset_us) / 1000.0
        all_arr   = self.arrivals(tx_pos, heading_deg, fired_ms if radio_ok else math.inf)

        readings = []
        for rx, arr in zip(self.receivers, all_arr):
            true_cm = float(np.hypot(*(rx.position - tx_pos)))
            los     = rx.state != "blocked" and not self.world.blocked(tx_pos, rx.position)
            if rx.state == "off":
                readings.append(Reading(0, "off", true_cm, None, None, los))
                continue

            heard = [a for a in arr if a.cycle is not None and a.time_ms >= e.rx_ready_ms]
            det   = min(heard, key=lambda a: a.detect_ms) if heard else None
            if det is not None and not e.threshold and det.source == "TX" \
                    and det.path.length > e.max_range_cm:
                det, status, echo_ms = None, "range", None
            elif det is not None:
                echo_ms = det.detect_ms
                status  = _status(det, ref_ms)
            elif not radio_ok:
                status, echo_ms = "radio", e.rx_ready_ms + e.timeout_ms
            elif e.fov_cutoff and any(a.source == "TX" and not self._in_fov(a.path, rx) for a in arr):
                status, echo_ms = "angle", None
            elif not e.threshold and any(a.source == "TX" for a in arr):
                status, echo_ms = "range", None
            else:
                status, echo_ms = "lost", e.rx_ready_ms + e.timeout_ms
            if status == "radio" or status == "lost":
                if echo_ms > e.window_ms:
                    status, echo_ms = "missed", None

            measured = None if echo_ms is None else c_cal * (echo_ms - e.burst_lead_ms - ref_ms)
            if measured is not None:
                measured += rng.normal(0.0, f.noise_cm) if f.noise_cm > 0 else 0.0
                if e.near_noise and true_cm < e.near_cm and status in ("ok", "walk"):
                    measured += rng.normal(0.0, e.near_noise_cm)
                    status = "near"
            # Injected faults
            if rng.random() < f.dropout_rate:
                measured, status, echo_ms = None, "dropout", None
            elif rng.random() < f.timeout_rate:
                measured, status = c_cal * e.timeout_ms, "timeout"
            elif measured is not None and rng.random() < f.spike_rate:
                p = f.spike_pct / 100.0
                measured *= 1.0 + rng.uniform(-p, p)
                status = "spike"

            readings.append(Reading(self._tick(rx, measured), status, true_cm, measured,
                                    echo_ms, los, det, arr))
        return readings

    def _tick(self, rx: Receiver, measured: float | None) -> int:
        if measured is None:
            return 0
        x = (measured - rx.intercept) / rx.slope
        t = math.floor(x + self.rng.random()) if self.env.tick_dither else round(x)
        return max(0, int(t))


def _angle(v) -> float:
    return math.degrees(math.atan2(v[1], v[0]))


def _status(det: Arrival, ref_ms: float) -> str:
    if det.source != "TX":
        return "crosstalk"
    if det.path.kind == "reflect":
        return "echo"
    if det.path.kind == "diffract":
        return "bent"
    return "walk" if det.cycle * ac.CYCLE_MS > ref_ms + 1.5 * ac.CYCLE_MS else "ok"
