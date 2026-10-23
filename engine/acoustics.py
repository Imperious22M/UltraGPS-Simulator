"""
Acoustics of one 40 kHz ultrasonic ping, as plain functions.

Standard textbook physics with the HC-SR04's numbers:

* the speed of sound in air, from temperature and humidity (moist air as an
  ideal-gas mixture of dry air and water vapour);
* atmospheric absorption at 40 kHz (ISO 9613-1): about 1.3 dB per metre in a
  room, so ultrasound fades far faster than audible sound;
* spherical spreading: amplitude falls as 1 / distance;
* the beam pattern of a circular piston transducer,
  ``2 J1(ka sin θ) / (ka sin θ)``, sized so the response is −6 dB at a chosen
  half-angle (the 400ST/SR160-type transducers on an HC-SR04 are about ±27°),
  with its nulls filled in as on a real transducer;
* threshold detection: the receiver's comparator fires on the first cycle at
  which the burst, ringing up through the resonant transducers, crosses a fixed
  threshold.  A weak burst crosses a few cycles later ("range walk": each 25 µs
  cycle reads as 0.86 cm farther) or, if it never rings up enough within the
  8-cycle burst, not at all (loss of signal);
* interference: bursts that reach a receiver within a few cycles of each other
  (the direct sound and an echo off a nearby wall) add up in or out of phase,
  so the comparator sees their sum, not each one alone.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.special import j1

FREQ_HZ      = 40_000.0      # HC-SR04 carrier
CYCLE_MS     = 1000.0 / FREQ_HZ
BURST_CYCLES = 8             # HC-SR04 sends an 8-cycle burst
X_6DB        = 2.215089      # 2 J1(x) / x = 0.5  (−6 dB point of a piston)
BACK_LEVEL   = 0.03          # response behind the transducer (−30 dB)
ENVELOPE     = 2.0 * math.sqrt(2.0 / math.pi)
X_ENVELOPE   = 2.4444576     # main lobe meets the side-lobe envelope here, without a step
BACK_FROM    = 90.0          # off-axis angle where the response starts falling to BACK_LEVEL ...
BACK_TO      = 120.0         # ... which it reaches here


def speed_of_sound(temp_c: float, humidity_pct: float = 0.0) -> float:
    """Speed of sound in air, m/s.  Moist air is lighter and a little less
    stiff than dry air: saturated air is about 1.3 m/s faster than dry air at
    20 °C, 2.4 m/s at 30 °C and 5.6 m/s at 45 °C."""
    x     = vapour_fraction(temp_c, humidity_pct)
    gamma = (3.5 * (1.0 - x) + 4.03 * x) / (2.5 * (1.0 - x) + 3.03 * x)
    molar = 28.97 * (1.0 - x) + 18.02 * x
    return (331.3 + 0.606 * temp_c) * math.sqrt(gamma / 1.4 * 28.97 / molar)


def vapour_fraction(temp_c: float, humidity_pct: float, pressure_kpa: float = 101.325) -> float:
    """Mole fraction of water vapour in air (ISO 9613-1 saturation pressure)."""
    T01 = 273.16
    c   = -6.8346 * (T01 / (temp_c + 273.15)) ** 1.261 + 4.6151
    return min(humidity_pct / 100.0 * 10.0 ** c * (101.325 / pressure_kpa), 1.0)


def wavelength_cm(temp_c: float = 20.0, humidity_pct: float = 50.0) -> float:
    return speed_of_sound(temp_c, humidity_pct) * 100.0 / FREQ_HZ


@lru_cache(maxsize=64)
def absorption_db_per_m(temp_c: float, humidity_pct: float, freq_hz: float = FREQ_HZ,
                        pressure_kpa: float = 101.325) -> float:
    """Atmospheric absorption, dB per metre (ISO 9613-1:1993, equations 3-5)."""
    T, T0, pr = temp_c + 273.15, 293.15, 101.325
    pa  = pressure_kpa
    h   = 100.0 * vapour_fraction(temp_c, humidity_pct, pa)   # molar water-vapour concentration, %
    frO = (pa / pr) * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = (pa / pr) * (T / T0) ** -0.5 * (9.0 + 280.0 * h * math.exp(-4.170 * ((T / T0) ** (-1 / 3) - 1.0)))
    f2  = freq_hz ** 2
    return 8.686 * f2 * (
        1.84e-11 * (pr / pa) * (T / T0) ** 0.5
        + (T / T0) ** -2.5 * (0.01275 * math.exp(-2239.1 / T) / (frO + f2 / frO)
                              + 0.1068 * math.exp(-3352.0 / T) / (frN + f2 / frN)))


def piston_ka(half_angle_deg: float) -> float:
    """``ka`` of a piston whose response is −6 dB at ``half_angle_deg`` off axis."""
    return X_6DB / math.sin(math.radians(min(max(half_angle_deg, 1.0), 89.0)))


def directivity(off_axis_deg: float, half_angle_deg: float) -> float:
    """Amplitude response (0..1) of a transducer ``off_axis_deg`` off its axis:
    a circular piston's main lobe, then the upper envelope of its side lobes
    (``2·sqrt(2/π)·x^-1.5``), since real transducers have no deep nulls.
    Beside it (90°) the response falls smoothly, in dB, to ``BACK_LEVEL``
    behind it (120° and beyond)."""
    off = abs((off_axis_deg + 180.0) % 360.0 - 180.0)
    if off >= BACK_TO:
        return BACK_LEVEL
    x = piston_ka(half_angle_deg) * math.sin(math.radians(min(off, BACK_FROM)))
    if x < 1e-9:
        return 1.0
    d = max(float(2.0 * j1(x) / x if x <= X_ENVELOPE else ENVELOPE * x ** -1.5), BACK_LEVEL)
    if off <= BACK_FROM:
        return d
    f = (off - BACK_FROM) / (BACK_TO - BACK_FROM)
    return d ** (1.0 - f) * BACK_LEVEL ** f                # straight line in dB


def spreading(length_cm: float, temp_c: float, humidity_pct: float) -> float:
    """Amplitude after ``length_cm`` of travel, relative to 1 m: 1/r spreading
    times atmospheric absorption over the whole length (so 1 m itself loses
    about 1.3 dB)."""
    length_cm = max(length_cm, 5.0)                     # transducer near field
    loss_db   = absorption_db_per_m(temp_c, humidity_pct) * length_cm / 100.0
    return (100.0 / length_cm) * 10.0 ** (-loss_db / 20.0)


def knife_edge_db(clearance_cm: float, d1_cm: float, d2_cm: float, wavelength: float) -> float:
    """Loss (dB) of sound bending round an edge ``clearance_cm`` off the straight
    line, ``d1``/``d2`` from each end (ITU-R P.526 single knife edge)."""
    v = clearance_cm * math.sqrt(2.0 / wavelength * (d1_cm + d2_cm) / max(d1_cm * d2_cm, 1e-9))
    if v <= -0.78:
        return 0.0
    return 6.9 + 20.0 * math.log10(math.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)


def detect_cycle(ratio: float, rise_cycles: float, burst_cycles: int = BURST_CYCLES) -> int | None:
    """First cycle (1, 2, ...) at which a burst whose full amplitude is ``ratio``
    times the detection threshold crosses it, or ``None`` if it never does.

    The burst envelope through the resonant transducers rises as
    ``1 − exp(−t / rise_cycles)`` and stops rising when the drive stops after
    ``burst_cycles``."""
    if not ratio > 1.0:                                 # also NaN
        return None
    t = -rise_cycles * math.log(1.0 - 1.0 / ratio)
    k = max(1, math.ceil(t - 1e-9))
    return k if k <= burst_cycles else None


def envelope(cycles, rise_cycles: float, burst_cycles: int = BURST_CYCLES):
    """Envelope (0..1 of full amplitude) of a burst ``cycles`` after it starts:
    it rings up as ``1 − exp(−t / rise_cycles)`` while driven, then rings down
    at the same rate."""
    t    = np.asarray(cycles, dtype=float)
    peak = 1.0 - math.exp(-burst_cycles / rise_cycles)
    up   = 1.0 - np.exp(-np.clip(t, 0.0, None) / rise_cycles)
    down = peak * np.exp(-np.clip(t - burst_cycles, 0.0, None) / rise_cycles)
    return np.where(t <= 0.0, 0.0, np.where(t <= burst_cycles, up, down))


def first_detection(start_ms, ratios, rise_cycles: float,
                    burst_cycles: int = BURST_CYCLES) -> tuple[float, int] | None:
    """When a comparator fires on bursts that may overlap, and on which.

    Burst ``i`` starts at ``start_ms[i]`` with full amplitude ``ratios[i]``
    times the threshold, and carrier phase set by its start (its path length).
    Bursts that overlap add as phasors; the comparator fires on the first peak
    of the summed carrier at which their summed envelope is over the threshold.
    A burst on its own fires exactly as ``detect_cycle`` says.  Returns
    ``(time ms, index of the burst contributing most then)``, or ``None``."""
    start  = np.asarray(start_ms, dtype=float)
    ratio  = np.asarray(ratios, dtype=float)
    order  = np.argsort(start, kind="stable")
    peak   = 1.0 - math.exp(-burst_cycles / rise_cycles)
    keep   = order[ratio[order] * peak >= 0.02]        # fainter than −34 dB: no effect
    span   = (burst_cycles + 3.0 * rise_cycles) * CYCLE_MS   # rung down to 5 % by then
    i, n   = 0, len(keep)
    while i < n:
        j = i + 1
        while j < n and start[keep[j]] - start[keep[j - 1]] < span:
            j += 1
        group = keep[i:j]
        k     = detect_cycle(ratio[group[0]], rise_cycles, burst_cycles)
        alone = None if k is None else start[group[0]] + k * CYCLE_MS
        if alone is not None and (len(group) == 1 or start[group[1]] >= alone):
            return alone, int(group[0])                 # fired before anything else arrived
        if len(group) > 1 and np.sum(ratio[group]) * peak >= 1.0:
            hit = _sum_detection(start[group], ratio[group], rise_cycles, burst_cycles)
            if hit is not None:
                return hit[0], int(group[hit[1]])
        i = j
    return None


def _sum_detection(start, ratio, rise_cycles, burst_cycles):
    """``first_detection`` within one group of overlapping bursts: every peak
    of the summed carrier, found by following its phase from each cycle of the
    first burst, and the first of them at which the summed envelope is over
    the threshold."""
    t0     = start[0]
    m      = np.arange(1, math.ceil((start[-1] - t0) / CYCLE_MS + burst_cycles) + 1)
    phasor = np.exp(-2j * math.pi * (start - t0) / CYCLE_MS)
    t      = t0 + m * CYCLE_MS
    for settle in range(3):
        g = envelope((t[:, None] - start[None, :]) / CYCLE_MS, rise_cycles, burst_cycles) * ratio
        s = g @ phasor
        if settle < 2:                                  # the peak moves with the sum's phase
            t = t0 + (m - np.angle(s) / (2.0 * math.pi)) * CYCLE_MS
    over = np.flatnonzero(np.abs(s) >= 1.0)
    if not len(over):
        return None
    k = over[0]
    return float(t[k]), int(np.argmax(g[k]))
