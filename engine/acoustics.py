"""
Acoustics of one 40 kHz ultrasonic ping, as plain functions.

Standard textbook physics with the HC-SR04's numbers:

* the speed of sound in air, from temperature and humidity;
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
  8-cycle burst, not at all (loss of signal).
"""

from __future__ import annotations

import math

from scipy.special import j1

FREQ_HZ      = 40_000.0      # HC-SR04 carrier
CYCLE_MS     = 1000.0 / FREQ_HZ
BURST_CYCLES = 8             # HC-SR04 sends an 8-cycle burst
X_6DB        = 2.215089      # 2 J1(x) / x = 0.5  (−6 dB point of a piston)
BACK_LEVEL   = 0.03          # response behind the transducer (−30 dB)
ENVELOPE     = 2.0 * math.sqrt(2.0 / math.pi)
X_ENVELOPE   = 2.5           # main lobe meets the side-lobe envelope here


def speed_of_sound(temp_c: float, humidity_pct: float = 0.0) -> float:
    """Speed of sound in air, m/s (humidity adds about 0.6 m/s from dry to saturated)."""
    return 331.3 + 0.606 * temp_c + 0.0124 * humidity_pct


def wavelength_cm(temp_c: float = 20.0, humidity_pct: float = 50.0) -> float:
    return speed_of_sound(temp_c, humidity_pct) * 100.0 / FREQ_HZ


def absorption_db_per_m(temp_c: float, humidity_pct: float, freq_hz: float = FREQ_HZ,
                        pressure_kpa: float = 101.325) -> float:
    """Atmospheric absorption, dB per metre (ISO 9613-1:1993, equations 3-5)."""
    T, T0, T01, pr = temp_c + 273.15, 293.15, 273.16, 101.325
    pa  = pressure_kpa
    c   = -6.8346 * (T01 / T) ** 1.261 + 4.6151
    h   = humidity_pct * 10.0 ** c * (pr / pa)          # molar water-vapour concentration, %
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
    (``2·sqrt(2/π)·x^-1.5``), since real transducers have no deep nulls."""
    off = abs((off_axis_deg + 180.0) % 360.0 - 180.0)
    if off >= 90.0:
        return BACK_LEVEL
    x = piston_ka(half_angle_deg) * math.sin(math.radians(off))
    if x < 1e-9:
        return 1.0
    d = 2.0 * j1(x) / x if x <= X_ENVELOPE else ENVELOPE * x ** -1.5
    return max(float(d), BACK_LEVEL)


def spreading(length_cm: float, temp_c: float, humidity_pct: float) -> float:
    """Amplitude after ``length_cm`` of travel, relative to 1 m: 1/r spreading
    times atmospheric absorption."""
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
    if ratio <= 1.0:
        return None
    t = -rise_cycles * math.log(1.0 - 1.0 / ratio)
    k = max(1, math.ceil(t - 1e-9))
    return k if k <= burst_cycles else None
