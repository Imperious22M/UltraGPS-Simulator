# UltraGPS positioning simulator

A standalone desktop simulator of the UltraGPS ultrasonic positioning system.
It simulates the six HC-SR04 receivers, the transmitter, the room and the base
station, feeds the tick counts into the **real** `ultragps_position` library,
and shows every positioning algorithm solving the same ping side by side.

![Simulator window](docs/img/main-vector.png)

**[MANUAL.md](MANUAL.md)** explains everything: the windows, the physics, the
algorithms, the algorithm lab, calibration, maps, recording and replay, the
benchmark, and what the simulator suggests about the real system.

## Run it

Python 3.11 or later:

```bash
python3 -m pip install -r requirements.txt     # PyQt6, numpy, scipy
python3 app.py
python3 app.py --preset Harsh --palette paper --seed 1
python3 -m unittest discover -s tests -t .
```

## What is in it

- A main window with dockable panels, menus, shortcuts and a remembered
  layout, drawn in a 2D vector style (dark or light, exports to SVG).
- A model of one ping through the real hardware: beam patterns, spreading,
  air absorption, threshold detection with range walk, loss of signal,
  reflections off walls and obstacles, sound bending round corners, a direct
  sound and its echo interfering, ambient ultrasonic noise, module-to-module
  differences, wind, radio trigger timing and loss, the HC-SR04 timeout and
  the base station's sampling window. An echo scope shows every arrival at
  every receiver.
- The library's OLS, LM and CEP (searching the 15 subsets UltraGPS-Control
  uses, or all of them) next to five simulator-only algorithms: CEP with 4+
  receivers, robust LM, LM with a common offset, RANSAC and a Kalman tracker.
- An algorithm lab that animates the library's CEP subset search, the effect
  of blocked receivers, and the LM optimiser, step by step with the library's
  own equations and numbers.
- Calibration through the real calibration library, with a search for the
  best calibration points in the room; coverage, multipath, geometry and
  accuracy maps; recording, and replay of real base-station tick logs and
  serial-monitor captures; a benchmark with a regression check against a
  recording.

## Relationship to the other UltraGPS repositories

This repository does not modify or depend on any other repository at run
time. These are copied in from
[UltraGPS-Control](https://github.com/Imperious22M/UltraGPS-Control) at commit
`f8f4f73`, and the simulator never edits them (a test checks their SHA-256):

- `ultragps_position/` — the positioning library (`libs/ultragps_position`)
- `ultragps_calibration/` — the calibration library (`libs/ultragps_calibration`)
- `config/config.toml` — receiver positions and calibration

The copies differ from that commit as follows, from earlier work on this
repository:

1. In the positioning library's sanity filter `_filter_receivers`, a distance
   must be greater than 0 to be kept, and the upper bound is a **top-band
   reject**, `max_distance`, a new optional constructor argument; `None` (the
   default) keeps the original bound, the widest receiver spacing. A receiver
   that is off or misses its echo reports 0 ticks, which calibrates to a
   negative distance, and one that loses the signal reports a very high
   value; the original check let the first through.
2. The calibration library takes 2 to 4 calibration points instead of
   exactly two (`cal_points` in config.toml, falling back to `cal_point_1` /
   `cal_point_2`), fitting a least-squares line through all of them; for two
   points it is exactly the original formula. A config.toml with three or
   four points is not fully readable by UltraGPS-Control's unmodified
   calibration library.

To pick up later library changes, copy the folders over again and re-apply
these changes (`diff` against UltraGPS-Control shows them).
