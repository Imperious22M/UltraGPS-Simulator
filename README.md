# UltraGPS positioning simulator

A standalone desktop simulator of the UltraGPS positioning system. It fakes the
six HC-SR04 receivers, feeds their tick counts into the **real**
`ultragps_position` library, and shows every positioning algorithm solving the
same frame side by side.

![Simulator window](screenshot.png)

## Run it

Python 3.11 or later:

```bash
python3 -m pip install -r requirements.txt
python3 app.py
python3 app.py --config path/to/config.toml --max-differential 50 --seed 1
```

It reads receiver positions and calibration from `config/config.toml`, a copy
of UltraGPS-Control's config, so the arena matches the real one. Point
`--config` at another config.toml to simulate a different arena.

## Relationship to the other UltraGPS repositories

This repository is self-contained and does not modify or depend on any other
repository at run time. Two things are copied in, unmodified, from
[UltraGPS-Control](https://github.com/Imperious22M/UltraGPS-Control) at commit
`f8f4f73`:

- `ultragps_position/` — the positioning library (`libs/ultragps_position`)
- `config/config.toml` — receiver positions and calibration

To pick up library changes, copy that folder over again.

## What you can do

| Control | Effect |
|---------|--------|
| Click the arena | The transmitter moves there at the set speed |
| **Circle path** | The transmitter drives a 100 cm circle |
| **Receivers** R0–R5 | Switch a receiver off (it reports 0 ticks, as unused pins do) |
| **Algorithms** | Show or hide each solver's estimate |
| Noise σ | Gaussian range noise in cm |
| Dropouts | Chance a reading is missed and reported as 0 ticks |
| Bad values | Chance a reading is a random wrong range (multipath, EMI) |
| Air temp | Speed-of-sound change against the 20 °C calibration |
| Speed | How fast the transmitter moves |

The panel on the right shows each algorithm's current and RMS error against the
true position, how many frames it failed, which receivers the library's filter
kept, which subset CEP picked, and the raw ticks and distances per receiver.

On the arena, each receiver's colour shows what its sensor did this frame
(green ok, grey off, orange missed or out of range, red bad value), a thick
outline marks the receivers CEP used, and each ring is the measured range —
solid if the library kept that reading, dotted red if its filter dropped it.

## Algorithms shown

All three come from `ultragps_position` and see the same filtered receivers,
because the simulator calls `get_position_full()` exactly as UltraGPS-Control's
`windows/position_window.py` does.

- **OLS** — linear least squares on the squared-range differences. This is the
  seed both other solvers start from.
- **LM** — the OLS seed refined by Levenberg-Marquardt over all kept receivers.
- **CEP** — tries every subset of 3 or more kept receivers (42 subsets with
  all six) and keeps the one with the smallest Circular Error Probable,
  falling back to the last good fix when out of bounds. The library's own
  default stops after 15 subsets; the simulator raises that cap through the
  library's `max_subsets` argument, so the library code itself is unchanged.

## Sensor model

`sensor_model.py` turns a true position into ticks: straight-line range,
speed-of-sound scaling for temperature, Gaussian noise, then the faults above.
Ranges beyond the HC-SR04's 400 cm read as 0 ticks. Distances are converted to
ticks by inverting each receiver's calibration (`tick = (d - intercept) / slope`),
so the library's own `tick * slope + intercept` recovers them.

## Code layout

| File | Role |
|------|------|
| `sensor_model.py` | Physics and faults: position → ticks |
| `solvers.py` | Runs each algorithm in the library on one frame |
| `ultragps_position/` | Copy of the positioning library (unmodified) |
| `simulation.py` | Run state: moves the transmitter, steps sensor + solvers, tracks error |
| `app.py` | Matplotlib window and controls |
| `test_simulator.py` | Headless checks: `python3 -m unittest test_simulator` |
