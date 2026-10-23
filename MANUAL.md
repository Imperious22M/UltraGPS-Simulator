# UltraGPS Simulator — Manual

This is the manual for the `max-effort` rewrite of the UltraGPS positioning
simulator. It covers what was rebuilt and why, how to use every window, the
physics the simulator now models, the positioning algorithms it compares, and
how each part can help with the real system.

![The simulator](docs/img/main-vector.png)

## Contents

1. [What changed](#1-what-changed)
2. [Running it](#2-running-it)
3. [The main window](#3-the-main-window)
4. [How a ping is simulated](#4-how-a-ping-is-simulated)
5. [Positioning algorithms](#5-positioning-algorithms)
6. [The algorithm lab](#6-the-algorithm-lab)
7. [Calibration](#7-calibration)
8. [Maps](#8-maps)
9. [Recording and replaying real data](#9-recording-and-replaying-real-data)
10. [Benchmark results](#10-benchmark-results)
11. [What the simulator says about the real system](#11-what-the-simulator-says-about-the-real-system)
12. [Code map and how to extend it](#12-code-map-and-how-to-extend-it)
13. [Assumptions and limits](#13-assumptions-and-limits)

---

## 1. What changed

| | Before (`main`) | Now (`max-effort`) |
|---|---|---|
| Toolkit | One matplotlib figure, widgets placed by coordinates, a second figure for calibration | PyQt6 (the same toolkit as UltraGPS-Control): a main window with dockable, floatable, tabbable panels, menus, a toolbar, keyboard shortcuts, a remembered layout, and separate tool windows |
| Look | Default matplotlib plot | A 2D vector-display style: thin glowing lines on a dark field (or ink on paper), outlined markers, hatched obstacles, monospace labels; exports to SVG |
| Sound | Straight-line range plus Gaussian noise and random faults | A model of one ping through the real hardware: beam patterns, spreading, air absorption, threshold detection with range walk, loss of signal, reflections off walls and obstacles, sound bending round corners, wind, radio trigger timing and loss, the HC-SR04 timeout and the base station's sampling window |
| Room | None | Walls and obstacles (`config/world.toml`), drawn and edited in the arena |
| Algorithms | OLS, LM, CEP (library) and CEP4 | The same, plus robust LM, LM with a common offset, RANSAC and a Kalman tracker, all outside the library |
| Seeing inside the algorithms | Error numbers | The algorithm lab: the library's CEP subset search, the effect of blocked receivers, and the LM optimiser, step by step and animated, with the equations and numbers |
| Seeing inside the sensor | Status colour | The echo scope: every sound arrival at every receiver, its strength against the detection threshold, and the moment each echo line dropped |
| Analysis | — | Coverage, geometry (HDOP), multipath and Monte-Carlo accuracy maps; a benchmark of every algorithm; recording to CSV; replay of real base-station tick logs |
| Libraries | `ultragps_position`, `ultragps_calibration` | **Unchanged.** A test checks their SHA-256 hashes. New ideas live in `engine/` |

The positioning and calibration libraries are exactly the copies on `main`
(the ones from UltraGPS-Control with the changes you asked for earlier). The
simulator only calls them, and the algorithm lab watches them by wrapping
their methods for the length of one call; nothing in them was edited.

## 2. Running it

Python 3.11 or later.

```bash
python3 -m pip install -r requirements.txt     # PyQt6, numpy, scipy
python3 app.py
```

Options:

```bash
python3 app.py --preset Ideal           # Ideal, Realistic (default) or Harsh physics
python3 app.py --palette paper          # light theme
python3 app.py --config path/to/config.toml --sensors path/to/sensors.toml --world path/to/world.toml
python3 app.py --seed 1                 # repeatable randomness
python3 app.py --screenshot shot.png --frames 60 --size 1600x1000   # render and exit
```

Command-line tools (no window):

```bash
python3 -m engine.benchmark --frames 300      # every algorithm under ten standard conditions
python3 -m engine.cal_points --apply          # best calibration points (see section 7)
python3 -m unittest discover -s tests -t .    # the test suite
```

On a machine without a display, set `QT_QPA_PLATFORM=offscreen`.

## 3. The main window

TODO: tour (filled in after the GUI is final).

## 4. How a ping is simulated

The simulator follows one ping through the real system, in the order the
firmware does it (`UltraGPS-Arduino/base-station/src/main.cpp`):

1. **Trigger.** The base station fires all six receiver HC-SR04s at once
   (t = 0). Each one raises its echo line when it is ready to listen
   (`rx_ready_ms`, 3 ms; the firmware notes the sensor "needs about 3 ms to
   send the burst and go into receiver mode").
2. **Transmitter.** The transmitter fires `burst_lead_ms` later on its radio
   trigger. The firmware asks for 2 ms (`SONIC_BURST_DELAY`); the calibration
   says about 4 ms in practice (an intercept of about −139 cm at 0.96 cm per
   tick is 145 ticks, about 4 ms), so the default is 4 ms. The radio link has
   no retries, so a trigger can be lost (`radio_loss`), and its timing can be
   off (`sync_offset_us`) or jitter (`sync_jitter_us`) — the same for every
   receiver.
3. **Sound paths.** The sound reaches each receiver along every path the room
   allows: straight, bounced off walls and obstacles (up to two bounces), or
   bent round an obstacle's corner when the straight line is blocked.
4. **Strength.** Each arrival's strength over the detection threshold is

   `strength = SNR(1 m) × beam_tx(angle) × beam_rx(angle) × spreading × absorption × reflection × diffraction`

   with every term from section 4.1, and a little random turbulence
   (`amp_jitter_db`).
5. **Detection.** The receiver's comparator fires on the first 40 kHz cycle at
   which an arrival's burst, ringing up through the resonant transducers,
   crosses its threshold. A strong arrival is caught on the first cycle; a
   weak one a few cycles later (each cycle is 25 µs, 0.86 cm); one too weak
   never. The receiver latches the **first** arrival it detects, so if the
   direct sound is blocked or too weak, it reads a reflection instead.
6. **Timeout.** If nothing is detected, the HC-SR04 gives up after its timeout
   (`timeout_ms`, 38 ms) and drops its echo line anyway: a very high value.
7. **Sampling.** The base station counts loop iterations until each echo line
   drops; it stops sampling at 80 % of the frame (`window_ms`, 80 ms at
   10 Hz). A line still high then, or a pin with nothing on it, reads 0.
8. **Ticks.** The time becomes ticks through each receiver's true response
   in `config/sensors.toml`. The sampling loop starts at a random phase to
   the sound (`tick_dither`), so readings wobble by ±1 tick.
9. **Faults.** Optional injected faults go on top: noise, dropouts, lost
   signal, bad values.

### 4.1 The physical effects

| Effect | Model | Control | What you see | Why it matters on the real system |
|---|---|---|---|---|
| Speed of sound | 331.3 + 0.606·T + 0.0124·RH m/s | Air temperature, humidity; calibration temperature | Every range scales by c_cal / c | 10 °C warmer than at calibration = 1.7 % long, 5 cm at 3 m |
| Absorption | ISO 9613-1 at 40 kHz (about 1.3 dB/m in a room) | Temperature, humidity | Far receivers get weaker, so detect later | Ultrasound fades 250 times faster than audible sound; humidity matters |
| Spreading | 1/r amplitude | — | Strength falls 6 dB per doubling of distance | Sets the useful range |
| Beam patterns | Circular piston, −6 dB at ±27.5° (400ST/SR160-type transducers), nulls filled in, −30 dB behind | Beam patterns on/off, receiver beam, transmitter mode (omni, heading, fixed) | Off-axis receivers detect later, then not at all; lobes drawn on the arena | Which receivers can hear which parts of the arena depends on how they are aimed |
| Threshold detection, range walk | Envelope 1 − e^(−t/τ), τ = 6 cycles, 8-cycle burst; first cycle over the threshold | SNR at 1 m, ring-up time, turbulence | Weak signals read up to several cm long; status `walk` | A bias that depends on position and angle, which a straight-line calibration cannot remove |
| Loss of signal | No cycle crosses the threshold within the burst | — | Status `lost`: the timeout's high value | The high values the top-band reject exists for |
| Reflections (multipath) | Image sources, up to 2 bounces, reflectivity per surface | Reflections, bounces, wall reflectivity, obstacles | Status `echo`: reads long by the extra path | When the direct sound is blocked or out of the beam, a wall echo is caught instead: large, one-sided errors |
| Diffraction | Shortest path round an obstacle's corners, knife-edge loss per corner (ITU-R P.526) | Diffraction | Status `bent`: a little long and weak | Sound does reach a receiver behind a small obstacle, late and weak |
| Wind | Each leg's speed c + w·û | Wind speed, direction | Ranges upwind long, downwind short | A fan or air conditioning skews positions |
| Trigger timing | Common offset and jitter | Sync offset, sync jitter | All ranges shift together | A common error that the BIAS solver removes |
| Radio loss | Transmitter never fires | Radio loss | Every receiver times out: status `radio` | One lost frame; the solvers must hold |
| Timeout vs window | 38 ms timeout inside an 80 ms window → high value; a clone with a 200 ms timeout → 0 | Timeout, window | `lost` (high) or `missed` (0) | Explains whether a dead reading shows as 0 or as a huge distance |
| Listen delay | Arrivals before `rx_ready_ms` are missed | Receiver ready, burst lead | Close to a receiver, the direct sound can arrive before it listens | A blind zone if the transmitter fires too early |
| Crosstalk | Receivers' own bursts are heard by the others (and as their own echoes) | Crosstalk | Status `crosstalk`: reads short | If the receivers' transmit transducers are not covered, a facing receiver's burst arrives before the transmitter's and is caught first |
| Survey error | True receiver positions differ from config.toml | Survey error σ, dragging receivers | Ghost receivers on the arena | Tape-measure errors in tower positions become position errors |
| Hard field of view | Nothing heard outside ±15° | Field-of-view cut-off | Status `angle`, 0 ticks; dashed red lines | The earlier simple model, kept for comparison |
| Near noise | Extra noise within 16 cm | Near-receiver noise | Status `near` | The earlier simple model, kept for comparison |

Presets: **Ideal** turns everything off except injected faults (the first
simulator's model). **Realistic** is the default. **Harsh** lowers the signal
margin, adds turbulence, second bounces, 30 µs trigger jitter, 2 % radio loss
and a light draught.

## 5. Positioning algorithms

All algorithms see the same frame. OLS, LM and CEP come from the
`ultragps_position` library, called exactly as UltraGPS-Control's position
window calls it (`get_position_full`). The others are simulator-only and live
in `engine/solvers.py`; they reuse the library's OLS seed and range equation.

Notation: receiver *i* at **r**ᵢ reports distance dᵢ; the residual of a fix
**p** is rᵢ = |**p** − **r**ᵢ| − dᵢ.

| Key | Name | How it works | Good at | Weak at |
|---|---|---|---|---|
| OLS | Linear least squares (library) | Subtract the first receiver's circle equation from the others: 2(x₁ − xᵢ)x + 2(y₁ − yᵢ)y = dᵢ² − d₁² − (xᵢ² + yᵢ²) + (x₁² + y₁²); solve in one step | Instant; no starting guess | Squares the ranges, so errors on long ranges count extra; the first receiver is special |
| LM | Levenberg–Marquardt (library) | Minimise Σrᵢ² from the OLS seed (scipy `least_squares(method="lm")`) | Best when every reading is good | One bad reading pulls the whole fix |
| CEP | CEP best subset (library) | Fit every subset of 3+ kept receivers (all 42 with six); weight each residual by wᵢ = 1/(\|rᵢ\| + 0.1); Σ = (JᵀWJ)⁻¹; CEP = 0.59·√(σx² + σy²)·√2; lowest wins; outside the receivers' box it holds the last good fix | Drops a bad receiver when the rest agree | A 3-receiver fit has near-zero residuals, so weights near 10 and a tiny CEP: it prefers 3-subsets even when one of the three is wrong; freezes outside the box |
| CEP4 | CEP, 4+ receivers | Subsets of 4 or more only; Σ = s²(JᵀJ)⁻¹ with s² = Σrᵢ²/(m − 2); CEP = 0.59(σx + σy); no box | A bad reading can't hide in an exact 3-fit | Needs 4 good readings |
| ROBUST | Robust LM | Minimise Σρ(rᵢ) with the soft-L1 loss ρ(r) = 2c²(√(1 + (r/c)²) − 1), c = 5 cm: quadratic for small residuals, linear for large | One smooth fit that discounts outliers; no subsets | Two bad readings that agree can still pull it |
| BIAS | LM + common offset | Minimise Σ(\|**p** − **r**ᵢ\| + b − dᵢ)² over x, y, b | Removes trigger timing error (radio/PLL), which shifts every range equally; reports b | Needs 4 receivers; slightly noisier when there is no offset |
| RANSAC | RANSAC consensus | Fit every 3-receiver subset; count the receivers within 8 cm of each fit; refit on the largest agreeing set | Throws out bad readings decisively | Needs most readings to be good |
| EKF | Kalman tracker | State x, y, vx, vy; constant velocity with random acceleration; each range updates the state on its own; a range more than 3.5σ from the prediction is rejected | Smooth, uses 1 or 2 receivers when that's all there is, coasts through dropouts, rejects spikes by the motion model | Lags hard turns; needs a few frames to lock on |

Every algorithm uses the receivers the library's filter kept (inside the
bounds, no jump over 50 cm). Set **Filter for simulator algorithms** to
*bounds only* to let their own outlier handling do the work instead; the
library's jump filter rejects the next good reading after a bad one, which
hurts them. The EKF always uses bounds only: its gate replaces the jump filter.
Like the library's solvers, the simulator algorithms hold their last good fix
when they cannot solve (shown hollow on the arena and counted as a failure).

TODO: lab, calibration, maps, recording, benchmark, findings, code map, limits.
