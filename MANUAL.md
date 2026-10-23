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

The arena sits in the middle. Every panel round it is a dock: drag its title
to move it to another edge, drop it on another dock to stack them as tabs,
double-click the title to float it as its own window (on a second monitor, for
instance), or close it and bring it back from **View**. The layout and the
palette are remembered between runs; **View > Reset layout** puts everything
back. The algorithm lab, calibration and maps are separate windows, so they
can sit next to the arena instead of replacing it.

![Light palette](docs/img/main-paper.png)

### 3.1 The arena

The arena draws what the sound did on the latest ping, in layers you can switch
off under **View > Arena layers**:

| Layer | What it shows |
|---|---|
| Grid | 25 cm and 100 cm lines (coarser when zoomed out), axes in cm, a scale bar |
| Walls and obstacles | The room's walls (ticked on the outside) and obstacles (hatched) |
| Beam patterns | Each receiver's beam pattern as a polar plot of its sensitivity (and the transmitter's, when it is not omnidirectional); with the hard field of view on, dashed lines at its limits |
| Measured ranges | A ring of the measured distance round where the library *believes* each receiver is; dotted when the library's filter dropped that reading |
| Sound paths | The path the detected sound really took: straight, bounced off a wall or obstacle (an echo), bent round a corner, or another receiver's burst; a dotted red line where the line of sight is blocked |
| Wavefronts | The ping slowed right down: the wave spreading from the transmitter (shadowed by obstacles), echoes spreading from their image sources, each receiver's ring flashing when its sound arrives |
| Trails | The last 40 fixes of each algorithm and the true path |
| Estimates | Each algorithm's fix, one marker shape and colour each; hollow and dim when it has no fix this frame and holds its last one |
| Uncertainty ellipses | The 2σ ellipse of each fix's covariance, where the algorithm has one |
| Labels | Receiver names, obstacle names, readouts |
| Calibration points | P1 to P4 |
| Legend | Each shown algorithm's error this frame |
| Map overlay | A map from the maps window (section 8) |

Hover over a receiver for its reading in full (status, tick, distance, the
arrival it detected and its strength), or over a fix for the algorithm's
details. The status line under the window shows the cursor's coordinates.

The mouse works in one of three tools (toolbar or **Tools**):

| Tool | Key | Left mouse | Also |
|---|---|---|---|
| Move | M | Click: send the transmitter there | |
| Obstacles | O | Drag: draw a rectangular obstacle | Right-click an obstacle: delete it |
| Receivers | E | Drag a receiver: move where it *really* is (the library keeps the config.toml position, so this is a survey error) | Wheel over a receiver: turn it 5° |

In every tool the wheel zooms, right- or middle-drag pans, double-click fits
the room and Esc goes back to Move. **File > Save world** writes the walls and
obstacles to a `world.toml` for next time (`--world` loads one);
**File > Export arena as SVG** saves the arena as a vector drawing.

### 3.2 The docks

| Dock | What is in it |
|---|---|
| **Scenario** | Motion (click to go, circle, figure eight, lawnmower, random waypoints), speed, path radius, timestep (0.1 s is the base station's 10 Hz), time scale, and a readout of the transmitter and the frame rate |
| **Receivers** | One row per receiver: on, blocked or off; what its sensor did this ping in words; the tick it reported, the distance the library makes of it, the true distance and the error; **K** if the library's filter kept it and **C** if it is in the CEP's best subset; the detected sound's strength over the threshold in dB. Under it, **Survey error σ** moves the real receivers away from where config.toml says they are, and **Re-roll** draws a new error |
| **Algorithms** | One row per algorithm: show it or not; its error now, RMS and 95th percentile over the last 300 frames, and failed frames; a note (CEP's subset and CEP, BIAS's offset, RANSAC's agreement). Under it, the simulator algorithms' settings (section 5) and **Reset stats** |
| **Environment** | A preset, then every physical effect of section 4.1 in five groups: Air, Transducers, Detection, Propagation, and Timing & electronics. Every control has a tooltip saying what it does |
| **Faults & filter** | Injected faults (range noise, dropouts, lost signal, bad values and how bad), and the library's own filter: top-band reject and jump filter |
| **Echo scope** | One row per receiver, like a logic analyser (below) |
| **Error** | Each shown algorithm's error over time; click **lin / log** to switch the scale, hover for the values at a moment |

A receiver that is **blocked** has something standing in its line of sight:
the direct sound cannot reach it, but a reflection or sound bending round the
obstacle still can, so it may read long or lose the signal. A receiver that is
**off** is unplugged: the base station reads 0 for it.

### 3.3 The echo scope

The echo scope shows *why* each receiver read what it did. Each row is one
receiver's echo line on the time axis of the ping (milliseconds after the
base station's trigger). The line goes high when the receiver starts
listening and drops when it detects sound; that moment is the reading. Each
stem is a sound arrival, as tall as its strength over the detection threshold
and coloured by path: direct, reflection, bent round an obstacle, or another
receiver's burst. The detected arrival is drawn bold, arrivals too weak to
detect hollow, and a small triangle marks when the direct sound would arrive.

![Obstacles](docs/img/obstacle.png)

Here a pillar stands between the transmitter and R1 and a box in front of R2.
R1 has no line of sight, but the sound bends round the pillar (status
`bent`): its reading is 4 cm long and much weaker. R2 hears nothing strong
enough at all: its echo line stays high until the HC-SR04's 38 ms timeout
(status `lost`, a reading of over 10 m), which the library's top-band reject
then drops. The other receivers detect the direct sound first; the reflections
off the walls arrive later and are ignored.

### 3.4 Menus and keys

| Key | Action |
|---|---|
| Space | Run or pause |
| Right | Pause and step one frame |
| R | Reset statistics and solver memory |
| F | Fit the arena to the window |
| C | Put the transmitter in the centre |
| M, O, E | Move, Obstacles and Receivers tools |
| Ctrl+0 … Ctrl+5 | Switch receiver R0 … R5 on or off |
| Ctrl+L | Algorithm lab |
| Ctrl+K | Calibration |
| Ctrl+M | Maps |
| Ctrl+R | Start or stop recording |
| Ctrl+O | Replay a tick log |
| Ctrl+S | Save the world |
| Ctrl+P | Save a screenshot |
| F1 | This manual |

**File** has replay, recording, saving the world, clearing obstacles, SVG
export and screenshots; **Simulation** has run control, the motion modes and
the physics presets; **View** has every dock, the arena layers, the palette
(vector or paper), fit and reset layout; **Tools** has the three tool windows
and the arena tools; **Help** has this manual and the list of shortcuts.

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
| BIAS | LM + common offset | Minimise Σ(\|**p** − **r**ᵢ\| + b − dᵢ)² over x, y, b | Removes trigger timing error (radio and processing delay), which shifts every range equally; reports b | Needs 4 receivers; one bad reading is spread over the offset and the fix alike; an offset over 1 m is refused as implausible |
| RANSAC | RANSAC consensus | Fit every 3-receiver subset; count the receivers within 8 cm of each fit; refit on the largest agreeing set | Throws out bad readings decisively | Needs most readings to be good |
| EKF | Kalman tracker | State x, y, vx, vy; constant velocity with random acceleration; each range updates the state on its own; a range more than 3.5σ from the prediction is rejected | Smooth, uses 1 or 2 receivers when that's all there is, coasts through dropouts, rejects spikes by the motion model | Lags hard turns; needs a few frames to lock on |

Every algorithm uses the receivers the library's filter kept (inside the
bounds, no jump over 50 cm). In the Algorithms dock, set **Receivers used**
to *Bounds only* to give the simulator algorithms every plausible reading and
let their own outlier handling do the work instead; the library's jump
filter rejects the next good reading after a bad one, which hurts them. The EKF always uses bounds only: its gate replaces the jump filter.
Like the library's solvers, the simulator algorithms hold their last good fix
when they cannot solve (shown hollow on the arena and counted as a failure).

## 6. The algorithm lab

**Tools > Algorithm lab** (Ctrl+L) opens a window for watching the
algorithms work on one scenario, step by step and animated, with the
library's own equations and numbers. A scenario is a transmitter position and
what each receiver reports. **Load latest frame** takes a snapshot of the
running simulation; or click the CEP or Blocking canvas to put the
transmitter somewhere and give each receiver an override in the **Scenario**
panel: *Clean*, *Blocked* (no line of sight, so it reads whatever sound
reaches it), *+offset* (an extra error you set with the slider) or *Off*.
**Previous frame was clean** decides what the library's jump filter compares
against: checked, the readings have just changed (as on the first frame after
a receiver is blocked); unchecked, they have been like this for a while.

Lab readings come from a private copy of the simulated hardware with its
random parts turned off, so a scenario gives the same readings every time.
Every number in the lab comes from `engine/lab.py`, which runs the **real**
library on a private copy and records what it does by wrapping its methods
for the length of one call. Nothing in the library is changed.

The player at the bottom works the same on every tab: first, previous,
play/pause, next and last step, and a speed slider. The keys are Space
(play/pause), Left and Right (step), Home and End.

### 6.1 CEP subsets

![CEP subsets](docs/img/lab-cep.png)

This tab replays the library's CEP search (`_find_best_subset`) one subset at
a time, in the library's own order. The first step is the sanity filter
(`_filter_receivers`): which receivers pass the bounds check and the jump
check, and why the others don't. Then, for each subset, the side panel shows
the library's calculation with the numbers filled in:

1. **Seed** — `_ols`: the linearised equations solved by least squares.
2. **Fit** — `_nlls`: `least_squares(method="lm")` from the seed.
3. **Residuals and weights** — rᵢ = |**p** − **R**ᵢ| − dᵢ and
   wᵢ = 1 / (|rᵢ| + 0.1), with the Jacobian rows Jᵢ = (**p** − **R**ᵢ) / |**p** − **R**ᵢ|.
4. **Covariance** — Σ = (JᵀWJ)⁻¹, W = diag(w).
5. **CEP** — 0.59 · √(σx² + σy²) · √2, and whether it beats the best so far.

The arena draws the subset's circles, the seed, the fit, the residuals, the
1σ ellipse, the CEP circle and the best fix so far. Under it a bar chart
shows every subset's CEP (log scale) with a dot grid of which receivers each
one uses, and the table on the right lists every subset's CEP, fix and true
error. The last step is the bounds check (`_validate_position`): a fix outside
the receivers' box is replaced by the last good one.

In the picture R2 reads 25 cm long. Most subsets that include R2 have large
residuals, small weights and a large CEP, and one without it wins. The table
also shows the library's preference for three-receiver subsets: their fits
have only one spare equation to absorb noise, so their residuals are tiny,
the weights near 10 and the CEP small. With 1 cm of noise and every reading
good, the winner is a 3-subset three times in four, which throws away three
good receivers. That is why the library's CEP is noisier than LM when every
reading is good (section 10).

### 6.2 Blocking

![Blocking](docs/img/lab-blocking.png)

**Fixes** shows how each algorithm copes with the scenario: every fix in
turn, with its error, the receivers it used, and a ranking. Here R3 is blocked
and reads a wall echo 68 cm long. The library's filter keeps it (it is within
bounds and has not jumped), so OLS, LM and BIAS, which use every kept
receiver, are pulled 16 to 35 cm off; the robust fit weighs R3 down; CEP,
CEP4 and RANSAC leave it out.

![Sweep](docs/img/lab-sweep.png)

**Sweep** pushes one receiver's reading from 0 to the offset you choose and
plots every algorithm's error against it (small multiples under the arena),
as when a receiver starts hearing a longer and longer reflection. The arena
marks where the library's CEP changes subset ("CEP drops R1 at +4 cm") and
draws the swept receiver's range as it grows. OLS, LM and BIAS grow steadily
with the offset; CEP, CEP4 and RANSAC stay flat once they drop the receiver;
the robust fit levels off at a couple of cm. The filter is not re-run along the sweep, because a
reading that stays wrong passes the jump check on every frame after the
first.

### 6.3 LM optimiser

![LM optimiser](docs/img/lab-lm.png)

This tab shows the surface Levenberg-Marquardt walks down: contours of the RMS
range residual over the arena, with the global minimum. Click anywhere to
start LM there (or **Start at OLS seed**, as the library does), and the player
steps through every point MINPACK evaluates: accepted steps (arrows), rejected
trials (crosses) and, with **Jacobian probes** on, the tiny moves it makes to
estimate the Jacobian by finite differences. The side panel shows the
residuals, J, JᵀJ and Jᵀr at the current iterate and the update
(JᵀJ + λD)δ = −Jᵀr; the table lists every evaluation, and the inset plots the
RMS residual per evaluation.

- **Receivers** — fit every kept receiver, or any three of them.
- **3 in a line** — fit three receivers that stand in a line (R0, R1 and R2,
  or R3, R4 and R5). Their distances cannot tell the transmitter from its
  mirror image, so the surface has a second minimum, and where LM ends depends
  on where it starts.
- **Many starts** — run LM from a 6 × 6 grid of starting points at once and
  colour each by the minimum it ends in (its basin of attraction).

With all six receivers the surface is a single smooth bowl: from every one
of 143 starts spread over and beyond the arena, LM reaches the right fix in a
handful of steps, so the starting guess hardly matters. With three receivers
in a line it does. Their equations say nothing about the side, so the OLS
seed lands on the middle line (x = 0), which happens to be on the inside; a
start beyond the line converges to the mirror image, with a zero residual and
no sign that anything is wrong.

## 7. Calibration

![Calibration](docs/img/calibration.png)

Two files hold each receiver's tick-to-distance line,
distance = tick × slope + intercept:

| File | What it is | Used by |
|---|---|---|
| `config/sensors.toml` | The simulated hardware's **true** response | The simulated sensors |
| `config/config.toml` | The **calibration**, what the system believes | The positioning library |

When they differ, every distance the library sees is off, as on the real
system. **Tools > Calibration** (Ctrl+K) works through calibrating in four
steps, top left to bottom right:

1. **Calibration points.** Choose 2, 3 or 4 points: **Search best points**
   finds the set that gives the smallest calibration error over the arena,
   **Pick on arena** lets you click them on the main arena, or type them in the
   table. The *arena score* is the RMS distance error (cm) per cm of range
   noise that the set would cause over the arena; lower is better. **Use
   these points** writes them to config.toml (with each receiver's distances
   to them).
2. **Calibration run.** **Calibrate** runs the calibration library's own
   `run_full_calibration()` on the simulated sensors: the transmitter is
   parked on each point in turn and pinged until every receiver's most
   frequent tick has been seen enough times; the library fits each receiver's
   line, writes it to config.toml, and the simulation reloads it straight
   away. The current physics, faults, obstacles and receiver states all apply,
   so you can see how each turns into calibration error. The simulation waits
   while it runs.
3. **Offsets.** Each receiver's slope and intercept next to its true values,
   and the distance error they cause at 100 and 300 cm. Edit them and **Save &
   apply**, **Reload from file**, or **Use true values** (a perfect
   calibration, to separate calibration error from everything else).
4. **Response.** One receiver's true and calibrated lines, the modal tick it
   read at each point, and the error between the lines underneath.

`python3 -m engine.cal_points` runs the same search from the command line and
checks each set by calibrating it many times on the simulated sensors (here
100 times, with the default physics and 1 cm noise; the distance error is
the RMS error of the calibrated lines over the arena):

| Set | Points (cm) | Slope error | Intercept error | Distance error |
|---|---|---|---|---|
| current | (0, 76), (−76, 0) | 3.10 % | 11.4 cm | 3.30 cm |
| best 2 | (−82, −152), (78, 153) | 0.74 % | 3.5 cm | 1.86 cm |
| best 3 | (−97, −92), (58, 123), (93, −147) | 0.76 % | 2.9 cm | 1.27 cm |
| best 4 | (−97, −142), (93, 98), (93, −147), (−97, 98) | 0.48 % | 1.5 cm | 1.28 cm |

A receiver's line is accurate when its calibration points sit at very
different distances from it. The current two points are close to the middle,
so some receivers see them at almost the same distance (R3: 292 and 276 cm),
and a one-tick difference becomes a large slope error. More points also
average out range walk, which differs from point to point.

## 8. Maps

**Tools > Maps** (Ctrl+M) computes one number for every point of the arena,
so you can see at a glance where the system works and why. A map is computed
on its own thread from a copy of the simulation, so the simulation keeps
running; hover for the value under the cursor, and **Show on arena** shades
the main arena with it.

| Map | Value at each point | Use it to |
|---|---|---|
| **Coverage** | Receivers that can detect the direct sound | Place and aim receivers so the area you drive in has 4 or more everywhere |
| **First arrival not direct (NLOS)** | Receivers that would time something other than the direct sound: an echo, bent sound, another receiver's burst, or nothing | See where outliers come from; keep calibration points where this is 0 |
| **Geometry (HDOP)** | cm of position error per cm of range error, from the receivers that hear the direct sound | Judge receiver placement independently of noise |
| **Range bias** | Mean range error with no noise: range walk, echoes, bent sound | Put calibration points where this is low and typical |
| **Accuracy (Monte Carlo)** | RMS position error of one algorithm over many pings, with every effect on | Compare algorithms across the whole arena |

The first four switch the random parts off, so they show the physics alone.

![Multipath map](docs/img/maps-multipath.png)

With a 50 × 70 cm box in the arena, each receiver has a shadow behind it.
Only 12 % of the arena (dark) has every receiver timing the direct sound;
near the box, up to four receivers read something else.

![Accuracy map](docs/img/maps-accuracy.png)

LM's accuracy in the same room: about 2 cm in the open, but more than 8 cm
(yellow) in the box's shadow next to R1, where R1 hears the sound bend round
the box or loses it, and along the edges beside and behind receivers, where
the transmitter is far off the axis of the receivers nearest to it.

## 9. Recording and replaying real data

**Record** (Ctrl+R) starts recording frames; press it again to stop and save
them, either as a CSV (time, true position, ticks, distances, every receiver's
status and every algorithm's fix and error) or as a base-station tick log
(`C: t0, t1, ...` lines, exactly as the base station prints in continuous
mode).

**File > Replay tick log** (Ctrl+O) feeds recorded ticks to the solvers
instead of the simulated hardware, frame by frame, looping at the end. It
reads:

- the base station's own output, `N: ...` or `C: ...` lines, parsed with the
  library's own `parse_message`, so a log captured from the real system's
  serial port can be replayed through every algorithm, including the ones
  that are not in the library, and stepped through in the lab; there is no
  true position, so no errors are shown;
- a CSV recorded by the simulator, with its true positions.

**File > Stop replay** goes back to the simulated hardware.

## 10. Benchmark results

`python3 -m engine.benchmark` drives the transmitter to random waypoints for
400 frames (40 s) under each of ten conditions and reports every algorithm's
RMS and 95th-percentile error and its failed frames (seed 1). Lower is
better; the best RMS in each row is bold.

| Condition | OLS | LM | CEP | CEP4 | ROBUST | BIAS | RANSAC | EKF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ideal (first simulator), noise 1 cm | 1.2 | 0.8 | 1.3 | 1.3 | 0.8 | 0.9 | 0.8 | **0.8** |
| Realistic room | 0.9 | 0.9 | 1.3 | 1.3 | 0.9 | **0.7** | 0.9 | 1.0 |
| Realistic + 40 cm box in the arena | 26.9 | 28.1 | 1.4 | 1.2 | 1.4 | 27.0 | **1.0** | 1.1 |
| Bad values 5 % (±50 %) | 4.5 | 4.3 | 1.3 | 1.3 | 1.8 | 4.6 | 1.4 | **1.1** |
| Dropouts 10 %, lost signal 5 % | 7.2 | 2.1 | 2.2 | 5.4 | 2.1 | 5.4 | 2.1 | **1.1** |
| Trigger jitter 100 µs | 1.8 | 2.3 | 4.6 | 4.7 | 3.0 | **0.7** | 3.3 | 3.9 |
| Air at 30 °C (calibrated at 20 °C) | 3.2 | 2.6 | 5.2 | 5.2 | 2.9 | **1.3** | 2.6 | 2.6 |
| Receiver R4 off | 1.0 | 0.9 | 1.3 | 1.1 | 0.9 | **0.8** | 0.9 | 1.0 |
| Receivers misplaced (σ 3 cm) | 1.7 | 1.2 | 3.0 | 2.6 | 1.2 | 1.3 | **1.2** | 1.2 |
| Harsh preset | 3.4 | 3.9 | 3.7 | 5.2 | 3.1 | 5.3 | 2.9 | **2.3** |

Failed frames (no fresh fix, the algorithm held its last one) were rare
except with dropouts and lost signals: 6 % for every algorithm that uses the
library's filter and 23 % for CEP4 and BIAS, which need four receivers,
against none for the Kalman tracker; and in the harsh preset, 5 to 7 %
against 2 %. `--markdown` also prints the 95th percentiles.

What stands out:

- **Every reading good:** LM, the robust fit, RANSAC and the Kalman tracker
  are all within 1 cm; the library's CEP and CEP4 are about 40 % worse,
  because the winning subset usually uses only three receivers.
- **A box in the arena:** in about one ping in eight, a receiver behind the
  box hears a wall echo, typically more than 2 m longer than the direct path,
  and the filter keeps it. OLS, LM and BIAS are then about 1 m off (their
  95th percentile is about 90 cm); everything that rejects outliers stays
  near 1 cm.
- **Bad values** split them the same way. With **dropouts** every algorithm
  that uses the library's filter loses fixes (each glitch costs two frames,
  section 11), and the Kalman tracker, which coasts, is best.
- **Trigger jitter** and **temperature** shift every range together; BIAS,
  which solves for a common offset, beats LM by a factor of 2 to 3, and the
  subset methods are worst.
- **Survey error** hurts the subset methods most.


## 11. What the simulator says about the real system

These come from the benchmark, the lab and the maps. They are predictions of
a model, so treat them as things to check on the real system, not as facts
about it.

1. **The jump filter rejects the good reading after a bad one.** The library
   stores every reading as "last", including the ones it rejected, so after a
   spike or a dropout the next good reading is rejected as well: each glitch
   costs a receiver two frames. A reading that stays wrong (a blocked
   receiver timing an echo) passes from the second frame on. Comparing
   against the last *kept* value would fix the first; only an algorithm that
   rejects outliers fixes the second.
2. **Blocked line of sight is the dangerous case.** A wall echo or sound bent
   round an obstacle reads long, inside the bounds and steadily, so the filter
   keeps it, and LM, OLS and BIAS are pulled tens of cm off (section 6.2). CEP,
   RANSAC and the Kalman tracker reject it. If anything can stand between the
   transmitter and a receiver, use one of those.
3. **The library's CEP prefers three-receiver subsets** (section 6.1), so it
   is noisier than LM when every reading is good, and it freezes on its last
   fix whenever the transmitter leaves the receivers' box. CEP4, RANSAC and the
   Kalman tracker keep its robustness without the noise.
4. **A common timing error is easy to remove.** Trigger timing (radio and
   processing delay, jitter) shifts every range by the same amount; BIAS
   solves for it and cuts the error to a third with 100 µs of jitter. The same
   solver also takes out part of a temperature change, which looks like a
   common offset over a small arena.
5. **Temperature matters more than noise.** 10 °C between calibration and use
   makes every range 1.7 % long: 5 cm at 3 m. Measuring the air temperature
   and scaling by c_cal / c costs nothing; so does calibrating in the
   conditions you run in.
6. **Range walk is a bias calibration cannot remove.** A weak signal is
   detected a cycle or more late (0.86 cm a cycle). It depends on distance
   and angle, so a straight-line calibration absorbs it only where it was
   calibrated. Aiming receivers at the middle of the area you drive in, and
   using 3 or 4 calibration points spread over it, keeps it small.
7. **Calibration points matter.** The current two points give a 3.3 cm
   distance error over the arena; the best two, 1.9 cm; the best three or
   four, 1.3 cm (section 7). `python3 -m engine.cal_points --apply` writes
   the best set into config.toml.
8. **The Kalman tracker is the most robust overall** in the simulation: it
   coasts through dropouts and lost frames, gates spikes by the motion model,
   and is best in the harsh preset. It lags hard turns.
9. **Crosstalk.** If the receivers' own HC-SR04 transmitters are not covered,
   a receiver facing another one hears its burst before the transmitter's
   (the crosstalk switch shows the result: readings far too short). Cover
   them.
10. **Survey the receivers.** Receivers 3 cm (σ) away from where config.toml
    says they are cost LM little (1.2 cm against 0.9), because six receivers
    average the errors out, but the library's CEP, which trusts three at a
    time, gets more than twice as bad (3.0 cm against 1.3).

## 12. Code map and how to extend it

| Path | Role |
|---|---|
| `app.py` | Command line and start-up |
| `engine/acoustics.py` | Pure physics: speed of sound, absorption, beam patterns, spreading, knife-edge diffraction, threshold detection |
| `engine/world.py` | Walls and obstacles; line of sight; image-source reflections; paths round corners |
| `engine/hardware.py` | One ping through the hardware: trigger timing, every arrival, detection, timeout, sampling window, ticks, faults; `Environment` (every physical setting), `PRESETS`, `Faults`, `STATUS` |
| `engine/config.py` | Reading config.toml, sensors.toml and world.toml |
| `engine/motion.py` | Where the transmitter goes |
| `engine/simulation.py` | One frame: move, ping, solve, score; recording and replay |
| `engine/solvers.py` | Runs the library's OLS, LM and CEP and the simulator's own algorithms |
| `engine/lab.py` | Traces of what the library does inside one call, for the lab |
| `engine/maps.py` | The arena maps |
| `engine/calibration.py` | The calibration library on the simulated sensors |
| `engine/cal_points.py` | Calibration point search |
| `engine/benchmark.py` | The benchmark |
| `gui/theme.py`, `gui/draw.py` | Palettes and the vector drawing kit everything is drawn with |
| `gui/runner.py` | Steps the simulation on its own thread |
| `gui/main_window.py` | The main window, docks, menus and shortcuts |
| `gui/arena.py`, `gui/charts.py`, `gui/panels.py` | The arena, echo scope and error chart, and the docks |
| `gui/lab_window.py`, `gui/calibration_window.py`, `gui/map_window.py` | The tool windows |
| `ultragps_position/`, `ultragps_calibration/` | The libraries, unchanged |
| `config/` | config.toml and sensors.toml (as on `main`), world.toml (new) |
| `tests/` | `python3 -m unittest discover -s tests -t .` |
| `docs/make_screenshots.py` | Regenerates the pictures in this manual |

**Adding an algorithm.** Write a function in `engine/solvers.py` that takes
the library, the distances and the receiver indices and returns an
`Estimate` (position, success, receivers used, covariance or None, a note);
add it to `ALGORITHMS` and call it in `Solvers.solve`; give it a colour in
`theme.py` and a marker in `ALGO_SHAPES`. It then appears in the arena, the
docks, the error chart, the lab's Blocking tab, the accuracy map and the
benchmark.

**Adding a physical effect.** Add a field to `Environment` in
`engine/hardware.py` and use it in `Hardware.arrivals` (anything that
changes a sound path's strength or time) or `Hardware.measure` (anything that
changes what the receiver or the base station does with it). Add a row for it
to `ENV_GROUPS` in `gui/panels.py` and it gets a control with a tooltip.

**Tests.** 170 tests cover the physics against their formulas, the library's
algorithms through the simulator, the simulator's algorithms, the lab traces
against the library, the maps, and every window driven headless. One test
checks the SHA-256 of the library files, so an accidental edit fails.

## 13. Assumptions and limits

- **2D.** Everything is in the plane of the receivers. Height differences,
  the floor and the ceiling are not modelled; in a real room the ceiling and
  floor give echoes too.
- **First arrival, no interference.** Each path is detected on its own; two
  arrivals within a burst of each other (a strong echo just behind the direct
  sound) would really add up and shift the detection, which is not modelled.
- **Transducers.** The beam pattern is a typical 40 kHz transducer's (−6 dB at
  ±27.5°); the HC-SR04's real pattern and the threshold circuit's behaviour
  vary between modules. The signal margin (30 dB on axis at 1 m) is an
  estimate: with it the direct sound is detected out to about 7 m on axis,
  3.5 m at 45° off axis and 2.5 m at 60°, so in this arena it is the angle,
  not the distance, that loses a receiver. Lower it to model weaker modules.
- **Timing.** The transmitter's 4 ms lead is inferred from the calibration's
  intercept, not measured. The base station's tick is a loop count, modelled
  through each receiver's line in sensors.toml.
- **The room.** `config/world.toml` puts walls 50 cm outside the receivers;
  that is a guess. Measure the real room and edit it (or draw it with the
  Obstacles tool and **Save world**) for echoes in the right places.
- **Surfaces.** Each wall and obstacle reflects a fixed fraction of the sound
  at every angle, like a smooth hard surface. Soft or rough surfaces scatter
  instead.
- **Speed.** A frame takes 50 to 80 ms to compute, mostly the library's CEP
  trying 42 subsets; the simulation runs on its own thread so the windows stay
  responsive, and at 10 Hz it keeps up with real time on a typical machine.
