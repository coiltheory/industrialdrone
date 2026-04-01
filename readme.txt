# IndustrialDrone
### Built on MarketSynth

## What this project does

IndustrialDrone converts financial market data (OHLC ticks) into generative **industrial drone audio**. It is a heavily reworked evolution of MarketSynth, retaining the core concept of fetching market bars and mapping price/volatility to audio parameters, but replacing the original staccato sine-wave approach with a dense, slow-moving drone engine designed to produce textures closer to industrial ambient music than digital bleeps.

## What changed from MarketSynth

MarketSynth rendered each market tick as a short, independent sine-wave burst mapped to the C Minor Pentatonic scale. The result was recognizably "musical" but erratic — pitch jumped instantly between notes, each tick was isolated, and the overall texture sounded like a sequencer rather than a machine.

IndustrialDrone makes five structural changes to address this.

---

### 1. Temporal Smearing and Overlapping Ticks

**MarketSynth:** each 1-minute tick produced exactly 1 second of audio. Ticks played back-to-back with no overlap, creating a staccato, grid-like rhythm.

**IndustrialDrone:** each tick now generates a long segment (default 10 seconds), but segment *start times* remain 1 second apart. Segments are written into a single overlap-add (OLA) buffer, meaning at any moment in the output you are hearing roughly 10 market states layered simultaneously. The result is a thick, slowly evolving soundscape where individual ticks dissolve into the whole rather than announcing themselves.

Configurable via `--seg-len` (segment length) and `--stride` (spacing between start times).

---

### 2. Complex Harmonics (Sawtooth + Sub-Octave Square)

**MarketSynth:** used a pure sine wave — clean, simple, and tonally thin.

**IndustrialDrone:** replaces the sine with a blend of two band-limited waveforms:
- A **sawtooth wave** (first 8 harmonics) provides a buzzy, harmonically rich upper register.
- A **sub-octave square wave** (odd harmonics at half the fundamental frequency) adds low-end weight and body.

The two are blended 70/30 (saw/square) and passed through a fixed `tanh` soft-clipper, giving the raw waveform significantly more "meat" for subsequent processing to act on.

---

### 3. Portamento (Exponential Frequency Gliding)

**MarketSynth:** pitch jumped instantly to the new target frequency at the start of each tick, producing a step-sequencer effect.

**IndustrialDrone:** pitch glides *continuously* from the previous tick's frequency to the next, using exponential interpolation (log-space linear ramp, which is perceptually linear in pitch). The instantaneous frequency is integrated sample-by-sample into a phase signal before any waveform is computed, eliminating all click artifacts at transitions. The effect is a slow, tectonic pitch drift rather than digital stepping.

---

### 4. Volatility → Low-Pass Filter Cutoff

**MarketSynth:** ATR (Average True Range) directly drove the `drive` parameter of a `tanh` distortion — higher volatility meant more aggressive clipping.

**IndustrialDrone:** ATR is instead mapped to the **cutoff frequency of a 1-pole IIR low-pass filter** applied to each segment. The cutoff range is 300 Hz (calm market, dark and muffled) to 8000 Hz (volatile market, bright and piercing). This creates a dramatically more expressive dynamic: quiet consolidation periods produce a subterranean rumble, while volatility spikes cause the harmonic content to tear open and become abrasive.

---

### 5. Feedback Delay

**MarketSynth:** audio segments played and ended — no temporal persistence between them.

**IndustrialDrone:** after all segments are overlap-added, the full mix is passed through a **comb-filter feedback delay**. A copy of the signal from `delay_ms` milliseconds ago is fed back into the buffer at a configurable `feedback` coefficient, then blended wet/dry into the output. The effect is that the sonic signature of past market states — a flash crash, a volatility spike — continues to decay and reverberate in the background long after the underlying data has moved on. The default settings (800ms delay, 0.55 feedback, 0.40 wet) produce a long, ghostly tail.

---

## Requirements

- macOS / Linux (tested)
- Python 3.11 or 3.12 (3.13+ can fail during `pandas_ta`/`numba` builds)
- Libraries (install via pip):
  - `numpy`
  - `yfinance`
  - `pandas_ta`
  - `scipy`

## Setup

```bash
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Usage

```bash
source .venv/bin/activate
python app_industrial_drone.py --ticker CL=F --start 2026-01-20 --end 2026-02-10
```

## All Options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--ticker` | `-t` | `CL=F` | Yahoo Finance ticker (e.g. `AAPL`, `^SPX`) |
| `--start` | `-s` | 7 days ago | Start date `YYYY-MM-DD` |
| `--end` | `-e` | today | End date `YYYY-MM-DD` |
| `--interval` | `-i` | `1m` | OHLC interval: `1m`, `5m`, `15m`, `1h`, `1d` |
| `--sr` | | `44100` | Sample rate in Hz |
| `--seg-len` | | `10.0` | Segment length per tick in seconds — longer = thicker smear |
| `--stride` | | `1.0` | Gap between segment start times in seconds — smaller = denser |
| `--delay-ms` | | `800.0` | Feedback delay time in milliseconds |
| `--feedback` | | `0.55` | Feedback coefficient 0–<1 — higher = longer decay tail |
| `--delay-wet` | | `0.40` | Wet/dry mix for delay (0 = dry, 1 = all-wet) |

## Suggested Presets

```bash
# Default: crude oil futures, last 7 days, 1-minute ticks
python app_industrial_drone.py

# Denser smear — fewer distinct pitches, more wall-of-sound
python app_industrial_drone.py --seg-len 20 --stride 0.5

# Longer feedback tail for a more haunted atmosphere
python app_industrial_drone.py --feedback 0.72 --delay-ms 1200 --delay-wet 0.55

# Daily candles over a year — slow tectonic shifts in pitch
python app_industrial_drone.py -t SPY -i 1d --start 2023-01-01 --seg-len 30 --stride 5
```

## Output

The script writes a normalized 16-bit WAV file named `{TICKER}_{START}_drone.wav` in the working directory.

## Behavior & Limitations

- Yahoo Finance restricts `1m` data to approximately the last 30 days. The script truncates start dates automatically for `1m` requests.
- Some tickers (futures, delisted symbols) may have no `1m` data — try `1d` or `5m`, or verify the ticker.
- Long segment lengths and dense strides can produce large in-memory buffers. For a week of 1-minute data at `--seg-len 10`, expect a buffer of roughly 50,000 samples × 10s × 44100 Hz ≈ a few hundred MB. Reduce `--seg-len` or increase `--stride` if memory is a concern.
- The feedback delay loop is a pure Python sample-by-sample loop and will be the slowest part of the pipeline on long files. This is acceptable for offline rendering but not suitable for real-time use.

## Development Notes

- `app_industrial_drone.py` contains all synthesis logic and the CLI entry point.
- `requirements.txt` lists pinned runtime dependencies (unchanged from MarketSynth).
- The `IndustrialDroneSynth` class can be imported and used directly without the CLI — instantiate with a sample rate and call `generate_drone()`.