MarketSynth
===============

What this project does
----------------------
MarketSynth converts financial market data (OHLC ticks) into generative drone audio.
It fetches high-resolution market bars, computes simple technical indicators (RSI, ATR),
and maps price and volatility to musical parameters (frequency, amplitude, distortion) to
produce a synthesized WAV file.

Highlights
----------
- Maps price to the C Minor Pentatonic scale across multiple octaves.
- Uses RSI to control amplitude (intensity) and ATR to drive soft-clipping distortion.
- Generates a readable, normalized 16-bit WAV file at the configured sample rate.

Requirements
------------
- macOS / Linux (tested)
- Python 3.11 or 3.12 (3.13+ can fail during `pandas_ta`/`numba` builds)
- Tools/libraries (installed via pip):
  - numpy
  - yfinance
  - pandas_ta
  - scipy

Setup
-----
1. Install a supported Python (pyenv recommended). Example using pyenv:

```bash
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Alternatively, use your system Python (ensure version is 3.11 or 3.12) and create/activate a venv before installing.

Quick usage
-----------
Run the synthesizer with a ticker and date range. Example:

```bash
source .venv/bin/activate
python app.py --ticker AAPL --start 2026-01-29 --end 2026-02-05 --seconds-per-tick 0.05
```

Options
-------
- `--ticker` or `-t`: Ticker symbol (e.g. `AAPL`, `CL=F`, `^SPX`).
- `--start` / `--end`: Date range in `YYYY-MM-DD` format. Defaults to the last 7 days.
- `--interval` / `-i`: Yahoo data interval: `1m`, `5m`, `15m`, `1h`, `1d`. Note: Yahoo limits `1m` to roughly the last 30 days.
- `--seconds-per-tick`: How many seconds of audio to generate per market bar.
- `--sr`: Sample rate (default 44100).

Behavior & limitations
----------------------
- Yahoo Finance (`yfinance`) restricts `1m` data to recent ranges (≈30 days). The script will truncate start dates automatically for `1m` requests and will raise a clear error if no data is available.
- Some tickers (futures, delisted symbols) may have no 1m data — try `1d` or `5m` intervals or verify ticker validity.
- Audio is synthesized as simple sine waves mapped to the pentatonic scale; this project is intended as a creative sonification tool rather than a production audio engine.

Output
------
- The script writes a normalized 16-bit WAV file named `{TICKER}_{START}.wav` in the project directory.

Development notes
-----------------
- `app.py` contains the main logic and a CLI.
- `requirements.txt` lists pinned runtime dependencies.
- If you want, enable automatic fallback intervals (e.g. from `1m` to `1d`) by requesting a change.

Contact
-------
For questions or improvements, open an issue or edit the code and create a PR.
