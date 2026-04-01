"""IndustrialDrone: convert market ticks to generative industrial drone audio.

Enhancements over base MarketSynth:
  1. Temporal Smearing  — each tick generates a long (10s) segment; segments are
                          overlap-added so multiple market states coexist at once.
  2. Complex Harmonics  — sawtooth + sub-octave square wave instead of pure sine.
  3. Portamento         — frequency glides exponentially between ticks instead of
                          jumping instantly.
  4. LPF Volatility     — ATR drives a 1-pole low-pass filter cutoff, not drive
                          level: calm = dark/muffled, volatile = bright/piercing.
  5. Feedback + Delay   — a comb-filter feedback loop creates long temporal
                          persistence so past market states haunt the mix.
"""

import argparse
import re
import sys
from datetime import datetime, timedelta

_MIN_PY = (3, 11)
_MAX_PY = (3, 12)
if not (_MIN_PY <= sys.version_info[:2] <= _MAX_PY):
    raise RuntimeError(
        "Unsupported Python version. Use Python 3.11 or 3.12 for this project. "
        "pandas_ta/numba builds are unstable or unavailable on newer versions."
    )

import numpy as np
import yfinance as yf
import pandas_ta as ta
from scipy.io.wavfile import write


# ---------------------------------------------------------------------------
# 1-pole low-pass filter (IIR)
# ---------------------------------------------------------------------------

def lowpass_1pole(signal: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    """Apply a simple 1-pole IIR low-pass filter."""
    if cutoff_hz <= 0:
        return np.zeros_like(signal)
    if cutoff_hz >= sample_rate / 2:
        return signal.copy()
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    dt = 1.0 / sample_rate
    alpha = float(dt / (rc + dt))
    out = signal.copy().astype(np.float64)
    y = 0.0
    for i in range(len(out)):
        y = alpha * out[i] + (1.0 - alpha) * y
        out[i] = y
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Feedback delay (comb filter)
# ---------------------------------------------------------------------------

def apply_feedback_delay(
    signal: np.ndarray,
    sample_rate: int,
    delay_ms: float = 800.0,
    feedback: float = 0.55,
    wet: float = 0.40,
) -> np.ndarray:
    """Comb-filter feedback delay for temporal persistence."""
    delay_samples = int(sample_rate * delay_ms / 1000.0)
    buf = np.zeros(delay_samples, dtype=np.float64)
    out = np.empty_like(signal, dtype=np.float64)
    head = 0
    sig = signal.astype(np.float64)
    for i in range(len(sig)):
        delayed = buf[head]
        new_val = sig[i] + feedback * delayed
        buf[head] = new_val
        out[i] = (1.0 - wet) * sig[i] + wet * delayed
        head = (head + 1) % delay_samples
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Main synthesiser class
# ---------------------------------------------------------------------------

class IndustrialDroneSynth:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = int(sample_rate)
        # C Minor Pentatonic: C, Eb, F, G, Bb (MIDI offsets: 0, 3, 5, 7, 10)
        self.scale_intervals = [0, 3, 5, 7, 10]

    def get_pentatonic_freq(self, index_1_to_100: float) -> float:
        """Maps a 0-100 index to C-Minor Pentatonic across several octaves."""
        index = max(0.0, min(100.0, float(index_1_to_100)))
        midi_pool = [
            12 + octave * 12 + interval
            for octave in range(9)
            for interval in self.scale_intervals
            if 12 + octave * 12 + interval <= 110
        ]
        idx = int((index / 100.0) * (len(midi_pool) - 1))
        midi_note = midi_pool[max(0, min(idx, len(midi_pool) - 1))]
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

    def _make_segment(
        self,
        freq_start: float,
        freq_end: float,
        intensity: float,
        cutoff_hz: float,
        seconds: float,
    ) -> np.ndarray:
        """Generate one overlapping drone segment with all five enhancements."""
        n = int(self.sample_rate * seconds)

        # --- Change 3: Exponential frequency portamento ---
        # Compute instantaneous frequency per sample via log-space interpolation.
        log_start = np.log(max(freq_start, 1.0))
        log_end   = np.log(max(freq_end,   1.0))
        instant_freq = np.exp(np.linspace(log_start, log_end, n))

        # Integrate to get phase — this avoids click artifacts at segment edges.
        dt = 1.0 / self.sample_rate
        phase = np.cumsum(instant_freq) * dt * 2.0 * np.pi

        # --- Change 2: Complex waveforms ---
        # Band-limited sawtooth (first 8 harmonics, phase-coherent)
        saw = np.zeros(n, dtype=np.float32)
        for k in range(1, 9):
            saw += ((-1) ** (k + 1)) / k * np.sin(k * phase).astype(np.float32)
        saw *= 2.0 / np.pi

        # Sub-octave square wave (odd harmonics, half-frequency) adds body
        sqr = np.zeros(n, dtype=np.float32)
        for k in range(1, 16, 2):
            sqr += (1.0 / k) * np.sin(k * phase / 2.0).astype(np.float32)
        sqr *= 4.0 / np.pi

        # Blend: 70% sawtooth (grit), 30% sub-octave square (weight)
        segment = (0.70 * saw + 0.30 * sqr).astype(np.float32)

        # Fixed soft-clip — brightness is now managed entirely by the LPF below
        segment = np.tanh(segment * 2.5).astype(np.float32)

        # --- Change 4: LPF with volatility-driven cutoff ---
        segment = lowpass_1pole(segment, cutoff_hz, self.sample_rate)

        # Amplitude envelope: RSI-driven level + raised-cosine fade (click-free)
        envelope = np.ones(n, dtype=np.float32)
        fade = min(int(self.sample_rate * 0.4), n // 4)
        if fade > 0:
            envelope[:fade]  = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade) / fade))
            envelope[-fade:] = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade, 0, -1) / fade))

        segment *= envelope * max(0.05, float(intensity))
        return segment.astype(np.float32)

    def generate_drone(
        self,
        ticker_symbol: str,
        start_time: str,
        end_time: str,
        seconds_per_tick: float = 10.0,
        tick_stride_seconds: float = 1.0,
        interval: str = "1m",
        delay_ms: float = 800.0,
        feedback: float = 0.55,
        delay_wet: float = 0.40,
    ):
        """Fetch market data and render an industrial drone audio file."""
        print(f"--- INDUSTRIAL DRONE: {ticker_symbol} ({start_time} to {end_time}) [{interval}] ---")

        # 1. Fetch & validate data
        try:
            data = yf.download(
                ticker_symbol, start=start_time, end=end_time,
                interval=interval, progress=False
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to download market data: {exc}") from exc

        if data is None or data.empty:
            raise ValueError(
                f"No market data for {ticker_symbol} ({start_time} to {end_time}, {interval}). "
                "Check ticker / date range."
            )

        data = data.copy()
        if hasattr(data.columns, "nlevels") and data.columns.nlevels > 1:
            data.columns = data.columns.get_level_values(0)

        # 2. Technical indicators
        data["RSI"] = ta.rsi(data["Close"], length=14)
        data["ATR"] = ta.atr(data["High"], data["Low"], data["Close"], length=14)
        data = data.dropna()
        if data.empty:
            raise ValueError("Insufficient data after indicator calculation (range too short).")

        # 3. Normalisation ranges
        price_min   = float(data["Close"].min())
        price_max   = float(data["Close"].max())
        price_range = price_max - price_min

        atr_vals  = data["ATR"].values.astype(float)
        atr_min   = float(np.nanmin(atr_vals))
        atr_max   = float(np.nanmax(atr_vals))
        atr_range = atr_max - atr_min if atr_max != atr_min else 1.0

        # LPF range: calm market -> 300 Hz (dark), volatile -> 8000 Hz (bright)
        LPF_MIN, LPF_MAX = 300.0, 8000.0

        # 4. Build overlap-add buffer (Change 1: Temporal Smearing)
        seconds_per_tick    = float(seconds_per_tick)
        tick_stride_seconds = float(tick_stride_seconds)
        n_ticks             = len(data)
        n_seg               = int(self.sample_rate * seconds_per_tick)
        stride              = int(self.sample_rate * tick_stride_seconds)

        total_samples = stride * n_ticks + n_seg
        ola_buffer    = np.zeros(total_samples, dtype=np.float64)

        print(
            f"Rendering {n_ticks} ticks x {seconds_per_tick}s segments "
            f"(stride {tick_stride_seconds}s) -> ~{total_samples / self.sample_rate:.1f}s total ..."
        )

        # Pre-compute target frequencies for portamento
        freqs = []
        for _, row in data.iterrows():
            price_idx = (
                50.0 if price_range == 0
                else float(((row["Close"] - price_min) / price_range) * 100.0)
            )
            freqs.append(self.get_pentatonic_freq(price_idx))

        # Render each tick and overlap-add
        for i, (_, row) in enumerate(data.iterrows()):
            freq_start = freqs[i]
            freq_end   = freqs[min(i + 1, n_ticks - 1)]

            intensity = float(row["RSI"]) / 100.0 if not np.isnan(row["RSI"]) else 0.5

            atr_norm  = float(row["ATR"] - atr_min) / atr_range if not np.isnan(row["ATR"]) else 0.5
            cutoff_hz = LPF_MIN + atr_norm * (LPF_MAX - LPF_MIN)

            seg = self._make_segment(freq_start, freq_end, intensity, cutoff_hz, seconds_per_tick)

            start_sample = i * stride
            ola_buffer[start_sample : start_sample + n_seg] += seg.astype(np.float64)

        # 5. Feedback delay (Change 5)
        print("Applying feedback delay ...")
        mix = apply_feedback_delay(
            ola_buffer.astype(np.float32),
            self.sample_rate,
            delay_ms=delay_ms,
            feedback=feedback,
            wet=delay_wet,
        )

        # 6. Normalise & export
        max_val = np.max(np.abs(mix))
        if max_val == 0 or np.isnan(max_val):
            raise RuntimeError("Generated silent audio — something went wrong.")
        mix = mix / max_val

        safe_ticker = re.sub(r"[^A-Za-z0-9_.\-]", "_", ticker_symbol)
        safe_start  = re.sub(r"[^A-Za-z0-9_.\-]", "_", str(start_time))
        filename    = f"{safe_ticker}_{safe_start}_drone.wav"

        write(filename, self.sample_rate, (mix * 32767).astype(np.int16))
        print(f"--- EXPORT COMPLETE: {filename} ({total_samples / self.sample_rate:.1f}s) ---")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="IndustrialDrone: market ticks -> industrial drone audio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ticker",    "-t", default="CL=F",   help="Yahoo Finance ticker")
    p.add_argument("--start",     "-s", default=None,      help="Start date YYYY-MM-DD")
    p.add_argument("--end",       "-e", default=None,      help="End date YYYY-MM-DD")
    p.add_argument("--interval",  "-i", default="1m",
                   choices=["1m", "5m", "15m", "1h", "1d"], help="OHLC interval")
    p.add_argument("--sr",        default=44100, type=int, help="Sample rate (Hz)")
    p.add_argument("--seg-len",   default=10.0,  type=float,
                   help="Segment length per tick in seconds (longer = thicker smear)")
    p.add_argument("--stride",    default=1.0,   type=float,
                   help="Gap between segment starts in seconds (smaller = denser)")
    p.add_argument("--delay-ms",  default=800.0, type=float,
                   help="Feedback delay time in milliseconds")
    p.add_argument("--feedback",  default=0.55,  type=float,
                   help="Feedback coefficient 0-<1 (higher = longer decay)")
    p.add_argument("--delay-wet", default=0.40,  type=float,
                   help="Delay wet/dry mix (0=dry, 1=all-wet)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    today = datetime.now().date()

    if args.start is None:
        args.start = (today - timedelta(days=7)).isoformat()
    if args.end is None:
        args.end = today.isoformat()

    def _parse_date(s, label):
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(f"Invalid {label} date '{s}'. Use YYYY-MM-DD.")

    start_date = _parse_date(args.start, "--start")
    end_date   = _parse_date(args.end,   "--end")

    if end_date > today:
        print("End date is in the future; clamping to today.")
        end_date = today

    if args.interval == "1m":
        max_lookback = today - timedelta(days=30)
        if start_date < max_lookback:
            print("1m interval: truncating start to last 30 days (Yahoo API limit).")
            start_date = max_lookback

    synth = IndustrialDroneSynth(sample_rate=args.sr)
    synth.generate_drone(
        ticker_symbol=args.ticker,
        start_time=start_date.isoformat(),
        end_time=end_date.isoformat(),
        seconds_per_tick=args.seg_len,
        tick_stride_seconds=args.stride,
        interval=args.interval,
        delay_ms=args.delay_ms,
        feedback=args.feedback,
        delay_wet=args.delay_wet,
    )