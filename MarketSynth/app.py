"""IndustrialDrone: convert market ticks to generative drone audio.

This script fetches 1-minute OHLC data and maps ticks to a C-minor
pentatonic sine drone, modulating amplitude with RSI and distortion with ATR.
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


class IndustrialDroneSynth:
    def __init__(self, sample_rate=44100):
        self.sample_rate = int(sample_rate)
        # C Minor Pentatonic: C, Eb, F, G, Bb (Midi offsets: 0, 3, 5, 7, 10)
        self.scale_intervals = [0, 3, 5, 7, 10]

    def get_pentatonic_freq(self, index_1_to_100: float) -> float:
        """Maps 1-100 index to C-Minor Pentatonic across several octaves.

        Args:
            index_1_to_100: value in range [0,100]

        Returns:
            Frequency in Hz
        """
        index = max(0.0, min(100.0, float(index_1_to_100)))
        midi_pool = []
        for octave in range(9):
            for interval in self.scale_intervals:
                note = 12 + (octave * 12) + interval
                if note <= 110:
                    midi_pool.append(note)

        idx = int((index / 100.0) * (len(midi_pool) - 1))
        midi_note = midi_pool[max(0, min(idx, len(midi_pool) - 1))]
        return 440.0 * (2 ** ((midi_note - 69) / 12.0))

    def generate_drone(self, ticker_symbol: str, start_time: str, end_time: str, seconds_per_tick: float = 1.0, interval: str = "1m"):
        print(f"--- INITIALIZING SYSTEM: {ticker_symbol} ({start_time} -> {end_time}) interval={interval} ---")

        # 1. Fetch Data
        try:
            data = yf.download(ticker_symbol, start=start_time, end=end_time, interval=interval)
        except Exception as e:
            raise RuntimeError(f"Failed to download market data: {e}")

        if data is None or data.empty:
            raise ValueError(f"No market data found for {ticker_symbol} in {start_time}->{end_time} interval={interval}. Check ticker validity or interval restrictions.")

        # 2. Calculate Industrial Metrics
        data = data.copy()
        # yfinance may return MultiIndex columns (Price, Ticker). Flatten to last level for single-ticker use.
        if hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
            # Keep primary price names (Close, High, Low, Open, Volume)
            data.columns = data.columns.get_level_values(0)

        data['RSI'] = ta.rsi(data['Close'], length=14)
        data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
        data = data.dropna()
        if data.empty:
            raise ValueError('Insufficient data after indicator calculation (too-short range).')

        # 3. Audio Processing Constants
        seconds_per_tick = float(seconds_per_tick)
        n_samples_per_tick = int(self.sample_rate * seconds_per_tick)
        if n_samples_per_tick <= 0:
            raise ValueError('seconds_per_tick too small')

        print(f"Processing {len(data)} market ticks into drone ({n_samples_per_tick} samples/tick)...")

        price_min = float(data['Close'].min())
        price_max = float(data['Close'].max())
        price_range = price_max - price_min
        atr_mean = float(data['ATR'].mean()) if data['ATR'].mean() not in (None, 0) else 0.0

        segments = []
        fade_size = max(1, int(self.sample_rate * 0.05))

        for _, row in data.iterrows():
            if price_range == 0:
                price_index = 50.0
            else:
                price_index = float(((row['Close'] - price_min) / price_range) * 100.0)

            freq = self.get_pentatonic_freq(price_index)
            intensity = float(row['RSI']) / 100.0 if not np.isnan(row['RSI']) else 0.0
            volatility = (float(row['ATR']) / atr_mean) if (atr_mean and not np.isnan(row['ATR'])) else 1.0

            t = np.linspace(0.0, seconds_per_tick, n_samples_per_tick, False)
            segment = np.sin(2.0 * np.pi * freq * t).astype(np.float32)

            drive = 1.0 + (volatility * 5.0)
            segment = np.tanh(segment * drive)

            segment *= intensity

            if fade_size * 2 < len(segment):
                fade_in = np.linspace(0.0, 1.0, fade_size, dtype=np.float32)
                fade_out = np.linspace(1.0, 0.0, fade_size, dtype=np.float32)
                segment[:fade_size] *= fade_in
                segment[-fade_size:] *= fade_out

            segments.append(segment)

        if not segments:
            raise RuntimeError('No audio segments generated')

        total_audio = np.concatenate(segments).astype(np.float32)

        max_val = np.max(np.abs(total_audio))
        if max_val == 0 or np.isnan(max_val):
            raise RuntimeError('Generated silent audio (max amplitude is zero)')

        total_audio = total_audio / max_val

        safe_ticker = re.sub(r"[^A-Za-z0-9_.-]", "_", ticker_symbol)
        safe_start = re.sub(r"[^A-Za-z0-9_.-]", "_", str(start_time))
        filename = f"{safe_ticker}_{safe_start}.wav"
        write(filename, self.sample_rate, (total_audio * 32767).astype(np.int16))

        print(f"--- EXPORT COMPLETE: {filename} ---")


def _parse_args():
    p = argparse.ArgumentParser(description='IndustrialDrone synth')
    p.add_argument('--ticker', '-t', default='CL=F', help='Ticker symbol')
    p.add_argument('--start', '-s', default=None, help='Start date YYYY-MM-DD')
    p.add_argument('--end', '-e', default=None, help='End date YYYY-MM-DD')
    p.add_argument('--interval', '-i', default='1m', choices=['1m', '5m', '15m', '1h', '1d'], help='Yahoo data interval')
    p.add_argument('--sr', default=44100, type=int, help='Sample rate')
    p.add_argument('--seconds-per-tick', default=1.0, type=float, help='Seconds per market tick')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    today = datetime.now().date()
    # sensible defaults: last 7 days
    if args.start is None:
        args.start = (today - timedelta(days=7)).isoformat()
    if args.end is None:
        args.end = today.isoformat()

    # Parse start/end dates and validate
    try:
        start_date = datetime.fromisoformat(args.start).date()
    except Exception:
        try:
            start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
        except Exception:
            raise ValueError('Invalid --start date. Use YYYY-MM-DD')

    try:
        end_date = datetime.fromisoformat(args.end).date()
    except Exception:
        try:
            end_date = datetime.strptime(args.end, '%Y-%m-%d').date()
        except Exception:
            raise ValueError('Invalid --end date. Use YYYY-MM-DD')

    # Clamp end_date to today
    if end_date > today:
        print('End date is in the future; clamping to today')
        end_date = today

    # If user requested 1-minute data, Yahoo restricts to the last ~30 days.
    interval = args.interval
    if interval == '1m':
        max_lookback = today - timedelta(days=30)
        if start_date < max_lookback:
            print('Requested 1m interval beginning earlier than 30 days ago. Truncating start date to last 30 days to satisfy Yahoo API.')
            start_date = max_lookback

    synth = IndustrialDroneSynth(sample_rate=args.sr)
    synth.generate_drone(
        ticker_symbol=args.ticker,
        start_time=start_date.isoformat(),
        end_time=end_date.isoformat(),
        seconds_per_tick=args.seconds_per_tick,
        interval=interval,
    )