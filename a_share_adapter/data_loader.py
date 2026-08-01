"""
A-Share K-line data loader for Kronos.
Reads JSON files from /home/dev/quant/stock_tracker/data/kline_cache/
and converts to Kronos-compatible DataFrame format.
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional


KLINE_CACHE_DIR = "/home/dev/quant/stock_tracker/data/kline_cache"

# JSON files have 7 columns: date, open, high, low, close, volume, amount
JSON_COLS = ["date", "open", "high", "low", "close", "volume", "amount"]
KRONOS_COLS = ["open", "high", "low", "close", "volume", "amount"]


def list_available_stocks(base_dir: str = KLINE_CACHE_DIR) -> List[str]:
    """List all stock symbols from the kline_cache directory."""
    path = Path(base_dir)
    files = sorted(path.glob("*.json"))
    return [f.stem for f in files]


def load_stock_kline(
    symbol: str,
    base_dir: str = KLINE_CACHE_DIR,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a single stock's K-line data.

    Returns:
        DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'amount']
        index: pd.DatetimeIndex
    """
    filepath = Path(base_dir) / f"{symbol}.json"
    if not filepath.exists():
        raise FileNotFoundError(f"No data for {symbol} at {filepath}")

    with open(filepath) as f:
        raw = json.load(f)

    df = pd.DataFrame(raw, columns=JSON_COLS)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Convert strings to float
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with zero volume (suspended trading)
    df = df[df["volume"] > 0]

    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]

    return df


def prepare_sliding_window_samples(
    df: pd.DataFrame,
    lookback: int = 90,
    predict_len: int = 10,
    max_samples: int = 100,
) -> List[Tuple[pd.DataFrame, pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Prepare sliding-window samples for Kronos training/inference.
    """
    samples = []
    total_len = len(df)

    step = max(1, (total_len - lookback - predict_len) // max_samples)
    start_idxs = range(0, total_len - lookback - predict_len, step)

    for idx in start_idxs:
        if len(samples) >= max_samples:
            break
        x_df = df.iloc[idx : idx + lookback][KRONOS_COLS].copy()
        x_ts = df.iloc[idx : idx + lookback].index.copy()
        y_ts = df.iloc[idx + lookback : idx + lookback + predict_len].index.copy()

        if len(x_df) < lookback or len(y_ts) < predict_len:
            continue
        if x_df.isnull().any().any():
            continue

        samples.append((x_df, x_ts, y_ts))

    return samples


def build_training_dataset(
    symbols: List[str],
    lookback: int = 90,
    predict_len: int = 10,
    samples_per_stock: int = 10,
    max_stocks: int = 5,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
) -> Dict:
    """Build a small training dataset from a set of stock symbols."""
    all_samples = []

    for symbol in symbols[:max_stocks]:
        try:
            df = load_stock_kline(symbol, start_date=start_date, end_date=end_date)
            if len(df) < lookback + predict_len + 10:
                print(f"  [SKIP] {symbol}: insufficient data (len={len(df)})")
                continue
            samples = prepare_sliding_window_samples(
                df, lookback=lookback, predict_len=predict_len, max_samples=samples_per_stock
            )
            all_samples.extend(samples)
            print(f"  [OK] {symbol}: {len(samples)} samples")
        except Exception as e:
            print(f"  [ERR] {symbol}: {e}")

    split = int(len(all_samples) * 0.8)
    return {"train_samples": all_samples[:split], "val_samples": all_samples[split:]}


def load_dataset_for_finetuning(
    symbols: Optional[List[str]] = None,
    lookback: int = 90,
    predict_len: int = 10,
    max_stocks: int = 5,
    max_samples_per_stock: int = 10,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
) -> Dict:
    """Convenience wrapper: build a dataset ready for Kronos fine-tuning."""
    if symbols is None:
        all_stocks = list_available_stocks()
        symbols = all_stocks[:max_stocks]

    print(f"Building dataset from {len(symbols[:max_stocks])} stocks...")
    dataset = build_training_dataset(
        symbols=symbols, lookback=lookback, predict_len=predict_len,
        samples_per_stock=max_samples_per_stock, max_stocks=max_stocks,
        start_date=start_date, end_date=end_date,
    )
    print(f"  Train samples: {len(dataset['train_samples'])}")
    print(f"  Val samples:   {len(dataset['val_samples'])}")
    return dataset


if __name__ == "__main__":
    stocks = list_available_stocks()
    print(f"Available stocks: {len(stocks)}")
    print(f"First 10: {stocks[:10]}")

    df = load_stock_kline("sh.600000", start_date="2023-01-01", end_date="2024-12-31")
    print(f"\nsh.600000 shape: {df.shape}")
    print(df.head())
    print(df.tail())

    print("\n--- Building test dataset ---")
    ds = load_dataset_for_finetuning(
        symbols=stocks[:5], lookback=90, predict_len=10,
        max_stocks=5, max_samples_per_stock=10,
    )