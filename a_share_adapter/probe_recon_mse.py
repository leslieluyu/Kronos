"""
Read-only diagnostic: measure pure reconstruction MSE (z vs x) for the
current fine-tuned Tokenizer checkpoint vs. the original pretrained base.
Does NOT touch the live training process — only reads the checkpoint files
on disk, so it's safe to run alongside train_mini.py.

Appends a row to outputs/mini_finetune/mse_probe.csv each run, so repeated
runs build up a trend over the course of training.

Usage:
    .venv/bin/python a_share_adapter/probe_recon_mse.py
"""
import sys, json, csv, time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import KronosTokenizer
from train_mini import list_stocks, load_stock, config

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "outputs" / "mini_finetune" / "tokenizer_checkpoints" / "best_model"
PROBE_CSV = ROOT / "outputs" / "mini_finetune" / "mse_probe.csv"

torch.manual_seed(42)
np.random.seed(42)


def build_probe_batch(n_samples=200, lookback=90):
    """Fixed, reproducible sample of real A-share windows, z-score normalized
    the same way train_mini.py does."""
    stocks = list_stocks()
    rng = np.random.RandomState(42)
    picked = rng.choice(stocks, size=min(400, len(stocks)), replace=False)

    windows = []
    cols = ["open", "high", "low", "close", "volume", "amount"]
    for sym in picked:
        if len(windows) >= n_samples:
            break
        df = load_stock(sym)
        if df is None or len(df) < lookback + 20:
            continue
        i = rng.randint(0, len(df) - lookback)
        x_df = df.iloc[i:i + lookback]
        if len(x_df) != lookback or x_df.isnull().any().any():
            continue
        x = x_df[cols].values.astype(np.float32)
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -config.data_clip, config.data_clip)
        windows.append(x)

    return torch.from_numpy(np.stack(windows))


@torch.no_grad()
def recon_mse(tokenizer, x, batch_size=32):
    tokenizer.eval()
    total, n = 0.0, 0
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size]
        (z_pre, z), bsq_loss, _, _ = tokenizer(xb)
        total += F.mse_loss(z, xb, reduction="sum").item()
        n += xb.numel()
    return total / n


def main():
    print("Building fixed probe batch from real A-share data ...")
    x = build_probe_batch()
    print(f"  Probe batch: {x.shape}")

    print("\nLoading pretrained base Tokenizer-2k (no fine-tuning) ...")
    base_tok = KronosTokenizer.from_pretrained(config.tokenizer_id)
    base_mse = recon_mse(base_tok, x)
    print(f"  Pretrained base recon MSE: {base_mse:.6f}")

    epoch_info = "n/a"
    if CKPT_DIR.exists():
        print(f"\nLoading fine-tuned checkpoint from {CKPT_DIR} ...")
        ft_tok = KronosTokenizer.from_pretrained(str(CKPT_DIR))
        ft_mse = recon_mse(ft_tok, x)
        print(f"  Fine-tuned recon MSE:      {ft_mse:.6f}")
        state_file = CKPT_DIR / "training_state.json"
        if state_file.exists():
            epoch_info = json.loads(state_file.read_text()).get("total_epochs", "n/a")
        delta_pct = (ft_mse - base_mse) / base_mse * 100
        print(f"\n  Change vs pretrained base: {delta_pct:+.2f}% ({'better' if delta_pct < 0 else 'worse'})")
        print(f"  Fine-tuned checkpoint total_epochs so far: {epoch_info}")
    else:
        ft_mse = None
        print(f"\nNo fine-tuned checkpoint found yet at {CKPT_DIR}")

    PROBE_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not PROBE_CSV.exists()
    with open(PROBE_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "total_epochs", "pretrained_base_mse", "finetuned_mse"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), epoch_info, f"{base_mse:.6f}",
                    f"{ft_mse:.6f}" if ft_mse is not None else ""])
    print(f"\nAppended result to {PROBE_CSV}")


if __name__ == "__main__":
    main()
