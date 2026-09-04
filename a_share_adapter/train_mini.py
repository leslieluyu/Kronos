"""
Kronos-mini Fine-Tuning on A-Share Data (CPU).

Steps:
1. Load pre-trained Kronos-mini + Tokenizer-2k from HuggingFace
2. Build training dataset from kline_cache JSON files
3. Fine-tune Tokenizer (stage 1)
4. Fine-tune Predictor (stage 2)
5. Save checkpoints

Usage:
    .venv/bin/python a_share_adapter/train_mini.py --epochs 5 --max-stocks 100
"""

import sys, os, time, json, math, argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import Kronos, KronosTokenizer

# ── Configuration ──────────────────────────────────────────────

class TrainConfig:
    # Data
    kline_dir = "/home/dev/quant/stock_tracker/data/kline_cache"
    output_dir = "./outputs/mini_finetune"

    # Model
    tokenizer_id = "NeoQuasar/Kronos-Tokenizer-2k"
    model_id = "NeoQuasar/Kronos-mini"

    # Training hyperparams (reduced for CPU test)
    lookback = 90
    predict_len = 10
    max_context = 2048  # mini supports longer context

    epochs = 5  # reduced for test; paper uses 30
    batch_size = 4
    n_train_iter = 4000  # reduced; paper uses 100k
    n_val_iter = 400

    tokenizer_lr = 2e-4
    predictor_lr = 4e-5
    adam_beta1 = 0.9
    adam_beta2 = 0.95
    weight_decay = 0.1
    clip = 5.0  # gradient clip norm
    data_clip = 5.0  # input z-score clip range — must match KronosPredictor.clip for train/inference parity
    seed = 100

    # Data sampling
    max_stocks = 100  # max stocks to load (out of 5003)
    max_samples_per_stock = 50
    start_date = "2020-01-01"
    end_date = "2025-12-31"
    val_split_date = "2025-01-01"  # samples fully before this -> train, fully after -> val (temporal split, no leakage)

    device = "cpu"


config = TrainConfig()

# ── Data Helpers ────────────────────────────────────────────────

def load_stock(symbol):
    p = Path(config.kline_dir) / f"{symbol}.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    ncols = len(data[0])
    if ncols == 6:
        cols = ["date","open","high","low","close","volume"]
    elif ncols == 7:
        cols = ["date","open","high","low","close","volume","amount"]
    elif ncols == 8:
        cols = ["date","open","high","low","close","volume","amount","turnover"]
    else:
        cols = ["date"] + [f"c{i}" for i in range(1, ncols)]
    df = pd.DataFrame(data, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "amount" not in df.columns:
        df["amount"] = (df["open"]+df["high"]+df["low"]+df["close"])/4*df["volume"]
    df = df[df["volume"] > 0]
    df = df[df.index >= pd.Timestamp(config.start_date)]
    df = df[df.index <= pd.Timestamp(config.end_date)]
    return df


def list_stocks():
    return sorted([
        f.stem for f in Path(config.kline_dir).glob("*.json")
        if f.stem.startswith(("sh.", "sz.")) and f.stem.split(".")[1][0] not in "015"
    ])


class KronosDataset(Dataset):
    """Sliding-window dataset for Kronos training."""

    def __init__(self, samples: list):
        self.samples = samples
        self.n_features = 6  # OHLCV + amount

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_df, _x_ts, _y_ts = self.samples[idx]
        cols = ["open","high","low","close","volume","amount"]
        x = x_df[cols].values.astype(np.float32)

        # Z-score normalize per-window, per-column — matches KronosPredictor.predict()
        # (model/kronos.py) so the fine-tuned weights match the inference-time distribution.
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -config.data_clip, config.data_clip)

        return torch.from_numpy(x)


def build_samples(symbols, max_stocks=100, max_per_stock=50):
    """Build sliding-window samples from stock files, split by TIME (not a random
    shuffle) to avoid leakage: a random split can put two heavily-overlapping windows
    from the same stock on opposite sides, letting validation loss "see" near-duplicate
    training data and look artificially good. Any window straddling val_split_date is
    dropped entirely rather than assigned to either side."""
    train_samples, val_samples = [], []
    lookback = config.lookback
    pred_len = config.predict_len
    val_cutoff = pd.Timestamp(config.val_split_date)

    for sym in tqdm(symbols[:max_stocks], desc="Loading stocks"):
        df = load_stock(sym)
        if df is None or len(df) < lookback + pred_len + 10:
            continue

        total = len(df)
        step = max(1, (total - lookback - pred_len) // max_per_stock)
        stock_train, stock_val = [], []
        for i in range(0, total - lookback - pred_len, step):
            x_df = df.iloc[i:i+lookback]
            x_ts = df.iloc[i:i+lookback].index
            y_ts = df.iloc[i+lookback:i+lookback+pred_len].index
            if len(x_df) != lookback or x_df.isnull().any().any():
                continue
            if y_ts[-1] < val_cutoff:
                stock_train.append((x_df, x_ts, y_ts))
            elif x_ts[0] >= val_cutoff:
                stock_val.append((x_df, x_ts, y_ts))
            # else: window straddles val_cutoff -> dropped to avoid leakage
            if len(stock_train) + len(stock_val) >= max_per_stock:
                break

        train_samples.extend(stock_train)
        val_samples.extend(stock_val)

    np.random.seed(config.seed)
    np.random.shuffle(train_samples)
    np.random.shuffle(val_samples)
    return train_samples, val_samples


# ── Training Helpers ────────────────────────────────────────────

def load_training_state(ckpt_path):
    """Read {best_loss, total_epochs} sidecar written next to a saved checkpoint, if any."""
    state_file = Path(ckpt_path) / "training_state.json"
    if state_file.exists():
        return json.loads(state_file.read_text())
    return {"best_loss": float("inf"), "total_epochs": 0}


def save_training_state(ckpt_path, opt, best_loss, total_epochs):
    """Persist optimizer state + best_loss/epoch bookkeeping so a later run can resume properly."""
    torch.save(opt.state_dict(), Path(ckpt_path) / "optimizer.pt")
    (Path(ckpt_path) / "training_state.json").write_text(
        json.dumps({"best_loss": best_loss, "total_epochs": total_epochs})
    )


def maybe_load_optimizer(opt, resume_from):
    if resume_from:
        opt_path = Path(resume_from) / "optimizer.pt"
        if opt_path.exists():
            opt.load_state_dict(torch.load(opt_path, map_location=config.device))
            print(f"  Resumed optimizer state from {opt_path}")


@torch.no_grad()
def eval_tokenizer_loss(tokenizer, val_loader):
    if val_loader is None or len(val_loader) == 0:
        return None
    tokenizer.eval()
    total_loss = 0.0
    for batch in val_loader:
        x = batch.float().to(config.device)
        (z_pre, z), bsq_loss, _, _ = tokenizer(x)
        loss = F.mse_loss(z, x) + bsq_loss
        total_loss += loss.item()
    return total_loss / len(val_loader)


@torch.no_grad()
def eval_predictor_loss(tokenizer, model, val_loader):
    if val_loader is None or len(val_loader) == 0:
        return None
    tokenizer.eval()
    model.eval()
    total_loss = 0.0
    for batch in val_loader:
        x = batch.float().to(config.device)
        x_token = tokenizer.encode(x, half=True)
        s1, s2 = x_token[0], x_token[1]
        s1_logits, s2_logits = model(s1, s2)
        target = s1[:, 1:]
        logits_pred = s1_logits[:, :-1, :]
        loss = F.cross_entropy(
            logits_pred.reshape(-1, logits_pred.size(-1)),
            target.reshape(-1).clamp(0, logits_pred.size(-1)-1))
        total_loss += loss.item()
    return total_loss / len(val_loader)


def save_epoch_snapshot(model, save_dir, total_epoch, train_loss, val_loss):
    """Unconditionally save this epoch's weights (no optimizer state) to its own folder,
    so a later probe can pick the true-best epoch instead of trusting the training loss
    (which can diverge from actual reconstruction/prediction quality, e.g. BSQ entropy
    regularization moving total loss down while MSE moves up)."""
    epoch_dir = Path(save_dir) / "epochs" / f"epoch_{total_epoch}"
    model.save_pretrained(str(epoch_dir))
    (epoch_dir / "epoch_info.json").write_text(
        json.dumps({"epoch": total_epoch, "train_loss": train_loss, "val_loss": val_loss})
    )


def train_tokenizer(tokenizer, train_dataset, val_dataset, resume_from=None):
    print(f"\n{'='*60}")
    print(f"Stage 1: Tokenizer Fine-Tuning")
    print(f"{'='*60}")

    tokenizer.train()
    for p in tokenizer.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(
        tokenizer.parameters(), lr=config.tokenizer_lr,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )

    n_iter = min(config.n_train_iter, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False) if len(val_dataset) > 0 else None

    save_dir = Path(config.output_dir) / "tokenizer_checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    # best_loss tracks validation loss when a val set exists (the honest, leakage-free
    # signal); falls back to train loss only when there's no val data (e.g. tiny test runs).
    state = load_training_state(resume_from) if resume_from else {"best_loss": float("inf"), "total_epochs": 0}
    best_loss = state["best_loss"]
    prior_epochs = state["total_epochs"]
    maybe_load_optimizer(opt, resume_from)
    if resume_from:
        print(f"  Resuming: prior best_loss={best_loss:.4f}, prior_epochs={prior_epochs}")

    for epoch in range(config.epochs):
        tokenizer.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Tok Epoch {epoch+1}/{config.epochs}")
        for batch in pbar:
            x = batch.float().to(config.device)
            (z_pre, z), bsq_loss, _, _ = tokenizer(x)
            loss = F.mse_loss(z, x) + bsq_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), config.clip)
            opt.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = total_loss / len(pbar)
        val_loss = eval_tokenizer_loss(tokenizer, val_loader)
        val_str = f"{val_loss:.4f}" if val_loss is not None else "n/a"
        print(f"  Epoch {epoch+1}: train loss={train_loss:.4f}  val loss={val_str}")

        total_epoch = prior_epochs + epoch + 1
        save_epoch_snapshot(tokenizer, save_dir, total_epoch, train_loss, val_loss)
        print(f"  Saved epoch snapshot to {save_dir}/epochs/epoch_{total_epoch}")

        # Save best (by val loss when available, else train loss)
        selection_loss = val_loss if val_loss is not None else train_loss
        if selection_loss < best_loss:
            best_loss = selection_loss
            ckpt_path = save_dir / "best_model"
            tokenizer.save_pretrained(str(ckpt_path))
            save_training_state(ckpt_path, opt, best_loss, total_epoch)
            print(f"  Saved best tokenizer to {ckpt_path} (total_epochs={total_epoch}, selection_loss={selection_loss:.4f})")

    # Restore best
    tokenizer = KronosTokenizer.from_pretrained(str(save_dir / "best_model"))
    tokenizer.eval()
    print("Tokenizer fine-tuning complete.")
    return tokenizer


def train_predictor(tokenizer, model, train_dataset, val_dataset, resume_from=None):
    print(f"\n{'='*60}")
    print(f"Stage 2: Predictor Fine-Tuning")
    print(f"{'='*60}")

    tokenizer.eval()
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(
        model.parameters(), lr=config.predictor_lr,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False) if len(val_dataset) > 0 else None

    save_dir = Path(config.output_dir) / "predictor_checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    # best_loss tracks validation loss when a val set exists (the honest, leakage-free
    # signal); falls back to train loss only when there's no val data (e.g. tiny test runs).
    state = load_training_state(resume_from) if resume_from else {"best_loss": float("inf"), "total_epochs": 0}
    best_loss = state["best_loss"]
    prior_epochs = state["total_epochs"]
    maybe_load_optimizer(opt, resume_from)
    if resume_from:
        print(f"  Resuming: prior best_loss={best_loss:.4f}, prior_epochs={prior_epochs}")

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Pred Epoch {epoch+1}/{config.epochs}")
        for batch in pbar:
            x = batch.float().to(config.device)
            with torch.no_grad():
                x_token = tokenizer.encode(x, half=True)
                s1, s2 = x_token[0], x_token[1]

            s1_logits, s2_logits = model(s1, s2)
            target = s1[:, 1:]
            logits_pred = s1_logits[:, :-1, :]
            loss = F.cross_entropy(
                logits_pred.reshape(-1, logits_pred.size(-1)),
                target.reshape(-1).clamp(0, logits_pred.size(-1)-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            opt.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = total_loss / len(pbar)
        val_loss = eval_predictor_loss(tokenizer, model, val_loader)
        val_str = f"{val_loss:.4f}" if val_loss is not None else "n/a"
        print(f"  Epoch {epoch+1}: train loss={train_loss:.4f}  val loss={val_str}")

        total_epoch = prior_epochs + epoch + 1
        save_epoch_snapshot(model, save_dir, total_epoch, train_loss, val_loss)
        print(f"  Saved epoch snapshot to {save_dir}/epochs/epoch_{total_epoch}")

        selection_loss = val_loss if val_loss is not None else train_loss
        if selection_loss < best_loss:
            best_loss = selection_loss
            ckpt_path = save_dir / "best_model"
            model.save_pretrained(str(ckpt_path))
            save_training_state(ckpt_path, opt, best_loss, total_epoch)
            print(f"  Saved best predictor to {ckpt_path} (total_epochs={total_epoch}, selection_loss={selection_loss:.4f})")

    model = Kronos.from_pretrained(str(save_dir / "best_model"))
    model.eval()
    print("Predictor fine-tuning complete.")
    return model


# ── Main ────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-stocks", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=config.batch_size)
    p.add_argument("--skip-tokenizer", action="store_true")
    p.add_argument("--skip-predictor", action="store_true")
    p.add_argument("--resume-tokenizer", type=str, default=None,
                    help="Local checkpoint dir to resume tokenizer from (e.g. outputs/mini_finetune/tokenizer_checkpoints/best_model). "
                         "Continues training/best_loss/optimizer state instead of starting from the HF hub base model.")
    p.add_argument("--resume-predictor", type=str, default=None,
                    help="Local checkpoint dir to resume predictor from (e.g. outputs/mini_finetune/predictor_checkpoints/best_model).")
    p.add_argument("--output-dir", type=str, default=config.output_dir,
                    help="Directory to write checkpoints/final model to (default: %(default)s).")
    args = p.parse_args()

    config.epochs = args.epochs
    config.max_stocks = args.max_stocks
    config.batch_size = args.batch_size
    config.output_dir = args.output_dir

    print(f"{'='*60}")
    print(f"Kronos-mini Fine-Tuning on A-Share Data")
    print(f"{'='*60}")
    print(f"  Epochs: {config.epochs}, Max stocks: {config.max_stocks}")
    print(f"  Batch: {config.batch_size}, Device: {config.device}")
    print(f"  Output: {config.output_dir}")

    # ── Load Models ──
    if args.resume_tokenizer:
        print(f"\nResuming Tokenizer from {args.resume_tokenizer} ...")
        tokenizer = KronosTokenizer.from_pretrained(args.resume_tokenizer)
    else:
        print(f"\nLoading Kronos-Tokenizer-2k (pretrained base) ...")
        tokenizer = KronosTokenizer.from_pretrained(config.tokenizer_id)

    if args.resume_predictor:
        print(f"Resuming Predictor from {args.resume_predictor} ...")
        model = Kronos.from_pretrained(args.resume_predictor)
    else:
        print(f"Loading Kronos-mini (pretrained base) ...")
        model = Kronos.from_pretrained(config.model_id)
    print(f"  Tokenizer: {sum(p.numel() for p in tokenizer.parameters()):,} params")
    print(f"  Predictor: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Build Dataset ──
    stocks = list_stocks()
    print(f"\nFound {len(stocks)} A-share stocks")
    print(f"Building dataset from {min(config.max_stocks, len(stocks))} stocks ...")

    train_samples, val_samples = build_samples(
        stocks, max_stocks=config.max_stocks, max_per_stock=config.max_samples_per_stock
    )
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples:   {len(val_samples)}")

    train_ds = KronosDataset(train_samples)
    val_ds = KronosDataset(val_samples)

    # ── Stage 1: Tokenizer ──
    if not args.skip_tokenizer:
        tokenizer = train_tokenizer(tokenizer, train_ds, val_ds, resume_from=args.resume_tokenizer)

    # ── Stage 2: Predictor ──
    if not args.skip_predictor:
        model = train_predictor(tokenizer, model, train_ds, val_ds, resume_from=args.resume_predictor)

    # ── Save final ──
    final_dir = Path(config.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(final_dir / "tokenizer"))
    model.save_pretrained(str(final_dir / "predictor"))
    print(f"\n{'='*60}")
    print(f"Fine-tuning complete! Models saved to {final_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()