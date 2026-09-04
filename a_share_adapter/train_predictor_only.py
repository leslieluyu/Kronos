"""
Predictor-only fine-tuning. Uses pre-trained Tokenizer checkpoint.
Usage: .venv/bin/python a_share_adapter/train_predictor_only.py --epochs 30 --max-stocks 200
"""
import sys, os, time, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Kronos, KronosTokenizer

# ── Config ──────────────────────────────────────────────
class Config:
    kline_dir = "/home/dev/quant/stock_tracker/data/kline_cache"
    tokenizer_ckpt = "./outputs/mini_finetune/tokenizer_checkpoints/best_model"
    predictor_ckpt = "./outputs/mini_finetune/predictor_checkpoints"
    output_dir = "./outputs/mini_finetune"
    lookback = 90; predict_len = 10
    epochs = 30; batch_size = 4
    lr = 4e-5; adam_beta1 = 0.9; adam_beta2 = 0.95; weight_decay = 0.1; clip = 5.0
    seed = 100; device = "cpu"
    max_stocks = 200; max_samples_per_stock = 50
    start_date = "2020-01-01"; end_date = "2025-12-31"

cfg = Config()

# ── Data ────────────────────────────────────────────────
def load_stock(symbol):
    p = Path(cfg.kline_dir) / f"{symbol}.json"
    if not p.exists(): return None
    data = json.loads(p.read_text())
    ncols = len(data[0])
    cols_map = {6: ["date","open","high","low","close","volume"],
                7: ["date","open","high","low","close","volume","amount"],
                8: ["date","open","high","low","close","volume","amount","turnover"]}
    cols = cols_map.get(ncols, ["date"] + [f"c{i}" for i in range(ncols)])
    df = pd.DataFrame(data, columns=cols)
    df["date"] = pd.to_datetime(df["date"]); df = df.set_index("date").sort_index()
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "amount" not in df.columns:
        df["amount"] = (df["open"]+df["high"]+df["low"]+df["close"])/4*df["volume"]
    df = df[df["volume"] > 0]
    df = df[(df.index >= pd.Timestamp(cfg.start_date)) & (df.index <= pd.Timestamp(cfg.end_date))]
    return df

def list_stocks():
    return sorted([f.stem for f in Path(cfg.kline_dir).glob("*.json")
                   if f.stem.startswith(("sh.","sz.")) and f.stem.split(".")[1][0] not in "015"])

class KronosDataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x_df, _, _ = self.samples[idx]
        x = x_df[["open","high","low","close","volume","amount"]].values.astype(np.float32)
        # Z-score normalize per-window, per-column — matches KronosPredictor.predict()
        # (model/kronos.py) so the fine-tuned weights match the inference-time distribution.
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -5.0, 5.0)
        return torch.from_numpy(x)

def build_samples(symbols):
    all_samples, L, P = [], cfg.lookback, cfg.predict_len
    for sym in tqdm(symbols[:cfg.max_stocks], desc="Loading stocks"):
        df = load_stock(sym)
        if df is None or len(df) < L + P + 10: continue
        step = max(1, (len(df) - L - P) // cfg.max_samples_per_stock)
        for i in range(0, len(df) - L - P, step):
            x_df = df.iloc[i:i+L]
            if len(x_df) == L and not x_df.isnull().any().any():
                all_samples.append((x_df, df.iloc[i:i+L].index, df.iloc[i+L:i+L+P].index))
    np.random.seed(cfg.seed); np.random.shuffle(all_samples)
    n = int(len(all_samples)*0.8)
    return all_samples[:n], all_samples[n:]

# ── Training ────────────────────────────────────────────
def train_predictor(tokenizer, model, train_ds):
    tokenizer.eval(); model.train()
    for p in model.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            betas=(cfg.adam_beta1, cfg.adam_beta2), weight_decay=cfg.weight_decay)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    best_loss = float("inf")
    Path(cfg.predictor_ckpt).mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train(); total_loss = 0.0
        pbar = tqdm(loader, desc=f"Pred Epoch {epoch+1}/{cfg.epochs}")
        for batch in pbar:
            x = batch.float().to(cfg.device)
            with torch.no_grad():
                tok_out = tokenizer.encode(x, half=True)
                s1, s2 = tok_out[0], tok_out[1]
            s1_logits, s2_logits = model(s1, s2)
            target = s1[:, 1:]; lp = s1_logits[:, :-1, :]
            loss = F.cross_entropy(lp.reshape(-1, lp.size(-1)), target.reshape(-1).clamp(0, lp.size(-1)-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip); opt.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg = total_loss / len(loader)
        print(f"  Epoch {epoch+1} avg loss: {avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            model.save_pretrained(str(Path(cfg.predictor_ckpt) / "best_model"))
            print(f"  Saved best to {cfg.predictor_ckpt}/best_model")

    return Kronos.from_pretrained(str(Path(cfg.predictor_ckpt) / "best_model"))

# ── Main ────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--max-stocks", type=int, default=200)
    args = p.parse_args()
    cfg.epochs = args.epochs; cfg.max_stocks = args.max_stocks

    print(f"{'='*60}\nKronos-mini Predictor Fine-Tuning\n{'='*60}")
    print(f"  Epochs: {cfg.epochs}, Stocks: {cfg.max_stocks}, Batch: {cfg.batch_size}")

    print(f"\nLoading best Tokenizer from {cfg.tokenizer_ckpt} ...")
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_ckpt)
    tokenizer.eval()
    print(f"  Tokenizer loaded ({sum(p.numel() for p in tokenizer.parameters()):,} params)")

    print(f"\nLoading Kronos-mini predictor ...")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
    print(f"  Predictor loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    stocks = list_stocks()
    print(f"\nFound {len(stocks)} A-shares, loading {cfg.max_stocks} ...")
    train_samples, _ = build_samples(stocks)
    print(f"  Train samples: {len(train_samples)}")
    train_ds = KronosDataset(train_samples)

    model = train_predictor(tokenizer, model, train_ds)

    final = Path(cfg.output_dir) / "final_predictor"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final))
    print(f"\n{'='*60}\nPredictor fine-tuning complete!\nModel saved to {final}\n{'='*60}")

if __name__ == "__main__":
    main()