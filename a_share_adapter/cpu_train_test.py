"""
Kronos CPU Training Efficiency Test with A-Share data.

Usage:
    .venv/bin/python a_share_adapter/cpu_train_test.py [--skip-train] [--skip-predict]
"""
import sys, os, time, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

KLINE_DIR = "/home/dev/quant/stock_tracker/data/kline_cache"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_stock(symbol, start=None, end=None):
    path = Path(KLINE_DIR) / f"{symbol}.json"
    with open(path) as f:
        raw = json.load(f)
    ncols = len(raw[0])
    if ncols == 6:
        cols = ["date", "open", "high", "low", "close", "volume"]
    else:
        cols = ["date", "open", "high", "low", "close", "volume", "amount"]
    df = pd.DataFrame(raw, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "amount" not in df.columns:
        df["amount"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0 * df["volume"]
    df = df[df["volume"] > 0]
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def load_models():
    from model import Kronos, KronosTokenizer
    print("Loading Kronos-Tokenizer-base ..."); t0 = time.time()
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    print(f"  Tokenizer loaded ({time.time()-t0:.1f}s)")
    print("Loading Kronos-small ..."); t0 = time.time()
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    print(f"  Predictor loaded ({time.time()-t0:.1f}s)")
    tokenizer.eval(); model.eval()
    tok_p = sum(p.numel() for p in tokenizer.parameters())
    mdl_p = sum(p.numel() for p in model.parameters())
    print(f"  Tokenizer params: {tok_p:,}")
    print(f"  Predictor params: {mdl_p:,}")
    return tokenizer, model


def test_tokenizer_speed(tokenizer):
    print(f"\n{'='*60}\nTest 1: Tokenizer Inference Speed")
    batch = torch.randn(1, 90, 6)  # OHLCV + amount
    with torch.no_grad():
        for _ in range(5):
            tokenizer(batch)
    n = 20; t0 = time.time()
    with torch.no_grad():
        for _ in range(n):
            (z_pre, z), bsq_loss, quantized, z_indices = tokenizer(batch)
    ms = (time.time() - t0) / n * 1000
    print(f"  Time per call: {ms:.1f}ms  |  z_indices: {z_indices.shape}  |  z_pre: {z_pre.shape}")
    return ms


def test_tokenizer_training(tokenizer):
    print(f"\n{'='*60}\nTest 2: Tokenizer Training (5 iters, batch=1)")

    tokenizer.train()
    for p in tokenizer.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(tokenizer.parameters(), lr=2e-4, betas=(0.9,0.95), weight_decay=0.1)

    df = load_stock("sh.600000", start="2024-01-01", end="2024-10-31")
    cols = ["open", "high", "low", "close", "volume", "amount"]
    x = torch.from_numpy(df[cols].iloc[:90].values.astype(np.float32)).unsqueeze(0)
    x = x / x[0,0,0]

    t0 = time.time()
    for i in range(5):
        (z_pre, z), bsq_loss, quantized, z_indices = tokenizer(x)
        loss = torch.nn.functional.mse_loss(z, x) + bsq_loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 5.0)
        opt.step()
        print(f"  Iter {i}: loss={loss.item():.4f}")
    t = (time.time() - t0) / 5
    print(f"  Avg: {t:.2f}s ({t*1000:.0f}ms)")

    orig_iters = 2000 * 50  # 100k/epoch
    epo = t * orig_iters; tot = epo * 30
    print(f"  Projection (100k iter/epoch, 30 epochs):")
    print(f"    Per epoch: {epo/60:.1f} min  |  Total: {tot/3600:.1f}h (~{tot/3600/24:.1f}d)")
    tokenizer.eval()
    return t


def test_predictor_training(tokenizer, model):
    print(f"\n{'='*60}\nTest 3: Predictor Training (5 iters, batch=1)")

    tokenizer.eval(); model.train()
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=4e-5, betas=(0.9,0.95), weight_decay=0.1)

    df = load_stock("sh.600000", start="2024-01-01", end="2024-10-31")
    cols = ["open", "high", "low", "close", "volume", "amount"]
    x = torch.from_numpy(df[cols].iloc[:90].values.astype(np.float32)).unsqueeze(0)
    x = x / x[0,0,0]

    with torch.no_grad():
        x_token = tokenizer.encode(x, half=True)  # returns (s1_ids, s2_ids)
        s1_ids, s2_ids = x_token[0], x_token[1]

    t0 = time.time()
    for i in range(5):
        s1_logits, s2_logits = model(s1_ids, s2_ids)
        # Simple next-token prediction loss on s1
        target = s1_ids[:, 1:]
        logits_pred = s1_logits[:, :-1, :]
        loss = torch.nn.functional.cross_entropy(
            logits_pred.reshape(-1, logits_pred.size(-1)),
            target.reshape(-1).clamp(0, logits_pred.size(-1)-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        print(f"  Iter {i}: loss={loss.item():.6f}")
    t = (time.time() - t0) / 5
    print(f"  Avg: {t:.2f}s ({t*1000:.0f}ms)")

    orig_iters = 2000 * 50
    epo = t * orig_iters; tot = epo * 30
    print(f"  Projection (100k iter/epoch, 30 epochs):")
    print(f"    Per epoch: {epo/60:.1f} min  |  Total: {tot/3600:.1f}h (~{tot/3600/24:.1f}d)")
    model.eval()
    return t


def test_prediction(tokenizer, model):
    print(f"\n{'='*60}\nTest 4: End-to-End Prediction")
    from model import KronosPredictor
    df = load_stock("sh.600000", start="2023-01-01", end="2024-06-30")
    lookback, pred_len = 200, 20
    x_df = df.iloc[:lookback][["open","high","low","close","volume","amount"]]
    x_ts = pd.Series(df.iloc[:lookback].index)
    y_ts = pd.Series(df.iloc[lookback:lookback+pred_len].index)
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    t0 = time.time()
    pred_df = predictor.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                                 pred_len=pred_len, T=1.0, top_p=0.9, sample_count=1)
    print(f"  Time: {time.time()-t0:.1f}s")
    print(f"  Forecast:\n{pred_df[['close']].to_string()}")
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12,5))
        actual = df["close"].iloc[lookback:lookback+pred_len]
        ax.plot(range(lookback), x_df["close"].values, label="History", color="blue", alpha=0.5)
        ax.plot(range(lookback, lookback+pred_len), actual.values, label="Actual", color="green", marker="o")
        ax.plot(range(lookback, lookback+pred_len), pred_df["close"].values, label="Predicted", color="red", marker="x")
        ax.axvline(x=lookback, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("Kronos Prediction - sh.600000"); ax.legend(); ax.grid(alpha=0.3)
        p = OUTPUT_DIR / "prediction_cpu_test.png"; fig.savefig(p, dpi=100, bbox_inches="tight")
        print(f"  Plot: {p}"); plt.close()
    except Exception as e:
        print(f"  Plot error: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-predict", action="store_true")
    args = p.parse_args()
    print(f"{'='*60}\nKronos CPU Training Test | Cores: {os.cpu_count()}\n{'='*60}")
    tokenizer, model = load_models()
    test_tokenizer_speed(tokenizer)
    if not args.skip_train:
        test_tokenizer_training(tokenizer)
        test_predictor_training(tokenizer, model)
    if not args.skip_predict:
        test_prediction(tokenizer, model)
    print(f"\n{'='*60}\nDone!\n{'='*60}")


if __name__ == "__main__":
    main()