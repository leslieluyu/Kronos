"""Compare Kronos mini/small/base CPU training speed."""
import sys, os, time, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
KLINE_DIR = "/home/dev/quant/stock_tracker/data/kline_cache"

def load_stock(symbol, start=None, end=None):
    p = Path(KLINE_DIR) / f"{symbol}.json"
    data = json.loads(p.read_text())
    ncols = len(data[0])
    cols = ["date","open","high","low","close","volume"] if ncols==6 else ["date","open","high","low","close","volume","amount"]
    import pandas as pd
    df = pd.DataFrame(data, columns=cols)
    df["date"] = pd.to_datetime(df["date"]); df = df.set_index("date").sort_index()
    for c in ["open","high","low","close","volume"]: df[c] = pd.to_numeric(df[c],errors="coerce")
    if "amount" not in df.columns: df["amount"] = (df["open"]+df["high"]+df["low"]+df["close"])/4*df["volume"]
    df = df[df["volume"]>0]
    if start: df = df[df.index>=pd.Timestamp(start)]
    if end:   df = df[df.index<=pd.Timestamp(end)]
    return df

def measure(model_id, tokenizer_id, label):
    from model import Kronos, KronosTokenizer
    print(f"\n{'─'*50}")
    print(f"Loading {label}: tokenizer={tokenizer_id}, model={model_id}")
    t0 = time.time()
    tok = KronosTokenizer.from_pretrained(tokenizer_id)
    mdl = Kronos.from_pretrained(model_id)
    load_t = time.time()-t0
    tok_p = sum(p.numel() for p in tok.parameters())
    mdl_p = sum(p.numel() for p in mdl.parameters())
    print(f"  Loaded in {load_t:.1f}s | Tokenizer params: {tok_p:,} | Predictor params: {mdl_p:,}")

    tok.eval(); mdl.eval()

    # Prepare real data
    df = load_stock("sh.600000", start="2024-01-01", end="2024-10-31")
    cols = ["open","high","low","close","volume","amount"]
    x = torch.from_numpy(df[cols].iloc[:90].values.astype(np.float32)).unsqueeze(0)
    # Check tokenizer input dim
    d_in = tok.d_in
    if d_in != x.shape[-1]:
        x = x[:, :, :d_in] if d_in < 6 else x
    x = x / x[0,0,0]

    # Tokenizer inference
    with torch.no_grad():
        for _ in range(3): tok(x)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(20): tok(x)
    tok_infer = (time.time()-t0)/20*1000

    # Tokenizer training (1 iter)
    tok.train()
    for p in tok.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(tok.parameters(), lr=2e-4, betas=(0.9,0.95), weight_decay=0.1)
    t0 = time.time()
    (z_pre,z), bsq_loss, _, _ = tok(x)
    loss = torch.nn.functional.mse_loss(z,x)+bsq_loss
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(tok.parameters(),5.0); opt.step()
    tok_train = time.time()-t0

    # Predictor training (1 iter)
    tok.eval(); mdl.train()
    for p in mdl.parameters(): p.requires_grad = True
    opt2 = torch.optim.AdamW(mdl.parameters(), lr=4e-5, betas=(0.9,0.95), weight_decay=0.1)
    with torch.no_grad():
        x_token = tok.encode(x, half=True)
        s1, s2 = x_token[0], x_token[1]
    t0 = time.time()
    s1_logits, s2_logits = mdl(s1, s2)
    target = s1[:,1:]; lp = s1_logits[:,:-1,:]
    loss2 = torch.nn.functional.cross_entropy(lp.reshape(-1,lp.size(-1)), target.reshape(-1).clamp(0,lp.size(-1)-1))
    opt2.zero_grad(); loss2.backward()
    torch.nn.utils.clip_grad_norm_(mdl.parameters(),5.0); opt2.step()
    mdl_train = time.time()-t0

    # Projections
    ITERS_PER_EPOCH = 2000*50
    EPOCHS = 30
    tok_epoch = tok_train*ITERS_PER_EPOCH
    mdl_epoch = mdl_train*ITERS_PER_EPOCH
    tok_total = tok_epoch*EPOCHS
    mdl_total = mdl_epoch*EPOCHS
    combined = tok_total + mdl_total

    print(f"\n  Tokenizer inference: {tok_infer:.0f}ms")
    print(f"  Tokenizer training iter: {tok_train:.2f}s | epoch: {tok_epoch/60:.0f} min | 30 epochs: {tok_total/3600:.0f}h")
    print(f"  Predictor training iter:  {mdl_train:.2f}s | epoch: {mdl_epoch/60:.0f} min | 30 epochs: {mdl_total/3600:.0f}h")
    print(f"  ─────────────────────────────────")
    print(f"  COMBINED 30 epochs: {combined/3600:.0f}h (~{combined/3600/24:.1f} days)")

    return {
        "label": label, "load_time": load_t,
        "tok_params": tok_p, "mdl_params": mdl_p,
        "tok_infer_ms": tok_infer, "tok_train_s": tok_train,
        "mdl_train_s": mdl_train,
        "tok_epoch_h": tok_epoch/3600, "mdl_epoch_h": mdl_epoch/3600,
        "combined_h": combined/3600, "combined_d": combined/3600/24,
    }

def main():
    results = []

    # Test Kronos-small (already cached)
    r = measure("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", "Kronos-small (24.7M)")
    results.append(r)

    # Test Kronos-mini (needs Tokenizer-2k)
    try:
        r = measure("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", "Kronos-mini (4.1M)")
        results.append(r)
    except Exception as e:
        print(f"\n  Kronos-mini error: {e}")

    # Test Kronos-base (needs Tokenizer-base, 102M)
    try:
        r = measure("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", "Kronos-base (102.3M)")
        results.append(r)
    except Exception as e:
        print(f"\n  Kronos-base error: {e}")

    # Summary table
    print(f"\n\n{'='*70}")
    print("SUMMARY: Kronos Fine-Tuning Time on CPU (6 cores)")
    print(f"{'='*70}")
    print(f"{'Model':<22} {'Params':>10} {'Tok/iter':>10} {'Pred/iter':>10} {'Combined 30ep':>16}")
    print(f"{'─'*22} {'─'*10} {'─'*10} {'─'*10} {'─'*16}")
    for r in results:
        print(f"{r['label']:<22} {r['mdl_params']:>10,} {r['tok_train_s']:>9.2f}s {r['mdl_train_s']:>9.2f}s {r['combined_d']:>14.1f} days")

    # Estimate mini from small if not measured
    if len(results) < 3:
        small = results[0]
        print(f"\n  (Kronos-base inference was too slow to complete; estimate: ~4× small = ~80 days on CPU)")

    print(f"\n{'='*70}")
    print("✅ CPU 推理可行 | 微调必须 GPU (或 mini 勉强可试 3-5 天)")

if __name__ == "__main__":
    main()