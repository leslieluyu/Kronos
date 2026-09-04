"""
Historical Backtest: Evaluate fine-tuned Kronos-mini on A-Share historical data.

For each stock, pick a historical cutoff date, use prior data as input,
predict the next N days' prices, and compare with actual data.

Key eval metrics:
  - Directional accuracy (up/down correctly predicted)
  - MAE, RMSE, MAPE on N-day forward returns
  - Rank IC (Spearman cross-sectional correlation)
  - Top/bottom decile hit rate
  - Long-short portfolio spread

Usage:
    .venv/bin/python a_share_adapter/historical_backtest.py \
        --cutoff 2025-03-15 --predict-days 10 --max-stocks 100
"""

import sys, os, json, argparse, time
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import Kronos, KronosTokenizer, KronosPredictor

# ── Configuration ────────────────────────────────────────────────

KLINE_DIR = "/home/dev/quant/stock_tracker/data/kline_cache"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "backtest"
MODEL_DIR = Path(__file__).resolve().parent.parent / "outputs" / "mini_finetune" / "final"


class BacktestConfig:
    cutoff_date = "2025-03-15"
    lookback = 200
    predict_days = 10
    max_stocks = 100
    max_context = 512
    sample_count = 1
    T = 0.8
    top_p = 0.9
    device = "cpu"
    price_cols = ["open", "high", "low", "close", "volume", "amount"]


config = BacktestConfig()


# ── 1. Data Loading ─────────────────────────────────────────────

def load_stock(symbol):
    """Load a single stock's K-line data from JSON cache. Returns pd.DataFrame or None."""
    p = Path(KLINE_DIR) / f"{symbol}.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    ncols = len(raw[0])
    if ncols == 6:
        cols = ["date", "open", "high", "low", "close", "volume"]
    elif ncols == 7:
        cols = ["date", "open", "high", "low", "close", "volume", "amount"]
    elif ncols == 8:
        cols = ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]
    else:
        cols = ["date"] + [f"c{i}" for i in range(1, ncols)]

    df = pd.DataFrame(raw, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "amount" not in df.columns:
        df["amount"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0 * df["volume"]
    df = df[df["volume"] > 0]
    return df


def list_valid_stocks(cutoff_date, lookback, predict_days):
    """List stocks with enough data before AND after cutoff_date."""
    valid = []
    cutoff_ts = pd.Timestamp(cutoff_date)
    min_rows = lookback + predict_days + 30
    max_date = cutoff_ts + pd.DateOffset(days=predict_days + 60)

    for f in sorted(Path(KLINE_DIR).glob("*.json")):
        symbol = f.stem
        if not symbol.startswith(("sh.", "sz.")):
            continue
        code = symbol.split(".")[1]
        if code[0] in "015" or code.startswith("9"):
            continue

        try:
            df = load_stock(symbol)
            if df is None or len(df) < min_rows:
                continue
            if df.index.min() > cutoff_ts or df.index.max() < max_date:
                continue
            pre_cutoff = df[df.index <= cutoff_ts]
            post_cutoff = df[df.index > cutoff_ts]
            if len(pre_cutoff) < lookback or len(post_cutoff) < predict_days:
                continue
            valid.append(symbol)
        except Exception:
            continue
    return valid


# ── 2. Prediction ───────────────────────────────────────────────

def make_prediction(predictor, df, cutoff_date, lookback, predict_days):
    """
    Use data up to cutoff_date to predict the next predict_days prices.

    Args:
        predictor: KronosPredictor instance
        df: full pd.DataFrame with DatetimeIndex
        cutoff_date: str like '2025-03-15'
        lookback: int, number of historical bars to feed
        predict_days: int, number of days to forecast

    Returns:
        (pred_df, actual_df) or (None, None) on failure.
        pred_df: pd.DataFrame with predicted OHLCV+amount, index = future dates
        actual_df: pd.DataFrame with actual OHLCV+amount, index = future dates
    """
    cutoff_ts = pd.Timestamp(cutoff_date)
    pre_df = df[df.index <= cutoff_ts].iloc[-lookback:].copy()

    if len(pre_df) < lookback:
        return None, None

    # Ensure all required price columns exist
    for col in config.price_cols:
        if col not in pre_df.columns:
            if col == "amount" and "volume" in pre_df.columns:
                pre_df[col] = pre_df["close"] * pre_df["volume"]
            else:
                return None, None

    # Get actual future data
    actual_df = df[df.index > cutoff_ts].iloc[:predict_days].copy()
    if len(actual_df) < predict_days:
        return None, None

    # Build timestamp Series (MUST be pd.Series for calc_time_stamps which uses .dt accessor)
    x_timestamp = pd.Series(pre_df.index.values, name="date")
    y_timestamp = pd.Series(actual_df.index[:predict_days].values, name="date")

    try:
        pred_df = predictor.predict(
            df=pre_df[config.price_cols],
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp[:predict_days],
            pred_len=predict_days,
            T=config.T,
            top_p=config.top_p,
            sample_count=config.sample_count,
            verbose=False,
        )
    except Exception as e:
        # Silently skip individual prediction failures
        return None, None

    return pred_df, actual_df


# ── 3. Metrics ──────────────────────────────────────────────────

def compute_metrics(all_results):
    """Compute evaluation metrics from a list of per-stock prediction results."""
    if not all_results:
        return {}

    df = pd.DataFrame(all_results)

    # Filter out extreme outliers (> 100% return)
    df = df[(df["pred_return"].abs() < 1.0) & (df["actual_return"].abs() < 1.0)].copy()
    if len(df) == 0:
        return {}

    # Directional accuracy
    df["pred_dir"] = np.sign(df["pred_return"])
    df["actual_dir"] = np.sign(df["actual_return"])
    df["dir_correct"] = (df["pred_dir"] == df["actual_dir"]).astype(int)
    dir_mask = df["actual_dir"] != 0
    dir_accuracy = df.loc[dir_mask, "dir_correct"].mean() if dir_mask.sum() > 0 else 0.0

    # Directional accuracy on significant moves (> 2%)
    sig_mask = df["actual_return"].abs() > 0.02
    dir_acc_sig = df.loc[sig_mask & dir_mask, "dir_correct"].mean() if (sig_mask & dir_mask).sum() > 0 else 0.0

    # Regression
    mae = np.mean(np.abs(df["pred_return"] - df["actual_return"]))
    rmse = np.sqrt(np.mean((df["pred_return"] - df["actual_return"]) ** 2))
    act_abs = df["actual_return"].abs().clip(lower=0.005)
    mape = np.mean(np.abs(df["pred_return"] - df["actual_return"]) / act_abs) * 100

    # Correlation
    corr = df["pred_return"].corr(df["actual_return"])
    try:
        rank_ic = df["pred_return"].corr(df["actual_return"], method="spearman")
    except Exception:
        rank_ic = np.nan

    # Top/bottom decile analysis
    n = len(df)
    top_n = max(1, n // 10)
    df_pred_sorted = df.sort_values("pred_return", ascending=False)
    top_pred = df_pred_sorted.head(top_n)
    bot_pred = df_pred_sorted.tail(top_n)

    df_actual_sorted = df.sort_values("actual_return", ascending=False)
    top_actual_set = set(df_actual_sorted.head(top_n)["symbol"])
    bot_actual_set = set(df_actual_sorted.tail(top_n)["symbol"])

    top_hit = len(set(top_pred["symbol"]) & top_actual_set) / top_n if top_n > 0 else 0
    bot_hit = len(set(bot_pred["symbol"]) & bot_actual_set) / top_n if top_n > 0 else 0

    long_ret = top_pred["actual_return"].mean()
    short_ret = bot_pred["actual_return"].mean()
    ls_spread = long_ret - short_ret

    metrics = {
        "n_stocks": len(df["symbol"].unique()),
        "n_predictions": len(df),
        "dir_accuracy": dir_accuracy,
        "dir_accuracy_significant": dir_acc_sig,
        "mae_return": mae,
        "rmse_return": rmse,
        "mape_pct": mape,
        "pearson_corr": corr,
        "rank_ic": rank_ic,
        "top_decile_hit": top_hit,
        "bot_decile_hit": bot_hit,
        "long_basket_return": long_ret,
        "short_basket_return": short_ret,
        "long_short_spread": ls_spread,
        "avg_pred_return": df["pred_return"].mean(),
        "avg_actual_return": df["actual_return"].mean(),
    }

    # Per-horizon breakdown
    for h in [1, 3, 5]:
        df_h = df[df["horizon_days"] == h] if "horizon_days" in df.columns else pd.DataFrame()
        if len(df_h) > 10:
            metrics[f"dir_acc_{h}d"] = (
                df_h.loc[df_h["actual_dir"] != 0, "dir_correct"].mean()
                if (df_h["actual_dir"] != 0).sum() > 0 else 0.0
            )
            metrics[f"rank_ic_{h}d"] = df_h["pred_return"].corr(df_h["actual_return"])

    return metrics


def print_metrics(metrics):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*70}")
    print(f"  HISTORICAL BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f"  Stocks evaluated:     {metrics.get('n_stocks', 0)}")
    print(f"  Predictions made:     {metrics.get('n_predictions', 0)}")
    print(f"{'='*70}")
    print(f"  DIRECTIONAL ACCURACY")
    print(f"    All moves:          {metrics.get('dir_accuracy', 0):.2%}")
    print(f"    |return| > 2%:       {metrics.get('dir_accuracy_significant', 0):.2%}")
    for k in sorted(metrics):
        if k.startswith("dir_acc_") and "significant" not in k:
            print(f"    {k.replace('dir_acc_', 'Horizon ')}:   {metrics[k]:.2%}")
    print(f"{'='*70}")
    print(f"  REGRESSION METRICS")
    print(f"    MAE (return):       {metrics.get('mae_return', 0):.4f}")
    print(f"    RMSE (return):      {metrics.get('rmse_return', 0):.4f}")
    print(f"    MAPE:               {metrics.get('mape_pct', 0):.1f}%")
    print(f"    Pearson r:          {metrics.get('pearson_corr', 0):.4f}")
    print(f"    Rank IC (Spearman): {metrics.get('rank_ic', 0):.4f}")
    for k in sorted(metrics):
        if k.startswith("rank_ic_") and "dir" not in k:
            print(f"    {k.replace('rank_ic_', 'Rank IC ')}:   {metrics[k]:.4f}")
    print(f"{'='*70}")
    print(f"  DECILE ANALYSIS (Top vs Bottom 10%)")
    print(f"    Top decile hit:     {metrics.get('top_decile_hit', 0):.2%}")
    print(f"    Bottom decile hit:  {metrics.get('bot_decile_hit', 0):.2%}")
    print(f"    Long basket ret:    {metrics.get('long_basket_return', 0):.4f}")
    print(f"    Short basket ret:   {metrics.get('short_basket_return', 0):.4f}")
    print(f"    Long-Short spread:  {metrics.get('long_short_spread', 0):.4f}")
    print(f"{'='*70}")
    print(f"  BENCHMARKS")
    print(f"    Avg model pred:     {metrics.get('avg_pred_return', 0):.4f}")
    print(f"    Avg actual return:  {metrics.get('avg_actual_return', 0):.4f}")
    print(f"{'='*70}")


# ── 4. Multi-horizon evaluation ─────────────────────────────────

def evaluate_multi_horizon(predictor, all_pre_dfs, all_post_dfs, symbols, horizons=[1, 3, 5, 10]):
    """Run predictions at multiple horizons for each stock."""
    all_results = []

    for sym, pre_df, post_df in tqdm(
        zip(symbols, all_pre_dfs, all_post_dfs),
        total=len(symbols),
        desc=f"Predicting {len(horizons)} horizons",
    ):
        for h in horizons:
            if len(post_df) < h:
                continue
            try:
                full_df = pd.concat([pre_df, post_df])
                cutoff_str = pre_df.index[-1].strftime("%Y-%m-%d")
                pred_df, actual_df = make_prediction(
                    predictor, full_df, cutoff_str, min(len(pre_df), config.lookback), h
                )
                if pred_df is None or actual_df is None or len(pred_df) == 0 or len(actual_df) == 0:
                    continue

                pred_close = pred_df["close"].values
                actual_close = actual_df["close"].values[:h]
                if len(pred_close) < h or len(actual_close) < h:
                    continue

                last_close = pre_df["close"].iloc[-1]
                pred_return = (pred_close[h - 1] / last_close) - 1.0
                actual_return = (actual_close[h - 1] / last_close) - 1.0

                all_results.append({
                    "symbol": sym,
                    "horizon_days": h,
                    "pred_return": pred_return,
                    "actual_return": actual_return,
                })
            except Exception:
                continue
    return all_results


# ── 5. Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kronos Historical Backtest")
    parser.add_argument("--cutoff", type=str, default=None,
                        help="Cutoff date YYYY-MM-DD (default: run 3 dates)")
    parser.add_argument("--predict-days", type=int, default=10)
    parser.add_argument("--max-stocks", type=int, default=100)
    parser.add_argument("--lookback", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--multi-horizon", action="store_true",
                        help="Also evaluate 1,3,5,10 day horizons")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR))
    args = parser.parse_args()

    # Update config
    if args.cutoff:
        cutoff_dates = [args.cutoff]
    else:
        cutoff_dates = ["2024-12-15", "2025-03-15", "2025-06-15"]

    config.lookback = args.lookback
    config.predict_days = args.predict_days
    config.max_stocks = args.max_stocks
    config.T = args.temperature
    config.sample_count = args.sample_count

    model_dir = Path(args.model_dir)

    print(f"\n{'='*70}")
    print(f"  KRONOS HISTORICAL BACKTEST")
    print(f"{'='*70}")
    print(f"  Model dir:     {model_dir}")
    print(f"  Cutoff dates:  {cutoff_dates}")
    print(f"  Predict days:  {config.predict_days}")
    print(f"  Lookback:      {config.lookback}")
    print(f"  Max stocks:    {config.max_stocks}")
    print(f"  Temperature:   {config.T}")
    print(f"  Sample count:  {config.sample_count}")
    print(f"  Multi-horizon: {args.multi_horizon}")
    print(f"{'='*70}")

    # ── Load Models ──
    print("\n[1/4] Loading fine-tuned models...")
    tokenizer_path = model_dir / "tokenizer"
    predictor_path = model_dir / "predictor"

    if not tokenizer_path.exists() or not predictor_path.exists():
        print(f"  ERROR: Model not found at {model_dir}")
        sys.exit(1)

    t0 = time.time()
    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))
    model = Kronos.from_pretrained(str(predictor_path))
    predictor = KronosPredictor(model, tokenizer, max_context=config.max_context)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Tokenizer: {sum(p.numel() for p in tokenizer.parameters()):,} params")
    print(f"  Predictor: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Iterate cutoff dates ──
    all_date_results = {}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for cutoff_date in cutoff_dates:
        print(f"\n[2/4] Finding valid stocks for cutoff {cutoff_date}...")
        valid_stocks = list_valid_stocks(cutoff_date, config.lookback, config.predict_days)
        print(f"  Found {len(valid_stocks)} valid stocks")

        if len(valid_stocks) == 0:
            print(f"  No valid stocks for {cutoff_date}, skipping.")
            continue

        # Set seed for reproducibility, but sample different stocks per cutoff date
        rng = np.random.RandomState(42 + hash(cutoff_date) % 100)
        max_n = min(config.max_stocks, len(valid_stocks))
        eval_stocks = rng.choice(valid_stocks, size=max_n, replace=False).tolist()

        print(f"  Selected {len(eval_stocks)} stocks for evaluation")

        # ── Predict ──
        print(f"\n[3/4] Loading data and predicting ({len(eval_stocks)} stocks)...")
        all_results = []
        pred_errors = []

        for sym in tqdm(eval_stocks, desc="Predicting"):
            df = load_stock(sym)
            if df is None:
                continue

            pred_df, actual_df = make_prediction(
                predictor, df, cutoff_date, config.lookback, config.predict_days
            )

            if pred_df is None or actual_df is None or len(pred_df) == 0 or len(actual_df) == 0:
                continue

            pred_close = pred_df["close"].values
            actual_close = actual_df["close"].values

            min_len = min(len(pred_close), len(actual_close), config.predict_days)
            if min_len == 0:
                continue

            last_close = df.loc[df.index <= pd.Timestamp(cutoff_date), "close"].iloc[-1]
            pred_return = (pred_close[min_len - 1] / last_close) - 1.0
            actual_return = (actual_close[min_len - 1] / last_close) - 1.0

            all_results.append({
                "symbol": sym,
                "horizon_days": min_len,
                "last_close": last_close,
                "pred_close": pred_close[min_len - 1],
                "actual_close": actual_close[min_len - 1],
                "pred_return": pred_return,
                "actual_return": actual_return,
            })

            # Track daily errors
            for d in range(min_len):
                pr = (pred_close[d] / last_close) - 1.0 if d < len(pred_close) else np.nan
                ar = (actual_close[d] / last_close) - 1.0 if d < len(actual_close) else np.nan
                if not np.isnan(pr) and not np.isnan(ar) and abs(ar) < 1.0:
                    pred_errors.append({
                        "symbol": sym,
                        "day": d + 1,
                        "pred_return": pr,
                        "actual_return": ar,
                        "abs_error": abs(pr - ar),
                    })

        # Multi-horizon
        if args.multi_horizon:
            print(f"\n  Running multi-horizon evaluation...")
            all_pre_dfs, all_post_dfs, valid_syms = [], [], []
            for sym in eval_stocks:
                df = load_stock(sym)
                if df is None: continue
                pre_df = df[df.index <= pd.Timestamp(cutoff_date)].iloc[-config.lookback:].copy()
                post_df = df[df.index > pd.Timestamp(cutoff_date)].copy()
                if len(pre_df) >= config.lookback and len(post_df) >= 10:
                    all_pre_dfs.append(pre_df)
                    all_post_dfs.append(post_df)
                    valid_syms.append(sym)
            mh_results = evaluate_multi_horizon(predictor, all_pre_dfs, all_post_dfs, valid_syms)
            all_results.extend(mh_results)

        # ── Metrics ──
        print(f"\n[4/4] Computing metrics...")
        print(f"  Total predictions: {len(all_results)}")
        metrics = compute_metrics(all_results)
        all_date_results[cutoff_date] = metrics
        print_metrics(metrics)

        # Save detailed CSV
        results_df = pd.DataFrame(all_results)
        csv_path = OUTPUT_DIR / f"backtest_{cutoff_date}_{config.predict_days}d.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\n  Detailed results: {csv_path}")

        # Daily error breakdown
        if pred_errors:
            err_df = pd.DataFrame(pred_errors)
            err_csv = OUTPUT_DIR / f"pred_errors_{cutoff_date}.csv"
            err_df.to_csv(err_csv, index=False)
            daily = err_df.groupby("day")[["abs_error", "pred_return", "actual_return"]].mean()
            print(f"\n  Daily average errors:")
            for d in sorted(daily.index):
                row = daily.loc[d]
                print(f"    Day {int(d)}: |Error|={row['abs_error']:.4f}  "
                      f"Pred={row['pred_return']:.4f}  Actual={row['actual_return']:.4f}")

    # ── Cross-date summary ──
    if len(all_date_results) > 1:
        print(f"\n{'='*70}")
        print(f"  CROSS-DATE SUMMARY")
        print(f"{'='*70}")
        summary_rows = []
        for date, m in all_date_results.items():
            summary_rows.append({
                "date": date,
                "dir_acc": f"{m.get('dir_accuracy', 0):.2%}",
                "dir_acc_sig": f"{m.get('dir_accuracy_significant', 0):.2%}",
                "rank_ic": f"{m.get('rank_ic', 0):.4f}",
                "ls_spread": f"{m.get('long_short_spread', 0):.4f}",
                "n_stocks": m.get("n_stocks", 0),
            })
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(OUTPUT_DIR / "cross_date_summary.csv", index=False)

    print(f"\n{'='*70}")
    print(f"  Backtest complete!")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()