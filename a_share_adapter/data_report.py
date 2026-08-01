"""A-Share data completeness report for Kronos training."""
import json
from pathlib import Path
import pandas as pd

KLINE_DIR = Path("/home/dev/quant/stock_tracker/data/kline_cache")
files = sorted(KLINE_DIR.glob("*.json"))

stats = []
for f in files:
    data = json.loads(f.read_text())
    if len(data) == 0:
        stats.append({"symbol": f.stem, "rows": 0, "start": None, "end": None, "type": "empty", "ncols": 0})
        continue

    name = f.stem
    nrows = len(data)
    start = data[0][0]
    end = data[-1][0]
    ncols = len(data[0]) if data else 0

    if name.startswith("bk."):
        stype = "sector"
    elif name.startswith(("sh.", "sz.")):
        code = name.split(".")[1]
        if name.startswith("sh.") and code.startswith("6"):
            stype = "sh_stock"
        elif name.startswith("sz.") and (code.startswith("0") or code.startswith("2") or code.startswith("3")):
            stype = "sz_stock"
        else:
            stype = "index/etf"
    else:
        stype = "other"

    stats.append({"symbol": name, "rows": nrows, "start": start, "end": end, "type": stype, "ncols": ncols})

df = pd.DataFrame(stats)
a_shares = df[df["type"].isin(["sh_stock", "sz_stock"])]

print("=" * 60)
print("A-Share Data Completeness Report for Kronos Training")
print("=" * 60)
print(f"Total files: {len(files)}")
print()

print("--- By type ---")
print(df["type"].value_counts().to_string())
print()

print(f"--- A-shares ---")
print(f"  sh_stock (Shanghai 6xx):  {len(a_shares[a_shares.type=='sh_stock'])}")
print(f"  sz_stock (Shenzhen 0xx/2xx/3xx): {len(a_shares[a_shares.type=='sz_stock'])}")
print(f"  Total: {len(a_shares)}")
print()

print("--- Row count distribution (A-shares) ---")
bins = [0, 100, 200, 500, 1000, 2000, 4000, 10000]
labels = ["<100", "100-199", "200-499", "500-999", "1000-1999", "2000-3999", "4000+"]
a_shares["bucket"] = pd.cut(a_shares["rows"], bins=bins, labels=labels, right=True)
print(a_shares["bucket"].value_counts().sort_index().to_string())
print()

min_required = 200
sufficient = a_shares[a_shares["rows"] >= min_required]
insufficient = a_shares[a_shares["rows"] < min_required]
print(f"--- Kronos Training Suitability (need >= {min_required} rows) ---")
print(f"  Sufficient: {len(sufficient)} ({len(sufficient)/len(a_shares)*100:.1f}%)")
print(f"  Insufficient: {len(insufficient)} ({len(insufficient)/len(a_shares)*100:.1f}%)")
if len(insufficient) > 0:
    print(f"  Examples: {insufficient.sort_values('rows')['symbol'].head(10).tolist()}")
print()

print("--- Date Coverage ---")
print(f"  Earliest: {a_shares['start'].min()}")
print(f"  Latest:   {a_shares['end'].max()}")
print()

latest_dates = a_shares["end"].value_counts().sort_index()
print("--- Top 5 latest dates ---")
for d, c in latest_dates.tail(5).items():
    print(f"  {d}: {c} stocks")
print()

print("--- Columns ---")
col6 = a_shares[a_shares["ncols"] == 6]
col7 = a_shares[a_shares["ncols"] == 7]
print(f"  6 cols (OHLCV only): {len(col6)}")
print(f"  7 cols (OHLCV+amount): {len(col7)}")

# Sample stocks for manual verification
print("\n--- Sample stocks ---")
for s in ["sh.600000", "sz.000001", "sz.300750", "sh.688981"]:
    row = a_shares[a_shares["symbol"] == s]
    if len(row) > 0:
        r = row.iloc[0]
        print(f"  {s}: {r['rows']} rows, {r['start']} ~ {r['end']}, cols={r['ncols']}")