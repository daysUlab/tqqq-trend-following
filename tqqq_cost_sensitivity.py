import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date

plt.style.use("dark_background")

# -- Paths ---------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROC_DIR   = os.path.join(BASE_DIR, "tqqq_cost_sensitivity", "processed")
CHARTS_DIR = os.path.join(BASE_DIR, "tqqq_cost_sensitivity", "charts")
os.makedirs(PROC_DIR,   exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

START          = "2010-02-11"
END            = date.today().isoformat()
INITIAL_EQUITY = 10_000.0
SMA_WIN        = 175

# Slippage levels in bps per side; total applied per switch = 2 * bps / 10000
# 2 bps per side matches all other strategy files (SLIP_RT = 0.0004)
BPS_LEVELS = [0, 1, 2, 5, 10, 15, 20, 50]

# -- Download ------------------------------------------------------------------
print(f"Downloading TQQQ, QQQ, BIL  |  {START} to {END}")
raw = yf.download(["TQQQ", "QQQ", "BIL"], start=START, end=END,
                  auto_adjust=True, progress=False)

if isinstance(raw.columns, pd.MultiIndex):
    prices = raw["Close"].copy()
else:
    prices = raw[["TQQQ", "QQQ", "BIL"]].copy()

prices.index = pd.to_datetime(prices.index)
prices.index.name = "date"
prices = prices.ffill().dropna(subset=["TQQQ", "QQQ", "BIL"])

tqqq  = prices["TQQQ"].astype(float)
qqq   = prices["QQQ"].astype(float)
bil   = prices["BIL"].astype(float)
dates = prices.index
print(f"  Data: {dates[0].date()} to {dates[-1].date()}  ({len(prices)} rows)")

tqqq_ret = tqqq.pct_change().fillna(0)
bil_ret  = bil.pct_change().fillna(0)
qqq_ret  = qqq.pct_change().fillna(0)

# -- Build position (same for all slippage levels) ----------------------------
sma      = qqq.rolling(SMA_WIN).mean()
position = (qqq > sma).astype(float).shift(1).fillna(0)

switch        = position.diff().abs()
switch.iloc[0] = float(position.iloc[0])
n_switches    = int((switch > 1e-9).sum())

base_strat    = position * tqqq_ret + (1 - position) * bil_ret  # pre-slippage returns

# -- QQQ buy-and-hold CAGR (benchmark threshold) ------------------------------
qqq_arr      = qqq.values.astype(float)
n_days       = len(qqq_arr)
qqq_bh_cagr  = (qqq_arr[-1] / qqq_arr[0]) ** (252 / n_days) - 1

# -- Run at each slippage level -----------------------------------------------
def calc_metrics(eq_vals, ret_vals):
    c    = eq_vals
    rets = ret_vals[~np.isnan(ret_vals)]
    total = c[-1] / c[0] - 1
    cagr  = (c[-1] / c[0]) ** (252 / len(rets)) - 1
    sh    = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    rm    = np.maximum.accumulate(c)
    mdd   = ((c - rm) / rm).min()
    return total, cagr, sh, mdd

results = []
zero_bps_cagr = None   # baseline for cost drag calculation

for bps in BPS_LEVELS:
    slip_total = 2 * bps / 10_000   # per-switch decimal cost (round trip)
    strat      = base_strat - switch * slip_total
    eq         = INITIAL_EQUITY * (1 + strat.fillna(0)).cumprod()
    total, cagr, sh, mdd = calc_metrics(eq.values, strat.values)

    if bps == 0:
        zero_bps_cagr = cagr

    results.append(dict(
        bps=bps, slip_total=slip_total,
        total=total, cagr=cagr, sharpe=sh, max_dd=mdd,
        eq=eq,
    ))

# -- Cost drag vs 0 bps baseline ----------------------------------------------
for r in results:
    r["cagr_drag"] = r["cagr"] - zero_bps_cagr   # negative number, in pp

# -- Break-even: interpolate where strategy CAGR crosses QQQ B&H CAGR --------
bps_arr  = np.array([r["bps"]  for r in results], dtype=float)
cagr_arr = np.array([r["cagr"] for r in results], dtype=float)
diff_arr = cagr_arr - qqq_bh_cagr

# Find where diff crosses zero (strategy CAGR drops below QQQ CAGR)
cross_idx = np.where(np.diff(np.sign(diff_arr)) < 0)[0]

if len(cross_idx) == 0:
    if diff_arr[-1] > 0:
        break_even_bps = None   # never crosses — always beats
    else:
        break_even_bps = 0.0    # already below at 0 bps
else:
    i  = cross_idx[0]
    x0, x1 = bps_arr[i], bps_arr[i + 1]
    y0, y1 = diff_arr[i], diff_arr[i + 1]
    break_even_bps = float(x0 - y0 * (x1 - x0) / (y1 - y0))

# -- Print results table -------------------------------------------------------
DIV  = "=" * 76
SDIV = "-" * 76

print(f"\n{DIV}")
print(f"  TQQQ COST SENSITIVITY  (QQQ {SMA_WIN}-day SMA filter  |  {n_switches} switches total)")
print(f"  {dates[0].date()} to {dates[-1].date()}")
print(f"  QQQ B&H CAGR benchmark: {qqq_bh_cagr*100:.2f}%")
print(DIV)
print(f"  {'Slippage':>10}{'Total Ret':>12}{'CAGR':>10}{'Sharpe':>10}"
      f"{'Max DD':>10}{'CAGR Drag':>12}")
print("  " + SDIV[2:])

for r in results:
    flag = " << baseline" if r["bps"] == 2 else ""
    beat = " [beats QQQ]" if r["cagr"] > qqq_bh_cagr else " [below QQQ]"
    print(f"  {r['bps']:>7} bps"
          f"{r['total']*100:>11.1f}%"
          f"{r['cagr']*100:>9.2f}%"
          f"{r['sharpe']:>10.3f}"
          f"{r['max_dd']*100:>9.1f}%"
          f"{r['cagr_drag']*100:>10.2f}pp"
          f"{flag}{beat}")

print("  " + SDIV[2:])
if break_even_bps is None:
    print(f"  Break-even: strategy beats QQQ B&H at all tested slippage levels")
elif break_even_bps == 0.0:
    print(f"  Break-even: strategy does not beat QQQ B&H even at 0 bps")
else:
    print(f"  Break-even slippage: {break_even_bps:.1f} bps per side"
          f"  (strategy CAGR = QQQ B&H CAGR {qqq_bh_cagr*100:.2f}%)")
print(DIV)

# -- Save CSV ------------------------------------------------------------------
csv_path = os.path.join(PROC_DIR, "cost_sensitivity.csv")
pd.DataFrame([{
    "bps_per_side":      r["bps"],
    "slip_per_switch":   round(r["slip_total"], 6),
    "total_return":      round(r["total"],   4),
    "cagr":              round(r["cagr"],    4),
    "sharpe":            round(r["sharpe"],  4),
    "max_dd":            round(r["max_dd"],  4),
    "cagr_drag_pp":      round(r["cagr_drag"] * 100, 4),
    "n_switches":        n_switches,
    "beats_qqq_bh":      r["cagr"] > qqq_bh_cagr,
} for r in results]).to_csv(csv_path, index=False)
print(f"\n  Results CSV -> {csv_path}")

# -- Chart: CAGR + Sharpe vs slippage -----------------------------------------
bps_vals   = [r["bps"]    for r in results]
cagr_vals  = [r["cagr"] * 100  for r in results]
sharpe_vals = [r["sharpe"] for r in results]

fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()

line1, = ax1.plot(bps_vals, cagr_vals,   color="#00FF7F", linewidth=2.0,
                  marker="o", markersize=6, label="CAGR (%)")
line2, = ax2.plot(bps_vals, sharpe_vals, color="#5599FF", linewidth=2.0,
                  marker="s", markersize=6, label="Sharpe Ratio")

# Reference lines
ax1.axhline(qqq_bh_cagr * 100, color="#FF8C00", linewidth=1.0, linestyle=":",
            alpha=0.85, label=f"QQQ B&H CAGR ({qqq_bh_cagr*100:.2f}%)")

ax1.axvline(2, color="#FFD700", linewidth=1.3, linestyle="--",
            alpha=0.85, label="2 bps baseline")

if break_even_bps is not None and 0 < break_even_bps <= max(bps_vals):
    ax1.axvline(break_even_bps, color="#FF4444", linewidth=1.3, linestyle="--",
                alpha=0.85, label=f"Break-even ({break_even_bps:.1f} bps)")

# Labels and formatting
ax1.set_xlabel("Slippage (bps per side)", fontsize=11)
ax1.set_ylabel("CAGR (%)", color="#00FF7F", fontsize=11)
ax1.tick_params(axis="y", labelcolor="#00FF7F")
ax2.set_ylabel("Sharpe Ratio", color="#5599FF", fontsize=11)
ax2.tick_params(axis="y", labelcolor="#5599FF")

ax1.set_title("Strategy Performance vs Transaction Cost Assumption", fontsize=14, pad=12)
ax1.set_xticks(bps_vals)
ax1.grid(alpha=0.2)

lines  = [line1, line2]
labels = [l.get_label() for l in lines]
extra_handles = [
    plt.Line2D([0], [0], color="#FF8C00", linewidth=1.0, linestyle=":",
               label=f"QQQ B&H CAGR ({qqq_bh_cagr*100:.2f}%)"),
    plt.Line2D([0], [0], color="#FFD700", linewidth=1.3, linestyle="--",
               label="2 bps baseline"),
]
if break_even_bps is not None and 0 < break_even_bps <= max(bps_vals):
    extra_handles.append(
        plt.Line2D([0], [0], color="#FF4444", linewidth=1.3, linestyle="--",
                   label=f"Break-even ({break_even_bps:.1f} bps)")
    )
ax1.legend(handles=lines + extra_handles, fontsize=9, loc="upper right")

fig.tight_layout()
p1 = os.path.join(CHARTS_DIR, "01_cost_sensitivity.png")
fig.savefig(p1, dpi=150)
plt.close(fig)
print(f"  Chart saved -> {p1}")
