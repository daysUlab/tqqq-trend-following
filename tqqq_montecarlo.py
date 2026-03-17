import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf
from datetime import date

plt.style.use("dark_background")

# -- Paths ---------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROC_DIR   = os.path.join(BASE_DIR, "tqqq_montecarlo", "processed")
CHARTS_DIR = os.path.join(BASE_DIR, "tqqq_montecarlo", "charts")
os.makedirs(PROC_DIR,   exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

START          = "2010-02-11"
END            = date.today().isoformat()
INITIAL_EQUITY = 10_000.0
SLIP_RT        = 0.0004
SMA_WIN        = 175
N_SIMS         = 10_000
N_SAMPLE_PATHS = 200

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

# -- Historical strategy -------------------------------------------------------
sma      = qqq.rolling(SMA_WIN).mean()
position = (qqq > sma).astype(float).shift(1).fillna(0)
strat    = position * tqqq_ret + (1 - position) * bil_ret

switch        = position.diff().abs()
switch.iloc[0] = float(position.iloc[0])
strat -= switch * SLIP_RT

eq_actual = INITIAL_EQUITY * (1 + strat.fillna(0)).cumprod()

# -- Historical metrics --------------------------------------------------------
def metrics_from_array(ec_arr, ret_arr):
    """ec_arr and ret_arr are numpy arrays."""
    total = ec_arr[-1] / ec_arr[0] - 1
    cagr  = (ec_arr[-1] / ec_arr[0]) ** (252 / len(ret_arr)) - 1
    sh    = (ret_arr.mean() / ret_arr.std()) * np.sqrt(252) if ret_arr.std() > 0 else 0
    neg   = ret_arr[ret_arr < 0]
    so    = (ret_arr.mean() / neg.std()) * np.sqrt(252) if len(neg) > 0 and neg.std() > 0 else 0
    ec_w  = np.concatenate([[ec_arr[0]], ec_arr])
    rm    = np.maximum.accumulate(ec_w)
    mdd   = ((ec_w - rm) / rm).min()
    return dict(total=total, cagr=cagr, sharpe=sh, sortino=so, max_dd=mdd)

c_actual  = eq_actual.values.astype(float)
r_actual  = strat.values.astype(float)
hist_m    = metrics_from_array(c_actual, r_actual)

# QQQ buy-and-hold CAGR (used as benchmark threshold)
qqq_arr      = qqq.values.astype(float)
qqq_rets_arr = qqq_ret.values.astype(float)
qqq_bh_cagr  = (qqq_arr[-1] / qqq_arr[0]) ** (252 / len(qqq_rets_arr)) - 1

# -- Print historical results --------------------------------------------------
DIV = "=" * 54
print(f"\n{DIV}")
print(f"  HISTORICAL STRATEGY  (QQQ {SMA_WIN}-day SMA Filter)")
print(f"  {dates[0].date()} to {dates[-1].date()}")
print(DIV)
print(f"  Total Return   : {hist_m['total']*100:>9.1f}%")
print(f"  CAGR           : {hist_m['cagr']*100:>9.2f}%")
print(f"  Sharpe         : {hist_m['sharpe']:>9.3f}")
print(f"  Sortino        : {hist_m['sortino']:>9.3f}")
print(f"  Max Drawdown   : {hist_m['max_dd']*100:>9.1f}%")
print(f"  QQQ B&H CAGR   : {qqq_bh_cagr*100:>9.2f}%  (benchmark threshold)")
print(DIV)

# -- Monte Carlo ---------------------------------------------------------------
print(f"\nRunning {N_SIMS:,} bootstrap simulations...")

n_days = len(r_actual)

cagr_sim   = np.zeros(N_SIMS)
mdd_sim    = np.zeros(N_SIMS)
sharpe_sim = np.zeros(N_SIMS)
final_sim  = np.zeros(N_SIMS)

# Pre-select which simulation indices to store as sample paths
rng             = np.random.default_rng(seed=None)
sample_path_set = set(rng.choice(N_SIMS, N_SAMPLE_PATHS, replace=False).tolist())
sample_paths    = []  # list of numpy arrays (equity curves)

for i in range(N_SIMS):
    # Bootstrap: resample with replacement to preserve return distribution
    r = rng.choice(r_actual, size=n_days, replace=True)
    ec = INITIAL_EQUITY * np.cumprod(1 + r)

    final_sim[i] = ec[-1]
    cagr_sim[i]  = (ec[-1] / INITIAL_EQUITY) ** (252 / n_days) - 1

    ec_w = np.concatenate([[INITIAL_EQUITY], ec])
    rm   = np.maximum.accumulate(ec_w)
    mdd_sim[i] = ((ec_w - rm) / rm).min()

    sharpe_sim[i] = (r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0

    if i in sample_path_set:
        sample_paths.append(ec)

    if (i + 1) % 1000 == 0:
        print(f"  {i+1:>6,} / {N_SIMS:,} complete")

print("  Done.")

# -- Distribution statistics ---------------------------------------------------
def pct_table(arr, label):
    p = np.percentile(arr, [5, 25, 50, 75, 95])
    print(f"\n  {label}")
    print(f"  {'─'*44}")
    print(f"  5th  pct : {p[0]:>10.3f}")
    print(f"  25th pct : {p[1]:>10.3f}")
    print(f"  Median   : {p[2]:>10.3f}")
    print(f"  75th pct : {p[3]:>10.3f}")
    print(f"  95th pct : {p[4]:>10.3f}")

print(f"\n{DIV}")
print(f"  MONTE CARLO DISTRIBUTION  ({N_SIMS:,} simulations)")
print(DIV)

pct_table(cagr_sim  * 100, "CAGR (%)")
pct_table(mdd_sim   * 100, "Max Drawdown (%)")
pct_table(sharpe_sim,      "Sharpe Ratio")

p_pos    = float((final_sim > INITIAL_EQUITY).mean()) * 100
p_qqq    = float((cagr_sim > qqq_bh_cagr).mean()) * 100
p_bigdd  = float((mdd_sim < -0.70).mean()) * 100

print(f"\n  Probabilities")
print(f"  {'─'*44}")
print(f"  P(positive total return)          : {p_pos:>6.1f}%")
print(f"  P(beat QQQ B&H CAGR {qqq_bh_cagr*100:.2f}%)    : {p_qqq:>6.1f}%")
print(f"  P(max drawdown exceeds -70%)      : {p_bigdd:>6.1f}%")
print(DIV)

# -- Helper: histogram layout --------------------------------------------------
def hist_vline(ax, val, color, label):
    ax.axvline(val, color=color, linewidth=1.6, linestyle="--",
               label=f"{label}: {val:.2f}", zorder=5)

# -- Chart 1: CAGR distribution ------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(cagr_sim * 100, bins=80, color="#00AA44", edgecolor="none", alpha=0.85)
hist_vline(ax, hist_m["cagr"] * 100,  "#FF4444", "Actual CAGR")
hist_vline(ax, qqq_bh_cagr   * 100,  "#FFD700", f"QQQ B&H CAGR")
ax.set_title(f"Monte Carlo — CAGR Distribution ({N_SIMS:,} simulations)", fontsize=13, pad=10)
ax.set_xlabel("CAGR (%)")
ax.set_ylabel("Frequency")
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
fig.tight_layout()
p1 = os.path.join(CHARTS_DIR, "01_cagr_distribution.png")
fig.savefig(p1, dpi=150)
plt.close(fig)
print(f"\n  Chart saved -> {p1}")

# -- Chart 2: Max drawdown distribution ----------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(mdd_sim * 100, bins=80, color="#BB2222", edgecolor="none", alpha=0.85)
hist_vline(ax, hist_m["max_dd"] * 100, "#00FF7F", "Actual Max DD")
ax.axvline(-70, color="#FFD700", linewidth=1.4, linestyle="--",
           label="-70% threshold", zorder=5)
ax.set_title(f"Monte Carlo — Max Drawdown Distribution ({N_SIMS:,} simulations)", fontsize=13, pad=10)
ax.set_xlabel("Max Drawdown (%)")
ax.set_ylabel("Frequency")
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
fig.tight_layout()
p2 = os.path.join(CHARTS_DIR, "02_maxdd_distribution.png")
fig.savefig(p2, dpi=150)
plt.close(fig)
print(f"  Chart saved -> {p2}")

# -- Chart 3: Sharpe distribution ----------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(sharpe_sim, bins=80, color="#2255BB", edgecolor="none", alpha=0.85)
hist_vline(ax, hist_m["sharpe"], "#FFD700", "Actual Sharpe")
ax.set_title(f"Monte Carlo — Sharpe Distribution ({N_SIMS:,} simulations)", fontsize=13, pad=10)
ax.set_xlabel("Sharpe Ratio")
ax.set_ylabel("Frequency")
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
fig.tight_layout()
p3 = os.path.join(CHARTS_DIR, "03_sharpe_distribution.png")
fig.savefig(p3, dpi=150)
plt.close(fig)
print(f"  Chart saved -> {p3}")

# -- Chart 4: Sample equity paths (log scale) ----------------------------------
def dollar_fmt(x, _):
    if x >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"${x/1_000:.0f}K"
    return f"${x:.0f}"

# All sample paths share the same date index as the actual strategy
fig, ax = plt.subplots(figsize=(14, 7))

for ec in sample_paths:
    ax.plot(dates, ec, color="#888888", linewidth=0.4, alpha=0.35)

ax.plot(dates, c_actual, color="#00FF7F", linewidth=2.0,
        label="Actual Historical Path", zorder=10)

ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(dollar_fmt))
ax.set_title(f"Monte Carlo — {N_SAMPLE_PATHS} Bootstrap Paths vs Actual  (QQQ {SMA_WIN}d SMA Filter)",
             fontsize=13, pad=10)
ax.set_ylabel("Portfolio Value ($10,000 start)")
ax.legend(fontsize=10)
ax.grid(alpha=0.2, which="both")
fig.tight_layout()
p4 = os.path.join(CHARTS_DIR, "04_sample_paths.png")
fig.savefig(p4, dpi=150)
plt.close(fig)
print(f"  Chart saved -> {p4}")
