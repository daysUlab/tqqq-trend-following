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
PROC_DIR   = os.path.join(BASE_DIR, "tqqq_final_comparison", "processed")
CHARTS_DIR = os.path.join(BASE_DIR, "tqqq_final_comparison", "charts")
os.makedirs(PROC_DIR,   exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

START          = "2010-02-11"
END            = date.today().isoformat()
INITIAL_EQUITY = 10_000.0
SLIP_RT        = 0.0004    # 2 bps per side

# -- Download ------------------------------------------------------------------
print(f"Downloading TQQQ, QQQ, TLT, BIL, ^VIX  |  {START} to {END}")
price_raw = yf.download(["TQQQ", "QQQ", "TLT", "BIL"],
                        start=START, end=END, auto_adjust=True, progress=False)
vix_raw   = yf.download("^VIX", start=START, end=END,
                         auto_adjust=False, progress=False)

if isinstance(price_raw.columns, pd.MultiIndex):
    prices = price_raw["Close"].copy()
else:
    prices = price_raw[["TQQQ", "QQQ", "TLT", "BIL"]].copy()

if isinstance(vix_raw.columns, pd.MultiIndex):
    vix_raw.columns = vix_raw.columns.get_level_values(0)

prices.index = pd.to_datetime(prices.index)
prices.index.name = "date"
prices = prices.ffill().dropna(subset=["TQQQ", "QQQ", "TLT", "BIL"])

tqqq  = prices["TQQQ"].astype(float)
qqq   = prices["QQQ"].astype(float)
tlt   = prices["TLT"].astype(float)
bil   = prices["BIL"].astype(float)
dates = prices.index
print(f"  Common data: {dates[0].date()} to {dates[-1].date()}  ({len(prices)} rows)")

tqqq_ret = tqqq.pct_change().fillna(0)
tlt_ret  = tlt.pct_change().fillna(0)
bil_ret  = bil.pct_change().fillna(0)
qqq_ret  = qqq.pct_change().fillna(0)

sma150   = qqq.rolling(150).mean()
position = (qqq > sma150).astype(float).shift(1).fillna(0)  # signal t-1, execute t

# -- Strategy 1: TQQQ baseline (100% TQQQ or 100% BIL) -----------------------
ret1   = position * tqqq_ret + (1 - position) * bil_ret
switch1 = position.diff().abs()
switch1.iloc[0] = float(position.iloc[0])
ret1  -= switch1 * SLIP_RT
eq1    = INITIAL_EQUITY * (1 + ret1.fillna(0)).cumprod()

# -- Strategy 2: TQQQ + TLT hedge (80/20 or 100% BIL) ------------------------
ret2   = position * (0.8 * tqqq_ret + 0.2 * tlt_ret) + (1 - position) * bil_ret
ret2  -= switch1 * SLIP_RT       # same switches as strategy 1
eq2    = INITIAL_EQUITY * (1 + ret2.fillna(0)).cumprod()

# -- Benchmark: QQQ buy and hold ----------------------------------------------
bh_qqq = INITIAL_EQUITY * (qqq / float(qqq.iloc[0]))

# -- Metrics -------------------------------------------------------------------
def calc_metrics(eq, daily_ret, pos=None):
    c    = eq.dropna().values.astype(float)
    rets = daily_ret.dropna().values.astype(float)
    rets = rets[~np.isnan(rets)]
    total = c[-1] / c[0] - 1
    cagr  = (c[-1] / c[0]) ** (252 / len(rets)) - 1
    sh    = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    neg   = rets[rets < 0]
    so    = (rets.mean() / neg.std()) * np.sqrt(252) if len(neg) > 0 and neg.std() > 0 else 0
    rm    = np.maximum.accumulate(c)
    mdd   = ((c - rm) / rm).min()
    sw    = int((position.diff().abs().fillna(float(position.iloc[0])) > 1e-9).sum()) if pos is not None else 0
    pct   = float(pos.mean()) * 100 if pos is not None else 100.0
    return dict(total=total, cagr=cagr, sharpe=sh, sortino=so, max_dd=mdd, switches=sw, pct_on=pct)

m1 = calc_metrics(eq1, ret1, position)
m2 = calc_metrics(eq2, ret2, position)
m3 = calc_metrics(bh_qqq, qqq_ret)

# -- Print comparison table ----------------------------------------------------
DIV  = "=" * 68
SDIV = "-" * 68
W    = 16

def fmt_col(val, fmt):
    return f"{val:{fmt}}"

rows = [
    ("TQQQ Baseline",     m1),
    ("TQQQ + TLT Hedge",  m2),
    ("QQQ Buy & Hold",    m3),
]

print(f"\n{DIV}")
print(f"  TQQQ FINAL COMPARISON  ({dates[0].date()} to {dates[-1].date()})")
print(DIV)
print(f"  {'Metric':<22}{'TQQQ Baseline':>{W}}{'TQQQ+TLT Hedge':>{W}}{'QQQ B&H':>{W}}")
print("  " + SDIV[2:])

metric_lines = [
    ("Total Return",  lambda m: f"{m['total']*100:.1f}%"),
    ("CAGR",          lambda m: f"{m['cagr']*100:.2f}%"),
    ("Sharpe",        lambda m: f"{m['sharpe']:.3f}"),
    ("Sortino",       lambda m: f"{m['sortino']:.3f}"),
    ("Max Drawdown",  lambda m: f"{m['max_dd']*100:.1f}%"),
    ("Switches",      lambda m: str(m['switches']) if m['switches'] > 0 else "B&H"),
    ("Time Risk-On",  lambda m: f"{m['pct_on']:.1f}%"),
]

for label, fn in metric_lines:
    vals = [fn(m) for _, m in rows]
    print(f"  {label:<22}" + "".join(f"{v:>{W}}" for v in vals))

print(DIV)

# -- Yearly returns ------------------------------------------------------------
def yearly_returns(ret_series):
    """Compound daily returns by calendar year."""
    return (1 + ret_series).groupby(ret_series.index.year).prod() - 1

yr1 = yearly_returns(ret1)
yr2 = yearly_returns(ret2)
yr3 = yearly_returns(qqq_ret)

all_years = sorted(set(yr1.index) | set(yr2.index) | set(yr3.index))
yr_rows = []
for y in all_years:
    yr_rows.append({
        "year":             y,
        "TQQQ_baseline":    round(float(yr1.get(y, np.nan)) * 100, 2),
        "TQQQ_TLT_hedge":   round(float(yr2.get(y, np.nan)) * 100, 2),
        "QQQ_BH":           round(float(yr3.get(y, np.nan)) * 100, 2),
    })

yr_df = pd.DataFrame(yr_rows)
yr_csv = os.path.join(PROC_DIR, "yearly_returns.csv")
yr_df.to_csv(yr_csv, index=False)
print(f"\n  Yearly returns CSV -> {yr_csv}")

# Print yearly returns
print(f"\n  {'Year':<8}{'TQQQ Baseline':>16}{'TQQQ+TLT Hedge':>16}{'QQQ B&H':>12}")
print("  " + "-" * 52)
for row in yr_rows:
    def pct(v):
        return f"{v:+.1f}%" if not np.isnan(v) else "  n/a"
    print(f"  {row['year']:<8}{pct(row['TQQQ_baseline']):>16}{pct(row['TQQQ_TLT_hedge']):>16}{pct(row['QQQ_BH']):>12}")

# -- Drawdown series -----------------------------------------------------------
def dd_series(eq):
    c  = eq.values.astype(float)
    rm = np.maximum.accumulate(c)
    return pd.Series((c - rm) / rm, index=eq.index)

def dollar_fmt(x, _):
    if x >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"${x/1_000:.0f}K"
    return f"${x:.0f}"

# -- Chart 1: Equity curves (log scale, 14x8) ----------------------------------
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(eq1.index,    eq1.values,    color="#00FF7F", linewidth=1.4,
        label="TQQQ + QQQ 150d SMA Filter")
ax.plot(eq2.index,    eq2.values,    color="#00BFFF", linewidth=1.4,
        label="TQQQ + TLT Hedge (80/20)")
ax.plot(bh_qqq.index, bh_qqq.values, color="#FFFFFF", linewidth=1.1,
        label="QQQ Buy & Hold", alpha=0.80)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(dollar_fmt))
ax.set_title("TQQQ Strategy Comparison -- Equity Curve (Log Scale)", fontsize=14, pad=12)
ax.set_ylabel("Portfolio Value ($10,000 start)")
ax.legend(fontsize=10)
ax.grid(alpha=0.2, which="both")
fig.tight_layout()
p1 = os.path.join(CHARTS_DIR, "01_comparison.png")
fig.savefig(p1, dpi=150)
plt.close(fig)
print(f"\n  Chart saved -> {p1}")

# -- Chart 2: Drawdown (14x6) --------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(dd_series(eq1).index,    dd_series(eq1).values * 100,    color="#00FF7F",
        linewidth=1.4, label="TQQQ + QQQ 150d SMA Filter")
ax.plot(dd_series(eq2).index,    dd_series(eq2).values * 100,    color="#00BFFF",
        linewidth=1.4, label="TQQQ + TLT Hedge (80/20)")
ax.plot(dd_series(bh_qqq).index, dd_series(bh_qqq).values * 100, color="#FFFFFF",
        linewidth=1.1, label="QQQ Buy & Hold", alpha=0.80)
ax.set_title("TQQQ Strategy Comparison -- Drawdown", fontsize=14, pad=12)
ax.set_ylabel("Drawdown (%)")
ax.legend(fontsize=10)
ax.grid(alpha=0.2)
fig.tight_layout()
p2 = os.path.join(CHARTS_DIR, "02_drawdown_comparison.png")
fig.savefig(p2, dpi=150)
plt.close(fig)
print(f"  Chart saved -> {p2}")
