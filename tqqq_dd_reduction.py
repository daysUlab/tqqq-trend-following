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
PROC_DIR   = os.path.join(BASE_DIR, "tqqq_dd_reduction", "processed")
CHARTS_DIR = os.path.join(BASE_DIR, "tqqq_dd_reduction", "charts")
os.makedirs(PROC_DIR,   exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

START          = "2010-02-11"   # TQQQ inception
END            = date.today().isoformat()
INITIAL_EQUITY = 10_000.0
SLIP_RT        = 0.0004         # 2 bps per side

# -- Download ------------------------------------------------------------------
print(f"Downloading TQQQ, QQQ, SPY, TLT, BIL, ^VIX  |  {START} to {END}")
price_raw = yf.download(["TQQQ", "QQQ", "SPY", "TLT", "BIL"],
                        start=START, end=END, auto_adjust=True, progress=False)
vix_raw   = yf.download("^VIX", start=START, end=END,
                         auto_adjust=False, progress=False)

if isinstance(price_raw.columns, pd.MultiIndex):
    prices = price_raw["Close"].copy()
else:
    prices = price_raw[["TQQQ", "QQQ", "SPY", "TLT", "BIL"]].copy()

if isinstance(vix_raw.columns, pd.MultiIndex):
    vix_raw.columns = vix_raw.columns.get_level_values(0)

prices.index = pd.to_datetime(prices.index)
prices.index.name = "date"
prices = prices.ffill().dropna(subset=["TQQQ", "QQQ", "SPY", "TLT", "BIL"])

vix_raw.index = pd.to_datetime(vix_raw.index)
vix_aligned   = vix_raw["Close"].astype(float).reindex(prices.index).ffill()

tqqq  = prices["TQQQ"].astype(float)
qqq   = prices["QQQ"].astype(float)
tlt   = prices["TLT"].astype(float)
bil   = prices["BIL"].astype(float)
dates = prices.index
n     = len(prices)
print(f"  Common data: {dates[0].date()} to {dates[-1].date()}  ({n} rows)")

tqqq_ret = tqqq.pct_change().fillna(0)
tlt_ret  = tlt.pct_change().fillna(0)
bil_ret  = bil.pct_change().fillna(0)

# SMAs on QQQ
sma50  = qqq.rolling(50).mean()
sma100 = qqq.rolling(100).mean()
sma150 = qqq.rolling(150).mean()

# VIX lagged one day (signal on close t-1, execute at close t)
vix_prev = vix_aligned.ffill().shift(1).fillna(20.0)

# -- Engine helpers ------------------------------------------------------------
def make_equity(strat_returns):
    return INITIAL_EQUITY * (1 + strat_returns.fillna(0)).cumprod()

def apply_slip(returns, position):
    """Slippage scales with size of position change."""
    switch = position.diff().abs()
    switch.iloc[0] = abs(float(position.iloc[0]))
    return returns - switch * SLIP_RT

def calc_metrics(eq, position=None):
    c    = eq.dropna().values.astype(float)
    rets = np.diff(c) / c[:-1]
    rets = rets[~np.isnan(rets)]
    if len(rets) == 0:
        return {}
    total = c[-1] / c[0] - 1
    cagr  = (c[-1] / c[0]) ** (252 / len(rets)) - 1
    sh    = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    neg   = rets[rets < 0]
    so    = (rets.mean() / neg.std()) * np.sqrt(252) if len(neg) > 0 and neg.std() > 0 else 0
    rm    = np.maximum.accumulate(c)
    mdd   = ((c - rm) / rm).min()
    if position is not None:
        sw = int((position.diff().abs().fillna(abs(float(position.iloc[0]))) > 1e-9).sum())
    else:
        sw = 0
    return dict(total=total, cagr=cagr, sharpe=sh, sortino=so, max_dd=mdd, switches=sw)

def dd_series(eq):
    c  = eq.values.astype(float)
    rm = np.maximum.accumulate(c)
    return pd.Series((c - rm) / rm, index=eq.index)

def compute_asymmetric_pos(qqq_s, sma100_s, sma150_s):
    """State machine: exit to BIL when QQQ < SMA100, re-enter when QQQ > SMA150."""
    arr_q    = qqq_s.values
    arr_s100 = sma100_s.values
    arr_s150 = sma150_s.values
    pos      = np.zeros(len(qqq_s))
    for i in range(1, len(qqq_s)):
        q    = arr_q[i - 1]
        s100 = arr_s100[i - 1]
        s150 = arr_s150[i - 1]
        prev = pos[i - 1]
        if np.isnan(s100) or np.isnan(s150):
            pos[i] = 0.0
        elif prev >= 0.5:               # currently in TQQQ
            pos[i] = 0.0 if q < s100 else 1.0
        else:                           # currently in BIL
            pos[i] = 1.0 if q > s150 else 0.0
    return pd.Series(pos, index=qqq_s.index)

# -- Strategy 1: Baseline (QQQ 150-day SMA) -----------------------------------
pos1  = (qqq > sma150).astype(float).shift(1).fillna(0)
ret1  = apply_slip(pos1 * tqqq_ret + (1 - pos1) * bil_ret, pos1)
eq1   = make_equity(ret1)

# -- Strategy 2: Asymmetric exit (exit SMA100, re-enter SMA150) ---------------
pos2  = compute_asymmetric_pos(qqq, sma100, sma150)
ret2  = apply_slip(pos2 * tqqq_ret + (1 - pos2) * bil_ret, pos2)
eq2   = make_equity(ret2)

# -- Strategy 3: VIX overlay ---------------------------------------------------
# VIX > 40  -> 100% BIL
# VIX 30-40 -> 50% TQQQ / 50% BIL
# VIX < 30  -> follow QQQ 150d SMA signal
base3 = (qqq > sma150).astype(float).shift(1).fillna(0)
pos3  = pd.Series(
    np.where(vix_prev > 40,  0.0,
    np.where(vix_prev >= 30, 0.5, base3.values)),
    index=dates
)
ret3  = apply_slip(pos3 * tqqq_ret + (1 - pos3) * bil_ret, pos3)
eq3   = make_equity(ret3)

# -- Strategy 4: Death cross / Golden cross (QQQ SMA50 vs SMA150) -------------
pos4  = (sma50 > sma150).astype(float).shift(1).fillna(0)
ret4  = apply_slip(pos4 * tqqq_ret + (1 - pos4) * bil_ret, pos4)
eq4   = make_equity(ret4)

# -- Strategy 5: Position scaling ----------------------------------------------
# QQQ > 5% above SMA150  -> 100% TQQQ
# QQQ 0-5% above SMA150  -> 50% TQQQ / 50% BIL
# QQQ below SMA150       -> 100% BIL
dist  = (qqq - sma150) / sma150
pos5  = pd.Series(
    np.where(dist >= 0.05, 1.0,
    np.where(dist >= 0.0,  0.5, 0.0)),
    index=dates
).shift(1).fillna(0)
ret5  = apply_slip(pos5 * tqqq_ret + (1 - pos5) * bil_ret, pos5)
eq5   = make_equity(ret5)

# -- Strategy 6: TQQQ + TLT hedge (80% TQQQ / 20% TLT risk-on, BIL risk-off) -
base6 = (qqq > sma150).astype(float).shift(1).fillna(0)
ret6  = apply_slip(
    base6 * (0.8 * tqqq_ret + 0.2 * tlt_ret) + (1 - base6) * bil_ret,
    base6
)
eq6   = make_equity(ret6)

# -- Strategy 7: Combined (asymmetric exit + VIX overlay) ---------------------
asym7 = compute_asymmetric_pos(qqq, sma100, sma150)
pos7  = pd.Series(
    np.where(vix_prev > 40,  0.0,
    np.where(vix_prev >= 30, asym7.values * 0.5, asym7.values)),
    index=dates
)
ret7  = apply_slip(pos7 * tqqq_ret + (1 - pos7) * bil_ret, pos7)
eq7   = make_equity(ret7)

# -- Benchmarks ----------------------------------------------------------------
bh_tqqq = INITIAL_EQUITY * (tqqq / float(tqqq.iloc[0]))
bh_qqq  = INITIAL_EQUITY * (qqq  / float(qqq.iloc[0]))

# -- Compute metrics -----------------------------------------------------------
ENTRIES = [
    ("1. Baseline (150d SMA)",        eq1, pos1),
    ("2. Asymmetric Exit",             eq2, pos2),
    ("3. VIX Overlay",                 eq3, pos3),
    ("4. Death / Golden Cross",        eq4, pos4),
    ("5. Position Scaling",            eq5, pos5),
    ("6. TQQQ + TLT Hedge (80/20)",   eq6, base6),
    ("7. Combined (VIX + Asym Exit)", eq7, pos7),
    ("TQQQ Buy & Hold",               bh_tqqq, None),
    ("QQQ Buy & Hold",                bh_qqq,  None),
]

results = []
for label, eq, pos in ENTRIES:
    m = calc_metrics(eq, pos)
    m["label"] = label
    m["eq"]    = eq
    results.append(m)

ranked = sorted(results, key=lambda x: x["sharpe"], reverse=True)

# -- Print results table -------------------------------------------------------
DIV  = "=" * 86
SDIV = "-" * 86

def fmt_row(r):
    cagr = f"{r['cagr']*100:.2f}%"
    mdd  = f"{r['max_dd']*100:.1f}%"
    sw   = str(r["switches"]) if r["switches"] > 0 else "B&H"
    return (f"  {r['label']:<32}"
            f"{cagr:>10}"
            f"{r['sharpe']:>10.3f}"
            f"{r['sortino']:>10.3f}"
            f"{mdd:>10}"
            f"{sw:>11}")

print(f"\n{DIV}")
print(f"  TQQQ DRAWDOWN REDUCTION  ({dates[0].date()} to {dates[-1].date()})  -- ranked by Sharpe")
print(DIV)
print(f"  {'Strategy':<32}{'CAGR':>10}{'Sharpe':>10}{'Sortino':>10}{'Max DD':>10}{'Switches':>11}")
print("  " + SDIV[2:])
for r in ranked:
    print(fmt_row(r))
print(DIV)

# -- Save CSV ------------------------------------------------------------------
csv_path = os.path.join(PROC_DIR, "results.csv")
pd.DataFrame([{
    "strategy": r["label"],
    "cagr":     round(r["cagr"],    4),
    "sharpe":   round(r["sharpe"],  4),
    "sortino":  round(r["sortino"], 4),
    "max_dd":   round(r["max_dd"],  4),
    "switches": r["switches"],
} for r in results]).to_csv(csv_path, index=False)
print(f"\n  Results CSV -> {csv_path}")

# -- Chart colors --------------------------------------------------------------
COLORS = {
    "1. Baseline (150d SMA)":        "#00FF7F",
    "2. Asymmetric Exit":             "#FFD700",
    "3. VIX Overlay":                 "#00BFFF",
    "4. Death / Golden Cross":        "#FF69B4",
    "5. Position Scaling":            "#FF8C00",
    "6. TQQQ + TLT Hedge (80/20)":   "#DA70D6",
    "7. Combined (VIX + Asym Exit)": "#FFFFFF",
    "TQQQ Buy & Hold":               "#FF4444",
    "QQQ Buy & Hold":                "#888888",
}

def dollar_fmt(x, _):
    if x >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"${x/1_000:.0f}K"
    return f"${x:.0f}"

# -- Chart 1: Equity curves (log scale) ----------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))
for label, eq, _ in ENTRIES:
    is_bh = "Buy" in label
    ax.plot(eq.index, eq.values,
            color=COLORS[label],
            linewidth=1.0 if is_bh else 1.3,
            linestyle="--" if is_bh else "-",
            alpha=0.70 if is_bh else 0.90,
            label=label)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(dollar_fmt))
ax.set_title("TQQQ Drawdown Reduction -- Equity Curves (Log Scale)", fontsize=14, pad=12)
ax.set_ylabel("Portfolio Value ($)")
ax.legend(fontsize=7.5, ncol=3)
ax.grid(alpha=0.2, which="both")
fig.tight_layout()
p1 = os.path.join(CHARTS_DIR, "01_equity_curves.png")
fig.savefig(p1, dpi=150)
plt.close(fig)
print(f"  Chart saved -> {p1}")

# -- Chart 2: Drawdowns --------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))
for label, eq, _ in ENTRIES:
    is_bh = "Buy" in label
    dd = dd_series(eq)
    ax.plot(dd.index, dd.values * 100,
            color=COLORS[label],
            linewidth=1.0 if is_bh else 1.3,
            linestyle="--" if is_bh else "-",
            alpha=0.70 if is_bh else 0.90,
            label=label)
ax.set_title("TQQQ Drawdown Reduction -- Drawdowns", fontsize=14, pad=12)
ax.set_ylabel("Drawdown (%)")
ax.legend(fontsize=7.5, ncol=3)
ax.grid(alpha=0.2)
fig.tight_layout()
p2 = os.path.join(CHARTS_DIR, "02_drawdowns.png")
fig.savefig(p2, dpi=150)
plt.close(fig)
print(f"  Chart saved -> {p2}")
