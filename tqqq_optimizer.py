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
PROC_DIR   = os.path.join(BASE_DIR, "tqqq_optimizer", "processed")
CHARTS_DIR = os.path.join(BASE_DIR, "tqqq_optimizer", "charts")
os.makedirs(PROC_DIR,   exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# -- Data Split -- HARDCODED, NEVER CHANGE ------------------------------------
IS_START  = "2010-02-11"   # TQQQ inception
IS_END    = "2018-12-31"
OOS_START = "2019-01-01"
OOS_END   = date.today().isoformat()

INITIAL_EQUITY = 10_000.0
SLIP_RT        = 0.0004      # 2 bps per side, round trip
SMA_WINDOWS    = [50, 100, 150, 200, 250]

# -- Download ------------------------------------------------------------------
print(f"Downloading TQQQ, QQQ, SPY, BIL  |  {IS_START} to {OOS_END}")
raw = yf.download(["TQQQ", "QQQ", "SPY", "BIL"], start=IS_START, end=OOS_END,
                  auto_adjust=True, progress=False)

if isinstance(raw.columns, pd.MultiIndex):
    prices = raw["Close"].copy()
else:
    prices = raw[["TQQQ", "QQQ", "SPY", "BIL"]].copy()

prices.index = pd.to_datetime(prices.index)
prices.index.name = "date"
prices = prices.ffill()
prices = prices.dropna(subset=["TQQQ", "QQQ", "SPY", "BIL"])

tqqq  = prices["TQQQ"].astype(float)
qqq   = prices["QQQ"].astype(float)
spy   = prices["SPY"].astype(float)
bil   = prices["BIL"].astype(float)
print(f"  Full data: {prices.index[0].date()} to {prices.index[-1].date()}  ({len(prices)} rows)")

# -- Backtest Engine -----------------------------------------------------------
def run_backtest(tqqq_s, qqq_s, bil_s, sma_window):
    """Run QQQ SMA trend-following backtest holding TQQQ/BIL."""
    sma      = qqq_s.rolling(sma_window).mean()
    signal   = (qqq_s > sma).astype(float)
    position = signal.shift(1).fillna(0)

    tqqq_ret = tqqq_s.pct_change().fillna(0)
    bil_ret  = bil_s.pct_change().fillna(0)
    strat    = position * tqqq_ret + (1 - position) * bil_ret

    switch        = position.diff().abs()
    switch.iloc[0] = position.iloc[0]
    strat[switch.astype(bool)] -= SLIP_RT

    equity = INITIAL_EQUITY * (1 + strat.fillna(0)).cumprod()
    return equity, strat, position

def calc_metrics(equity, strat, position=None):
    c    = equity.dropna().values.astype(float)
    rets = strat.dropna().values.astype(float)
    rets = rets[~np.isnan(rets)]
    if len(rets) == 0 or c[0] == 0:
        return dict(total_ret=0, cagr=0, sharpe=0, sortino=0, max_dd=0, n_switches=0)

    total_ret = c[-1] / c[0] - 1
    cagr      = (c[-1] / c[0]) ** (252 / len(rets)) - 1
    sharpe    = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    neg       = rets[rets < 0]
    sortino   = (rets.mean() / neg.std()) * np.sqrt(252) if len(neg) > 0 and neg.std() > 0 else 0
    roll_max  = np.maximum.accumulate(c)
    max_dd    = ((c - roll_max) / roll_max).min()
    if position is not None:
        switches = int(position.diff().abs().fillna(position.iloc[0]).sum())
    else:
        switches = 0

    return dict(total_ret=total_ret, cagr=cagr, sharpe=sharpe,
                sortino=sortino, max_dd=max_dd, n_switches=switches)

def bh_metrics(price_s):
    """Buy-and-hold metrics for a price series."""
    c    = price_s.dropna().values.astype(float)
    rets = np.diff(c) / c[:-1]
    rets = rets[~np.isnan(rets)]
    if len(rets) == 0:
        return dict(total_ret=0, cagr=0, sharpe=0, sortino=0, max_dd=0, n_switches=0)
    total_ret = c[-1] / c[0] - 1
    cagr      = (c[-1] / c[0]) ** (252 / len(rets)) - 1
    sharpe    = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
    neg       = rets[rets < 0]
    sortino   = (rets.mean() / neg.std()) * np.sqrt(252) if len(neg) > 0 and neg.std() > 0 else 0
    roll_max  = np.maximum.accumulate(c)
    max_dd    = ((c - roll_max) / roll_max).min()
    return dict(total_ret=total_ret, cagr=cagr, sharpe=sharpe,
                sortino=sortino, max_dd=max_dd, n_switches=0)

# -- Slice Data ----------------------------------------------------------------
tqqq_is  = tqqq.loc[IS_START:IS_END]
qqq_is   = qqq.loc[IS_START:IS_END]
spy_is   = spy.loc[IS_START:IS_END]
bil_is   = bil.loc[IS_START:IS_END]

tqqq_oos = tqqq.loc[OOS_START:]
qqq_oos  = qqq.loc[OOS_START:]
spy_oos  = spy.loc[OOS_START:]
bil_oos  = bil.loc[OOS_START:]

# -- In-Sample Optimization ----------------------------------------------------
print("\nRunning IN-SAMPLE optimization...")
is_results = {}
is_curves  = {}

for w in SMA_WINDOWS:
    eq, strat, pos = run_backtest(tqqq_is, qqq_is, bil_is, w)
    is_results[w]  = calc_metrics(eq, strat, pos)
    is_curves[w]   = eq

best_window = max(SMA_WINDOWS, key=lambda w: is_results[w]["sharpe"])

# -- Out-of-Sample Evaluation --------------------------------------------------
print("Running OUT-OF-SAMPLE evaluation...")
oos_results = {}
oos_curves  = {}

for w in SMA_WINDOWS:
    eq, strat, pos = run_backtest(tqqq_oos, qqq_oos, bil_oos, w)
    oos_results[w]  = calc_metrics(eq, strat, pos)
    oos_curves[w]   = eq

oos_best_window = max(SMA_WINDOWS, key=lambda w: oos_results[w]["sharpe"])
winner_held      = (best_window == oos_best_window)

# -- Benchmark Curves ----------------------------------------------------------
bh_tqqq_is  = INITIAL_EQUITY * (tqqq_is  / float(tqqq_is.iloc[0]))
bh_qqq_is   = INITIAL_EQUITY * (qqq_is   / float(qqq_is.iloc[0]))
bh_spy_is   = INITIAL_EQUITY * (spy_is   / float(spy_is.iloc[0]))

bh_tqqq_oos = INITIAL_EQUITY * (tqqq_oos / float(tqqq_oos.iloc[0]))
bh_qqq_oos  = INITIAL_EQUITY * (qqq_oos  / float(qqq_oos.iloc[0]))
bh_spy_oos  = INITIAL_EQUITY * (spy_oos  / float(spy_oos.iloc[0]))

# -- Print Results -------------------------------------------------------------
DIV  = "=" * 78
SDIV = "-" * 78
W    = 11

def fmt_metrics(m, flag=""):
    return (f"{m['cagr']*100:>9.2f}%"
            f"{m['sharpe']:>{W}.3f}"
            f"{m['sortino']:>{W}.3f}"
            f"{m['max_dd']*100:>9.1f}%"
            f"{m['n_switches']:>{W}}"
            f"  {flag}")

hdr_line = (f"  {'Strategy':<22}"
            + "".join(f"{h:>{W}}" for h in ["CAGR", "Sharpe", "Sortino", "Max DD", "Switches"]))

print(f"\n{DIV}")
print(f"  IN-SAMPLE RESULTS  ({IS_START} to {IS_END})  -- optimization period")
print(DIV)
print(hdr_line)
print("  " + SDIV[2:])
for w in SMA_WINDOWS:
    flag = "<< WINNER" if w == best_window else ""
    print(f"  SMA-{w:<18}" + fmt_metrics(is_results[w], flag))
print("  " + SDIV[2:])
print(f"  {'TQQQ Buy & Hold':<22}" + fmt_metrics(bh_metrics(tqqq_is)))
print(f"  {'QQQ Buy & Hold':<22}"  + fmt_metrics(bh_metrics(qqq_is)))
print(f"  {'SPY Buy & Hold':<22}"  + fmt_metrics(bh_metrics(spy_is)))
print(DIV)

print(f"\n{DIV}")
print(f"  OUT-OF-SAMPLE RESULTS  ({OOS_START} to present)  -- evaluation period")
print(DIV)
print(hdr_line)
print("  " + SDIV[2:])
for w in SMA_WINDOWS:
    flags = []
    if w == best_window:     flags.append("IS-winner")
    if w == oos_best_window: flags.append("OOS-best")
    print(f"  SMA-{w:<18}" + fmt_metrics(oos_results[w], "  ".join(flags)))
print("  " + SDIV[2:])
print(f"  {'TQQQ Buy & Hold':<22}" + fmt_metrics(bh_metrics(tqqq_oos)))
print(f"  {'QQQ Buy & Hold':<22}"  + fmt_metrics(bh_metrics(qqq_oos)))
print(f"  {'SPY Buy & Hold':<22}"  + fmt_metrics(bh_metrics(spy_oos)))
print(DIV)

print(f"\n  In-sample  winner : SMA-{best_window}")
print(f"  Out-of-sample best: SMA-{oos_best_window}")
if winner_held:
    print("  Winner HELD out-of-sample -- parameter is robust")
else:
    print("  Winner did NOT hold out-of-sample -- be cautious")

# -- Save Results CSV ----------------------------------------------------------
rows = []
for w in SMA_WINDOWS:
    im = is_results[w]
    om = oos_results[w]
    rows.append({
        "sma_window":   w,
        "is_cagr":      round(im["cagr"],      4),
        "is_sharpe":    round(im["sharpe"],     4),
        "is_sortino":   round(im["sortino"],    4),
        "is_max_dd":    round(im["max_dd"],     4),
        "is_switches":  im["n_switches"],
        "oos_cagr":     round(om["cagr"],       4),
        "oos_sharpe":   round(om["sharpe"],     4),
        "oos_sortino":  round(om["sortino"],    4),
        "oos_max_dd":   round(om["max_dd"],     4),
        "oos_switches": om["n_switches"],
        "is_winner":    w == best_window,
        "oos_best":     w == oos_best_window,
    })
csv_path = os.path.join(PROC_DIR, "optimization_results.csv")
pd.DataFrame(rows).to_csv(csv_path, index=False)
print(f"\n  Results CSV -> {csv_path}")

# -- Chart Helpers -------------------------------------------------------------
PALETTE = {50: "#00FF7F", 100: "#FFD700", 150: "#00BFFF", 200: "#FF69B4", 250: "#FF8C00"}
C_TQQQ  = "#FFD700"
C_QQQ   = "#FFFFFF"
C_SPY   = "#FF4444"

def dollar_fmt(x, _):
    if x >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"${x/1_000:.0f}K"
    return f"${x:.0f}"

def plot_equity_curves(curves, bh_tqqq, bh_qqq, bh_spy, title, filepath):
    fig, ax = plt.subplots(figsize=(14, 6))
    for w, eq in curves.items():
        lw    = 1.8 if w == best_window else 1.0
        label = f"SMA-{w}" + (" << IS-winner" if w == best_window else "")
        ax.plot(eq.index, eq.values, color=PALETTE[w], linewidth=lw,
                label=label, alpha=0.9)
    ax.plot(bh_tqqq.index, bh_tqqq.values, color=C_TQQQ, linewidth=1.1,
            label="TQQQ B&H", alpha=0.85, linestyle="--")
    ax.plot(bh_qqq.index,  bh_qqq.values,  color=C_QQQ,  linewidth=1.0,
            label="QQQ B&H",  alpha=0.70, linestyle="--")
    ax.plot(bh_spy.index,  bh_spy.values,  color=C_SPY,  linewidth=1.0,
            label="SPY B&H",  alpha=0.70, linestyle="--")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(dollar_fmt))
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Chart saved -> {filepath}")

plot_equity_curves(
    is_curves, bh_tqqq_is, bh_qqq_is, bh_spy_is,
    f"TQQQ Optimizer -- In-Sample ({IS_START} to {IS_END})",
    os.path.join(CHARTS_DIR, "01_insample_equity_curves.png")
)
plot_equity_curves(
    oos_curves, bh_tqqq_oos, bh_qqq_oos, bh_spy_oos,
    f"TQQQ Optimizer -- Out-of-Sample ({OOS_START} to present)",
    os.path.join(CHARTS_DIR, "02_outsample_equity_curves.png")
)

# -- Final Warning -------------------------------------------------------------
print(f"\n{'!' * 78}")
print("  WARNING: Out-of-sample results are the only ones that matter.")
print("           In-sample results are for parameter selection only.")
print(f"{'!' * 78}")
