import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import yfinance as yf
from datetime import date

plt.style.use("dark_background")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROC_DIR   = os.path.join(BASE_DIR, "tqqq_trend", "processed")
CHARTS_DIR = os.path.join(BASE_DIR, "tqqq_trend", "charts")
os.makedirs(PROC_DIR,   exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

START          = "2010-02-11"   # TQQQ inception
END            = date.today().isoformat()
INITIAL_EQUITY = 10_000.0
SMA_WIN        = 150
SLIP_RT        = 0.0004         # 2 bps per side

# ── Download — work directly in memory ────────────────────────────────────────
print(f"Downloading TQQQ, QQQ, SPY, BIL  |  {START} to {END}")
raw = yf.download(["TQQQ", "QQQ", "SPY", "BIL"], start=START, end=END,
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
dates = prices.index
n     = len(prices)
print(f"  Common data: {dates[0].date()} to {dates[-1].date()}  ({n} rows)")

# ── Strategy 4: TQQQ with QQQ 150-day SMA Filter ─────────────────────────────
sma150   = qqq.rolling(SMA_WIN).mean()
signal   = (qqq > sma150).astype(float)
position = signal.shift(1).fillna(0)   # execute tomorrow on today's signal

tqqq_ret = tqqq.pct_change().fillna(0)
bil_ret  = bil.pct_change().fillna(0)
strat    = position * tqqq_ret + (1 - position) * bil_ret

switch        = position.diff().abs()
switch.iloc[0] = position.iloc[0]
strat[switch.astype(bool)] -= SLIP_RT

filtered_eq = INITIAL_EQUITY * (1 + strat.fillna(0)).cumprod()

# ── Buy-and-Hold Curves ───────────────────────────────────────────────────────
tqqq_bh = INITIAL_EQUITY * (tqqq / float(tqqq.iloc[0]))
qqq_bh  = INITIAL_EQUITY * (qqq  / float(qqq.iloc[0]))
spy_bh  = INITIAL_EQUITY * (spy  / float(spy.iloc[0]))

# ── Metrics Helper ────────────────────────────────────────────────────────────
def calc_metrics(equity, daily_rets, position=None):
    c    = equity.dropna().values.astype(float)
    rets = daily_rets.dropna().values.astype(float)
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
    switches = int(position.diff().abs().fillna(position.iloc[0]).sum()) if position is not None else 0
    pct_tqqq = float(position.mean()) * 100 if position is not None else None
    return dict(total=total, cagr=cagr, sharpe=sh, sortino=so,
                max_dd=mdd, switches=switches, pct_tqqq=pct_tqqq)

# ── Print Results ─────────────────────────────────────────────────────────────
def print_metrics(label, m):
    print(f"\n  {label}")
    print(f"  {'─'*46}")
    print(f"  Total return   : {m['total']*100:>9.1f}%")
    print(f"  CAGR           : {m['cagr']*100:>9.2f}%")
    print(f"  Sharpe         : {m['sharpe']:>9.3f}")
    print(f"  Sortino        : {m['sortino']:>9.3f}")
    print(f"  Max drawdown   : {m['max_dd']*100:>9.1f}%")
    if m['switches']:
        pct_bil = 100 - m['pct_tqqq']
        print(f"  Switches       : {m['switches']:>9}")
        print(f"  Time in TQQQ   : {m['pct_tqqq']:>8.1f}%")
        print(f"  Time in BIL    : {pct_bil:>8.1f}%")

DIV = "=" * 50
print(f"\n{DIV}")
print("  TQQQ TREND FILTER BACKTEST")
print(f"  {dates[0].date()} to {dates[-1].date()}")
print(DIV)

m_filtered = calc_metrics(filtered_eq, strat, position)
m_tqqq_bh  = calc_metrics(tqqq_bh, tqqq.pct_change().fillna(0))
m_qqq_bh   = calc_metrics(qqq_bh,  qqq.pct_change().fillna(0))
m_spy_bh   = calc_metrics(spy_bh,  spy.pct_change().fillna(0))

print_metrics("TQQQ + QQQ 150-day SMA Filter", m_filtered)
print_metrics("TQQQ Buy & Hold", m_tqqq_bh)
print_metrics("QQQ Buy & Hold",  m_qqq_bh)
print_metrics("SPY Buy & Hold",  m_spy_bh)
print(f"\n{DIV}")

# ── Trade Log ─────────────────────────────────────────────────────────────────
trade_log  = []
in_tqqq    = False
entry_date = None
entry_px   = None

for i, dt in enumerate(dates):
    pos_today = int(position.iloc[i])
    # Entered TQQQ
    if pos_today == 1 and not in_tqqq:
        in_tqqq    = True
        entry_date = dt
        entry_px   = float(tqqq.iloc[i])
    # Exited TQQQ
    elif pos_today == 0 and in_tqqq:
        exit_px = float(tqqq.iloc[i])
        trade_log.append({
            "entry_date":  entry_date.date(),
            "exit_date":   dt.date(),
            "entry_tqqq":  round(entry_px, 4),
            "exit_tqqq":   round(exit_px, 4),
            "hold_days":   (dt - entry_date).days,
            "return_pct":  round(exit_px / entry_px - 1, 6),
        })
        in_tqqq = False

# Close any open position at end
if in_tqqq:
    exit_px = float(tqqq.iloc[-1])
    trade_log.append({
        "entry_date":  entry_date.date(),
        "exit_date":   dates[-1].date(),
        "entry_tqqq":  round(entry_px, 4),
        "exit_tqqq":   round(exit_px, 4),
        "hold_days":   (dates[-1] - entry_date).days,
        "return_pct":  round(exit_px / entry_px - 1, 6),
    })

tlog_path = os.path.join(PROC_DIR, "trade_log.csv")
pd.DataFrame(trade_log).to_csv(tlog_path, index=False)
print(f"\n  Trade log saved ({len(trade_log)} entries) -> {tlog_path}")

# ── Drawdown Helper ───────────────────────────────────────────────────────────
def drawdown_series(equity):
    c  = equity.values.astype(float)
    rm = np.maximum.accumulate(c)
    dd = (c - rm) / rm
    return pd.Series(dd, index=equity.index)

dd_filtered = drawdown_series(filtered_eq)
dd_tqqq     = drawdown_series(tqqq_bh)
dd_qqq      = drawdown_series(qqq_bh)
dd_spy      = drawdown_series(spy_bh)

C_FILTERED = "#00FF7F"   # bright green
C_TQQQ     = "#FFD700"   # yellow
C_SPY      = "#FF4444"   # red
C_QQQ      = "#FFFFFF"   # white

# ── Chart 1: Equity Curves (log scale) ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(filtered_eq.index, filtered_eq.values, color=C_FILTERED, linewidth=1.4,
        label="TQQQ + QQQ 150d SMA Filter", zorder=4)
ax.plot(tqqq_bh.index,    tqqq_bh.values,    color=C_TQQQ,     linewidth=1.1,
        label="TQQQ Buy & Hold", alpha=0.85)
ax.plot(qqq_bh.index,     qqq_bh.values,     color=C_QQQ,      linewidth=1.0,
        label="QQQ Buy & Hold",  alpha=0.75)
ax.plot(spy_bh.index,     spy_bh.values,     color=C_SPY,      linewidth=1.0,
        label="SPY Buy & Hold",  alpha=0.75)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.set_title("TQQQ Trend Filter — Equity Curve (Log Scale)", fontsize=14, pad=12)
ax.set_ylabel("Portfolio Value ($)")
ax.legend(fontsize=9)
ax.grid(alpha=0.2, which="both")
fig.tight_layout()
p1 = os.path.join(CHARTS_DIR, "01_equity_curve.png")
fig.savefig(p1, dpi=150)
plt.close(fig)
print(f"  Chart saved -> {p1}")

# ── Chart 2: Drawdown ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.fill_between(dd_filtered.index, dd_filtered.values * 100, 0,
                color=C_FILTERED, alpha=0.25)
ax.plot(dd_filtered.index, dd_filtered.values * 100, color=C_FILTERED, linewidth=1.3,
        label="TQQQ + QQQ 150d SMA Filter")
ax.plot(dd_tqqq.index,     dd_tqqq.values * 100,     color=C_TQQQ,     linewidth=1.0,
        label="TQQQ Buy & Hold", alpha=0.85)
ax.plot(dd_qqq.index,      dd_qqq.values * 100,      color=C_QQQ,      linewidth=1.0,
        label="QQQ Buy & Hold",  alpha=0.75)
ax.plot(dd_spy.index,      dd_spy.values * 100,       color=C_SPY,      linewidth=1.0,
        label="SPY Buy & Hold",  alpha=0.75)
ax.set_title("TQQQ Trend Filter — Drawdown", fontsize=14, pad=12)
ax.set_ylabel("Drawdown (%)")
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
fig.tight_layout()
p2 = os.path.join(CHARTS_DIR, "02_drawdown.png")
fig.savefig(p2, dpi=150)
plt.close(fig)
print(f"  Chart saved -> {p2}")
