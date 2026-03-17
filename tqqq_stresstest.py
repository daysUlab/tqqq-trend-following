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
PROC_DIR   = os.path.join(BASE_DIR, "tqqq_stresstest", "processed")
CHARTS_DIR = os.path.join(BASE_DIR, "tqqq_stresstest", "charts")
os.makedirs(PROC_DIR,   exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

END            = date.today().isoformat()
INITIAL_EQUITY = 10_000.0
SLIP_RT        = 0.0004
SMA_WIN        = 175
DAILY_DECAY    = 0.01 / 252    # 1% annual drag for synthetic TQQQ
BIL_PROXY_RATE = 0.03 / 252   # 3% annual T-bill proxy for pre-BIL dates

# -- Download ------------------------------------------------------------------
print("Downloading QQQ (1999-), TQQQ (2010-), BIL (2004-)...")
qqq_raw  = yf.download("QQQ",  start="1999-01-01", end=END, auto_adjust=True,  progress=False)
tqqq_raw = yf.download("TQQQ", start="2010-02-11", end=END, auto_adjust=True,  progress=False)
bil_raw  = yf.download("BIL",  start="2004-01-01", end=END, auto_adjust=True,  progress=False)

for df in (qqq_raw, tqqq_raw, bil_raw):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

qqq_raw.index  = pd.to_datetime(qqq_raw.index)
tqqq_raw.index = pd.to_datetime(tqqq_raw.index)
bil_raw.index  = pd.to_datetime(bil_raw.index)

qqq  = qqq_raw["Close"].astype(float)
tqqq = tqqq_raw["Close"].astype(float)
bil  = bil_raw["Close"].astype(float)

print(f"  QQQ : {qqq.index[0].date()} to {qqq.index[-1].date()}")
print(f"  TQQQ: {tqqq.index[0].date()} to {tqqq.index[-1].date()}")
print(f"  BIL : {bil.index[0].date()} to {bil.index[-1].date()}")

# -- Build synthetic TQQQ (3x QQQ - daily drag) --------------------------------
qqq_ret       = qqq.pct_change().fillna(0)
synth_ret     = 3.0 * qqq_ret - DAILY_DECAY
synth_tqqq    = 100.0 * (1 + synth_ret).cumprod()

# -- Build unified BIL series (actual + proxy for pre-launch dates) ------------
# Proxy: constant daily return of BIL_PROXY_RATE, anchored at 100
proxy_arr = 100.0 * (1 + BIL_PROXY_RATE) ** np.arange(len(qqq))
bil_proxy = pd.Series(proxy_arr, index=qqq.index)

# Scale and splice: proxy before BIL inception, actual after
first_bil = bil.index[0]
scale     = float(bil_proxy.loc[first_bil]) / float(bil.iloc[0])
bil_scaled = bil * scale
bil_full  = bil_proxy.copy()
bil_full[bil_full.index >= first_bil] = bil_scaled.reindex(
    bil_full.index[bil_full.index >= first_bil]
).ffill()

# -- SMA + signal on full QQQ history ------------------------------------------
sma_full    = qqq.rolling(SMA_WIN).mean()
signal_full = (qqq > sma_full).astype(float).shift(1).fillna(0)

# -- Scenario engine -----------------------------------------------------------
def run_scenario(label, start_str, end_str, use_synth):
    qqq_s = qqq.loc[start_str:end_str]
    pos_s = signal_full.loc[start_str:end_str]
    bil_s = bil_full.loc[start_str:end_str]

    if use_synth:
        tqqq_s     = synth_tqqq.loc[start_str:end_str]
        tqqq_label = "Synth TQQQ B&H"
    else:
        tqqq_s     = tqqq.reindex(qqq_s.index).ffill()
        tqqq_label = "TQQQ B&H"

    tqqq_r = tqqq_s.pct_change().fillna(0)
    bil_r  = bil_s.pct_change().fillna(0)
    qqq_r  = qqq_s.pct_change().fillna(0)

    strat          = pos_s * tqqq_r + (1 - pos_s) * bil_r
    sw             = pos_s.diff().abs()
    sw.iloc[0]     = float(pos_s.iloc[0])
    strat         -= sw * SLIP_RT

    eq_strat = INITIAL_EQUITY * (1 + strat.fillna(0)).cumprod()
    eq_tqqq  = INITIAL_EQUITY * (tqqq_s / float(tqqq_s.iloc[0]))
    eq_qqq   = INITIAL_EQUITY * (qqq_s  / float(qqq_s.iloc[0]))

    def mdd(eq):
        c  = np.concatenate([[float(eq.iloc[0])], eq.values.astype(float)])
        rm = np.maximum.accumulate(c)
        return ((c - rm) / rm).min()

    n          = len(qqq_s)
    d_tqqq     = int((pos_s > 0.5).sum())
    d_bil      = n - d_tqqq

    return dict(label=label, tqqq_label=tqqq_label,
                start=qqq_s.index[0], end=qqq_s.index[-1], n_days=n,
                eq_strat=eq_strat, eq_tqqq=eq_tqqq, eq_qqq=eq_qqq,
                mdd_strat=mdd(eq_strat), mdd_tqqq=mdd(eq_tqqq), mdd_qqq=mdd(eq_qqq),
                final_strat=float(eq_strat.iloc[-1]),
                final_tqqq=float(eq_tqqq.iloc[-1]),
                saved=float(eq_strat.iloc[-1]) - float(eq_tqqq.iloc[-1]),
                d_tqqq=d_tqqq, d_bil=d_bil)

# -- Run scenarios 1-4 ---------------------------------------------------------
print("\nRunning stress scenarios...")
s1 = run_scenario("Dot-com Crash",          "2000-03-10", "2002-10-09", use_synth=True)
s2 = run_scenario("2008 Financial Crisis",  "2007-10-09", "2009-03-09", use_synth=True)
s3 = run_scenario("2020 COVID Crash",       "2020-02-19", "2020-03-23", use_synth=False)
s4 = run_scenario("2022 Inflation Bear",    "2021-11-19", "2022-12-28", use_synth=False)

# -- Scenario 5: dot-com return sequence applied starting today ----------------
dot_strat_rets = s1["eq_strat"].pct_change().fillna(0).values
dot_tqqq_rets  = s1["eq_tqqq"].pct_change().fillna(0).values
dot_qqq_rets   = s1["eq_qqq"].pct_change().fillna(0).values

n5           = len(dot_strat_rets)
future_dates = pd.bdate_range(start=END, periods=n5)

def arr_mdd(arr):
    c  = np.concatenate([[arr[0]], arr])
    rm = np.maximum.accumulate(c)
    return ((c - rm) / rm).min()

s5_eq_strat = pd.Series(INITIAL_EQUITY * np.cumprod(1 + dot_strat_rets), index=future_dates)
s5_eq_tqqq  = pd.Series(INITIAL_EQUITY * np.cumprod(1 + dot_tqqq_rets),  index=future_dates)
s5_eq_qqq   = pd.Series(INITIAL_EQUITY * np.cumprod(1 + dot_qqq_rets),   index=future_dates)

s5 = dict(label="Worst Case: Dot-com Replay from Today",
          tqqq_label="Synth TQQQ B&H",
          start=future_dates[0], end=future_dates[-1], n_days=n5,
          eq_strat=s5_eq_strat, eq_tqqq=s5_eq_tqqq, eq_qqq=s5_eq_qqq,
          mdd_strat=arr_mdd(s5_eq_strat.values),
          mdd_tqqq=arr_mdd(s5_eq_tqqq.values),
          mdd_qqq=arr_mdd(s5_eq_qqq.values),
          final_strat=float(s5_eq_strat.iloc[-1]),
          final_tqqq=float(s5_eq_tqqq.iloc[-1]),
          saved=float(s5_eq_strat.iloc[-1]) - float(s5_eq_tqqq.iloc[-1]),
          d_tqqq=s1["d_tqqq"], d_bil=s1["d_bil"])

all_scenarios = [s1, s2, s3, s4, s5]

# -- Print results -------------------------------------------------------------
DIV  = "=" * 64
SDIV = "-" * 64

def print_scenario(s):
    n = s["n_days"]
    print(f"\n{DIV}")
    print(f"  {s['label']}")
    print(f"  {s['start'].date()} to {s['end'].date()}  ({n} trading days)")
    print(SDIV)
    print(f"  Peak-to-Trough Drawdown:")
    print(f"    Strategy (175d SMA filter)   : {s['mdd_strat']*100:>8.1f}%")
    print(f"    {s['tqqq_label']:<30} : {s['mdd_tqqq']*100:>8.1f}%")
    print(f"    QQQ Buy & Hold               : {s['mdd_qqq']*100:>8.1f}%")
    print(f"  Equity (start $10,000):")
    print(f"    Strategy final               : ${s['final_strat']:>10,.0f}")
    print(f"    {s['tqqq_label']:<30} : ${s['final_tqqq']:>10,.0f}")
    print(f"    Capital saved vs TQQQ B&H    : ${s['saved']:>+10,.0f}")
    mdd_diff = s["mdd_strat"] - s["mdd_tqqq"]
    print(f"    MDD improvement              : {mdd_diff*100:>+8.1f}pp")
    if mdd_diff > 0.10:
        verdict = "PROTECTED -- filter significantly reduced drawdown"
    elif mdd_diff > 0.02:
        verdict = "PARTIAL -- filter provided modest protection"
    else:
        verdict = "MINIMAL -- crash too fast or filter already triggered"
    print(f"  Verdict: {verdict}")
    pct_t = s["d_tqqq"] / n * 100
    pct_b = s["d_bil"]  / n * 100
    print(f"  Days in TQQQ : {s['d_tqqq']:>4} ({pct_t:.1f}%)")
    print(f"  Days in BIL  : {s['d_bil']:>4} ({pct_b:.1f}%)")

for s in all_scenarios:
    print_scenario(s)
print(f"\n{DIV}")

# -- Save summary CSV ----------------------------------------------------------
csv_rows = []
for s in all_scenarios:
    csv_rows.append({
        "scenario":           s["label"],
        "start":              s["start"].date(),
        "end":                s["end"].date(),
        "trading_days":       s["n_days"],
        "mdd_strategy":       round(s["mdd_strat"],  4),
        "mdd_tqqq_bh":        round(s["mdd_tqqq"],   4),
        "mdd_qqq_bh":         round(s["mdd_qqq"],    4),
        "final_strategy":     round(s["final_strat"], 2),
        "final_tqqq_bh":      round(s["final_tqqq"],  2),
        "capital_saved":      round(s["saved"],        2),
        "days_in_tqqq":       s["d_tqqq"],
        "days_in_bil":        s["d_bil"],
    })
csv_path = os.path.join(PROC_DIR, "scenario_results.csv")
pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
print(f"\n  Results CSV -> {csv_path}")

# -- Chart helpers -------------------------------------------------------------
C_STRAT = "#00FF7F"
C_TQQQ  = "#FFD700"
C_QQQ   = "#FFFFFF"

def dollar_ax(ax):
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )

def plot_scenario_ax(ax, s, title, show_legend=True):
    ax.plot(s["eq_strat"].index, s["eq_strat"].values,
            color=C_STRAT, linewidth=1.4, label="Strategy (175d SMA)")
    ax.plot(s["eq_tqqq"].index,  s["eq_tqqq"].values,
            color=C_TQQQ,  linewidth=1.1, label=s["tqqq_label"], alpha=0.85)
    ax.plot(s["eq_qqq"].index,   s["eq_qqq"].values,
            color=C_QQQ,   linewidth=1.0, label="QQQ B&H", alpha=0.75)
    ax.axhline(INITIAL_EQUITY, color="#555555", linewidth=0.6, linestyle="--")
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_ylabel("Portfolio ($)")
    dollar_ax(ax)
    if show_legend:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

# -- Charts 1-4: individual -----------------------------------------------------
chart_specs = [
    (s1, f"Dot-com Crash  |  {s1['start'].date()} to {s1['end'].date()}",
         "01_dotcom.png"),
    (s2, f"2008 Financial Crisis  |  {s2['start'].date()} to {s2['end'].date()}",
         "02_2008.png"),
    (s3, f"2020 COVID Crash  |  {s3['start'].date()} to {s3['end'].date()}",
         "03_2020.png"),
    (s4, f"2022 Inflation Bear  |  {s4['start'].date()} to {s4['end'].date()}",
         "04_2022.png"),
]

for s, title, fname in chart_specs:
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_scenario_ax(ax, s, title)
    fig.tight_layout()
    p = os.path.join(CHARTS_DIR, fname)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Chart saved -> {p}")

# -- Chart 5: 2x2 combined -----------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (s, title, _) in zip(axes.flatten(), chart_specs):
    plot_scenario_ax(ax, s, title, show_legend=(ax is axes[0, 0]))

# Single shared legend in the first panel
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3,
           fontsize=9, bbox_to_anchor=(0.5, 1.01))

fig.suptitle(f"TQQQ {SMA_WIN}-day SMA Filter  --  Crisis Stress Tests  ($10,000 start)",
             fontsize=13, y=1.04)
fig.tight_layout()
p5 = os.path.join(CHARTS_DIR, "05_combined.png")
fig.savefig(p5, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Chart saved -> {p5}")
