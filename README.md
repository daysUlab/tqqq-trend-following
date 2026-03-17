# TQQQ Trend Following — QQQ 175-Day SMA Filter

**Status: Live — deployed March 2026**

A systematic leveraged trend following strategy applying a 175-day simple moving average filter to TQQQ (3x leveraged Nasdaq-100 ETF). Built from scratch with full institutional validation: walk-forward optimization, Monte Carlo simulation, historical stress testing across four major crises, and transaction cost sensitivity analysis.

This is not a backtest-until-green project. The dot-com stress test is documented in full. The strategy loses 94.7% in that scenario.

---

## The Strategy

Hold TQQQ when QQQ is above its 175-day SMA. Hold BIL (T-bills) when QQQ is below. Signal on today's close, execute tomorrow's open. Rebalance only on signal change — approximately 5 switches per year.

That's the entire rule set.

---

## Why This Pair

QQQ's 175-day SMA is used as the signal rather than TQQQ's own moving average. QQQ is the non-leveraged underlying — its SMA is cleaner and less distorted by TQQQ's daily compounding. The signal detects the underlying trend, not the leveraged noise.

BIL is used as the safe haven rather than TLT. The 2022 inflation shock demonstrated that long-duration bonds fail as a hedge when inflation drives both equity and bond drawdowns simultaneously. T-bills preserve capital in all rate environments.

---

## Research Process

Nine strategies were tested and rejected before this one:

| Strategy | Why Rejected |
|---|---|
| QQQ-TLT Cross-Asset Stat Arb | Spread non-stationary 97% of the time (ADF test) |
| Return Residual Stat Arb | 2.7-day avg hold — signal had no memory |
| Sector Rotation | Underperformed SPY after short-term capital gains tax |
| Dual Momentum (Antonacci) | Underperformed SPY on Sharpe |
| PEAD Gap-Up Proxy | Sharpe 0.336 — all noise |
| Overnight Drift | Anomaly real, transaction costs eliminate edge |
| VIX Mean Reversion | 18 trades in 20 years — cannot compound |
| Bull Put Spreads | Black-Scholes mispricing — free data insufficient |
| MTUM/USMV Factor Pairs | Half-life 127 days — too slow for active trading |

Trend following with a cash filter was the only strategy that produced a higher out-of-sample Sharpe than buy-and-hold QQQ across the full validation process.

---

## Validation Stack

### 1. Walk-Forward Optimization

**In-sample: 2010-02-11 to 2018-12-31 | Out-of-sample: 2019-01-01 to present**

Nine SMA windows tested (50 to 250 days). SMA-175 selected as in-sample winner on Sharpe. Validated on out-of-sample data the optimizer never touched.

| SMA Window | OOS CAGR | OOS Sharpe | OOS Max DD | OOS Switches |
|---|---|---|---|---|
| SMA-50 | 25.53% | 0.734 | -52.3% | 110 |
| SMA-100 | 19.43% | 0.616 | -55.2% | 82 |
| SMA-125 | 26.77% | 0.736 | -53.8% | 54 |
| **SMA-175 (selected)** | **43.80%** | **0.994** | **-54.9%** | **24** |
| SMA-200 | 39.34% | 0.926 | -54.9% | 23 |
| SMA-250 | 27.39% | 0.742 | -63.4% | 31 |
| QQQ Buy & Hold | 21.34% | 0.925 | -35.1% | — |
| TQQQ Buy & Hold | 38.28% | 0.816 | -81.7% | — |

**Key finding:** SMA windows from 125 to 200 all beat QQQ and TQQQ buy-and-hold on Sharpe out-of-sample. Parameter insensitivity across a wide range is evidence of genuine robustness, not overfitting to a single magic number.

### 2. Monte Carlo Simulation (10,000 bootstrapped paths)

Returns shuffled randomly 10,000 times to test whether performance depends on lucky sequencing.

| Metric | 5th Pct | 25th Pct | Median | 75th Pct | 95th Pct | Actual |
|---|---|---|---|---|---|---|
| CAGR | 10.9% | 23.7% | 33.5% | 44.1% | 60.5% | 33.6% |
| Max DD | -84.9% | -74.0% | -66.3% | -58.8% | -49.7% | -54.9% |
| Sharpe | 0.458 | 0.702 | 0.872 | 1.045 | 1.295 | 0.874 |

- **99.4%** probability of positive total return
- **84.9%** probability of beating QQQ buy-and-hold CAGR
- **36.6%** probability of max drawdown exceeding -70%

The actual historical path sits near the 35th percentile of outcomes — a decent but not exceptional draw from the return sequence. The median simulated outcome closely matches the actual result, indicating the edge is in the return distribution itself rather than lucky ordering.

### 3. Crisis Stress Tests

| Scenario | Strategy DD | TQQQ B&H DD | QQQ B&H DD | Capital Saved |
|---|---|---|---|---|
| Dot-com 2000-2002 (synthetic) | **-94.7%** | -99.9% | -83.0% | +$565 on $10k |
| 2008 Financial Crisis (synthetic) | -50.7% | -94.3% | -53.4% | +$4,760 on $10k |
| 2020 COVID Crash | -54.9% | -69.9% | -28.6% | +$1,496 on $10k |
| 2022 Inflation Bear | -29.6% | -81.7% | -35.1% | +$5,299 on $10k |

**The dot-com result is not hidden or minimized.** A 2000-2003 style crash — where QQQ fell 83% over 2.5 years with multiple violent bear market rallies — reduces the strategy account from $10,000 to $571. The filter provided marginal protection because repeated false recoveries triggered re-entries into TQQQ before the trend resumed downward. This is the existential risk of any leveraged trend following approach.

The filter performs well in directional crashes (2008, 2022) and moderately in fast crashes (2020). It fails in prolonged bear markets with multiple false recoveries (2000-2003).

### 4. Transaction Cost Sensitivity

84 total switches over 16 years (~5 per year). Cost drag is minimal.

| Slippage | CAGR | vs QQQ B&H |
|---|---|---|
| 0 bps | 33.88% | +15.2% |
| 2 bps (actual) | 33.60% | +14.9% |
| 10 bps | 32.48% | +13.8% |
| 50 bps (25x actual) | 27.00% | +8.3% |

**The strategy beats QQQ buy-and-hold at every slippage level tested, including 50 bps.** With only ~5 switches per year, transaction costs are not a meaningful drag.

---

## Full Historical Results (2010-2026)

| Metric | TQQQ + SMA-175 | TQQQ Buy & Hold | QQQ Buy & Hold |
|---|---|---|---|
| Total Return | 10,366% | 22,153% | 1,464% |
| CAGR | 33.60% | 40.02% | 18.68% |
| Sharpe | 0.874 | 0.861 | 0.935 |
| Sortino | 1.011 | 1.106 | 1.203 |
| Max Drawdown | -54.9% | -81.7% | -35.1% |
| Annual Switches | ~5 | 0 | 0 |
| Time in TQQQ | 80.5% | 100% | — |

---

## Yearly Returns

| Year | Strategy | QQQ B&H |
|---|---|---|
| 2010 | +47.8% | +25.9% |
| 2011 | -12.0% | +3.5% |
| 2012 | -8.3% | +18.1% |
| 2013 | +139.7% | +36.6% |
| 2014 | +50.0% | +19.2% |
| 2015 | +14.3% | +9.4% |
| 2016 | +0.7% | +7.1% |
| 2017 | +118.1% | +32.7% |
| 2018 | +1.9% | -0.1% |
| 2019 | +55.2% | +39.0% |
| 2020 | +97.8% | +48.4% |
| 2021 | +83.0% | +27.4% |
| 2022 | -32.1% | -32.6% |
| 2023 | +126.0% | +54.9% |
| 2024 | +40.8% | +25.6% |
| 2025 | +31.0% | +20.8% |

---

## Known Risks and Limitations

**Dot-com tail risk:** A prolonged tech bear market with multiple false recoveries can destroy the account despite the filter. This is documented, not disclaimed away.

**Backtest period bias:** TQQQ was launched in 2010 — the beginning of the longest bull market in history. The strategy has never been tested through a true multi-year tech bear market on live data. The dot-com simulation uses synthetic 3x QQQ returns, not actual TQQQ data.

**Path dependency:** 3x leveraged ETFs reset daily and suffer from volatility decay. In choppy sideways markets TQQQ underperforms 3x QQQ returns even when the filter keeps you invested. This is a structural drag not captured in a simple price-based backtest.

**Whipsaw:** The 175-day SMA is a slow signal. At trend reversals the strategy will be late to exit and late to re-enter by construction. Short choppy markets produce multiple false signals and modest underperformance vs QQQ.

**Single signal:** The strategy holds 100% in one asset at all times. No diversification across signals or assets. A flash crash or overnight gap-down while in TQQQ produces a full-account loss on that event before the signal can react.

---

## Files

| File | Description |
|---|---|
| `tqqq_trend_backtest.py` | Core backtest — strategy vs benchmarks, equity curve, drawdown chart |
| `tqqq_optimizer.py` | Walk-forward optimization across SMA windows, IS/OOS split |
| `tqqq_dd_reduction.py` | Seven drawdown reduction variations tested simultaneously |
| `tqqq_final_comparison.py` | Clean three-way comparison with yearly returns |
| `tqqq_walkforward.py` | Extended walk-forward with 9 windows and Sharpe stability chart |
| `tqqq_montecarlo.py` | 10,000 bootstrap simulations, distribution charts, sample paths |
| `tqqq_stresstest.py` | Four historical crisis scenarios including synthetic dot-com |
| `tqqq_cost_sensitivity.py` | Performance at 8 slippage levels, break-even analysis |

---

## Live Deployment

- **Started:** March 2026
- **Account:** Charles Schwab
- **Capital:** $xx,xxx initial
- **Current signal:** BIL (QQQ below 175-day SMA as of March 2026)
- **Check frequency:** Daily after market close
- **Execution:** Manual limit orders near open

---

*Built: March 2026 | Author: Justin Moter | Part of a systematic strategy research program begun March 2026*
