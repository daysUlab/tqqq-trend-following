# GLD vs BIL as the SMA-175 risk-off asset

## Question

What happens if GLD is used instead of BIL during the exact same SMA-175 risk-off periods?

## Method

- Use the existing QQQ SMA-175 signal described for the live strategy.
- Hold TQQQ while risk-on; compare BIL with GLD while risk-off.
- Reuse one signal and position path for both variants.
- Lag the close signal by one trading day, with the SMA calculated from QQQ history beginning in 2009 so the evaluation period has a proper warm-up.
- Apply the repository's 0.04% cost per switch.
- Do not optimize any parameter.

The table below uses the repository scripts' close-to-close return convention and data through 2026-09-09.

## Results

| Period | Risk-off asset | CAGR | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| Full (2010-02-11+) | BIL | 36.05% | 0.899 | -54.87% |
| Full (2010-02-11+) | GLD | 38.99% | 0.938 | -60.38% |
| 2019+ | BIL | 49.38% | 1.043 | -54.87% |
| 2019+ | GLD | 51.94% | 1.069 | -60.38% |

Historically, GLD increased CAGR and slightly improved Sharpe, but it also increased maximum drawdown. It may therefore be an interesting alternative risk-off asset for investors willing to accept more drawdown risk; these results do not suggest replacing BIL outright.

I also checked a next-open variant. The same basic trade-off remained: higher CAGR and Sharpe with GLD, but deeper drawdowns.

The repository contains both SMA-150 and SMA-175 research artifacts. This comparison does not try to resolve that difference; it fixes SMA-175 to match the live strategy described in the root README.

## Reproduce

From the repository root, using the same Python packages already imported by the project:

```bash
python3 gld_riskoff_comparison/gld_riskoff_comparison.py
```

The script downloads adjusted Yahoo Finance data and rewrites `summary.csv`. Pass `--end YYYY-MM-DD` to test a later completed trading day; the default cutoff reproduces the checked-in table.
