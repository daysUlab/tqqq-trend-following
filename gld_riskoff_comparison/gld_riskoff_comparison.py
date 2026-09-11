#!/usr/bin/env python3
"""Compare BIL and GLD under the same QQQ SMA-175 position path."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


OUT = Path(__file__).resolve().parent
TICKERS = ["QQQ", "TQQQ", "BIL", "GLD"]
DOWNLOAD_START = "2009-01-01"  # provides QQQ history for the SMA warm-up
DEFAULT_END = "2026-09-09"  # fixed cutoff for the checked-in results
SMA_WINDOW = 175
INITIAL_VALUE = 10_000.0
SWITCH_COST = 0.0004  # repository convention: 2 bps per side
OOS_START = pd.Timestamp("2019-01-01")


def download_prices(end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    end_date = pd.Timestamp(end)
    raw = yf.download(
        TICKERS,
        start=DOWNLOAD_START,
        # A short buffer matches the original experiment's request; rows after
        # the requested cutoff are discarded below.
        end=(end_date + pd.Timedelta(days=4)).date().isoformat(),
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=True,
    )
    if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError("Yahoo Finance returned no usable multi-ticker data")

    close = raw["Close"].reindex(columns=TICKERS)
    open_ = raw["Open"].reindex(columns=TICKERS)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    open_.index = pd.to_datetime(open_.index).tz_localize(None)
    close = close.loc[:end_date]
    open_ = open_.loc[:end_date]

    aligned = pd.concat({"close": close.ffill(), "open": open_.ffill()}, axis=1)
    required = [(field, ticker) for field in ("close", "open") for ticker in TICKERS]
    common = aligned.dropna(subset=required)
    if common.empty or common.index.max() != end_date:
        raise RuntimeError(f"no complete data through requested end date {end}")
    return close, common


def build_position(close: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    qqq = close["QQQ"].dropna()
    sma = qqq.rolling(SMA_WINDOW, min_periods=SMA_WINDOW).mean()
    signal = qqq.gt(sma)

    # Today's holding uses only the signal known after the previous close.
    position = signal.shift(1).reindex(index).fillna(False).astype(int)
    expected = qqq.shift(1).gt(sma.shift(1)).reindex(index).fillna(False).astype(int)
    pd.testing.assert_series_equal(position, expected, check_names=False)
    if not position.isin([0, 1]).all() or not position.diff().abs().gt(0).any():
        raise AssertionError("expected a changing binary position series")
    return position.rename("tqqq_position")


def switch_cost(position: pd.Series) -> pd.Series:
    switches = position.diff().abs().fillna(float(position.iloc[0]))
    return switches * SWITCH_COST


def close_returns(
    common: pd.DataFrame, position: pd.Series, risk_off: str
) -> pd.Series:
    tqqq = common[("close", "TQQQ")].pct_change().fillna(0.0)
    defensive = common[("close", risk_off)].pct_change().fillna(0.0)
    return position * tqqq + (1 - position) * defensive - switch_cost(position)


def next_open_returns(
    common: pd.DataFrame, position: pd.Series, risk_off: str
) -> pd.Series:
    """Hold the old asset overnight and the new asset after a switch-day open."""
    returns = np.zeros(len(common), dtype=float)
    holdings = position.to_numpy(dtype=int)
    for i in range(1, len(common)):
        old_asset = "TQQQ" if holdings[i - 1] else risk_off
        new_asset = "TQQQ" if holdings[i] else risk_off
        if old_asset == new_asset:
            factor = common[("close", new_asset)].iloc[i] / common[
                ("close", new_asset)
            ].iloc[i - 1]
        else:
            overnight = common[("open", old_asset)].iloc[i] / common[
                ("close", old_asset)
            ].iloc[i - 1]
            intraday = common[("close", new_asset)].iloc[i] / common[
                ("open", new_asset)
            ].iloc[i]
            factor = overnight * intraday
        returns[i] = float(factor - 1.0)
        if old_asset != new_asset:
            returns[i] -= SWITCH_COST
    return pd.Series(returns, index=common.index)


def metrics(
    returns: pd.Series,
    position: pd.Series,
    start: pd.Timestamp,
    period: str,
    execution: str,
    risk_off: str,
) -> dict[str, object]:
    r = returns.loc[start:]
    p = position.loc[r.index]
    equity = INITIAL_VALUE * (1 + r).cumprod()
    wealth = np.r_[INITIAL_VALUE, equity.to_numpy()]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    volatility = float(r.to_numpy().std(ddof=0))
    return {
        "execution": execution,
        "period": period,
        "risk_off_asset": risk_off,
        "start": r.index.min().date().isoformat(),
        "end": r.index.max().date().isoformat(),
        "trading_days": len(r),
        "cagr": float((equity.iloc[-1] / INITIAL_VALUE) ** (252 / len(r)) - 1),
        "sharpe": float(r.mean() / volatility * math.sqrt(252)),
        "max_drawdown": float(drawdown.min()),
        "switches": int(position.diff().abs().fillna(position.iloc[0]).loc[r.index].sum()),
        "tqqq_exposure": float(p.mean()),
    }


def run(end: str) -> pd.DataFrame:
    close, common = download_prices(end)
    position = build_position(close, common.index)
    returns = {
        ("existing_close", asset): close_returns(common, position, asset)
        for asset in ("BIL", "GLD")
    }
    returns.update(
        {
            ("next_open", asset): next_open_returns(common, position, asset)
            for asset in ("BIL", "GLD")
        }
    )

    # Both alternatives must use exactly the same dates and position path.
    for series in returns.values():
        if not series.index.equals(position.index) or series.isna().any():
            raise AssertionError("return series is not aligned to the shared position")

    periods = {"Full": common.index.min(), "2019+": OOS_START}
    rows = [
        metrics(returns[(execution, asset)], position, start, period, execution, asset)
        for execution in ("existing_close", "next_open")
        for period, start in periods.items()
        for asset in ("BIL", "GLD")
    ]
    result = pd.DataFrame(rows)
    for _, group in result.groupby(["execution", "period"]):
        if group["switches"].nunique() != 1 or group["tqqq_exposure"].nunique() != 1:
            raise AssertionError("BIL and GLD did not share the same position path")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        help=f"inclusive data cutoff (default: {DEFAULT_END})",
    )
    args = parser.parse_args()
    result = run(args.end)
    summary = result.round(
        {"cagr": 4, "sharpe": 3, "max_drawdown": 4, "tqqq_exposure": 4}
    )
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
