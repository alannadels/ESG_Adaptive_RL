"""Build, cache, and summarize the three per-regime datasets.

Runs the split pipeline once on live data and writes an inspectable CSV per regime, so
the labeling can be sanity-checked before the (expensive) per-regime training and so
training can load the cached splits without re-downloading:

    1. Download the tradable universe (prices + real ESG) and the regime index (SPY).
    2. Label each trading day bull / neutral / bear (causal 50/200 SMA crossover).
    3. Split the dataset into three regime subsets.
    4. Write one long-format CSV per regime (date, ticker, return, E/S/G), plus a
       regime-label CSV, and print a summary (day-counts and date ranges).

Run from the repository root:

    python build_regime_datasets.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from esg_adaptive_rl import config as base_config
from esg_adaptive_rl.data import load_index_close, load_market_data
from esg_adaptive_rl.regimes import RegimeConfig, label_regimes, split_by_regime

# How far back / forward to pull. START is intentionally early; the effective start is
# the youngest ticker's first trading day (rows with any missing ticker are dropped).
START_DATE = "2005-01-01"
END_DATE = "2026-08-09"
REGIME_INDEX = "SPY"
ESG_PATH = "Dataset/ESG_2000-26-ESGC.csv"
ESG_ANCHOR_YEAR = 2025
OUTPUT_DIR = "Dataset/regime_datasets"


def _to_long_frame(subset) -> pd.DataFrame:
    """Flatten a regime :class:`MarketData` subset into a tidy long CSV frame.

    Args:
        subset: A regime subset (dates x tickers arrays for returns and E/S/G).

    Returns:
        A DataFrame with one row per (date, ticker): date, ticker, return, E, S, G.
    """
    records = []
    for i, date in enumerate(subset.dates):
        for j, ticker in enumerate(subset.tickers):
            records.append(
                {
                    "date": pd.Timestamp(date).date(),
                    "ticker": ticker,
                    "return": subset.returns[i, j],
                    "esg_E": subset.esg["E"][i, j],
                    "esg_S": subset.esg["S"][i, j],
                    "esg_G": subset.esg["G"][i, j],
                }
            )
    return pd.DataFrame.from_records(records)


def main() -> None:
    """Build and cache the three per-regime datasets, and print a summary."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Universe (prices + real ESG) and the regime-defining index.
    print(f"Downloading universe ({len(base_config.UNIVERSE)} tickers) + {REGIME_INDEX}...")
    data = load_market_data(
        base_config.UNIVERSE,
        start=START_DATE,
        end=END_DATE,
        esg_source="refinitiv",
        esg_path=ESG_PATH,
        esg_anchor_year=ESG_ANCHOR_YEAR,
    )
    index_prices = load_index_close(REGIME_INDEX, START_DATE, END_DATE)

    # 2. Regime labels on the index. Pinned to the causal 50/200 SMA crossover — the
    # live-matched rule-based detector this project standardizes on. (RegimeConfig now
    # defaults to the walk-forward HMM; it is benchmarked in
    # esg_regime/results/HEURISTICS_BENCHMARKS.md but pinned off here so the cached
    # datasets and labels are reproducible from the crossover with no extra dependency.)
    labels = label_regimes(index_prices, RegimeConfig(detector="crossover"))

    # 3. Split the dataset into regime subsets.
    subsets = split_by_regime(data, labels)

    # 4a. Write one CSV per regime.
    for regime, subset in subsets.items():
        path = os.path.join(OUTPUT_DIR, f"{regime}.csv")
        _to_long_frame(subset).to_csv(path, index=False)

    # 4b. Write the day-by-day regime labels (aligned to the dataset's dates).
    aligned = labels.reindex(pd.to_datetime(pd.Index(data.dates)), method="ffill")
    labels_frame = pd.DataFrame(
        {"date": [pd.Timestamp(d).date() for d in data.dates], "regime": aligned.to_numpy()}
    )
    labels_frame.to_csv(os.path.join(OUTPUT_DIR, "regime_labels.csv"), index=False)

    # Summary.
    first, last = pd.Timestamp(data.dates[0]).date(), pd.Timestamp(data.dates[-1]).date()
    print(f"\nData span: {first} -> {last}  ({len(data.dates)} trading days, "
          f"{len(data.tickers)} tickers)")
    print("Per-regime datasets written to", os.path.abspath(OUTPUT_DIR) + ":")
    for regime, subset in subsets.items():
        n = subset.returns.shape[0]
        r_first = pd.Timestamp(subset.dates[0]).date()
        r_last = pd.Timestamp(subset.dates[-1]).date()
        pct = 100.0 * n / len(data.dates)
        print(f"  {regime:<8} {n:>5} days ({pct:4.1f}%)  {r_first} .. {r_last}  -> {regime}.csv")


if __name__ == "__main__":
    main()
