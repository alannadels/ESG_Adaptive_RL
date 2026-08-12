"""Build per-regime datasets labeled by the walk-forward HMM (not the crossover).

A sibling of :mod:`build_regime_datasets`, identical in every respect except the
regime detector: labels come from the 3-state walk-forward Gaussian HMM
(:mod:`esg_regime`) instead of the 50/200 SMA crossover. Both emit the same
``bull``/``neutral``/``bear`` vocabulary, so the downstream RL pipeline
(``PortfolioEnv`` -> PPO -> evolutionary search) consumes either without change.

Why a separate builder rather than a flag: ``Dataset/regime_datasets/*.csv`` is
pinned to the crossover so the committed splits stay reproducible without the
``hmmlearn`` dependency. This writes to ``Dataset/regime_datasets_hmm/`` so the
two labelings can be trained and compared side by side.

Run from the repository root:

    python build_regime_datasets_hmm.py
"""

from __future__ import annotations

import os

import pandas as pd

from build_regime_datasets import (
    END_DATE,
    ESG_ANCHOR_YEAR,
    ESG_PATH,
    REGIME_INDEX,
    START_DATE,
    _to_long_frame,
)
from esg_adaptive_rl import config as base_config
from esg_adaptive_rl.data import load_index_close, load_market_data
from esg_adaptive_rl.regimes import RegimeConfig, label_regimes, split_by_regime

OUTPUT_DIR = "Dataset/regime_datasets_hmm"


def main() -> None:
    """Build and cache the three HMM-labeled per-regime datasets."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    # The only difference from build_regime_datasets: the detector. The HMM path
    # reuses esg_regime's cached walk-forward label file for REGIME_INDEX, which is
    # point-in-time (monthly expanding-window refits, filtered posteriors).
    print("Labeling regimes with the walk-forward HMM...")
    labels = label_regimes(index_prices, RegimeConfig(detector="hmm", ticker=REGIME_INDEX))

    subsets = split_by_regime(data, labels)

    for regime, subset in subsets.items():
        _to_long_frame(subset).to_csv(os.path.join(OUTPUT_DIR, f"{regime}.csv"), index=False)

    aligned = labels.reindex(pd.to_datetime(pd.Index(data.dates)), method="ffill")
    pd.DataFrame({
        "date": [pd.Timestamp(d).date() for d in data.dates],
        "regime": aligned.to_numpy(),
    }).to_csv(os.path.join(OUTPUT_DIR, "regime_labels.csv"), index=False)

    first, last = pd.Timestamp(data.dates[0]).date(), pd.Timestamp(data.dates[-1]).date()
    print(f"\nData span: {first} -> {last}  ({len(data.dates)} trading days, "
          f"{len(data.tickers)} tickers)")
    print("HMM-labeled per-regime datasets written to", os.path.abspath(OUTPUT_DIR) + ":")
    for regime, subset in subsets.items():
        n = subset.returns.shape[0]
        pct = 100.0 * n / len(data.dates)
        print(f"  {regime:<8} {n:>5} days ({pct:4.1f}%)  "
              f"{pd.Timestamp(subset.dates[0]).date()} .. "
              f"{pd.Timestamp(subset.dates[-1]).date()}  -> {regime}.csv")


if __name__ == "__main__":
    main()
