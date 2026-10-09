"""Build per-regime datasets labeled by the EMA (exponential) 50/200 crossover.

The third regime detector, a sibling of :mod:`build_regime_datasets` (SMA crossover)
and :mod:`build_regime_datasets_hmm` (walk-forward HMM). It uses the same fast/slow
crossover rule but with *exponential* moving averages (``ma_type="ema"``), which weight
recent prices more heavily and so react a little sooner to turns than the equal-weighted
SMA. Emitting the same ``bull``/``neutral``/``bear`` vocabulary, it feeds the identical
downstream RL pipeline and lets the paper report the finding across three methodologically
distinct regime definitions (SMA / EMA / HMM).

Design choices that keep the three detector families strictly comparable:

  * **Shared universe.** The tradable universe (prices + real ESG) is *reconstructed*
    from the committed crossover CSVs (``Dataset/regime_datasets/{bull,neutral,bear}.csv``,
    which partition the same days) rather than re-downloaded. This pins the EMA family to
    the exact same per-day returns and ESG as the committed SMA and HMM families, so the
    only thing that differs across the three is the regime labeling.
  * **Same index source as the SMA crossover.** Regimes are labeled on *adjusted* SPY
    (``load_index_close``), identical to :mod:`build_regime_datasets`, so EMA vs SMA is a
    clean averaging-method comparison and not a data-source artifact.

Run from the repository root:

    python build_regime_datasets_ema.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from build_regime_datasets import (
    END_DATE,
    REGIME_INDEX,
    START_DATE,
    _to_long_frame,
)
from esg_adaptive_rl.data import MarketData, load_index_close
from esg_adaptive_rl.regimes import RegimeConfig, label_regimes, split_by_regime

# The committed crossover splits, whose union is the full universe (offline, fixed).
CROSSOVER_DIR = "Dataset/regime_datasets"
OUTPUT_DIR = "Dataset/regime_datasets_ema"


def reconstruct_universe(crossover_dir: str = CROSSOVER_DIR) -> MarketData:
    """Rebuild the full-universe :class:`MarketData` from the committed crossover CSVs.

    The three crossover regime files partition every trading day exactly once, so
    concatenating their long-format rows and pivoting reconstructs the complete
    (date x ticker) returns and E/S/G arrays with no re-download.

    Args:
        crossover_dir: Directory holding ``bull.csv``, ``neutral.csv``, ``bear.csv``.

    Returns:
        The full-universe :class:`MarketData`, dates sorted ascending.
    """
    frames = [
        pd.read_csv(os.path.join(crossover_dir, f"{regime}.csv"), parse_dates=["date"])
        for regime in ("bull", "neutral", "bear")
    ]
    long = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"])
    tickers = sorted(long["ticker"].unique())
    returns = long.pivot(index="date", columns="ticker", values="return").reindex(columns=tickers)
    esg = {
        pillar: long.pivot(index="date", columns="ticker", values=f"esg_{pillar}")
        .reindex(columns=tickers)
        .to_numpy(dtype=np.float64)
        for pillar in ("E", "S", "G")
    }
    return MarketData(
        dates=returns.index,
        tickers=list(tickers),
        returns=returns.to_numpy(dtype=np.float64),
        esg=esg,
    )


def main() -> None:
    """Build and cache the three EMA-labeled per-regime datasets."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Reconstruct the shared universe (offline) and load the adjusted regime index.
    print("Reconstructing universe from committed crossover CSVs...")
    data = reconstruct_universe()
    print(f"  universe: {len(data.dates)} trading days x {len(data.tickers)} tickers")
    index_prices = load_index_close(REGIME_INDEX, START_DATE, END_DATE)

    # 2. Label regimes with the EMA (exponential) 50/200 crossover.
    print("Labeling regimes with the EMA 50/200 crossover...")
    ema_labels = label_regimes(index_prices, RegimeConfig(detector="crossover", ma_type="ema"))

    # Cross-check: the SMA crossover on this same index should reproduce the committed
    # crossover labels, confirming the index source matches what the committed datasets
    # used (so EMA vs SMA is a pure averaging-method difference).
    sma_labels = label_regimes(index_prices, RegimeConfig(detector="crossover", ma_type="sma"))
    committed = pd.read_csv(os.path.join(CROSSOVER_DIR, "regime_labels.csv"), parse_dates=["date"])
    sma_aligned = sma_labels.reindex(pd.to_datetime(committed["date"]), method="ffill").to_numpy()
    agree = float(np.mean(sma_aligned == committed["regime"].to_numpy()))
    print(f"  SMA-vs-committed crossover label agreement: {100 * agree:.2f}% "
          f"(confirms the SPY source matches)")

    # 3. Split the shared universe by the EMA labels and write the CSVs.
    subsets = split_by_regime(data, ema_labels)
    for regime, subset in subsets.items():
        _to_long_frame(subset).to_csv(os.path.join(OUTPUT_DIR, f"{regime}.csv"), index=False)

    aligned = ema_labels.reindex(pd.to_datetime(pd.Index(data.dates)), method="ffill")
    pd.DataFrame({
        "date": [pd.Timestamp(d).date() for d in data.dates],
        "regime": aligned.to_numpy(),
    }).to_csv(os.path.join(OUTPUT_DIR, "regime_labels.csv"), index=False)

    # 4. Summary.
    first, last = pd.Timestamp(data.dates[0]).date(), pd.Timestamp(data.dates[-1]).date()
    print(f"\nData span: {first} -> {last}  ({len(data.dates)} trading days)")
    print("EMA-labeled per-regime datasets written to", os.path.abspath(OUTPUT_DIR) + ":")
    for regime, subset in subsets.items():
        n = subset.returns.shape[0]
        pct = 100.0 * n / len(data.dates)
        print(f"  {regime:<8} {n:>5} days ({pct:4.1f}%)  "
              f"{pd.Timestamp(subset.dates[0]).date()} .. "
              f"{pd.Timestamp(subset.dates[-1]).date()}  -> {regime}.csv")


if __name__ == "__main__":
    main()
