"""Build the legacy (biased, fixed-50) regime-split pickle for one detector.

Mirrors evolve_regimes_worker.build_regime_splits's ``legacy`` branch -- the original
survivorship-biased universe (``config.UNIVERSE``, prices + real Refinitiv ESG) -- but
parametrized by detector so the biased control can be run across all three detectors
(SMA crossover / EMA crossover / walk-forward HMM), matching the survivorship-free sweep.

Writes ``Dataset/cache/regime_splits_legacy_<detector>.pkl``.

    python build_legacy_splits.py --detector hmm
"""

from __future__ import annotations

import argparse
import os
import pickle

from esg_adaptive_rl import config as base_config
from esg_adaptive_rl.data import load_index_close, load_market_data
from esg_adaptive_rl.regimes import RegimeConfig, label_regimes, split_by_regime

REGIME_INDEX = "SPY"
ESG_PATH = "Dataset/ESG_2000-26-ESGC.csv"
ESG_ANCHOR_YEAR = 2025
OUT_PICKLE = "Dataset/cache/regime_splits_legacy_{detector}.pkl"


def _regime_cfg(detector: str) -> RegimeConfig:
    """Map a study detector key to a dev RegimeConfig (ema = ema-crossover; hmm = walk-forward)."""
    if detector == "crossover":
        return RegimeConfig(ticker=REGIME_INDEX, detector="crossover", ma_type="sma")
    if detector == "ema":
        return RegimeConfig(ticker=REGIME_INDEX, detector="crossover", ma_type="ema")
    if detector == "hmm":
        return RegimeConfig(ticker=REGIME_INDEX, detector="hmm")
    raise ValueError(f"unknown detector {detector!r}; expected crossover|ema|hmm")


def main() -> None:
    """Load the fixed-universe data, label regimes with the chosen detector, split, pickle."""
    ap = argparse.ArgumentParser(description="Build legacy fixed-universe regime splits for one detector.")
    ap.add_argument("--detector", choices=("crossover", "ema", "hmm"), default="crossover")
    args = ap.parse_args()
    out_pickle = OUT_PICKLE.format(detector=args.detector)

    data = load_market_data(
        base_config.UNIVERSE,
        start=base_config.START_DATE,
        end=base_config.END_DATE,
        esg_source="refinitiv",
        esg_path=ESG_PATH,
        esg_anchor_year=ESG_ANCHOR_YEAR,
    )
    index_prices = load_index_close(REGIME_INDEX, base_config.START_DATE, base_config.END_DATE)
    labels = label_regimes(index_prices, _regime_cfg(args.detector))
    splits = split_by_regime(data, labels)

    os.makedirs(os.path.dirname(out_pickle), exist_ok=True)
    with open(out_pickle, "wb") as fh:
        pickle.dump(splits, fh)
    print(f"[{args.detector}] wrote {out_pickle} | "
          + ", ".join(f"{r}={s.returns.shape[0]}d" for r, s in splits.items()))


if __name__ == "__main__":
    main()
