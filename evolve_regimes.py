"""Entry point: evolve a specialist reward-weight strategy for each market regime.

This ties the pieces together into the project's core experiment:

    1. Label each trading day bull / neutral / bear from a market index (SPY) with a
       causal moving-average rule (:mod:`esg_adaptive_rl.regimes`).
    2. Split the dataset into three regime subsets.
    3. For each regime, run the eight-optimizer nature-inspired search
       (:mod:`evolution`) to discover the reward weighting that performs best on that
       regime's days — an ad-hoc, regime-specialist strategy.

The three regimes' evolved weight vectors, compared side by side, are the headline
finding: which of (return, E, S, G, tail risk) matters most in which market regime.

Note on honest evaluation: the per-regime numbers here assume the regime is known; the
realistic, live number comes from the meta-controller running the *same* causal detector
used for this split. Day-counts per regime are printed so the (small) bear sample is
visible.

Run from the repository root:

    python evolve_regimes.py
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from esg_adaptive_rl import config as base_config
from esg_adaptive_rl.data import (
    load_index_close,
    load_market_data,
    split_by_fraction,
)
from esg_adaptive_rl.regimes import RegimeConfig, label_regimes, split_by_regime
from esg_adaptive_rl.reward import FACTOR_NAMES, RewardWeights
from evolution.config import EvolutionConfig
from evolution.search import run_search
from evolve_weights import make_objective

# The market index whose trend defines the regimes, and the real ESG source.
REGIME_INDEX = "SPY"
ESG_PATH = "Dataset/ESG_2000-26-ESGC.csv"
ESG_ANCHOR_YEAR = 2025


def main() -> None:
    """Label regimes, split the dataset, and evolve a strategy per regime."""
    evo_cfg = EvolutionConfig()
    regime_cfg = RegimeConfig()  # 50/200 SMA crossover, causal min-dwell
    np.random.seed(evo_cfg.seed)

    # Load the tradable universe (prices + real ESG) and the regime-defining index.
    print("Loading universe data (prices + real ESG) and the regime index...")
    data = load_market_data(
        base_config.UNIVERSE,
        start=base_config.START_DATE,
        end=base_config.END_DATE,
        esg_source="refinitiv",
        esg_path=ESG_PATH,
        esg_anchor_year=ESG_ANCHOR_YEAR,
    )
    index_prices = load_index_close(REGIME_INDEX, base_config.START_DATE, base_config.END_DATE)

    # Label regimes on the index and split the dataset into regime subsets.
    labels = label_regimes(index_prices, regime_cfg)
    subsets = split_by_regime(data, labels)
    print("Regime day-counts:", {r: s.returns.shape[0] for r, s in subsets.items()})

    # Evolve a specialist strategy for each regime, comparing all eight optimizers.
    for regime, subset in subsets.items():
        train, validation = split_by_fraction(subset, train_fraction=0.7)
        if train.returns.shape[0] <= base_config.LOOKBACK or validation.returns.shape[0] <= base_config.LOOKBACK:
            print(f"\n[{regime}] too few days to train/validate; skipping.")
            continue

        objective = make_objective(train, validation, evo_cfg)
        header = f"\n=== {regime.upper()}  (train={train.returns.shape[0]}, val={validation.returns.shape[0]}) ==="
        print(header)
        print(f"{'algo':<8}{evo_cfg.fitness_metric:>9}" + "".join(f"{n:>10}" for n in FACTOR_NAMES))

        for algorithm in evo_cfg.algorithms:
            result = run_search(objective, replace(evo_cfg, algorithm=algorithm))
            raw = result.best_weights
            total = raw.sum()
            normalized = raw / total if total > 0 else raw
            row = f"{algorithm:<8}{result.best_fitness:>9.3f}"
            row += "".join(f"{w:>10.3f}" for w in normalized)
            print(row)


if __name__ == "__main__":
    main()
