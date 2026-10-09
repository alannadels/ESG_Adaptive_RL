"""Ablation: naive allocation baselines (equal-weight and buy-and-hold).

Computes two non-RL reference strategies on the *same* per-regime validation windows the
RL allocator was evaluated on, so the learned allocator can be contextualized against
trivial strategies:

  - equal-weight : rebalance to 1/N every day (the 1/N benchmark).
  - buy-and-hold : start at 1/N and let the weights drift with returns (no rebalancing).

For each (detector, regime) it reports the self-contained panel (return, Sharpe, Sortino,
CVaR, drawdown) plus beta/alpha vs SPY and the ESG ETFs, matching the RL evaluation. No
RL, no search — pure arithmetic on the cached returns. Writes results/ablation_naive.csv.

Run from the repository root:

    python ablate_naive_baselines.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from esg_adaptive_rl import config as base_config
from esg_adaptive_rl.benchmarks import DEFAULT_BENCHMARKS, align_pair, benchmark_returns_for_dates
from esg_adaptive_rl.data import load_regime_dataset, split_by_fraction
from esg_adaptive_rl.metrics import benchmark_metrics, summarize

DETECTORS = {"crossover": "Dataset/regime_datasets",
             "ema": "Dataset/regime_datasets_ema",
             "hmm": "Dataset/regime_datasets_hmm"}
REGIMES = ["bull", "neutral", "bear"]
LB = base_config.LOOKBACK   # match the RL trajectory, which starts at val.dates[LOOKBACK:]


def _equal_weight(returns: np.ndarray) -> np.ndarray:
    """Daily portfolio returns for a 1/N book rebalanced every day."""
    return returns.mean(axis=1)


def _buy_and_hold(returns: np.ndarray) -> np.ndarray:
    """Daily portfolio returns for a 1/N book held (weights drift with returns)."""
    n = returns.shape[1]
    w = np.full(n, 1.0 / n)
    out = np.empty(returns.shape[0])
    for t in range(returns.shape[0]):
        out[t] = float(np.dot(w, returns[t]))
        grown = w * (1.0 + returns[t])
        w = grown / grown.sum()
    return out


def main() -> None:
    """Compute the two naive baselines per (detector, regime) and write the CSV."""
    rows = []
    for detector, path in DETECTORS.items():
        for regime in REGIMES:
            data = load_regime_dataset(os.path.join(path, f"{regime}.csv"))
            _train, val = split_by_fraction(data, train_fraction=0.7)
            # Match the RL eval window: the trajectory begins once a lookback is available.
            rets = val.returns[LB:]
            dates = val.dates[LB:]
            bench = benchmark_returns_for_dates(dates, DEFAULT_BENCHMARKS)
            for name, port in (("equal_weight", _equal_weight(rets)),
                               ("buy_and_hold", _buy_and_hold(rets))):
                m = summarize({"net_returns": port, "esg_E": [], "esg_S": [], "esg_G": [],
                               "turnover": []}, alpha=base_config.CVAR_ALPHA)
                row = {"detector": detector, "regime": regime, "strategy": name,
                       "annual_return": round(m["annual_return"], 4),
                       "sharpe": round(m["sharpe"], 3), "sortino": round(m["sortino"], 3),
                       "max_drawdown": round(m["max_drawdown"], 4),
                       "cvar_5": round(m["cvar_5"], 4)}
                for b in DEFAULT_BENCHMARKS:
                    ps, bs = align_pair(port, bench[b])
                    bm = benchmark_metrics(ps, bs)
                    row[f"beta_{b}"] = round(bm["beta"], 3)
                    row[f"alpha_{b}"] = round(bm["alpha"], 4)
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("results/ablation_naive.csv", index=False)
    print("wrote results/ablation_naive.csv")
    # quick view: Sharpe by strategy vs regime, crossover
    piv = df[df.detector == "crossover"].pivot_table(index="regime", columns="strategy", values="sharpe")
    print("\ncrossover Sharpe (naive baselines):")
    print(piv.reindex(["bull", "neutral", "bear"]).to_string())


if __name__ == "__main__":
    main()
