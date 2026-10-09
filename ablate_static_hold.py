"""Ablation: single static portfolio held across the entire timeline (no regime adaptation).

The other baselines all evaluate per regime window. This one holds ONE unchanging book
across the full continuous history and never adapts to a regime, isolating whether the
regime-detect-and-adapt machinery adds anything over simply holding the screened universe:

  - full_1N     : rebalance to 1/N every day across the whole period (regime-agnostic).
  - full_BH     : buy 1/N once at the start and never rebalance (drift through all regimes).

Reports full-period metrics (annualized return, Sharpe, alpha/beta vs SPY) and, for context,
attributes the same static book's realized returns to each regime's dates. Pure arithmetic
on the reconstructed universe -- no RL, no search. Writes results/ablation_static_hold.csv.

Run from the repository root:

    python ablate_static_hold.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from esg_adaptive_rl import config as base_config
from esg_adaptive_rl.benchmarks import DEFAULT_BENCHMARKS, align_pair, benchmark_returns_for_dates
from esg_adaptive_rl.metrics import benchmark_metrics, summarize
from build_regime_datasets_ema import reconstruct_universe

DETECTORS = {"crossover": "Dataset/regime_datasets",
             "ema": "Dataset/regime_datasets_ema",
             "hmm": "Dataset/regime_datasets_hmm"}
REGIMES = ["bull", "neutral", "bear"]


def _daily_1n(returns: np.ndarray) -> np.ndarray:
    """Daily portfolio returns for a 1/N book rebalanced to equal weight every day."""
    return returns.mean(axis=1)


def _buy_hold(returns: np.ndarray) -> np.ndarray:
    """Daily portfolio returns for 1/N bought once and never rebalanced (weights drift)."""
    n = returns.shape[1]
    w = np.full(n, 1.0 / n)
    out = np.empty(returns.shape[0])
    for t in range(returns.shape[0]):
        out[t] = float(np.dot(w, returns[t]))
        grown = w * (1.0 + returns[t])
        w = grown / grown.sum()
    return out


def _panel(port: np.ndarray, dates) -> dict:
    """Headline + SPY-relative metrics for a daily-return series over given dates."""
    m = summarize({"net_returns": port, "esg_E": [], "esg_S": [], "esg_G": [], "turnover": []},
                  alpha=base_config.CVAR_ALPHA)
    bench = benchmark_returns_for_dates(dates, DEFAULT_BENCHMARKS)
    ps, bs = align_pair(port, bench["SPY"])
    bm = benchmark_metrics(ps, bs)
    return {"annual_return": round(m["annual_return"], 4), "sharpe": round(m["sharpe"], 3),
            "sortino": round(m["sortino"], 3), "max_drawdown": round(m["max_drawdown"], 4),
            "alpha_SPY": round(bm["alpha"], 4), "beta_SPY": round(bm["beta"], 3),
            "days": len(port)}


def main() -> None:
    """Compute the full-period static holds and their per-regime attribution; write CSV."""
    rows = []
    for detector, path in DETECTORS.items():
        full = reconstruct_universe(path)
        dates = full.dates
        series = {"full_1N": _daily_1n(full.returns), "full_BH": _buy_hold(full.returns)}
        # regime membership for attribution: which full-period dates fall in each regime file
        regime_dates = {reg: set(pd.read_csv(os.path.join(path, f"{reg}.csv"),
                                              parse_dates=["date"])["date"])
                        for reg in REGIMES}
        for sname, port in series.items():
            # full-period (the genuinely new number: hold across all regimes, measure once)
            rows.append({"detector": detector, "strategy": sname, "scope": "full_period",
                         **_panel(port, dates)})
            # attribution: same static book, sliced to each regime's dates
            for reg in REGIMES:
                mask = np.array([d in regime_dates[reg] for d in dates])
                if mask.sum() < 5:
                    continue
                rows.append({"detector": detector, "strategy": sname, "scope": reg,
                             **_panel(port[mask], dates[mask])})
    df = pd.DataFrame(rows)
    df.to_csv("results/ablation_static_hold.csv", index=False)
    print("wrote results/ablation_static_hold.csv\n")
    show = df[(df.strategy == "full_1N")][["detector", "scope", "annual_return", "sharpe", "alpha_SPY", "beta_SPY", "days"]]
    print("full_1N (single unchanging 1/N book):")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
