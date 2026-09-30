"""Ablation: regime-indexed vs global reward schedule (all three detectors).

Tests the paper's central claim — that indexing the evolved reward *by regime* adds
value over a single, regime-agnostic schedule. For each (detector, optimizer):

  1. Evolve ONE **global** reward schedule on that detector's full (un-split) universe.
  2. For each regime, train a PPO allocator on that regime's train window **under the
     global weights** and evaluate on the regime's val window (10 seeds).

The result is each global schedule's per-regime performance, directly comparable to the
regime-indexed schedules already in results/full_regime_metrics.csv (same training data
and eval window per regime; only the reward weights differ). Run across all three
detectors (SMA crossover / EMA crossover / HMM) so the ablation matches the main sweep.

Run from the repository root:

    python ablate_regime_vs_global.py
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from esg_adaptive_rl.data import split_by_fraction
from esg_adaptive_rl.reward import RewardWeights
from evolution.search import run_search
from evolve_weights import make_objective
from build_regime_datasets_ema import reconstruct_universe

# Reuse the main runner's config + evaluation helpers so the comparison is apples-to-apples.
import evolve_regimes_full as R

DETECTORS_TO_RUN = ["crossover", "ema", "hmm"]   # SMA crossover / EMA crossover / HMM
REGIMES = ["bull", "neutral", "bear"]
OUT = "results/ablation_regime_vs_global.jsonl"


def run_cell(args) -> dict:
    """Evolve a global schedule for one (detector, optimizer) and evaluate it per regime."""
    detector, optimizer = args
    # 1. That detector's full un-split period -> evolve one global reward schedule.
    full = reconstruct_universe(R.DETECTORS[detector])
    full_train, full_val = split_by_fraction(full, train_fraction=0.7)
    result = run_search(make_objective(full_train, full_val, R.SEARCH_CFG),
                        replace(R.SEARCH_CFG, algorithm=optimizer))
    gw = np.asarray(result.best_weights, dtype=float)
    total = gw.sum()
    out = {"detector": detector, "optimizer": optimizer,
           "global_weights_normalized": (gw / total if total > 0 else gw).tolist(),
           "global_search_fitness": float(result.best_fitness),
           "per_regime": {}}

    # 2. Apply the global weights per regime: train on regime-train, eval on regime-val.
    for regime in REGIMES:
        prep = R._prepare(detector, regime)
        if prep is None:
            continue
        train, val, bench_arrays, _tickers = prep
        agg, _holdings, n_ok = R._multiseed_eval(RewardWeights(*gw), train, val, bench_arrays)
        out["per_regime"][regime] = {"metrics": agg, "n_eval_ok": n_ok}
    return out


def main() -> None:
    """Run the global-schedule ablation across all detectors x optimizers; write the jsonl."""
    os.makedirs("results", exist_ok=True)
    tasks = [(d, o) for d in DETECTORS_TO_RUN for o in R.SEARCH_CFG.algorithms]
    n_workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"Regime-vs-global ablation: {len(tasks)} cells "
          f"({len(DETECTORS_TO_RUN)} detectors x {len(R.SEARCH_CFG.algorithms)} optimizers), "
          f"{n_workers} workers, {R.N_EVAL_SEEDS} eval seeds.", flush=True)
    with open(OUT, "w") as fh, Pool(processes=n_workers) as pool:
        for rec in pool.imap_unordered(run_cell, tasks):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            sh = {k: v["metrics"].get("sharpe_mean") for k, v in rec["per_regime"].items()}
            print(f"  {rec['detector']:<10}{rec['optimizer']:<7} "
                  f"global_fit={rec['global_search_fitness']:.3f} | global-schedule Sharpe: "
                  + ", ".join(f"{k}={sh[k]:.2f}" for k in REGIMES if k in sh), flush=True)
    print("done ->", os.path.abspath(OUT), flush=True)


if __name__ == "__main__":
    main()
