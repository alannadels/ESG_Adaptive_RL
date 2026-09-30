"""Ablation add-on: uniform reward vector (all five components weighted equally).

Runs the single fixed-reward variant equal_all = (0.2, 0.2, 0.2, 0.2, 0.2) across every
(detector, regime) with the same 10-seed evaluation protocol as the main run. This is the
explicit null for the evolved-reward finding: a reward that expresses no preference among
return, the three ESG pillars, or risk. Writes results/ablation_equal_reward.jsonl.

Run from the repository root:

    python ablate_equal_reward.py
"""

from __future__ import annotations

import json
import os
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

from esg_adaptive_rl.reward import RewardWeights

import evolve_regimes_full as R

WEIGHTS = RewardWeights(0.2, 0.2, 0.2, 0.2, 0.2)
DETECTORS = ["crossover", "ema", "hmm"]
REGIMES = ["bull", "neutral", "bear"]
OUT = "results/ablation_equal_reward.jsonl"


def run_cell(args) -> dict:
    """Evaluate the uniform reward on one (detector, regime), 10 seeds."""
    detector, regime = args
    prep = R._prepare(detector, regime)
    if prep is None:
        return {"detector": detector, "regime": regime, "variant": "equal_all", "error": "too few days"}
    train, val, bench_arrays, _tickers = prep
    agg, _holdings, n_ok = R._multiseed_eval(WEIGHTS, train, val, bench_arrays)
    return {"detector": detector, "regime": regime, "variant": "equal_all",
            "metrics": agg, "n_eval_ok": n_ok}


def main() -> None:
    """Run the uniform-reward null across detectors x regimes; write the jsonl."""
    os.makedirs("results", exist_ok=True)
    tasks = [(d, r) for d in DETECTORS for r in REGIMES]
    n_workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"Equal-reward null: {len(tasks)} cells, {n_workers} workers, {R.N_EVAL_SEEDS} seeds.", flush=True)
    with open(OUT, "w") as fh, Pool(processes=n_workers) as pool:
        for rec in pool.imap_unordered(run_cell, tasks):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            s = "ERR" if "error" in rec else f"{rec['metrics']['sharpe_mean']:.3f}"
            print(f"  {rec['detector']:<10}{rec['regime']:<9} sharpe={s}", flush=True)
    print("done ->", os.path.abspath(OUT), flush=True)


if __name__ == "__main__":
    main()
