"""Ablation: reward-component breakdown (fixed-reward evals).

Isolates what each term of the multi-factor reward contributes, by evaluating fixed
reward-weight vectors (no evolutionary search) with the same 10-seed protocol as the
main run. Together with the existing baselines (return-only, default) and the evolved
result, this gives the full lattice:

    return alone   (= return_only baseline, already run)
    risk alone     (0, 0, 0, 0, 0.5)        <- new
    ESG alone      (0, 0.1, 0.1, 0.1, 0)     <- new
    return + risk  (1, 0, 0, 0, 0.5)         <- new
    all / default  (1, 0.1, 0.1, 0.1, 0.5)   (= default baseline, already run)
    evolved        (search-optimized)        (main run)

The three new variants keep each component at its *default* weight and zero the rest
("ablate the default"). Run across all three detectors. Writes the jsonl.

Run from the repository root:

    python ablate_reward_breakdown.py
"""

from __future__ import annotations

import json
import os
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

from esg_adaptive_rl.reward import RewardWeights

import evolve_regimes_full as R

# New fixed-reward variants (ablate-the-default: default weight for the kept term, 0 else).
VARIANTS = {
    "risk_alone": RewardWeights(w_return=0.0, w_e=0.0, w_s=0.0, w_g=0.0, w_risk=0.5),
    "return_risk": RewardWeights(w_return=1.0, w_e=0.0, w_s=0.0, w_g=0.0, w_risk=0.5),
    "esg_alone": RewardWeights(w_return=0.0, w_e=0.1, w_s=0.1, w_g=0.1, w_risk=0.0),
}
DETECTORS = ["crossover", "ema", "hmm"]
REGIMES = ["bull", "neutral", "bear"]
OUT = "results/ablation_reward_breakdown.jsonl"


def run_cell(args) -> dict:
    """Evaluate one fixed reward variant on one (detector, regime), 10 seeds."""
    detector, regime, vname = args
    prep = R._prepare(detector, regime)
    if prep is None:
        return {"detector": detector, "regime": regime, "variant": vname, "error": "too few days"}
    train, val, bench_arrays, _tickers = prep
    agg, _holdings, n_ok = R._multiseed_eval(VARIANTS[vname], train, val, bench_arrays)
    return {"detector": detector, "regime": regime, "variant": vname,
            "metrics": agg, "n_eval_ok": n_ok}


def main() -> None:
    """Evaluate all new reward variants across detectors x regimes; write the jsonl."""
    os.makedirs("results", exist_ok=True)
    tasks = [(d, r, v) for d in DETECTORS for r in REGIMES for v in VARIANTS]
    n_workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"Reward breakdown: {len(tasks)} cells "
          f"({len(DETECTORS)} detectors x {len(REGIMES)} regimes x {len(VARIANTS)} variants), "
          f"{n_workers} workers, {R.N_EVAL_SEEDS} eval seeds.", flush=True)
    with open(OUT, "w") as fh, Pool(processes=n_workers) as pool:
        for rec in pool.imap_unordered(run_cell, tasks):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            s = "ERR" if "error" in rec else f"{rec['metrics']['sharpe_mean']:.3f}"
            print(f"  {rec['detector']:<10}{rec['regime']:<9}{rec['variant']:<12} sharpe={s}", flush=True)
    print("done ->", os.path.abspath(OUT), flush=True)


if __name__ == "__main__":
    main()
