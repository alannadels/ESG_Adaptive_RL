"""Per-regime reward-weight evolution under HMM vs crossover regime labels.

Answers the question the regime work exists to settle: does labeling regimes with
the walk-forward HMM change what the RL learns, versus the 50/200 crossover?

For each labeling x regime it:
  1. loads the cached regime dataset (bull / neutral / bear);
  2. splits it chronologically 70/30 into train / validation;
  3. evolves the reward weighting (return, E, S, G, risk) with CMA-ES, where each
     candidate is scored by training a PPO allocator on `train` and measuring
     validation Sharpe;
  4. re-evaluates the evolved weights against two baselines on the same
     validation window: the project's DEFAULT_WEIGHTS and a return-only agent.

Budget note: this runs CMA-ES only (the project's primary optimizer) with a
reduced population/generation/timestep budget so the full 2 x 3 grid is
tractable. `evolve_regimes.py` remains the full eight-optimizer sweep.

Run from the repository root:

    python evolve_regimes_compare.py            # both labelings, all regimes
    python evolve_regimes_compare.py hmm bear   # one cell
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import replace
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

warnings.filterwarnings("ignore")

from esg_adaptive_rl import config as base_config  # noqa: E402
from esg_adaptive_rl.data import load_regime_dataset, split_by_fraction  # noqa: E402
from esg_adaptive_rl.reward import FACTOR_NAMES, RewardWeights  # noqa: E402
from evolution.config import EvolutionConfig  # noqa: E402
from evolution.fitness import evaluate_weights  # noqa: E402
from evolution.search import run_search  # noqa: E402
from evolve_weights import make_objective  # noqa: E402

REGIMES = ["bull", "neutral", "bear"]
LABELINGS = {
    "hmm": "Dataset/regime_datasets_hmm",     # walk-forward 3-state Gaussian HMM
    "crossover": "Dataset/regime_datasets",   # causal 50/200 SMA crossover
}
OUT = "esg_regime/results/rl_regime_comparison.json"

# Reduced search budget (see module docstring).
CFG = EvolutionConfig(
    algorithm="cma",
    population_size=10,
    max_generations=10,
    fitness_timesteps=10_000,
    fitness_metric="sharpe",
)
# Larger budget for the final head-to-head evaluation of the winning weights.
EVAL_TIMESTEPS = 30_000


def run_cell(args) -> dict:
    """Evolve and evaluate one (labeling, regime) cell."""
    labeling, regime = args
    path = os.path.join(LABELINGS[labeling], f"{regime}.csv")
    if not os.path.exists(path):
        return {"labeling": labeling, "regime": regime, "error": f"missing {path}"}

    data = load_regime_dataset(path)
    train, val = split_by_fraction(data, train_fraction=0.7)
    n_train, n_val = train.returns.shape[0], val.returns.shape[0]
    if n_train <= base_config.LOOKBACK or n_val <= base_config.LOOKBACK:
        return {"labeling": labeling, "regime": regime,
                "error": f"too few days (train={n_train}, val={n_val})"}

    t0 = time.time()
    result = run_search(make_objective(train, val, CFG), CFG)
    raw = np.asarray(result.best_weights, dtype=float)
    total = raw.sum()
    norm = (raw / total) if total > 0 else raw
    evolved = RewardWeights(*raw)

    # Head-to-head on the SAME validation window, at a larger PPO budget.
    baselines = {
        "evolved": evolved,
        "default": base_config.DEFAULT_WEIGHTS,
        "return_only": RewardWeights(w_return=1.0, w_e=0.0, w_s=0.0, w_g=0.0, w_risk=0.0),
    }
    scores = {
        name: float(evaluate_weights(w, train, val, timesteps=EVAL_TIMESTEPS,
                                     seed=CFG.seed, metric=CFG.fitness_metric))
        for name, w in baselines.items()
    }

    return {
        "labeling": labeling, "regime": regime,
        "days_total": int(data.returns.shape[0]), "days_train": int(n_train),
        "days_val": int(n_val),
        "search_fitness": float(result.best_fitness),
        "weights_raw": raw.tolist(),
        "weights_normalized": norm.tolist(),
        "eval_sharpe": scores,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main() -> None:
    if len(sys.argv) == 3:
        cells = [(sys.argv[1], sys.argv[2])]
    else:
        cells = [(lab, reg) for lab in LABELINGS for reg in REGIMES]

    print(f"Running {len(cells)} cells (CMA-ES, pop={CFG.population_size}, "
          f"gens={CFG.max_generations}, {CFG.fitness_timesteps} steps/candidate)...")
    with Pool(processes=min(len(cells), max(1, (os.cpu_count() or 4) - 2))) as pool:
        results = pool.map(run_cell, cells)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=2)

    ok = [r for r in results if "error" not in r]
    print("\n" + "=" * 104)
    print("  EVOLVED REWARD WEIGHTS PER REGIME (normalized), by regime labeling")
    print("=" * 104)
    print(f"{'labeling':<11}{'regime':<9}{'days':>7}{'fitness':>9}"
          + "".join(f"{n:>10}" for n in FACTOR_NAMES))
    for r in sorted(ok, key=lambda x: (x["labeling"], REGIMES.index(x["regime"]))):
        print(f"{r['labeling']:<11}{r['regime']:<9}{r['days_total']:>7}"
              f"{r['search_fitness']:>9.3f}"
              + "".join(f"{w:>10.3f}" for w in r["weights_normalized"]))

    print("\n" + "=" * 104)
    print("  VALIDATION SHARPE — evolved weights vs baselines (same window, "
          f"{EVAL_TIMESTEPS} PPO steps)")
    print("=" * 104)
    print(f"{'labeling':<11}{'regime':<9}{'evolved':>10}{'default':>10}"
          f"{'return_only':>13}{'evolved-default':>17}")
    for r in sorted(ok, key=lambda x: (x["labeling"], REGIMES.index(x["regime"]))):
        s = r["eval_sharpe"]
        print(f"{r['labeling']:<11}{r['regime']:<9}{s['evolved']:>10.3f}"
              f"{s['default']:>10.3f}{s['return_only']:>13.3f}"
              f"{s['evolved'] - s['default']:>+17.3f}")

    for r in results:
        if "error" in r:
            print(f"\nSKIPPED {r['labeling']}/{r['regime']}: {r['error']}")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
