"""Paper-grade per-regime evolutionary reward search across all three detectors.

The definitive run behind the paper. For every cell in the grid

    {8 optimizers} x {3 detectors: crossover / ema / hmm} x {3 regimes: bull/neutral/bear}

it:

  1. loads the cached per-regime dataset (offline; no re-download);
  2. splits it chronologically 70/30 into train / validation;
  3. evolves the reward weighting (return, E, S, G, risk) with the given optimizer,
     scored by validation Sharpe of a PPO allocator (single search seed — see below);
  4. re-evaluates the evolved weights across N_EVAL_SEEDS PPO seeds and reports the
     full performance panel as mean +/- std, plus the seed-averaged mean holdings;
  5. computes benchmark-relative metrics (beta / alpha / information ratio / tracking
     error) against SPY and the ESG benchmarks SUSA, DSI, SPYX.

Seeding (workshop scope): the schedule *search* uses a single seed; robustness of the
discovered weight pattern is evidenced by agreement across 8 optimizers and 3 detectors.
The reported *performance* is averaged over 10 evaluation seeds with standard deviations.
A future (conference/journal) extension can average the search over multiple seeds too.

Outputs (written to results/):
    full_regime_weights.csv    - evolved reward weights per cell (normalized)
    full_regime_metrics.csv    - full performance panel per cell (mean & std over seeds)
    full_regime_holdings.csv   - seed-averaged mean allocation to each of the 50 names
    full_regime_baselines.csv  - default & return-only baselines per (detector, regime)
    full_regime_results.jsonl  - raw per-cell records, written incrementally

Run from the repository root:

    python evolve_regimes_full.py                 # full grid (all detectors)
    python evolve_regimes_full.py crossover        # one detector only (contained run)
    python evolve_regimes_full.py hmm bull          # one detector, one regime
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
from esg_adaptive_rl.benchmarks import (  # noqa: E402
    DEFAULT_BENCHMARKS,
    align_pair,
    benchmark_returns_for_dates,
)
from esg_adaptive_rl.data import load_regime_dataset, split_by_fraction  # noqa: E402
from esg_adaptive_rl.metrics import benchmark_metrics, summarize  # noqa: E402
from esg_adaptive_rl.reward import FACTOR_NAMES, RewardWeights  # noqa: E402
from evolution.config import EvolutionConfig  # noqa: E402
from evolution.fitness import evaluate_weights  # noqa: E402
from evolution.search import run_search  # noqa: E402
from evolve_weights import make_objective  # noqa: E402

# ----------------------------------------------------------------------------- config
# Detector name -> directory of cached per-regime datasets (all share one universe).
DETECTORS = {
    "crossover": "Dataset/regime_datasets",
    "ema": "Dataset/regime_datasets_ema",
    "hmm": "Dataset/regime_datasets_hmm",
}
REGIMES = ["bull", "neutral", "bear"]

# Single-seed schedule search at the project's full budget (pop 12 x gen 15 x 20k steps).
SEARCH_CFG = EvolutionConfig()          # defaults; algorithm set per cell below
SEARCH_SEED = SEARCH_CFG.seed           # 42

# 10-seed evaluation of the evolved weights, at a larger PPO budget for convergence.
N_EVAL_SEEDS = 10
EVAL_SEEDS = [SEARCH_SEED + i for i in range(N_EVAL_SEEDS)]   # 42..51
EVAL_TIMESTEPS = 30_000

# Internal baselines re-evaluated on the same windows for the evolved-vs-naive delta.
BASELINES = {
    "default": base_config.DEFAULT_WEIGHTS,
    "return_only": RewardWeights(w_return=1.0, w_e=0.0, w_s=0.0, w_g=0.0, w_risk=0.0),
}

OUT_DIR = "results"


# --------------------------------------------------------------------------- helpers
def _aggregate(dicts: list) -> dict:
    """Reduce a list of per-seed metric dicts to ``{key_mean, key_std}`` entries.

    Args:
        dicts: One metric dict per evaluation seed (all sharing the same keys).

    Returns:
        A flat dict with a ``_mean`` and ``_std`` entry for every metric key.
    """
    out = {}
    if not dicts:
        return out
    for key in dicts[0]:
        vals = np.array([d[key] for d in dicts], dtype=np.float64)
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
    return out


def _eval_one_seed(weights, train, val, bench_arrays, seed):
    """Train under ``weights`` at one seed and return its metric panel and holdings.

    Args:
        weights: The reward weighting to evaluate.
        train: PPO training window (regime subset).
        val: Validation window the metrics are scored on.
        bench_arrays: ``{ticker: aligned_return_array}`` over the trajectory dates.
        seed: PPO/eval seed.

    Returns:
        A ``(metrics, holdings)`` tuple, or ``(None, None)`` if the run failed.
        ``metrics`` includes the self-contained panel plus per-benchmark beta/alpha/
        info-ratio/tracking-error; ``holdings`` is the mean weight vector over the episode.
    """
    fitness, hist = evaluate_weights(
        weights, train, val, timesteps=EVAL_TIMESTEPS, seed=seed,
        metric=SEARCH_CFG.fitness_metric, return_history=True,
    )
    if hist is None:
        return None, None

    metrics = summarize(hist, alpha=base_config.CVAR_ALPHA)
    port = hist["net_returns"]
    for ticker, arr in bench_arrays.items():
        port_sub, bench_sub = align_pair(port, arr)
        bm = benchmark_metrics(port_sub, bench_sub)
        metrics[f"beta_{ticker}"] = bm["beta"]
        metrics[f"alpha_{ticker}"] = bm["alpha"]
        metrics[f"ir_{ticker}"] = bm["info_ratio"]
        metrics[f"te_{ticker}"] = bm["tracking_error"]
        metrics[f"cov_{ticker}"] = bm["cov_days"]

    holdings = hist["weights"].mean(axis=0)  # seed-specific mean allocation (N,)
    return metrics, holdings


def _multiseed_eval(weights, train, val, bench_arrays):
    """Evaluate ``weights`` across all evaluation seeds; aggregate metrics and holdings.

    Args:
        weights: The reward weighting to evaluate.
        train, val: Train / validation windows.
        bench_arrays: Benchmark returns aligned to the trajectory dates.

    Returns:
        A ``(agg_metrics, mean_holdings, n_ok)`` tuple: aggregated ``mean``/``std``
        metrics over the seeds that succeeded, the holdings averaged over those seeds,
        and the number of successful seeds.
    """
    per_seed, holdings = [], []
    for seed in EVAL_SEEDS:
        m, h = _eval_one_seed(weights, train, val, bench_arrays, seed)
        if m is not None:
            per_seed.append(m)
            holdings.append(h)
    agg = _aggregate(per_seed)
    mean_holdings = (np.mean(holdings, axis=0) if holdings else None)
    return agg, mean_holdings, len(per_seed)


def _prepare(detector, regime):
    """Load and split a cell's dataset and build its benchmark return arrays.

    Args:
        detector: Detector name (key of :data:`DETECTORS`).
        regime: Regime name.

    Returns:
        A ``(train, val, bench_arrays, tickers)`` tuple, or ``None`` if the regime has
        too few days to form train/validation windows.
    """
    data = load_regime_dataset(os.path.join(DETECTORS[detector], f"{regime}.csv"))
    train, val = split_by_fraction(data, train_fraction=0.7)
    if (train.returns.shape[0] <= base_config.LOOKBACK
            or val.returns.shape[0] <= base_config.LOOKBACK):
        return None
    # The evaluation trajectory starts once a full lookback window is available, so it
    # maps to val.dates[LOOKBACK:]; align the benchmarks to exactly those days.
    traj_dates = val.dates[base_config.LOOKBACK:]
    bench_arrays = benchmark_returns_for_dates(traj_dates, DEFAULT_BENCHMARKS)
    return train, val, bench_arrays, data.tickers


# ------------------------------------------------------------------------------ cells
def run_cell(args) -> dict:
    """Evolve and multi-seed-evaluate one (detector, regime, optimizer) cell."""
    detector, regime, optimizer = args
    prepared = _prepare(detector, regime)
    if prepared is None:
        return {"detector": detector, "regime": regime, "optimizer": optimizer,
                "error": "too few days"}
    train, val, bench_arrays, tickers = prepared

    t0 = time.time()
    # Single-seed schedule search at full budget.
    objective = make_objective(train, val, SEARCH_CFG)
    result = run_search(objective, replace(SEARCH_CFG, algorithm=optimizer))
    raw = np.asarray(result.best_weights, dtype=float)
    total = raw.sum()
    norm = (raw / total) if total > 0 else raw

    # 10-seed evaluation of the evolved weights.
    agg, holdings, n_ok = _multiseed_eval(RewardWeights(*raw), train, val, bench_arrays)

    return {
        "detector": detector, "regime": regime, "optimizer": optimizer,
        "days_train": int(train.returns.shape[0]), "days_val": int(val.returns.shape[0]),
        "search_fitness": float(result.best_fitness),
        "weights_normalized": norm.tolist(),
        "metrics": agg, "n_eval_ok": n_ok,
        "holdings": (holdings.tolist() if holdings is not None else None),
        "tickers": tickers,
        "elapsed_s": round(time.time() - t0, 1),
    }


def run_baseline(args) -> dict:
    """Multi-seed-evaluate the fixed baselines for one (detector, regime)."""
    detector, regime = args
    prepared = _prepare(detector, regime)
    if prepared is None:
        return {"detector": detector, "regime": regime, "error": "too few days"}
    train, val, bench_arrays, _ = prepared

    out = {"detector": detector, "regime": regime, "baselines": {}}
    for name, weights in BASELINES.items():
        agg, _, n_ok = _multiseed_eval(weights, train, val, bench_arrays)
        out["baselines"][name] = {"metrics": agg, "n_eval_ok": n_ok}
    return out


# ------------------------------------------------------------------------------- main
def _fmt(seconds: float) -> str:
    """Format a duration in seconds as compact ``H:MM:SS`` (or ``M:SS`` under an hour)."""
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _parse_scope():
    """Resolve the optimizer/detector/regime grid from the command line.

    Returns:
        A ``(cells, base_cells, tag)`` tuple: the (detector, regime, optimizer) cells to
        run, the (detector, regime) combinations for the baseline pass, and a filename
        ``tag`` (e.g. ``"_crossover"``) so a scoped run does not overwrite another's
        output files (empty string for the full grid).
    """
    detectors = list(DETECTORS)
    regimes = list(REGIMES)
    parts = []
    if len(sys.argv) >= 2 and sys.argv[1] in DETECTORS:
        detectors = [sys.argv[1]]
        parts.append(sys.argv[1])
    if len(sys.argv) >= 3 and sys.argv[2] in REGIMES:
        regimes = [sys.argv[2]]
        parts.append(sys.argv[2])
    cells = [(d, r, o) for d in detectors for r in regimes for o in SEARCH_CFG.algorithms]
    base_cells = [(d, r) for d in detectors for r in regimes]
    tag = ("_" + "_".join(parts)) if parts else ""
    return cells, base_cells, tag


def main() -> None:
    """Run the full grid (search + multi-seed eval) and write all result artifacts."""
    os.makedirs(OUT_DIR, exist_ok=True)
    cells, base_cells, tag = _parse_scope()
    raw_jsonl = os.path.join(OUT_DIR, f"full_regime_results{tag}.jsonl")
    n_workers = max(1, (os.cpu_count() or 4) - 2)
    total = len(cells)
    print(f"Grid: {total} cells (+{len(base_cells)} baseline evals), "
          f"{n_workers} workers, {N_EVAL_SEEDS} eval seeds, eval={EVAL_TIMESTEPS} steps.",
          flush=True)

    # Resume support: load any cells already completed in a prior (possibly interrupted)
    # run of this scope, skip them, and APPEND new results rather than truncating the file.
    # This means an interruption (e.g. the WSL VM idle-dropping mid-run) never loses the
    # finished cells — a relaunch picks up where it left off. Only successful records count
    # as done; error records are left out of ``done_keys`` so they get retried.
    results, done_keys = [], set()
    if os.path.exists(raw_jsonl):
        with open(raw_jsonl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "error" not in rec:
                    results.append(rec)
                    done_keys.add((rec["detector"], rec["regime"], rec["optimizer"]))
    remaining = [c for c in cells if c not in done_keys]
    n_done = len(done_keys)
    if n_done:
        print(f"Resuming: {n_done}/{total} cells already complete in {raw_jsonl}; "
              f"running the remaining {len(remaining)}.", flush=True)

    # Each completed cell is appended immediately (flush=True streams live through a tee).
    # ``status`` is kept distinct from the filename-scope ``tag`` so per-cell text does not
    # clobber the output-file suffix; the ETA uses only cells completed this session.
    start = time.time()
    done = n_done
    with open(raw_jsonl, "a") as fh, Pool(processes=n_workers) as pool:
        for rec in pool.imap_unordered(run_cell, remaining):
            results.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done += 1
            elapsed = time.time() - start
            fresh = done - n_done  # cells finished this session, for the rate/ETA
            eta = (elapsed / fresh) * (len(remaining) - fresh) if fresh else 0.0
            status = "ERROR" if "error" in rec else f"{rec['search_fitness']:.3f} ({rec['elapsed_s']}s)"
            print(f"[{done:>2}/{total}  {100 * done // total:>3}%  elapsed {_fmt(elapsed)}  "
                  f"eta {_fmt(eta)}]  {rec['detector']}/{rec['regime']}/{rec.get('optimizer')}: "
                  f"{status}", flush=True)

    # Baseline pass (once per detector/regime, not per optimizer) — also progress-reported.
    print(f"\nBaselines: {len(base_cells)} (detector, regime) combos...", flush=True)
    baselines = []
    bstart = time.time()
    for i, b in enumerate(_imap_baselines(base_cells, n_workers), start=1):
        baselines.append(b)
        print(f"[{i:>2}/{len(base_cells)}  elapsed {_fmt(time.time() - bstart)}]  "
              f"baseline {b['detector']}/{b['regime']}", flush=True)

    _write_csvs(results, baselines, tag)
    print(f"\nWrote result CSVs to {os.path.abspath(OUT_DIR)} (tag='{tag or 'full'}')", flush=True)
    print(f"Total wall time: {_fmt(time.time() - start)}", flush=True)


def _imap_baselines(base_cells, n_workers):
    """Yield baseline results as they complete, so the baseline pass can report progress."""
    with Pool(processes=n_workers) as pool:
        for b in pool.imap_unordered(run_baseline, base_cells):
            yield b


def _write_csvs(results, baselines, tag="") -> None:
    """Assemble the weights / metrics / holdings / baselines CSVs from raw records.

    Args:
        results: Per-cell evolutionary records from :func:`run_cell`.
        baselines: Per-(detector, regime) baseline records from :func:`run_baseline`.
        tag: Filename suffix scoping the run (e.g. ``"_crossover"``) so a per-detector
            run does not overwrite the full-grid or another detector's CSVs.
    """
    import pandas as pd

    ok = [r for r in results if "error" not in r]

    # 1. Evolved weights (normalized).
    wrows = []
    for r in ok:
        row = {"detector": r["detector"], "regime": r["regime"], "optimizer": r["optimizer"],
               "days_train": r["days_train"], "days_val": r["days_val"],
               "search_fitness": round(r["search_fitness"], 6),
               "elapsed_s": r.get("elapsed_s")}  # per-cell wall-time (search + 10-seed eval)
        row.update({name: round(w, 6) for name, w in zip(FACTOR_NAMES, r["weights_normalized"])})
        wrows.append(row)
    pd.DataFrame(wrows).to_csv(os.path.join(OUT_DIR, f"full_regime_weights{tag}.csv"), index=False)

    # 2. Full metric panel (mean & std over seeds), one row per cell.
    mrows = []
    for r in ok:
        row = {"detector": r["detector"], "regime": r["regime"], "optimizer": r["optimizer"],
               "n_eval_ok": r["n_eval_ok"]}
        row.update({k: round(v, 6) for k, v in r["metrics"].items()})
        mrows.append(row)
    pd.DataFrame(mrows).to_csv(os.path.join(OUT_DIR, f"full_regime_metrics{tag}.csv"), index=False)

    # 3. Seed-averaged mean holdings, one row per cell, one column per ticker.
    hrows = []
    for r in ok:
        if r["holdings"] is None:
            continue
        row = {"detector": r["detector"], "regime": r["regime"], "optimizer": r["optimizer"]}
        row.update({tk: round(w, 6) for tk, w in zip(r["tickers"], r["holdings"])})
        hrows.append(row)
    pd.DataFrame(hrows).to_csv(os.path.join(OUT_DIR, f"full_regime_holdings{tag}.csv"), index=False)

    # 4. Baselines (default & return-only) per detector/regime.
    brows = []
    for b in baselines:
        if "error" in b:
            continue
        for name, payload in b["baselines"].items():
            row = {"detector": b["detector"], "regime": b["regime"], "baseline": name,
                   "n_eval_ok": payload["n_eval_ok"]}
            row.update({k: round(v, 6) for k, v in payload["metrics"].items()})
            brows.append(row)
    pd.DataFrame(brows).to_csv(os.path.join(OUT_DIR, f"full_regime_baselines{tag}.csv"), index=False)


if __name__ == "__main__":
    main()
