"""Entry point: run ONE (regime, optimizer) evolutionary search as a standalone job.

The parallel counterpart to ``evolve_regimes.py``. The full sweep is 3 regimes x 8
optimizers = 24 independent searches, each itself a sequence of PPO train-and-evaluate
fitness evaluations. This worker runs exactly one such search so the sweep can be fanned
out across CPU cores (the fitness evaluations are CPU-bound: tiny MLP policies gain
nothing from a GPU). Results are written as one JSON file per job, so partial sweeps
resume cleanly and the headline table can be assembled offline with
``assemble_regime_sweep.py``.

The dataset build (yfinance download + regime labeling + per-regime split) mirrors
``evolve_regimes.py`` but is cached to a pickle so the 24 workers do not each
re-download. Two universes are supported via ``--universe``:

    - ``snapshot`` (default): the point-in-time snapshot universe built from
      ``Dataset/universe_snapshots.csv`` — the superset of every yearly snapshot with a
      per-day tradability mask the environment enforces (survivorship-free);
    - ``legacy``: the fixed 50-name ``config.UNIVERSE`` (the original pipeline).

Build the cache once up front:

    python evolve_regimes_worker.py --prepare-only

Then run one worker (normally driven by ``run_regime_sweep.sh``):

    python evolve_regimes_worker.py --regime bear --algorithm cma
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import asdict, replace

# Pin BLAS threads before numpy imports, so each of the N parallel workers uses one
# core instead of every worker spawning N intra-op threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from esg_adaptive_rl import config as base_config
from esg_adaptive_rl.data import (
    MarketData,
    load_index_close,
    load_market_data,
    load_snapshot_market_data,
    split_by_fraction,
)
from esg_adaptive_rl.regimes import RegimeConfig, label_regimes, split_by_regime
from esg_adaptive_rl.reward import FACTOR_NAMES, RewardWeights
from evolution.config import ALGORITHMS, EvolutionConfig
from evolution.search import run_search
from evolve_weights import make_objective

REGIME_INDEX = "SPY"
ESG_PATH = "Dataset/ESG_2000-26-ESGC.csv"
ESG_ANCHOR_YEAR = 2025
SNAPSHOT_UNIVERSE_PATH = "Dataset/universe_snapshots.csv"
DATA_CACHE_PATH = "Dataset/cache/regime_splits_{universe}.pkl"
DEFAULT_OUT_DIR = "results/regime_sweep"
REGIMES = ("bull", "neutral", "bear")


def _limit_torch_threads() -> None:
    """Pin each worker to a single torch thread.

    The sweep parallelizes across processes; without this, every worker's torch would
    spawn 16 intra-op threads and the 16 workers would oversubscribe the node. The
    environment variables are set at module import (before numpy); this catches torch's
    own thread pools.
    """
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # already initialized in this process
    except ImportError:
        pass


def build_regime_splits(universe: str = "legacy") -> dict:
    """Build the per-regime dataset splits.

    Args:
        universe: ``"legacy"`` for the fixed 50-name list in
            :mod:`esg_adaptive_rl.config` (the original pipeline), or ``"snapshot"``
            for the point-in-time snapshot universe (``Dataset/universe_snapshots.csv``):
            the superset of every yearly snapshot with a per-day tradability mask the
            environment enforces — the survivorship-free pipeline.

    Returns:
        A mapping ``{"bull"|"neutral"|"bear": MarketData}``.

    Raises:
        ValueError: For an unknown universe key.
    """
    if universe == "snapshot":
        print(f"Loading point-in-time snapshot universe from {SNAPSHOT_UNIVERSE_PATH} "
              f"(prices + real ESG + tradability mask)...")
        data = load_snapshot_market_data(
            snapshot_path=SNAPSHOT_UNIVERSE_PATH,
            start=base_config.START_DATE,
            end=base_config.END_DATE,
            esg_path=ESG_PATH,
            esg_anchor_year=ESG_ANCHOR_YEAR,
        )
        print(f"Snapshot superset: {len(data.tickers)} ever-selected tickers; "
              f"tradable name-days: {int(data.tradable.sum())} of "
              f"{data.tradable.size}.")
        index_prices = load_index_close(REGIME_INDEX, base_config.START_DATE, base_config.END_DATE)
    elif universe == "legacy":
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
    else:
        raise ValueError(f"unknown universe {universe!r}; expected 'legacy' or 'snapshot'")

    # Causal 50/200 SMA crossover, matching the committed Dataset/regime_datasets
    # labels (the walk-forward HMM default is benchmarked but pinned off here).
    regime_cfg = RegimeConfig(ticker=REGIME_INDEX, detector="crossover")
    labels = label_regimes(index_prices, regime_cfg)
    subsets = split_by_regime(data, labels)
    print("Regime day-counts:", {r: s.returns.shape[0] for r, s in subsets.items()})
    return {regime: subsets[regime] for regime in REGIMES}


def load_or_build_splits(cache_path: str, universe: str = "legacy") -> dict:
    """Load the cached regime splits, building and caching them if absent.

    Args:
        cache_path: Pickle path for the ``{regime: MarketData}`` mapping.
        universe: Which universe to build when the cache is absent (see
            :func:`build_regime_splits`).

    Returns:
        The per-regime dataset mapping.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached regime splits from {cache_path} ...")
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)

    splits = build_regime_splits(universe)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    temp_path = f"{cache_path}.{os.getpid()}.tmp"
    with open(temp_path, "wb") as handle:
        pickle.dump(splits, handle)
    os.replace(temp_path, cache_path)
    print(f"Cached regime splits to {cache_path}")
    return splits


def run_one_search(
    regime: str,
    algorithm: str,
    evo_cfg: EvolutionConfig,
    cache_path: str,
    universe: str = "legacy",
) -> dict:
    """Run one (regime, optimizer) search and return its result record.

    Args:
        regime: Which regime specialist to evolve.
        algorithm: Optimizer key from :data:`evolution.config.ALGORITHMS`.
        evo_cfg: Evolutionary hyperparameters (algorithm field is overridden).
        cache_path: Pickle path for the cached regime splits.
        universe: Which universe the splits were built with.

    Returns:
        A JSON-serializable record with the best fitness, raw and normalized weights,
        the search history, and the run's configuration.
    """
    splits = load_or_build_splits(cache_path, universe)
    subset = splits[regime]
    train, validation = split_by_fraction(subset, train_fraction=0.7)
    if train.returns.shape[0] <= base_config.LOOKBACK or validation.returns.shape[0] <= base_config.LOOKBACK:
        raise ValueError(f"[{regime}] too few days to train/validate; cannot search.")

    cfg = replace(evo_cfg, algorithm=algorithm)
    objective = make_objective(train, validation, cfg)
    started = time.perf_counter()
    result = run_search(objective, cfg)
    elapsed = time.perf_counter() - started

    raw = np.asarray(result.best_weights, dtype=np.float64)
    total = raw.sum()
    normalized = raw / total if total > 0 else raw
    return {
        "regime": regime,
        "algorithm": algorithm,
        "best_fitness": float(result.best_fitness),
        "best_weights_raw": raw.tolist(),
        "best_weights_normalized": normalized.tolist(),
        "n_evaluations": len(result.history),
        "history": [float(x) for x in result.history],
        "train_days": int(train.returns.shape[0]),
        "val_days": int(validation.returns.shape[0]),
        "elapsed_seconds": elapsed,
        "config": asdict(cfg),
        "factor_names": list(FACTOR_NAMES),
    }


def main() -> None:
    """Parse the CLI, optionally prepare the data cache, and run one search."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--regime", choices=REGIMES, help="Regime specialist to evolve.")
    parser.add_argument("--algorithm", choices=ALGORITHMS, help="Optimizer to run.")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Build/cache the regime splits, then exit (no search).")
    parser.add_argument("--population-size", type=int, default=EvolutionConfig.population_size)
    parser.add_argument("--max-generations", type=int, default=EvolutionConfig.max_generations)
    parser.add_argument("--fitness-timesteps", type=int, default=EvolutionConfig.fitness_timesteps)
    parser.add_argument("--fitness-metric", default=EvolutionConfig.fitness_metric)
    parser.add_argument("--seed", type=int, default=EvolutionConfig.seed)
    parser.add_argument("--universe", choices=("legacy", "snapshot"), default="snapshot",
                        help="legacy = fixed 50-name config.UNIVERSE; snapshot = "
                             "point-in-time snapshot universe with a tradability mask "
                             "(survivorship-free).")
    parser.add_argument("--data-cache", default=None,
                        help="Pickle cache path; defaults to a per-universe path.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    _limit_torch_threads()

    if args.data_cache is None:
        args.data_cache = DATA_CACHE_PATH.format(universe=args.universe)

    if args.prepare_only:
        load_or_build_splits(args.data_cache, args.universe)
        return

    if args.regime is None or args.algorithm is None:
        parser.error("--regime and --algorithm are required unless --prepare-only is given.")

    evo_cfg = EvolutionConfig(
        population_size=args.population_size,
        max_generations=args.max_generations,
        fitness_timesteps=args.fitness_timesteps,
        fitness_metric=args.fitness_metric,
        seed=args.seed,
    )

    print(f"[{args.regime}/{args.algorithm}|{args.universe}] "
          f"pop={evo_cfg.population_size} gens={evo_cfg.max_generations} "
          f"steps={evo_cfg.fitness_timesteps} seed={evo_cfg.seed}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir,
        f"{args.universe}_{args.regime}_{args.algorithm}_seed{evo_cfg.seed}.json",
    )
    if os.path.exists(out_path):
        print(f"Result already exists: {out_path} (delete it to re-run this search).")
        return

    record = run_one_search(args.regime, args.algorithm, evo_cfg, args.data_cache,
                            args.universe)
    record["universe"] = args.universe
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)

    row = f"{record['algorithm']:<8}{record['best_fitness']:>9.3f}"
    row += "".join(f"{w:>10.3f}" for w in record["best_weights_normalized"])
    print(f"[{args.regime}] done in {record['elapsed_seconds']:.0f}s -> {out_path}")
    print(f"{'algo':<8}{evo_cfg.fitness_metric:>9}"
          + "".join(f"{n:>10}" for n in FACTOR_NAMES))
    print(row)


if __name__ == "__main__":
    main()
