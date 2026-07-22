"""Entry point: evolve reward-weight strategies with nature-inspired search.

The outer-loop counterpart to ``train_single.py``. Instead of training one allocator
under a hand-picked reward weighting, it *searches* for the weighting that yields the
best out-of-sample financial performance — running each of the eight nature-inspired
optimizers and reporting the ad-hoc strategy (evolved weights) that each discovers.

Single search per optimizer (no separate generalization phase): each search produces the
strategy directly. Per regime, this whole sweep is re-run with a ``regime_mask`` (from
:mod:`esg_regime`) so the fitness is scored on one regime's days — yielding the per-regime
schedules that are the project's headline finding.

Run from the repository root:

    python evolve_weights.py
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np

from esg_adaptive_rl import config as base_config
from esg_adaptive_rl.data import MarketData, load_market_data, split_by_date
from esg_adaptive_rl.reward import FACTOR_NAMES, RewardWeights
from evolution.config import EvolutionConfig
from evolution.fitness import evaluate_weights
from evolution.search import run_search


def make_objective(
    train: MarketData,
    validation: MarketData,
    cfg: EvolutionConfig,
) -> Callable[[np.ndarray], float]:
    """Build the fitness objective for a search over reward weights.

    Args:
        train: Market data the PPO allocator trains on.
        validation: Held-out data the fitness is scored on.
        cfg: Evolutionary configuration (fitness budget, metric, seed).

    Returns:
        A function mapping a length-5 reward-weight array to its validation fitness.
    """
    def objective(weight_vector: np.ndarray) -> float:
        return evaluate_weights(
            weights=RewardWeights.from_array(weight_vector),
            train_data=train,
            eval_data=validation,
            timesteps=cfg.fitness_timesteps,
            seed=cfg.seed,
            metric=cfg.fitness_metric,
        )

    return objective


def main() -> None:
    """Run the nature-inspired search sweep over all optimizers and report each result."""
    cfg = EvolutionConfig()
    np.random.seed(cfg.seed)

    # Load data and split into a PPO-training window and a fitness-scoring window.
    print("Loading market data and building the ESG table...")
    data = load_market_data(base_config.UNIVERSE, start=base_config.START_DATE, end=base_config.END_DATE)
    train, validation = split_by_date(data, split_date=cfg.split_date)
    print(f"train={len(train.dates)} days, validation={len(validation.dates)} days.\n")

    objective = make_objective(train, validation, cfg)

    # Run each optimizer and collect the ad-hoc strategy (evolved weights) it discovers.
    results = {}
    for algorithm in cfg.algorithms:
        print(f"Searching with {algorithm.upper()} "
              f"(pop={cfg.population_size}, gens={cfg.max_generations})...")
        result = run_search(objective, replace(cfg, algorithm=algorithm))
        results[algorithm] = result

    # Report: for each optimizer, its validation fitness and the normalized evolved
    # weights (summing to one) for interpretability, mirroring the reporting style of the
    # author's developmental-schedule work.
    header = f"\n{'algo':<8}{cfg.fitness_metric:>9}" + "".join(f"{n:>10}" for n in FACTOR_NAMES)
    print(header)
    print("-" * len(header))
    for algorithm, result in results.items():
        raw = result.best_weights
        total = raw.sum()
        normalized = raw / total if total > 0 else raw
        row = f"{algorithm:<8}{result.best_fitness:>9.3f}"
        row += "".join(f"{w:>10.3f}" for w in normalized)
        print(row)


if __name__ == "__main__":
    main()
