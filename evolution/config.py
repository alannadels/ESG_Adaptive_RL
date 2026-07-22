"""Configuration for the evolutionary reward-weight search.

Kept separate from :mod:`esg_adaptive_rl.config` (universe, dates, PPO settings) so the
nature-inspired-algorithm hyperparameters live in one obvious place. Defaults are modest
so the loop is runnable on a laptop; scale ``population_size``, ``max_generations``, and
``fitness_timesteps`` up for final results.

Eight nature-inspired optimizers are supported, spanning four families, so their
reward-weight discoveries can be compared head-to-head:

    Evolution Strategies  : ``cma`` (CMA-ES), ``xnes`` (Exponential NES)
    Differential Evolution: ``de``, ``lshade`` (L-SHADE)
    Swarm intelligence    : ``pso`` (Particle Swarm), ``gwo`` (Grey Wolf), ``abc`` (Bee)
    Ant colony (continuous): ``acor`` (ACO-R, the continuous-domain ACO)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# Every supported optimizer key (see :mod:`evolution.search` for the backing libraries).
ALGORITHMS: Tuple[str, ...] = ("cma", "xnes", "de", "lshade", "pso", "acor", "gwo", "abc")


@dataclass
class EvolutionConfig:
    """Hyperparameters for the evolutionary search over reward weights.

    Attributes:
        algorithm: The optimizer to run for a single search (one of :data:`ALGORITHMS`).
        algorithms: The set to compare when running the full head-to-head sweep.
        weight_low: Lower bound for every reward weight (weights are non-negative).
        weight_high: Upper bound for every reward weight. The search box is
            ``[weight_low, weight_high]^5`` over ``(return, E, S, G, risk)``.
        population_size: Candidates per generation.
        max_generations: Generations (CMA-ES iterations / mealpy epochs; for xNES the
            function-evaluation budget is ``population_size * max_generations``).
        cma_sigma0: Initial step size (fraction of box width) for CMA-ES and xNES.
        fitness_timesteps: PPO training budget *per candidate*. Small because the search
            evaluates many candidates.
        fitness_metric: Which :func:`esg_adaptive_rl.metrics.summarize` key is maximized.
            ``"sharpe"`` by default — a purely *financial* objective, so the evolved
            E/S/G weights reflect what genuinely helps risk-adjusted return within the
            values-compliant universe.
        seed: Seed for the PPO inner loop and the optimizer, so candidates and algorithms
            are compared on equal footing.
        split_date: Train/validation boundary. PPO trains on dates before it; the
            evolutionary fitness is scored on the validation window (on/after it). There
            is deliberately no separate generalization phase — each search produces an
            ad-hoc strategy directly.
    """

    algorithm: str = "cma"
    algorithms: Tuple[str, ...] = ALGORITHMS

    # Search space: a non-negative box over (return, E, S, G, risk).
    weight_low: float = 0.0
    weight_high: float = 3.0

    # Evolutionary budget.
    population_size: int = 12
    max_generations: int = 15
    cma_sigma0: float = 0.5

    # PPO fitness-evaluation budget (short during search for tractability).
    fitness_timesteps: int = 20_000
    fitness_metric: str = "sharpe"

    seed: int = 42

    # Train/validation split (PPO trains before; fitness scored on/after).
    split_date: str = "2021-01-01"

    def bounds(self) -> list:
        """Return per-weight ``(low, high)`` bounds for the five reward weights.

        Returns:
            A list of five ``(low, high)`` tuples, as expected by the search drivers.
        """
        return [(self.weight_low, self.weight_high)] * 5
