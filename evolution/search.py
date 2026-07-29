"""Evolutionary search drivers — eight nature-inspired optimizers, one interface.

Every optimizer maximizes the same ``objective(x) -> fitness`` that maps a length-5
reward-weight array to a scalar fitness. A small tracker wraps the objective so that,
whichever backend runs, we recover the best-seen weights, the best fitness, and a
best-so-far history (the convergence-curve analogue used in the author's prior work).

Backends (all reputable, documented libraries; heavy imports are local to each adapter
so importing this module needs only the backend actually used):

    cma                         -> ``cma``       (canonical CMA-ES)
    xnes                        -> ``pypop7``    (Exponential Natural Evolution Strategy)
    de, lshade, pso, acor,
    gwo, abc                    -> ``mealpy``    (unified metaheuristics; acor = ACO-R)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from evolution.config import EvolutionConfig

# The reward-weight vector has five entries: (return, E, S, G, risk).
_N_WEIGHTS: int = 5

# mealpy optimizers, keyed by our algorithm name -> (submodule, class name).
_MEALPY: Dict[str, Tuple[str, str]] = {
    "de": ("evolutionary_based.DE", "OriginalDE"),
    "lshade": ("evolutionary_based.SHADE", "L_SHADE"),
    "pso": ("swarm_based.PSO", "OriginalPSO"),
    "acor": ("swarm_based.ACOR", "OriginalACOR"),
    "gwo": ("swarm_based.GWO", "OriginalGWO"),
    "abc": ("swarm_based.ABC", "OriginalABC"),
}


@dataclass
class SearchResult:
    """Outcome of an evolutionary search.

    Attributes:
        best_weights: The best reward-weight vector found, shape ``(5,)``.
        best_fitness: The fitness of ``best_weights`` (maximized objective).
        history: Best-so-far fitness recorded after each candidate evaluation.
    """

    best_weights: np.ndarray
    best_fitness: float
    history: List[float]


class _BestTracker:
    """Wrap a maximization objective to record the best candidate seen.

    Calling the tracker evaluates the wrapped objective, updates the running best, and
    appends the best-so-far fitness to ``history`` — giving a backend-agnostic record.
    """

    def __init__(self, objective: Callable[[np.ndarray], float]) -> None:
        self._objective = objective
        self.best_x: np.ndarray = np.zeros(_N_WEIGHTS)
        self.best_fitness: float = -np.inf
        self.history: List[float] = []

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        fitness = self._objective(x)
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_x = x.copy()
        self.history.append(self.best_fitness)
        return fitness


def _run_cma(tracker: _BestTracker, cfg: EvolutionConfig) -> None:
    """Run CMA-ES (``cma`` library) over the reward-weight box.

    Args:
        tracker: Best-tracking wrapper around the (maximization) objective.
        cfg: Evolutionary configuration.
    """
    import cma

    low, high = cfg.weight_low, cfg.weight_high
    x0 = [0.5 * (low + high)] * _N_WEIGHTS
    sigma0 = cfg.cma_sigma0 * (high - low)
    options = {
        "bounds": [[low] * _N_WEIGHTS, [high] * _N_WEIGHTS],
        "popsize": cfg.population_size,
        "seed": cfg.seed,
        "maxiter": cfg.max_generations,
        "verbose": -9,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, options)
    while not es.stop():
        candidates = es.ask()
        # CMA-ES minimizes -> feed it the negated fitness.
        es.tell(candidates, [-tracker(x) for x in candidates])


def _run_xnes(tracker: _BestTracker, cfg: EvolutionConfig) -> None:
    """Run xNES (Exponential Natural Evolution Strategy, ``pypop7``) over the box.

    Args:
        tracker: Best-tracking wrapper around the (maximization) objective.
        cfg: Evolutionary configuration.
    """
    from pypop7.optimizers.nes.xnes import XNES

    low, high = cfg.weight_low, cfg.weight_high
    problem = {
        "fitness_function": lambda x: -tracker(x),  # pypop7 minimizes
        "ndim_problem": _N_WEIGHTS,
        "lower_boundary": np.full(_N_WEIGHTS, low),
        "upper_boundary": np.full(_N_WEIGHTS, high),
    }
    options = {
        "max_function_evaluations": cfg.population_size * cfg.max_generations,
        "seed_rng": cfg.seed,
        "x": np.full(_N_WEIGHTS, 0.5 * (low + high)),
        "sigma": cfg.cma_sigma0 * (high - low),
        "verbose": False,
    }
    XNES(problem, options).optimize()


def _run_mealpy(algorithm: str, tracker: _BestTracker, cfg: EvolutionConfig) -> None:
    """Run one of the mealpy optimizers (DE, L-SHADE, PSO, ACO-R, GWO, ABC).

    Args:
        algorithm: The mealpy algorithm key (must be in :data:`_MEALPY`).
        tracker: Best-tracking wrapper around the (maximization) objective.
        cfg: Evolutionary configuration.
    """
    import importlib

    from mealpy import FloatVar

    submodule, class_name = _MEALPY[algorithm]
    optimizer_cls = getattr(importlib.import_module(f"mealpy.{submodule}"), class_name)

    problem = {
        # mealpy maximizes directly (minmax="max"), so the tracker is the objective.
        "obj_func": tracker,
        "bounds": FloatVar(
            lb=[cfg.weight_low] * _N_WEIGHTS,
            ub=[cfg.weight_high] * _N_WEIGHTS,
        ),
        "minmax": "max",
        "log_to": None,
    }
    # mealpy requires a population of at least 5; clamp so small configs still run.
    pop_size = max(5, cfg.population_size)
    model = optimizer_cls(epoch=cfg.max_generations, pop_size=pop_size)
    model.solve(problem, seed=cfg.seed)


def run_search(
    objective: Callable[[np.ndarray], float],
    cfg: EvolutionConfig,
) -> SearchResult:
    """Search the reward-weight space with the configured nature-inspired optimizer.

    Args:
        objective: Maps a length-5 reward-weight array to a scalar fitness to maximize
            (typically :func:`evolution.fitness.evaluate_weights` bound to fixed data).
        cfg: Evolutionary configuration, including ``cfg.algorithm``.

    Returns:
        A :class:`SearchResult` with the best weights, their fitness, and the best-so-far
        history.

    Raises:
        ValueError: If ``cfg.algorithm`` is not one of :data:`evolution.config.ALGORITHMS`.
    """
    tracker = _BestTracker(objective)
    algorithm = cfg.algorithm
    if algorithm == "cma":
        _run_cma(tracker, cfg)
    elif algorithm == "xnes":
        _run_xnes(tracker, cfg)
    elif algorithm in _MEALPY:
        _run_mealpy(algorithm, tracker, cfg)
    else:
        raise ValueError(
            f"unknown algorithm {algorithm!r}; expected one of "
            f"cma, xnes, de, lshade, pso, acor, gwo, abc"
        )

    return SearchResult(
        best_weights=tracker.best_x,
        best_fitness=tracker.best_fitness,
        history=tracker.history,
    )
