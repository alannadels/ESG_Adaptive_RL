"""evolution — nature-inspired search over the reward-weight vector.

The nature-inspired (evolutionary) outer loop of the project. It searches the
multi-factor reward weighting ``(return, E, S, G, risk)`` that, when a PPO allocator is
trained under it, yields the best *financial* out-of-sample performance. This is the
piece that turns the fixed-weight backbone into an *evolved-reward* allocator, and — run
separately per market regime — produces the per-regime reward schedules that are the
project's headline finding.

Design (mirrors the author's developmental-reward-schedule framework):

    - Outer loop  : an evolutionary algorithm (CMA-ES or Differential Evolution)
                    proposes candidate reward-weight vectors.
    - Inner loop  : a PPO allocator is trained under each candidate.
    - Fitness     : the *true* financial objective (e.g. out-of-sample Sharpe) — NOT the
                    shaped reward being searched, so the search cannot game its own signal.
    - Two phases  : (1) search for the best weights on a fixed seed; (2) a generalization
                    check that retrains under those weights across several seeds.

Submodules:
    - ``config``   : configuration for the evolutionary search
    - ``fitness``  : train-a-PPO-and-score-it fitness evaluation
    - ``search``   : the CMA-ES and DE search drivers
"""

from evolution.config import ALGORITHMS, EvolutionConfig  # noqa: F401
from evolution.fitness import evaluate_weights  # noqa: F401
from evolution.search import run_search  # noqa: F401

__version__ = "0.0.1"
