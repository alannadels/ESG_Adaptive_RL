"""Fitness evaluation for the evolutionary search.

A single fitness evaluation is: take a candidate reward-weight vector, train a PPO
allocator under it on the training window, roll the trained policy through the validation
window, and return a *financial* score (e.g. Sharpe). The financial score — not the
shaped reward being searched — is what the evolutionary loop maximizes, so the search
cannot inflate its own signal by simply cranking a reward weight up.

A regime mask can restrict the score to the days of a single market regime, which is how
the same machinery produces a *per-regime* schedule: evolve the weights that make the
allocator perform best specifically in calm, choppy, or stressed markets.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from stable_baselines3 import PPO

from esg_adaptive_rl import config as base_config
from esg_adaptive_rl.data import MarketData
from esg_adaptive_rl.env import PortfolioEnv
from esg_adaptive_rl.metrics import summarize
from esg_adaptive_rl.reward import RewardWeights

# Fitness returned when a candidate fails to train/evaluate, so the evolutionary loop
# treats it as strongly undesirable rather than crashing.
_FAILURE_FITNESS: float = -1e6


def _filter_history(history: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Restrict a trajectory history to the steps selected by a boolean mask.

    Args:
        history: Trajectory dict from :meth:`esg_adaptive_rl.env.PortfolioEnv.get_history`.
        mask: Boolean array aligned to the trajectory length; ``True`` keeps the step.

    Returns:
        A new history dict containing only the masked steps.
    """
    return {key: values[mask] for key, values in history.items()}


def evaluate_weights(
    weights: RewardWeights,
    train_data: MarketData,
    eval_data: MarketData,
    timesteps: int,
    seed: int,
    metric: str = "sharpe",
    lookback: int = base_config.LOOKBACK,
    transaction_cost_rate: float = base_config.TRANSACTION_COST_RATE,
    regime_mask: Optional[np.ndarray] = None,
) -> float:
    """Train a PPO allocator under ``weights`` and score it out-of-sample.

    Args:
        weights: The candidate reward weighting to evaluate.
        train_data: Market data the PPO allocator trains on.
        eval_data: Held-out market data the fitness is scored on.
        timesteps: PPO training budget (environment steps).
        seed: RNG seed for PPO and the environment, so candidates are comparable.
        metric: Which :func:`esg_adaptive_rl.metrics.summarize` key to return as the
            fitness (maximized).
        lookback: Env trailing-window length.
        transaction_cost_rate: Env transaction-cost rate.
        regime_mask: Optional boolean array aligned to the evaluation trajectory; if
            given, the fitness is computed only over the selected (single-regime) steps.

    Returns:
        The scalar fitness (higher is better). Returns a large negative value if training
        or evaluation fails, so the search discards the candidate gracefully.
    """
    try:
        # Inner loop: train a PPO allocator under the candidate reward weights.
        train_env = PortfolioEnv(
            data=train_data,
            reward_weights=weights,
            lookback=lookback,
            transaction_cost_rate=transaction_cost_rate,
        )
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            policy_kwargs={"net_arch": base_config.POLICY_NET_ARCH},
            seed=seed,
            verbose=0,
        )
        model.learn(total_timesteps=timesteps)

        # Roll the trained policy deterministically through the evaluation window.
        eval_env = PortfolioEnv(
            data=eval_data,
            reward_weights=weights,
            lookback=lookback,
            transaction_cost_rate=transaction_cost_rate,
        )
        obs, _ = eval_env.reset(seed=seed)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = eval_env.step(action)
            done = terminated or truncated

        history = eval_env.get_history()

        # Optionally restrict the score to a single regime's days.
        if regime_mask is not None:
            mask = np.asarray(regime_mask, dtype=bool)
            if mask.shape[0] != history["net_returns"].shape[0]:
                raise ValueError(
                    "regime_mask length does not match the evaluation trajectory length."
                )
            if not mask.any():
                # No days of this regime in the window -> uninformative candidate.
                return _FAILURE_FITNESS
            history = _filter_history(history, mask)

        metrics = summarize(history, alpha=base_config.CVAR_ALPHA)
        fitness = float(metrics[metric])
        # Guard against non-finite scores (e.g. a degenerate all-zero-return window).
        return fitness if np.isfinite(fitness) else _FAILURE_FITNESS

    except Exception:
        # A bad candidate (e.g. numerically unstable reward) should not kill the search.
        return _FAILURE_FITNESS
