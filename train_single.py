"""Entry point: train and evaluate a single fixed-weight ESG allocator.

This is the v0 "vertical slice" of the project. It runs the entire pipeline end-to-end
for the simplest case — one fixed reward weighting, no market regimes and no evolutionary
search yet — so the data, environment, reward, agent, and metrics can all be exercised
and validated together.

Pipeline:

    1. Download prices and build the (placeholder) ESG table.
    2. Split chronologically into train and test periods.
    3. Train a PPO allocator on the train period under the configured fixed weights.
    4. Roll the trained policy deterministically through the test period.
    5. Print headline performance and ESG metrics.

Run from the repository root:

    python train_single.py
"""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO

from esg_adaptive_rl import config
from esg_adaptive_rl.data import load_market_data, split_by_date
from esg_adaptive_rl.env import PortfolioEnv
from esg_adaptive_rl.metrics import summarize


def build_env(data, weights) -> PortfolioEnv:
    """Construct a :class:`PortfolioEnv` from a data bundle and reward weights.

    Args:
        data: A :class:`esg_adaptive_rl.data.MarketData` bundle.
        weights: The :class:`esg_adaptive_rl.reward.RewardWeights` for the episode.

    Returns:
        A configured environment ready for training or evaluation.
    """
    return PortfolioEnv(
        data=data,
        reward_weights=weights,
        lookback=config.LOOKBACK,
        transaction_cost_rate=config.TRANSACTION_COST_RATE,
    )


def evaluate(model: PPO, env: PortfolioEnv) -> dict:
    """Roll a trained policy deterministically through one episode and summarise it.

    Args:
        model: A trained PPO model.
        env: The (test) environment to evaluate on.

    Returns:
        A dictionary of metrics from :func:`esg_adaptive_rl.metrics.summarize`.
    """
    obs, _ = env.reset(seed=config.SEED)
    done = False
    while not done:
        # Deterministic action so the evaluation is reproducible and reflects the
        # learned policy rather than exploration noise.
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, _info = env.step(action)
        done = terminated or truncated
    return summarize(env.get_history(), alpha=config.CVAR_ALPHA)


def main() -> None:
    """Run the full single-regime train-and-evaluate pipeline and print results."""
    # Reproducibility for the NumPy-side randomness (price/ESG construction is already
    # deterministic; this covers any incidental sampling).
    np.random.seed(config.SEED)

    # 1-2. Load data and split chronologically into train and test periods.
    print("Downloading market data and building the placeholder ESG table...")
    data = load_market_data(config.UNIVERSE, start=config.START_DATE, end=config.END_DATE)
    train_data, test_data = split_by_date(data, split_date=config.SPLIT_DATE)
    print(
        f"Loaded {len(data.dates)} trading days for {len(data.tickers)} assets "
        f"(train: {len(train_data.dates)}, test: {len(test_data.dates)})."
    )

    # 3. Train a PPO allocator on the train period under the fixed reward weights.
    train_env = build_env(train_data, config.DEFAULT_WEIGHTS)
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        policy_kwargs={"net_arch": config.POLICY_NET_ARCH},
        seed=config.SEED,
        verbose=1,
    )
    print(f"Training PPO for {config.TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS)

    # 4-5. Evaluate on the held-out test period and report metrics.
    test_env = build_env(test_data, config.DEFAULT_WEIGHTS)
    metrics = evaluate(model, test_env)

    print("\nHeld-out test performance (fixed-weight allocator):")
    for name, value in metrics.items():
        print(f"  {name:<18}: {value: .4f}")


if __name__ == "__main__":
    main()
