"""The portfolio-allocation environment (custom Gymnasium env).

A single environment step corresponds to one trading day. On each step the agent emits
a vector of raw scores (one per asset); these are mapped to long-only portfolio weights
via a softmax, the portfolio is held for the next day, and the realised return, ESG
profile, transaction cost, and tail-risk proxy feed the multi-factor reward.

Timeline / look-ahead discipline (important):

    - At decision index ``t`` the agent observes information available *up to and
      including day t-1*: trailing return statistics over ``[t-lookback, t-1]`` and the
      ESG scores known as of day ``t-1``.
    - The chosen weights are held over day ``t`` and earn ``returns[t]``.

This ordering ensures the action never depends on the same-day return it is graded on.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from esg_adaptive_rl.data import ESG_FACTORS, MarketData
from esg_adaptive_rl.reward import RewardWeights, compute_reward


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Map raw action scores to long-only weights that sum to one.

    Args:
        logits: Raw per-asset action values.

    Returns:
        Non-negative weights summing to 1 (a point on the simplex).
    """
    # Subtract the max for numerical stability before exponentiating.
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


class PortfolioEnv(gym.Env):
    """A long-only, fully-invested portfolio-allocation environment.

    Observation (per step), a flat float32 vector of length ``6 * N``:

        - per asset: trailing mean return, trailing return volatility, and the
          E, S, G scores known at decision time (``5 * N`` values);
        - the portfolio weights held going into the step (``N`` values).

    Action: a length-``N`` real vector, mapped to portfolio weights by softmax.

    Reward: produced by :func:`esg_adaptive_rl.reward.compute_reward` from the net
    return, the portfolio ESG profile, and a downside-risk proxy.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: MarketData,
        reward_weights: RewardWeights,
        lookback: int = 20,
        transaction_cost_rate: float = 0.0010,
    ) -> None:
        """Initialise the environment.

        Args:
            data: Aligned returns and ESG arrays to trade over.
            reward_weights: The reward trade-off used for every step in this env. (In
                later stages a regime layer will select which weights are active; for
                the v0 single-regime backbone the weights are fixed for the episode.)
            lookback: Window length for the trailing return statistics in the
                observation.
            transaction_cost_rate: Cost charged per unit of turnover (sum of absolute
                weight changes). For example, ``0.0010`` is 10 basis points per unit
                turnover.

        Raises:
            ValueError: If there are not enough days for at least one decision step.
        """
        super().__init__()

        self._returns = data.returns
        self._esg = data.esg
        self._tickers = data.tickers
        self._weights_cfg = reward_weights
        self._lookback = int(lookback)
        self._cost_rate = float(transaction_cost_rate)

        self._n_days, self._n_assets = self._returns.shape
        # Need at least ``lookback`` days of history before the first decision, and at
        # least one further day whose return can be realised.
        if self._n_days <= self._lookback:
            raise ValueError(
                f"Need more than lookback={self._lookback} days; got {self._n_days}."
            )

        # Action: one real score per asset (softmax-normalised inside step()).
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self._n_assets,), dtype=np.float32
        )
        # Observation: 6 * N features (see class docstring). Bounds are generous and
        # finite so they are valid for the deep-RL backend.
        obs_dim = 6 * self._n_assets
        self.observation_space = spaces.Box(
            low=-1e3, high=1e3, shape=(obs_dim,), dtype=np.float32
        )

        # Episode state, initialised properly in reset().
        self._t: int = self._lookback
        self._prev_weights: np.ndarray = np.full(self._n_assets, 1.0 / self._n_assets)
        self._history: Dict[str, List[float]] = {}

    def _get_observation(self) -> np.ndarray:
        """Build the observation for the current decision index ``self._t``.

        Uses only information available up to day ``t-1`` (see the look-ahead note in
        the module docstring).

        Returns:
            A float32 observation vector of length ``6 * N``.
        """
        t = self._t
        # Trailing window of returns over days [t-lookback, t-1].
        window = self._returns[t - self._lookback : t]  # shape (lookback, N)
        mean_return = window.mean(axis=0)
        vol_return = window.std(axis=0)

        # ESG scores known at decision time are those of the previous day, t-1.
        esg_now = np.stack([self._esg[f][t - 1] for f in ESG_FACTORS], axis=1)  # (N, 3)

        # Per-asset block: [mean_return, vol_return, E, S, G] -> shape (N, 5).
        per_asset = np.column_stack([mean_return, vol_return, esg_now])

        # Flatten per-asset features and append the incoming portfolio weights.
        obs = np.concatenate([per_asset.reshape(-1), self._prev_weights])
        return obs.astype(np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset the environment to the start of the data window.

        Args:
            seed: Optional RNG seed (forwarded to the base class).
            options: Unused; present for Gymnasium API compatibility.

        Returns:
            A ``(observation, info)`` tuple.
        """
        super().reset(seed=seed)

        # First decision happens once a full lookback window is available.
        self._t = self._lookback
        # Start from an equal-weight portfolio so the first step's turnover is defined.
        self._prev_weights = np.full(self._n_assets, 1.0 / self._n_assets)
        # Fresh trajectory record for evaluation/metrics.
        self._history = {
            "net_returns": [],
            "gross_returns": [],
            "turnover": [],
            "esg_E": [],
            "esg_S": [],
            "esg_G": [],
        }

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one trading day.

        Args:
            action: Raw per-asset scores; mapped to weights by softmax.

        Returns:
            A ``(observation, reward, terminated, truncated, info)`` tuple following the
            Gymnasium API.
        """
        t = self._t

        # 1. Convert raw scores to long-only weights on the simplex.
        weights = _softmax(np.asarray(action, dtype=np.float64))

        # 2. Realise the portfolio return over day t and charge transaction costs on the
        #    turnover relative to the previously held weights.
        gross_return = float(np.dot(weights, self._returns[t]))
        turnover = float(np.sum(np.abs(weights - self._prev_weights)))
        cost = self._cost_rate * turnover
        net_return = gross_return - cost

        # 3. Portfolio ESG profile, using the scores known at decision time (day t-1).
        esg_e = float(np.dot(weights, self._esg["E"][t - 1]))
        esg_s = float(np.dot(weights, self._esg["S"][t - 1]))
        esg_g = float(np.dot(weights, self._esg["G"][t - 1]))

        # 4. Per-step tail-risk proxy: the magnitude of a loss (0 when the step is up).
        #    Richer measures can replace this without changing the reward interface.
        downside_risk = max(0.0, -net_return)

        # 5. Scalar reward from the configurable multi-factor reward function.
        reward = compute_reward(
            net_return=net_return,
            esg_e=esg_e,
            esg_s=esg_s,
            esg_g=esg_g,
            downside_risk=downside_risk,
            weights=self._weights_cfg,
        )

        # 6. Record the step for downstream evaluation/metrics.
        self._history["net_returns"].append(net_return)
        self._history["gross_returns"].append(gross_return)
        self._history["turnover"].append(turnover)
        self._history["esg_E"].append(esg_e)
        self._history["esg_S"].append(esg_s)
        self._history["esg_G"].append(esg_g)

        # 7. Carry the chosen weights forward and advance the clock.
        self._prev_weights = weights
        self._t += 1

        # The episode ends once there is no further day whose return can be realised.
        terminated = self._t >= self._n_days
        truncated = False

        # When finished there is no valid next observation; return zeros of the right
        # shape (the value is unused once ``terminated`` is True).
        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_observation()

        info = {
            "net_return": net_return,
            "turnover": turnover,
            "esg_E": esg_e,
            "esg_S": esg_s,
            "esg_G": esg_g,
        }
        return obs, float(reward), terminated, truncated, info

    def get_history(self) -> Dict[str, np.ndarray]:
        """Return the recorded trajectory of the most recent episode.

        Returns:
            A dictionary of 1-D arrays (net/gross returns, turnover, per-step ESG
            profile), suitable for the metrics in :mod:`esg_adaptive_rl.metrics`.
        """
        return {key: np.asarray(values, dtype=np.float64) for key, values in self._history.items()}
