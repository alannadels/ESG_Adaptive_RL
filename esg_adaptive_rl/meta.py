"""Regime-switching meta-controller over per-regime specialist policies.

The meta-controller is the layer that closes the loop between the regime
detector and the RL allocators: three regime-specialist PPO policies (one per
``bull`` / ``neutral`` / ``bear``), each trained under its regime's reward
weighting, are held simultaneously and switched between in live sequence by a
causal regime label.

Timing / look-ahead discipline (must match :class:`esg_adaptive_rl.env.PortfolioEnv`):

    - At decision index ``t`` the agent observes information up to and including
      day ``t-1`` (trailing return statistics and E/S/G scores as of ``t-1``).
    - The switching signal is the regime label of day ``t-1`` — the most recent
      complete day — so the choice of specialist never uses same-day information.
    - The chosen weights are held over day ``t`` and earn ``returns[t]``.

This is exactly the as-of convention used by ``split_by_regime`` (for building
the per-regime training subsets) and by the overlay backtests in ``esg_regime``
(signals lagged one day), so a specialist trained on one regime's cached subset
and a switch evaluated on the full timeline are mutually consistent and agree
with the published regime results.

Known approximation (documented honestly): a specialist is trained only on its
own regime's days (non-contiguous calendar stretches), while the meta-controller
deploys it on the same regime's days embedded in the continuous timeline. The
observation's trailing window therefore sometimes spans days of neighbouring
regimes during deployment. This is inherent to the specialist design and is
reported as such in ``evolve_meta.py``.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
from stable_baselines3 import PPO

from esg_adaptive_rl import config as base_config
from esg_adaptive_rl.data import MarketData
from esg_adaptive_rl.env import PortfolioEnv
from esg_adaptive_rl.metrics import TRADING_DAYS, summarize
from esg_adaptive_rl.reward import RewardWeights


class MetaController:
    """Switch between per-regime specialist PPO policies on a causal label.

    The controller owns:

        - ``policies`` — one trained :class:`stable_baselines3.PPO` allocator per
          regime (keys are the ``bull`` / ``neutral`` / ``bear`` labels);
        - ``labels`` — the point-in-time regime path (indexed by date) that
          drives the switching. Labels at day ``t-1`` select the specialist
          whose action is held over day ``t``;
        - ``default_policy`` — the fallback allocator used for any step whose
          label has no specialist (e.g. a regime whose training subset was too
          small). Typically the single full-window policy.

    :meth:`evaluate` rolls the switched strategy deterministically through a
    held-out :class:`~esg_adaptive_rl.data.MarketData` window and returns the
    usual :func:`~esg_adaptive_rl.metrics.summarize` metrics plus switching
    statistics (number of switches, switches per year, per-regime exposure).

    Args:
        policies: Mapping ``{regime label: trained PPO}`` for the available
            specialists.
        labels: Regime label series indexed by date (from
            :func:`esg_adaptive_rl.regimes.label_regimes` or the cached
            ``regime_labels.csv`` files). Aligned as-of onto the evaluation
            dates; any gap is filled with ``"neutral"``.
        default_policy: Fallback PPO used when the day's label has no
            specialist in ``policies`` (may be ``None`` only if ``policies``
            covers every label that occurs).
        lookback: Env trailing-window length (must match the specialists'
            training configuration).
        transaction_cost_rate: Cost charged per unit of turnover, applied on
            every switch exactly as in the underlying env.
    """

    def __init__(
        self,
        policies: Dict[str, PPO],
        labels: pd.Series,
        default_policy: Optional[PPO] = None,
        lookback: int = base_config.LOOKBACK,
        transaction_cost_rate: float = base_config.TRANSACTION_COST_RATE,
    ) -> None:
        if not policies and default_policy is None:
            raise ValueError("MetaController needs at least one policy.")
        self.policies = policies
        self.default_policy = default_policy
        self.labels = labels.sort_index()
        self.lookback = int(lookback)
        self.transaction_cost_rate = float(transaction_cost_rate)

    def evaluate(
        self,
        data: MarketData,
        seed: int = base_config.SEED,
    ) -> dict:
        """Roll the switched strategy through ``data`` and summarise it.

        Args:
            data: Held-out :class:`~esg_adaptive_rl.data.MarketData` to trade.
            seed: Environment RNG seed (deterministic policy roll).

        Returns:
            A dict with the :func:`~esg_adaptive_rl.metrics.summarize` metrics
            plus ``days``, ``steps``, ``switches``, ``switches_per_year``, and
            ``regime_exposure`` (per-label percentage of steps).

        Raises:
            ValueError: If no policy exists for a step's label and no default
                policy was provided.
        """
        aligned = self.labels.reindex(
            pd.to_datetime(pd.Index(data.dates)), method="ffill"
        )
        labels_arr = aligned.fillna("neutral").to_numpy()

        env = PortfolioEnv(
            data=data,
            reward_weights=RewardWeights(),
            lookback=self.lookback,
            transaction_cost_rate=self.transaction_cost_rate,
        )
        obs, _ = env.reset(seed=seed)

        n_days = data.returns.shape[0]
        steps = n_days - self.lookback  # decisions taken (env starts at lookback)
        switches = 0
        uses: Dict[str, int] = {}
        previous_label: Optional[str] = None

        t = self.lookback
        while t < n_days:
            # The switch signal is the most recent complete day's label.
            label = str(labels_arr[t - 1])
            if previous_label is not None and label != previous_label:
                switches += 1
            previous_label = label
            uses[label] = uses.get(label, 0) + 1

            policy = self.policies.get(label) or self.default_policy
            if policy is None:
                raise ValueError(
                    f"no specialist for regime {label!r} and no default policy."
                )
            action, _ = policy.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(action)
            t += 1
            if terminated or truncated:
                break

        metrics = summarize(env.get_history(), alpha=base_config.CVAR_ALPHA)
        metrics["days"] = n_days
        metrics["steps"] = steps
        metrics["switches"] = switches
        years = n_days / TRADING_DAYS
        metrics["switches_per_year"] = switches / years if years > 0 else 0.0
        metrics["regime_exposure"] = {
            regime: round(100.0 * count / steps, 1)
            for regime, count in sorted(uses.items())
        }
        return metrics
