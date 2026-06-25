"""The configurable multi-factor reward function.

This is the central seam of the whole project. The agent's reward is a weighted
combination of the portfolio's financial return, its Environmental / Social /
Governance (E/S/G) profile, and a tail-risk penalty. The weight vector ``w`` is passed
in explicitly rather than hard-coded, because every later stage of the project plugs in
exactly here:

    - the evolutionary outer loop will *search* the space of weight vectors;
    - the regime layer will hold a *separate* weight vector per market regime;
    - the headline finding is a *comparison* of the discovered weight vectors.

Keeping ``w`` an explicit, swappable object now means none of those later stages
require touching the environment or the agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

# Canonical ordering of the reward weights. The evolutionary loop will operate on the
# flat array form, so a single source of truth for the ordering avoids subtle bugs.
FACTOR_NAMES: List[str] = ["w_return", "w_e", "w_s", "w_g", "w_risk"]


@dataclass
class RewardWeights:
    """Weights for the multi-factor reward.

    Attributes:
        w_return: Weight on the (net-of-cost) portfolio return. Larger values make the
            agent more profit-seeking.
        w_e: Weight on the portfolio's Environmental score.
        w_s: Weight on the portfolio's Social score.
        w_g: Weight on the portfolio's Governance score.
        w_risk: Weight on the tail-risk penalty. This term is *subtracted*, so a larger
            value makes the agent more risk-averse.
    """

    w_return: float = 1.0
    w_e: float = 0.0
    w_s: float = 0.0
    w_g: float = 0.0
    w_risk: float = 0.0

    def as_array(self) -> np.ndarray:
        """Return the weights as a flat array in canonical ``FACTOR_NAMES`` order.

        Returns:
            A 1-D float array ``[w_return, w_e, w_s, w_g, w_risk]``. This is the form the
            evolutionary search will mutate.
        """
        return np.array(
            [self.w_return, self.w_e, self.w_s, self.w_g, self.w_risk],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, values: np.ndarray) -> "RewardWeights":
        """Build a :class:`RewardWeights` from a flat array in canonical order.

        Args:
            values: A length-5 sequence ordered as ``FACTOR_NAMES``.

        Returns:
            The corresponding :class:`RewardWeights` instance.

        Raises:
            ValueError: If ``values`` does not have exactly five elements.
        """
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.shape[0] != len(FACTOR_NAMES):
            raise ValueError(
                f"Expected {len(FACTOR_NAMES)} weights ({FACTOR_NAMES}), "
                f"got {values.shape[0]}."
            )
        return cls(
            w_return=float(values[0]),
            w_e=float(values[1]),
            w_s=float(values[2]),
            w_g=float(values[3]),
            w_risk=float(values[4]),
        )


def compute_reward(
    net_return: float,
    esg_e: float,
    esg_s: float,
    esg_g: float,
    downside_risk: float,
    weights: RewardWeights,
) -> float:
    """Compute the scalar reward for a single allocation step.

    The reward is the linear combination::

        r = w_return * net_return
          + w_e * esg_e + w_s * esg_s + w_g * esg_g
          - w_risk * downside_risk

    Args:
        net_return: Portfolio return for the step, already net of transaction costs.
        esg_e: Portfolio-weighted Environmental score (typically in ``[0, 1]``).
        esg_s: Portfolio-weighted Social score.
        esg_g: Portfolio-weighted Governance score.
        downside_risk: A non-negative per-step tail-risk proxy. In the v0 backbone this
            is the realised downside (the magnitude of a negative return), so that
            losses are penalised. Richer measures (rolling volatility, downside
            deviation) can replace it here without changing the interface.
        weights: The :class:`RewardWeights` defining the trade-off. In later stages this
            object is what the evolutionary loop searches and what the regime layer
            swaps per market regime.

    Returns:
        The scalar reward for the step.
    """
    return (
        weights.w_return * net_return
        + weights.w_e * esg_e
        + weights.w_s * esg_s
        + weights.w_g * esg_g
        - weights.w_risk * downside_risk
    )
