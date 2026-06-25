"""ESG-Adaptive RL.

A research package for sustainable portfolio allocation in which a reinforcement-
learning agent's reward weighting over financial return, the E/S/G factors, and tail
risk can later be evolved and indexed by market regime.

This module exposes the version string only; the concrete building blocks live in the
submodules:

    - ``config``  : central configuration for the v0 single-regime pipeline
    - ``data``    : price download and (placeholder) ESG-table construction
    - ``reward``  : the configurable multi-factor reward function
    - ``env``     : the custom Gymnasium portfolio-allocation environment
    - ``metrics`` : evaluation metrics (return, Sharpe, CVaR, drawdown, ESG profile)
"""

__version__ = "0.0.1"
