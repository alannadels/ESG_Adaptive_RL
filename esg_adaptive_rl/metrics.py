"""Evaluation metrics for a portfolio return trajectory.

These functions summarise a sequence of (daily) portfolio returns produced by rolling a
trained policy through a held-out period. They are deliberately framework-agnostic: each
takes a plain array of returns so they can grade any strategy (RL, baseline, or
classical optimiser) on equal footing.

Note on tail risk: the per-step reward uses a simple downside proxy for tractability,
but the *evaluation* here reports true Conditional Value at Risk (CVaR) computed over the
realised return distribution, which is the quantity the project ultimately cares about.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

# Trading days per year, used to annualise return and volatility.
TRADING_DAYS: int = 252


def annualized_return(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    """Geometric annualised return of a daily return series.

    Args:
        returns: Daily simple returns.
        periods_per_year: Number of periods per year (252 for daily data).

    Returns:
        The annualised return, or 0.0 for an empty series.
    """
    returns = np.asarray(returns, dtype=np.float64)
    n = returns.shape[0]
    if n == 0:
        return 0.0
    # Compound the period returns, then scale the growth to a one-year horizon.
    cumulative_growth = float(np.prod(1.0 + returns))
    return cumulative_growth ** (periods_per_year / n) - 1.0


def annualized_volatility(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised standard deviation of a daily return series.

    Args:
        returns: Daily simple returns.
        periods_per_year: Number of periods per year.

    Returns:
        The annualised volatility, or 0.0 for a series shorter than two points.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.shape[0] < 2:
        return 0.0
    # Sample standard deviation (ddof=1), scaled by sqrt(periods) to annualise.
    return float(np.std(returns, ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: int = TRADING_DAYS,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualised Sharpe ratio of a daily return series.

    Args:
        returns: Daily simple returns.
        periods_per_year: Number of periods per year.
        risk_free_rate: Annual risk-free rate, converted to per-period internally.

    Returns:
        The annualised Sharpe ratio, or 0.0 if volatility is zero/undefined.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.shape[0] < 2:
        return 0.0
    # Convert the annual risk-free rate to a per-period rate and take excess returns.
    per_period_rf = risk_free_rate / periods_per_year
    excess = returns - per_period_rf
    std = np.std(excess, ddof=1)
    if std == 0.0:
        return 0.0
    return float(np.mean(excess) / std * math.sqrt(periods_per_year))


def conditional_value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Conditional Value at Risk (Expected Shortfall) at level ``alpha``.

    This is the average return over the worst ``alpha`` fraction of periods, i.e. the
    expected loss conditional on being in the left tail. It is reported as a (typically
    negative) return.

    Args:
        returns: Daily simple returns.
        alpha: Tail probability (e.g. 0.05 for the worst 5% of days).

    Returns:
        The mean of the worst ``alpha`` fraction of returns, or 0.0 for an empty series.
    """
    returns = np.asarray(returns, dtype=np.float64)
    n = returns.shape[0]
    if n == 0:
        return 0.0
    # Number of tail observations; at least one so the measure is always defined.
    tail_count = max(1, int(math.ceil(alpha * n)))
    worst = np.sort(returns)[:tail_count]
    return float(np.mean(worst))


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown of the cumulative equity curve.

    Args:
        returns: Daily simple returns.

    Returns:
        The most negative peak-to-trough decline as a fraction (e.g. -0.32 for -32%),
        or 0.0 for an empty series.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.shape[0] == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    running_peak = np.maximum.accumulate(equity)
    drawdowns = (equity - running_peak) / running_peak
    return float(np.min(drawdowns))


def summarize(history: Dict[str, np.ndarray], alpha: float = 0.05) -> Dict[str, float]:
    """Summarise an episode trajectory into headline performance and ESG metrics.

    Args:
        history: A trajectory dict as returned by
            :meth:`esg_adaptive_rl.env.PortfolioEnv.get_history`, containing at least
            ``net_returns`` and the per-step ``esg_E``/``esg_S``/``esg_G`` profiles.
        alpha: Tail level for the CVaR metric.

    Returns:
        A dictionary of named scalar metrics.
    """
    net_returns = np.asarray(history["net_returns"], dtype=np.float64)
    return {
        "annual_return": annualized_return(net_returns),
        "annual_volatility": annualized_volatility(net_returns),
        "sharpe": sharpe_ratio(net_returns),
        f"cvar_{int(alpha * 100)}": conditional_value_at_risk(net_returns, alpha),
        "max_drawdown": max_drawdown(net_returns),
        # Average realised ESG exposure of the held portfolio over the episode.
        "avg_esg_E": float(np.mean(history["esg_E"])) if len(history["esg_E"]) else 0.0,
        "avg_esg_S": float(np.mean(history["esg_S"])) if len(history["esg_S"]) else 0.0,
        "avg_esg_G": float(np.mean(history["esg_G"])) if len(history["esg_G"]) else 0.0,
        "avg_turnover": float(np.mean(history["turnover"])) if len(history["turnover"]) else 0.0,
    }
