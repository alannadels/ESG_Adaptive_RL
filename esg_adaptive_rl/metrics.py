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


def sortino_ratio(
    returns: np.ndarray,
    periods_per_year: int = TRADING_DAYS,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualised Sortino ratio: excess return per unit of *downside* deviation.

    Like the Sharpe ratio but penalising only harmful (below-target) volatility, so a
    strategy is not charged for upside variability. The target is the per-period
    risk-free rate.

    Args:
        returns: Daily simple returns.
        periods_per_year: Number of periods per year.
        risk_free_rate: Annual risk-free rate, converted to per-period internally.

    Returns:
        The annualised Sortino ratio, or 0.0 if the downside deviation is zero/undefined.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.shape[0] < 2:
        return 0.0
    per_period_rf = risk_free_rate / periods_per_year
    excess = returns - per_period_rf
    # Downside deviation: RMS of the shortfall below the target (upside set to zero).
    shortfall = np.minimum(excess, 0.0)
    downside_dev = math.sqrt(float(np.mean(shortfall ** 2)))
    if downside_dev == 0.0:
        return 0.0
    return float(np.mean(excess) / downside_dev * math.sqrt(periods_per_year))


def calmar_ratio(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    """Calmar ratio: annualised return divided by the magnitude of the max drawdown.

    A higher value means more return earned per unit of worst-case peak-to-trough pain.

    Args:
        returns: Daily simple returns.
        periods_per_year: Number of periods per year (for the annualised return).

    Returns:
        The Calmar ratio, or 0.0 if the maximum drawdown is zero/undefined.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.shape[0] == 0:
        return 0.0
    mdd = abs(max_drawdown(returns))
    if mdd == 0.0:
        return 0.0
    return float(annualized_return(returns, periods_per_year) / mdd)


def _align_nonempty(port: np.ndarray, bench: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Coerce a portfolio/benchmark return pair to equal-length float arrays.

    Args:
        port: Portfolio daily returns.
        bench: Benchmark daily returns (already date-aligned to ``port`` by the caller).

    Returns:
        The two arrays as float64, truncated to their common length.
    """
    port = np.asarray(port, dtype=np.float64)
    bench = np.asarray(bench, dtype=np.float64)
    n = min(port.shape[0], bench.shape[0])
    return port[:n], bench[:n]


def beta(port: np.ndarray, bench: np.ndarray) -> float:
    """Market beta: sensitivity of the portfolio's returns to the benchmark's.

    ``beta = cov(port, bench) / var(bench)``. A beta of 1 moves one-for-one with the
    benchmark; below 1 is less market-sensitive.

    Args:
        port: Portfolio daily returns.
        bench: Benchmark daily returns, date-aligned to ``port``.

    Returns:
        The beta, or 0.0 if the benchmark variance is zero or fewer than two points.
    """
    port, bench = _align_nonempty(port, bench)
    if port.shape[0] < 2:
        return 0.0
    var_bench = float(np.var(bench, ddof=1))
    if var_bench == 0.0:
        return 0.0
    cov = float(np.cov(port, bench, ddof=1)[0, 1])
    return cov / var_bench


def jensen_alpha(
    port: np.ndarray,
    bench: np.ndarray,
    periods_per_year: int = TRADING_DAYS,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualised Jensen's alpha: CAPM-style excess return not explained by market beta.

    ``alpha_period = mean(port - rf) - beta * mean(bench - rf)``, annualised by scaling
    by ``periods_per_year``. Positive alpha means the portfolio out-earned what its market
    exposure alone would predict.

    Args:
        port: Portfolio daily returns.
        bench: Benchmark daily returns, date-aligned to ``port``.
        periods_per_year: Number of periods per year (annualisation factor).
        risk_free_rate: Annual risk-free rate, converted to per-period internally.

    Returns:
        The annualised alpha, or 0.0 for a series shorter than two points.
    """
    port, bench = _align_nonempty(port, bench)
    if port.shape[0] < 2:
        return 0.0
    per_period_rf = risk_free_rate / periods_per_year
    b = beta(port, bench)
    alpha_period = float(np.mean(port - per_period_rf) - b * np.mean(bench - per_period_rf))
    return alpha_period * periods_per_year


def information_ratio(
    port: np.ndarray,
    bench: np.ndarray,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised information ratio: mean active return over tracking error.

    ``IR = mean(port - bench) / std(port - bench)``, annualised. Measures active return
    earned per unit of active risk taken versus the benchmark.

    Args:
        port: Portfolio daily returns.
        bench: Benchmark daily returns, date-aligned to ``port``.
        periods_per_year: Number of periods per year (annualisation factor).

    Returns:
        The annualised information ratio, or 0.0 if active risk is zero/undefined.
    """
    port, bench = _align_nonempty(port, bench)
    if port.shape[0] < 2:
        return 0.0
    active = port - bench
    std_active = float(np.std(active, ddof=1))
    if std_active == 0.0:
        return 0.0
    return float(np.mean(active) / std_active * math.sqrt(periods_per_year))


def tracking_error(
    port: np.ndarray,
    bench: np.ndarray,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised tracking error: volatility of the active (portfolio minus benchmark) return.

    Args:
        port: Portfolio daily returns.
        bench: Benchmark daily returns, date-aligned to ``port``.
        periods_per_year: Number of periods per year (annualisation factor).

    Returns:
        The annualised tracking error, or 0.0 for a series shorter than two points.
    """
    port, bench = _align_nonempty(port, bench)
    if port.shape[0] < 2:
        return 0.0
    active = port - bench
    return float(np.std(active, ddof=1) * math.sqrt(periods_per_year))


def benchmark_metrics(
    port: np.ndarray,
    bench: np.ndarray,
    periods_per_year: int = TRADING_DAYS,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Bundle the benchmark-relative metrics for one (portfolio, benchmark) pair.

    Args:
        port: Portfolio daily returns.
        bench: Benchmark daily returns, date-aligned to ``port`` by the caller.
        periods_per_year: Number of periods per year (annualisation factor).
        risk_free_rate: Annual risk-free rate for the alpha calculation.

    Returns:
        A dict with ``beta``, ``alpha`` (annualised), ``info_ratio``, ``tracking_error``,
        and ``cov_days`` (the number of aligned observations the metrics used).
    """
    port, bench = _align_nonempty(port, bench)
    return {
        "beta": beta(port, bench),
        "alpha": jensen_alpha(port, bench, periods_per_year, risk_free_rate),
        "info_ratio": information_ratio(port, bench, periods_per_year),
        "tracking_error": tracking_error(port, bench, periods_per_year),
        "cov_days": int(port.shape[0]),
    }


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
        "sortino": sortino_ratio(net_returns),
        "calmar": calmar_ratio(net_returns),
        f"cvar_{int(alpha * 100)}": conditional_value_at_risk(net_returns, alpha),
        "max_drawdown": max_drawdown(net_returns),
        # Average realised ESG exposure of the held portfolio over the episode.
        "avg_esg_E": float(np.mean(history["esg_E"])) if len(history["esg_E"]) else 0.0,
        "avg_esg_S": float(np.mean(history["esg_S"])) if len(history["esg_S"]) else 0.0,
        "avg_esg_G": float(np.mean(history["esg_G"])) if len(history["esg_G"]) else 0.0,
        "avg_turnover": float(np.mean(history["turnover"])) if len(history["turnover"]) else 0.0,
    }
