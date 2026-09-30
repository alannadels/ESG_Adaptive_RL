"""Offline benchmark return series for alpha/beta-style evaluation.

The per-regime evaluation reports how the evolved allocator compares to market and ESG
benchmarks (beta, Jensen's alpha, information ratio, tracking error). Those metrics need
a benchmark daily-return series aligned to the exact trading days of a given evaluation
window.

This module loads the committed benchmark price CSVs (``esg_regime/data/<TICKER>.csv``,
each with a ``date`` and ``close`` column) and returns daily returns aligned to a
requested set of dates. Everything is read from disk, so evaluation stays fully offline
and reproducible — no network, no rebuild.

The default panel:

    SPY   - S&P 500, the broad market (beta baseline)
    SUSA  - iShares MSCI USA ESG Select (deep-history broad US ESG)
    DSI   - iShares MSCI KLD 400 Social (deep-history socially-screened)
    SPYX  - SPDR S&P 500 Fossil Fuel Reserves Free (matches the no-fossil mandate)

Younger ETFs may not span an entire window; alignment is an inner join on date, so a
benchmark contributes only the days it actually covers (the caller reports that count).
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

# Directory holding the committed benchmark price histories.
_BENCHMARK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "esg_regime", "data"
)

# The default benchmark panel (see module docstring for what each represents).
DEFAULT_BENCHMARKS: List[str] = ["SPY", "SUSA", "DSI", "SPYX"]


def _load_benchmark_returns(ticker: str) -> pd.Series:
    """Load one benchmark's daily simple returns, indexed by date.

    Args:
        ticker: Benchmark ticker whose ``esg_regime/data/<ticker>.csv`` is read.

    Returns:
        A date-indexed Series of daily simple returns (from the ``close`` column).

    Raises:
        FileNotFoundError: If the benchmark CSV is missing.
    """
    path = os.path.join(_BENCHMARK_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"benchmark price file not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    close = df.set_index("date")["close"].astype(float)
    # Daily simple returns; the first day becomes NaN and is dropped.
    return close.pct_change().dropna()


def benchmark_returns_for_dates(
    dates,
    tickers: List[str] = None,
) -> Dict[str, np.ndarray]:
    """Return daily benchmark returns aligned to ``dates``, one array per benchmark.

    For each benchmark, the returns are inner-joined to ``dates`` on the calendar date,
    so the result contains only the days the benchmark actually covers, in the order of
    ``dates``. This makes each benchmark's coverage explicit: a younger ETF simply yields
    a shorter array, and the caller can pair it with the matching portfolio-return days.

    Args:
        dates: The evaluation window's trading dates (e.g. ``val.dates`` slice), in order.
        tickers: Benchmarks to load; defaults to :data:`DEFAULT_BENCHMARKS`.

    Returns:
        A mapping ``{ticker: np.ndarray}`` where each array has the same length as
        ``dates`` and holds that benchmark's daily return per day, with ``NaN`` on days
        the benchmark does not cover. Pair it with the portfolio returns and drop the
        uncovered days with :func:`align_pair` before computing metrics.
    """
    tickers = tickers or DEFAULT_BENCHMARKS
    target = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates)))
    out: Dict[str, np.ndarray] = {}
    for ticker in tickers:
        series = _load_benchmark_returns(ticker)
        # Reindex onto the requested dates; NaN where the benchmark has no data.
        aligned = series.reindex(target)
        out[ticker] = aligned.to_numpy(dtype=np.float64)
    return out


def align_pair(port: np.ndarray, bench: np.ndarray):
    """Drop days where the benchmark has no data, keeping the portfolio pair in sync.

    Args:
        port: Portfolio daily returns over the full evaluation window.
        bench: Benchmark daily returns over the same window, with ``NaN`` on days the
            benchmark does not cover (as produced by :func:`benchmark_returns_for_dates`).

    Returns:
        A ``(port_sub, bench_sub)`` pair restricted to the days both series cover.
    """
    port = np.asarray(port, dtype=np.float64)
    bench = np.asarray(bench, dtype=np.float64)
    n = min(port.shape[0], bench.shape[0])
    port, bench = port[:n], bench[:n]
    mask = ~np.isnan(bench)
    return port[mask], bench[mask]
