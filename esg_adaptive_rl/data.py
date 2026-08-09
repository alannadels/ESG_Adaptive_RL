"""Market-data loading and (placeholder) ESG-table construction.

This module produces the numerical inputs the environment consumes:

    - a matrix of daily simple returns (time x asset);
    - a time-indexed E/S/G table (one matrix per factor, time x asset).

IMPORTANT — the ESG table here is a deterministic *placeholder*, not real data. It
exists so the full pipeline can run end-to-end before real ESG histories are sourced.
It is built so that ESG enters the environment as a *time-indexed per-asset* input, so
that a real (and crucially, look-ahead-free) ESG history can be dropped in later with no
change to the environment or the agent.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# Names of the three ESG sub-factors, used as keys in the ESG dictionary throughout.
ESG_FACTORS: List[str] = ["E", "S", "G"]


@dataclass
class MarketData:
    """A bundle of aligned market inputs over a single date range.

    All arrays share the same time axis (``dates``) and the same asset axis
    (``tickers``), so row ``t`` and column ``j`` refer to the same day/asset across
    ``returns`` and every entry of ``esg``.

    Attributes:
        dates: Trading dates, length ``T``.
        tickers: Asset tickers, length ``N``, in column order.
        returns: Daily simple returns, shape ``(T, N)``.
        esg: Mapping ``{"E"|"S"|"G": array}`` with each array shape ``(T, N)`` and
            values in ``[0, 1]``.
    """

    dates: pd.DatetimeIndex
    tickers: List[str]
    returns: np.ndarray
    esg: Dict[str, np.ndarray]


def _stable_seed(text: str) -> int:
    """Map a string to a deterministic 32-bit seed.

    Python's built-in ``hash`` is salted per process, so it cannot be used for
    reproducibility. An MD5 digest gives the same seed on every run and machine.

    Args:
        text: The string to hash (here, a ticker symbol).

    Returns:
        A non-negative integer suitable as a NumPy random seed.
    """
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    # Use the first 8 hex characters (32 bits) as the seed.
    return int(digest[:8], 16)


def _synthetic_esg(tickers: List[str], n_days: int) -> Dict[str, np.ndarray]:
    """Build a deterministic placeholder E/S/G table.

    Each asset gets a fixed baseline E/S/G level (derived deterministically from its
    ticker) plus a slow sinusoidal drift and a small fixed noise component, all clipped
    to ``[0, 1]``. The result is fully reproducible and varies over time, which lets the
    environment treat ESG as a time-indexed input exactly as it would for real data.

    Args:
        tickers: Asset tickers, length ``N``.
        n_days: Number of trading days, ``T``.

    Returns:
        Mapping ``{"E"|"S"|"G": array}`` with each array of shape ``(T, N)``.
    """
    n_assets = len(tickers)
    esg = {factor: np.zeros((n_days, n_assets), dtype=np.float64) for factor in ESG_FACTORS}

    # A normalised time axis in [0, 1] drives the slow drift identically for all assets.
    time_fraction = np.linspace(0.0, 1.0, n_days)

    for col, ticker in enumerate(tickers):
        rng = np.random.default_rng(_stable_seed(ticker))
        # One baseline level per factor, kept away from the [0, 1] edges.
        baseline = rng.uniform(0.2, 0.8, size=len(ESG_FACTORS))
        for k, factor in enumerate(ESG_FACTORS):
            # Slow seasonal drift, phase-shifted per factor so E/S/G are not identical.
            drift = 0.05 * np.sin(2.0 * np.pi * (time_fraction + 0.1 * k))
            # Small deterministic noise (drawn from the per-ticker generator).
            noise = rng.normal(0.0, 0.01, size=n_days)
            esg[factor][:, col] = np.clip(baseline[k] + drift + noise, 0.0, 1.0)

    return esg


def load_market_data(
    tickers: List[str],
    start: str,
    end: str,
    esg_source: str = "synthetic",
    esg_path: Optional[str] = None,
    esg_anchor_year: int = 2025,
) -> MarketData:
    """Download prices and assemble aligned returns and an ESG table.

    Args:
        tickers: Asset tickers to include, in the desired column order.
        start: Start date (``YYYY-MM-DD``), inclusive.
        end: End date (``YYYY-MM-DD``), exclusive per yfinance convention.
        esg_source: ``"synthetic"`` for the deterministic placeholder table, or
            ``"refinitiv"`` to load the real ESG scores from ``esg_path``.
        esg_path: Path to the Refinitiv ESG CSV; required when
            ``esg_source == "refinitiv"``.
        esg_anchor_year: Calendar year that fiscal-year offset 0 (FY0) corresponds to in
            the Refinitiv export (only used for ``esg_source == "refinitiv"``).

    Returns:
        A :class:`MarketData` bundle with aligned returns and ESG arrays.

    Raises:
        ValueError: If no usable price data is returned, or if ``esg_source`` is
            ``"refinitiv"`` without an ``esg_path``.
    """
    # auto_adjust=True returns split/dividend-adjusted prices under the "Close" field.
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        repair=True,  # fix yfinance split/price glitches (e.g. an unadjusted JCI split)
    )
    if raw.empty:
        raise ValueError("yfinance returned no data for the requested universe/date range.")

    # With multiple tickers the columns are a (field, ticker) MultiIndex; select "Close".
    close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    # Force a consistent column order and drop any rows with missing prices.
    close = close.reindex(columns=tickers).dropna(how="any")
    if close.shape[0] < 2:
        raise ValueError("Not enough overlapping price history to compute returns.")

    # Daily simple returns; the first row becomes NaN and is dropped.
    returns_df = close.pct_change().dropna(how="any")

    dates = returns_df.index
    returns = returns_df.to_numpy(dtype=np.float64)

    # Attach the ESG table from the requested source. The real loader is imported lazily
    # so a synthetic-only run needs no ESG file.
    if esg_source == "refinitiv":
        if esg_path is None:
            raise ValueError("esg_path is required when esg_source='refinitiv'.")
        from esg_adaptive_rl.esg_data import build_esg_arrays, load_refinitiv_esg

        parsed = load_refinitiv_esg(esg_path)
        esg = build_esg_arrays(parsed, list(tickers), dates, anchor_year=esg_anchor_year)
    else:
        esg = _synthetic_esg(tickers, n_days=returns.shape[0])

    return MarketData(dates=dates, tickers=list(tickers), returns=returns, esg=esg)


def split_by_date(
    data: MarketData,
    split_date: str,
) -> Tuple[MarketData, MarketData]:
    """Split a :class:`MarketData` bundle into train (before) and test (on/after).

    The split is purely chronological, which keeps the test period strictly in the
    future relative to training and avoids leaking future information into training.

    Args:
        data: The full :class:`MarketData` bundle.
        split_date: Boundary date (``YYYY-MM-DD``). Rows strictly before it go to the
            train split; rows on or after it go to the test split.

    Returns:
        A ``(train, test)`` tuple of :class:`MarketData` bundles.
    """
    boundary = pd.Timestamp(split_date)
    train_mask = data.dates < boundary
    test_mask = ~train_mask

    def _subset(mask: np.ndarray) -> MarketData:
        return MarketData(
            dates=data.dates[mask],
            tickers=data.tickers,
            returns=data.returns[mask],
            esg={factor: matrix[mask] for factor, matrix in data.esg.items()},
        )

    return _subset(train_mask), _subset(test_mask)


def split_by_fraction(
    data: MarketData,
    train_fraction: float = 0.7,
) -> Tuple[MarketData, MarketData]:
    """Split a bundle chronologically by row fraction into (train, validation).

    Useful for a regime subset, whose days span the whole timeline but are already in
    date order: the earliest ``train_fraction`` of its rows become training data and the
    remainder validation, keeping the validation period later than training.

    Args:
        data: The :class:`MarketData` bundle (rows assumed in date order).
        train_fraction: Fraction of rows assigned to the training split.

    Returns:
        A ``(train, validation)`` tuple of :class:`MarketData` bundles.
    """
    n = data.returns.shape[0]
    cut = int(n * train_fraction)

    def _slice(sl: slice) -> MarketData:
        return MarketData(
            dates=data.dates[sl],
            tickers=data.tickers,
            returns=data.returns[sl],
            esg={factor: matrix[sl] for factor, matrix in data.esg.items()},
        )

    return _slice(slice(0, cut)), _slice(slice(cut, n))


def load_index_close(ticker: str, start: str, end: str) -> pd.Series:
    """Download the adjusted daily close for a single index/ETF (e.g. SPY).

    Used to define market regimes independently of the tradable universe.

    Args:
        ticker: The index/ETF ticker (e.g. ``"SPY"``).
        start: Start date (``YYYY-MM-DD``), inclusive.
        end: End date (``YYYY-MM-DD``), exclusive per yfinance convention.

    Returns:
        The adjusted-close price series indexed by date.

    Raises:
        ValueError: If no price data is returned.
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, repair=True)
    if raw.empty:
        raise ValueError(f"yfinance returned no data for index {ticker!r}.")
    close = raw["Close"]
    # A single-ticker download can still come back as a one-column frame.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


def load_regime_dataset(path: str) -> MarketData:
    """Load a cached regime CSV back into a :class:`MarketData` bundle.

    Reads a long-format regime file (``date, ticker, return, esg_E, esg_S, esg_G``, as
    written by ``build_regime_datasets.py``) and reshapes it into the arrays the
    environment expects — so the team can train directly on the shared, look-ahead-safe
    splits without re-downloading anything.

    Args:
        path: Path to a regime CSV (e.g. ``Dataset/regime_datasets/bear.csv``).

    Returns:
        A :class:`MarketData` bundle for that regime's days.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    tickers = sorted(df["ticker"].unique())
    returns = df.pivot(index="date", columns="ticker", values="return").reindex(columns=tickers)
    esg = {
        pillar: df.pivot(index="date", columns="ticker", values=f"esg_{pillar}")
        .reindex(columns=tickers)
        .to_numpy(dtype=np.float64)
        for pillar in ("E", "S", "G")
    }
    return MarketData(
        dates=returns.index,
        tickers=list(tickers),
        returns=returns.to_numpy(dtype=np.float64),
        esg=esg,
    )
