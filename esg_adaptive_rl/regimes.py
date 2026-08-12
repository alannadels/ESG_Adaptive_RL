"""Market-regime labeling (bull / neutral / bear) — causal, no look-ahead.

Labels each trading day of a market index (e.g. SPY) as ``bull``, ``neutral``, or
``bear``. These labels split the daily dataset into three regime subsets, each of
which trains its own specialist allocator.

Three pluggable detectors, driven by :class:`RegimeConfig`:

    - ``"hmm"`` (DEFAULT) : the 3-state walk-forward Gaussian HMM from
      :mod:`esg_regime` (filtered posteriors, monthly expanding-window refits —
      the one causal Markov-switching protocol). It emits bull/neutral/bear
      directly; its states are vol-ordered and that ordering coincides with the
      trend ordering on every universe tested. Chosen as default after
      a 11-detector comparison across 6 universes (see
      esg_regime/results/HEURISTICS_BENCHMARKS.md): stress/calm vol separation
      2.73 vs 1.70 for the 50/200 crossover, overlay dSharpe +0.11 vs +0.03,
      with SPY->ESG label transfer validated at 85-96% agreement.

    - ``"crossover"`` : a fast/slow moving-average crossover (default 50/200). The fast
      average tracks the recent trend, the slow the long trend; the sign of their gap
      (with a neutral band around the crossover) gives the regime. More responsive — the
      intended "live" detector, so training and live switching use the same signal.
    - ``"trend"``     : price versus a single slow moving average (default 200), with a
      neutral band. Smoother and laggier — the baseline.

Either detector can use a simple (``"sma"``) or exponential (``"ema"``) moving average;
the EMA reacts faster to recent prices. A causal minimum-dwell filter absorbs brief
threshold flip-flops, confirming a regime switch only after the new label persists for
``min_dwell`` days (which introduces an honest detection lag).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

# The three regime labels, ordered from most bullish to most bearish.
REGIMES: List[str] = ["bull", "neutral", "bear"]


@dataclass
class RegimeConfig:
    """Configuration for the rule-based regime detector.

    Attributes:
        detector: ``"hmm"`` (walk-forward 3-state HMM — the default),
            ``"crossover"`` (fast/slow MA crossover) or ``"trend"`` (price vs a
            single slow MA — the baseline).
        ticker: The index whose cached OHLC history feeds the HMM detector
            (the HMM's range-volatility feature needs high/low, which the
            close-only ``prices`` input does not carry).
        fast_window: Fast moving-average window (crossover detector only).
        slow_window: Slow moving-average window (both detectors).
        ma_type: ``"sma"`` (equal-weighted) or ``"ema"`` (recent-weighted, more
            responsive).
        neutral_band: Half-width of the neutral zone, as a fraction. The day is
            ``neutral`` when the two averages (crossover) or price and the average
            (trend) are within this band of each other.
        min_dwell: Minimum consecutive days a new raw label must persist before the
            switch is confirmed (causal anti-whipsaw smoothing). ``0`` disables it.
    """

    detector: str = "hmm"
    ticker: str = "SPY"
    fast_window: int = 50
    slow_window: int = 200
    ma_type: str = "sma"
    neutral_band: float = 0.02
    min_dwell: int = 10


def _moving_average(series: pd.Series, window: int, ma_type: str) -> pd.Series:
    """Compute a simple or exponential moving average.

    Args:
        series: The price series.
        window: The averaging window (span, for the EMA).
        ma_type: ``"sma"`` or ``"ema"``.

    Returns:
        The moving-average series (leading values are NaN for the SMA until the window
        is full).

    Raises:
        ValueError: If ``ma_type`` is not ``"sma"`` or ``"ema"``.
    """
    if ma_type == "sma":
        return series.rolling(window=window).mean()
    if ma_type == "ema":
        # ``adjust=False`` gives the standard recursive EMA; span = window.
        return series.ewm(span=window, adjust=False).mean()
    raise ValueError(f"unknown ma_type {ma_type!r}; expected 'sma' or 'ema'")


def _raw_labels(prices: pd.Series, cfg: RegimeConfig) -> List[str]:
    """Compute the un-smoothed per-day regime labels.

    The signal is a relative gap: ``fast/slow - 1`` for the crossover detector, or
    ``price/slow - 1`` for the trend detector. Above ``+neutral_band`` is ``bull``, below
    ``-neutral_band`` is ``bear``, and in between is ``neutral``. Days before the slow
    average is defined are ``neutral``.

    Args:
        prices: The index price series.
        cfg: The detector configuration.

    Returns:
        A list of raw labels aligned to ``prices``.

    Raises:
        ValueError: If ``cfg.detector`` is not ``"crossover"`` or ``"trend"``.
    """
    slow = _moving_average(prices, cfg.slow_window, cfg.ma_type)
    if cfg.detector == "crossover":
        fast = _moving_average(prices, cfg.fast_window, cfg.ma_type)
        gap = fast / slow - 1.0
    elif cfg.detector == "trend":
        gap = prices / slow - 1.0
    else:
        raise ValueError(f"unknown detector {cfg.detector!r}; expected 'crossover' or 'trend'")

    labels: List[str] = []
    for value in gap.to_numpy():
        if not np.isfinite(value):
            labels.append("neutral")  # slow average not yet defined
        elif value > cfg.neutral_band:
            labels.append("bull")
        elif value < -cfg.neutral_band:
            labels.append("bear")
        else:
            labels.append("neutral")
    return labels


def _apply_min_dwell(raw: List[str], min_dwell: int) -> List[str]:
    """Confirm regime switches only after a new label persists ``min_dwell`` days.

    This is a forward-only (causal) debounce: on each day it uses only that day's and
    prior raw labels, so it introduces a detection *lag* but never look-ahead. Brief
    flip-flops shorter than ``min_dwell`` are absorbed into the standing regime.

    Args:
        raw: The un-smoothed labels.
        min_dwell: Minimum consecutive days a challenger label must hold to be confirmed.

    Returns:
        The smoothed labels, same length as ``raw``.
    """
    if min_dwell <= 1 or not raw:
        return list(raw)

    confirmed = raw[0]
    candidate = confirmed
    candidate_run = 0
    out: List[str] = []
    for label in raw:
        if label == confirmed:
            # Back to the standing regime; reset any pending challenger.
            candidate_run = 0
        else:
            # A challenger: extend its run, or start a new one.
            if label == candidate:
                candidate_run += 1
            else:
                candidate = label
                candidate_run = 1
            # Confirm the switch once the challenger has held long enough.
            if candidate_run >= min_dwell:
                confirmed = candidate
                candidate_run = 0
        out.append(confirmed)
    return out


def label_regimes(prices: pd.Series, cfg: RegimeConfig = RegimeConfig()) -> pd.Series:
    """Label each day of a market index bull / neutral / bear.

    Args:
        prices: Daily index prices (e.g. SPY close), indexed by date.
        cfg: Detector configuration.

    Returns:
        A Series of regime labels aligned to ``prices.index``.
    """
    prices = pd.Series(prices).astype(float)
    if cfg.detector == "hmm":
        return _hmm_labels(prices, cfg)
    raw = _raw_labels(prices, cfg)
    smoothed = _apply_min_dwell(raw, cfg.min_dwell)
    return pd.Series(smoothed, index=prices.index, name="regime")


def _hmm_labels(prices: pd.Series, cfg: RegimeConfig) -> pd.Series:
    """Walk-forward HMM labels for ``cfg.ticker``, aligned to ``prices.index``.

    Delegates to :func:`esg_regime.benchmark_compare.hmm_labels`, which returns
    (and caches) the point-in-time label path: the HMM is refit monthly on an
    expanding window and decoded with filtered posteriors, so no day is labeled
    by a model that saw it. The HMM's own OHLC history for ``cfg.ticker`` is
    used for features; labels are then as-of aligned onto the requested dates
    (same forward-fill convention as :func:`split_by_regime`). Days before the
    HMM's warm-up are labeled ``neutral``.

    Raises:
        ImportError: If the ``esg_regime`` package (and its requirements, e.g.
            ``hmmlearn``) is not importable — fall back to
            ``RegimeConfig(detector="crossover")``.
    """
    from esg_regime.benchmark_compare import hmm_labels
    from esg_regime.classifier import normalize_regimes

    lab = hmm_labels(cfg.ticker)
    # esg_regime emits bull/neutral/bear directly; normalize_regimes translates
    # any label paths still cached under the pre-rename S1/S2/S3 names.
    series = normalize_regimes(
        lab.assign(date=pd.to_datetime(lab["date"])).set_index("date")["regime"])
    aligned = series.reindex(pd.to_datetime(prices.index), method="ffill")
    return pd.Series(aligned.fillna("neutral").values, index=prices.index,
                     name="regime")


def split_by_regime(data, labels: pd.Series) -> Dict[str, "object"]:
    """Split a market dataset into its bull / neutral / bear subsets.

    The regime labels (defined on the market index) are aligned to the dataset's trading
    days by an as-of forward fill, and each regime's rows are gathered into their own
    :class:`esg_adaptive_rl.data.MarketData` bundle. Rows are concatenated in date order;
    they need not be calendar-contiguous (regimes recur in separate stretches).

    Args:
        data: The full :class:`esg_adaptive_rl.data.MarketData` bundle.
        labels: Regime labels from :func:`label_regimes`, indexed by date.

    Returns:
        A mapping ``{regime: MarketData}`` for each regime that has at least one day.
    """
    # Imported here to avoid a circular import at module load time.
    from esg_adaptive_rl.data import MarketData

    # As-of align the index-based labels onto the dataset's trading days.
    aligned = labels.reindex(
        pd.to_datetime(pd.Index(data.dates)), method="ffill"
    ).to_numpy()

    subsets: Dict[str, MarketData] = {}
    for regime in REGIMES:
        mask = aligned == regime
        if mask.any():
            subsets[regime] = MarketData(
                dates=data.dates[mask],
                tickers=data.tickers,
                returns=data.returns[mask],
                esg={factor: matrix[mask] for factor, matrix in data.esg.items()},
            )
    return subsets
