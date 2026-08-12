"""Market-regime detection for the ESG universe.

This module is the integration layer between the low-level HMM (``features`` +
``classifier``) and the rest of the ESG-Adaptive-RL pipeline. It mirrors the
conventions of :mod:`esg_adaptive_rl.config` and :mod:`esg_adaptive_rl.data`:
central configuration in a dataclass, a chronological train/test split that
avoids look-ahead, and Google-style docstrings.

The detector consumes a daily OHLC price frame for the ESG universe (either the
high-ESG index built by ``build_esg_index.py`` or an ESG ETF proxy) and labels
each day as one of three regimes:

    bull     — low realised volatility, above trend, near highs
    neutral  — elevated volatility, range-bound
    bear     — high volatility, below trend, deep drawdown

Downstream, these labels are what the evolutionary layer will condition its
per-regime reward weights on: the RL agent can hold a different E/S/G-vs-return
trade-off in bull markets than in bear ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from esg_regime.classifier import (
    RegimeClassifier,
    RegimeConfig as _HMMConfig,
    finalize_regimes,
    in_sample_regimes,
    normalize_regimes,
    walk_forward_regimes,
)
from esg_regime.features import LOCKED_FEATURES, compute_features

# The three regime labels, ordered most-bullish/calmest to most-bearish/stressed.
REGIMES: List[str] = ["bull", "neutral", "bear"]


@dataclass
class RegimeDetectorConfig:
    """Central configuration for the ESG regime detector.

    Attributes:
        features: The locked feature set fed to the HMM (realised-vol term
            structure, trend, realised vol, drawdown, momentum, Parkinson vol).
        n_states: Number of hidden regimes (3).
        n_init: Random restarts per HMM fit; the best log-likelihood wins.
        conf_threshold: Minimum filtered posterior to accept a regime switch
            (anti-whipsaw hysteresis).
        dwell: Minimum consecutive days in a new raw regime before it is
            confirmed (anti-whipsaw hysteresis).
        min_train: Days in the initial expanding window for walk-forward mode.
        retrain_every: Refit cadence (days) for walk-forward mode.
        split_date: Chronological boundary for the train/test evaluation; days
            before it train the model, days on/after it are scored out-of-sample.
    """

    features: tuple = LOCKED_FEATURES
    n_states: int = 3
    n_init: int = 8
    conf_threshold: float = 0.60
    dwell: int = 2
    min_train: int = 252
    retrain_every: int = 21
    split_date: str = "2021-01-01"
    label_by: str = "vol"

    def to_hmm(self) -> _HMMConfig:
        """Project onto the low-level HMM config used by the classifier."""
        return _HMMConfig(
            n_states=self.n_states,
            n_init=self.n_init,
            conf_threshold=self.conf_threshold,
            dwell=self.dwell,
            features=self.features,
            label_by=self.label_by,
        )


class MarketRegimeDetector:
    """Fit / predict wrapper around the 3-state ESG regime HMM.

    Two usage modes:

    * :meth:`fit` then :meth:`predict` — the honest chronological protocol. Fit
      the HMM on a training window, freeze it, and decode a (later) test window
      with filtered posteriors so no test bar is seen during training.
    * :meth:`walk_forward` — a rolling variant that retrains monthly on an
      expanding window; every bar is labeled by a model trained only on its past.
    """

    def __init__(self, config: Optional[RegimeDetectorConfig] = None) -> None:
        self.config = config or RegimeDetectorConfig()
        self._clf: Optional[RegimeClassifier] = None

    # -- feature engineering -----------------------------------------------
    def compute_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Turn a raw OHLC frame into the standardized feature frame.

        Args:
            prices: Daily frame with ``date, open, high, low, close`` (and
                optionally ``vix, vix3m``).

        Returns:
            The standardized, look-ahead-free feature frame.
        """
        return compute_features(prices)

    # -- chronological fit / predict ---------------------------------------
    def fit(self, train_features: pd.DataFrame) -> "MarketRegimeDetector":
        """Fit the HMM on a training feature frame and pin the vol-ordered labels.

        Args:
            train_features: Feature frame restricted to the training period.

        Returns:
            ``self``, with a fitted classifier.
        """
        self._clf = RegimeClassifier(self.config.to_hmm()).fit(train_features)
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Decode regimes for a feature frame with the frozen model.

        Uses filtered (forward-only) posteriors, so labeling day ``t`` never uses
        data after ``t``. Safe to call on the held-out test window.

        Args:
            features: Feature frame to label (train, test, or the full series).

        Returns:
            A frame with ``date, close, regime, confidence`` and guardrail flags.
        """
        if self._clf is None:
            raise RuntimeError("call fit() before predict()")
        return self._clf.predict_regimes(features)

    # -- walk-forward ------------------------------------------------------
    def walk_forward(self, features: pd.DataFrame) -> pd.DataFrame:
        """Label the whole series out-of-sample via monthly expanding-window refits.

        Args:
            features: The full feature frame.

        Returns:
            The regime frame over all bars, each labeled point-in-time.
        """
        return walk_forward_regimes(
            features, self.config.to_hmm(),
            min_train=self.config.min_train,
            retrain_every=self.config.retrain_every,
        )

    def in_sample(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fit once on the full history and decode (optimistic, in-sample)."""
        return in_sample_regimes(features, self.config.to_hmm())


def split_features_by_date(
    features: pd.DataFrame, split_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split a feature frame into train (before) / test (on-after).

    Mirrors :func:`esg_adaptive_rl.data.split_by_date` but operates on the regime
    feature frame. The split is purely by date, keeping the test period strictly
    in the future relative to training.

    Args:
        features: The full feature frame (must carry a ``date`` column).
        split_date: Boundary date (``YYYY-MM-DD``).

    Returns:
        A ``(train, test)`` tuple of feature frames.
    """
    boundary = pd.Timestamp(split_date)
    dates = pd.to_datetime(features["date"])
    train = features[dates < boundary].reset_index(drop=True)
    test = features[dates >= boundary].reset_index(drop=True)
    return train, test


def label_regimes(
    prices: pd.DataFrame,
    config: Optional[RegimeDetectorConfig] = None,
    mode: str = "walk_forward",
) -> pd.DataFrame:
    """One-call regime labeling for an ESG price series.

    Args:
        prices: Raw daily OHLC frame for the ESG universe.
        config: Detector configuration; defaults are used if omitted.
        mode: ``"walk_forward"`` (out-of-sample, default) or ``"in_sample"``.

    Returns:
        The regime frame aligned to the (feature-valid) trading dates.
    """
    det = MarketRegimeDetector(config)
    feats = det.compute_features(prices)
    if mode == "walk_forward":
        return det.walk_forward(feats)
    if mode == "in_sample":
        return det.in_sample(feats)
    raise ValueError(f"unknown mode: {mode!r}")
