"""ESG regime feature engineering — pure, point-in-time.

Adapted from the SPY regime model (TradingAgentV2/v2/regime_core). The reference
leans on VIX/VIX9d option-implied term structure, which is SPX-specific. This
ESG variant is SELF-CONTAINED on an ESG ETF's own OHLCV: it substitutes a
*realized-vol* term structure (short-window rv / long-window rv) for the
implied-vol term structure, and drops the variance-risk-premium feature (needs
option-implied vol). Broad ^VIX is kept only as an optional macro candidate.

`compute_features(df)` takes a raw daily frame carrying (date, OHLCV) and
optionally (vix, vix3m) and returns the standardized feature frame.
Standardization uses a LAGGED expanding window so the scaler never sees the
future (no look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ESG core-5: realized-vol term structure replaces the VIX term structure;
# vrp is dropped (needs option-implied vol). Mirrors the reference core set.
CORE_FEATURES = ["rv_term_structure", "trend", "realized_vol", "drawdown", "mom_20"]
# LOCKED ESG model: core-5 + Parkinson OHLC range-vol (kept from the reference,
# which validated it OOS as a robust addition to close-to-close vol).
LOCKED_FEATURES = tuple(CORE_FEATURES) + ("parkinson_vol",)
# Candidates (tested, optional): broad-market fear + alternative OHLC vol.
CANDIDATE_FEATURES = ["vix_ts", "vix_level", "parkinson_vol", "yz_vol",
                      "true_range", "overnight_gap", "downside_vol"]
ALL_FEATURES = CORE_FEATURES + CANDIDATE_FEATURES
STD_MIN_PERIODS = 60


def _parkinson_vol(bars: pd.DataFrame, window: int = 20) -> pd.Series:
    hl = np.log(bars["high"].astype(float) / bars["low"].astype(float)) ** 2
    return np.sqrt(hl.rolling(window).mean() / (4 * np.log(2)) * 252)


def _yang_zhang_vol(bars: pd.DataFrame, window: int = 20) -> pd.Series:
    o, h, l, c = (np.log(bars[x].astype(float)) for x in ("open", "high", "low", "close"))
    overnight = o - c.shift(1)
    open_close = c - o
    rs = (h - c) * (h - o) + (l - c) * (l - o)
    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return np.sqrt((overnight.rolling(n).var() + k * open_close.rolling(n).var()
                    + (1 - k) * rs.rolling(n).mean()).clip(lower=0) * 252)


def _lagged_zscore(s: pd.Series, min_periods: int = STD_MIN_PERIODS) -> pd.Series:
    """Standardize using expanding mean/std THROUGH t-1 (no look-ahead)."""
    m = s.expanding(min_periods=min_periods).mean().shift(1)
    sd = s.expanding(min_periods=min_periods).std().shift(1).replace(0.0, np.nan)
    return (s - m) / sd


def compute_features(df: pd.DataFrame, vol_window: int = 20,
                     rv_short: int = 10, rv_long: int = 60) -> pd.DataFrame:
    """Raw frame (date, open/high/low/close/volume, [vix, vix3m]) ->
    standardized feature frame. Rows lacking CORE features are dropped; absent
    candidate sources are omitted.

    `rv_term_structure = log(rv_short / rv_long)` is the realized-vol analogue of
    the reference's `log(vix9d / vix)`: >0 means short-horizon vol is elevated
    relative to long-horizon vol (a stress / backwardation signal)."""
    required = {"date", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"compute_features missing columns: {sorted(missing)}")
    df = df.sort_values("date").reset_index(drop=True).copy()
    for c in ("vix", "vix3m"):
        if c in df.columns:
            df[c] = df[c].ffill()

    close = df["close"].astype(float)
    logret = np.log(close / close.shift(1))
    prev_close = close.shift(1)

    # realized-vol term structure (self-contained substitute for VIX term struct)
    rv_s = logret.rolling(rv_short).std() * np.sqrt(252)
    rv_l = logret.rolling(rv_long).std() * np.sqrt(252)
    df["rv_term_structure"] = np.log(rv_s / rv_l)

    # core
    df["realized_vol"] = logret.rolling(vol_window).std() * np.sqrt(252)
    df["sma200"] = close.rolling(200).mean()
    df["trend"] = close / df["sma200"] - 1.0
    df["drawdown"] = close / close.rolling(252).max() - 1.0
    df["mom_20"] = close.pct_change(20)

    # OHLC-based vol
    df["parkinson_vol"] = _parkinson_vol(df, vol_window)
    df["yz_vol"] = _yang_zhang_vol(df, 20)

    # downside (semi) vol — only negative returns
    neg = logret.where(logret < 0, 0.0)
    df["downside_vol"] = neg.rolling(vol_window).std() * np.sqrt(252)

    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    df["true_range"] = (tr / close).rolling(20).mean()
    df["overnight_gap"] = np.log(df["open"] / prev_close).abs().rolling(20).mean()

    # optional broad-market macro candidates
    if "vix" in df.columns and "vix3m" in df.columns:
        df["vix_ts"] = np.log(df["vix"] / df["vix3m"])
    if "vix" in df.columns:
        df["vix_level"] = df["vix"]

    present = [f for f in ALL_FEATURES if f in df.columns]
    for f in present:
        df[f + "_z"] = _lagged_zscore(df[f])

    raw_keep = ["date", "close", "sma200", "rv_term_structure", "realized_vol",
                "trend", "drawdown", "mom_20"] + \
               [f for f in CANDIDATE_FEATURES if f in df.columns]
    keep = list(dict.fromkeys(raw_keep)) + [f + "_z" for f in present]
    return df[keep].dropna(subset=[f + "_z" for f in CORE_FEATURES]).reset_index(drop=True)
