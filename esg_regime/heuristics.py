"""Rule-based (heuristic) regime detectors — causal, point-in-time.

Standard practitioner / academic rules for labeling equity-market regimes, each
mapped onto the same 3-state scheme the HMM uses (bull / neutral /
bear) so every detector can be graded by the same harness
(:mod:`esg_regime.benchmark_compare`).

Detectors (all causal — each day's label uses only data through that day):

  ``ma_crossover``    The repo's production rule (esg_adaptive_rl.regimes):
                      50/200 SMA crossover, 2% neutral band, 10-day dwell.
  ``trend_200``       The repo's baseline: price vs 200 SMA with band.
  ``drawdown_bear``   The classic industry convention: >=20% below the running
                      peak is a bear market, >=10% a correction (neutral),
                      else bull. (S&P Dow Jones / press convention.)
  ``consec_down``     Streak rule (user-suggested family): N trailing
                      consecutive negative closes. streak>=5 -> bear,
                      streak in {3,4} -> neutral, else bull.
  ``downday_frac``    Smoother cousin: fraction of down days in the last 10.
                      >=0.7 -> bear, >=0.5 -> neutral, else bull.
  ``vol_percentile``  Realized 20d vol vs its own expanding distribution
                      through t-1: >85th pct -> bear, >60th -> neutral, else bull.
  ``vix_threshold``   Classic VIX cutoffs: VIX<20 -> bull, 20-30 -> neutral,
                      >30 -> bear. (Uses broad ^VIX for every universe.)
  ``mom_12m``         Time-series momentum sign (Moskowitz-Ooi-Pedersen 2012):
                      trailing 252d return > +5% -> bull, < -5% -> bear, else neutral.
  ``lunde_timmermann``Lunde-Timmermann (2004) first-passage filter, baseline
                      20%/15% thresholds — the causal academic bull/bear rule.
  ``ret_sign_60``     Trailing 60d return sign with +/-5% band — the deep-RL
                      trading literature's bull/neutral/bear convention.

Retrospective dating algorithms (Pagan-Sossounov 2003, Bry-Boschan/BBQ, NBER
dating) are deliberately excluded: they identify peaks/troughs using data
*after* the turning point (local-extremum windows, end censoring, announcement
lags), so they cannot produce honest point-in-time labels. See
results/regime_methods_catalog.json for the full researched catalog.
"""
from __future__ import annotations

import os
from typing import Callable, Dict

import numpy as np
import pandas as pd

BULL, NEUTRAL, BEAR = "bull", "neutral", "bear"
REGIME_ORDER = [BULL, NEUTRAL, BEAR]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


# --------------------------------------------------------------------- helpers
def _out(dates: pd.Series, labels) -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(dates).values, "regime": list(labels)})


def _load_vix() -> pd.Series:
    """Cached ^VIX close, indexed by date (downloads once)."""
    cache = os.path.join(DATA, "VIX.csv")
    if os.path.exists(cache):
        v = pd.read_csv(cache, parse_dates=["date"])
    else:
        import yfinance as yf
        raw = yf.download("^VIX", start="2000-01-01", progress=False, auto_adjust=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        v = raw["Close"].rename("vix").reset_index().rename(columns={"Date": "date"})
        v.to_csv(cache, index=False)
    return v.set_index("date")["vix"]


# ------------------------------------------------------------------- detectors
def ma_crossover(prices: pd.DataFrame) -> pd.DataFrame:
    """The repo's production 50/200 crossover detector, verbatim."""
    from esg_adaptive_rl.regimes import RegimeConfig, label_regimes
    s = pd.Series(prices["close"].values, index=pd.to_datetime(prices["date"]))
    lab = label_regimes(s, RegimeConfig(detector="crossover"))
    return _out(prices["date"], list(lab))       # already bull/neutral/bear


def trend_200(prices: pd.DataFrame) -> pd.DataFrame:
    """The repo's baseline price-vs-200SMA detector, verbatim."""
    from esg_adaptive_rl.regimes import RegimeConfig, label_regimes
    s = pd.Series(prices["close"].values, index=pd.to_datetime(prices["date"]))
    lab = label_regimes(s, RegimeConfig(detector="trend"))
    return _out(prices["date"], list(lab))       # already bull/neutral/bear


def drawdown_bear(prices: pd.DataFrame,
                  correction: float = 0.10, bear: float = 0.20) -> pd.DataFrame:
    """Classic 10% correction / 20% bear convention off the running peak."""
    close = prices["close"].astype(float)
    dd = close / close.cummax() - 1.0
    lab = np.where(dd <= -bear, BEAR, np.where(dd <= -correction, NEUTRAL, BULL))
    return _out(prices["date"], lab)


def consec_down(prices: pd.DataFrame, stress_n: int = 5, choppy_n: int = 3) -> pd.DataFrame:
    """Trailing consecutive-negative-close streak (the user-suggested rule)."""
    ret = prices["close"].astype(float).pct_change()
    streak, out = 0, []
    for r in ret:
        if np.isfinite(r) and r < 0:
            streak += 1
        else:
            streak = 0
        out.append(BEAR if streak >= stress_n else NEUTRAL if streak >= choppy_n else BULL)
    return _out(prices["date"], out)


def downday_frac(prices: pd.DataFrame, window: int = 10,
                 stress_f: float = 0.7, choppy_f: float = 0.5) -> pd.DataFrame:
    """Fraction of down days in a rolling window — smoother streak cousin."""
    ret = prices["close"].astype(float).pct_change()
    frac = (ret < 0).rolling(window).mean()
    lab = np.where(frac >= stress_f, BEAR, np.where(frac >= choppy_f, NEUTRAL, BULL))
    return _out(prices["date"], lab)


def vol_percentile(prices: pd.DataFrame, window: int = 20,
                   choppy_q: float = 0.60, stress_q: float = 0.85) -> pd.DataFrame:
    """Realized vol vs its own expanding distribution through t-1 (causal)."""
    ret = np.log(prices["close"].astype(float)).diff()
    vol = ret.rolling(window).std() * np.sqrt(252)
    q_lo = vol.expanding(min_periods=120).quantile(choppy_q).shift(1)
    q_hi = vol.expanding(min_periods=120).quantile(stress_q).shift(1)
    lab = np.where(vol >= q_hi, BEAR, np.where(vol >= q_lo, NEUTRAL, BULL))
    lab = np.where(q_lo.isna(), BULL, lab)          # warm-up: default bull
    return _out(prices["date"], lab)


def vix_threshold(prices: pd.DataFrame, calm_max: float = 20.0,
                  stress_min: float = 30.0) -> pd.DataFrame:
    """Classic VIX cutoffs (<20 bull, 20-30 neutral, >30 bear)."""
    vix = _load_vix()
    d = pd.to_datetime(prices["date"])
    v = vix.reindex(d, method="ffill").to_numpy()
    lab = np.where(v > stress_min, BEAR, np.where(v > calm_max, NEUTRAL, BULL))
    lab = np.where(~np.isfinite(v), BULL, lab)
    return _out(prices["date"], lab)


def mom_12m(prices: pd.DataFrame, band: float = 0.05) -> pd.DataFrame:
    """12-month time-series momentum sign with a +/-5% neutral band."""
    close = prices["close"].astype(float)
    mom = close / close.shift(252) - 1.0
    lab = np.where(mom > band, BULL, np.where(mom < -band, BEAR, NEUTRAL))
    lab = np.where(mom.isna(), NEUTRAL, lab)
    return _out(prices["date"], lab)


def lunde_timmermann(prices: pd.DataFrame, bull_thresh: float = 0.20,
                     bear_thresh: float = 0.15) -> pd.DataFrame:
    """Lunde-Timmermann (2004) first-passage filter — the causal academic
    bull/bear dating rule (baseline thresholds 20%/15%).

    In a bull state, track the running max since the last trough; switch to
    bear when price falls ``bear_thresh`` below it. In a bear state, track the
    running min since the last peak; switch to bull when price rises
    ``bull_thresh`` above it. Two states only: bull and bear.
    """
    close = prices["close"].astype(float).to_numpy()
    state, ext = BULL, close[0] if len(close) else np.nan   # ext: running max/min
    out = []
    for p in close:
        if state == BULL:
            ext = max(ext, p)
            if p <= ext * (1 - bear_thresh):
                state, ext = BEAR, p
        else:
            ext = min(ext, p)
            if p >= ext * (1 + bull_thresh):
                state, ext = BULL, p
        out.append(state)
    return _out(prices["date"], out)


def ret_sign_60(prices: pd.DataFrame, window: int = 60, band: float = 0.05) -> pd.DataFrame:
    """Trailing return-sign rule — the deep-RL trading literature's own
    bull/neutral/bear convention: cumulative return over a trailing window,
    with a +/-band neutral zone."""
    close = prices["close"].astype(float)
    cum = close / close.shift(window) - 1.0
    lab = np.where(cum > band, BULL, np.where(cum < -band, BEAR, NEUTRAL))
    lab = np.where(cum.isna(), NEUTRAL, lab)
    return _out(prices["date"], lab)


HEURISTICS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "ma_crossover": ma_crossover,
    "trend_200": trend_200,
    "drawdown_bear": drawdown_bear,
    "consec_down": consec_down,
    "downday_frac": downday_frac,
    "vol_percentile": vol_percentile,
    "vix_threshold": vix_threshold,
    "mom_12m": mom_12m,
    "lunde_timmermann": lunde_timmermann,
    "ret_sign_60": ret_sign_60,
}
