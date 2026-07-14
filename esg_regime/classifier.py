"""3-state ESG regime HMM — pure.

Same architecture as the reference SPY model (TradingAgentV2/v2/regime_core):
  - Gaussian-emission HMM, diagonal covariance, 3 states.
  - Label-pin by REALIZED-VOL mean (lowest->Calm S1 .. highest->Stress S3).
  - FILTERED posteriors (forward algorithm) — point-in-time, no look-ahead.
  - Anti-whipsaw hysteresis + deterministic guardrail
    (close<SMA200 & realized-vol backwardation -> Stress).

`in_sample_regimes(feats)` fits once on full history (optimistic);
`walk_forward_regimes(feats)` is the honest OOS path (retrain monthly on an
expanding window, freeze between, decode filtered).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from esg_regime.features import LOCKED_FEATURES

REGIMES = ["S1_calm", "S2_choppy", "S3_stress"]


@dataclass
class RegimeConfig:
    n_states: int = 3
    n_init: int = 8
    conf_threshold: float = 0.60
    dwell: int = 2
    seed: int = 0
    features: tuple = LOCKED_FEATURES

    @property
    def cols(self) -> list[str]:
        return [f + "_z" for f in self.features]


class RegimeClassifier:
    def __init__(self, cfg: RegimeConfig | None = None):
        self.cfg = cfg or RegimeConfig()
        self.model: GaussianHMM | None = None
        self.state_to_regime: dict[int, str] = {}

    def fit(self, feats: pd.DataFrame) -> "RegimeClassifier":
        X = feats[self.cfg.cols].to_numpy(float)
        best = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for seed in range(self.cfg.n_init):
                m = GaussianHMM(n_components=self.cfg.n_states, covariance_type="diag",
                                n_iter=200, tol=1e-3, random_state=seed)
                try:
                    m.fit(X)
                    ll = m.score(X)
                except Exception:
                    continue
                if best is None or ll > best[0]:
                    best = (ll, m)
        if best is None:
            raise RuntimeError("HMM fit failed on all restarts")
        self.model = best[1]
        states = self.model.predict(X)
        rv = feats["realized_vol"].to_numpy(float)
        means = {s: float(np.nanmean(rv[states == s])) if np.any(states == s) else np.inf
                 for s in range(self.cfg.n_states)}
        order = sorted(means, key=lambda s: means[s])
        self.state_to_regime = {s: REGIMES[rank] for rank, s in enumerate(order)}
        return self

    def _filtered_posteriors(self, X: np.ndarray) -> np.ndarray:
        m = self.model
        eps = 1e-300
        log_start = np.log(m.startprob_ + eps)
        log_trans = np.log(m.transmat_ + eps)
        n = self.cfg.n_states
        log_emit = np.empty((len(X), n))
        for j in range(n):
            log_emit[:, j] = multivariate_normal(
                mean=m.means_[j], cov=m.covars_[j], allow_singular=True).logpdf(X)
        bad = ~np.isfinite(log_emit).any(axis=1)
        log_emit[bad] = 0.0

        def _norm(la: np.ndarray) -> np.ndarray:
            z = logsumexp(la)
            return (la - z) if np.isfinite(z) else np.full(n, -np.log(n))

        out = np.empty((len(X), n))
        la = _norm(log_start + log_emit[0])
        out[0] = la
        for t in range(1, len(X)):
            la = _norm(logsumexp(la[:, None] + log_trans, axis=0) + log_emit[t])
            out[t] = la
        return np.exp(out)

    def predict_regimes(self, feats: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("fit() first")
        post = self._filtered_posteriors(feats[self.cfg.cols].to_numpy(float))
        raw = [self.state_to_regime[s] for s in post.argmax(axis=1)]
        return finalize_regimes(feats, raw, post.max(axis=1),
                                self.cfg.conf_threshold, self.cfg.dwell)


def finalize_regimes(feats: pd.DataFrame, raw_regime: list[str], raw_conf,
                     conf_threshold: float, dwell: int) -> pd.DataFrame:
    """Anti-whipsaw hysteresis + deterministic guardrail. Model-agnostic."""
    raw_conf = np.asarray(raw_conf, float)
    confirmed: list[str] = []
    cur = raw_regime[0]
    run_val, run_len = raw_regime[0], 1
    for i in range(len(raw_regime)):
        if i > 0:
            run_len = run_len + 1 if raw_regime[i] == run_val else 1
            run_val = raw_regime[i]
        if raw_regime[i] != cur and raw_conf[i] >= conf_threshold and run_len >= dwell:
            cur = raw_regime[i]
        confirmed.append(cur)
    out = feats[["date", "close", "sma200", "rv_term_structure"]].copy().reset_index(drop=True)
    out["regime_raw"] = raw_regime
    out["confidence"] = raw_conf.round(3)
    out["regime"] = confirmed
    # ESG guardrail: below the 200d trend AND short-vol elevated vs long-vol
    guard = (out["close"] < out["sma200"]) & (out["rv_term_structure"] > 0)
    out.loc[guard, "regime"] = "S3_stress"
    out["guardrail"] = guard
    return out


def in_sample_regimes(feats: pd.DataFrame, cfg: RegimeConfig | None = None) -> pd.DataFrame:
    """Fit once on full history, decode filtered (optimistic — in-sample)."""
    return RegimeClassifier(cfg).fit(feats).predict_regimes(feats)


def walk_forward_regimes(feats: pd.DataFrame, cfg: RegimeConfig | None = None,
                         min_train: int = 252, retrain_every: int = 21) -> pd.DataFrame:
    """Out-of-sample: retrain monthly on an expanding window, freeze between,
    decode filtered. No bar is labeled by a model that trained on it."""
    feats = feats.reset_index(drop=True)
    n = len(feats)
    cols = (cfg or RegimeConfig()).cols
    if n <= min_train:
        return in_sample_regimes(feats, cfg)
    raw_regime: list[str | None] = [None] * n
    raw_conf = np.zeros(n)

    def _label(train_end: int, decode_end: int) -> None:
        clf = RegimeClassifier(cfg).fit(feats.iloc[:train_end])
        post = clf._filtered_posteriors(feats.iloc[:decode_end][cols].to_numpy(float))
        st, cf = post.argmax(1), post.max(1)
        lo = 0 if train_end == min_train else train_end
        for i in range(lo, decode_end):
            raw_regime[i] = clf.state_to_regime[st[i]]
            raw_conf[i] = cf[i]

    _label(min_train, min_train)
    r = min_train
    while r < n:
        end = min(r + retrain_every, n)
        _label(r, end)
        r = end
    c = cfg or RegimeConfig()
    return finalize_regimes(feats, [x or "S1_calm" for x in raw_regime], raw_conf,
                            c.conf_threshold, c.dwell)
