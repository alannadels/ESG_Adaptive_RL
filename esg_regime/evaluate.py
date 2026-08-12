"""Evaluate the ESG regime detector: chronological train/test + walk-forward.

Two validation protocols, both look-ahead-free:

  TRAIN/TEST  — fit the HMM once on dates < SPLIT_DATE, freeze it, then decode
                the test period (>= SPLIT_DATE) with filtered posteriors. The
                strategy overlay is graded separately on train and test so we can
                see whether the edge survives out of sample.
  WALK-FORWARD — retrain monthly on an expanding window; every bar labeled by a
                model trained only on its past. Uses the whole series.

Strategy overlay on the ESG series:
  bull -> 100% invested   neutral -> 60%   bear -> 0% (cash)
Signals are lagged one day; cash earns a flat rate; turnover is charged.

Data sources:
  esg_index  -> data/esg_index.csv  (real high-ESG basket; build_esg_index.py)
  SUSA / DSI -> downloaded ESG ETFs  (cached to data/<ticker>.csv)
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from esg_regime.regime import (
    MarketRegimeDetector,
    RegimeDetectorConfig,
    split_features_by_date,
)

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

TRADING_DAYS = 252
CASH_ANNUAL = 0.02
COST_BPS = 1.0
WEIGHTS = {"bull": 1.0, "neutral": 0.6, "bear": 0.0}


# --------------------------------------------------------------------------- data
def load_prices(source: str, start: str = "2005-01-01") -> pd.DataFrame:
    """Load a raw OHLC frame for the ESG universe, caching downloads to data/."""
    if source == "esg_index":
        df = pd.read_csv(os.path.join(DATA, "esg_index.csv"), parse_dates=["date"])
        return df
    cache = os.path.join(DATA, f"{source}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, parse_dates=["date"])
    px = yf.download(source, start=start, progress=False, auto_adjust=False)
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(0)
    px = px.rename(columns=str.lower)
    adj = px["adj close"] / px["close"]
    for c in ("open", "high", "low"):
        px[c] = px[c] * adj
    px["close"] = px["adj close"]
    df = px.reset_index().rename(columns={"Date": "date"})[
        ["date", "open", "high", "low", "close", "volume"]]
    df.to_csv(cache, index=False)
    return df


# ----------------------------------------------------------------------- metrics
def perf(returns: pd.Series) -> Dict[str, float]:
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ["CAGR", "Vol", "Sharpe", "Sortino", "MaxDD",
                                    "Calmar", "HitRate"]}
    eq = (1 + r).cumprod()
    cagr = eq.iloc[-1] ** (TRADING_DAYS / len(r)) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = (r.mean() * TRADING_DAYS - CASH_ANNUAL) / vol if vol else np.nan
    dvol = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (r.mean() * TRADING_DAYS - CASH_ANNUAL) / dvol if dvol else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Sortino": sortino,
            "MaxDD": dd, "Calmar": cagr / abs(dd) if dd else np.nan,
            "HitRate": (r > 0).mean()}


def overlay_returns(reg: pd.DataFrame) -> pd.DataFrame:
    d = reg[["date", "close", "regime"]].copy()
    d["asset_ret"] = d["close"].pct_change()
    d["w"] = d["regime"].map(WEIGHTS).shift(1).fillna(1.0)
    cash = CASH_ANNUAL / TRADING_DAYS
    cost = d["w"].diff().abs().fillna(0.0) * (COST_BPS / 1e4)
    d["strat_ret"] = d["w"] * d["asset_ret"] + (1 - d["w"]) * cash - cost
    d["bh_ret"] = d["asset_ret"]
    return d.dropna(subset=["asset_ret"])


def regime_conditional(reg: pd.DataFrame) -> pd.DataFrame:
    """Next-day asset return grouped by the (lagged) regime label."""
    d = reg[["date", "close", "regime"]].copy()
    d["fwd"] = d["close"].pct_change().shift(-1)          # return earned AFTER label
    g = d.dropna(subset=["fwd"]).groupby("regime")["fwd"].agg(["mean", "std", "count"])
    g["ann_ret"] = g["mean"] * TRADING_DAYS
    g["ann_vol"] = g["std"] * np.sqrt(TRADING_DAYS)
    g["sharpe"] = g["ann_ret"] / g["ann_vol"]
    return g.reindex(["bull", "neutral", "bear"])[
        ["ann_ret", "ann_vol", "sharpe", "count"]]


# --------------------------------------------------------------------- reporting
def _fmt(stats: Dict[str, float]) -> str:
    return (f"CAGR {stats['CAGR']*100:6.2f}%  Vol {stats['Vol']*100:6.2f}%  "
            f"Sharpe {stats['Sharpe']:5.2f}  Sortino {stats['Sortino']:5.2f}  "
            f"MaxDD {stats['MaxDD']*100:7.2f}%  Calmar {stats['Calmar']:5.2f}")


def evaluate(source: str, cfg: Optional[RegimeDetectorConfig] = None) -> Dict:
    cfg = cfg or RegimeDetectorConfig()
    print("\n" + "=" * 92)
    print(f"  ESG REGIME EVALUATION — {source}")
    print("=" * 92)
    prices = load_prices(source)
    det = MarketRegimeDetector(cfg)
    feats = det.compute_features(prices)
    print(f"Bars: {len(feats)}  ({feats['date'].min().date()} -> {feats['date'].max().date()})")

    # ---------- PROTOCOL 1: chronological train / test ----------
    train_f, test_f = split_features_by_date(feats, cfg.split_date)
    print(f"\n[TRAIN/TEST split @ {cfg.split_date}]  train={len(train_f)}  test={len(test_f)}")
    det.fit(train_f)
    reg_train = det.predict(train_f)
    reg_test = det.predict(test_f)          # frozen model, filtered decode = OOS

    for tag, reg in (("TRAIN (in-sample)", reg_train), ("TEST  (out-of-sample)", reg_test)):
        ov = overlay_returns(reg)
        mix = reg["regime"].value_counts(normalize=True).reindex(
            ["bull", "neutral", "bear"]).fillna(0)
        print(f"\n  {tag}   regime mix: "
              f"bull {mix['bull']:.0%} / neutral {mix['neutral']:.0%} / bear {mix['bear']:.0%}")
        print(f"    Buy & Hold : {_fmt(perf(ov['bh_ret']))}")
        print(f"    Overlay    : {_fmt(perf(ov['strat_ret']))}")

    print("\n  TEST regime-conditional next-day return (does the OOS label mean anything?)")
    rc = regime_conditional(reg_test)
    print(rc.to_string(float_format=lambda x: f"{x:8.3f}"))

    # ---------- PROTOCOL 2: walk-forward ----------
    print("\n[WALK-FORWARD  retrain monthly, expanding window]")
    reg_wf = det.walk_forward(feats)
    ov = overlay_returns(reg_wf)
    mix = reg_wf["regime"].value_counts(normalize=True).reindex(
        ["bull", "neutral", "bear"]).fillna(0)
    print(f"  regime mix: bull {mix['bull']:.0%} / neutral {mix['neutral']:.0%} / "
          f"bear {mix['bear']:.0%}")
    print(f"    Buy & Hold : {_fmt(perf(ov['bh_ret']))}")
    print(f"    Overlay    : {_fmt(perf(ov['strat_ret']))}")

    # save the walk-forward path + equity for plotting / audit
    out = reg_wf.merge(ov[["date", "w", "strat_ret", "bh_ret"]], on="date", how="left")
    out["strat_equity"] = (1 + out["strat_ret"].fillna(0)).cumprod()
    out["bh_equity"] = (1 + out["bh_ret"].fillna(0)).cumprod()
    out.to_csv(os.path.join(RESULTS, f"{source}_walkforward.csv"), index=False)

    # save a compact summary row set
    summary = pd.DataFrame({
        "train_bh": perf(overlay_returns(reg_train)["bh_ret"]),
        "train_overlay": perf(overlay_returns(reg_train)["strat_ret"]),
        "test_bh": perf(overlay_returns(reg_test)["bh_ret"]),
        "test_overlay": perf(overlay_returns(reg_test)["strat_ret"]),
        "wf_bh": perf(ov["bh_ret"]),
        "wf_overlay": perf(ov["strat_ret"]),
    }).T
    summary.to_csv(os.path.join(RESULTS, f"{source}_summary.csv"))
    return {"reg_wf": reg_wf, "reg_test": reg_test, "summary": summary,
            "regime_conditional_test": rc}


def main() -> None:
    sources = sys.argv[1:] or ["esg_index", "SUSA", "DSI"]
    for s in sources:
        evaluate(s)
    print("\nDone. Per-source CSVs written to results/.")


if __name__ == "__main__":
    main()
