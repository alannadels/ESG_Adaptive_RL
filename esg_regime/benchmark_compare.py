"""Heuristics vs HMM, and ESG universes vs broad benchmarks (SPX/QQQ/SPY).

Three questions, one harness:

  1. HEURISTICS — how do standard rule-based regime definitions (incl. the
     repo's production 50/200 crossover and the user-suggested consecutive-
     down-days rule) compare to the walk-forward HMM, on the same universes,
     by the same metrics?
  2. BENCHMARKS — does the same machinery detect regimes on broad indices
     (^GSPC, QQQ, SPY) as well as it does on ESG ETFs? Are ESG regimes just
     market regimes?
  3. TRANSFER — the repo's ``evolve_regimes.py`` labels regimes on SPY and
     applies them to the ESG dataset. Does a SPY-detected label work as well
     on an ESG universe as that universe's own label?

Metrics per (universe, detector): state mix, switches/year, next-day vol per
state (point-in-time labels -> honest), stress/calm vol ratio, 100/60/0
overlay vs buy & hold (Sharpe, MaxDD), and agreement with the HMM label.

Usage:  python -m esg_regime.benchmark_compare
"""
from __future__ import annotations

import os
import sys
import warnings
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
RESULTS = os.path.join(HERE, "results")

from esg_regime.evaluate import load_prices, overlay_returns, perf  # noqa: E402
from esg_regime.heuristics import HEURISTICS, REGIME_ORDER  # noqa: E402
from esg_regime.regime import MarketRegimeDetector, RegimeDetectorConfig  # noqa: E402

BULL, NEUTRAL, BEAR = REGIME_ORDER
ESG_UNIVERSES = ["SUSA", "DSI", "ESGU"]
BENCHMARKS = ["^GSPC", "QQQ", "SPY"]
UNIVERSES = ESG_UNIVERSES + BENCHMARKS
TRADING_DAYS = 252


# ------------------------------------------------------------------ hmm labels
def hmm_labels(ticker: str) -> pd.DataFrame:
    """Walk-forward HMM labels for a ticker, cached to results/labels_<t>.csv."""
    safe = ticker.replace("^", "")
    cache = os.path.join(RESULTS, f"labels_{safe if ticker.startswith('^') else ticker}.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache, parse_dates=["date"])
    prices = load_prices(ticker)
    det = MarketRegimeDetector(RegimeDetectorConfig())
    reg = det.walk_forward(det.compute_features(prices))
    out = reg[["date", "regime"]].copy()
    out.to_csv(cache, index=False)
    return out


def _hmm_worker(t: str) -> str:
    try:
        hmm_labels(t)
        return f"{t}: ok"
    except Exception as e:  # pragma: no cover
        return f"{t}: ERROR {e}"


# -------------------------------------------------------------------- metrics
def grade(prices: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Grade one (universe prices, label path) pair."""
    d = prices[["date", "close"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    lab = labels.copy()
    lab["date"] = pd.to_datetime(lab["date"])
    d = d.merge(lab, on="date", how="inner").dropna(subset=["regime"])
    if len(d) < 300:
        return {}

    # state mix / churn / per-state day counts
    mix = d["regime"].value_counts(normalize=True)
    counts = d["regime"].value_counts()
    years = len(d) / TRADING_DAYS
    switches = (d["regime"] != d["regime"].shift(1)).sum() - 1

    # next-day vol/return per state (label at t, return t->t+1)
    d["fwd"] = d["close"].pct_change().shift(-1)
    g = d.dropna(subset=["fwd"]).groupby("regime")["fwd"]
    vol = (g.std() * np.sqrt(TRADING_DAYS)).reindex(REGIME_ORDER)
    ret = (g.mean() * TRADING_DAYS).reindex(REGIME_ORDER)

    # overlay backtest (100/60/0, lagged, costs) via the shared harness
    ov = overlay_returns(d.rename(columns={"regime": "regime"})[["date", "close", "regime"]])
    bh, st = perf(ov["bh_ret"]), perf(ov["strat_ret"])

    return {
        "days": len(d),
        "calm_pct": mix.get(BULL, 0.0), "choppy_pct": mix.get(NEUTRAL, 0.0),
        "stress_pct": mix.get(BEAR, 0.0),
        "stress_days": int(counts.get(BEAR, 0)),
        "switches_per_year": switches / years if years else np.nan,
        "calm_vol": vol[BULL], "stress_vol": vol[BEAR],
        "vol_ratio": vol[BEAR] / vol[BULL] if vol[BULL] and np.isfinite(vol[BULL]) else np.nan,
        "stress_ann_ret": ret[BEAR],
        "bh_sharpe": bh["Sharpe"], "ov_sharpe": st["Sharpe"],
        "d_sharpe": st["Sharpe"] - bh["Sharpe"],
        "bh_dd": bh["MaxDD"], "ov_dd": st["MaxDD"],
        "dd_cut": 1 - abs(st["MaxDD"]) / abs(bh["MaxDD"]) if bh["MaxDD"] else np.nan,
    }


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """% of shared dates with identical labels."""
    j = a.merge(b, on="date", suffixes=("_a", "_b")).dropna()
    return (j["regime_a"] == j["regime_b"]).mean() if len(j) else np.nan


# ---------------------------------------------------------------------- main
def main() -> None:
    # 1) ensure HMM labels exist for every universe (parallel where missing)
    missing = [t for t in UNIVERSES
               if not os.path.exists(os.path.join(
                   RESULTS, f"labels_{t.replace('^', '')}.csv"))]
    if missing:
        print(f"Fitting walk-forward HMM on: {missing}")
        with Pool(processes=min(len(missing), 3)) as pool:
            for msg in pool.map(_hmm_worker, missing):
                print(" ", msg)

    # 2) grade the full grid
    rows, label_store = [], {}
    for uni in UNIVERSES:
        prices = load_prices(uni)
        prices = prices[pd.to_datetime(prices["date"]) >= "2005-01-01"]
        labs = {"hmm": hmm_labels(uni)}
        for name, fn in HEURISTICS.items():
            try:
                labs[name] = fn(prices)
            except Exception as e:
                print(f"  {uni}/{name} failed: {e}")
        label_store[uni] = labs
        for det, lab in labs.items():
            m = grade(prices, lab)
            if m:
                m.update({"universe": uni, "detector": det,
                          "agree_hmm": agreement(lab, labs["hmm"])})
                rows.append(m)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "heuristics_benchmarks.csv"), index=False)

    # 3) transfer test: SPY labels applied to ESG universes (repo design)
    trows = []
    for target in ["SUSA", "ESGU"]:
        tp = load_prices(target)
        tp = tp[pd.to_datetime(tp["date"]) >= "2005-01-01"]
        for det in ["hmm", "ma_crossover", "trend_200", "drawdown_bear", "vol_percentile"]:
            own = label_store[target].get(det)
            spy = label_store["SPY"].get(det)
            if own is None or spy is None:
                continue
            # as-of forward-fill SPY labels onto the target's dates (repo method)
            s = spy.set_index("date")["regime"]
            aligned = s.reindex(pd.to_datetime(tp["date"]), method="ffill")
            spy_on_target = pd.DataFrame({"date": tp["date"].values,
                                          "regime": aligned.values})
            m_own, m_spy = grade(tp, own), grade(tp, spy_on_target)
            if m_own and m_spy:
                trows.append({
                    "target": target, "detector": det,
                    "own_vol_ratio": m_own["vol_ratio"], "spy_vol_ratio": m_spy["vol_ratio"],
                    "own_d_sharpe": m_own["d_sharpe"], "spy_d_sharpe": m_spy["d_sharpe"],
                    "own_dd_cut": m_own["dd_cut"], "spy_dd_cut": m_spy["dd_cut"],
                    "label_agreement": agreement(own, spy_on_target),
                })
    tdf = pd.DataFrame(trows)
    tdf.to_csv(os.path.join(RESULTS, "transfer_test.csv"), index=False)

    # 4) report
    pd.set_option("display.width", 200)
    print("\n" + "=" * 112)
    print("  DETECTOR GRID — stress/calm vol ratio | overlay dSharpe | drawdown cut | switches/yr | stress days")
    print("=" * 112)
    for uni in UNIVERSES:
        sub = df[df.universe == uni].sort_values("vol_ratio", ascending=False)
        print(f"\n--- {uni} ---")
        print(f"{'detector':<16}{'volratio':>9}{'dSharpe':>9}{'ddcut':>8}{'sw/yr':>8}"
              f"{'stress%':>9}{'stressN':>9}{'agreeHMM':>10}")
        for _, r in sub.iterrows():
            print(f"{r.detector:<16}{r.vol_ratio:>9.2f}{r.d_sharpe:>+9.2f}"
                  f"{r.dd_cut*100:>7.0f}%{r.switches_per_year:>8.1f}"
                  f"{r.stress_pct*100:>8.1f}%{r.stress_days:>9d}{r.agree_hmm*100:>9.0f}%")

    print("\n" + "=" * 112)
    print("  TRANSFER TEST — regime labels detected on SPY, applied to ESG universes (the evolve_regimes.py design)")
    print("=" * 112)
    print(f"{'target':<7}{'detector':<15}{'own volratio':>13}{'SPY volratio':>13}"
          f"{'own dSh':>9}{'SPY dSh':>9}{'own ddcut':>10}{'SPY ddcut':>10}{'agree':>7}")
    for _, r in tdf.iterrows():
        print(f"{r.target:<7}{r.detector:<15}{r.own_vol_ratio:>13.2f}{r.spy_vol_ratio:>13.2f}"
              f"{r.own_d_sharpe:>+9.2f}{r.spy_d_sharpe:>+9.2f}"
              f"{r.own_dd_cut*100:>9.0f}%{r.spy_dd_cut*100:>9.0f}%{r.label_agreement*100:>6.0f}%")

    print("\nSaved results/heuristics_benchmarks.csv + results/transfer_test.csv")


if __name__ == "__main__":
    main()
