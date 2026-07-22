"""Compare regime HMMs fit on published ESG indices vs the score-constructed index.

Question: does the regime model need a bespoke, ESG-score-constructed universe, or
does fitting the same HMM directly on an off-the-shelf ESG index work just as well?

For every ESG index in the panel we run the IDENTICAL pipeline (features -> 3-state
HMM -> walk-forward labeling -> overlay backtest) and report:

  * regime quality  — the OOS stress/calm volatility ratio (the model's real job:
                      separating volatility states);
  * economic value  — overlay vs buy & hold (Sharpe, max drawdown);
  * agreement       — how often each index's regime label matches the
                      score-constructed esg_index label on shared dates.

Indices are processed in parallel (one process each) since each is independent.
"""
from __future__ import annotations

import os
import sys
import warnings
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
RESULTS = os.path.join(HERE, "results")

from esg_regime.evaluate import (  # noqa: E402
    load_prices, overlay_returns, perf, regime_conditional,
)
from esg_regime.regime import MarketRegimeDetector, RegimeDetectorConfig  # noqa: E402

# ESG index panel: ticker -> (name, style)
PANEL = {
    "esg_index": ("Score-constructed top-40 ESG basket", "constructed"),
    "SUSA": ("iShares MSCI USA ESG Select", "broad ESG"),
    "DSI": ("iShares MSCI KLD 400 Social", "values-screened"),
    "ERTH": ("Invesco MSCI Sustainable Future", "environmental"),
    "ICLN": ("iShares Global Clean Energy", "clean-energy thematic"),
    "CRBN": ("iShares MSCI ACWI Low Carbon", "low carbon (global)"),
    "SPYX": ("SPDR S&P 500 Fossil Fuel Free", "fossil-fuel free"),
    "ESGU": ("iShares ESG Aware MSCI USA", "broad ESG"),
    "ESGD": ("iShares ESG Aware MSCI EAFE", "international ESG"),
    "NULV": ("Nuveen ESG Large-Cap Value", "ESG value"),
    "ESGV": ("Vanguard ESG US Stock", "broad ESG"),
    "SUSL": ("iShares ESG MSCI USA Leaders", "ESG leaders"),
}


def run_one(ticker: str) -> dict:
    """Fit + evaluate the regime model on a single ESG index."""
    try:
        prices = load_prices(ticker)
        det = MarketRegimeDetector(RegimeDetectorConfig())
        feats = det.compute_features(prices)
        if len(feats) < 600:
            return {"ticker": ticker, "error": f"only {len(feats)} usable bars"}

        reg = det.walk_forward(feats)
        ov = overlay_returns(reg)
        rc = regime_conditional(reg)

        bh, st = perf(ov["bh_ret"]), perf(ov["strat_ret"])
        calm_vol = rc.loc["S1_calm", "ann_vol"] if "S1_calm" in rc.index else np.nan
        stress_vol = rc.loc["S3_stress", "ann_vol"] if "S3_stress" in rc.index else np.nan
        mix = reg["regime"].value_counts(normalize=True)

        # persist the label path so we can measure cross-index agreement
        reg[["date", "regime"]].to_csv(
            os.path.join(RESULTS, f"labels_{ticker}.csv"), index=False)

        return {
            "ticker": ticker, "name": PANEL[ticker][0], "style": PANEL[ticker][1],
            "bars": len(feats),
            "start": str(feats["date"].min().date()), "end": str(feats["date"].max().date()),
            "calm_vol": calm_vol, "stress_vol": stress_vol,
            "vol_ratio": stress_vol / calm_vol if calm_vol else np.nan,
            "stress_pct": mix.get("S3_stress", 0.0),
            "bh_sharpe": bh["Sharpe"], "ov_sharpe": st["Sharpe"],
            "d_sharpe": st["Sharpe"] - bh["Sharpe"],
            "bh_dd": bh["MaxDD"], "ov_dd": st["MaxDD"],
            "dd_cut": 1 - abs(st["MaxDD"]) / abs(bh["MaxDD"]) if bh["MaxDD"] else np.nan,
            "bh_cagr": bh["CAGR"], "ov_cagr": st["CAGR"],
        }
    except Exception as exc:  # keep the panel running if one ticker fails
        return {"ticker": ticker, "error": str(exc)[:120]}


def agreement() -> pd.DataFrame:
    """% of shared dates where each index's regime label matches esg_index's."""
    base_path = os.path.join(RESULTS, "labels_esg_index.csv")
    if not os.path.exists(base_path):
        return pd.DataFrame()
    base = pd.read_csv(base_path, parse_dates=["date"]).set_index("date")["regime"]
    rows = []
    for t in PANEL:
        if t == "esg_index":
            continue
        p = os.path.join(RESULTS, f"labels_{t}.csv")
        if not os.path.exists(p):
            continue
        s = pd.read_csv(p, parse_dates=["date"]).set_index("date")["regime"]
        j = pd.concat([base.rename("a"), s.rename("b")], axis=1).dropna()
        if len(j) < 100:
            continue
        rows.append({"ticker": t, "shared_days": len(j),
                     "exact_match": (j.a == j.b).mean(),
                     "stress_both": ((j.a == "S3_stress") & (j.b == "S3_stress")).sum()
                     / max(1, (j.a == "S3_stress").sum())})
    return pd.DataFrame(rows).sort_values("exact_match", ascending=False)


def main() -> None:
    tickers = list(PANEL)
    print(f"Fitting the regime HMM on {len(tickers)} ESG indices (parallel)...\n")
    with Pool(processes=min(6, os.cpu_count() or 4)) as pool:
        out = pool.map(run_one, tickers)

    ok = [r for r in out if "error" not in r]
    bad = [r for r in out if "error" in r]
    df = pd.DataFrame(ok)
    df.to_csv(os.path.join(RESULTS, "index_comparison.csv"), index=False)

    # ---------- regime quality ----------
    print("=" * 108)
    print("  REGIME QUALITY — does the HMM separate volatility states on each ESG index?")
    print("=" * 108)
    q = df.sort_values("vol_ratio", ascending=False)
    print(f"{'Index':<10}{'Style':<24}{'Bars':>6}{'Calm vol':>10}{'Stress vol':>12}"
          f"{'Ratio':>8}{'Stress%':>9}")
    for _, r in q.iterrows():
        print(f"{r.ticker:<10}{r.style:<24}{r.bars:>6}{r.calm_vol*100:>9.1f}%"
              f"{r.stress_vol*100:>11.1f}%{r.vol_ratio:>8.2f}{r.stress_pct*100:>8.0f}%")

    # ---------- economic value ----------
    print("\n" + "=" * 108)
    print("  ECONOMIC VALUE — regime overlay vs buy & hold (walk-forward, net of costs)")
    print("=" * 108)
    e = df.sort_values("d_sharpe", ascending=False)
    print(f"{'Index':<10}{'B&H Sharpe':>12}{'Ovl Sharpe':>12}{'ΔSharpe':>10}"
          f"{'B&H MaxDD':>12}{'Ovl MaxDD':>12}{'DD cut':>9}")
    for _, r in e.iterrows():
        print(f"{r.ticker:<10}{r.bh_sharpe:>12.2f}{r.ov_sharpe:>12.2f}{r.d_sharpe:>+10.2f}"
              f"{r.bh_dd*100:>11.1f}%{r.ov_dd*100:>11.1f}%{r.dd_cut*100:>8.0f}%")

    # ---------- headline comparison ----------
    print("\n" + "=" * 108)
    print("  CONSTRUCTED BASKET vs PUBLISHED ESG INDICES")
    print("=" * 108)
    con = df[df.ticker == "esg_index"]
    pub = df[df.ticker != "esg_index"]
    if len(con):
        c = con.iloc[0]
        print(f"  Score-constructed basket : vol ratio {c.vol_ratio:.2f} | "
              f"ΔSharpe {c.d_sharpe:+.2f} | DD cut {c.dd_cut*100:.0f}%")
    print(f"  Published ESG indices    : vol ratio {pub.vol_ratio.mean():.2f} "
          f"(median {pub.vol_ratio.median():.2f}, range {pub.vol_ratio.min():.2f}"
          f"–{pub.vol_ratio.max():.2f}) | ΔSharpe {pub.d_sharpe.mean():+.2f} "
          f"| DD cut {pub.dd_cut.mean()*100:.0f}%")
    print(f"  Indices where the overlay IMPROVES Sharpe : "
          f"{(pub.d_sharpe > 0).sum()}/{len(pub)}")
    print(f"  Indices where the overlay CUTS drawdown   : "
          f"{(pub.dd_cut > 0).sum()}/{len(pub)}")

    # ---------- label agreement ----------
    ag = agreement()
    if len(ag):
        print("\n" + "=" * 108)
        print("  LABEL AGREEMENT with the score-constructed basket (shared dates)")
        print("=" * 108)
        print(f"{'Index':<10}{'Shared days':>13}{'Exact match':>14}{'Stress recall':>15}")
        for _, r in ag.iterrows():
            print(f"{r.ticker:<10}{r.shared_days:>13}{r.exact_match*100:>13.0f}%"
                  f"{r.stress_both*100:>14.0f}%")
        ag.to_csv(os.path.join(RESULTS, "index_agreement.csv"), index=False)

    if bad:
        print("\nSkipped:", ", ".join(f"{b['ticker']} ({b['error']})" for b in bad))
    print("\nSaved results/index_comparison.csv + results/index_agreement.csv")


if __name__ == "__main__":
    main()
