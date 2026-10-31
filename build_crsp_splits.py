"""Build the CRSP+tresg survivorship-free regime-split pickle for the sweep.

Produces the exact artifact the sweep worker consumes -- a ``{regime: MarketData}``
mapping cached at ``Dataset/cache/regime_splits_crsp.pkl`` -- but sourced entirely from
WRDS CRSP (delisting-inclusive daily returns) + Refinitiv ``tresg`` (ESG pillars) via the
cached pulls in ``wrds_data/`` and the point-in-time ``universe_snapshots_crsp.csv``.

Because ``evolve_regimes_worker.load_or_build_splits`` loads an existing cache pickle
without rebuilding, pre-building this file lets the standard sweep run on survivorship-free
data with no change to the worker: it never touches yfinance.

Mirrors ``load_snapshot_market_data``: superset of every yearly snapshot, a ``(T, N)``
``tradable`` mask (reused ``snapshot_tradability``), returns zero-filled where masked, and
E/S/G daily arrays built with the same publication-lag convention (a score for calendar
year ``c`` is known from June of ``c+1``) but on tresg's already-[0,1] pillar scores.
Regimes are labelled on SPY with the identical causal 50/200 crossover detector, so the
labels match the yfinance snapshot run and only the universe data differs.

Run from the repository root:

    python build_crsp_splits.py --detector crossover
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from esg_adaptive_rl import config as base_config
from esg_adaptive_rl.data import MarketData, load_index_close, snapshot_tradability
from esg_adaptive_rl.regimes import RegimeConfig, label_regimes, split_by_regime

WRDS_DIR = os.environ.get("WRDS_DATA_DIR", "wrds_data")  # relative to repo root; override via env
SNAPSHOT_CSV = "Dataset/universe_snapshots_crsp.csv"
OUT_PICKLE = "Dataset/cache/regime_splits_crsp_{detector}.pkl"
DATE_START, DATE_END = "2010-01-01", "2024-12-31"
REGIME_INDEX = "SPY"


def _regime_cfg(detector: str) -> RegimeConfig:
    """Map a study detector key to a dev RegimeConfig (ema = ema-crossover; hmm = walk-forward)."""
    if detector == "crossover":
        return RegimeConfig(ticker=REGIME_INDEX, detector="crossover", ma_type="sma")
    if detector == "ema":
        return RegimeConfig(ticker=REGIME_INDEX, detector="crossover", ma_type="ema")
    if detector == "hmm":
        return RegimeConfig(ticker=REGIME_INDEX, detector="hmm")
    raise ValueError(f"unknown detector {detector!r}; expected crossover|ema|hmm")


def _rep_ticker(names: pd.DataFrame) -> pd.Series:
    """permno -> most-recent CRSP ticker (same map used by the snapshot builder)."""
    n = names.dropna(subset=["ticker"]).sort_values(["permno", "namedt"])
    return n.groupby("permno").tail(1).set_index("permno")["ticker"]


def _effective_esg_year(dates: pd.DatetimeIndex) -> np.ndarray:
    """Newest ESG fiscal year usable on each date (published June of c+1, no look-ahead)."""
    y = dates.year.to_numpy()
    return np.where(dates.month.to_numpy() >= 6, y - 1, y - 2)


def _build_returns(superset, rep, dates):
    """(T, N) return matrix for the superset tickers; NaN where no quote that day."""
    ret = pd.read_parquet(f"{WRDS_DIR}/daily_returns.parquet", columns=["permno", "date", "ret"])
    ret["date"] = pd.to_datetime(ret["date"])
    ret = ret[(ret.date >= dates[0]) & (ret.date <= dates[-1])].copy()
    ret["ticker"] = ret["permno"].map(rep)
    ret = ret.dropna(subset=["ticker"])
    wide = ret.pivot_table(index="date", columns="ticker", values="ret", aggfunc="first")
    wide = wide.reindex(index=dates, columns=superset)
    return wide.to_numpy(dtype=np.float64)


def _build_esg(superset, rep, names, dates):
    """{E,S,G} daily (T, N) arrays from tresg pillars with publication-lag + mean-fill."""
    esg = pd.read_csv(f"{WRDS_DIR}/esg_panel.csv", dtype={"cusip9": str})
    cusip2permno = (names.dropna(subset=["cusip9"]).drop_duplicates("cusip9")
                    .set_index("cusip9")["permno"])
    esg["permno"] = esg["cusip9"].map(cusip2permno)
    esg["ticker"] = esg["permno"].map(rep)
    esg = esg.dropna(subset=["ticker"])
    eff = _effective_esg_year(dates)                      # newest usable FY per day
    col_of = {t: j for j, t in enumerate(superset)}
    out = {}
    for pillar in ("E", "S", "G"):
        wide = (esg.pivot_table(index="ticker", columns="year", values=pillar, aggfunc="mean")
                .reindex(index=superset))
        wide = wide.sort_index(axis=1).ffill(axis=1)       # latest-known-up-to-each-year
        years = list(wide.columns)
        matrix = np.full((len(dates), len(superset)), np.nan)
        for t, ey in enumerate(eff):
            usable = [y for y in years if y <= ey]
            if not usable:
                continue
            matrix[t, :] = wide[usable[-1]].to_numpy()
        # fill names/dates with no score by the daily cross-sectional mean (-> 0.5 fallback)
        row_means = np.nanmean(matrix, axis=1, keepdims=True)
        row_means = np.where(np.isfinite(row_means), row_means, 0.5)
        matrix = np.where(np.isnan(matrix), row_means, matrix)
        out[pillar] = np.clip(matrix, 0.0, 1.0)
    return out


def main() -> None:
    """Assemble the survivorship-free MarketData, label regimes on SPY, split, and pickle."""
    import argparse
    ap = argparse.ArgumentParser(description="Build CRSP survivorship-free regime splits for one detector.")
    ap.add_argument("--detector", choices=("crossover", "ema", "hmm"), default="crossover")
    args = ap.parse_args()
    out_pickle = OUT_PICKLE.format(detector=args.detector)

    names = pd.read_csv(f"{WRDS_DIR}/names.csv", dtype={"cusip9": str}, parse_dates=["namedt", "nameenddt"])
    rep = _rep_ticker(names)
    snapshots = pd.read_csv(SNAPSHOT_CSV)
    superset = sorted(snapshots["ticker"].astype(str).unique())

    # trading calendar = CRSP trading days present in the window
    cal = pd.read_parquet(f"{WRDS_DIR}/daily_returns.parquet", columns=["date"])
    cal["date"] = pd.to_datetime(cal["date"])
    dates = pd.DatetimeIndex(sorted(cal[(cal.date >= DATE_START) & (cal.date <= DATE_END)]["date"].unique()))

    print(f"superset {len(superset)} tickers; {len(dates)} trading days {dates[0].date()}..{dates[-1].date()}")
    returns = _build_returns(superset, rep, dates)
    esg = _build_esg(superset, rep, names, dates)
    tradable = snapshot_tradability(snapshots, superset, dates, returns)
    returns = np.where(np.isnan(returns), 0.0, returns)     # mask makes these unreachable

    data = MarketData(dates=dates, tickers=superset, returns=returns, esg=esg, tradable=tradable)

    # Regimes on SPY with the requested detector (same causal labelers as the main study).
    index_prices = load_index_close(REGIME_INDEX, base_config.START_DATE, base_config.END_DATE)
    labels = label_regimes(index_prices, _regime_cfg(args.detector))
    splits = split_by_regime(data, labels)

    os.makedirs(os.path.dirname(out_pickle), exist_ok=True)
    with open(out_pickle, "wb") as fh:
        pickle.dump(splits, fh)

    print(f"\n[{args.detector}] wrote {out_pickle}")
    for reg, md in splits.items():
        trd = md.tradable
        avg_tradable = float(trd.sum(axis=1).mean()) if trd is not None else float("nan")
        print(f"  {reg:<8} days={md.returns.shape[0]:>5}  assets={md.returns.shape[1]}  "
              f"avg tradable names/day={avg_tradable:.1f}")
    print(f"\nverification: mask avg should be ~50 (the per-year universe size); "
          f"superset {len(superset)} > 50 confirms names enter/exit across years (survivorship-free).")


if __name__ == "__main__":
    main()
