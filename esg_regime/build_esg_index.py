"""Build a high-ESG price index from the repo's real Refinitiv ESG scores.

The regime model detects when the *ESG universe* is in a stress state. To make
that universe genuinely ESG-driven (rather than a third-party ESG ETF), we:

  1. read ``Dataset/ESG_2000-2026.csv`` (real Refinitiv ESG scores);
  2. keep liquid US large-caps and rank by current ESG score (FY0);
  3. take the top-``TOP_N`` names as the sustainable universe;
  4. download their adjusted prices and form an equal-weight daily index;
  5. reconstruct a synthetic OHLC bar for the index (so the OHLC range-vol
     features still work) and cache it to ``data/esg_index.csv``.

The cached CSV is the self-contained input the regime pipeline runs on, so the
whole thing is reproducible offline once built.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ESG_CSV = os.path.join(REPO, "Dataset", "ESG_2000-2026.csv")
OUT_CSV = os.path.join(HERE, "data", "esg_index.csv")
UNIVERSE_CSV = os.path.join(HERE, "data", "esg_universe.csv")

TOP_N = 40
MIN_MKT_CAP = 5e9
START = "2010-01-01"


def ric_to_yahoo(ric: str) -> str:
    """Map a Refinitiv RIC (e.g. 'NVDA.OQ', 'JPM.N') to a Yahoo ticker.

    The exchange suffix after the dot is dropped; Yahoo uses the bare root symbol
    for US listings. Class shares (dots inside the root) are left untouched.
    """
    return ric.split(".")[0].replace(" ", "")


def build_universe() -> pd.DataFrame:
    df = pd.read_csv(ESG_CSV)
    df.columns = [c.replace("\n", " ").strip() for c in df.columns]
    df = df.rename(columns={
        "Identifier (RIC)": "ric", "Company Name": "name",
        "Country of Exchange": "country", "GICS Industry Name": "industry",
        "Company Market Capitalization (USD)": "mkt_cap", "ESG Score (FY0)": "esg",
    })
    df = df[df["country"] == "United States of America"].copy()
    df["mkt_cap"] = pd.to_numeric(df["mkt_cap"], errors="coerce")
    df["esg"] = pd.to_numeric(df["esg"], errors="coerce")
    df = df.dropna(subset=["esg", "mkt_cap"])
    df = df[df["mkt_cap"] >= MIN_MKT_CAP]
    df["ticker"] = df["ric"].map(ric_to_yahoo)
    df = df.sort_values("esg", ascending=False).head(TOP_N).reset_index(drop=True)
    return df[["ticker", "name", "industry", "mkt_cap", "esg"]]


def build_index(uni: pd.DataFrame) -> pd.DataFrame:
    tickers = uni["ticker"].tolist()
    print(f"Downloading {len(tickers)} high-ESG names since {START} ...")
    px = yf.download(tickers, start=START, auto_adjust=True, progress=False)
    close = px["Close"] if isinstance(px.columns, pd.MultiIndex) else px
    close = close.dropna(axis=1, how="all")
    # keep names with a reasonably complete history
    good = close.columns[close.notna().mean() > 0.9]
    close = close[good].dropna(how="any")
    print(f"Retained {close.shape[1]} names with full overlapping history: "
          f"{close.index.min().date()} -> {close.index.max().date()}")

    # equal-weight daily rebalanced index -> level series
    rets = close.pct_change().dropna(how="any")
    idx_ret = rets.mean(axis=1)                     # equal weight
    level = 100.0 * (1 + idx_ret).cumprod()
    level.iloc[0] = 100.0

    # cross-sectional daily high/low across constituents (normalised to the index
    # level) gives a genuine intraday range proxy for the OHLC vol estimators.
    hi = (close / close.shift(1)).max(axis=1).reindex(level.index)
    lo = (close / close.shift(1)).min(axis=1).reindex(level.index)
    prev = level.shift(1)
    out = pd.DataFrame({
        "date": level.index,
        "open": prev.fillna(level),
        "high": np.maximum(level, prev.fillna(level) * hi.fillna(1.0)),
        "low": np.minimum(level, prev.fillna(level) * lo.fillna(1.0)),
        "close": level.values,
        "volume": 0.0,
    }).reset_index(drop=True)
    return out


def main() -> None:
    uni = build_universe()
    uni.to_csv(UNIVERSE_CSV, index=False)
    print(f"Top {len(uni)} ESG names (mean score {uni['esg'].mean():.1f}):")
    print(uni[["ticker", "name", "esg"]].head(15).to_string(index=False))
    idx = build_index(uni)
    idx.to_csv(OUT_CSV, index=False)
    print(f"\nSaved index ({len(idx)} bars) -> {OUT_CSV}")
    print(f"Saved universe -> {UNIVERSE_CSV}")


if __name__ == "__main__":
    main()
