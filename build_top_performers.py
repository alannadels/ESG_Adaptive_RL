"""Rebuild correct ESG and returns top-performer tables, and intersect them.

Fixes the data-quality problems in the exploratory workbook (e.g. NVIDIA missing from
the returns rankings) by rebuilding both tables from reliable, complete sources over a
single consistent universe:

    - Universe : current S&P 500 constituents (Wikipedia) that also have Refinitiv ESG
                 scores, with each name's GICS sector.
    - Returns  : computed from adjusted prices (yfinance) — calendar-year total returns.
    - ESG      : the Refinitiv overall ESG score per fiscal year.

For each year and sector it ranks the top-N names by return and by ESG, writes both as
CSVs, then finds the tickers that are *consistently* top in BOTH — the starting point for
the portfolio universe.

Caveats:
    - Uses *current* S&P 500 membership, so it carries survivorship bias (names dropped
      from the index over the years are absent). Appropriate for picking a currently
      investable universe; not a point-in-time historical study.
    - Refinitiv ESG coverage is only solid for roughly the last eight years, so the
      intersection window is restricted to where both signals are reliable.

Run from the repository root:

    python build_top_performers.py
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

SP500_PATH = "Dataset/sp500_constituents.csv"      # ticker, sector (built in a prior step)
ESG_PATH = "Dataset/ESG_2000-26-ESGC.csv"
# S&P/yfinance ticker -> Refinitiv ESG ticker, bridging share-class string mismatches
# (e.g. Berkshire trades as BRK-B but Refinitiv lists it under the Class-A RIC BRKa;
# ESG is a company-level score, so the class does not matter).
TICKER_ALIAS = {"BRK-B": "BRKa", "BF-B": "BFb"}
ESG_ANCHOR_YEAR = 2025                              # calendar year that FY0 maps to
OUTPUT_DIR = "Dataset"
TOP_N = 5                                           # ranked per sector per year
YEARS = list(range(2017, 2026))                    # window where ESG + returns are both reliable
MIN_APPEARANCES = 3                                 # min top-N years in a signal to count as "consistent"


def load_universe() -> pd.DataFrame:
    """Load the S&P 500 names that also carry Refinitiv ESG, with sector and RIC ticker.

    Returns:
        A DataFrame indexed by ticker with a ``sector`` column.
    """
    sp = pd.read_csv(SP500_PATH)
    esg = pd.read_csv(ESG_PATH)
    esg.columns = [str(c).replace("\n", " ").strip() for c in esg.columns]
    esg["ticker"] = esg["Identifier (RIC)"].str.split(".").str[0]
    have_esg = set(esg["ticker"])
    # Also keep names whose ESG lives under an aliased (share-class) ticker.
    have_esg |= {sp_t for sp_t, ric in TICKER_ALIAS.items() if ric in set(esg["ticker"])}
    sp = sp[sp["ticker"].isin(have_esg)].drop_duplicates("ticker").set_index("ticker")
    return sp[["sector"]]


def esg_annual_scores() -> pd.DataFrame:
    """Build a per-ticker, per-year Refinitiv overall-ESG table.

    Returns:
        A DataFrame indexed by ticker with integer-year columns (overall ESG score).
    """
    esg = pd.read_csv(ESG_PATH)
    esg.columns = [str(c).replace("\n", " ").strip() for c in esg.columns]
    esg["ticker"] = esg["Identifier (RIC)"].str.split(".").str[0]
    esg = esg.drop_duplicates("ticker").set_index("ticker")
    scores = {}
    for offset in range(0, 26):
        col = "ESG Score (FY0)" if offset == 0 else f"ESG Score (FY-{offset})"
        if col in esg.columns:
            scores[ESG_ANCHOR_YEAR - offset] = pd.to_numeric(esg[col], errors="coerce")
    table = pd.DataFrame(scores)  # index = Refinitiv ticker, columns = years
    # Duplicate aliased companies under their S&P/yfinance ticker so the ESG table keys
    # match the returns/universe tables (e.g. add a BRK-B row copied from BRKa).
    for sp_ticker, ric in TICKER_ALIAS.items():
        if ric in table.index:
            table.loc[sp_ticker] = table.loc[ric]
    return table


def annual_returns(tickers: List[str]) -> pd.DataFrame:
    """Download prices and compute calendar-year total returns per ticker.

    Args:
        tickers: The universe tickers.

    Returns:
        A DataFrame indexed by ticker with integer-year columns (annual return).
    """
    # Download in chunks to stay within yfinance limits; keep adjusted close.
    closes = []
    for i in range(0, len(tickers), 100):
        chunk = tickers[i : i + 100]
        raw = yf.download(chunk, start="1999-12-01", end="2026-08-10",
                          auto_adjust=True, progress=False)
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
        closes.append(close)
        print(f"  downloaded {min(i + 100, len(tickers))}/{len(tickers)} tickers")
    close = pd.concat(closes, axis=1)

    # Year-end price -> year-over-year return.
    year_end = close.resample("YE").last()
    yearly = year_end.pct_change()
    yearly.index = yearly.index.year
    return yearly.T  # rows = ticker, columns = year


def rank_top(values_by_year: pd.DataFrame, universe: pd.DataFrame, signal: str) -> pd.DataFrame:
    """Rank the top-N tickers per sector per year for one signal.

    Args:
        values_by_year: Ticker-by-year values (ESG scores or returns).
        universe: Ticker -> sector table.
        signal: Label stored in the output ("esg" or "return").

    Returns:
        A long DataFrame: year, sector, rank, ticker, value, signal.
    """
    rows = []
    for year in YEARS:
        if year not in values_by_year.columns:
            continue
        col = values_by_year[year].dropna()
        for sector, members in universe.groupby("sector").groups.items():
            vals = col[col.index.isin(members)].sort_values(ascending=False).head(TOP_N)
            for rank, (ticker, value) in enumerate(vals.items(), start=1):
                rows.append({"year": year, "sector": sector, "rank": rank,
                             "ticker": ticker, "value": round(float(value), 4), "signal": signal})
    return pd.DataFrame(rows)


def main() -> None:
    """Rebuild both top-performer tables and write the intersection universe."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    universe = load_universe()
    print(f"Universe: {len(universe)} S&P 500 names with ESG, across {universe['sector'].nunique()} sectors")

    esg_tbl = esg_annual_scores()
    print("Downloading prices and computing annual returns...")
    ret_tbl = annual_returns(list(universe.index))

    # Rank per sector per year and save the corrected tables.
    esg_top = rank_top(esg_tbl, universe, "esg")
    ret_top = rank_top(ret_tbl, universe, "return")
    esg_top.to_csv(os.path.join(OUTPUT_DIR, "esg_top_performers.csv"), index=False)
    ret_top.to_csv(os.path.join(OUTPUT_DIR, "returns_top_performers.csv"), index=False)

    # Count how often each ticker is a top-N performer in each signal (its own sector).
    esg_appear = esg_top["ticker"].value_counts()
    ret_appear = ret_top["ticker"].value_counts()

    # Intersection: consistently top in BOTH signals.
    inter = pd.DataFrame({"esg_years": esg_appear, "ret_years": ret_appear}).fillna(0).astype(int)
    inter["sector"] = universe["sector"].reindex(inter.index)
    inter = inter[(inter["esg_years"] >= MIN_APPEARANCES) & (inter["ret_years"] >= MIN_APPEARANCES)]
    inter["min_years"] = inter[["esg_years", "ret_years"]].min(axis=1)
    inter = inter.sort_values(["sector", "min_years"], ascending=[True, False])
    inter.to_csv(os.path.join(OUTPUT_DIR, "intersection_universe.csv"))

    print(f"\nWrote esg_top_performers.csv, returns_top_performers.csv, intersection_universe.csv "
          f"to {os.path.abspath(OUTPUT_DIR)}")
    print(f"\nIntersection universe ({len(inter)} names; top-{TOP_N} in BOTH ESG and returns "
          f">= {MIN_APPEARANCES} years over {YEARS[0]}-{YEARS[-1]}):\n")
    print(inter[["sector", "esg_years", "ret_years", "min_years"]].to_string())


if __name__ == "__main__":
    main()
