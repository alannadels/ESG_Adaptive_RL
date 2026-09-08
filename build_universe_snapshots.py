"""Entry point: build point-in-time, year-stamped universe snapshots.

Replaces the retrospective fixed universe (``config.UNIVERSE``, picked from
2017-2025 ESG scores but traded over 2005-2026) with a *dynamic* universe
re-selected every year from only that year's known membership and scores. This
kills the survivorship bias and the direct look-ahead of the legacy list.

Selection timing (look-ahead-free, matching ``esg_adaptive_rl.esg_data``'
publication-lag convention and the membership model of
``build_membership_history.py``):

    - the snapshot for year ``Y`` is the universe effective from Jan 1 of ``Y``;
    - membership as of Jan 1 ``Y`` comes from the S&P 500 year-end revision of
      year ``Y-1`` (see ``Dataset/sp500_membership_history.csv``);
    - a Refinitiv ESG score for calendar year ``c`` is treated as published in
      June of ``c+1``, so at Jan 1 ``Y`` the newest usable fiscal year is
      ``Y-2``: the ESG ranking window is the average over FY ``Y-6 .. Y-2``;
    - annual returns for year ``c`` are known by Dec 31 ``c``, so the return
      ranking window is the calendar years ``Y-5 .. Y-1``.

Per snapshot year the selection mirrors ``build_top_performers.py``'s
protocol inside the trailing windows: top-``N`` per GICS sector by ESG, top-``N``
per sector by annual return, intersect names top-``N`` in BOTH signals on
``>= min_appearances`` years, then top the sector back up to ``N`` with the
best-ESG remaining names (the exclusion-replacement rule of the mandate).
Sectors/industries never investable under the strict mandate (Energy; Tobacco;
Aerospace & Defense; the defense-contractor ticker list) are dropped before
ranking; names without a sector or without ESG data cannot rank and are
skipped, exactly as in the legacy pipeline's Refinitiv join.

Coverage window: the first honest snapshot is ``2008`` because the Wikipedia
constituents table (the membership source) only exists from 2007. The 2005-2007
window remains covered by the legacy fixed universe in the original pipeline.

Run from the repository root (downloads prices via yfinance for every ever-
member name; needs the membership CSV, produced by ``build_membership_history``):

    python build_universe_snapshots.py

Writes: ``Dataset/universe_snapshots.csv`` plus a printed per-year summary.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from build_top_performers import (
    MIN_APPEARANCES,
    TOP_N,
    annual_returns,
    esg_annual_scores,
)

DATA_DIR = "Dataset"
MEMBERSHIP_PATH = os.path.join(DATA_DIR, "sp500_membership_history.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "universe_snapshots.csv")
ESG_PATH = os.path.join(DATA_DIR, "ESG_2000-26-ESGC.csv")

# Publication-lag conventions (must match esg_adaptive_rl.esg_data.build_esg_arrays):
# an ESG score for calendar year c is known from June of c+1.
ESG_LAG_YEARS: int = 2        # at Jan 1 Y the newest usable FY is Y-2
RETURN_LAG_YEARS: int = 1     # at Jan 1 Y the newest usable annual return is Y-1
# Selection lookback mirrors build_top_performers' ~9-year ranking window
# (it used FY 2017-2025 of a 2026 selection). Returns are known a year earlier
# than ESG, so the windows keep a matching one-year offset.
ESG_WINDOW_YEARS: int = 10    # FY years Y-11 .. Y-2 are averaged
RETURN_WINDOW_YEARS: int = 10  # calendar years Y-10 .. Y-1 are ranked

# The strict mandate (mirrors UNIVERSE_AND_TRAINING.md): these never enter a
# snapshot, regardless of score.
EXCLUDED_SECTORS = {"Energy"}
EXCLUDED_INDUSTRIES = {"Tobacco", "Aerospace & Defense"}
EXCLUDED_TICKERS = {
    "LDOS", "RTX", "NOC", "GD", "LHX", "HII", "TXT", "BAH", "GE", "LMT", "BA",
}

# First honest snapshot year (first year-end constituents table is 2007).
FIRST_SNAPSHOT_YEAR: int = 2008


def load_membership_history(path: str = MEMBERSHIP_PATH) -> pd.DataFrame:
    """Load the membership-history span table.

    Args:
        path: Path to ``Dataset/sp500_membership_history.csv`` (built by
            ``build_membership_history.py``).

    Returns:
        A DataFrame with ``ticker, sector, industry, entry_date, exit_date``
        (dates parsed; ``exit_date`` may be NaT for open spans).

    Raises:
        ValueError: If required columns are missing or an entry date is invalid.
    """
    df = pd.read_csv(path, dtype={"ticker": str})
    for column in ("ticker", "sector", "entry_date", "exit_date"):
        if column not in df.columns:
            raise ValueError(
                f"membership history {path} is missing column {column!r}; "
                f"regenerate it with build_membership_history.py."
            )
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_date"] = pd.to_datetime(df["exit_date"], errors="coerce")
    bad = df[df["entry_date"].isna()]
    if not bad.empty:
        raise ValueError(
            f"unparseable entry dates in {path}: {bad['ticker'].tolist()[:10]}"
        )
    return df


def members_as_of(membership: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Members whose membership span covers ``as_of``.

    Args:
        membership: Span table from :func:`load_membership_history`.
        as_of: The date to test membership on.

    Returns:
        One row per member (tickers unique; overlapping spans are impossible
        by construction of the source, but deduplicated defensively).
    """
    entry_ok = membership["entry_date"] <= as_of
    exit_ok = membership["exit_date"].isna() | (membership["exit_date"] > as_of)
    return membership.loc[entry_ok & exit_ok].drop_duplicates("ticker")


def _top_n_counts(
    values_by_year: Dict[int, pd.Series],
    members: pd.DataFrame,
    window_years: range,
    top_n: int,
) -> Tuple[Dict[str, int], pd.DataFrame]:
    """Count top-``top_n`` appearances per ticker within a year window.

    ``members`` is the as-of pool (index = ticker, has a ``sector`` column);
    for each window year the members with a score in that year are ranked by
    the score within their sector and the top ``top_n`` get an appearance.

    Args:
        values_by_year: ``year -> Series(index=ticker, value)``.
        members: The pool to rank within.
        window_years: The selection window (inclusive boundaries given by the
            caller's ``range``).
        top_n: How many names per sector per year count as "top".

    Returns:
        ``(counts, detail)``: per-ticker appearance counts, and a long
        DataFrame ``year, sector, ticker, value`` of every top-``n`` slot.
    """
    counts: Dict[str, int] = {}
    detail: List[dict] = []
    for year in window_years:
        values = values_by_year.get(year)
        if values is None:
            continue
        column = values.dropna()
        for sector, group in members.groupby("sector", sort=True):
            candidates = column[column.index.isin(group.index)]
            winners = candidates.sort_values(ascending=False).head(top_n)
            for ticker, value in winners.items():
                counts[ticker] = counts.get(ticker, 0) + 1
                detail.append({"year": year, "sector": sector,
                               "ticker": ticker, "value": value})
    return counts, pd.DataFrame(detail)


def select_for_year(
    membership: pd.DataFrame,
    esg_tbl: pd.DataFrame,
    ret_tbl: pd.DataFrame,
    year: int,
    top_n: int = TOP_N,
    min_appearances: int = MIN_APPEARANCES,
) -> pd.DataFrame:
    """Select one year's universe snapshot, strictly as-of Jan 1 of ``year``.

    Args:
        membership: Span table from :func:`load_membership_history`.
        esg_tbl: Ticker-by-year Refinitiv overall-ESG scores (from
            ``build_top_performers.esg_annual_scores``).
        ret_tbl: Ticker-by-year calendar annual returns (from
            ``build_top_performers.annual_returns``).
        year: The snapshot year (universe effective Jan 1 of ``year``).
        top_n: Names per sector per signal.
        min_appearances: Minimum window years in BOTH signals for the
            top-``top_n`` intersection to count as "selected".

    Returns:
        One row per selected/filler name with columns ``year, ticker, sector,
        esg_rank, ret_rank, esg_years, ret_years, min_years, esg_mean,
        selected``.
    """
    as_of = pd.Timestamp(year, 1, 1)
    pool = members_as_of(membership, as_of)
    if pool.empty:
        return pd.DataFrame(columns=[
            "year", "ticker", "sector", "esg_rank", "ret_rank", "esg_years",
            "ret_years", "min_years", "esg_mean", "selected"])

    # Mandate: drop never-investable names before ranking.
    pool = pool[~pool["sector"].isin(EXCLUDED_SECTORS)]
    pool = pool[~pool["industry"].isin(EXCLUDED_INDUSTRIES)]
    pool = pool[~pool["ticker"].isin(EXCLUDED_TICKERS)]

    # Names without a sector (removed members whose export GICS industry is
    # unavailable/unmapped) cannot be bucketed by GICS and must never rank.
    pool = pool[pool["sector"].notna() & (pool["sector"].astype(str).str.strip() != "")]

    # Only ESG-scored names can rank (legacy pipeline's Refinitiv join).
    have_esg = set(esg_tbl.index)
    pool = pool[pool["ticker"].isin(have_esg)]
    if pool.empty:
        return pd.DataFrame(columns=[
            "year", "ticker", "sector", "esg_rank", "ret_rank", "esg_years",
            "ret_years", "min_years", "esg_mean", "selected"])

    # From here on the ticker is the DataFrame index (the ranking helpers rely
    # on it; the membership table's row numbers are not tickers).
    pool = pool.set_index("ticker")

    # Look-ahead-free windows for this snapshot year.
    esg_window = range(year - ESG_WINDOW_YEARS - 1, year - ESG_LAG_YEARS + 1)
    ret_window = range(year - RETURN_WINDOW_YEARS, year - RETURN_LAG_YEARS + 1)
    esg_cols = {y: esg_tbl[y] for y in esg_window if y in esg_tbl.columns}
    ret_cols = {y: ret_tbl[y] for y in ret_window if y in ret_tbl.columns}

    esg_counts, _ = _top_n_counts(esg_cols, pool, esg_window, top_n)
    ret_counts, _ = _top_n_counts(ret_cols, pool, ret_window, top_n)
    esg_mean = pd.DataFrame(esg_cols).mean(axis=1) if esg_cols else pd.Series(dtype=float)
    ret_mean = pd.DataFrame(ret_cols).mean(axis=1) if ret_cols else pd.Series(dtype=float)

    rows = pd.DataFrame({
        "ticker": pool.index,
        "sector": pool["sector"].values,
        "esg_years": [esg_counts.get(t, 0) for t in pool.index],
        "ret_years": [ret_counts.get(t, 0) for t in pool.index],
        "esg_mean": esg_mean.reindex(pool.index).to_numpy(),
        "ret_mean": ret_mean.reindex(pool.index).to_numpy(),
    })
    rows["min_years"] = rows[["esg_years", "ret_years"]].min(axis=1)
    rows["selected"] = (
        (rows["esg_years"] >= min_appearances)
        & (rows["ret_years"] >= min_appearances)
    )

    # Rank within sector: by appearance count, then by mean score.
    rows = rows.sort_values(
        ["sector", "esg_years", "esg_mean", "ticker"],
        ascending=[True, False, False, True], na_position="last",
    )
    rows["esg_rank"] = rows.groupby("sector", sort=True).cumcount() + 1
    rows = rows.sort_values(
        ["sector", "ret_years", "ret_mean", "ticker"],
        ascending=[True, False, False, True], na_position="last",
    )
    rows["ret_rank"] = rows.groupby("sector", sort=True).cumcount() + 1

    # Cap each sector at exactly `top_n`: keep the best-`top_n` names by ESG
    # rank, then top the sector back up with the best-ESG unselected names if
    # it has fewer than `top_n` qualified candidates. Never exceeds `top_n`.
    chosen: List[pd.DataFrame] = []
    for sector, group in rows.groupby("sector", sort=True):
        ranked = group.sort_values(
            ["esg_rank", "ticker"], ascending=[True, True]).head(top_n)
        excluded = set(ranked["ticker"])
        fillers = group[~group["ticker"].isin(excluded)].dropna(
            subset=["esg_mean"]).sort_values(
            ["esg_mean", "ticker"], ascending=[False, True])
        need = max(0, top_n - len(ranked))
        chosen.append(pd.concat([ranked, fillers.head(need)]))
    result = pd.concat(chosen, ignore_index=True) if chosen else rows.iloc[0:0]

    result["year"] = year
    return result[["year", "ticker", "sector", "esg_rank", "ret_rank",
                   "esg_years", "ret_years", "min_years", "esg_mean",
                   "selected"]]


def summarize(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Per-year summary rows for the console report.

    Args:
        snapshots: Concatenated :func:`select_for_year` output.

    Returns:
        ``year, names, selected, filled, sectors`` counts.
    """
    rows = []
    for year, group in snapshots.groupby("year", sort=True):
        rows.append({
            "year": year,
            "names": int(group["ticker"].nunique()),
            "selected": int(group["selected"].sum()),
            "filled": int((~group["selected"]).sum()),
            "sectors": int(group["sector"].nunique()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    """Build all year-stamped snapshots and write the universe table."""
    parser = argparse.ArgumentParser(
        description="Build Dataset/universe_snapshots.csv (point-in-time, "
                    "look-ahead-free per-year universe)."
    )
    parser.add_argument(
        "--start-year", type=int, default=FIRST_SNAPSHOT_YEAR,
        help=f"First snapshot year (default {FIRST_SNAPSHOT_YEAR}).",
    )
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--membership", default=MEMBERSHIP_PATH)
    parser.add_argument("--esg-path", default=ESG_PATH)
    parser.add_argument(
        "--returns-csv",
        default=None,
        help="Optional cache for the downloaded ticker-by-year returns table: "
             "read it if present, write it after a live download.",
    )
    parser.add_argument("--out", default=OUTPUT_PATH)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    parser.add_argument("--min-appearances", type=int, default=MIN_APPEARANCES)
    args = parser.parse_args()

    membership = load_membership_history(args.membership)
    print(f"membership: {len(membership)} spans, "
          f"{membership['ticker'].nunique()} tickers")

    esg_tbl = esg_annual_scores()
    pool = set(membership["ticker"])
    if args.returns_csv and os.path.exists(args.returns_csv):
        ret_tbl = pd.read_csv(args.returns_csv, index_col=0)
        ret_tbl.index = ret_tbl.index.astype(str)
        ret_tbl.columns = pd.to_numeric(ret_tbl.columns, errors="coerce")
        print(f"loaded annual returns from {args.returns_csv} "
              f"({ret_tbl.shape[0]} tickers)")
    else:
        print(f"downloading annual returns for {len(pool)} ever-members...")
        ret_tbl = annual_returns(sorted(pool))
        if args.returns_csv:
            os.makedirs(os.path.dirname(args.returns_csv) or ".", exist_ok=True)
            ret_tbl.to_csv(args.returns_csv)

    frames = []
    for year in range(args.start_year, args.end_year + 1):
        frame = select_for_year(
            membership, esg_tbl, ret_tbl, year,
            top_n=args.top_n, min_appearances=args.min_appearances,
        )
        frames.append(frame)
        summary = frame.groupby("sector", sort=True).size()
        print(f"  {year}: {len(frame):3d} names across "
              f"{frame['sector'].nunique()} sectors "
              f"(selected {int(frame['selected'].sum())}, "
              f"filled {int((~frame['selected']).sum())})")

    snapshots = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    snapshots.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(snapshots)} rows, years "
          f"{args.start_year}-{args.end_year})")
    print(summarize(snapshots).to_string(index=False))


if __name__ == "__main__":
    main()
