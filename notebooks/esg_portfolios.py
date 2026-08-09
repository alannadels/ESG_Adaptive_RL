"""Annually-rebalanced ESG quintile portfolios with an enforced publication lag.

The central correctness concern is look-ahead. An ESG score for fiscal year ``Y`` is not
public until roughly 6-12 months after that year ends, so a portfolio that holds it from
1 January of year ``Y`` is trading on information nobody had. :func:`formation_schedule`
builds the (formation date -> scoring year) map, and :func:`assert_no_lookahead` turns the
intended guarantee into a check that actually runs.

The lag is a parameter rather than a constant so the look-ahead premium can be *measured*:
run the study at ``formation_lag_months=12`` and again at ``0``, and the difference is the
size of the bias, which is more informative than asserting it exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Metrics already exist in the project; do not reimplement them here.
from esg_adaptive_rl.metrics import (
    annualized_return,
    annualized_volatility,
    conditional_value_at_risk,
    max_drawdown,
    sharpe_ratio,
)


def formation_schedule(
    years: list[int],
    formation_lag_months: int = 12,
) -> pd.DataFrame:
    """Map each scoring year to the date its scores may first be acted on.

    Args:
        years: Scoring years available in the ESG panel.
        formation_lag_months: Months after fiscal-year end before scores are treated as
            public. Fiscal years are taken to end 31 December, so 6 puts formation at the
            start of July in ``year + 1`` and 12 at the end of that December. The lag is
            applied literally — there is no second snapping step — so the number in the
            argument is the true information delay.

    Returns:
        Frame of ``score_year`` and ``formation_date``, sorted by formation date.
    """
    schedule = pd.DataFrame({"score_year": sorted(years)})
    # Fiscal year Y is treated as ending 31 Dec Y; add the lag to get the tradable date.
    schedule["formation_date"] = [
        pd.Timestamp(year=y, month=12, day=31) + pd.DateOffset(months=formation_lag_months)
        for y in schedule["score_year"]
    ]
    return schedule.sort_values("formation_date").reset_index(drop=True)


def assert_no_lookahead(schedule: pd.DataFrame, formation_lag_months: int) -> None:
    """Fail if any formation date precedes the end of the fiscal year it scores.

    A zero lag is permitted (it is an explicit sensitivity case) but is still flagged by the
    caller; anything negative is nonsense and raises.
    """
    if formation_lag_months < 0:
        raise ValueError("formation_lag_months must be >= 0")
    year_end = pd.to_datetime(
        schedule["score_year"].astype(str) + "-12-31", format="%Y-%m-%d"
    )
    violations = schedule.loc[schedule["formation_date"] < year_end]
    if len(violations):
        raise AssertionError(
            f"{len(violations)} formation date(s) precede their fiscal year end:\n{violations}"
        )


def assign_quintiles(
    panel: pd.DataFrame,
    score_col: str = "esg_score",
    n_buckets: int = 5,
    min_names: int = 50,
) -> pd.DataFrame:
    """Sort each year's cross-section into equal-count buckets on ``score_col``.

    Args:
        panel: Long ESG panel with ``ticker``, ``year`` and ``score_col``.
        score_col: Column to rank on — raw score, or the sector-neutral z-score.
        n_buckets: Number of buckets; 5 gives quintiles.
        min_names: Years with fewer scored names than this are dropped rather than sorted
            into buckets too thin to mean anything. The default of 50 excludes 2002, where
            only 26 of the 181 names are scored and quintiles would hold 5 stocks each.

    Returns:
        ``panel`` plus a 1-based integer ``bucket`` column, restricted to usable years.
    """
    usable = panel.dropna(subset=[score_col]).copy()
    counts = usable.groupby("year")["ticker"].count()
    keep_years = counts[counts >= min_names].index
    usable = usable[usable["year"].isin(keep_years)]

    def _bucket(group: pd.DataFrame) -> pd.Series:
        # rank(method="first") breaks ties deterministically so bucket sizes stay balanced.
        ranks = group[score_col].rank(method="first")
        return pd.cut(ranks, bins=n_buckets, labels=range(1, n_buckets + 1)).astype(int)

    usable["bucket"] = (
        usable.groupby("year", group_keys=False)[[score_col]]
        .apply(lambda g: _bucket(g))
        .astype(int)
    )
    return usable


def build_bucket_returns(
    buckets: pd.DataFrame,
    returns: pd.DataFrame,
    formation_lag_months: int = 12,
) -> pd.DataFrame:
    """Compute daily equal-weight returns for each bucket under an annual rebalance.

    Each bucket is held from its formation date until the next formation date. Holdings come
    from the scoring year implied by the lag, never from a later year.

    Args:
        buckets: Output of :func:`assign_quintiles`.
        returns: Daily return panel, dates on the index, tickers as columns.
        formation_lag_months: Months between fiscal-year end and tradability.

    Returns:
        Daily returns per bucket (columns ``Q1``..``Qn``) plus ``EW_universe``, over the
        period actually covered by formations.
    """
    schedule = formation_schedule(sorted(buckets["year"].unique()), formation_lag_months)
    assert_no_lookahead(schedule, formation_lag_months)

    n_buckets = int(buckets["bucket"].max())
    frames = []
    for i, row in schedule.iterrows():
        start = row["formation_date"]
        end = (
            schedule.loc[i + 1, "formation_date"]
            if i + 1 < len(schedule)
            else returns.index.max() + pd.Timedelta(days=1)
        )
        window = returns.loc[(returns.index >= start) & (returns.index < end)]
        if window.empty:
            continue

        year_slice = buckets[buckets["year"] == row["score_year"]]
        period = {}
        for bucket in range(1, n_buckets + 1):
            names = [
                t for t in year_slice.loc[year_slice["bucket"] == bucket, "ticker"]
                if t in window.columns
            ]
            period[f"Q{bucket}"] = window[names].mean(axis=1) if names else np.nan
        held = [t for t in year_slice["ticker"] if t in window.columns]
        period["EW_universe"] = window[held].mean(axis=1)
        frames.append(pd.DataFrame(period, index=window.index))

    return pd.concat(frames).sort_index()


def summarize_buckets(bucket_returns: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Headline risk/return statistics per bucket, via the project's metrics module."""
    rows = []
    for column in bucket_returns.columns:
        series = bucket_returns[column].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "portfolio": column,
                "ann_return": annualized_return(series),
                "ann_vol": annualized_volatility(series),
                "sharpe": sharpe_ratio(series),
                "cvar_5pct": conditional_value_at_risk(series, alpha=alpha),
                "max_drawdown": max_drawdown(series),
                "n_days": len(series),
            }
        )
    return pd.DataFrame(rows).set_index("portfolio")


def spread_series(bucket_returns: pd.DataFrame, n_buckets: int = 5) -> pd.Series:
    """Daily top-minus-bottom bucket return, the long/short ESG spread."""
    return bucket_returns[f"Q{n_buckets}"] - bucket_returns["Q1"]


def rank_persistence(panel: pd.DataFrame, score_col: str = "esg_score") -> pd.Series:
    """Year-over-year Spearman correlation of the cross-sectional score ranking.

    Near-1.0 persistence means the quintiles are almost a fixed set of companies, so any
    return difference between them is closer to a stock-picking accident than an ESG effect.
    """
    wide = panel.pivot_table(index="ticker", columns="year", values=score_col)
    years = sorted(wide.columns)
    return pd.Series(
        {
            year: wide[[prev, year]].dropna().corr(method="spearman").iloc[0, 1]
            for prev, year in zip(years, years[1:])
        },
        name="rank_autocorr",
    )
