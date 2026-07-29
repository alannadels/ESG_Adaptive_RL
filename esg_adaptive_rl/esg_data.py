"""Loader for the real Refinitiv/LSEG ESG scores.

Reads the exported ESG spreadsheet (``Dataset/ESG_2000-26-ESGC.csv``) and turns its
fiscal-year columns into the time-indexed E/S/G table the environment consumes — a
drop-in replacement for the synthetic placeholder in :mod:`esg_adaptive_rl.data`.

The export stores, per company, annual scores as offset columns:

    - overall ESG        : ``ESG Score (FY0)`` … ``ESG Score (FY-25)``   (scale 0-100)
    - E / S / G pillars  : ``<Pillar> Pillar ESG Score (FY0..FY-5)``      (scale 0-5)

Two realities the loader is built to handle honestly:

    - Coverage is far shallower than the column count implies: overall ESG is well
      populated for roughly the last eight years, and the pillar scores for only about
      the last three (FY0-FY-2). Missing years are simply skipped.
    - Fiscal years are mapped to calendar years via ``anchor_year`` (the calendar year
      ``FY0`` corresponds to) and made available only after a reporting lag, so a score
      for calendar year Y is not used until it would plausibly have been published — this
      keeps the ESG signal look-ahead-free.

Scores are normalized to ``[0, 1]`` (pillars / 5, overall / 100) to match the range the
environment and reward expect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

# Column-name stems for each field in the export.
_OVERALL_STEM = "ESG Score"
_PILLAR_STEM = {
    "E": "Environmental Pillar ESG Score",
    "S": "Social Pillar ESG Score",
    "G": "Governance Pillar ESG Score",
}

# Observed native scales in the export, used to normalize to [0, 1].
_PILLAR_SCALE = 5.0
_OVERALL_SCALE = 100.0


def _fy_column(stem: str, offset: int) -> str:
    """Return the export column name for a field at a given fiscal-year offset.

    Args:
        stem: The column stem (e.g. ``"ESG Score"``).
        offset: Fiscal-year offset back from the latest (0 = FY0, 1 = FY-1, ...).

    Returns:
        The full column name, e.g. ``"ESG Score (FY-1)"``.
    """
    return f"{stem} (FY0)" if offset == 0 else f"{stem} (FY-{offset})"


@dataclass
class RefinitivESG:
    """Parsed ESG scores keyed by ticker and fiscal-year offset.

    Attributes:
        overall: Overall ESG (0-100), index = ticker, columns = fiscal-year offset.
        pillars: Mapping ``{"E"|"S"|"G": DataFrame}`` (0-5), same index/column layout.
    """

    overall: pd.DataFrame
    pillars: Dict[str, pd.DataFrame]


def load_refinitiv_esg(
    path: str,
    max_overall_offset: int = 25,
    max_pillar_offset: int = 5,
) -> RefinitivESG:
    """Parse the ESG export into per-ticker annual score tables.

    Args:
        path: Path to the ESG CSV export.
        max_overall_offset: Highest fiscal-year offset to read for overall ESG.
        max_pillar_offset: Highest fiscal-year offset to read for the pillars.

    Returns:
        A :class:`RefinitivESG` bundle.
    """
    df = pd.read_csv(path)
    # Column headers in the export contain embedded newlines; normalize them.
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]

    # The identifier is a Reuters Instrument Code ("AAPL.OQ"); the base ticker is the
    # part before the exchange suffix.
    df["ticker"] = df["Identifier (RIC)"].str.split(".").str[0]
    df = df.drop_duplicates("ticker").set_index("ticker")

    def _collect(stem: str, max_offset: int) -> pd.DataFrame:
        """Gather the available fiscal-year columns for one field into a frame."""
        series = {}
        for offset in range(0, max_offset + 1):
            column = _fy_column(stem, offset)
            if column in df.columns:
                series[offset] = pd.to_numeric(df[column], errors="coerce")
        return pd.DataFrame(series)

    overall = _collect(_OVERALL_STEM, max_overall_offset)
    pillars = {pillar: _collect(stem, max_pillar_offset) for pillar, stem in _PILLAR_STEM.items()}
    return RefinitivESG(overall=overall, pillars=pillars)


def _annual_to_daily(
    annual: pd.Series,
    dates: pd.DatetimeIndex,
    anchor_year: int,
    publish_month: int,
) -> np.ndarray:
    """Expand one ticker's annual scores onto a daily date index, look-ahead-free.

    Each fiscal-year score is stamped with a *publish date* — the point at which it would
    plausibly have been known — and each trading day takes the most recent score whose
    publish date is on or before it. A score for the calendar year ``anchor_year-offset``
    is treated as published in ``publish_month`` of the *following* year.

    Args:
        annual: Scores indexed by fiscal-year offset (may contain NaNs).
        dates: Target daily date index.
        anchor_year: Calendar year that fiscal-year offset 0 (FY0) corresponds to.
        publish_month: Month (1-12) of the year *after* the fiscal year in which the
            score becomes available.

    Returns:
        A 1-D array of daily scores aligned to ``dates`` (NaN only if no score exists at
        all for this ticker).
    """
    published = {}
    for offset, value in annual.items():
        if pd.isna(value):
            continue
        calendar_year = anchor_year - int(offset)
        # Available from publish_month of the following year -> no look-ahead.
        publish_date = pd.Timestamp(year=calendar_year + 1, month=publish_month, day=1)
        published[publish_date] = float(value)

    if not published:
        return np.full(len(dates), np.nan)

    known = pd.Series(published).sort_index()
    # As-of forward fill: each date takes the latest score known on or before it.
    combined = known.reindex(known.index.union(dates)).sort_index().ffill()
    daily = combined.reindex(dates)
    # Dates before the earliest published score have no as-of value; backfill them with
    # the earliest available score (a documented approximation for the pre-coverage tail).
    return daily.bfill().to_numpy()


def build_esg_arrays(
    esg: RefinitivESG,
    tickers: List[str],
    dates: pd.DatetimeIndex,
    anchor_year: int = 2025,
    publish_month: int = 6,
    normalize: bool = True,
) -> Dict[str, np.ndarray]:
    """Build the time-indexed E/S/G arrays the environment expects.

    Args:
        esg: Parsed ESG scores from :func:`load_refinitiv_esg`.
        tickers: Universe tickers, in column order.
        dates: Daily trading dates, in row order.
        anchor_year: Calendar year that FY0 corresponds to. Confirm this against the
            export's vintage; it shifts every score's effective date.
        publish_month: Reporting-lag month (see :func:`_annual_to_daily`).
        normalize: If True, scale pillars by 1/5 to ``[0, 1]``.

    Returns:
        Mapping ``{"E"|"S"|"G": array}`` with each array shape ``(len(dates), len(tickers))``.
        Any ticker with no ESG data at all is filled with the per-day cross-sectional mean
        (falling back to a neutral 0.5) so the environment never sees NaNs.
    """
    dates = pd.to_datetime(pd.Index(dates))
    n_days, n_assets = len(dates), len(tickers)
    out: Dict[str, np.ndarray] = {}

    for pillar in ("E", "S", "G"):
        matrix = np.full((n_days, n_assets), np.nan)
        pillar_scores = esg.pillars[pillar]
        for col, ticker in enumerate(tickers):
            if ticker in pillar_scores.index:
                matrix[:, col] = _annual_to_daily(
                    pillar_scores.loc[ticker], dates, anchor_year, publish_month
                )
        if normalize:
            matrix = matrix / _PILLAR_SCALE

        # Replace any remaining NaN (a ticker absent from the export) with the daily
        # cross-sectional mean, and clip to the valid [0, 1] band.
        row_means = np.nanmean(matrix, axis=1, keepdims=True)
        row_means = np.where(np.isfinite(row_means), row_means, 0.5)
        matrix = np.where(np.isnan(matrix), row_means, matrix)
        out[pillar] = np.clip(matrix, 0.0, 1.0)

    return out
