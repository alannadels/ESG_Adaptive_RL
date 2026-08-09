"""Reshape the LSEG/Refinitiv ESG extract into a tidy panel joined to the price universe.

The source file ``Dataset/ESG_2000-2026.csv`` is a wide, snapshot-style extract:

- one row per company, keyed by RIC (``AAPL.OQ``);
- 26 columns of *combined* ESG score at **relative fiscal-year offsets**, ``ESG Score (FY0)``
  through ``(FY-25)`` — not calendar dates;
- E/S/G pillar scores and the Controversies score for **FY0 only**, with no history.

Two consequences drive the design here.

**The time axis has to be reconstructed.** ``FY0`` is the most recently completed fiscal year
(2025 for this extract), so ``FY-n`` maps to calendar ``FY0_YEAR - n``. Nothing in the file
records this, so it is a parameter rather than a hard-coded constant, and every consumer
should surface which mapping it used.

**Pillars are not a time series.** ``load_esg_panel`` returns only the combined score, and the
FY0-only fields live behind a separate loader so they cannot be silently melted into a panel
and mistaken for history. A time-varying E/S/G breakdown — what the RL environment ultimately
wants — is not obtainable from this file.

Sector neutralisation happens at **sector**, not industry, level. The 181-name universe spans
51 GICS industries with a median of 3 names each and 18 singletons; within-industry ranking is
meaningless there (a singleton's demeaned score is identically zero). Rolling up to the 11
GICS sectors gives ~16 names per group.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# FY0 is the most recently completed fiscal year in this extract. Confirmed by the user, and
# corroborated by coverage: it places the collapse to ~14% at calendar 2002, which is when
# Refinitiv/LSEG ESG history begins.
FY0_YEAR = 2025

DATASET_DIR = Path(__file__).resolve().parent.parent / "ESG_Adaptive_RL" / "Dataset"
ESG_CSV = DATASET_DIR / "ESG_2000-2026.csv"
RETURNS_CSV = DATASET_DIR / "daily_returns.csv"
AUDIT_CSV = Path(__file__).resolve().parent.parent / "reference" / "universe_audit.csv"

# Column names carry embedded newlines in the raw export.
COL_RIC = "Identifier (RIC)"
COL_NAME = "Company Name"
COL_INDUSTRY = "GICS Industry Name"
COL_MKTCAP = "Company Market Capitalization\n(USD)"
FY0_FIELDS = {
    "Environmental Pillar ESG Score\n(FY0)": "e_pillar",
    "Social Pillar ESG Score\n(FY0)": "s_pillar",
    "Governance Pillar ESG Score\n(FY0)": "g_pillar",
    "Controversies Score\n(FY0)": "controversies",
    "ESG Score\n(FY0)": "esg_score",
}

# Standard GICS industry -> sector rollup, covering all 74 industries present in the extract.
GICS_INDUSTRY_TO_SECTOR: dict[str, str] = {
    # Energy
    "Energy Equipment & Services": "Energy",
    "Oil, Gas & Consumable Fuels": "Energy",
    # Materials
    "Chemicals": "Materials",
    "Construction Materials": "Materials",
    "Containers & Packaging": "Materials",
    "Metals & Mining": "Materials",
    "Paper & Forest Products": "Materials",
    # Industrials
    "Aerospace & Defense": "Industrials",
    "Air Freight & Logistics": "Industrials",
    "Building Products": "Industrials",
    "Commercial Services & Supplies": "Industrials",
    "Construction & Engineering": "Industrials",
    "Electrical Equipment": "Industrials",
    "Ground Transportation": "Industrials",
    "Industrial Conglomerates": "Industrials",
    "Machinery": "Industrials",
    "Marine Transportation": "Industrials",
    "Passenger Airlines": "Industrials",
    "Professional Services": "Industrials",
    "Trading Companies & Distributors": "Industrials",
    "Transportation Infrastructure": "Industrials",
    # Consumer Discretionary
    "Automobile Components": "Consumer Discretionary",
    "Automobiles": "Consumer Discretionary",
    "Broadline Retail": "Consumer Discretionary",
    "Distributors": "Consumer Discretionary",
    "Diversified Consumer Services": "Consumer Discretionary",
    "Hotels, Restaurants & Leisure": "Consumer Discretionary",
    "Household Durables": "Consumer Discretionary",
    "Leisure Products": "Consumer Discretionary",
    "Specialty Retail": "Consumer Discretionary",
    "Textiles, Apparel & Luxury Goods": "Consumer Discretionary",
    # Consumer Staples
    "Beverages": "Consumer Staples",
    "Consumer Staples Distribution & Retail": "Consumer Staples",
    "Food Products": "Consumer Staples",
    "Household Products": "Consumer Staples",
    "Personal Care Products": "Consumer Staples",
    "Tobacco": "Consumer Staples",
    # Health Care
    "Biotechnology": "Health Care",
    "Health Care Equipment & Supplies": "Health Care",
    "Health Care Providers & Services": "Health Care",
    "Health Care Technology": "Health Care",
    "Life Sciences Tools & Services": "Health Care",
    "Pharmaceuticals": "Health Care",
    # Financials
    "Banks": "Financials",
    "Capital Markets": "Financials",
    "Consumer Finance": "Financials",
    "Financial Services": "Financials",
    "Insurance": "Financials",
    "Mortgage Real Estate Investment Trusts (REITs)": "Financials",
    # Information Technology
    "Communications Equipment": "Information Technology",
    "Electronic Equipment, Instruments & Components": "Information Technology",
    "IT Services": "Information Technology",
    "Semiconductors & Semiconductor Equipment": "Information Technology",
    "Software": "Information Technology",
    "Technology Hardware, Storage & Peripherals": "Information Technology",
    # Communication Services
    "Diversified Telecommunication Services": "Communication Services",
    "Entertainment": "Communication Services",
    "Interactive Media & Services": "Communication Services",
    "Media": "Communication Services",
    "Wireless Telecommunication Services": "Communication Services",
    # Utilities
    "Electric Utilities": "Utilities",
    "Gas Utilities": "Utilities",
    "Independent Power and Renewable Electricity Producers": "Utilities",
    "Multi-Utilities": "Utilities",
    "Water Utilities": "Utilities",
    # Real Estate
    "Diversified REITs": "Real Estate",
    "Health Care REITs": "Real Estate",
    "Hotel & Resort REITs": "Real Estate",
    "Industrial REITs": "Real Estate",
    "Office REITs": "Real Estate",
    "Real Estate Management & Development": "Real Estate",
    "Residential REITs": "Real Estate",
    "Retail REITs": "Real Estate",
    "Specialized REITs": "Real Estate",
}


def _read_raw() -> pd.DataFrame:
    """Read the extract and derive a plain ticker from the RIC."""
    raw = pd.read_csv(ESG_CSV)
    # RICs suffix the exchange (.OQ Nasdaq, .N NYSE, .PK pink sheets); strip to bare ticker.
    raw["ticker"] = raw[COL_RIC].astype(str).str.split(".").str[0].str.upper()
    return raw


def audited_universe() -> list[str]:
    """The tickers that survived the price-continuity audit in ``universe_audit.py``."""
    audit = pd.read_csv(AUDIT_CSV)
    return sorted(audit.loc[audit["keep"], "ticker"])


def attach_sector(frame: pd.DataFrame, industry_col: str = COL_INDUSTRY) -> pd.DataFrame:
    """Add a ``sector`` column via the GICS rollup, failing loudly on unmapped industries."""
    frame = frame.copy()
    frame["sector"] = frame[industry_col].map(GICS_INDUSTRY_TO_SECTOR)
    unmapped = sorted(set(frame.loc[frame["sector"].isna(), industry_col].dropna()))
    if unmapped:
        raise KeyError(f"GICS industries missing from the sector map: {unmapped}")
    return frame


def load_esg_panel(
    fy0_year: int = FY0_YEAR,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Melt the wide fiscal-year columns into a tidy ``(ticker, year, esg_score)`` panel.

    Args:
        fy0_year: Calendar year that ``FY0`` denotes. ``FY-n`` becomes ``fy0_year - n``.
        tickers: Restrict to these tickers. Defaults to the full file.

    Returns:
        Long panel with one row per company-year that has a score, carrying company name,
        GICS industry and sector. Rows with no score are dropped, so per-year counts show
        the real coverage rather than a padded rectangle.
    """
    raw = _read_raw()
    if tickers is not None:
        raw = raw[raw["ticker"].isin(set(tickers))]

    score_cols = {
        col: fy0_year - (0 if col.endswith("(FY0)") else int(col.split("FY-")[1].rstrip(")")))
        for col in raw.columns
        if col.startswith("ESG Score\n(FY")
    }

    panel = raw.melt(
        id_vars=["ticker", COL_NAME, COL_INDUSTRY],
        value_vars=list(score_cols),
        var_name="fy_col",
        value_name="esg_score",
    )
    panel["year"] = panel["fy_col"].map(score_cols)
    panel = panel.drop(columns="fy_col").dropna(subset=["esg_score"])
    panel = panel.rename(columns={COL_NAME: "company", COL_INDUSTRY: "industry"})
    panel = attach_sector(panel, industry_col="industry")
    return panel.sort_values(["ticker", "year"]).reset_index(drop=True)


def load_fy0_snapshot(tickers: list[str] | None = None) -> pd.DataFrame:
    """Load the FY0-only fields: E/S/G pillars and Controversies.

    Deliberately separate from :func:`load_esg_panel`. These are a single cross-section with
    no time dimension, and running a return analysis against them would be pure look-ahead.
    """
    raw = _read_raw()
    if tickers is not None:
        raw = raw[raw["ticker"].isin(set(tickers))]
    snapshot = raw[["ticker", COL_NAME, COL_INDUSTRY, COL_MKTCAP, *FY0_FIELDS]].rename(
        columns={COL_NAME: "company", COL_INDUSTRY: "industry", COL_MKTCAP: "market_cap", **FY0_FIELDS}
    )
    return attach_sector(snapshot, industry_col="industry").reset_index(drop=True)


def sector_neutral_score(panel: pd.DataFrame) -> pd.Series:
    """Within-sector, within-year z-score of the raw ESG score.

    Removes the level difference between sectors — Health Care Equipment averages ~80 while
    Insurance averages ~58 — so the ranking reflects standing among peers rather than which
    industry a company happens to sit in. Sectors with a single name in a given year get a
    degenerate z-score; the caller should require a minimum group size.
    """
    grouped = panel.groupby(["year", "sector"])["esg_score"]
    return (panel["esg_score"] - grouped.transform("mean")) / grouped.transform("std")


def sector_rank(panel: pd.DataFrame) -> pd.Series:
    """Within-sector, within-year z-score of the raw ESG score.

    Removes the level difference between sectors — Health Care Equipment averages ~80 while
    Insurance averages ~58 — so the ranking reflects standing among peers rather than which
    industry a company happens to sit in. Sectors with a single name in a given year get a
    degenerate z-score; the caller should require a minimum group size.
    """
    grouped = panel.groupby(["year", "sector"])["esg_score"]
    return grouped.rank(ascending=False)

def load_daily_returns(tickers: list[str] | None = None) -> pd.DataFrame:
    """Load the precomputed daily return panel, indexed by date."""
    returns = pd.read_csv(RETURNS_CSV, parse_dates=["date"]).set_index("date")
    if tickers is not None:
        returns = returns.reindex(columns=[t for t in tickers if t in returns.columns])
    return returns.sort_index()
