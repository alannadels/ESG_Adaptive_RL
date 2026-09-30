"""Build ``Dataset/sp500_membership_history.csv`` from English Wikipedia.

The membership-history file is the point-in-time replacement for the current-
membership snapshot in ``Dataset/sp500_constituents.csv``. Instead of one set of
retrospectively-selected survivors it records, per ticker, every *continuous
membership span* in the S&P 500 (``ticker, sector, industry, entry_date,
exit_date``; a ticker with several rows re-entered the index), so downstream
snapshot selection can ask "who was a member on date *t*" without look-ahead.

The change-event tables Wikipedia keeps ("Selected changes", and the newer
"Historical components" page) are *incomplete* for long stretches (e.g. they
omit Bear Stearns, WaMu, Merrill Lynch, Ford's 2008 removal and GE's 2018
removal), so this builder does not rely on them. Instead it samples the
**constituents table of the article itself at year-end revisions**: the page
maintains the full component list, and MediaWiki keeps every historical
revision, so the table as it stood at the end of year ``Y`` is a complete,
as-of membership list for year ``Y``.

Membership model (annual granularity, matching the annual snapshot protocol
of ``build_universe_snapshots.py``):

    - a name present in the year-end table of year ``Y`` is treated as a
      member *from Jan 1 of ``Y+1``*, i.e. the snapshot at Jan 1 of year
      ``Y+1`` is built from the year-end table of year ``Y`` (the first
      revision for which a last day of the year table exists is 2006, hence
      the first honest snapshot year is 2007);
    - a name whose last year-end appearance is ``Y`` exits on Jan 1 of
      ``Y+2``. Names that enter and leave within one calendar year are missed
      (rare, and the annual protocol cannot select them either way);
    - names entered before the first sample default to an entry of
      ``1900-01-01``, downgraded to the exact entry date from the current
      constituents table when one exists.

Sector/industry metadata: current members take their sector and GICS
sub-industry from the current constituents table; removed names take theirs
from the committed Refinitiv export (``Dataset/ESG_2000-26-ESGC.csv``, mapped
GICS industry -> sector via :data:`INDUSTRY_TO_SECTOR`). Both are current-
vintage GICS: structural changes (e.g. the 2016 Real Estate split) are not
restated historically.

Conventions:

    - tickers use the repo's yfinance style (``BRK-B``, ``BF-B``; Wikipedia
      dots are replaced by dashes) to match ``build_top_performers.py``;
    - ``exit_date`` is the first day the name is NOT a member (empty = still
      a member as of the build date).

Run from the repository root (needs network to the English Wikipedia API and
pandas' read_html backend, ``lxml`` or ``html5lib``):

    python build_membership_history.py

For offline/CI use: download the current-components page as ``components.html``
and the historical-changes page as ``changes.html``, pass ``--html-dir``, and
reuse a previously fetched year-end sample cache with ``--samples-json``:

    python build_membership_history.py --html-dir /path --samples-json samples.json

Writes: ``Dataset/sp500_membership_history.csv`` plus a printed sanity report
(per-year membership counts and known-index-event spot checks).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

import pandas as pd

COMPONENTS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
API = "https://en.wikipedia.org/w/api.php"

DATA_DIR = "Dataset"
OUT_PATH = os.path.join(DATA_DIR, "sp500_membership_history.csv")
ESG_PATH = os.path.join(DATA_DIR, "ESG_2000-26-ESGC.csv")

# Wikimedia requires a descriptive User-Agent; the default urllib agent is blocked.
WIKI_USER_AGENT = ("ESG-Adaptive-RL/0.1 (academic research; "
                   "https://github.com/esg-adaptive-rl)")
# Be polite to the API: pause between calls (the API rate-limits aggressively).
API_PAUSE_S = 1.5

# Nominal entry for a name whose membership predates the first sampled revision.
BACKFILL_ENTRY: str = "1900-01-01"

# First year whose last-day-of-year constituents table is available (the article
# was created on 2005-09-14 without a table; the full table was added during
# 2007, whose year-end revision is the first usable sample). The first snapshot
# that can rely on a year-end revision is therefore 2008.
FIRST_SAMPLE_YEAR: int = 2007

NORMALIZE_TICKERS = {
    "BRK.B": "BRK-B", "BF.B": "BF-B",          # dot-style class B symbols
    "BRKB": "BRK-B", "BFB": "BF-B",            # lowercase-class template era (2009-2010)
    "WMI": "WM",                               # Waste Management renamed WMI -> WM (2008)
}

# Share-class aliases between yfinance-style tickers and Refinitiv RIC bases,
# mirroring build_top_performers.TICKER_ALIAS.
TICKER_ALIAS = {"BRK-B": "BRKa", "BF-B": "BFb"}


# --------------------------------------------------------------------- network
def wapi(params: str, tries: int = 6) -> dict:
    """Call the MediaWiki API with backoff on rate limiting.

    Args:
        params: Query-string parameters (without the leading ``?``).
        tries: Max attempts before giving up.

    Returns:
        The decoded JSON payload.

    Raises:
        RuntimeError: If the response is still rate-limited after ``tries``.
    """
    url = f"{API}?{params}"
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": WIKI_USER_AGENT})
            return json.load(urllib.request.urlopen(request, timeout=120))
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < tries - 1:
                time.sleep(3 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"wikimedia API: still rate-limited for {url}")


def fetch_page(url: str) -> str:
    """Download one page as UTF-8 HTML (used for the current-constituents page).

    Args:
        url: The page URL.

    Returns:
        The page HTML.

    Raises:
        RuntimeError: If the request fails.
    """
    request = urllib.request.Request(url, headers={"User-Agent": WIKI_USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - network path
        raise RuntimeError(
            f"failed to fetch {url}: {exc}. For offline use, download the page as "
            f"HTML and pass --html-dir (see the module docstring)."
        ) from exc


def year_end_revision(year: int) -> Tuple[int, str]:
    """Return the ``(revid, timestamp)`` of the last revision in ``year``.

    Args:
        year: The calendar year.

    Returns:
        Revision id and ISO timestamp of the last revision at or before
        ``year``-12-31.
    """
    params = (
        "action=query&prop=revisions"
        f"&titles=List_of_S%26P_500_companies&rvstart={year}-12-31T23:59:59Z"
        "&rvdir=older&rvlimit=1&rvprop=ids|timestamp&format=json"
    )
    page = wapi(params)["query"]["pages"]
    pages = list(page.values())
    rev = pages[0]["revisions"][0]
    return int(rev["revid"]), str(rev["timestamp"])


def fetch_wikitext(revid: int) -> str:
    """Return the wikitext of one revision."""
    params = f"action=parse&prop=wikitext&oldid={revid}&format=json"
    return wapi(params)["parse"]["wikitext"]["*"]


# -------------------------------------------------------------------- html
def _find_table(html: str, required_columns: List[str]) -> pd.DataFrame:
    """Parse the first HTML table that contains every ``required_columns``.

    Args:
        html: Page HTML.
        required_columns: Column names the target table must contain (matched
            case-insensitively against the flattened header).

    Returns:
        The matching table with flattened column labels.

    Raises:
        RuntimeError: If no matching table is found.
    """
    tables = pd.read_html(io.StringIO(html))
    for table in tables:
        header = [str(c) for c in table.columns]
        flat = [str(c) for pair in zip(header[::2], header[1::2]) for c in pair]
        lowered = {h.lower() for h in flat}
        if all(any(r.lower() in h for h in lowered) for r in required_columns):
            return table
    raise RuntimeError(
        f"no table with columns {required_columns} found in the parsed HTML."
    )


def normalize_ticker(value) -> str:
    """Normalize a Wikipedia symbol to the repo's yfinance style.

    Args:
        value: Raw symbol cell (may be NaN/empty, may carry artifacts).

    Returns:
        The normalized ticker, or ``""`` for an empty cell.
    """
    if pd.isna(value):
        return ""
    value = str(value).strip().strip("|").strip().upper()
    return NORMALIZE_TICKERS.get(value, value.replace(".", "-"))


def parse_components(html: str) -> pd.DataFrame:
    """Extract current constituents: ticker, sector, industry, entry date.

    Args:
        html: The current-components page HTML.

    Returns:
        One row per current member with columns
        ``ticker, sector, industry, entry_date``.
    """
    table = _find_table(html, ["symbol", "date added", "gics sector"])
    frame = table[["Symbol", "GICS Sector", "GICS Sub-Industry", "Date added"]].copy()
    frame.columns = ["ticker", "sector", "industry", "entry_date"]
    frame["ticker"] = frame["ticker"].map(normalize_ticker)
    frame["entry_date"] = pd.to_datetime(frame["entry_date"], errors="coerce")
    frame = frame.dropna(subset=["entry_date", "ticker"]).drop_duplicates("ticker")
    frame = frame[frame["ticker"] != ""]
    return frame


def parse_changes(html: str) -> pd.DataFrame:
    """Extract index-change events (effective date, added/removed ticker).

    Used only for the sanity report, not for the spans themselves (the table
    is known-incomplete).

    Args:
        html: The historical-components page HTML.

    Returns:
        One row per event with columns ``date, a_ticker, r_ticker``.
    """
    table = _find_table(html, ["effective date", "removed", "reason"])
    frame = table.copy()
    frame.columns = ["date", "a_ticker", "a_security", "r_ticker", "r_security",
                     "reason", "refs"]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["a_ticker"] = frame["a_ticker"].map(normalize_ticker)
    frame["r_ticker"] = frame["r_ticker"].map(normalize_ticker)
    return frame[["date", "a_ticker", "r_ticker"]].dropna(subset=["date"])


# ------------------------------------------------------------ revision samples
def _cell_ticker(cell: str) -> str:
    """Extract a ticker from one wikitext table cell.

    Handles template symbols (``{{NyseSymbol|MMM}}``, ``{{NasdaqSymbol|ADBE}}``)
    and plain text, after stripping comments, refs, and templates. Returns ""
    unless the cell is a plausible ticker (uppercase, at most 8 characters).

    Args:
        cell: The raw cell text.

    Returns:
        The normalized ticker, or ``""`` if none can be extracted.
    """
    cell = re.sub(r"<!--.*?-->", " ", cell, flags=re.S)
    cell = re.sub(r"<ref[^>]*>.*?</ref>", " ", cell, flags=re.S)
    for pattern in (
        r"\{\{\s*[^|}]*[Ss]ymbol\s*\|(.*?)\}\}",
        r"\{\{\s*(?:NYSE|NASDAQ|NYSEArca|NMS)[^|}]*\|(.*?)\}\}",
    ):
        match = re.search(pattern, cell)
        if match:
            ticker = normalize_ticker(match.group(1))
            return ticker if re.fullmatch(r"[A-Z0-9.\-^]{1,8}", ticker) else ""
    cell = re.sub(r"\{\{.*?\}\}", " ", cell, flags=re.S)
    cell = re.sub(r"\[\[([^|\]]*)\]\]", r"\1", cell)
    ticker = normalize_ticker(cell)
    return ticker if re.fullmatch(r"[A-Z0-9.\-^]{1,8}", ticker) else ""


def parse_symbols(wikitext: str) -> List[str]:
    """Extract the constituents table's tickers from one revision's wikitext.

    The table's column order changed across revisions (2007: Company-first;
    2008+: symbol-first), so the column holding the ticker is located from the
    header line (``! ... !! ...``). Caption lines (``|+ ...``), the navbox, and
    the (incomplete) change-event table are ignored; the first table with
    450+ parsed tickers is taken for the constituents table.

    Args:
        wikitext: The revision's wikitext.

    Returns:
        De-duplicated, normalized tickers in table order.
    """
    for table in re.findall(r"\{\|(.*?)\|\}", wikitext, flags=re.S):
        ticker_col = 0
        header = re.search(r"(?m)^!\s*(.*?)$", table)
        if header:
            cells = [c.strip() for c in header.group(1).split("!!") if c.strip()]
            idx = next(
                (i for i, cell in enumerate(cells)
                 if any(word in cell.lower() for word in ("ticker", "symbol"))),
                None,
            )
            if idx is not None:
                ticker_col = idx

        symbols: List[str] = []
        for block in re.split(r"(?m)^\|-", table):
            data = block.lstrip()
            if not data.startswith("|"):
                continue  # attribute/caption/class lines, empty blocks
            cells = [cell for cell in data[1:].split("||")]
            if len(cells) <= ticker_col:
                continue
            ticker = _cell_ticker(cells[ticker_col])
            if ticker:
                symbols.append(ticker)

        symbols = list(dict.fromkeys(symbols))
        if len(symbols) >= 450:
            return symbols
    return []


def collect_samples(samples_json: Optional[str]) -> Dict[int, List[str]]:
    """Collect year-end constituents samples for 2006 .. 2025.

    Reads/writes an optional JSON cache ``{year: [tickers]}`` to make offline
    re-runs possible without re-hitting the API.

    Args:
        samples_json: Optional cache path (read if it exists, written after a
            live fetch).

    Returns:
        Mapping ``year -> ticker list`` from the last revision of that year.

    Raises:
        RuntimeError: If a sampled revision parses to an implausibly small list.
    """
    if samples_json and os.path.exists(samples_json):
        raw = json.load(open(samples_json, encoding="utf-8"))
        print(f"loaded {len(raw)} year-end samples from {samples_json}")
        return {
            int(year): [normalize_ticker(t) for t in tickers if normalize_ticker(t)]
            for year, tickers in raw.items()
        }

    samples: Dict[int, List[str]] = {}
    for year in range(FIRST_SAMPLE_YEAR, 2026):
        revid, timestamp = year_end_revision(year)
        wikitext = fetch_wikitext(revid)
        symbols = parse_symbols(wikitext)
        if len(symbols) < 450:
            raise RuntimeError(
                f"revision {revid} ({year}-12-31) parsed only {len(symbols)} "
                f"symbols -- the page layout may have changed; rerun after "
                f"updating parse_symbols()."
            )
        samples[year] = symbols
        print(f"sample {year}: {revid} {timestamp} -> {len(symbols)} tickers")
        time.sleep(API_PAUSE_S)

    if samples_json:
        json.dump({str(y): t for y, t in samples.items()},
                  open(samples_json, "w"), indent=0)
        print(f"wrote year-end samples to {samples_json}")
    return samples


# --------------------------------------------------------------------- spans
def build_spans(
    samples: Dict[int, List[str]], components: pd.DataFrame
) -> pd.DataFrame:
    """Resolve yearly membership samples into continuous spans per ticker.

    A name present in the year-end tables of years ``y0..y1`` (contiguous) is
    a member for snapshot years ``y0+1 .. y1+1``, i.e. its span is
    ``(entry, exit)`` with ``entry = Jan 1 of (y0+1)`` (or a nominal backfill
    for the earliest sample, downgraded to the exact current-constituent entry
    date when there is one) and ``exit = Jan 1 of (y1+2)``. A run still open at
    the last sample produces an open span; current constituents absent from
    every sample (recent additions) get an open span from their exact entry
    date.

    Args:
        samples: ``year -> ticker list`` from :func:`collect_samples`.
        components: Current-constituent table from :func:`parse_components`.

    Returns:
        DataFrame ``ticker, entry_date, exit_date`` (exit NaT = open span).
    """
    years = sorted(samples)
    first_year, last_year = years[0], years[-1]
    entry_dates = components.set_index("ticker")["entry_date"]

    present: Dict[str, List[int]] = {}
    for year in years:
        for ticker in samples[year]:
            present.setdefault(ticker, []).append(year)

    rows: List[Dict[str, object]] = []
    for ticker, appearances in present.items():
        runs: List[Tuple[int, int]] = []
        start = prev = appearances[0]
        for year in appearances[1:]:
            if year == prev + 1:
                prev = year
            else:
                runs.append((start, prev))
                start = prev = year
        runs.append((start, prev))

        for y0, y1 in runs:
            if y0 == first_year and runs[0][0] == y0 and len(runs) == 1:
                # A single run reaching back to the first sample: membership
                # predates observation. Prefer the current constituents' exact
                # entry date when it predates the sampled era; otherwise the
                # membership is taken as continuous from the nominal backfill
                # (the constituents date may describe a later re-listing, e.g.
                # GOOG's class-C entry or JCI's post-Tyco re-entry).
                entry = pd.Timestamp(BACKFILL_ENTRY)
                if ticker in entry_dates and pd.Timestamp(entry_dates[ticker]) < pd.Timestamp(f"{first_year + 1}-01-01"):
                    entry = max(entry, pd.Timestamp(entry_dates[ticker]))
                if y1 == last_year:
                    rows.append({"ticker": ticker, "entry_date": entry,
                                 "exit_date": pd.NaT})
                    continue
                exit_date = pd.Timestamp(f"{y1 + 2}-01-01")
                if entry < exit_date:
                    rows.append({"ticker": ticker, "entry_date": entry,
                                 "exit_date": exit_date})
                continue

            if y0 == first_year:
                entry = pd.Timestamp(BACKFILL_ENTRY)
            else:
                entry = pd.Timestamp(f"{y0 + 1}-01-01")
                # Current members with a known exact (re-)entry get the later date.
                if ticker in entry_dates:
                    entry = max(entry, pd.Timestamp(entry_dates[ticker]))
            if y1 == last_year:
                rows.append({"ticker": ticker, "entry_date": entry,
                             "exit_date": pd.NaT})
            else:
                exit_date = pd.Timestamp(f"{y1 + 2}-01-01")
                if entry < exit_date:
                    rows.append({"ticker": ticker, "entry_date": entry,
                                 "exit_date": exit_date})

    # Recent additions present only in the current constituents table.
    for _, row in components.iterrows():
        if row["ticker"] not in present:
            rows.append({"ticker": row["ticker"],
                         "entry_date": pd.Timestamp(row["entry_date"]),
                         "exit_date": pd.NaT})

    return pd.DataFrame(rows)


# ------------------------------------------------------------------ metadata
def load_export_industries(path: str = ESG_PATH) -> Dict[str, str]:
    """Map Refinitiv RIC-base tickers to GICS industry names from the export.

    Args:
        path: Path to the committed Refinitiv export.

    Returns:
        Mapping ``RIC base ticker -> GICS Industry Name``.
    """
    df = pd.read_csv(path, usecols=["Identifier (RIC)", "GICS Industry Name"])
    out: Dict[str, str] = {}
    for base, industry in zip(df["Identifier (RIC)"], df["GICS Industry Name"]):
        if pd.isna(base) or pd.isna(industry):
            continue
        out.setdefault(str(base).split(".")[0], str(industry).strip())
    return out


INDUSTRY_TO_SECTOR: Dict[str, str] = {
    # Energy
    "Energy Equipment & Services": "Energy",
    "Oil, Gas & Consumable Fuels": "Energy",
    # Materials
    "Chemicals": "Materials", "Construction Materials": "Materials",
    "Containers & Packaging": "Materials", "Metals & Mining": "Materials",
    "Paper & Forest Products": "Materials",
    # Industrials
    "Aerospace & Defense": "Industrials",
    "Air Freight & Logistics": "Industrials", "Building Products": "Industrials",
    "Commercial Services & Supplies": "Industrials",
    "Construction & Engineering": "Industrials",
    "Electrical Equipment": "Industrials",
    "Ground Transportation": "Industrials",
    "Industrial Conglomerates": "Industrials", "Machinery": "Industrials",
    "Marine Transportation": "Industrials", "Passenger Airlines": "Industrials",
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
    "Banks": "Financials", "Capital Markets": "Financials",
    "Consumer Finance": "Financials", "Financial Services": "Financials",
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
    "Electric Utilities": "Utilities", "Gas Utilities": "Utilities",
    "Independent Power and Renewable Electricity Producers": "Utilities",
    "Multi-Utilities": "Utilities", "Water Utilities": "Utilities",
    # Real Estate
    "Diversified REITs": "Real Estate", "Health Care REITs": "Real Estate",
    "Hotel & Resort REITs": "Real Estate", "Industrial REITs": "Real Estate",
    "Office REITs": "Real Estate",
    "Real Estate Management & Development": "Real Estate",
    "Residential REITs": "Real Estate", "Retail REITs": "Real Estate",
    "Specialized REITs": "Real Estate",
}


def attach_metadata(
    spans: pd.DataFrame,
    components: pd.DataFrame,
    export_industries: Dict[str, str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Attach sector/industry to every span; report names without a sector.

    Current members take their sector/industry from the constituents table.
    Removed names take theirs from the Refinitiv export's GICS industry
    (mapped to a sector via :data:`INDUSTRY_TO_SECTOR`); names with no
    industry at all are kept with an empty sector so downstream can skip them
    explicitly (they have no ESG data and can never be ranked).

    Args:
        spans: Span table from :func:`build_spans`.
        components: Current-constituent table.
        export_industries: ``RIC base -> industry`` mapping from the export.

    Returns:
        A ``(annotated, unresolved, unmapped)`` tuple: spans with
        ``sector``/``industry`` columns; the list of tickers left without a
        sector; and the export GICS industry names that had no entry in
        :data:`INDUSTRY_TO_SECTOR` (taxonomy drift worth surfacing).
    """
    meta = components[["ticker", "sector", "industry"]].drop_duplicates("ticker")
    meta = meta.set_index("ticker")

    rows = []
    unresolved: List[str] = []
    unmapped: List[str] = []
    for _, span in spans.iterrows():
        ticker = span["ticker"]
        if ticker in meta.index:
            sector = meta.loc[ticker, "sector"]
            industry = meta.loc[ticker, "industry"]
        else:
            industry = export_industries.get(TICKER_ALIAS.get(ticker, ticker), "")
            if industry and industry not in INDUSTRY_TO_SECTOR:
                unmapped.append(industry)
            sector = INDUSTRY_TO_SECTOR.get(industry, "")
            if not sector:
                unresolved.append(ticker)
        rows.append({"ticker": ticker, "sector": sector, "industry": industry,
                     "entry_date": span["entry_date"], "exit_date": span["exit_date"]})
    return pd.DataFrame(rows), sorted(set(unresolved)), sorted(set(unmapped))


# ------------------------------------------------------------------- report
def report(membership: pd.DataFrame) -> None:
    """Print a compact sanity report for the generated file.

    Args:
        membership: The annotated span table.
    """
    print(f"spans: {len(membership)}  tickers: {membership['ticker'].nunique()}")

    known = {
        "BRK-B": "joined Feb 2010; never left (2009-2010 tables used BRKb)",
        "BF-B": "long-standing member (2010 table used BFb)",
        "F": "continuous per year-end tables (GM is the captured GFC removal)",
        "GOOGL": "added 2006 (as GOOG, class-A line)",
        "GE": "continuous per year-end tables",
        "LEH": "removed Sep 2008 (bankruptcy)",
        "BSC": "Bear Stearns: removed June 2008",
        "WM": "Waste Management (WMI pre-2008; WaMu used WM until Sept 2008)",
        "MER": "Merrill Lynch: removed Dec 2008",
        "GM": "removed June 2009, re-added May 2013",
        "DOW": "removed 2017 (DowDuPont DWDP), re-added 2019",
        "TWX": "removed 2018 (AT&T acquisition)",
        "AET": "removed 2018 (CVS acquisition)",
    }
    for ticker, expectation in known.items():
        sub = membership[membership["ticker"] == ticker]
        if sub.empty:
            print(f"  {ticker:<6}: (absent from history)  expected: {expectation}")
        else:
            lines = []
            for _, span in sub.iterrows():
                exit_s = (span["exit_date"].date()
                          if not pd.isna(span["exit_date"]) else "open")
                lines.append(f"{span['entry_date'].date()} -> {exit_s}")
            print(f"  {ticker:<6}: {'; '.join(lines)}  expected: {expectation}")

    counts = []
    for year in range(2008, 2027):
        as_of = pd.Timestamp(year, 1, 1)
        n = int(((membership["entry_date"] <= as_of)
                 & (membership["exit_date"].isna()
                    | (membership["exit_date"] > as_of))).sum())
        counts.append(f"{year}:{n}")
    print("members as of Jan 1 (snapshot pool): " + " ".join(counts))


# --------------------------------------------------------------------- main
def main() -> None:
    """Regenerate the membership-history CSV and print the sanity report."""
    parser = argparse.ArgumentParser(
        description="Build Dataset/sp500_membership_history.csv from Wikipedia."
    )
    parser.add_argument(
        "--html-dir",
        default=None,
        help="Directory containing components.html and changes.html (offline).",
    )
    parser.add_argument(
        "--samples-json",
        default=None,
        help="Cache/read year-end samples as JSON ({year: [tickers]}).",
    )
    parser.add_argument("--out", default=OUT_PATH, help="Output CSV path.")
    args = parser.parse_args()

    if args.html_dir:
        components_html = open(
            os.path.join(args.html_dir, "components.html"), encoding="utf-8"
        ).read()
        changes_html = open(
            os.path.join(args.html_dir, "changes.html"), encoding="utf-8"
        ).read()
    else:
        components_html = fetch_page(COMPONENTS_URL)
        changes_html = fetch_page(CHANGES_URL)

    components = parse_components(components_html)
    print(f"current constituents parsed: {len(components)}")
    changes = parse_changes(changes_html)
    print(f"change events parsed (report-only, known-incomplete): {len(changes)}")

    samples = collect_samples(args.samples_json)
    spans = build_spans(samples, components)
    membership, unresolved, unmapped = attach_metadata(
        spans, components, load_export_industries()
    )

    membership = membership.sort_values(["ticker", "entry_date"]).reset_index(drop=True)
    membership["exit_date"] = membership["exit_date"].fillna(pd.NaT)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    membership.to_csv(args.out, index=False, na_rep="")

    print(f"membership spans written to {args.out}")
    print(f"first sampled year: {min(samples)} -> first snapshot year: "
          f"{min(samples) + 1}")
    if unresolved:
        print(f"WARNING: {len(unresolved)} tickers without a sector (kept empty): "
              f"{unresolved[:20]}{' ...' if len(unresolved) > 20 else ''}")
        if unmapped:
            print(f"  of which {len(unmapped)} industry name(s) missing from "
                  f"INDUSTRY_TO_SECTOR: {unmapped[:10]}"
                  f"{' ...' if len(unmapped) > 10 else ''}")
    report(membership)


if __name__ == "__main__":
    main()
