"""Audit a candidate ticker universe before handing it to ``load_market_data``.

Two distinct problems bite when you intersect a historical constituent list with a
current one and feed the result to yfinance.

**1. Insufficient price history.** ``data.py`` does ``dropna(how="any")`` across all
tickers, so the panel is truncated to the latest-starting column. One ticker whose
history begins in 2022 silently costs you 22 years on every other name.

**2. Identifier collision.** A ticker string is not an identifier. Symbols get
recycled, so the same three letters can denote unrelated companies at two points in
time. ``CEG`` was Constellation Energy Group (acquired by Exelon, 2012) and is now
Constellation Energy Corporation (spun out of Exelon, 2022).

Why the drop rule is price continuity and nothing else
------------------------------------------------------
Three candidate tests were evaluated against the 191-name intersection of the
Jan-2000 and current S&P 500 lists. Only one is safe to automate:

- **Price continuity** (used here as the drop rule). Objective, and directly answers
  the question that matters mechanically: does a usable series exist over the window?
  It caught all 9 names that truncate the panel.

- **SEC CIK first-filing date** (kept as *evidence*, never as a rule). Flags 18 of 191,
  but is dominated by false positives: holding-company reorganisations and
  redomiciliations mint a fresh CIK for an unbroken business. Disney, Oracle, Exxon,
  Cigna, DuPont, ConocoPhillips, Comcast, Duke, Newmont, Northrop and Medtronic all
  trip it despite continuous history. Useful to a human, useless as a filter.

- **Fuzzy name matching** against the 1999 SPY holdings schedule (tested, rejected).
  It fails on precisely the cases that matter: "Constellation Energy Corp" scores
  1.00 against "Constellation Energy Group, Inc" and "Ingersoll Rand Inc." scores 1.00
  against "Ingersoll-Rand Co." — both genuine collisions where the successor kept the
  name. Meanwhile Disney and Exxon score below 0.70 while being the same company.
  High confidence in the wrong direction is worse than no signal.

The honest conclusion is that no free automated test resolves identity. Price
continuity gates the panel; CIK evidence is surfaced for review; anything that
survives both still carries a caveat that belongs in the write-up. A permanent
identifier (CRSP PERMNO, Compustat GVKEY) is the only real fix.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

# The SEC requires a descriptive User-Agent with contact details on automated requests.
SEC_USER_AGENT = "Georgia Tech OMSCS research lsavasta90@gmail.com"
SEC_RATE_LIMIT_S = 0.11  # stay under the documented 10 requests/second ceiling

CACHE_DIR = Path(__file__).resolve().parent / ".cache"


def _sec_get(url: str) -> dict:
    """Fetch and parse a JSON document from sec.gov."""
    request = urllib.request.Request(url, headers={"User-Agent": SEC_USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def sec_ticker_to_cik(use_cache: bool = True) -> dict[str, tuple[int, str]]:
    """Map current ticker -> (CIK, registrant name) from the SEC's published index.

    Note this reflects ticker ownership *today*, which is exactly the point: it tells
    you which registrant currently holds a symbol, so a modern CIK on a symbol that
    was in the index in 2000 is a hint that the holder changed.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / "sec_company_tickers.json"
    if use_cache and cache.exists():
        payload = json.loads(cache.read_text())
    else:
        payload = _sec_get("https://www.sec.gov/files/company_tickers.json")
        cache.write_text(json.dumps(payload))
    return {
        entry["ticker"].replace(".", "-").upper(): (entry["cik_str"], entry["title"])
        for entry in payload.values()
    }


def sec_first_filing_date(cik: int, use_cache: bool = True) -> str | None:
    """Earliest filing date on record for a CIK, as ``YYYY-MM-DD``.

    A date after the backtest start means the *registrant* did not exist then. That is
    evidence of a collision, not proof — see the module docstring on reorganisations.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"cik_{cik:010d}.json"
    if use_cache and cache.exists():
        submissions = json.loads(cache.read_text())
    else:
        submissions = _sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        cache.write_text(json.dumps(submissions))
        time.sleep(SEC_RATE_LIMIT_S)

    filings = submissions.get("filings", {})
    dates = list(filings.get("recent", {}).get("filingDate", []))
    # Long filing histories are paginated into older archive files carrying a range.
    for archive in filings.get("files", []):
        if archive.get("filingFrom"):
            dates.append(archive["filingFrom"])
    return min(dates) if dates else None


def audit_universe(
    candidates: Iterable[str],
    start: str,
    end: str,
    tolerance_days: int = 30,
    check_sec: bool = True,
) -> pd.DataFrame:
    """Score each candidate ticker on price continuity and registrant identity.

    Args:
        candidates: Ticker symbols to evaluate.
        start: Backtest start date (``YYYY-MM-DD``).
        end: Backtest end date (``YYYY-MM-DD``).
        tolerance_days: Grace period after ``start`` within which a first price is
            still considered acceptable, absorbing holidays and listing quirks.
        check_sec: Whether to attach SEC CIK evidence. Costs one request per ticker
            on a cold cache.

    Returns:
        One row per candidate with ``keep`` (the price-continuity drop rule) and
        ``review`` (CIK evidence suggesting a changed registrant). Sorted worst-first.
    """
    candidates = sorted({t.replace(".", "-").upper() for t in candidates})

    # Probe from slightly before `start` so a name trading on day one is not clipped.
    probe_start = (pd.Timestamp(start) - pd.Timedelta(days=210)).strftime("%Y-%m-%d")
    closes = yf.download(
        candidates, start=probe_start, end=end, auto_adjust=True, progress=False
    )["Close"]
    first_price = closes.apply(lambda series: series.first_valid_index())

    rows = []
    cik_map = sec_ticker_to_cik() if check_sec else {}
    for ticker in candidates:
        cik, sec_name = cik_map.get(ticker, (None, None))
        first_filing = None
        if cik is not None:
            try:
                first_filing = sec_first_filing_date(cik)
            except Exception:
                # Network hiccups must not sink the audit; a missing date reads as
                # "no evidence", which the review flag treats as benign.
                first_filing = None
        rows.append(
            {
                "ticker": ticker,
                "sec_name": sec_name,
                "cik": cik,
                "first_price": first_price.get(ticker),
                "first_filing": first_filing,
            }
        )

    audit = pd.DataFrame(rows)
    audit["first_price"] = pd.to_datetime(audit["first_price"])
    audit["first_filing"] = pd.to_datetime(audit["first_filing"])

    cutoff = pd.Timestamp(start) + pd.Timedelta(days=tolerance_days)
    # The drop rule: a usable series must actually exist at the start of the window.
    audit["keep"] = audit["first_price"].notna() & (audit["first_price"] <= cutoff)
    # Evidence only. Never gate on this - see the module docstring.
    audit["review"] = audit["first_filing"].notna() & (
        audit["first_filing"] > pd.Timestamp(start)
    )
    audit["missing_years"] = (
        (audit["first_price"] - pd.Timestamp(start)).dt.days / 365.25
    ).clip(lower=0).round(1)

    return audit.sort_values(
        ["keep", "review", "first_price"], ascending=[True, False, False]
    ).reset_index(drop=True)
