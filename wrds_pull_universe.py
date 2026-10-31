"""Pull survivorship-free raw data from WRDS (CRSP + Refinitiv tresg).

Caches five artifacts under ``wrds_data/`` that feed the point-in-time S&P 500
universe rebuild. Every name that was an S&P 500 constituent at any point from
``ERA_START`` onward is included -- crucially, that covers names later delisted
or removed, which a current-constituent universe silently drops. Returns come
from the CIZ daily file where the delisting return is already baked into
``dlyret`` (verified on Lehman Brothers, permno 80599).

Artifacts:
  membership.csv      - crsp.dsp500list spells (permno, start, ending)
  names.csv           - crsp.stocknames_v2 (permno, ticker, cusip9, name spans)
  sector.csv          - GICS sector per permno via CCM link -> comp.company
  esg_panel.csv       - tresg.wrds_ref_esg annual panel (cusip9, year, ESG + E/S/G)
  daily_returns.parquet - crsp.stkdlysecuritydata (permno, date, dlyret, prc, cap)

Run from the repo root after WRDS is set up (see wrds_setup.py / wrds_conn.py):

    python wrds_pull_universe.py
"""

from __future__ import annotations

import os
import time

import pandas as pd

from wrds_conn import engine

OUT = "wrds_data"
ERA_START = "2004-01-01"   # include anyone who was a member from here on (era + lookback headroom)
RET_START = "1998-01-01"   # daily returns back far enough for the Y-10..Y-1 return-ranking window
ESG_START = 2002           # tresg coverage begins ~2002

# Reusable subquery: permnos that were S&P 500 members at any point in the era.
MEMBERS = f"(select distinct permno from crsp.dsp500list where ending >= date '{ERA_START}')"


def _pull(eng, name, query, parquet=False):
    """Run one query, report shape + timing, and cache to CSV or parquet."""
    t = time.time()
    df = pd.read_sql_query(query, eng)
    path = os.path.join(OUT, f"{name}.{'parquet' if parquet else 'csv'}")
    (df.to_parquet(path, index=False) if parquet else df.to_csv(path, index=False))
    print(f"  {name:<16} {df.shape[0]:>9,} rows x {df.shape[1]} cols  "
          f"({time.time() - t:.0f}s)  -> {path}", flush=True)
    return df


def main() -> None:
    """Pull and cache all five raw artifacts."""
    os.makedirs(OUT, exist_ok=True)
    eng = engine()
    print(f"Pulling survivorship-free universe data (members since {ERA_START}):", flush=True)

    _pull(eng, "membership",
          f"select permno, start, ending from crsp.dsp500list "
          f"where ending >= date '{ERA_START}' order by permno, start")

    _pull(eng, "names",
          f"select permno, ticker, cusip9, namedt, nameenddt, issuernm, primaryexch "
          f"from crsp.stocknames_v2 where permno in {MEMBERS} order by permno, namedt")

    # GICS sector via the CCM link (permno<->gvkey) joined to Compustat company header.
    _pull(eng, "sector",
          f"select distinct l.lpermno::int as permno, c.gvkey, c.gsector, c.ggroup, "
          f"c.gind, c.gsubind, c.conm "
          f"from crsp_a_ccm.ccmxpf_lnkhist l "
          f"join comp.company c on c.gvkey = l.gvkey "
          f"where l.linktype in ('LU','LC') and l.linkprim in ('P','C') "
          f"and l.lpermno in {MEMBERS}")

    # Refinitiv ESG annual panel: overall + three pillars, pivoted wide by fieldname.
    esg = _pull(eng, "esg_long",
          f"select cusip as cusip9, year, fieldname, valuescore, valuedate "
          f"from tresg.wrds_ref_esg "
          f"where year >= {ESG_START} and cusip is not null and valuescore is not null "
          f"and fieldname in ('ESGScore','EnvironmentPillarScore',"
          f"'SocialPillarScore','GovernancePillarScore')")
    wide = (esg.pivot_table(index=["cusip9", "year"], columns="fieldname",
                            values="valuescore", aggfunc="first").reset_index())
    wide = wide.rename(columns={"ESGScore": "esg", "EnvironmentPillarScore": "E",
                                "SocialPillarScore": "S", "GovernancePillarScore": "G"})
    wide.to_csv(os.path.join(OUT, "esg_panel.csv"), index=False)
    print(f"  esg_panel        {wide.shape[0]:>9,} rows x {wide.shape[1]} cols  "
          f"-> {os.path.join(OUT, 'esg_panel.csv')}", flush=True)

    _pull(eng, "daily_returns",
          f"select permno, dlycaldt as date, dlyret as ret, dlyprc as prc, dlycap as cap "
          f"from crsp.stkdlysecuritydata "
          f"where permno in {MEMBERS} and dlycaldt >= date '{RET_START}' "
          f"order by permno, dlycaldt", parquet=True)

    print("done.", flush=True)


if __name__ == "__main__":
    main()
