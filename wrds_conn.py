"""Reusable non-interactive WRDS connection via the local ``~/.pgpass``.

The ``wrds`` package always injects a password into its connection URI and, when
that first attempt fails, drops to an interactive prompt -- which cannot work in a
headless session. This helper sidesteps that by connecting through SQLAlchemy with
the username only, letting libpq resolve the password from ``~/.pgpass`` natively
(the standard PostgreSQL mechanism). It reads only the *username* field from the
pgpass entry for the WRDS host; the password is never read or handled here.

Usage:
    from wrds_conn import engine, sql
    df = sql("select permno, start, ending from crsp.dsp500list limit 5")
"""

from __future__ import annotations

import os

import pandas as pd
import sqlalchemy as sa
import wrds.sql as W


def _pgpass_username(host: str = W.WRDS_POSTGRES_HOST) -> str:
    """Return the username for the WRDS host in ``~/.pgpass`` (password untouched).

    If several entries exist for the host (e.g. a stale one plus a freshly created
    one), the most recently written entry -- the last matching line -- is used.
    """
    path = os.path.expanduser("~/.pgpass")
    if not os.path.exists(path):
        raise RuntimeError("~/.pgpass not found -- run wrds_setup.py in a terminal first.")
    found = []
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) >= 4 and parts[0] == host:
                found.append(parts[3])
    if not found:
        raise RuntimeError(f"No ~/.pgpass entry for host {host}.")
    return found[-1]


def engine() -> sa.Engine:
    """Build a SQLAlchemy engine for WRDS using the pgpass-stored username + libpq password."""
    user = _pgpass_username()
    uri = f"postgresql://{user}@{W.WRDS_POSTGRES_HOST}:{W.WRDS_POSTGRES_PORT}/{W.WRDS_POSTGRES_DB}"
    return sa.create_engine(uri, isolation_level="AUTOCOMMIT", connect_args=W.WRDS_CONNECT_ARGS)


def sql(query: str, eng: sa.Engine | None = None) -> pd.DataFrame:
    """Run a read-only SQL query against WRDS and return a DataFrame."""
    eng = eng or engine()
    with eng.connect() as conn:
        return pd.read_sql_query(sa.text(query), conn)
