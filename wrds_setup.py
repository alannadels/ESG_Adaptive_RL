"""Interactive WRDS connection setup and entitlement check.

Run this once, at your own terminal, to authenticate to WRDS and confirm which
datasets your account can reach. It prompts for your WRDS username/password and
offers to write a ``~/.pgpass`` file so later, non-interactive pulls connect
without a password.

Because it needs an interactive terminal for the password prompt, run it from a
normal shell:

    python wrds_setup.py

It does not download any data -- it only connects, lists the libraries you are
entitled to, highlights the ones this project needs (CRSP for survivorship-free
prices + delisting returns, Compustat for GICS sectors, and any ESG vendor), and
runs one tiny sanity query against the S&P 500 constituent table.
"""

from __future__ import annotations

import sys


# Libraries this project cares about, with why.
NEEDED = {
    "crsp": "survivorship-free prices, point-in-time S&P 500 membership, delisting returns",
    "comp": "Compustat -- GICS sectors, fundamentals",
    "crsp_a_ccm": "CRSP/Compustat merged link (permno <-> gvkey)",
}
# ESG vendors appear under various library names depending on entitlement.
ESG_HINTS = ("esg", "reprisk", "sustainalytics", "msci", "trucost", "refinitiv",
             "asset4", "tr_", "spglobal", "iss")


def main() -> int:
    """Connect to WRDS, report entitlements, and sanity-check CRSP access."""
    try:
        import wrds
    except ImportError:
        print("wrds package not installed. Run: python -m pip install wrds")
        return 1

    print("Connecting to WRDS -- enter your WRDS username and password when prompted.")
    print("If asked to create a .pgpass file, answer 'y' for passwordless future runs.\n")
    try:
        db = wrds.Connection()
    except Exception as exc:  # noqa: BLE001 -- surface any auth/transport error plainly
        print(f"\nConnection failed: {exc}")
        return 1

    print("\nConnected. Checking entitlements...\n")
    try:
        libs = sorted(db.list_libraries())
    except Exception as exc:  # noqa: BLE001
        print(f"Could not list libraries: {exc}")
        db.close()
        return 1

    print("Project-required libraries:")
    for lib, why in NEEDED.items():
        mark = "OK " if lib in libs else "MISSING"
        print(f"  [{mark}] {lib:<12} {why}")

    esg_libs = [l for l in libs if any(h in l.lower() for h in ESG_HINTS)]
    print("\nESG-related libraries you can see:")
    print("  " + (", ".join(esg_libs) if esg_libs else "(none found -- may need a separate ESG entitlement)"))

    # Tiny sanity query: does the point-in-time S&P 500 membership table respond?
    print("\nSanity query on crsp.dsp500list (point-in-time S&P 500 membership):")
    try:
        df = db.raw_sql("select permno, start, ending from crsp.dsp500list "
                        "order by start desc limit 3")
        print(df.to_string(index=False))
        print("\nCRSP membership table is reachable -- survivorship-free universe is buildable.")
    except Exception as exc:  # noqa: BLE001
        print(f"  query failed: {exc}")
        print("  (crsp.dsp500list may be named differently under your entitlement; "
              "the library check above still tells us what you have.)")

    print(f"\nTotal libraries accessible: {len(libs)}")
    db.close()
    print("Done. Connection verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
