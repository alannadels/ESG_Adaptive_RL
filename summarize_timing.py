"""Summarize per-cell wall-times from a sweep's results jsonl.

Reads ``results/full_regime_results*.jsonl`` and reports timing grouped by optimizer and
by detector (cell count, mean seconds, total seconds), so the compute cost of each
nature-inspired algorithm can be compared head-to-head. Works on a partial, still-running
jsonl too — each finished cell adds one line, so this can be run live for a progress view.

Usage:
    python summarize_timing.py [path/to/full_regime_results.jsonl]
"""

from __future__ import annotations

import collections
import json
import sys


def _load(path: str) -> list:
    """Load completed, non-error cell records (those carrying an ``elapsed_s``)."""
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "error" not in rec and rec.get("elapsed_s") is not None:
                recs.append(rec)
    return recs


def _report(recs: list, key: str, label: str) -> None:
    """Print a timing table grouped by ``key`` (e.g. optimizer or detector)."""
    groups = collections.defaultdict(list)
    for rec in recs:
        groups[rec[key]].append(rec["elapsed_s"])
    print(f"\n=== timing by {label} ===")
    print(f"{label:<12}{'cells':>7}{'mean_s':>10}{'total_s':>12}{'total_min':>11}")
    for k in sorted(groups, key=lambda g: -sum(groups[g])):  # most expensive first
        v = groups[k]
        print(f"{k:<12}{len(v):>7}{sum(v) / len(v):>10.1f}{sum(v):>12.1f}{sum(v) / 60:>11.1f}")


def main() -> None:
    """Load the jsonl and print per-optimizer and per-detector timing summaries."""
    path = sys.argv[1] if len(sys.argv) > 1 else "results/full_regime_results.jsonl"
    recs = _load(path)
    if not recs:
        print(f"{path}: no completed cells yet.")
        return

    print(f"Completed cells: {len(recs)} / 72")
    _report(recs, "optimizer", "optimizer")
    _report(recs, "detector", "detector")
    total = sum(r["elapsed_s"] for r in recs)
    print(f"\nTotal compute across finished cells: {total:.0f}s ({total / 60:.1f} min)")


if __name__ == "__main__":
    main()
