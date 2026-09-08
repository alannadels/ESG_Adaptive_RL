"""Assemble the per-regime sweep table from ``evolve_regimes_worker.py`` JSON results.

Scans the results directory for ``{regime}_{algorithm}_seed{seed}.json`` records and
prints the same headline table ``evolve_regimes.py`` would have printed — fitness and
normalized evolved weights per optimizer, one block per regime — so a fan-out sweep can
be reported (and re-reported as more jobs finish) without re-running anything.

Run from the repository root:

    python assemble_regime_sweep.py [--results-dir results/regime_sweep]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

from esg_adaptive_rl.reward import FACTOR_NAMES


def load_records(results_dir: str) -> List[dict]:
    """Load every result record in the sweep directory.

    Args:
        results_dir: Directory containing the worker JSON files.

    Returns:
        Parsed JSON records, sorted by (regime, algorithm, seed).
    """
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path, encoding="utf-8") as handle:
            records.append(json.load(handle))
    return records


def main() -> None:
    """Print the per-regime table of fitness and normalized evolved weights."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", default="results/regime_sweep")
    args = parser.parse_args()

    records = load_records(args.results_dir)
    if not records:
        print(f"No result files found in {args.results_dir}")
        return

    groups: Dict[tuple, List[dict]] = {}
    for record in records:
        key = (record.get("universe", "legacy"), record["regime"])
        groups.setdefault(key, []).append(record)

    for (universe, regime) in sorted(groups, key=str):
        group = groups[(universe, regime)]
        first = group[0]
        header = f"\n=== {universe}/{regime.upper()}  (train={first['train_days']}, " \
                 f"val={first['val_days']}, steps={first['config']['fitness_timesteps']}) ==="
        print(header)
        print(f"{'algo':<8}{first['config']['fitness_metric']:>9}"
              + "".join(f"{n:>10}" for n in FACTOR_NAMES))
        print("-" * (8 + 9 + 10 * len(FACTOR_NAMES)))
        for record in sorted(group, key=lambda r: r["algorithm"]):
            row = f"{record['algorithm']:<8}{record['best_fitness']:>9.3f}"
            row += "".join(f"{w:>10.3f}" for w in record["best_weights_normalized"])
            print(row)

    seeds = sorted({record["config"]["seed"] for record in records})
    print(f"\n{len(records)} result(s) from {args.results_dir}; seeds covered: {seeds}")


if __name__ == "__main__":
    main()
