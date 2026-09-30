#!/usr/bin/env bash
# Turnkey setup + sanity check for the paper-grade run on a fresh WSL/Linux box.
# Creates an isolated venv, installs the CPU-only dependency stack, and runs a tiny
# end-to-end dry run (one cell, minimal budget) to confirm the environment before the
# multi-hour job. It does NOT launch the full run — do that after the dry run passes:
#
#     nohup python evolve_regimes_full.py crossover > results/crossover.log 2>&1 &
#     nohup python evolve_regimes_full.py ema       > results/ema.log       2>&1 &
#     nohup python evolve_regimes_full.py hmm       > results/hmm.log       2>&1 &
#
# (Run them one at a time if you want to watch each; each writes results/full_regime_*<detector>.csv.)
set -euo pipefail

echo "== Python =="
python3 --version

echo "== venv + deps =="
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

mkdir -p results

echo "== dry run (one cell, tiny budget) =="
python - <<'PY'
import warnings; warnings.filterwarnings("ignore")
import evolve_regimes_full as R
from evolution.config import EvolutionConfig
# Shrink the search and eval so this finishes in seconds.
R.SEARCH_CFG = EvolutionConfig(population_size=4, max_generations=1, fitness_timesteps=600)
R.EVAL_SEEDS = [42, 43]
R.EVAL_TIMESTEPS = 600
cell = R.run_cell(("crossover", "bear", "cma"))
assert "error" not in cell, cell
assert abs(sum(cell["holdings"]) - 1.0) < 1e-6, "holdings must sum to 1"
m = cell["metrics"]
print("DRY RUN OK | search_fitness=%.3f | n_eval_ok=%d | sharpe_mean=%.3f | beta_SPY_mean=%.3f"
      % (cell["search_fitness"], cell["n_eval_ok"], m["sharpe_mean"], m["beta_SPY_mean"]))
PY

echo ""
echo "Environment ready. Cores available: $(nproc)"
echo "Launch the full run per-detector (see header of this script)."
