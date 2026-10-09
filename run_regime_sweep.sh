#!/usr/bin/env bash
# Fan out the full per-regime sweep (3 regimes x 8 optimizers) across CPU cores.
#
# Usage (from the repository root, on the HPC login/compute node):
#   bash run_regime_sweep.sh              # 16 parallel workers, snapshot universe
#   NPROC=8 bash run_regime_sweep.sh      # fewer parallel workers
#   UNIVERSE=legacy bash run_regime_sweep.sh   # the original fixed 50-name universe
#
# Each worker is pinned to one BLAS/torch thread so N workers use N cores. Results land
# in results/regime_sweep_snapshot/ (or _legacy/) as one JSON per job; assemble with:
#   python assemble_regime_sweep.py --results-dir results/regime_sweep_snapshot
#
# All 24 jobs share the cached regime-split pickle; it is built once up front
# (--prepare-only) so the fan-out never touches yfinance concurrently.

set -euo pipefail

NPROC="${NPROC:-16}"
OUT_DIR="${OUT_DIR:-results/regime_sweep_snapshot}"
UNIVERSE="${UNIVERSE:-snapshot}"
DATA_CACHE="${DATA_CACHE:-Dataset/cache/regime_splits_${UNIVERSE}.pkl}"

# Tiny MLP policies: CPU only. Also cap BLAS threads in case a worker skips the
# in-process pinning.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Building/caching regime splits once (universe=${UNIVERSE})..."
python evolve_regimes_worker.py --prepare-only --universe "${UNIVERSE}" --data-cache "${DATA_CACHE}"

echo "Launching 24 searches on ${NPROC} cores..."
# Slow optimizers first (abc/acor take ~2x) so the makespan is not set by stragglers.
for regime in bull neutral bear; do
  for algorithm in abc acor cma de gwo lshade pso xnes; do
    printf '%s\n' "--regime ${regime} --algorithm ${algorithm}"
  done
done | xargs -P "${NPROC}" -n 4 python evolve_regimes_worker.py \
  --universe "${UNIVERSE}" --data-cache "${DATA_CACHE}" --out-dir "${OUT_DIR}"

echo "Sweep complete. Headline table:"
python assemble_regime_sweep.py --results-dir "${OUT_DIR}"
