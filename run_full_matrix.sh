#!/usr/bin/env bash
# Full survivorship study matrix.
#
#   universes : crsp (survivorship-free, CRSP+tresg) x legacy (original biased fixed-50)
#   detectors : crossover (SMA 50/200) x ema (EMA 50/200) x hmm (walk-forward)
#   steps     : 30000 x 100000 PPO fitness budget  (the "did 100k help?" axis)
#   each cell : 3 regimes x 8 optimizers = 24 evolutionary searches
#
# => 2 x 3 x 2 = 12 cells x 24 = 288 searches, then each cell is re-evaluated into full
# portfolio-performance artifacts (metrics + daily returns + holdings + 1/N & SPY baselines)
# so post-run analysis has everything it needs.
#
# PREREQUISITES (run from the dev-branch repo root):
#   - the three CRSP pickles shipped into Dataset/cache/:
#       regime_splits_crsp_{crossover,ema,hmm}.pkl
#   - python env with the project deps + hmmlearn (for the hmm detector) + pyarrow
#   - internet (SPY download for regime labels + baselines; yfinance for the legacy universe)
#
# USAGE:
#   NPROC=16 bash run_full_matrix.sh
#   # subsets, e.g. only the survivorship-free 100k cells first:
#   UNIVERSES=crsp STEPS_LIST=100000 bash run_full_matrix.sh
#
# Resumable: a cell whose evolved_metrics.csv already exists is skipped.

set -euo pipefail

PYTHON="${PYTHON:-python}"
NPROC="${NPROC:-16}"
SEEDS="${SEEDS:-3}"
STEPS_LIST="${STEPS_LIST:-100000 30000}"
UNIVERSES="${UNIVERSES:-crsp legacy}"
DETECTORS="${DETECTORS:-crossover ema hmm}"
MATRIX_DIR="${MATRIX_DIR:-results/matrix}"
CACHE_DIR="Dataset/cache"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Make sure THIS directory's esg_adaptive_rl (the dev/snapshot code) wins over any
# version installed in the venv's site-packages.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

mkdir -p "$MATRIX_DIR"
MASTER_LOG="$MATRIX_DIR/matrix.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"; }

pickle_path() { echo "$CACHE_DIR/regime_splits_${1}_${2}.pkl"; }   # (universe, detector)

# --- 1. ensure every (universe, detector) regime-split pickle exists ---
log "Preparing regime-split pickles (universes: $UNIVERSES; detectors: $DETECTORS)"
for universe in $UNIVERSES; do
  for detector in $DETECTORS; do
    pkl="$(pickle_path "$universe" "$detector")"
    if [[ -f "$pkl" ]]; then
      log "  have $pkl"
      continue
    fi
    if [[ "$universe" == "crsp" ]]; then
      # CRSP pickles are survivorship-free and must be shipped (need wrds_data to rebuild).
      if [[ -f "wrds_data/daily_returns.parquet" && -f "Dataset/universe_snapshots_crsp.csv" ]]; then
        log "  building $pkl from CRSP cache"
        "$PYTHON" -W ignore build_crsp_splits.py --detector "$detector" 2>&1 | tee -a "$MASTER_LOG"
      else
        log "ERROR: $pkl missing and no wrds_data/ to rebuild it -- ship the CRSP pickle first."
        exit 1
      fi
    else
      log "  building $pkl (legacy universe, yfinance)"
      "$PYTHON" -W ignore build_legacy_splits.py --detector "$detector" 2>&1 | tee -a "$MASTER_LOG"
    fi
  done
done

# --- 2. run every cell: 24-job sweep, then full portfolio-data save ---
for steps in $STEPS_LIST; do
  for universe in $UNIVERSES; do
    for detector in $DETECTORS; do
      cell="${universe}_${detector}_${steps}s"
      out="$MATRIX_DIR/$cell"
      pkl="$(pickle_path "$universe" "$detector")"
      umode="snapshot"; [[ "$universe" == "legacy" ]] && umode="legacy"

      if [[ -f "$out/evolved_metrics.csv" ]]; then
        log "SKIP $cell (already analysed)"
        continue
      fi
      mkdir -p "$out"
      log "CELL $cell  (pickle=$pkl, steps=$steps, nproc=$NPROC)"

      # 2a. fan the 24 (regime, optimizer) searches across cores (slow optimizers first).
      for regime in bull neutral bear; do
        for algorithm in abc acor cma de gwo lshade pso xnes; do
          printf '%s\n' "--regime ${regime} --algorithm ${algorithm}"
        done
      done | xargs -P "$NPROC" -n 4 "$PYTHON" evolve_regimes_worker.py \
          --universe "$umode" --data-cache "$pkl" \
          --fitness-timesteps "$steps" --out-dir "$out" \
          2>&1 | tee -a "$out/sweep.log" | tee -a "$MASTER_LOG"

      # 2b. re-evaluate evolved weights -> full portfolio artifacts + 1/N & SPY baselines.
      log "CELL $cell  saving portfolio data (${SEEDS} seeds)"
      "$PYTHON" -W ignore save_portfolio_data.py \
          --results-dir "$out" --data-cache "$pkl" \
          --timesteps "$steps" --seeds "$SEEDS" --out "$out" \
          2>&1 | tee -a "$out/analysis.log" | tee -a "$MASTER_LOG"

      log "CELL $cell DONE"
    done
  done
done

log "ALL CELLS COMPLETE. Artifacts per cell under $MATRIX_DIR/<universe>_<detector>_<steps>s/:"
log "  raw sweep JSONs, evolved_{metrics,returns,holdings}, baselines_{metrics,returns}, sweep.log, analysis.log"
