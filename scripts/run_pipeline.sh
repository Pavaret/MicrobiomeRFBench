#!/usr/bin/env bash
# Convenience runner for the MTX-response random-forest pipeline.
#
# Resolves paths relative to the repository, so it works from any checkout
# location without editing. Override the compute settings with the environment
# variables below, e.g.:
#
#   N_CORES=16 RF_N_PERMUTATIONS=1000 ./scripts/run_pipeline.sh
#
# A fast smoke test that finishes in a couple of minutes:
#
#   RF_N_REPEATS=3 RF_N_PERMUTATIONS=5 EXTRA="--fast-grid --groups RA --feature-sets pathway_only --out-dir /tmp/rf_smoke" \
#     ./scripts/run_pipeline.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

N_CORES="${N_CORES:-8}"
RF_N_REPEATS="${RF_N_REPEATS:-100}"
RF_BASE_SEED="${RF_BASE_SEED:-42}"
RF_N_PERMUTATIONS="${RF_N_PERMUTATIONS:-1000}"
RF_PRESCREEN="${RF_PRESCREEN:-rf_fallback}"
EXTRA="${EXTRA:-}"

echo "Repository : $REPO_DIR"
echo "Cores      : $N_CORES"
echo "Repeats    : $RF_N_REPEATS (base seed $RF_BASE_SEED)"
echo "Permutations: $RF_N_PERMUTATIONS"
echo "Pre-screen : $RF_PRESCREEN"

python "$SCRIPT_DIR/rf_mtx_response_pipeline.py" \
  --data-dir "$REPO_DIR/data" \
  --out-dir "$REPO_DIR/results" \
  --n-cores "$N_CORES" \
  --n-repeats "$RF_N_REPEATS" \
  --base-seed "$RF_BASE_SEED" \
  --n-permutations "$RF_N_PERMUTATIONS" \
  --prescreen "$RF_PRESCREEN" \
  --fig-format svg \
  $EXTRA
