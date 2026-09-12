#!/bin/bash
#SBATCH --job-name=e33diag
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%j.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%j.err
#SBATCH --propagate=NONE
#
# Fixed-data-vs-fresh-data x DSM-target-vs-exact-target diagnostic (review
# Phase 2 / §10.3). Six seeds x four conditions x one architecture x n=2048 --
# the bidirectional net's own reference-bundle cost dominates (grid_bp_batch
# at n=2048, ~10s each per the exp_31 benchmark) more than its training does
# (BiMessagePassing h=64 ~7ms/step, exp_31 benchmark job 634698), so this is
# cheap: single 6h slot, sequential like exp_31/exp_32 for the same reason
# (resume flushes the whole CSV per cell; concurrent writers would race).
set -euo pipefail

SRC=${NGBP_SRC:?set NGBP_SRC}
OUT=${NGBP_OUT:?set NGBP_OUT}
cd "$HOME/$SRC/research/nongaussian-bp"
source $HOME/nongaussian-bp/gpu_env.sh
export NGBP_REQUIRE_CLEAN=1
PY=$HOME/nongaussian-bp/.venv/bin/python

echo "start=$(date -Is) host=$(hostname)"
cat REVISION

"$PY" experiments/exp_33_target_diagnostic.py --only run \
    --output-dir "$HOME/$OUT/exp_33_diagnostic"
"$PY" experiments/exp_33_target_diagnostic.py --only summary \
    --output-dir "$HOME/$OUT/exp_33_diagnostic"

echo "done=$(date -Is)"
