#!/bin/bash
#SBATCH --job-name=e34ladder
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%A_%a.err
#SBATCH --propagate=NONE
#
# The MDN-inside-BP rung of the review's information ladder (§10.4, Phase 3).
# Uses free capacity alongside the exp_31/exp_32 chains -- this is one job,
# not a chain, and resumes per (seed, size, arm) cell like the others.
set -euo pipefail

SRC=${NGBP_SRC:?set NGBP_SRC}
OUT=${NGBP_OUT:?set NGBP_OUT}
cd "$HOME/$SRC/research/nongaussian-bp"
source $HOME/nongaussian-bp/gpu_env.sh
export NGBP_REQUIRE_CLEAN=1
PY=$HOME/nongaussian-bp/.venv/bin/python

echo "start=$(date -Is) host=$(hostname)"
cat REVISION

"$PY" experiments/exp_34_information_ladder.py --only run \
    --output-dir "$HOME/$OUT/exp_34_ladder"

echo "done=$(date -Is)"
