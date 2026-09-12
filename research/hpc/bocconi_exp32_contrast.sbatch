#!/bin/bash
#SBATCH --job-name=e32contrast
#SBATCH --partition=medium_cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%j.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%j.err
#
# The predeclared C=16 vs C=8 equivalence check, run once after the sweep
# chain (bocconi_exp32_capacity.sbatch) has produced capacity_equivalence.csv.
# Seconds of CPU work reading a CSV -- no GPU needed.
set -euo pipefail

SRC=${NGBP_SRC:?set NGBP_SRC}
OUT=${NGBP_OUT:?set NGBP_OUT}
cd "$HOME/$SRC/research/nongaussian-bp"
export NGBP_REQUIRE_CLEAN=1
PY=$HOME/nongaussian-bp/.venv/bin/python

echo "start=$(date -Is) host=$(hostname)"
"$PY" experiments/exp_32_capacity_equivalence.py --only contrast \
    --output-dir "$HOME/$OUT/exp_32_capacity"
echo "done=$(date -Is)"
