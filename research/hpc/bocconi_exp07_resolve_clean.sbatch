#!/bin/bash
#SBATCH --job-name=e07resolve
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --propagate=NONE
#
# Clean-deploy resolve rerun for the 7 cells still unresolved at M=801.
#
# outputs/README_m801_resolve.md records that job 634478 (16 seeds, M=801,
# old stamp+rsync deploy) took the excluded-cell count from 16 to 7, not to 0,
# and that its own outputs are not certifiable: 3 of the dirty tree's files
# were on exp_07's own code path. This reruns ONLY the surviving 7 -- all at
# n_chains=32, seeds 1 and 12 -- through hpc/deploy_clean.sh, at M=1201, so a
# result this time is either committable or the exclusion is confirmed as
# durable rather than a deployment artefact.
#
#   seed  1: t in {2.0676, 3.0}
#   seed 12: t in {0.4665, 0.6769, 0.9821, 1.425, 2.0676}
#
# One array task per seed rather than a loop in one job, so a slow seed cannot
# block the other and a failure is attributable.
set -euo pipefail

SRC=${NGBP_SRC:?set NGBP_SRC}
OUT=${NGBP_OUT:?set NGBP_OUT}
cd "$HOME/$SRC/research/nongaussian-bp"
source $HOME/nongaussian-bp/gpu_env.sh
export BP_DEVICE=gpu
export NGBP_REQUIRE_CLEAN=1
PY=$HOME/nongaussian-bp/.venv/bin/python

SEEDS=(1 12)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}

echo "start=$(date -Is) host=$(hostname) seed=$SEED"
cat REVISION
nvidia-smi --query-gpu=name --format=csv,noheader || true
"$PY" -c "import sys; from src.backend import gpu_available; sys.exit(0 if gpu_available() else 1)" \
    || { echo "NO USABLE GPU - refusing"; exit 3; }

"$PY" experiments/exp_07_em_vs_score_network.py \
    --only sample_efficiency_val \
    --output-dir "$HOME/$OUT/exp_07_m1201_resolve_seed${SEED}" \
    --set "seed=$SEED" \
    --set "grid_size=1201" \
    --set "sizes=(32,)"

echo "done=$(date -Is)"
