#!/bin/bash
# Seed-replicated sample-efficiency sweep: the headline curve, with error bars.
#
# The single-node run has one replicate per N, which is why its EM curve wobbles
# non-monotonically. This array runs the same sweep under independent training
# seeds — the held-out test set and its exact-BP reference stay fixed across
# tasks, so every replicate is judged against the same yardstick and the
# comparison remains paired.
#
# Submit:   sbatch hpc/slurm_replicates.sbatch
# Merge:    python tools/merge_replicates.py outputs/replicates
#
#SBATCH --job-name=em-bp-reps
#SBATCH --array=0-31
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=logs/reps_%A_%a.out
#SBATCH --error=logs/reps_%A_%a.err
##SBATCH --account=CHANGE_ME          # SITE
##SBATCH --partition=CHANGE_ME        # SITE

set -euo pipefail

# SITE: replace with your cluster's environment setup.
# module load python/3.12
# source "$HOME/venvs/diffusion/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

SEED="$SLURM_ARRAY_TASK_ID"
OUT="outputs/replicates/seed_${SEED}"

echo "host=$(hostname) seed=$SEED cpus=${SLURM_CPUS_PER_TASK:-?}"

srun python experiments/exp_07_em_vs_score_network.py \
    --only sample_efficiency \
    --output-dir "$OUT" \
    --set "seed=$SEED" \
    --set 'sizes=(32,64,128,256,512,1024,2048,4096,8192)' \
    --set net_steps=60000
