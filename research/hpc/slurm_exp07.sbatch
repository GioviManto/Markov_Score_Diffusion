#!/bin/bash
# One array task per part of exp_07 (EM-BP vs the score network).
#
# Submit:   sbatch hpc/slurm_exp07.sbatch
#
#SBATCH --job-name=em-bp-exp07
#SBATCH --array=0-3
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/exp07_%A_%a.out
#SBATCH --error=logs/exp07_%A_%a.err
##SBATCH --account=CHANGE_ME          # SITE
##SBATCH --partition=CHANGE_ME        # SITE

set -euo pipefail

# SITE: replace with your cluster's environment setup.
# module load python/3.12
# source "$HOME/venvs/diffusion/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# Fail with a clear message rather than a bare "unbound variable" if this is
# run outside a job array.
: "${SLURM_ARRAY_TASK_ID:?must be run as a job array (sbatch --array=...)}"

PARTS=(sample_efficiency capacity transfer inference_cost)
PART="${PARTS[$SLURM_ARRAY_TASK_ID]}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

echo "host=$(hostname) part=$PART cpus=${SLURM_CPUS_PER_TASK:-?}"

case "$PART" in
  sample_efficiency)
    # Follow the network well past N=2048: on one node it was still improving
    # there, so "EM-BP needs >=64x less data" is currently a lower bound read
    # off a truncated curve. Extending it either closes the gap or shows the
    # network plateauing, and both are worth knowing.
    EXTRA=(--set 'sizes=(32,64,128,256,512,1024,2048,4096,8192,16384)'
           --set net_steps=100000)
    ;;
  capacity)
    EXTRA=(--set 'archs=((32,32),(128,128),(256,256),(512,512),(1024,1024))'
           --set 'step_counts=(1000,5000,20000,100000,400000)'
           --set capacity_n=4096)
    ;;
  transfer)
    EXTRA=(--set net_steps=100000 --set capacity_n=4096)
    ;;
  inference_cost)
    EXTRA=(--set 'inference_batches=(32,128,512,2048)')
    ;;
esac

srun python experiments/exp_07_em_vs_score_network.py \
    --only "$PART" \
    --output-dir "outputs/exp_07_em_vs_score_network" \
    ${EXTRA[@]+"${EXTRA[@]}"}
