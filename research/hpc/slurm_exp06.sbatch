#!/bin/bash
# One array task per part of exp_06. Parts write disjoint CSVs into the shared
# output directory, so no merge step is needed.
#
# Submit:   sbatch hpc/slurm_exp06.sbatch
# Monitor:  squeue -u "$USER"; tail -f logs/exp06_*.out
#
#SBATCH --job-name=em-bp-exp06
#SBATCH --array=0-4
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/exp06_%A_%a.out
#SBATCH --error=logs/exp06_%A_%a.err
##SBATCH --account=CHANGE_ME          # SITE
##SBATCH --partition=CHANGE_ME        # SITE

set -euo pipefail

# SITE: replace with your cluster's environment setup.
# module load python/3.12
# source "$HOME/venvs/diffusion/bin/activate"

# Pin BLAS threads to the allocation. Left unset, numpy assumes the whole node
# and concurrent array tasks oversubscribe it.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# Fail with a clear message rather than a bare "unbound variable" if this is
# run outside a job array.
: "${SLURM_ARRAY_TASK_ID:?must be run as a job array (sbatch --array=...)}"

PARTS=(monotonicity price_of_noising misspecified rate quantization)
PART="${PARTS[$SLURM_ARRAY_TASK_ID]}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

echo "host=$(hostname) part=$PART cpus=${SLURM_CPUS_PER_TASK:-?}"

# Per-part scale-ups. The defaults finish in about an hour on one node; these
# are the settings the single-node runs could not afford. Trim if your wall
# clock is tighter than the --time above.
case "$PART" in
  price_of_noising)
    EXTRA=(--set n_rep=64 --set n_info=2000
           --set 't_price=(0.02,0.05,0.1,0.2,0.4,0.8,1.6,2.4)')
    ;;
  rate)
    EXTRA=(--set n_rep_rate=32
           --set 'sizes_rate=(64,128,256,512,1024,2048,4096)')
    ;;
  quantization)
    # rho values deliberately off the low-denominator lattice, plus the
    # simple rational 0.8 as the contrast case.
    EXTRA=(--set 'quantization_rhos=(0.8,0.77,0.813,0.7913,0.8377)'
           --set 'quantization_grids=(201,401,801,1601,3201)'
           --set n_chains_quantization=512)
    ;;
  misspecified)
    EXTRA=(--set 'components=(2,3,5,8,12)'
           --set 'sizes_kurtosis=(128,256,512,1024,2048,4096)')
    ;;
  *)
    EXTRA=()
    ;;
esac

srun python experiments/exp_06_em_parameter_recovery.py \
    --only "$PART" \
    --output-dir "outputs/exp_06_em_parameter_recovery" \
    ${EXTRA[@]+"${EXTRA[@]}"}
