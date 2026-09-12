#!/bin/bash
# One array task per part of the Layer-6 experiments (exp_13, exp_14). As with
# exp_06/exp_07, parts write disjoint CSVs into their own output directory, so
# no merge step is needed.
#
# Submit:   sbatch hpc/slurm_layer6.sbatch
# Monitor:  squeue -u "$USER"; tail -f logs/layer6_*.out
#
#SBATCH --job-name=em-bp-layer6
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/layer6_%A_%a.out
#SBATCH --error=logs/layer6_%A_%a.err
##SBATCH --account=CHANGE_ME          # SITE
##SBATCH --partition=CHANGE_ME        # SITE

set -euo pipefail

# SITE: replace with your cluster's environment setup.
# module load python/3.12
# source "$HOME/venvs/diffusion/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

: "${SLURM_ARRAY_TASK_ID:?must be run as a job array (sbatch --array=...)}"

# exp_13 parts first, then exp_14.
EXPS=(exp_13_speciation_cascade exp_13_speciation_cascade exp_13_speciation_cascade
      exp_13_speciation_cascade exp_13_speciation_cascade
      exp_14_memorization_collapse exp_14_memorization_collapse
      exp_14_memorization_collapse)
PARTS=(spectra cascade levels ordering filtering
       budget collapse time)

EXP="${EXPS[$SLURM_ARRAY_TASK_ID]}"
PART="${PARTS[$SLURM_ARRAY_TASK_ID]}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

echo "host=$(hostname) exp=$EXP part=$PART cpus=${SLURM_CPUS_PER_TASK:-?}"

# Per-part scale-ups. The defaults were chosen to fit a 4-core box; a cluster
# node can afford the settings that actually answer the questions.
#
# The one number to be careful with is `em_iters`. Tree EM converges several
# times more slowly than chain EM -- the internal nodes are never observed, so
# the missing-information fraction is much larger -- and an under-budgeted run
# produces a plausible-looking wrong answer rather than an obvious failure.
# 150 is enough at depth 3-4; deeper trees need more, not fewer.
case "$EXP/$PART" in
  exp_13_speciation_cascade/cascade)
    EXTRA=(--set cascade_depth=6 --set n_samples=20000 --set n_paths=2048
           --set n_steps_sde=800 --set t_min=0.005)
    ;;
  exp_13_speciation_cascade/levels)
    EXTRA=(--set depth=5 --set n_train=1024 --set n_test=512
           --set em_iters=400 --set dsm_steps=60000 --set grid_size=401)
    ;;
  exp_13_speciation_cascade/ordering)
    EXTRA=(--set depth=5 --set n_train=1024 --set em_iters=400
           --set 'probe_steps=(250,500,1000,2000,4000,8000,16000,32000,64000)')
    ;;
  exp_13_speciation_cascade/filtering)
    EXTRA=(--set depth=6 --set n_train=1024 --set n_test=512
           --set em_iters=400 --set dsm_steps=60000)
    ;;
  exp_13_speciation_cascade/spectra)
    EXTRA=(--set 'tree_depths=(2,3,4,5,6,7,8)'
           --set 'chain_sizes=(8,16,32,64,128,256,512,1024,2048)')
    ;;
  exp_14_memorization_collapse/collapse)
    # The point of scaling this one is to straddle the entropic wall: at
    # rho = 0.85 it sits at ~89 chains for n = 8 and ~1e9 for n = 33, so the
    # sweep has to reach both sides for at least one n.
    EXTRA=(--set 'n_sites=(8,12,16,24,32)'
           --set 'train_sizes=(32,64,128,256,1024,4096,16384)'
           --set n_gen=512 --set n_steps_sde=400 --set em_iters=120
           --set dsm_steps=40000)
    ;;
  exp_14_memorization_collapse/time)
    EXTRA=(--set 'time_sites=(8,12,16,24,32,48)'
           --set 'time_sizes=(32,128,512,2048,8192)'
           --set time_batch=1024 --set time_grid_points=200)
    ;;
  *)
    EXTRA=()
    ;;
esac

srun python "experiments/${EXP}.py" \
    --only "$PART" \
    --output-dir "outputs/${EXP}" \
    ${EXTRA[@]+"${EXTRA[@]}"}
