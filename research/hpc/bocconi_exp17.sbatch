#!/bin/bash
#SBATCH --job-name=exp17
#SBATCH --partition=medium_cpu
#SBATCH --time=05:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%A_%a.err
#
# exp_17 (the depth law) on the Bocconi cluster.
#
#   Reference curve, cheap, run once and inspect before anything else:
#     sbatch --array=0-0 --export=ALL,MODE=law bocconi_exp17.sbatch
#
#   Depth sweep, one array task per depth (all seeds inside the task):
#     sbatch --array=0-7 --export=ALL,MODE=depth bocconi_exp17.sbatch
#
#   Budget control, one task per (depth,width) pair:
#     sbatch --array=0-4 --export=ALL,MODE=budget bocconi_exp17.sbatch
#
# CPU, not GPU, and deliberately. The whole package is pure numpy with no device code, so a
# GPU allocation would run the same host-side arithmetic while holding an accelerator idle.
# The work here is many small independent fits, which is what CPU array tasks are for.
#
# QOS caps at this site: 30 jobs submitted, 10 running per user. Array sizes above that are
# rejected at submission rather than queued, so keep each array small and chain them.

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ROOT=$HOME/nongaussian-bp
PYTHON=$ROOT/.venv/bin/python
cd "$ROOT/research/nongaussian-bp"

TASK=${SLURM_ARRAY_TASK_ID:-0}
MODE=${MODE:-depth}

echo "host=$(hostname)  job=${SLURM_JOB_ID:-NA}  task=$TASK  mode=$MODE"
echo "start=$(date -Is)"
echo

case "$MODE" in

  law)
    # No training, pure linear algebra. This is the yardstick every other number is
    # measured against, so it runs alone and its output is meant to be read by a human
    # before the sweeps are trusted.
    "$PYTHON" experiments/exp_17_depth_law.py \
        --only law --output-dir "$ROOT/outputs/exp_17/law"
    ;;

  depth)
    DEPTHS=(1 2 3 4 6 8 10 12)
    D=${DEPTHS[$TASK]}
    "$PYTHON" experiments/exp_17_depth_law.py \
        --only depth \
        --output-dir "$ROOT/outputs/exp_17/depth_d${D}" \
        --set depths="($D,)"
    ;;

  budget)
    PAIRS=("(2,58)" "(4,40)" "(6,32)" "(8,28)" "(12,22)")
    P=${PAIRS[$TASK]}
    "$PYTHON" experiments/exp_17_depth_law.py \
        --only budget \
        --output-dir "$ROOT/outputs/exp_17/budget_${TASK}" \
        --set budget_pairs="($P,)"
    ;;

  *)
    echo "unknown MODE=$MODE (expected law, depth or budget)" >&2
    exit 2
    ;;
esac

echo
echo "end=$(date -Is)"
