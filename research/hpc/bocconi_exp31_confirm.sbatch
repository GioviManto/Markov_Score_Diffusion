#!/bin/bash
#SBATCH --job-name=e31conf
#SBATCH --partition=long_gpuh200
#SBATCH --gres=gpu:1
#SBATCH --time=2-23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%j.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%j.err
#SBATCH --propagate=NONE
#
# Stage 2: the confirmatory comparison, sixteen seeds, selection already frozen.
#
# The winner is read from the screening CSV by `_winners_from_screening` rather
# than pasted into this script. Two reasons: the choice stays reproducible from
# a committed artefact, and a hand-copied hyperparameter is one more place the
# job and the record can disagree -- which is the class of defect this whole
# revision is about. The shared-window head is carried forward whether or not
# it wins, because dropping the review's reference baseline when it loses would
# be selective.
#
# SIZED FROM MEASUREMENT (job 634698), 16 seeds x 4 sizes = 64 cells:
#
#     EM, 1200 iterations, on GPU        ~0.40 h   (2.5x the CPU rate)
#       + checkpoint selection over 13 rungs, val+test
#     window head                         5.4 min
#     winner network                     25-36 min depending on architecture
#                              per cell  ~1.1 h  ->  ~70 h for 64 cells
#
# That is at the 71 h limit with no margin, and the estimate is an upper bound
# because `fit_em` stops early on its evidence tolerance in most cells. So the
# experiment writes confirm.csv after EVERY (seed, size) cell and skips
# completed cells on restart: if the wall clock runs out, resubmitting this
# same script continues rather than starting over. Check progress with
#     wc -l $HOME/$NGBP_OUT/exp_31_confirm/confirm.csv
#
# This is the stage the GPU actually earns: EM dominates the cost here, and it
# is the arm that accelerates.
set -euo pipefail

SRC=${NGBP_SRC:?set NGBP_SRC}
OUT=${NGBP_OUT:?set NGBP_OUT}
SCREEN=${NGBP_SCREEN:-$HOME/$OUT/exp_31_screen/screening.csv}
cd "$HOME/$SRC/research/nongaussian-bp"
source $HOME/nongaussian-bp/gpu_env.sh
export NGBP_REQUIRE_CLEAN=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
PY=$HOME/nongaussian-bp/.venv/bin/python

echo "start=$(date -Is)  host=$(hostname)  job=$SLURM_JOB_ID"
cat REVISION

if [ ! -s "$SCREEN" ]; then
    echo "FATAL: no screening results at $SCREEN." >&2
    echo "Stage 1 must finish first; selecting the architecture here would let" >&2
    echo "the confirmatory seeds influence the choice." >&2
    exit 1
fi

$PY experiments/exp_31_structured_baseline.py --only confirm \
    --set seeds=16 \
    --set sizes='(32,128,512,2048)' \
    --set winners="'auto:$SCREEN'" \
    --output-dir "$HOME/$OUT/exp_31_confirm"

echo "done=$(date -Is)"
