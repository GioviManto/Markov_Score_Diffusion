#!/bin/bash
#SBATCH --job-name=e31screen
#SBATCH --partition=long_gpuh200
#SBATCH --gres=gpu:1
#SBATCH --time=2-23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%j.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%j.err
#SBATCH --propagate=NONE
#
# Stage 1 of the definitive structured-baseline comparison: choose the
# architecture and optimiser on DEVELOPMENT seeds, which are disjoint from the
# sixteen confirmatory seeds by construction (10000+, see part_screen).
#
# ONE JOB, NOT AN ARRAY. Sharding exp_12 across array tasks by seed is what
# introduced the `eff_seed0` override whose stamp does not match any committed
# revision -- the provenance defect that voided the previous result. A single
# task with an internal loop has no merge step and no per-shard override to
# mis-record.
#
# SIZED FROM MEASUREMENT (job 634698 on this partition), not from estimates:
#
#     window  5.4 min/candidate x 16 candidates          =  87 min
#     conv    16.6 (c=64) / 35.7 (c=128) x 8 candidates  = 209 min
#     bimp     7.9 (h=64) / 25.4 (h=128) x 8 candidates  = 134 min
#                                        per (seed,size) = 7.2 h
#     x 4 development seeds x 2 sizes                    = ~57 h
#
# against a 71 h limit. Two deliberate reductions from the review's Phase 1
# sweep, both disclosed rather than silent:
#
#   sizes {32, 2048} rather than all four. The architecture ranking is selected
#   once and frozen, so screening at the two endpoints catches a candidate that
#   wins only at one end, which is the failure mode that matters. Selecting per
#   size would be per-cell tuning.
#
#   learning rates {1e-3, 3e-3} rather than three. 3e-4 is the slowest-learning
#   combination on a ladder that stops at 64k steps, so it is the one most
#   likely to select the cap and be flagged unsaturated anyway. The two kept
#   values bracket an order of magnitude.
#
# The GPU is worth requesting for the EM arm only. Measured on this node:
# EM 2.5x faster on an H200, and the networks IDENTICAL (conv 15.27 -> 15.53
# ms/step) because src/seq_nets.py is pure numpy and does not read BP_DEVICE.
# Screening is almost all network training, so this stage is CPU-bound in
# practice; the partition is chosen for its three-day limit, which is the
# binding constraint.
set -euo pipefail

SRC=${NGBP_SRC:?set NGBP_SRC}
OUT=${NGBP_OUT:?set NGBP_OUT}
cd "$HOME/$SRC/research/nongaussian-bp"
source $HOME/nongaussian-bp/gpu_env.sh
export NGBP_REQUIRE_CLEAN=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
PY=$HOME/nongaussian-bp/.venv/bin/python

echo "start=$(date -Is)  host=$(hostname)  job=$SLURM_JOB_ID"
echo "src=$SRC"
cat REVISION

$PY experiments/exp_31_structured_baseline.py --only screen \
    --set dev_seeds=4 \
    --set sizes='(32,2048)' \
    --set lrs='(1e-3,3e-3)' \
    --output-dir "$HOME/$OUT/exp_31_screen"

echo "done=$(date -Is)"
