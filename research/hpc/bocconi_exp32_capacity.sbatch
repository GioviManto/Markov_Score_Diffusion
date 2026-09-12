#!/bin/bash
#SBATCH --job-name=e32cap
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%j.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%j.err
#SBATCH --propagate=NONE
#
# Capacity equivalence (review §10.6 / Phase 5), chained sequentially like the
# exp_31 screen/confirm jobs, for the same reason: part_sweep's resume support
# writes the WHOLE accumulated CSV on every flush (write_csv(dest, rows)), so
# two processes appending to the same file concurrently would race and the
# loser's work would vanish silently. Sequential, one worker at a time,
# resuming across shards, is what makes that safe rather than an array of
# concurrent tasks writing to one directory.
#
# Sized from the exp_31 EM benchmark (job 634698: n=512 EM 1.21s/iter GPU,
# 2.99s/iter CPU): 16 seeds x 2 sizes x 5 capacities x 2 inits = 320 fits,
# tolerance-stopped well under the 400-iteration cap in most cells (evidence
# settles early, per Remark in ch11) -- call it ~3 min/fit average, ~16h total.
# Each shard here requests 6h, matching the backfill-friendly sizing that got
# the exp_31 chain scheduled after the unthrottled 24h requests sat pending
# for hours under fair-share contention from other users' jobs.
set -euo pipefail

SRC=${NGBP_SRC:?set NGBP_SRC}
OUT=${NGBP_OUT:?set NGBP_OUT}
cd "$HOME/$SRC/research/nongaussian-bp"
source $HOME/nongaussian-bp/gpu_env.sh
export NGBP_REQUIRE_CLEAN=1
PY=$HOME/nongaussian-bp/.venv/bin/python

echo "start=$(date -Is) host=$(hostname) job=$SLURM_JOB_ID"
cat REVISION

$PY experiments/exp_32_capacity_equivalence.py --only sweep \
    --output-dir "$HOME/$OUT/exp_32_capacity"

echo "done=$(date -Is)"
