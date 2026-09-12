#!/bin/bash
#SBATCH --job-name=e07cert
#SBATCH --partition=medium_cpu
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%A_%a.err
#SBATCH --array=0-15%4
#SBATCH --propagate=NONE
#
# Rerun the certified headline table (16 seeds, sample_efficiency_val) through
# the clean-archive deploy path. Two things this buys at once:
#
#   1. THE HELLINGER WEIGHTING (round-two review, "Hellinger" section). The
#      appendix's transition-Hellinger row was the unweighted median over grid
#      columns -- one vote per parent grid point regardless of how much
#      posterior mass ever lands there -- and the review named the pi_0-weighted
#      version as "the right upgrade... not computed here". src/metrics.py now
#      computes it (hellinger_weighted_mean, parent law = standard normal,
#      matching Var(a_i)=1 everywhere in this project) and
#      exp_07_em_vs_score_network.py records it in every row as of this
#      revision. The original certified CSVs predate the field, so getting the
#      real number means rerunning, not recomputing from what is on disk.
#
#   2. INDEPENDENT REPRODUCIBILITY EVIDENCE for the headline table itself. The
#      certified table (commit 2ae68ac3, 0 dirty paths) was never in question
#      -- unlike exp_12_scaled and the n=8192 row -- but it has also never been
#      rerun through hpc/deploy_clean.sh. If this run's score-based numbers
#      (ratio, em_hellinger, resolution count) do not match the frozen ones
#      closely, that is itself a finding worth having before the thesis
#      defence, not after.
#
# CPU, not GPU: this is the same job bocconi_frozen.sbatch's certified branch
# ran, on medium_cpu, ~35-45 min/seed observed there. It does not touch the
# 6-GPU QOS cap the exp_31 and resolve jobs are using, so it runs alongside
# them rather than queueing behind them.
#
# Writes beside the frozen outputs, never into them -- diffing against
# outputs/frozen/exp_07_certified_seed*/ is the whole point, so nothing here
# may overwrite what it is being compared against.
#
# %4: the QOS caps a user at 10 RUNNING jobs total (sacctmgr show qos normal),
# counted across every job regardless of partition or GPU use. Sixteen array
# tasks running at once would consume most of that budget and block the
# exp_31 screen/confirm chain and the M=1201 resolve rerun -- both of which
# need a GPU slot and matter more than this diagnostic does -- from starting
# at all. Four concurrent leaves six job slots free for everything else.
set -euo pipefail

SRC=${NGBP_SRC:?set NGBP_SRC}
OUT=${NGBP_OUT:?set NGBP_OUT}
cd "$HOME/$SRC/research/nongaussian-bp"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export NGBP_REQUIRE_CLEAN=1
PY=$HOME/nongaussian-bp/.venv/bin/python

SEED=${SLURM_ARRAY_TASK_ID:-0}
echo "start=$(date -Is) host=$(hostname) seed=$SEED"
cat REVISION

"$PY" experiments/exp_07_em_vs_score_network.py \
    --only sample_efficiency_val \
    --output-dir "$HOME/$OUT/exp_07_certified_clean_seed${SEED}" \
    --set "seed=$SEED"

echo "done=$(date -Is)"
