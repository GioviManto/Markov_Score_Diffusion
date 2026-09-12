#!/bin/bash
#SBATCH --job-name=e21lap
#SBATCH --partition=gpuh200
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%A_%a.err
#SBATCH --propagate=NONE
#
# The Laplace arm of exp_21, at the frozen budget.
#
# WHY THIS EXISTS, and why it is not hpc/bocconi_nonmarkov_frozen.sbatch.
#
# That script has array tasks 10-14 for exactly this, and they were never run:
# outputs/exp_21_frozen/ holds only gauss_*. So the thesis' claim that the
# rank-one/long-range split also holds under a Laplace innovation rested on
# outputs/exp_21/, which used em_iters=80 and net_steps=8000 against the frozen
# protocol's 400 and 20000. An undertrained baseline inflates every ratio, and
# the Laplace claim is precisely "the ratio stays above 1" -- so the defect
# points the same way as the claim, and the claim had to be dropped from
# Chapter 9 rather than merely caveated. This run is what restores it.
#
# The old script also hardcodes $HOME/nongaussian-bp/, the pre-
# deploy_clean tree, so its outputs would carry that tree's provenance rather
# than an archive digest. This one takes NGBP_SRC/NGBP_OUT from
# hpc/.last_deploy.env like the exp_31 and exp_32 jobs, so the thing that is
# hashed is the thing that runs.
#
# One beta per array task. Unlike exp_32's sweep this is safe to parallelise:
# each task writes its own laplace_beta<B>/ directory, so there is no shared
# accumulating CSV to race on. Sized from the Gaussian arm of the same
# experiment (~900 s of EM per cell plus two 20k-step network fits); 6h is
# backfill-friendly and roughly four times the observed need.
set -euo pipefail

SRC=${NGBP_SRC:?set NGBP_SRC}
OUT=${NGBP_OUT:?set NGBP_OUT}
cd "$HOME/$SRC/research/nongaussian-bp"
source $HOME/nongaussian-bp/gpu_env.sh
export NGBP_REQUIRE_CLEAN=1
PY=$HOME/nongaussian-bp/.venv/bin/python

BETAS=(0.0 0.1 0.25 0.5 1.0)
B=${BETAS[${SLURM_ARRAY_TASK_ID:-0}]}

echo "start=$(date -Is) host=$(hostname) job=$SLURM_JOB_ID task=${SLURM_ARRAY_TASK_ID:-0} beta=$B"
cat REVISION

$PY experiments/exp_21_nonmarkov.py --only laplace \
    --output-dir "$HOME/$OUT/exp_21_frozen/laplace_beta$B" \
    --set betas="($B,)" --set em_iters=400 --set net_steps=20000

echo "done=$(date -Is)"
