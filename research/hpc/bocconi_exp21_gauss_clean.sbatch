#!/bin/bash
#SBATCH --job-name=e21gc
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
# The Gaussian arm of exp_21, rerun through the clean deployment path.
#
# WHY THIS EXISTS. outputs/exp_21_frozen/ -- the tree Section 9.5 currently
# reports -- was produced with src/em.py uncommitted, in all ten cells. The
# recorded commit therefore cannot reconstruct the program that ran, which is
# the exp_12 defect verbatim, and it is on the executed code path rather than
# merely elsewhere in the tree:
#
#     provenance_gate.require_clean(..., allow_legacy=True)  ->  STILL REFUSED
#
# Legacy provenance (exp_07, exp_27) is a missing archive digest and is
# disclosable. This is not that. The numbers may well be right -- em.py's
# uncommitted state may have been unrelated to this experiment -- but "may well
# be right" is not the standard Appendix C claims to enforce, and the table is
# load-bearing for the misspecification section.
#
# Same protocol as the run it replaces, so the comparison is like for like:
# em_iters=400, net_steps=20000, five beta and five gamma strengths, one cell
# per array task, each writing its own directory so there is no shared CSV to
# race on.
set -euo pipefail

SRC=${NGBP_SRC:?set NGBP_SRC}
OUT=${NGBP_OUT:?set NGBP_OUT}
cd "$HOME/$SRC/research/nongaussian-bp"
source $HOME/nongaussian-bp/gpu_env.sh
export NGBP_REQUIRE_CLEAN=1
PY=$HOME/nongaussian-bp/.venv/bin/python

T=${SLURM_ARRAY_TASK_ID:-0}
BETAS=(0.0 0.1 0.25 0.5 1.0)
GAMMAS=(0.0 0.05 0.1 0.2 0.4)

echo "start=$(date -Is) host=$(hostname) job=$SLURM_JOB_ID task=$T"
cat REVISION

if [ "$T" -lt 5 ]; then
  B=${BETAS[$T]}
  $PY experiments/exp_21_nonmarkov.py --only gauss \
      --output-dir "$HOME/$OUT/exp_21_clean/gauss_beta$B" \
      --set betas="($B,)" --set gammas="()" \
      --set em_iters=400 --set net_steps=20000
else
  G=${GAMMAS[$((T-5))]}
  $PY experiments/exp_21_nonmarkov.py --only gauss \
      --output-dir "$HOME/$OUT/exp_21_clean/gauss_gamma$G" \
      --set betas="()" --set gammas="($G,)" \
      --set em_iters=400 --set net_steps=20000
fi

echo "done=$(date -Is)"
