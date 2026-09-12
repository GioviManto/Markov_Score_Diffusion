#!/bin/bash
#SBATCH --job-name=nmfrozen
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
cd $HOME/nongaussian-bp/research/nongaussian-bp
source $HOME/nongaussian-bp/gpu_env.sh 2>/dev/null || true
P=$HOME/nongaussian-bp/.venv/bin/python
export BP_DEVICE=gpu OMP_NUM_THREADS=4
$P -c 'from src.backend import gpu_available; import sys; sys.exit(0 if gpu_available() else 1)' || { echo 'FATAL: no usable CUDA device'; exit 1; }
# exp_21 at the frozen budgets. The published run (627165) used em_iters=80 and
# net_steps=8000 against the frozen protocol's 400 and 20000, so its BASELINE was
# trained on 40% of the budget with no checkpoint selection -- which is why its
# ratios (18.3, 24.1, 37.0) sit above the certified table's 15.5 at the same
# nseq, and why the compendium refuses to pool them. Undertraining the baseline
# flatters us, so this is the one mismatch that had to be removed rather than
# merely disclosed.
T=${SLURM_ARRAY_TASK_ID:-0}
BETAS=(0.0 0.1 0.25 0.5 1.0)
GAMMAS=(0.0 0.05 0.1 0.2 0.4)
FROZEN_SET=(--set em_iters=400 --set net_steps=20000)
if [ $T -lt 5 ]; then
  B=${BETAS[$T]}; OUT=outputs/exp_21_frozen/gauss_beta$B
  $P experiments/exp_21_nonmarkov.py --only gauss --output-dir $HOME/nongaussian-bp/$OUT \
     --set betas="($B,)" --set gammas="()" "${FROZEN_SET[@]}"
elif [ $T -lt 10 ]; then
  G=${GAMMAS[$((T-5))]}; OUT=outputs/exp_21_frozen/gauss_gamma$G
  $P experiments/exp_21_nonmarkov.py --only gauss --output-dir $HOME/nongaussian-bp/$OUT \
     --set betas="()" --set gammas="($G,)" "${FROZEN_SET[@]}"
else
  B=${BETAS[$((T-10))]}; OUT=outputs/exp_21_frozen/laplace_beta$B
  $P experiments/exp_21_nonmarkov.py --only laplace --output-dir $HOME/nongaussian-bp/$OUT \
     --set betas="($B,)" "${FROZEN_SET[@]}"
fi
