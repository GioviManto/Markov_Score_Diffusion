#!/bin/bash
#SBATCH --job-name=cpoint
#SBATCH --partition=medium_cpu
#SBATCH --time=05:45:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%A_%a.err
#
# Does the richer mixture kernel that fixes the generated TAIL also keep its
# pointwise advantage? The component sweep shows generated kurtosis climbing
# monotonically in C and overtaking the CNN between C=4 and C=8, but that sweep
# ran generation only. If pointwise error degrades as C grows, the two axes
# trade off and the story is different from "more components is simply better".
# CPU: pure numpy, no device code needed.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ROOT=$HOME/nongaussian-bp
cd $ROOT/research/nongaussian-bp
CS=(2 4 8 12 16)
C=${CS[${SLURM_ARRAY_TASK_ID:-0}]}
$ROOT/.venv/bin/python experiments/exp_16_sampling_validation.py \
    --only pointwise --output-dir "$ROOT/outputs/exp_16/cpoint_C${C}" \
    --set sizes="(2048,)" --set em_components="$C" --set n_seed=3
