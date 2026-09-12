#!/bin/bash
#SBATCH --job-name=rf16
#SBATCH --partition=medium_cpu
#SBATCH --time=05:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%A_%a.err
#
# See hpc/bocconi_frozen.sbatch: the login node runs with `ulimit -t 600` and
# Slurm propagates the submitter's rlimits, so a six-hour job dies by SIGXCPU
# after ten minutes of CPU time unless this is set.
#SBATCH --propagate=NONE
#
# exp_12 at frozen scale -- the structured-baseline comparison, promoted.
#
# WHY. The headline table compares EM-BP against a fully connected MLP, which
# carries no locality prior, and reports 7-18x. exp_12 already showed that a
# weight-shared window predictor -- a 1-D CNN, the architecture the data
# actually calls for -- closes most of that gap: at four seeds the
# validation-selected CNN sits 2.3x to 4.2x behind EM-BP rather than ~10x.
# That is the number a reviewer will ask for, and at four seeds, three sizes
# and five noise levels it is not yet reportable next to a sixteen-seed table.
#
# This raises it to the frozen protocol: sixteen seeds, four sizes, and a
# radius sweep that now reaches r = 16, the full receptive field at
# n_sites = 33. If the advantage survives a weight-shared head that can see
# every site, "bounded receptive field" is eliminated as the explanation.
#
# One seed per array task. A monolithic sixteen-seed run is many hours; a
# single seed is minutes-to-an-hour, and `eff_seed0` shifts the window without
# touching the RNG keys, so shard k draws exactly what seed k would have drawn.
#
#   Gate first -- one seed, small, confirms the tree and the knobs:
#     sbatch --parsable --time=00:40:00 --array=0 \
#            --export=ALL,MODE=smoke hpc/bocconi_exp12_scaled.sbatch
#
#   Then the sweep:
#     sbatch --dependency=afterok:$GATE --array=0-15 \
#            hpc/bocconi_exp12_scaled.sbatch
#
# Merge with tools/merge_exp12.py, which concatenates the per-seed CSVs and
# refuses to merge shards that disagree on any scientific setting.

set -euo pipefail

ROOT=$HOME/nongaussian-bp
PYTHON=$ROOT/.venv/bin/python
cd "$ROOT/research/nongaussian-bp"

SEED=${SLURM_ARRAY_TASK_ID:-0}
MODE=${MODE:-full}

THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS

OUT=outputs/exp_12_scaled/seed${SEED}

# NOTE on --set and commas: values are parsed with ast.literal_eval, so tuples
# must survive the shell as a single argv element. They are quoted here rather
# than passed through --export, because `sbatch --export=ALL,X=(a,b)` splits on
# the comma and silently truncates the value -- that bug cost a whole sweep on
# 2026-08-19. Nothing scientific travels through --export in this script.
if [[ "$MODE" == "smoke" ]]; then
    SIZES="(128,)"
    RADII="(3,6)"
    STEPS=1500
else
    SIZES="(32,128,512,2048)"
    RADII="(3,6,12,16)"
    STEPS=8000
fi

echo "exp_12 scaled: seed=$SEED mode=$MODE sizes=$SIZES radii=$RADII steps=$STEPS"

$PYTHON experiments/exp_12_receptive_field.py \
    --only efficiency,efficiency_val \
    --output-dir "$OUT" \
    --set "eff_seed0=$SEED" \
    --set "eff_seeds=1" \
    --set "eff_sizes=$SIZES" \
    --set "eff_radii=$RADII" \
    --set "eff_steps=$STEPS"

echo "done -> $OUT"
