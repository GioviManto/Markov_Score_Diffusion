#!/bin/bash
#SBATCH --job-name=nonmark
#SBATCH --partition=medium_cpu
#SBATCH --time=05:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%A_%a.err
#
# exp_21 -- what the chain estimator does when the data is NOT a chain.
#
# The largest remaining gap: every result so far fitted a chain to a chain, so the estimator's
# advantage was measured only under a correct structural assumption. Here the assumption is
# false by construction and the networks carry none, so the advantage must shrink and at some
# point invert. Locating that crossover is the result.
#
# Held until phase P0 landed, deliberately. `_fit_all_arms` trains the neural baselines, and
# before P0.B those trained on five discrete noise levels while being evaluated across the
# whole schedule -- launching earlier would have baked that handicap into the one experiment
# whose entire purpose is a fair structural comparison.
#
#   Gate:
#     SMOKE=$(sbatch --parsable --time=00:40:00 --export=ALL,MODE=smoke bocconi_exp21.sbatch)
#
#   Gaussian non-Markov, exact linear-algebra reference, one task per violation strength:
#     sbatch --dependency=afterok:$SMOKE --array=0-9 --export=ALL,MODE=gauss bocconi_exp21.sbatch
#
#   Non-Gaussian AND non-Markov, exact g-marginalised reference, one task per beta:
#     sbatch --dependency=afterok:$SMOKE --array=0-4 --export=ALL,MODE=laplace bocconi_exp21.sbatch
#
# The laplace mode gets more cores: its reference runs one batched grid-BP pass per
# Gauss-Hermite node (41 of them) per evaluation, which is the only part of this package that
# genuinely benefits from BLAS threads.

set -euo pipefail

ROOT=$HOME/nongaussian-bp
PYTHON=$ROOT/.venv/bin/python
cd "$ROOT/research/nongaussian-bp"

TASK=${SLURM_ARRAY_TASK_ID:-0}
MODE=${MODE:-gauss}

THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS

echo "host=$(hostname)  job=${SLURM_JOB_ID:-NA}  task=$TASK  mode=$MODE  threads=$THREADS"
echo "start=$(date -Is)"
echo

case "$MODE" in

  smoke)
    "$PYTHON" -m pytest tests/ -q
    echo "--- exp_21 --quick ---"
    "$PYTHON" experiments/exp_21_nonmarkov.py --quick \
        --output-dir "$ROOT/outputs/smoke/exp_21"
    ;;

  gauss)
    # Tasks 0-4 sweep beta (global latent); tasks 5-9 sweep gamma (long-range coupling).
    # Each task runs ONE strength so the array parallelises, and the zero entries are the
    # Markov controls that must reproduce the existing numbers.
    BETAS=(0.0 0.1 0.25 0.5 1.0)
    GAMMAS=(0.0 0.05 0.1 0.2 0.4)
    if [ "$TASK" -lt 5 ]; then
        B=${BETAS[$TASK]}
        OUT=$ROOT/outputs/exp_21/gauss_beta${B}
        SET_ARGS=(--set betas="($B,)" --set gammas="()")
    else
        G=${GAMMAS[$((TASK - 5))]}
        OUT=$ROOT/outputs/exp_21/gauss_gamma${G}
        SET_ARGS=(--set betas="()" --set gammas="($G,)")
    fi
    echo "gauss -> $OUT"
    "$PYTHON" experiments/exp_21_nonmarkov.py --only gauss \
        --output-dir "$OUT" "${SET_ARGS[@]}"
    ;;

  laplace)
    BETAS=(0.0 0.1 0.25 0.5 1.0)
    B=${BETAS[$TASK]}
    OUT=$ROOT/outputs/exp_21/laplace_beta${B}
    echo "laplace -> $OUT"
    "$PYTHON" experiments/exp_21_nonmarkov.py --only laplace \
        --output-dir "$OUT" --set betas="($B,)"
    ;;

  *)
    echo "unknown MODE=$MODE (expected smoke, gauss or laplace)" >&2
    exit 2
    ;;
esac

echo
echo "end=$(date -Is)"
