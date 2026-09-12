#!/bin/bash
#SBATCH --job-name=valproto
#SBATCH --partition=medium_cpu
#SBATCH --time=05:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%A_%a.err
#
# Model selection moved off the test set.
#
# Two selections in this project happen on the bundle they are then judged on:
#
#   exp_07 emits dsm_net_eps and dsm_net_x0 as separate rows and the better is taken per
#   noise level downstream -- oracle post-selection over two models at every t.
#
#   exp_12 part 3 takes min(...) over every trained head and every global net on the test
#   bundle -- an oracle over receptive-field radius AND parameterisation at once.
#
# Both oracles are granted to the BASELINES, not to EM-BP, so they understate our own
# result rather than inflating it. That is exactly why the fix is worth running: it removes
# a real objection and is expected to move the ratios in our favour. If it moves them the
# other way, the validation bundles are too small for their selections to be signal, and
# n_val is the knob.
#
# The new parts reuse the RNG keys of the parts they mirror, so the models are the same
# fits; only the bundle the choice is made on differs. That is what makes the difference
# attributable to the protocol rather than to a different training run.
#
#   Gate first:
#     SMOKE=$(sbatch --parsable --time=00:40:00 --export=ALL,MODE=smoke bocconi_valprotocol.sbatch)
#
#   Then the two experiments, one per array task:
#     sbatch --dependency=afterok:$SMOKE --array=0-1 bocconi_valprotocol.sbatch
#
# Single-threaded on purpose, matching bocconi_exp16.sbatch: the per-task linear algebra is
# a 401x401 transition matrix, so thread parallelism buys little while one CPU per task
# lets many tasks pack onto a node.

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ROOT=$HOME/nongaussian-bp
PYTHON=$ROOT/.venv/bin/python
cd "$ROOT/research/nongaussian-bp"

TASK=${SLURM_ARRAY_TASK_ID:-0}
MODE=${MODE:-run}

echo "host=$(hostname)  job=${SLURM_JOB_ID:-NA}  task=$TASK  mode=$MODE"
echo "python=$($PYTHON -V 2>&1)  numpy=$($PYTHON -c 'import numpy;print(numpy.__version__)')"
echo "start=$(date -Is)"
echo

case "$MODE" in

  smoke)
    # The gate. Nothing here runs on a laptop, so the suite and both --quick passes run on
    # the cluster and the production array is chained behind this with --dependency=afterok.
    echo "--- pytest ---"
    "$PYTHON" -m pytest tests/ -q
    echo "--- exp_07 --quick ---"
    "$PYTHON" experiments/exp_07_em_vs_score_network.py \
        --quick --only sample_efficiency_val \
        --output-dir "$ROOT/outputs/smoke/exp_07"
    echo "--- exp_12 --quick ---"
    "$PYTHON" experiments/exp_12_receptive_field.py \
        --quick --only efficiency_val \
        --output-dir "$ROOT/outputs/smoke/exp_12"
    ;;

  run)
    case "$TASK" in
      0)
        "$PYTHON" experiments/exp_07_em_vs_score_network.py \
            --only sample_efficiency_val \
            --output-dir "$ROOT/outputs/exp_07_em_vs_score_network"
        ;;
      1)
        "$PYTHON" experiments/exp_12_receptive_field.py \
            --only efficiency_val \
            --output-dir "$ROOT/outputs/exp_12_receptive_field"
        ;;
      *)
        echo "unknown TASK=$TASK (expected 0 for exp_07 or 1 for exp_12)" >&2
        exit 2
        ;;
    esac
    ;;

  *)
    echo "unknown MODE=$MODE (expected smoke or run)" >&2
    exit 2
    ;;
esac

echo
echo "end=$(date -Is)"
