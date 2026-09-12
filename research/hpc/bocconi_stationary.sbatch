#!/bin/bash
#SBATCH --job-name=station
#SBATCH --partition=medium_cpu
#SBATCH --time=05:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%A_%a.err
#
# Strict stationarity, and the locality measurement it repairs.
#
# The priors start at a_1 ~ N(0,1) and then apply non-Gaussian innovations, so the chain is
# covariance-stationary but not strictly stationary: the marginal drifts towards the
# invariant law over about six sites at rho = 0.85. That was corrected in the text, but it
# has a consequence the text did not draw.
#
# exp_11 measures the receptive field by running grid BP on a window [C-r, C+r]. A window of
# a Markov chain is a chain with the same kernel and the marginal of its LEFT ENDPOINT as
# initial law, so the window estimator equals the exact E[a_C | x_window] only when BP is
# handed that law. grid_bp_batch defaults it to N(0,1) -- correct for the Gaussian chain at
# every site, wrong for every other family past the first. The measurement was therefore
# exact for precisely the family the others are normalised against, and approximate for the
# families whose departure from it is the reported 1.12-1.46x result.
#
# MODE=locality runs three arms so the two effects separate:
#   n01        the committed protocol, reproduced as the baseline to compare against
#   n01_exact  the same data, each window given the exact marginal of its own left endpoint
#              -- so n01 vs n01_exact is the error in the committed numbers, nothing else
#              changed
#   invariant  strictly stationary chains throughout, where the estimator is exact at every
#              site
#
# MODE=closure tests a prediction the note makes without a run behind it: that sampling the
# invariant law would not change the closure results. The Gaussian baseline depends on the
# prior only through (rho, q), and neither moves under a change of initial law, so only the
# reference arm can shift. If it shifts appreciably the prediction was wrong.
#
#   Gate first:
#     SMOKE=$(sbatch --parsable --time=00:40:00 --export=ALL,MODE=smoke bocconi_stationary.sbatch)
#
#   Locality, one task per rho (both/all arms inside the task, so they stay paired):
#     sbatch --dependency=afterok:$SMOKE --array=0-2 --export=ALL,MODE=locality \
#            bocconi_stationary.sbatch
#
#   Closure rerun, a single task -- exp_03 has no --set overrides, so it sweeps all three
#   rho values itself, and the whole sweep is a couple of minutes at M=801:
#     sbatch --dependency=afterok:$SMOKE --export=ALL,MODE=closure bocconi_stationary.sbatch
#
# Single-threaded, matching the other CPU scripts here: the work is a 401x401 transition
# matrix per task, so one CPU per task packs more tasks onto a node than threading each.

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ROOT=$HOME/nongaussian-bp
PYTHON=$ROOT/.venv/bin/python
cd "$ROOT/research/nongaussian-bp"

TASK=${SLURM_ARRAY_TASK_ID:-0}
MODE=${MODE:-locality}
RHOS=(0.5 0.85 0.95)
RHO=${RHOS[$TASK]}

echo "host=$(hostname)  job=${SLURM_JOB_ID:-NA}  task=$TASK  mode=$MODE"
echo "python=$($PYTHON -V 2>&1)  numpy=$($PYTHON -c 'import numpy;print(numpy.__version__)')"
echo "start=$(date -Is)"
echo

case "$MODE" in

  smoke)
    echo "--- pytest ---"
    "$PYTHON" -m pytest tests/ -q
    echo "--- exp_11 --quick ---"
    "$PYTHON" experiments/exp_11_nongaussian_locality.py \
        --quick --only stationary \
        --output-dir "$ROOT/outputs/smoke/exp_11"
    echo "--- exp_03 --quick ---"
    "$PYTHON" experiments/exp_03_nongaussian_innovation_sweep.py \
        --quick --only stationary \
        --output-dir "$ROOT/outputs/smoke/exp_03"
    ;;

  locality)
    OUT=$ROOT/outputs/exp_11_nongaussian_locality/stationary_rho${RHO}
    echo "locality: rho=$RHO -> $OUT"
    "$PYTHON" experiments/exp_11_nongaussian_locality.py \
        --only stationary \
        --output-dir "$OUT" \
        --set rhos="($RHO,)"
    ;;

  closure)
    # One task, all three rho values: exp_03 takes no --set overrides and sweeps RHOS
    # itself. Writes innovation_sweep_stationary.csv beside the committed baseline CSV
    # rather than over it.
    OUT=$ROOT/outputs/exp_03_nongaussian_innovation_sweep
    echo "closure: all rho -> $OUT"
    "$PYTHON" experiments/exp_03_nongaussian_innovation_sweep.py \
        --only stationary \
        --output-dir "$OUT"
    ;;

  *)
    echo "unknown MODE=$MODE (expected smoke, locality or closure)" >&2
    exit 2
    ;;
esac

echo
echo "end=$(date -Is)"
