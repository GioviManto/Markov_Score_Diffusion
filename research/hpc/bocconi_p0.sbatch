#!/bin/bash
#SBATCH --job-name=p0fix
#SBATCH --partition=medium_cpu
#SBATCH --time=05:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%A_%a.err
#
# Phase P0: protocol and reproducibility repairs from the 7 August external review.
#
# P0.D, run here:
#
#   * exp_18 seeded from Python's `hash`, which is salted per process, so every committed
#     output under outputs/exp_18/ was not reproducible from a fresh interpreter. Replaced
#     with the project's deterministic rng_for. NOTE what this did NOT invalidate: the
#     kernel column-mass residuals are pure functions of (prior, A, N_g) with no sample in
#     them, so the truncation-versus-quadrature separation stands. Only the edge-mass column
#     was ever sample-dependent, and it was computed from ONE trajectory and described as a
#     bound -- now 256 chains per cell, reported as max and upper quantiles, and called a
#     diagnostic.
#
#   * fit_em returned a kernel one M-step past the last logged evidence, on both the
#     converged and the budget-exhausted path. The returned kernel's likelihood was never
#     evaluated and the monotonicity trace never covered the final update.
#
#   Gate (test suite must pass before regenerating anything):
#     SMOKE=$(sbatch --parsable --time=00:40:00 --export=ALL,MODE=smoke bocconi_p0.sbatch)
#
#   Regenerate exp_18 under deterministic seeds:
#     sbatch --dependency=afterok:$SMOKE --export=ALL,MODE=exp18 bocconi_p0.sbatch
#
#   Determinism check -- run exp18 twice into different directories and diff. This is the
#   only way to demonstrate the seeding fix rather than assert it, since the failure mode is
#   precisely that two processes disagree:
#     sbatch --dependency=afterok:$SMOKE --export=ALL,MODE=determinism bocconi_p0.sbatch

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ROOT=$HOME/nongaussian-bp
PYTHON=$ROOT/.venv/bin/python
cd "$ROOT/research/nongaussian-bp"

MODE=${MODE:-smoke}

echo "host=$(hostname)  job=${SLURM_JOB_ID:-NA}  mode=$MODE"
echo "python=$($PYTHON -V 2>&1)  numpy=$($PYTHON -c 'import numpy;print(numpy.__version__)')"
echo "start=$(date -Is)"
echo

case "$MODE" in

  smoke)
    "$PYTHON" -m pytest tests/ -q
    ;;

  exp18)
    # --out is not optional here. Without it exp_18 writes to its in-repo default
    # (research/nongaussian-bp/outputs/exp_18) while every other experiment writes to
    # $ROOT/outputs, so an rsync of $ROOT/outputs silently misses it and the figures get
    # rebuilt from stale data -- which is exactly what happened once. make_figures.py exits
    # non-zero on a MISSING input but cannot detect a STALE one.
    "$PYTHON" experiments/exp_18_revision_diagnostics.py --parts boundary,emtrace,density \
        --out "$ROOT/outputs/exp_18"
    ;;

  determinism)
    # Two independent interpreters, deliberately with DIFFERENT hash seeds. Under the old
    # `hash`-based seeding these diverge; under rng_for they must agree byte for byte.
    D1=$ROOT/outputs/_determinism/run1
    D2=$ROOT/outputs/_determinism/run2
    rm -rf "$D1" "$D2"; mkdir -p "$D1" "$D2"

    ( cd "$ROOT/research/nongaussian-bp" && PYTHONHASHSEED=1 OUT_OVERRIDE="$D1" \
        "$PYTHON" experiments/exp_18_revision_diagnostics.py --parts boundary --out "$D1" )
    ( cd "$ROOT/research/nongaussian-bp" && PYTHONHASHSEED=999 OUT_OVERRIDE="$D2" \
        "$PYTHON" experiments/exp_18_revision_diagnostics.py --parts boundary --out "$D2" )

    echo "--- diffing boundary.csv across two hash seeds ---"
    if diff -q "$D1/boundary.csv" "$D2/boundary.csv"; then
        echo "DETERMINISM OK: identical under PYTHONHASHSEED=1 and 999"
    else
        echo "DETERMINISM FAILED: outputs differ across hash seeds" >&2
        diff "$D1/boundary.csv" "$D2/boundary.csv" | head -20 >&2
        exit 3
    fi
    ;;

  shape)
    # P0.E: efficient Fisher information for a genuine SHAPE parameter at fixed variance.
    #
    # exp_06 measured information for (rho, q) in a Gaussian model and concluded that
    # higher-order structure is destroyed faster. Both of those are second-order: q is a
    # scale. This runs the generalised-Gaussian family, where beta moves the fourth moment
    # with the variance held exactly at q, and reports the NUISANCE-CORRECTED information
    # 1/(J^{-1})[x,x] rather than the raw diagonal -- comparing diagonals ignores the
    # coupling between coordinates, which was the second half of the same mistake.
    "$PYTHON" experiments/exp_22_shape_information.py \
        --only information \
        --output-dir "$ROOT/outputs/exp_22"
    ;;

  *)
    echo "unknown MODE=$MODE (expected smoke, exp18, determinism or shape)" >&2
    exit 2
    ;;
esac

echo
echo "end=$(date -Is)"
