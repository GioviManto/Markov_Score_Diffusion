#!/bin/bash
#SBATCH --job-name=exp16
#SBATCH --partition=medium_cpu
#SBATCH --time=05:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%A_%a.err
#
# exp_16 (sampling validation) on the Bocconi cluster.
#
# Unlike the templates in this directory, this one is filled in for a specific site:
# partition `medium_cpu`, account implicit, and the QOS caps that actually apply here
# (30 jobs submitted, 10 running per user) -- so array sizes above 30 will be rejected
# outright rather than queued.
#
#   Calibration (must run and be inspected FIRST -- see below):
#     sbatch --array=0-5 --export=ALL,MODE=calibrate bocconi_exp16.sbatch
#
#   Main run, one task per (size, seed) cell:
#     sbatch --array=0-15 --export=ALL,MODE=generate,NSTEPS=<from calibration> \
#            bocconi_exp16.sbatch
#
#   Gate (runs the test suite and a --quick pass; chain everything else behind it):
#     SMOKE=$(sbatch --parsable --time=00:40:00 --export=ALL,MODE=smoke bocconi_exp16.sbatch)
#
#   Likelihood and runtime against capacity, one task per C:
#     sbatch --dependency=afterok:$SMOKE --array=0-4 --export=ALL,MODE=cpoint2 \
#            bocconi_exp16.sbatch
#
#   Capacity vs generated shape, rerun at a converged EM budget (see the mode body for why):
#     sbatch --partition=defq --time=24:00:00 --array=2 \
#            --export=ALL,MODE=components_converged bocconi_exp16.sbatch   # calibrate first
#     sbatch --partition=defq --time=24:00:00 --array=0-14 \
#            --export=ALL,MODE=components_converged bocconi_exp16.sbatch   # then the rest
#
# WHY CALIBRATION COMES FIRST. The reference arm of this experiment is the *exact* score,
# and a smoke run showed it does not reproduce the true innovation kurtosis at small step
# counts: 1.78 at 60 steps and 2.32 at 120, against a true 3.0. That is the Euler-Maruyama
# discretization, not the score. Until the exact arm reproduces the truth, a comparison
# between arms measures the integrator. So: run MODE=calibrate, read off the step count at
# which the exact arm's kurtosis stops moving, and pass it as NSTEPS to the main run.
# Do not skip this and do not guess the number.
#
# Single-threaded on purpose: the per-task linear algebra is a 401x401 transition matrix,
# so thread parallelism buys little while one CPU per task lets many tasks pack onto a node.

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ROOT=$HOME/nongaussian-bp
PYTHON=$ROOT/.venv/bin/python
CODE=$ROOT/research/nongaussian-bp

cd "$CODE"

TASK=${SLURM_ARRAY_TASK_ID:-0}
MODE=${MODE:-generate}

echo "host=$(hostname)  job=${SLURM_JOB_ID:-NA}  task=$TASK  mode=$MODE"
echo "python=$($PYTHON -V 2>&1)  numpy=$($PYTHON -c 'import numpy;print(numpy.__version__)')"
echo "start=$(date -Is)"
echo

case "$MODE" in

  calibrate)
    # One task per step count. Only the `exact` arm matters here, but the part runs all
    # four; the others are cheap by comparison and their trend is informative too.
    LADDER=(100 200 400 800 1600 3200)
    NS=${LADDER[$TASK]}
    OUT=$ROOT/outputs/exp_16/calibrate_steps${NS}
    echo "calibrating at n_steps=$NS -> $OUT"
    "$PYTHON" experiments/exp_16_sampling_validation.py \
        --only steps \
        --output-dir "$OUT" \
        --set steps_ladder="($NS,)" \
        --set n_generate=2000 \
        --set sizes="(2048,)" \
        --set n_seed=1
    ;;

  generate)
    # One task per (size, seed). Cells write disjoint output directories, so there is no
    # merge step beyond concatenating CSVs.
    SIZES=(32 128 512 2048)
    NSEED=4
    SI=$(( TASK / NSEED ))
    SEED=$(( TASK % NSEED ))
    SZ=${SIZES[$SI]}
    NSTEPS=${NSTEPS:?"NSTEPS must be set from the calibration run"}
    OUT=$ROOT/outputs/exp_16/gen_n${SZ}_seed${SEED}
    echo "generate: size=$SZ seed=$SEED n_steps=$NSTEPS -> $OUT"
    "$PYTHON" experiments/exp_16_sampling_validation.py \
        --only generate,pointwise \
        --output-dir "$OUT" \
        --set sizes="($SZ,)" \
        --set n_seed=1 \
        --set n_steps="$NSTEPS" \
        --set seed_offset="$SEED"
    ;;

  components)
    # Does EM-BP's under-dispersed generated tail come from the ESTIMATOR'S CAPACITY or from
    # the INFORMATION in noisy data? The main sweep fits a 4-component innovation mixture and
    # generates samples with excess kurtosis 0.897 against a target of 1.910, while its
    # pointwise error is the best of every arm by 10x. Two readings, and they differ:
    #
    #   capacity-limited  -> kurtosis should climb toward 1.910 as C grows, since exp_06
    #                        showed the FITTED innovation law improves monotonically with C
    #                        (2.71 at C=8 against 1.95 at C=2).
    #   information-limited -> it should plateau, because the Fisher information for the
    #                        innovation SCALE falls 142x between t=0.05 and t=1.6 while that
    #                        for the correlation falls only 26x. Non-Gaussianity is the first
    #                        thing the channel destroys, and EM never sees a clean sample.
    #
    # One task per component count, at the largest data budget so capacity is the only
    # binding constraint.
    CS=(2 4 8 12 16)
    C=${CS[$TASK]}
    OUT=$ROOT/outputs/exp_16/components_C${C}
    echo "components: C=$C -> $OUT"
    "$PYTHON" experiments/exp_16_sampling_validation.py \
        --only generate \
        --output-dir "$OUT" \
        --set sizes="(2048,)" \
        --set em_components="$C" \
        --set n_seed=3 \
        --set n_steps=400
    ;;

  components_converged)
    # `components` above, and the fig_capacity curve it feeds, used the default
    # em_iters=40. audit/AUDIT_NOTE.md's audit_rate_vs_bias.py measured that the innovation
    # shape needs on the order of 2000 EM iterations to settle -- 40 is a ~50x under-budget
    # -- and a direct check attributed roughly 96% of the apparent capacity effect at C=40
    # iterations to convergence rate, not capacity (CLAIMS_TO_UPDATE.md #2). This mode reruns
    # the same sweep at em_iters=2000 so the generated-shape vs C curve stops being confounded
    # with under-convergence. The held-out-evidence/runtime curve (`cpoint2`) is NOT
    # rerun here: CLAIMS_TO_UPDATE.md #2 already argues that one is unaffected, because it
    # is a paired comparison scored by an average over the whole density that saturates
    # early in the optimisation.
    #
    # One task per (C, seed) rather than per C with 3 seeds looped inside -- at 2000
    # iterations the EM fit alone is of order 2000 x 14s =~ 7.8h (13.5-14.3s/iter measured
    # at the old budget, assumed to scale linearly), so 3 seeds in one task would not fit
    # any partition's walltime cap with margin. 15 tasks (5 C x 3 seeds) instead, each
    # requesting the whole of defq's 24:10:00 cap for headroom over the generation cost on
    # top of the EM fit (unmeasured at this iteration count -- this is why cell 0 should be
    # submitted alone first as a timing check before the other 14; see hpc/README or ask
    # before blindly submitting the full array).
    #
    #   Calibration cell (submit alone, inspect elapsed time before the rest):
    #     sbatch --partition=defq --time=24:00:00 --array=2 \
    #            --export=ALL,MODE=components_converged bocconi_exp16.sbatch
    #
    #   Full array, once the calibration cell's walltime is known to fit:
    #     sbatch --partition=defq --time=24:00:00 --array=0-14 \
    #            --export=ALL,MODE=components_converged bocconi_exp16.sbatch
    CS=(2 4 8 12 16)
    NSEED=3
    CI=$(( TASK / NSEED ))
    SEED=$(( TASK % NSEED ))
    C=${CS[$CI]}
    OUT=$ROOT/outputs/exp_16/components_converged_C${C}_seed${SEED}
    echo "components_converged: C=$C seed=$SEED em_iters=2000 -> $OUT"
    "$PYTHON" experiments/exp_16_sampling_validation.py \
        --only generate \
        --output-dir "$OUT" \
        --set sizes="(2048,)" \
        --set em_components="$C" \
        --set n_seed=1 \
        --set seed_offset="$SEED" \
        --set em_iters=2000 \
        --set n_steps=400
    ;;

  cpoint2)
    # The two coordinates the capacity sweep never had: marginal likelihood and cost.
    #
    # The `cpoint` array settled that pointwise error and generated tail fidelity both
    # improve monotonically in C, so the two axes do not trade off in 2..16. It could not
    # say whether the extra components are *justified*, because it never looked at the
    # likelihood, nor what they cost. Both were already being computed inside every EM run
    # and discarded at the end of fit_arms.
    #
    # Read the two curves differently. Training evidence is monotone in C by nesting -- a
    # C-component mixture contains every smaller one -- so it carries no model-selection
    # content and is emitted only as a correctness check. The held-out curve, on fresh
    # chains noised the same way, is the one that can turn over.
    #
    # Same array shape as `cpoint`, which completed cleanly as 617485_[0-4], so the runtime
    # is known: pointwise and heldout each refit every arm at each C.
    CS=(2 4 8 12 16)
    C=${CS[$TASK]}
    OUT=$ROOT/outputs/exp_16/cpoint2_C${C}
    echo "cpoint2: C=$C -> $OUT"
    "$PYTHON" experiments/exp_16_sampling_validation.py \
        --only pointwise,heldout \
        --output-dir "$OUT" \
        --set sizes="(2048,)" \
        --set em_components="$C" \
        --set n_seed=3
    ;;

  generate2)
    # THE headline rerun. Same design as `generate`, but under the P0.B protocol: networks
    # trained continuously over [t_min, t_max] instead of on five discrete levels, and one
    # parameterisation chosen per arm on validation instead of switched per probe level
    # mid-integration.
    #
    # Writes to gen2_* rather than gen_*, deliberately. The committed gen_* outputs were
    # produced under the old protocol and the DIFFERENCE between the two is the quantity that
    # answers the review's first and most serious point -- how much of the reported
    # pointwise-versus-generative dissociation was the training/evaluation mismatch rather
    # than the score estimators. Overwriting them would destroy that comparison.
    SIZES=(32 128 512 2048)
    NSEED=4
    SI=$(( TASK / NSEED ))
    SEED=$(( TASK % NSEED ))
    SZ=${SIZES[$SI]}
    NSTEPS=${NSTEPS:-400}
    OUT=$ROOT/outputs/exp_16/gen2_n${SZ}_seed${SEED}
    echo "generate2: size=$SZ seed=$SEED n_steps=$NSTEPS -> $OUT"
    "$PYTHON" experiments/exp_16_sampling_validation.py \
        --only generate,pointwise \
        --output-dir "$OUT" \
        --set sizes="($SZ,)" \
        --set n_seed=1 \
        --set n_steps="$NSTEPS" \
        --set seed_offset="$SEED"
    ;;

  smoke)
    # The gate. Nothing in this file runs on a laptop, so the test suite and the --quick
    # pass have to run here too; production arrays are chained behind this with
    # --dependency=afterok so a broken tree never reaches a six-hour allocation.
    echo "--- pytest ---"
    "$PYTHON" -m pytest tests/ -q
    echo "--- exp_16 --quick ---"
    "$PYTHON" experiments/exp_16_sampling_validation.py \
        --quick --only pointwise,heldout \
        --output-dir "$ROOT/outputs/smoke/exp_16"
    ;;

  *)
    echo "unknown MODE=$MODE (expected smoke, calibrate, generate, generate2, components," \
         "components_converged or cpoint2)" >&2
    exit 2
    ;;
esac

echo
echo "end=$(date -Is)"
