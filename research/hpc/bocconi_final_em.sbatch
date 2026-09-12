#!/bin/bash
#SBATCH --job-name=finalem
#SBATCH --partition=compute
#SBATCH --time=2-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%A_%a.err
#
# The final EM-BP sweep set. Filled in for Bocconi: partition `compute`
# (3-day limit, cnode05-07), account implicit. QOS `normal` caps at 30 SUBMITTED
# and 10 RUNNING tasks per user, so no single submission may exceed 30 and the
# waves below are sized to stay under it together.
#
# Pin the run root ONCE, before wave 1, and pass it to every wave. See the
# RUN_STAMP note further down for what happens if you do not.
#   S=$(date +%Y%m%d)
#
# Wave 1 (submit together, 20 tasks):
#   GATE=$(sbatch --parsable --partition=medium_cpu --time=00:45:00 \
#          --export=ALL,RUN_STAMP=$S,MODE=smoke hpc/bocconi_final_em.sbatch)
#   sbatch --dependency=afterok:$GATE --array=0-15%8 --cpus-per-task=4 --mem=12G \
#          --export=ALL,RUN_STAMP=$S,MODE=reps hpc/bocconi_final_em.sbatch
#   sbatch --dependency=afterok:$GATE --array=0-3%4 --time=1-12:00:00 \
#          --export=ALL,RUN_STAMP=$S,MODE=recovery hpc/bocconi_final_em.sbatch
#
# Wave 2 (all 18 shape cells, packed 6 to a job -- only 3 tasks):
#   sbatch --array=0-2 --cpus-per-task=12 --mem=64G --time=1-00:00:00 \
#          --nice=200 --export=ALL,RUN_STAMP=$S,MODE=shape,CELLS_PER_TASK=6 \
#          hpc/bocconi_final_em.sbatch
#
# `shape` also runs happily on medium_cpu (PriorityJobFactor 200 against
# compute's 10, and a far shorter queue): six cells in parallel finish well
# inside the 6:10:00 cap. Use --partition=medium_cpu --time=06:00:00 when
# compute is congested -- that is what got wave 2 started on 2026-08-12.
#
# The --nice=200 is deliberate: wave 2 must yield to wave 1. Both waves fit
# under the 30-submit ceiling at once now (20 + 3), but the 10-RUNNING ceiling
# is still shared, and `reps` is the headline table. nice lowers wave 2 in the
# ordering without holding it, so it takes slots only when wave 1 is not.
#
# WHY `compute` AND NOT `defq`, AND WHY THIS IS NOT NEGOTIABLE. `defq` is the
# cluster default and it looks attractive -- 176 CPUs against 120, six nodes
# against three. It is the wrong answer twice over, and on 2026-08-11 three
# arrays were cancelled off `compute` and re-submitted onto it, which is what
# this paragraph exists to prevent happening again:
#
#   1. defq MaxTime is 1-00:10:00. `reps` cannot be given more than 23:50 there.
#      Nothing in this script checkpoints, and the .ok marker below is written
#      only after the payload returns 0, so a walltime kill contributes exactly
#      nothing to a merged table. A too-short walltime does not degrade the
#      result, it deletes it -- after burning a day of node time per task.
#   2. defq is where everyone else already is. Measured 2026-08-11 20:0x:
#      defq 46 pending / 14 running, compute 1 pending / 4 running. The default
#      partition is congested precisely because it is the default.
#
# `compute` costs PriorityJobFactor 10 against defq's 50, so a compute job
# starts lower in the ordering. That is the trade being made deliberately: a
# later start on a partition that can actually finish the job beats an earlier
# start on one that will kill it at hour 24.
#
# Read the walltime guard below before changing any --time or --partition here.
#
# WHY THE GATE. Every downstream conclusion depends on two corrections made on
# 2026-08-10: the sign of the mixture rho gradient, and the returned-iterate
# alignment of the gradient route. A sweep launched against a stale checkout
# would burn days and produce numbers indistinguishable from the ones we are
# replacing. MODE=smoke runs the suite and refuses to let anything start if it
# fails, so a stale or half-synced tree cannot silently become a result.
#
# WHY CPU AND NOT GPU. The recursion carries a CuPy backend (src/backend.py) and
# does run on the A100 nodes, but this cluster has two of them against eight CPU
# nodes, and the QOS ceiling of 10 running tasks is what actually bounds this
# sweep. Twelve independent CPU tasks finish sooner than two faster GPU ones.
# The GPU path is validated separately by hpc/bocconi_gpu.sbatch, which runs the
# CPU/GPU parity suite before it is allowed to produce anything.

set -euo pipefail

ROOT=$HOME/nongaussian-bp
PYTHON=$ROOT/.venv/bin/python
cd "$ROOT/research/nongaussian-bp"
mkdir -p "$ROOT/logs"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

TASK=${SLURM_ARRAY_TASK_ID:-0}
MODE=${MODE:-smoke}
# ---------------------------------------------------------------------------
# The run root, and a trap it set on 2026-08-12.
#
# STAMP used to be an unconditional `date +%Y%m%d`, evaluated inside each task at
# the moment that task started. A sweep whose tasks are throttled (%8) does not
# start all at once, so the 16-task `reps` array straddled midnight and wrote 12
# seeds into outputs/final_em/20260811/ and 4 into 20260812/. Nothing failed and
# nothing warned; the sweep is simply in two places. Anything that globs a single
# date silently reads 12 of 16 seeds and reports a complete-looking table.
#
# RUN_STAMP makes the root an input rather than a side effect of when a node
# happened to free up. Pass the same value to every wave of one sweep:
#
#   S=$(date +%Y%m%d)
#   sbatch --export=ALL,RUN_STAMP=$S,MODE=reps ...
#
# The default is kept so an un-exported submission still works, but a sweep that
# matters should set it. Merge steps should glob outputs/final_em/*/ regardless,
# since the split runs already on disk cannot be un-split.
# ---------------------------------------------------------------------------
STAMP=${RUN_STAMP:-$(date +%Y%m%d)}
RUN_ROOT="outputs/final_em/${STAMP}"

# ---------------------------------------------------------------------------
# Walltime preflight.
#
# The failure this prevents is silent and total: a task killed at the walltime
# never reaches the .ok marker, so the merge step skips it and the merged table
# is simply missing those cells -- with no error anywhere except a log nobody
# reads. It costs a day of node time per task to find out. Refuse to start at
# all rather than start something that cannot finish.
#
# The floors below are what each mode was sized for, not measured runtimes.
# `reps` in particular has never been timed at this configuration: it extends
# `sizes` to 4096 (the full default stops at 2048) and triples net_steps to
# 60000, so it is heavier than any exp_07 run on record. 36 h is therefore a
# floor on the *request*, chosen so the job cannot land on a partition that
# caps below it -- defq (1-00:10:00) and medium_cpu (6:10:00) both fail it.
# Replace these with measured numbers once a reps task has actually finished.
# ---------------------------------------------------------------------------
mode_min_minutes() {
    case "$1" in
        smoke)    echo 30   ;;
        recovery) echo 360  ;;   # 6 h
        shape)    echo 360  ;;   # 6 h
        reps)     echo 2160 ;;   # 36 h -- only `compute` can grant this
        *)        echo 0    ;;
    esac
}

# "2-12:00:00" | "23:50:00" | "45:00" | "30" | "UNLIMITED" -> minutes
slurm_time_to_minutes() {
    local t="$1" d=0 h=0 m=0 s=0
    if [ "$t" = "UNLIMITED" ] || [ -z "$t" ]; then echo 999999; return; fi
    # Reject anything that is not digits, colons and dashes before it reaches the
    # arithmetic. A bad $(( )) is a fatal shell error under `set -e` that no
    # `|| echo 0` at the call site can catch, so it has to be refused here.
    case "$t" in *[!0-9:-]*) echo 0; return ;; esac
    case "$t" in *-*) d=${t%%-*}; t=${t#*-} ;; esac
    local IFS=:
    # shellcheck disable=SC2086
    set -- $t
    case $# in
        3) h=$1; m=$2; s=$3 ;;
        2) m=$1; s=$2 ;;
        1) m=$1 ;;
    esac
    echo $(( 10#${d:-0} * 1440 + 10#${h:-0} * 60 + 10#${m:-0} \
             + ( 10#${s:-0} > 0 ? 1 : 0 ) ))
}

# `head -n1` is load-bearing. For an array job `squeue -j` can print one row per
# task, and without it the values concatenate: on 2026-08-12 task 622854_2 read
# its own limit as "6:00:006:00:006:00:00", parsed that as 0 minutes, and refused
# to start a job whose walltime was in fact correct. Its two siblings, which
# happened to query at a moment when only one row matched, ran fine.
GRANTED_RAW=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o "%l" 2>/dev/null \
              | head -n1 | tr -d '[:space:]' || true)
if [ -n "${GRANTED_RAW:-}" ]; then
    # `|| echo 0` so a string the parser cannot handle at all becomes the
    # fail-open case below rather than an arithmetic error that aborts under
    # `set -e` -- which would be the same false positive by another route.
    GRANTED=$(slurm_time_to_minutes "$GRANTED_RAW" 2>/dev/null || echo 0)
    GRANTED=${GRANTED:-0}
    NEED=$(mode_min_minutes "$MODE")
    if [ "$GRANTED" -le 0 ]; then
        # Unparseable. This guard exists to catch a mis-parameterised submission,
        # so it must not become an independent way to lose a good one: a real
        # too-short walltime parses fine, and anything that does not parse is a
        # bug here rather than evidence about the allocation. Warn and proceed.
        echo "preflight: could not parse walltime '${GRANTED_RAW}' -- proceeding" \
             "unchecked (mode=$MODE, partition=${SLURM_JOB_PARTITION:-unknown})" >&2
    elif [ "$GRANTED" -lt "$NEED" ]; then
        echo "REFUSING TO START: mode=$MODE was granted ${GRANTED_RAW}" \
             "(${GRANTED} min) on partition ${SLURM_JOB_PARTITION:-unknown}," \
             "but needs at least ${NEED} min." >&2
        echo "  Nothing here checkpoints. Starting would burn the allocation" \
             "and write no usable output. Resubmit on a partition that can" \
             "grant it: compute has MaxTime=3-00:10:00, medium_cpu 06:10:00," \
             "defq only 1-00:10:00." >&2
        exit 78
    else
        echo "preflight ok: mode=$MODE granted=${GRANTED_RAW} (${GRANTED} min)" \
             "partition=${SLURM_JOB_PARTITION:-unknown}"
    fi
fi

echo "host=$(hostname) mode=$MODE task=$TASK cpus=${SLURM_CPUS_PER_TASK:-?} \
job=${SLURM_JOB_ID:-none} array=${SLURM_ARRAY_JOB_ID:-none}"
$PYTHON -c "import numpy,scipy;print('numpy',numpy.__version__,'scipy',scipy.__version__)"

case "$MODE" in

smoke)
    # The gate. Nothing else is allowed to run if this fails.
    $PYTHON -m pytest tests/ -q
    $PYTHON experiments/exp_27_shape_convergence.py --quick \
        --output-dir "$RUN_ROOT/smoke/exp27"
    $PYTHON experiments/exp_06_em_parameter_recovery.py --quick \
        --only clean_raw_mle --output-dir "$RUN_ROOT/smoke/exp06"
    $PYTHON experiments/exp_07_em_vs_score_network.py --quick \
        --only sample_efficiency_val --output-dir "$RUN_ROOT/smoke/exp07"
    $PYTHON ../../tools/audit_em_bp_provenance.py
    echo "SMOKE OK"
    ;;

reps)
    # Headline sample-efficiency table at 16 independent training seeds, up from
    # the 6 the current table rests on. `sample_efficiency_val` is used rather
    # than `sample_efficiency` because it emits the validation-selected AND the
    # test-oracle column from the SAME fitted models -- so the honest protocol
    # and the sensitivity analysis cannot drift apart, and the networks are not
    # refitted twice.
    SEED="$TASK"
    OUT="$RUN_ROOT/reps/seed_${SEED}"
    $PYTHON experiments/exp_07_em_vs_score_network.py \
        --only sample_efficiency_val \
        --output-dir "$OUT" \
        --set "seed=$SEED" \
        --set 'sizes=(32,64,128,256,512,1024,2048,4096)' \
        --set net_steps=60000
    ;;

recovery)
    # Parameter recovery at replication high enough to settle the questions that
    # three replicates could not. Task 0 is the one that matters most: it is the
    # grid-free clean MLE against the grid-binned fit, which is what retired the
    # "discretisation floor" reading of the clean arm.
    case "$TASK" in
      0) $PYTHON experiments/exp_06_em_parameter_recovery.py --only clean_raw_mle \
             --set n_rep_raw=64 --output-dir "$RUN_ROOT/recovery/clean_raw" ;;
      1) $PYTHON experiments/exp_06_em_parameter_recovery.py --only clean_vs_noised \
             --set n_rep_clean=32 --output-dir "$RUN_ROOT/recovery/clean_vs_noised" ;;
      2) $PYTHON experiments/exp_06_em_parameter_recovery.py --only rate \
             --set n_rep_rate=32 --set 'sizes_rate=(64,128,256,512,1024,2048)' \
             --output-dir "$RUN_ROOT/recovery/rate" ;;
      3) $PYTHON experiments/exp_06_em_parameter_recovery.py \
             --only monotonicity,quantization \
             --set n_inits=16 --output-dir "$RUN_ROOT/recovery/monotonicity" ;;
      *) echo "no recovery cell $TASK" >&2; exit 1 ;;
    esac
    ;;

shape)
    # The convergence sweep: 4 seeds and 800 updates per (N, C, design) cell,
    # because "is 120 iterations enough" has a different answer in every cell and
    # the claim has been revised three times for want of this measurement.
    #
    # PACKED, AND WHY. The binding constraint on this cluster is QOS `normal`,
    # which caps 30 SUBMITTED and 10 RUNNING *jobs* per user. It does not cap
    # CPUs at all -- MaxTRESPU is gres/gpu=6,gres/shard=64 and says nothing about
    # cpu. So the cost of a cell is measured in submit slots, not cores, and the
    # 18 cells were originally one-per-task: 18 of your 30 slots to keep 8 cells
    # moving (--array=0-17%8). That is the expensive way round.
    #
    # Instead: one array task takes CELLS_PER_TASK cells and runs them
    # CONCURRENTLY, splitting its cores between them. 18 cells become 3 jobs of 6
    # at 12 cores, so all 18 are in flight against 3 submit slots and 3 running
    # slots -- leaving wave 1 the room it needs under the same ceiling. The cells
    # are independent processes over disjoint output dirs, so this is only a
    # question of how the work is boxed, not of what is computed.
    #
    # Each cell writes its OWN marker. A packed task must not be all-or-nothing:
    # if one cell dies, the other five have still produced valid traces and the
    # merge must be able to see them. The task still exits nonzero so the failure
    # is visible, but it does not retract its siblings' work.
    # Overridable, because the first run of this sweep could not answer its own
    # question. At n_updates=800 every `high` design cell settled at 788-797 --
    # that is the trace running out, not a settling time, so for that design the
    # answer is ">= 800" and no more. `low` and `uniform` settled at 30-418 and
    # are not censored. Rerunning only the censored design at a longer trace is
    # much cheaper than rerunning the grid.
    #
    #   SHAPE_DESIGNS="high" SHAPE_UPDATES=2400 CELLS_PER_TASK=3
    #
    # Defaults reproduce the original 18-cell sweep exactly.
    IFS=' ' read -r -a SIZES   <<< "${SHAPE_SIZES:-512 2048}"
    IFS=' ' read -r -a COMPS   <<< "${SHAPE_COMPS:-4 8 16}"
    IFS=' ' read -r -a DESIGNS <<< "${SHAPE_DESIGNS:-low uniform high}"
    N_UPDATES=${SHAPE_UPDATES:-800}
    N_SEEDS=${SHAPE_SEEDS:-4}
    N_CELLS=$(( ${#SIZES[@]} * ${#COMPS[@]} * ${#DESIGNS[@]} ))
    N_DES=${#DESIGNS[@]}
    N_CD=$(( ${#COMPS[@]} * N_DES ))
    CPT=${CELLS_PER_TASK:-6}

    THREADS=$(( ${SLURM_CPUS_PER_TASK:-1} / CPT ))
    if [ "$THREADS" -lt 1 ]; then THREADS=1; fi
    export OMP_NUM_THREADS="$THREADS"
    export OPENBLAS_NUM_THREADS="$THREADS"
    export MKL_NUM_THREADS="$THREADS"

    first=$(( TASK * CPT ))
    if [ "$first" -ge "$N_CELLS" ]; then
        echo "no shape cells for task $TASK (CELLS_PER_TASK=$CPT)" >&2
        exit 1
    fi

    pids=(); labels=()
    for (( k = 0; k < CPT; k++ )); do
        cell=$(( first + k ))
        if [ "$cell" -ge "$N_CELLS" ]; then break; fi
        si=$(( cell / N_CD )); ci=$(( (cell % N_CD) / N_DES )); di=$(( cell % N_DES ))
        N=${SIZES[$si]}; C=${COMPS[$ci]}; D=${DESIGNS[$di]}
        tag="N${N}_C${C}_${D}"
        # A longer rerun must not overwrite the 800-update trace it is checking:
        # both are wanted, and comparing them is the point.
        if [ "$N_UPDATES" -ne 800 ]; then tag="${tag}_u${N_UPDATES}"; fi
        cell_out="$RUN_ROOT/shape/$tag"
        echo "cell $cell -> N=$N C=$C design=$D threads=$THREADS out=$cell_out"
        (
            set -euo pipefail
            $PYTHON experiments/exp_27_shape_convergence.py --only trace \
                --set "sizes=($N,)" --set "components=($C,)" \
                --set "designs=('$D',)" \
                --set "n_seeds=$N_SEEDS" --set "n_updates=$N_UPDATES" \
                --output-dir "$cell_out" \
                > "$ROOT/logs/shape_${tag}_${SLURM_JOB_ID:-0}.log" 2>&1
            mkdir -p "$cell_out"
            date -Is > "$cell_out/.ok_cell"
        ) &
        pids+=("$!"); labels+=("cell $cell ($tag)")
    done

    fail=0
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "OK   ${labels[$i]}"
        else
            echo "FAIL ${labels[$i]} -- see $ROOT/logs/shape_*_${SLURM_JOB_ID:-0}.log" >&2
            fail=1
        fi
    done
    if [ "$fail" -ne 0 ]; then
        echo "shape task $TASK: at least one cell failed" >&2
        exit 1
    fi
    ;;

*)
    echo "unknown MODE=$MODE" >&2; exit 1 ;;
esac

# Success marker, written only after the payload returned 0. `merge` steps skip
# any directory without one, so a task killed by the walltime cannot contribute
# a truncated CSV to a merged table.
MARKER="$RUN_ROOT/.ok_${MODE}_${TASK}"
mkdir -p "$(dirname "$MARKER")"
date -Is > "$MARKER"
echo "DONE mode=$MODE task=$TASK"
