#!/bin/bash
#SBATCH --job-name=stabil
#SBATCH --partition=medium_gpuh200
#SBATCH --gres=gpu:1
#SBATCH --time=05:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=$HOME/nongaussian-bp/logs/%x_%A_%a.out
#SBATCH --error=$HOME/nongaussian-bp/logs/%x_%A_%a.err
#SBATCH --propagate=NONE
#
# The two runs the review names as required before the headline can be called
# stable, both of which are cheap in wall clock and expensive only in queue
# time -- so they go out together, one seed per array task.
#
# MODE=resolve  (review Priority 4)
#   16 of 1,728 certified cells are EXCLUDED from the headline table because
#   the narrowest fitted mixture component came out under two grid cells wide.
#   The exclusion is disclosed and moves no ratio by more than 0.08, but the
#   criterion is a property of the fitted estimator itself, so the dropped
#   cells are not missing at random: it is exactly the bad estimator outcomes
#   that get dropped. Refitting at M = 801 replaces exclusion with resolution
#   and removes the argument entirely. Same data, same budgets, same seeds --
#   only the mesh changes.
#
# MODE=budget  (review Priority 2)
#   Both arms often select their LARGEST offered budget at the big sample
#   sizes, so the selection is right-censored: validation error is still
#   falling when the search range ends. One calibration point exists, at
#   nseq=2048, and it does not certify nseq=4096 or 8192. This runs the top
#   three sizes at 2x and 4x the caps on the same seeds, so saturation can be
#   read off a curve rather than asserted. Acceptance is either an interior
#   validation optimum or <1% test-error movement under the final doubling.
#
# Both write beside the frozen outputs rather than into them; nothing here
# overwrites a certified directory.
#
#   Gate:     sbatch --parsable --time=00:40:00 --array=0 \
#                    --export=ALL,MODE=resolve,SMOKE=1 hpc/bocconi_stabilise.sbatch
#   Resolve:  sbatch --dependency=afterok:$G --array=0-15 \
#                    --export=ALL,MODE=resolve hpc/bocconi_stabilise.sbatch
#   Budget:   sbatch --dependency=afterok:$G --array=0-15 \
#                    --export=ALL,MODE=budget  hpc/bocconi_stabilise.sbatch
#
# NOTE on --export: values are split on commas by sbatch, so no value here may
# contain one. MODE and SMOKE are bare tokens for that reason; every tuple is
# built inside the script. This cost a whole sweep on 2026-08-19.

set -euo pipefail

source $HOME/nongaussian-bp/gpu_env.sh
export BP_DEVICE=gpu

ROOT=$HOME/nongaussian-bp
PYTHON=$ROOT/.venv/bin/python
cd "$ROOT/research/nongaussian-bp"

SEED=${SLURM_ARRAY_TASK_ID:-0}
MODE=${MODE:-resolve}
SMOKE=${SMOKE:-0}

echo "host=$(hostname) seed=$SEED mode=$MODE smoke=$SMOKE"
nvidia-smi --query-gpu=name --format=csv,noheader
"$PYTHON" -c "from src.backend import get_xp, device_name; print('device:', device_name(get_xp()))"

# BP_DEVICE=gpu is an explicit request, so a silent numpy fallback must be a
# failure and not a warning: the whole point of this sweep is that the numbers
# come off the hardware they claim to.
# An all-skipped parity run exits 0, so pytest's status alone cannot gate this.
# tests/test_backend_parity.py skips its whole module when no device is visible,
# which is correct on a laptop and catastrophic here: the sweep would proceed on
# numpy having "passed" its GPU check. Assert the device FIRST, so the tests
# actually execute and the exit code means something.
"$PYTHON" -c "import sys; from src.backend import gpu_available; sys.exit(0 if gpu_available() else 1)" \
    || { echo "NO USABLE GPU - refusing to run a GPU sweep on the CPU"; exit 3; }
"$PYTHON" -m pytest tests/test_backend_parity.py -q \
    || { echo "PARITY FAILED - refusing to run"; exit 3; }

case "$MODE" in
  resolve)
    # M = 801 doubles the mesh; the narrowest production component sits at
    # ~9.4 cells at M=401, so the 16 flagged cells clear two cells with room.
    OUT=outputs/exp_07_m801_seed${SEED}
    GRID=801
    [[ "$SMOKE" == "1" ]] && GRID=401
    SIZES="(32,64,128)"
    $PYTHON experiments/exp_07_em_vs_score_network.py \
        --only sample_efficiency_val \
        --output-dir "$OUT" \
        --set "seed=$SEED" \
        --set "grid_size=$GRID" \
        --set "sizes=$SIZES"
    ;;

  budget)
    # 2x and 4x on BOTH arms, so the ratio cannot move merely because one arm
    # was given more room than the other.
    for MULT in 2 4; do
      OUT=outputs/exp_07_budget${MULT}x_seed${SEED}
      EM=$(( 3200 * MULT ))
      NET=$(( 100000 * MULT ))
      if [[ "$SMOKE" == "1" ]]; then EM=200; NET=2000; fi
      echo "--- budget x${MULT}: em_iters=$EM net_steps=$NET"
      $PYTHON experiments/exp_07_em_vs_score_network.py \
          --only sample_efficiency_val \
          --output-dir "$OUT" \
          --set "seed=$SEED" \
          --set "sizes=(1024,2048,4096)" \
          --set "em_iters=$EM" \
          --set "net_steps=$NET"
    done
    ;;

  *) echo "unknown MODE=$MODE" >&2; exit 2 ;;
esac

echo "done=$(date -Is)"
