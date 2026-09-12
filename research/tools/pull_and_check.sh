#!/usr/bin/env bash
# Pull whatever the cluster has finished, then verify what arrived.
#
# WHY THIS EXISTS. Results are produced on a login node and have to be brought
# back by hand, and every step of that has already gone wrong once:
#
#   * A sweep was committed while two of its four tasks were still running, so
#     the repository carried a headline table computed from half an arm.
#   * The reps array straddled midnight and wrote its seeds into two dated run
#     roots; counting one of them shows a complete-looking twelve of sixteen.
#   * A pull reported success and transferred nothing, because `--info=stats2`
#     is unsupported by this rsync and the error went into a pipeline that was
#     grepping for success lines.
#
# So this script pulls, then *counts what it has*, and says plainly whether the
# sweep is complete rather than whether the transfer exited zero.
#
# It is deliberately additive. `--delete` has twice destroyed work here: once
# taking the other branch's source files, once a 163 MB CIFAR archive that
# exists in no branch. There is no flag to enable it.
#
#   ./tools/pull_and_check.sh           # pull, count, run the quick checks
#   ./tools/pull_and_check.sh --no-check   # pull and count only
#
# Exits non-zero if the pull fails or the quick checks fail. A sweep that is
# merely still running is reported, not failed.

set -uo pipefail
cd "$(dirname "$0")/.."

# Cluster coordinates come from the environment, never from this file:
# a public repository should not carry an institutional account number
# or an internal hostname. Set BP_CLUSTER_HOST and BP_CLUSTER_USER (a
# ~/.bp_cluster_env sourced from your shell profile is the easy way).
HOST=${BP_CLUSTER_HOST:-}
USER_AT=${BP_CLUSTER_USER:-}
if [[ -z "${HOST:-}" || -z "${USER_AT:-}" ]]; then
    echo "Set BP_CLUSTER_HOST and BP_CLUSTER_USER first." >&2
    exit 2
fi

REMOTE=${BP_CLUSTER_ROOT:-nongaussian-bp}
# THERE ARE TWO OUTPUTS TREES ON THE CLUSTER, and which one a result lands in
# depends on which sbatch produced it:
#
#   bocconi_final_em.sbatch   RUN_ROOT="outputs/final_em/..." -- relative to its
#                             WorkDir, so research/nongaussian-bp/outputs/
#   bocconi_wavelet.sbatch    "$ROOT/outputs/..." where ROOT=~/nongaussian-bp,
#                             so ~/nongaussian-bp/outputs/ -- one level up
#
# This script pulled only the first for its whole life, which is why exp_24
# reported "0 files" for six hours while fit.csv sat on the login node. Both are
# searched now. Fixing the sbatch scripts to agree would be better and is a
# change to paths that running jobs depend on, so it is not done here.
BASE_RESEARCH="$USER_AT@$HOST:/home/$USER_AT/$REMOTE/research/nongaussian-bp/outputs"
BASE_ROOT="$USER_AT@$HOST:/home/$USER_AT/$REMOTE/outputs"

RUN_CHECKS=1
[[ "${1:-}" == "--no-check" ]] && RUN_CHECKS=0

# Directories worth pulling back. outputs/ as a whole is too blunt: it would drag
# the entire committed result set across on every run.
DIRS=(final_em exp_24_wavelet_fit exp_23_wavelet_statistics
      exp_25_wavelet_generation exp_26_video)

echo "pulling from $HOST"
rc=0
for d in "${DIRS[@]}"; do
    found=0; total=0
    for base in "$BASE_RESEARCH" "$BASE_ROOT"; do
        remote_dir="${base#*:}/$d"
        ssh -o BatchMode=yes -o ConnectTimeout=20 "$USER_AT@$HOST" \
            "test -d '$remote_dir'" 2>/dev/null || continue
        found=1
        mkdir -p "outputs/$d"
        # --stats, not --info=stats2: the latter is unsupported here and fails.
        n=$(rsync -az --stats "$base/$d/" "outputs/$d/" 2>/dev/null \
            | awk '/Number of files transferred|Number of regular files transferred/ {print $NF; exit}')
        if [[ -z "${n:-}" ]]; then
            printf "  %-28s \033[31mPULL FAILED\033[0m from %s\n" "$d" "${base##*:}"; rc=1
        else
            total=$(( total + n ))
        fi
    done
    if [[ $found -eq 0 ]]; then
        printf "  %-28s not on cluster yet\n" "$d"
    else
        printf "  %-28s %s file(s) transferred\n" "$d" "$total"
    fi
done

# ---------------------------------------------------------------------------
echo
echo "sweep completeness (globbing ALL run roots -- see the midnight note above)"
# ---------------------------------------------------------------------------
if [[ -d outputs/final_em ]]; then
    # Only 8-digit dated directories are run roots. final_em/ also holds
    # gpu_bench/ from the GPU parity benchmark, which is not a sweep.
    roots=$(ls -d outputs/final_em/*/ 2>/dev/null \
            | grep -E '/[0-9]{8}/$' | wc -l | tr -d ' ')
    reps=$(ls outputs/final_em/*/.ok_reps_* 2>/dev/null | wc -l | tr -d ' ')
    rec=$(ls outputs/final_em/*/.ok_recovery_* 2>/dev/null | wc -l | tr -d ' ')
    # Exclude rerun cells (<tag>_u<N>) from the 18-cell baseline. They are extra
    # traces of cells already counted, so folding them in reported "19/18" --
    # which is not a completeness figure, it is two different things added up.
    shp=$(find outputs/final_em/*/shape -name .ok_cell 2>/dev/null \
          | grep -v '_u[0-9]*/' | wc -l | tr -d ' ')
    rerun=$(find outputs/final_em/*/shape -name .ok_cell 2>/dev/null \
            | grep -c '_u[0-9]*/' || true)
    printf "  run roots %s\n" "$roots"
    for spec in "reps:$reps:16" "recovery:$rec:4" "shape cells:$shp:18"; do
        name=${spec%%:*}; rest=${spec#*:}; have=${rest%%:*}; want=${rest##*:}
        if [[ "$have" -ge "$want" ]]; then
            printf "  \033[32mcomplete\033[0m  %-12s %s/%s\n" "$name" "$have" "$want"
        else
            printf "  \033[33mpartial \033[0m  %-12s %s/%s\n" "$name" "$have" "$want"
        fi
    done
    if [[ "${rerun:-0}" -gt 0 ]]; then
        printf "  \033[36mextra   \033[0m  %-12s %s cell(s) at a non-default trace length\n" \
               "reruns" "$rerun"
    fi
    if [[ "$roots" -gt 1 ]]; then
        echo "  note: $roots dated run roots -- any tool reading one of them sees a subset."
    fi
else
    echo "  outputs/final_em not present"
fi

echo
echo "uncommitted after pull:"
git status --porcelain outputs 2>/dev/null | head -12 | sed 's/^/  /'
git status --porcelain outputs 2>/dev/null | tail -n +13 | wc -l | tr -d ' ' \
    | xargs -I{} sh -c '[ {} -gt 0 ] && echo "  ... and {} more" || true'

if [[ $RUN_CHECKS -eq 1 ]]; then
    echo
    ./tools/check_all.sh --quick || rc=1
fi

exit $rc
