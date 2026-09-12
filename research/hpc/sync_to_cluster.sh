#!/usr/bin/env bash
# Push the working tree to the Bocconi cluster, additively, and verify.
#
# WHY THIS EXISTS. On 2026-08-11 two agents were syncing two different git
# worktrees into the same cluster directory, both with `rsync --delete`. Each
# deletion looked correct from inside its own branch: files belonging to the
# other branch are simply absent locally, so --delete removes them remotely.
# The EM branch lost src/wavelet.py, src/video_*.py and exp_23..exp_26; the
# wavelet branch lost exp_27_shape_convergence.py, src/clean_mle.py and
# hpc/bocconi_final_em.sbatch. In both cases a sweep was queued against a tree
# missing the files it calls, and one gate job failed for exactly that reason.
#
# So: NO --delete, ever, from this script. A stale extra file on the cluster is
# harmless. A deleted one costs a day of queue time and produces nothing.
#
#   ./hpc/sync_to_cluster.sh            # sync and verify
#   ./hpc/sync_to_cluster.sh --check    # verify only, exit 1 if anything is missing
set -euo pipefail

# Cluster coordinates come from the environment, never from this file:
# a public repository should not carry an institutional account number
# or an internal hostname. Set BP_CLUSTER_HOST and BP_CLUSTER_USER (a
# ~/.bp_cluster_env sourced from your shell profile is the easy way).
HOST=${BP_CLUSTER_HOST:-}
if [[ -z "${HOST:-}" ]]; then
    echo "Set BP_CLUSTER_HOST first." >&2
    exit 2
fi

REMOTE=${BP_CLUSTER_ROOT:-nongaussian-bp}
cd "$(dirname "$0")/.."                       # research/nongaussian-bp

# Files each branch owns. Both lists are checked after every sync, whichever
# branch you are on, because the failure mode is deleting the OTHER branch's work.
EM_FILES=(experiments/exp_27_shape_convergence.py src/clean_mle.py
          hpc/bocconi_final_em.sbatch src/em.py src/kernels.py)
WAVELET_FILES=(src/wavelet.py src/wavelet_bp.py src/video_bp.py src/video_model.py
               src/image_data.py experiments/exp_23_wavelet_statistics.py
               experiments/exp_26_video.py hpc/bocconi_wavelet.sbatch)
# Downloaded datasets live on the cluster and in nobody's branch, so --delete ate a 163 MB
# CIFAR tarball too and a wavelet job died 10 seconds in for want of it. Data is checked like
# code, and excluded from the push so a laptop copy can never overwrite the cluster's.
DATA_FILES=(data/cifar-10-python.tar.gz)

verify() {
    local missing=0
    for f in "${EM_FILES[@]}" "${WAVELET_FILES[@]}" "${DATA_FILES[@]}"; do
        ssh -o BatchMode=yes "$HOST" "test -e '$REMOTE/research/nongaussian-bp/$f'" \
            || { echo "MISSING on cluster: $f"; missing=1; }
    done
    if [[ $missing -eq 0 ]]; then
        echo "cluster has both branches' files"
    else
        echo
        echo "Restore from the branch that owns them before submitting anything."
        echo "The wavelet/video files live in the eloquent-wu-07a351 worktree;"
        echo "data/ is a download -- see the curl command in hpc/bocconi_wavelet.sbatch."
    fi
    return $missing
}

if [[ "${1:-}" == "--check" ]]; then verify; exit $?; fi

if [[ "${1:-}" == *--delete* ]]; then
    echo "refusing: --delete is what broke this before. See the header." >&2
    exit 2
fi

echo "syncing $(git rev-parse --short HEAD 2>/dev/null || echo 'no-git') -> $HOST:$REMOTE"
rsync -az --exclude '.git' --exclude '.claude' --exclude '__pycache__' \
      --exclude '*.pyc' --exclude '.venv' --exclude 'outputs/final_em' --exclude 'data' \
      ./ "$HOST:$REMOTE/research/nongaussian-bp/"
[[ -d ../../tools ]] && rsync -az ../../tools/ "$HOST:$REMOTE/tools/"
echo "synced. verifying..."
verify
