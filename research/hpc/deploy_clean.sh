#!/usr/bin/env bash
# Ship a reproducible source archive to the cluster. Replaces stamp+rsync.
#
# WHY THIS EXISTS
# ---------------
# The old path was two independent steps: `stamp_revision.sh` recorded the git
# state, then `sync_to_cluster.sh` rsynced the WORKING TREE. Nothing tied them
# together, so editing a file between them produced outputs whose stamp
# described a different program than the one that ran.
#
# That is not hypothetical. outputs/exp_12_scaled/*/params_*.json record
#
#     git_commit = 286b305...
#     overrides  = ["eff_seed0=0", ...]
#
# but 286b305's exp_12_receptive_field.py has no `eff_seed0` setting, and
# `apply_overrides` raises SystemExit on an unknown key. That command would
# have died instantly at that commit. The dirty list does not mention
# exp_12_receptive_field.py either, so the stamp was already stale when it was
# written. The source that produced those numbers is not recoverable.
#
# The fix is to make the archive the unit of deployment, so the thing that is
# hashed is the thing that runs:
#
#   1. refuse to build from a dirty tree             (no untracked drift)
#   2. `git archive` the commit                      (not the working tree)
#   3. SHA-256 the tarball                           (bytes, not beliefs)
#   4. extract on the cluster under the commit id    (no in-place overwrite)
#   5. outputs live OUTSIDE the source checkout      (a run cannot edit itself)
#
# Point 4 matters as much as the rest: the old script extracted over the
# previous deployment, so two half-synced revisions could interleave in one
# directory and no digest of anything would reveal it.
#
# Usage
#   hpc/deploy_clean.sh                 # deploy HEAD
#   hpc/deploy_clean.sh --allow-dirty   # scratch only; stamps source_is_clean=false
#
# Prints the remote source dir and the digest. Job scripts should source the
# emitted env file rather than hardcoding either.
set -euo pipefail

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

REMOTE_ROOT=${BP_CLUSTER_ROOT:-ngbp-runs}
cd "$(dirname "$0")/../../.."                 # repo root (Diffusion/)

ALLOW_DIRTY=0
[[ "${1:-}" == "--allow-dirty" ]] && ALLOW_DIRTY=1

git rev-parse --git-dir >/dev/null 2>&1 || {
    echo "FATAL: not a git repository" >&2; exit 1; }

dirty=$(git status --porcelain)
if [[ -n "$dirty" && $ALLOW_DIRTY -eq 0 ]]; then
    echo "FATAL: working tree is dirty. A deployment built from an uncommitted" >&2
    echo "tree cannot be reconstructed later -- this is the exp_12 defect." >&2
    echo >&2
    echo "$dirty" | sed 's/^/    /' >&2
    echo >&2
    echo "Commit (or stash) first, or pass --allow-dirty for a scratch run whose" >&2
    echo "outputs must never be cited." >&2
    exit 1
fi

commit=$(git rev-parse HEAD)
short=$(git rev-parse --short HEAD)
stamp=$(date -u +%Y%m%dT%H%M%SZ)
tag="${short}-${stamp}"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# `git archive` reads the COMMIT, never the working tree, so an uncommitted edit
# simply is not in the tarball. With --allow-dirty that is the point: the
# deployed code is the committed code, and the stamp says the tree had extra
# changes that did NOT ship.
tarball="$work/ngbp-${tag}.tar"
git archive --format=tar -o "$tarball" HEAD research/nongaussian-bp tools 2>/dev/null \
    || git archive --format=tar -o "$tarball" HEAD

if command -v sha256sum >/dev/null; then
    digest=$(sha256sum "$tarball" | cut -d' ' -f1)
else
    digest=$(shasum -a 256 "$tarball" | cut -d' ' -f1)
fi

# The REVISION file the run reads. key=value so it can grow a field without
# breaking the parser; `dirty` is one line so the format stays flat.
cat > "$work/REVISION" <<EOF
commit=${commit}
archive_sha256=${digest}
deployed_at=${stamp}
dirty=$(echo "$dirty" | tr '\n' ';' | sed 's/;$//')
EOF

SRC="${REMOTE_ROOT}/src/${tag}"
OUT="${REMOTE_ROOT}/outputs/${tag}"

echo "commit  ${commit}"
echo "digest  ${digest}"
echo "remote  ${SRC}"
[[ -n "$dirty" ]] && echo "WARNING: tree dirty; those changes are NOT in the archive"

ssh -o BatchMode=yes "${USER_AT}@${HOST}" "mkdir -p '${SRC}' '${OUT}'"
scp -q "$tarball" "${USER_AT}@${HOST}:${SRC}/source.tar"
scp -q "$work/REVISION" "${USER_AT}@${HOST}:${SRC}/REVISION.tmp"

# Verify the bytes survived the wire before extracting. A truncated scp that
# still extracts is the failure this catches.
remote_digest=$(ssh -o BatchMode=yes "${USER_AT}@${HOST}" \
    "sha256sum '${SRC}/source.tar' | cut -d' ' -f1")
if [[ "$remote_digest" != "$digest" ]]; then
    echo "FATAL: archive digest mismatch after transfer" >&2
    echo "  local  $digest" >&2
    echo "  remote $remote_digest" >&2
    exit 1
fi

ssh -o BatchMode=yes "${USER_AT}@${HOST}" bash -s <<REMOTE
set -euo pipefail
cd '${SRC}'
tar xf source.tar
# The REVISION must sit where common.py looks for it: package root.
mv REVISION.tmp research/nongaussian-bp/REVISION
# Outputs outside the checkout, so a run cannot mutate its own source and a
# second deployment of the same commit cannot inherit the first one's results.
ln -sfn '${HOME}/${OUT}' research/nongaussian-bp/outputs-run 2>/dev/null || true
echo "extracted \$(find research/nongaussian-bp -name '*.py' | wc -l) python files"
REMOTE

cat > "research/nongaussian-bp/hpc/.last_deploy.env" <<EOF
# Written by deploy_clean.sh. Source this in a job script.
NGBP_SRC=${SRC}
NGBP_OUT=${OUT}
NGBP_COMMIT=${commit}
NGBP_SOURCE_SHA256=${digest}
NGBP_REQUIRE_CLEAN=1
EOF

echo
echo "deployed. In the sbatch:"
echo "    source hpc/.last_deploy.env   # or paste:"
echo "    cd ~/${SRC}/research/nongaussian-bp"
echo "    export NGBP_REQUIRE_CLEAN=1"
echo "    --output-dir ~/${OUT}/<name>"
