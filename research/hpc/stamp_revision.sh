#!/usr/bin/env bash
# Write the REVISION file that provenance() falls back to on the cluster.
#
# Run this on the machine that HAS the repo, immediately before rsyncing the code
# up. Every frozen run before 18 Aug 2026 recorded git_commit="" because the code
# is deployed without .git and `git rev-parse` on a compute node has nothing to
# read -- so the runs that most needed a revision are exactly the ones that had
# none. This closes that.
#
#     hpc/stamp_revision.sh && rsync -a --exclude .git . <host>:<path>/
#
# Two lines: the commit, then the porcelain status (empty when the tree is clean).
# A non-empty second line means the deployed code did not match the commit, which
# is worth knowing later and impossible to reconstruct after the fact.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "FATAL: not in a git repository -- run this where the repo is, not on the cluster" >&2
    exit 1
fi

commit=$(git rev-parse HEAD)
dirty=$(git status --porcelain)

{ echo "$commit"; [ -n "$dirty" ] && echo "$dirty"; } > REVISION

if [ -n "$dirty" ]; then
    echo "stamped $commit  (WARNING: tree is dirty, $(echo "$dirty" | wc -l | tr -d ' ') file(s))"
    echo "$dirty" | sed 's/^/    /'
else
    echo "stamped $commit  (clean)"
fi
