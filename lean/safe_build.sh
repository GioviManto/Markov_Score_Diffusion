#!/bin/bash
# Serialized lake build: multiple agents may call this concurrently; builds run one at a time.
# Usage: ./safe_build.sh <module>   e.g. ./safe_build.sh ThesisAudit.Ch02
set -u
cd "$(dirname "$0")"
LOCKDIR=".lake/agent-build.lock"
MODULE="${1:-ThesisAudit}"
# spin until we hold the lock (mkdir is atomic); stale locks older than 15 min are removed
for i in $(seq 1 1800); do
  if mkdir "$LOCKDIR" 2>/dev/null; then
    trap 'rmdir "$LOCKDIR" 2>/dev/null' EXIT
    ~/.elan/bin/lake build "$MODULE"
    exit $?
  fi
  if [ -d "$LOCKDIR" ] && [ "$(find "$LOCKDIR" -maxdepth 0 -mmin +30 2>/dev/null)" ]; then
    rmdir "$LOCKDIR" 2>/dev/null
  fi
  sleep 1
done
echo "safe_build.sh: timed out waiting for build lock" >&2
exit 1
