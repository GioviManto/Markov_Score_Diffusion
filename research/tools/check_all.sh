#!/usr/bin/env bash
# Every local check this project has, in one command.
#
# WHY THIS EXISTS. The checks were all here already -- a pytest suite, a
# provenance auditor, a cross-process reproducibility checker, executable
# notebooks -- and nothing ran them together, so each was run when someone
# happened to think of it. The defects that actually reached the repository in
# August 2026 were the ones no single check owned:
#
#   * documentation that outlived the runs it described (CLAIMS_TO_UPDATE.md
#     telling readers to skip the one section it had finished; a compendium
#     listing a completed sweep under "Not started")
#   * a cluster tree assembled from two branches, whose test suite failed nine
#     ways while both branches passed locally
#   * a sweep split across two output roots by a date stamp evaluated per task
#
# None of those are unit-testable. What they have in common is that a human had
# to notice. So the checks below include the boring structural ones -- do the
# advertised files exist, do the counts in the README match reality -- alongside
# the real test suite.
#
#   ./tools/check_all.sh            # everything (~40 min, notebooks run twice)
#   ./tools/check_all.sh --quick    # skip notebooks and reproducibility (~1 min)
#
# Exits non-zero if any check fails. Run from research/nongaussian-bp.

set -uo pipefail
cd "$(dirname "$0")/.."

QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

PY=.venv/bin/python
[[ -x "$PY" ]] || PY=python3
REPO_TOOLS=../../tools          # audit_em_bp_provenance.py, summarize_generation_rerun.py
                                # NOTE: this project has two tools/ directories.
                                # This one (research/nongaussian-bp/tools) holds the
                                # per-experiment helpers; the repo-root one holds the
                                # cross-cutting auditors. Easy to grep the wrong one.

PASS=0; FAIL=0; SKIP=0
declare -a FAILED

ok()   { printf "  \033[32mPASS\033[0m  %s\n" "$1"; PASS=$((PASS+1)); }
bad()  { printf "  \033[31mFAIL\033[0m  %s\n" "$1"; FAIL=$((FAIL+1)); FAILED+=("$1"); }
skip() { printf "  \033[33mSKIP\033[0m  %s (%s)\n" "$1" "$2"; SKIP=$((SKIP+1)); }
head_() { printf "\n\033[1m%s\033[0m\n" "$1"; }

# ---------------------------------------------------------------------------
head_ "1. Test suite"
# ---------------------------------------------------------------------------
if [[ $QUICK -eq 1 ]]; then
    skip "pytest" "--quick"
else
    if out=$($PY -m pytest tests/ -q 2>&1); then
        ok "pytest -- $(grep -oE '[0-9]+ passed[^)]*' <<<"$out" | tail -1)"
    else
        bad "pytest -- $(grep -oE '[0-9]+ failed[^)]*' <<<"$out" | tail -1)"
        grep -E "^FAILED" <<<"$out" | head -5 | sed 's/^/        /'
    fi
fi

# ---------------------------------------------------------------------------
head_ "2. Provenance -- do committed manifests match committed data?"
# ---------------------------------------------------------------------------
if [[ -f "$REPO_TOOLS/audit_em_bp_provenance.py" ]]; then
    if out=$($PY "$REPO_TOOLS/audit_em_bp_provenance.py" 2>&1); then
        ok "audit_em_bp_provenance"
    else
        bad "audit_em_bp_provenance"; sed 's/^/        /' <<<"$out" | head -8
    fi
else
    skip "audit_em_bp_provenance" "not found at $REPO_TOOLS"
fi

# ---------------------------------------------------------------------------
head_ "3. Structural -- do advertised paths actually exist?"
# ---------------------------------------------------------------------------
# The README sent readers to docs/PAPER_CONNECTIONS.md for weeks while that file
# existed only inside a gitignored agent worktree. Cheap to check, and the class
# of bug is "a document confidently references something that is not there".
missing=0
while read -r target; do
    [[ -z "$target" ]] && continue
    [[ -e "$target" ]] || { echo "        missing: $target"; missing=1; }
done <<'EOF'
docs/PAPER_CONNECTIONS.md
notebooks/README.md
hpc/bocconi_final_em.sbatch
hpc/bocconi_wavelet.sbatch
hpc/sync_to_cluster.sh
src/kernels.py
src/em.py
EOF
[[ $missing -eq 0 ]] && ok "advertised paths present" || bad "advertised paths present"

# ---------------------------------------------------------------------------
head_ "4. Counts in prose vs counts on disk"
# ---------------------------------------------------------------------------
# README has carried "101 tests" (against 321), "exp_01 ... exp_15" (against 27)
# and "notebooks 01-04" (against 6). Prose drifts; disk does not.
n_exp=$(ls experiments/exp_*.py 2>/dev/null | wc -l | tr -d ' ')
n_nb=$(ls notebooks/*.ipynb 2>/dev/null | wc -l | tr -d ' ')
n_tests=$($PY -m pytest tests/ -q --collect-only 2>/dev/null | grep -cE "^tests/" || echo 0)
echo "        experiments=$n_exp  notebooks=$n_nb  tests collected=$n_tests"
if grep -q "exp_01 \.\.\. exp_27" README.md 2>/dev/null; then
    ok "README experiment range"
else
    bad "README experiment range (disk has $n_exp experiments)"
fi

# A notebook written but never added to the index is invisible; an index entry
# whose file was renamed is a dead link a reader hits before they hit the work.
# Both are silent, and both have happened here.
unlisted=0; broken=0
for nb in notebooks/*.ipynb; do
    grep -q "$(basename "$nb")" notebooks/README.md 2>/dev/null \
        || { echo "        not in index: $(basename "$nb")"; unlisted=1; }
done
for link in $(grep -oE '\]\([0-9]{2}_[a-z_]+\.ipynb\)' notebooks/README.md 2>/dev/null \
              | tr -d '])(' | sort -u); do
    [[ -f "notebooks/$link" ]] || { echo "        dead index link: $link"; broken=1; }
done
[[ $unlisted -eq 0 && $broken -eq 0 ]] && ok "notebook index complete, links resolve" \
                                       || bad "notebook index out of sync"

# ---------------------------------------------------------------------------
head_ "5. Sweep completeness -- is a run root silently half-present?"
# ---------------------------------------------------------------------------
# The reps array straddled midnight and wrote 12 seeds to one dated root and 4
# to another. Counting one root gives a complete-looking twelve. Always glob.
if [[ -d outputs/final_em ]]; then
    reps=$(ls outputs/final_em/*/.ok_reps_* 2>/dev/null | wc -l | tr -d ' ')
    rec=$(ls outputs/final_em/*/.ok_recovery_* 2>/dev/null | wc -l | tr -d ' ')
    # Rerun cells (<tag>_u<N>) are extra traces of cells already counted; folding
    # them into the total reported "19/18", which is two things added together.
    shp=$(find outputs/final_em/*/shape -name .ok_cell 2>/dev/null \
          | grep -v '_u[0-9]*/' | wc -l | tr -d ' ')
    # Only 8-digit dated directories are run roots. final_em/ also holds
    # gpu_bench/ from the GPU parity benchmark, which is not a sweep.
    roots=$(ls -d outputs/final_em/*/ 2>/dev/null \
            | grep -E '/[0-9]{8}/$' | wc -l | tr -d ' ')
    echo "        run roots=$roots  reps=$reps/16  recovery=$rec/4  shape cells=$shp/18"
    [[ "$reps" -eq 16 ]] && ok "reps complete (16 seeds across $roots roots)" \
                         || bad "reps incomplete: $reps/16"
    [[ "$rec" -eq 4 ]]  && ok "recovery complete" || bad "recovery incomplete: $rec/4"
    [[ "$shp" -eq 18 ]] && ok "shape complete" \
                        || skip "shape" "$shp/18 -- sweep still running, pull again when done"
else
    skip "sweep completeness" "outputs/final_em not present"
fi

# ---------------------------------------------------------------------------
head_ "6. Cluster tree parity (read-only; needs ssh)"
# ---------------------------------------------------------------------------
# A tree assembled from two branches passed every local test and failed nine on
# the cluster. Parity is the only thing that catches that.
if [[ $QUICK -eq 1 ]]; then
    skip "cluster parity" "--quick"
elif ./hpc/sync_to_cluster.sh --check >/dev/null 2>&1; then
    ok "cluster has both branches' files"
else
    skip "cluster parity" "unreachable or files missing -- run ./hpc/sync_to_cluster.sh --check"
fi

# ---------------------------------------------------------------------------
head_ "7. Notebooks re-execute"
# ---------------------------------------------------------------------------
# Notebooks are committed with outputs, so a stale one looks authoritative
# forever. Executing to a temp dir leaves the committed outputs untouched.
if [[ $QUICK -eq 1 ]]; then
    skip "notebook execution" "--quick"
elif [[ ! -x .venv/bin/jupyter ]]; then
    skip "notebook execution" "jupyter not in .venv"
else
    tmp=$(mktemp -d)
    for nb in notebooks/*.ipynb; do
        b=$(basename "$nb" .ipynb)
        if .venv/bin/jupyter nbconvert --to notebook --execute \
              --output-dir="$tmp" --output="$b.ipynb" \
              --ExecutePreprocessor.timeout=1200 "$nb" >"$tmp/$b.log" 2>&1; then
            ok "notebook $b"
        else
            bad "notebook $b -- $(grep -oE '[A-Za-z]*Error[^\"]*' "$tmp/$b.log" | head -1 | cut -c1-70)"
        fi
    done
    rm -rf "$tmp"
fi

# ---------------------------------------------------------------------------
head_ "8. Notebooks are FRESH, not merely green"
# ---------------------------------------------------------------------------
# Step 7 asks whether each notebook runs. This asks whether what it displays is
# still what the data says. A cluster job overwrote outputs/exp_26_video/fit.csv
# in place on 2026-08-13 and notebook 08 went on showing pre-overwrite numbers
# while executing cleanly -- green and stale at the same time, which is the worse
# of the two because a stale notebook still looks authoritative.
if [[ $QUICK -eq 1 ]]; then
    skip "notebook freshness" "--quick"
elif [[ ! -f tools/check_notebooks_fresh.py ]]; then
    skip "notebook freshness" "tools/check_notebooks_fresh.py not found"
else
    if out=$($PY tools/check_notebooks_fresh.py 2>&1); then
        ok "notebook outputs reproduce -- $(tail -1 <<<"$out")"
    else
        bad "notebook outputs are stale"
        grep -E "^  FAIL" <<<"$out" | head -5 | sed 's/^/      /'
    fi
fi

# ---------------------------------------------------------------------------
head_ "9. Cross-process reproducibility"
# ---------------------------------------------------------------------------
# rng_for once seeded from builtin hash(), which PEP 456 salts per process: every
# run drew different data while looking deterministic inside one interpreter.
# Only a separate-process check sees it.
if [[ $QUICK -eq 1 ]]; then
    skip "reproducibility" "--quick"
elif [[ -f tools/check_reproducible.py ]]; then
    if $PY tools/check_reproducible.py exp_09_mixture_message_closure --only exact_family \
         >/dev/null 2>&1; then
        ok "exp_09 reproduces across processes"
    else
        bad "exp_09 reproducibility"
    fi
else
    skip "reproducibility" "tools/check_reproducible.py not found"
fi

# ---------------------------------------------------------------------------
printf "\n\033[1m%s\033[0m\n" "Summary"
printf "  %d passed, %d failed, %d skipped\n" "$PASS" "$FAIL" "$SKIP"
if [[ $FAIL -gt 0 ]]; then
    printf "\n  failed:\n"
    for f in "${FAILED[@]}"; do printf "    - %s\n" "$f"; done
    exit 1
fi
exit 0
