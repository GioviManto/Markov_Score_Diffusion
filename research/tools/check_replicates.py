#!/usr/bin/env python3
"""Every replicate count behind a paper number must come from the frozen config.

Why this exists
---------------
Three real gaps motivated it, all found the same way -- by reading a result that
looked fine and then checking how many replicates were behind it:

    exp_06  n_rep       = 10
    exp_06  n_rep_rate  = 4     -> fitted n^{-1/2} slopes of -0.196 and -0.700
                                   against a predicted -0.500, non-monotone
    exp_06  n_rep_clean = 3     -> exactly the count behind the withdrawn
                                   "flat curve" reading in the previous draft
    exp_07  seed        = one per run, four runs submitted, sixteen promised

`experiments/frozen_config.py` exposes `n_seeds`. None of those knobs read it,
so `provenance()['is_frozen']` returned true for every one of those runs. A
configuration that cannot reach the knob that matters is not freezing anything.

What is checked
---------------
Assignments named `n_rep*` or `n_seeds` inside a settings dict, in any
experiment that feeds the paper. A literal integer is a failure unless the line
carries an explicit `# frozen-exempt: <reason>`.

Exempt by construction, without needing a marker:
  * anything inside a dict literal assigned to a name containing "quick" --
    smoke settings are never a reported number;
  * experiments not in PAPER_EXPERIMENTS -- the compendium is the development
    environment and is deliberately not gated.

    python3 tools/check_replicates.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"

# Only the experiments that produce numbers in the paper or the workshop.
PAPER_EXPERIMENTS = {
    "exp_01_grid_validation.py",
    "exp_02_laplace_gaussian_message_error.py",
    "exp_03_nongaussian_innovation_sweep.py",
    "exp_06_em_parameter_recovery.py",
    "exp_07_em_vs_score_network.py",
    "exp_08_gradient_vs_exact_mstep.py",
    "exp_18_revision_diagnostics.py",
    "exp_27_shape_convergence.py",
    "exp_28_ring_em.py",
}

def WATCHED(k: str) -> bool:
    """Settings the frozen config owns and an experiment must not set itself.

    Two families, found the hard way and one at a time. Replicate counts came
    first: four experiments carried private `n_rep*` knobs and ran at three,
    four and eight replicates while reporting `is_frozen: true`.

    Iteration budgets are the same defect and were missed by the first version
    of this check, because they are not written in a settings dict at all --
    exp_07 passed `n_iters=120` as a keyword argument, in four places, while
    FROZEN.em_max_iters said 400 and nothing in the tree read it. A check that
    only inspects dict literals cannot see a call site, so this one now walks
    both.
    """
    return (
        k == "n_seeds"
        or k.startswith("n_rep")
        or k in {"n_iters", "em_iters", "max_iters", "em_max_iters"}
    )


def offenders(path: Path) -> list[tuple[int, str, int]]:
    src = path.read_text()
    lines = src.splitlines()
    tree = ast.parse(src)
    found: list[tuple[int, str, int]] = []

    # Every node sitting under an `if args.quick:` -- smoke settings, exempt by
    # construction, exactly as a dict named `quick` is below. Without this the
    # check fires on the `settings.update(em_iters=60)` that every experiment
    # uses for its smoke path, which would train everyone to ignore it.
    in_quick: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "quick" in ast.dump(node.test).lower():
            for sub in ast.walk(node):
                in_quick.add(id(sub))

    # Keyword arguments at call sites: `fit_em(..., n_iters=120)`.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or id(node) in in_quick:
            continue
        for kw in node.keywords:
            if kw.arg is None or not WATCHED(kw.arg):
                continue
            if not (isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, int)
                    and not isinstance(kw.value.value, bool)):
                continue  # an expression, e.g. FROZEN.em_max_iters -- fine
            if "frozen-exempt" in lines[kw.value.lineno - 1]:
                continue
            found.append((kw.value.lineno, kw.arg, kw.value.value))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        # Is this dict assigned to a name containing "quick"? Then it is smoke
        # settings and exempt by construction.
        parent_is_quick = False
        for anc in ast.walk(tree):
            if isinstance(anc, ast.Assign) and anc.value is node:
                parent_is_quick = any(
                    isinstance(t, ast.Name) and "quick" in t.id.lower()
                    for t in anc.targets
                )
        if parent_is_quick:
            continue

        for key, val in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            if not WATCHED(key.value):
                continue
            if not (isinstance(val, ast.Constant) and isinstance(val.value, int)):
                continue  # an expression, e.g. FROZEN.n_seeds -- fine
            line = lines[val.lineno - 1]
            if "frozen-exempt" in line:
                continue
            found.append((val.lineno, key.value, val.value))
    return found


def main() -> int:
    bad = {}
    for path in sorted(EXP.glob("exp_*.py")):
        if path.name not in PAPER_EXPERIMENTS:
            continue
        try:
            hits = offenders(path)
        except SyntaxError as exc:
            print(f"  [FAIL] {path.name}: cannot parse ({exc})")
            return 1
        if hits:
            bad[path.name] = hits

    n_exempt = sum(
        p.read_text().count("frozen-exempt")
        for p in EXP.glob("exp_*.py")
        if p.name in PAPER_EXPERIMENTS
    )

    if not bad:
        print(f"  [PASS] every replicate count is frozen or explicitly exempt "
              f"({n_exempt} exemption(s))")
        return 0

    print("  [FAIL] setting(s) fixed locally, not governed by the frozen config:")
    for name, hits in sorted(bad.items()):
        for lineno, key, value in sorted(hits):
            print(f"         {name}:{lineno}  {key} = {value}")
    print("         Replicates: set to FROZEN.n_seeds. Iteration budgets: set to")
    print("         FROZEN.em_max_iters, and justify against the SLOWEST coordinate")
    print("         the run reports -- exp_27 puts shape settling at a median of 229")
    print("         updates against 80 for rho. Otherwise `# frozen-exempt: <reason>`.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
