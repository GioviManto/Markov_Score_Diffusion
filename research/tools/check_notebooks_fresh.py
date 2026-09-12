#!/usr/bin/env python3
"""Do the notebooks' committed outputs still match the data they claim to derive from?

WHY THIS EXISTS. `check_all.sh` re-executes every notebook and passes if none of
them raises. That is not the same question. On 2026-08-13 the videoall cluster
job overwrote `outputs/exp_26_video/fit.csv` in place, and notebook 08 kept
displaying the numbers from before the overwrite -- caterpillar -5902.5 against
a committed -5939.9. It executed cleanly the whole time. A notebook can be
simultaneously green and stale, and a stale notebook is worse than a broken one
because it still looks authoritative.

The notebooks in this project state as their convention that they re-derive every
number from the committed CSVs rather than transcribing prose. This checks that
the convention actually holds.

METHOD. Execute each notebook to a temporary copy, then compare the numbers in
its stream output against the numbers in the committed copy, positionally. Text
is ignored: prose gets reworded, and that is not staleness. Only numeric drift
beyond tolerance is reported.

Timings are the obvious false positive -- a cell that prints "12.4s" prints
something else every run -- so a cell whose output matches the timing patterns
below is skipped rather than flagged.

    python tools/check_notebooks_fresh.py                 # all notebooks
    python tools/check_notebooks_fresh.py 08 12           # only these

Exits non-zero if any notebook's committed numbers no longer reproduce.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

NUM = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")
# Lines that legitimately differ run to run.
VOLATILE = re.compile(r"\b(seconds?|s\b|elapsed|wall|took|time)\b", re.I)
RTOL = 1e-6
ATOL = 1e-9


def stream_numbers(nb: dict) -> list[tuple[int, float]]:
    """(cell_index, value) for every number printed by a code cell, in order."""
    out: list[tuple[int, float]] = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for o in cell.get("outputs", []):
            if o.get("output_type") != "stream":
                continue
            for line in "".join(o.get("text", "")).splitlines():
                if VOLATILE.search(line):
                    continue
                for m in NUM.finditer(line):
                    try:
                        out.append((i, float(m.group())))
                    except ValueError:
                        pass
    return out


def close(a: float, b: float) -> bool:
    return abs(a - b) <= max(ATOL, RTOL * max(abs(a), abs(b)))


def check(path: str, workdir: str) -> tuple[bool, str]:
    committed = json.load(open(path))
    name = os.path.basename(path)
    with tempfile.TemporaryDirectory() as tmp:
        rc = subprocess.run(
            [".venv/bin/jupyter", "nbconvert", "--to", "notebook", "--execute",
             "--output-dir", tmp, "--output", name,
             "--ExecutePreprocessor.timeout=1200", path],
            cwd=workdir, capture_output=True, text=True,
        )
        if rc.returncode != 0:
            err = re.search(r"[A-Za-z]*Error[^\n]*", rc.stderr or "")
            return False, f"failed to execute: {err.group() if err else 'unknown'}"
        fresh = json.load(open(os.path.join(tmp, name)))

    a, b = stream_numbers(committed), stream_numbers(fresh)
    if len(a) != len(b):
        return False, (f"output shape changed: committed prints {len(a)} numbers, "
                       f"a fresh run prints {len(b)}")
    bad = [(i, x, y) for (i, x), (_, y) in zip(a, b) if not close(x, y)]
    if bad:
        i, x, y = bad[0]
        return False, (f"{len(bad)} number(s) differ; first in cell {i}: "
                       f"committed {x!r} vs fresh {y!r}")
    return True, f"{len(a)} numbers reproduce"


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    nbdir = os.path.join(root, "notebooks")
    wanted = sys.argv[1:]

    paths = sorted(
        os.path.join(nbdir, f) for f in os.listdir(nbdir)
        if f.endswith(".ipynb") and (not wanted or any(w in f for w in wanted))
    )
    if not paths:
        print("no notebooks matched", file=sys.stderr)
        return 1

    fails = 0
    for p in paths:
        ok, msg = check(p, root)
        print(f"  {'PASS' if ok else 'FAIL'}  {os.path.basename(p):<52} {msg}")
        fails += not ok
    print(f"\n{len(paths) - fails} fresh, {fails} stale")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
