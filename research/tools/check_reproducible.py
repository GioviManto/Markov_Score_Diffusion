"""Verify that an experiment reproduces bit-for-bit from a fresh interpreter.

Why this exists as a tool rather than a test: the check has to run the
experiment in a *separate process*, because the bug it guards against was
precisely a per-process effect. `src.utils.rng_for` used to seed from Python's
builtin `hash`, which PEP 456 salts per process for strings, so every run of
every experiment drew different data while looking perfectly deterministic from
inside a single interpreter. A pytest assertion could not have caught it.

Why it compares columns rather than bytes: results CSVs carry wall-clock
timings, which of course differ between runs. A plain `cmp` or `diff` on the
files reports a difference and looks like a reproducibility failure. This tool
excludes timing columns by name and compares everything else exactly, so a real
regression is not hidden behind an expected one.

    python tools/check_reproducible.py exp_09_mixture_message_closure --only exact_family

Exits non-zero if any non-timing column differs.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from src.utils import read_csv  # noqa: E402

# Columns that legitimately vary between runs.
TIMING_HINTS = ("second", "elapsed", "runtime", "wall")


def _is_timing(name: str) -> bool:
    return any(h in name.lower() for h in TIMING_HINTS)


def run_once(experiment: str, out_dir: Path, extra: list[str]) -> None:
    cmd = [
        sys.executable,
        str(PACKAGE_ROOT / "experiments" / f"{experiment}.py"),
        "--quick",
        "--output-dir",
        str(out_dir),
        *extra,
    ]
    proc = subprocess.run(cmd, cwd=PACKAGE_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"{experiment} failed with code {proc.returncode}")


def compare(dir_a: Path, dir_b: Path) -> int:
    csvs = sorted(p.name for p in dir_a.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSV outputs found in {dir_a}")

    failures = 0
    for name in csvs:
        rows_a = read_csv(dir_a / name)
        rows_b = read_csv(dir_b / name)
        if len(rows_a) != len(rows_b):
            print(f"  {name}: DIFFERENT ROW COUNT ({len(rows_a)} vs {len(rows_b)})")
            failures += 1
            continue

        cols = [c for c in rows_a[0] if not _is_timing(c)]
        skipped = [c for c in rows_a[0] if _is_timing(c)]
        bad = set()
        for ra, rb in zip(rows_a, rows_b):
            for c in cols:
                if ra[c] != rb[c]:
                    bad.add(c)
        if bad:
            print(f"  {name}: DIFFERS in {sorted(bad)}")
            failures += 1
        else:
            note = f" (timing columns skipped: {skipped})" if skipped else ""
            print(f"  {name}: identical across {len(cols)} columns{note}")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment", help="module name under experiments/, no .py")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="extra flags passed through, e.g. --only exact_family")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="reprocheck-"))
    try:
        a, b = tmp / "run_a", tmp / "run_b"
        print(f"Running {args.experiment} twice in separate processes ...")
        run_once(args.experiment, a, args.rest)
        run_once(args.experiment, b, args.rest)
        print("Comparing outputs:")
        failures = compare(a, b)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        raise SystemExit(f"{failures} file(s) not reproducible")
    print("Reproducible.")


if __name__ == "__main__":
    main()
