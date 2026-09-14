#!/usr/bin/env python3
"""Summarise the Lean formula audit from AUDIT_RESULTS/*.json.

Usage:
    ./audit_status.py            # per-chapter table + totals
    ./audit_status.py -v         # also list every discrepancy in full
"""
import json
import pathlib
import sys
from collections import Counter

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "AUDIT_RESULTS"
ORDER = ["Ch05a", "Ch05b", "Ch06", "Ch08a", "Ch08b",
         "Ch07", "Ch10", "Ch04", "Ch03", "Ch02", "Ch01"]
STATUSES = ["proved", "instance_checked", "defined", "skipped", "discrepancy"]

# Display equations per chapter in the LaTeX source, for coverage tracking.
EXPECTED = {"Ch01": 5, "Ch02": 42, "Ch03": 38, "Ch04": 32, "Ch05a": 53,
            "Ch05b": 57, "Ch06": 42, "Ch07": 20, "Ch08a": 34, "Ch08b": 36,
            "Ch10": 34}


def load():
    out = {}
    for p in sorted(RESULTS.glob("*.json")):
        try:
            out[p.stem] = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            print(f"  !! {p.name} is not valid JSON: {e}", file=sys.stderr)
    return out


def main():
    verbose = "-v" in sys.argv
    data = load()
    if not data:
        print("No results yet in AUDIT_RESULTS/.")
        return

    totals = Counter()
    discrepancies = []
    print(f"{'module':<8} {'items':>5} {'exp':>4} {'prov':>5} {'inst':>5} "
          f"{'def':>4} {'skip':>5} {'DISCR':>6}  build")
    print("-" * 62)
    for mod in ORDER:
        d = data.get(mod)
        if not d:
            continue
        items = d.get("items", [])
        c = Counter(i.get("status") for i in items)
        totals.update(c)
        discrepancies += [(mod, i) for i in items
                          if i.get("status") == "discrepancy"]
        exp = EXPECTED.get(mod, "?")
        build = "ok" if d.get("build_ok") else "FAIL"
        print(f"{mod:<8} {len(items):>5} {exp:>4} {c['proved']:>5} "
              f"{c['instance_checked']:>5} {c['defined']:>4} "
              f"{c['skipped']:>5} {c['discrepancy']:>6}  {build}")

    print("-" * 62)
    n = sum(totals[s] for s in STATUSES)
    print(f"{'TOTAL':<8} {n:>5} {sum(EXPECTED.values()):>4} "
          f"{totals['proved']:>5} {totals['instance_checked']:>5} "
          f"{totals['defined']:>4} {totals['skipped']:>5} "
          f"{totals['discrepancy']:>6}")

    missing = [m for m in ORDER if m not in data]
    if missing:
        print(f"\nnot yet audited: {', '.join(missing)}")

    if discrepancies:
        print(f"\n*** {len(discrepancies)} DISCREPANCIES ***")
        for mod, i in discrepancies:
            print(f"\n[{mod}] {i['id']}  (lines {i.get('tex_lines', '?')})")
            print(f"  {i.get('summary', '')}")
            note = i.get("note", "")
            print(f"  {note if verbose else note[:400]}")
    else:
        print("\nNo discrepancies found in the chapters audited so far.")


if __name__ == "__main__":
    main()
