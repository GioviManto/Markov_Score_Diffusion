# tools — gates and generators

Scripts that check the work or produce what the documents include. Run from the
package root (`research/nongaussian-bp`).

## Gates

| Script | What it enforces |
|---|---|
| `check_paper.sh` | page limits, unfilled `\needsdata`, code references, hand-typed ratio ranges, orphaned shared sections, stray configuration |
| `check_all.sh` | the above plus the full test suite |
| `check_replicates.py` | every experiment's replicate count against the frozen config, block-aware so a `--quick` dict is not mistaken for a production one |
| `check_reproducible.py` | re-runs deterministic outputs and diffs them |
| `check_notebooks_fresh.py` | notebooks match the code they document |

`check_paper.sh` verifies its own inputs exist before running anything. Every
check in it greps a file, so a wrong path meant grep found nothing and the check
reported PASS — which is exactly what happened when the documents moved to
`overleaf/` and this script still pointed at the old tree.

## Generators — their output is not to be hand-edited

| Script | Writes |
|---|---|
| `make_tab_efficiency.py` | `overleaf/shared/sections/tab-efficiency.tex` and `efficiency-numbers.tex` |
| `make_figures.py` | every figure into `overleaf/shared/figures/` |

`efficiency-numbers.tex` defines the macros the prose cites (`\ratiolo`,
`\nfreeparams`, …). The abstract once held "between 8 and 14" for three weeks
after the table said 7.3–15.7; macros make that impossible rather than merely
detectable. `make_tab_efficiency.py` also refuses to build the table if any cell
failed the mixture-resolution check, and reports "NOT CERTIFIED" rather than
passing quietly for outputs that predate that column.

## Cluster

| Script | What it does |
|---|---|
| `merge_replicates.py` | merges sharded output into one CSV |
| `pull_and_check.sh` | rsyncs results down and validates them |

See `../hpc/` for the sbatch files and `stamp_revision.sh`, which writes the
`REVISION` file that provenance falls back to on compute nodes, where `.git` is
not present and `git rev-parse` would otherwise record nothing.
