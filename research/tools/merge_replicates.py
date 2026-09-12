"""Merge seed-replicated sample-efficiency runs into a curve with error bars.

Usage:
    python tools/merge_replicates.py outputs/replicates [--out outputs/replicates]

Reads every `seed_*/sample_efficiency.csv` under the given root, checks that the
replicates actually agree on what was swept, and writes:

    merged_raw.csv       every row, tagged with its seed
    merged_summary.csv   per (method, n_chains): mean, sd, and standard error
                         over seeds, for the score and posterior-mean errors
    merged_curve.png     the headline curve with +-1 s.e. bands

The summary is over *seeds*, having first averaged each seed over the noise
schedule -- so a standard error here answers "how much would this curve move if
we reran the experiment", which is the question the single-replicate run in
`outputs/exp_07_em_vs_score_network/` cannot answer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from src.plotting import new_figure, save_figure  # noqa: E402
from src.utils import read_csv, write_csv  # noqa: E402

METRICS = ("score_rel_l2", "mean_rel_l2")


def load(root: Path) -> list[dict]:
    rows: list[dict] = []
    seed_dirs = sorted(root.glob("seed_*"))
    if not seed_dirs:
        raise SystemExit(f"No seed_* directories under {root}")
    for d in seed_dirs:
        csv = d / "sample_efficiency.csv"
        if not csv.exists():
            print(f"  skipping {d.name}: no sample_efficiency.csv (still running?)")
            continue
        seed = d.name.split("_", 1)[1]
        for r in read_csv(csv):
            r["seed"] = seed
            rows.append(r)
    if not rows:
        raise SystemExit("No completed replicates found.")
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Average each seed over t first, then take mean/sd across seeds."""
    per_seed: dict[tuple[str, float, str], list[float]] = {}
    for r in rows:
        key = (r["method"], float(r["n_chains"]), r["seed"])
        per_seed.setdefault(key, []).append(r)

    # {(method, n): {metric: [one value per seed]}}
    grouped: dict[tuple[str, float], dict[str, list[float]]] = {}
    for (method, n, _seed), rs in per_seed.items():
        bucket = grouped.setdefault((method, n), {m: [] for m in METRICS})
        for m in METRICS:
            bucket[m].append(float(np.mean([float(r[m]) for r in rs])))

    out = []
    for (method, n), metrics in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        row = {"method": method, "n_chains": int(n),
               "n_seeds": len(metrics[METRICS[0]])}
        for m, vals in metrics.items():
            arr = np.asarray(vals, dtype=float)
            row[f"{m}_mean"] = float(arr.mean())
            row[f"{m}_sd"] = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
            row[f"{m}_se"] = (
                float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1
                else float("nan")
            )
        out.append(row)
    return out


def check_consistency(rows: list[dict]) -> None:
    """Replicates must agree on the sweep, or the summary averages apples and oranges."""
    by_seed: dict[str, set] = {}
    for r in rows:
        by_seed.setdefault(r["seed"], set()).add((r["method"], r["n_chains"]))
    reference = None
    for seed, cells in sorted(by_seed.items()):
        if reference is None:
            reference, ref_seed = cells, seed
        elif cells != reference:
            missing = reference - cells
            extra = cells - reference
            raise SystemExit(
                f"Replicate seed={seed} does not match seed={ref_seed}: "
                f"missing {sorted(missing)[:4]}, extra {sorted(extra)[:4]}. "
                "Merging these would average over different sweeps."
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="Directory containing seed_* subdirs.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.root

    rows = load(args.root)
    check_consistency(rows)
    summary = summarize(rows)
    n_seeds = len({r["seed"] for r in rows})
    print(f"Merged {n_seeds} replicates, {len(rows)} rows.")

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "merged_raw.csv", rows)
    write_csv(out / "merged_summary.csv", summary)

    fig, ax = new_figure(ncols=2, figsize=(11.0, 4.2))
    methods = sorted({r["method"] for r in summary})
    styles = {"em_bp": "o-", "dsm_net_eps": "s-", "dsm_net_x0": "^-"}
    for j, metric in enumerate(METRICS):
        for method in methods:
            sub = sorted([r for r in summary if r["method"] == method],
                         key=lambda r: r["n_chains"])
            ns = [r["n_chains"] for r in sub]
            mean = np.array([r[f"{metric}_mean"] for r in sub])
            se = np.array([r[f"{metric}_se"] for r in sub])
            ax[j].loglog(ns, mean, styles.get(method, "d-"), label=method)
            ok = np.isfinite(se)
            if ok.any():
                ax[j].fill_between(np.array(ns)[ok], (mean - se)[ok],
                                   (mean + se)[ok], alpha=0.2)
        ax[j].set_xlabel("number of training chains $N$")
        ax[j].set_ylabel(f"relative {'score' if j == 0 else 'posterior-mean'} error")
        ax[j].set_title(f"averaged over $t$, $\\pm$1 s.e. over {n_seeds} seeds")
        ax[j].legend()
    save_figure(fig, out / "merged_curve.png")
    print(f"Wrote merged_summary.csv, merged_raw.csv, merged_curve.png -> {out}")


if __name__ == "__main__":
    main()
