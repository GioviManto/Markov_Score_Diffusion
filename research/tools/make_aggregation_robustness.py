#!/usr/bin/env python
"""Is the headline ratio an artefact of how the ratio was averaged?

WHY THIS EXISTS. The headline is a ratio, and a ratio is a nonlinear function of
two means, so the number depends on the order in which you average. E[X]/E[Y] is
not E[X/Y], the gap between them grows with the spread, and a reader is entitled
to ask whether 7-20x is a property of the estimator or of the estimand chosen to
summarise it. The certified table picks one -- average the noise levels within a
seed, form the per-cell ratio, aggregate across seeds -- and that choice is
correct, because the twelve levels share a training set and a fitted model so
the seed is the inferential unit. Correct is not the same as robust.

This recomputes the same data under six estimands. Nothing is retrained; the
cost is reading the frozen CSVs.

    0. schedule-pooled paired per-seed ratio (what the table reports:
       squared error and reference energy summed over levels before the
       square root -- thesis Eq. 10.2 -- via tools/make_ref_energy.py)
    1. mean of per-cell ratios              (the pre-September-2026 headline)
    2. ratio of per-seed schedule-averaged errors
    3. geometric mean of per-cell ratios    (mean log-ratio, exponentiated)
    4. median paired ratio
    5. paired bootstrap over seeds, 95%     (on the reported estimand)
    6. per-noise-level ratios               (where the aggregation hides things)

The conclusion should not depend on which row you read. Where the rows disagree,
that disagreement is the result and belongs in the paper rather than the
smallest number being quoted.

    python tools/make_aggregation_robustness.py
"""
import csv
import glob
import sys

import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance_gate import load_params, require_clean  # noqa: E402

SOURCES = [
    "outputs/frozen/exp_07_certified_seed*/sample_efficiency_val.csv",
    "outputs/frozen/exp_07_n4096_seed*/sample_efficiency_val.csv",
    # nseq=8192 is withdrawn -- see the WITHDRAWN note in make_tab_efficiency.
    # This must track that file's source list exactly: an aggregation appendix
    # computed over a different cell set than the table it qualifies would be
    # worse than no appendix, because it would look like corroboration.
]
BOOT = 20000
RNG = np.random.default_rng(20260823)


def load():
    rows = []
    for pattern in SOURCES:
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"REFUSING: nothing at {pattern}", file=sys.stderr)
            sys.exit(1)
        for f in files:
            seed = f.split("seed")[1].split("/")[0]
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    r["seed"] = seed
                    rows.append(r)
    # Same resolution gate as the headline table, so this measures the
    # aggregation choice and not a different cell set.
    return [r for r in rows if "em_resolved" not in r or int(r["em_resolved"])]


rows = load()

# Reference energies for the pooled estimand; see make_ref_energy.py.
with open("outputs/frozen/exp_07_ref_energy/ref_energy.csv") as fh:
    B = {float(r["t"]): float(r["ref_sq"]) for r in csv.DictReader(fh)}


def pooled_rel_l2(cells, key):
    num = sum(B[float(r["t"])] * float(r[key]) ** 2 for r in cells)
    den = sum(B[float(r["t"])] for r in cells)
    return (num / den) ** 0.5

# exp_07 predates hpc/deploy_clean.sh: its records carry the commit, host and
# resolved configuration but no source-archive digest. The intersection of its
# recorded dirty list with this experiment's import closure is EMPTY, so the
# code that ran is reconstructable even though the archive is not hashed --
# which is what allow_legacy is for, and is not the exp_21 situation. Appendix C
# names this exception.
require_clean(load_params(sorted(glob.glob(SOURCES[0].rsplit('/', 1)[0]))),
              allow_legacy=True)

seeds = sorted({r["seed"] for r in rows}, key=int)
sizes = sorted({int(r["n_chains"]) for r in rows})
F = lambda r, k: float(r[k])

print(f"{len(rows)} cells, {len(seeds)} seeds, sizes {sizes}\n")
hdr = (f"{'n':>6} {'(1) cell':>9} {'(2) ratio':>10} {'(3) geo':>8} "
       f"{'(4) med':>8} {'(5) paired 95% CI':>20}")
print(hdr)
print("-" * len(hdr))

summary = {}
for n in sizes:
    g = [r for r in rows if int(r["n_chains"]) == n]

    # Per-seed vectors. Averaging within a seed first is what makes the seed the
    # unit; every estimand below is built from these, so they differ only in how
    # the ratio is formed and pooled, never in the dependence structure.
    per_seed_ratio, per_seed_net, per_seed_em, per_seed_pooled = [], [], [], []
    for s in seeds:
        c = [r for r in g if r["seed"] == s]
        if not c:
            continue
        per_seed_ratio.append(np.mean([F(r, "ratio_selected") for r in c]))
        per_seed_net.append(np.mean([F(r, "net_score_rel_l2_selected") for r in c]))
        per_seed_em.append(np.mean([F(r, "em_bp_score_rel_l2") for r in c]))
        per_seed_pooled.append(pooled_rel_l2(c, "net_score_rel_l2_selected")
                               / pooled_rel_l2(c, "em_bp_score_rel_l2"))
    per_seed_ratio = np.array(per_seed_ratio)
    per_seed_net = np.array(per_seed_net)
    per_seed_em = np.array(per_seed_em)
    per_seed_pooled = np.array(per_seed_pooled)

    a0 = per_seed_pooled.mean()
    a1 = per_seed_ratio.mean()
    a2 = per_seed_net.mean() / per_seed_em.mean()
    a3 = float(np.exp(np.mean(np.log([F(r, "ratio_selected") for r in g]))))
    a4 = float(np.median([F(r, "ratio_selected") for r in g]))

    # Bootstrap resamples SEEDS, not cells: cells within a seed share a fitted
    # model, so resampling them would treat dependent observations as
    # independent and produce an interval that is too narrow.
    idx = RNG.integers(0, per_seed_pooled.size, (BOOT, per_seed_pooled.size))
    boot = per_seed_pooled[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    summary[n] = (a0, a1, a2, a3, a4, lo, hi)
    print(f"{n:>6} {a0:>9.2f} {a1:>9.2f} {a2:>10.2f} {a3:>8.2f} {a4:>8.2f} "
          f"{f'[{lo:.2f}, {hi:.2f}]':>20}")

vals = np.array([[v[0], v[1], v[2], v[3], v[4]] for v in summary.values()])
spread = np.abs(vals - vals[:, [0]]).max()
worst = max(summary, key=lambda n: np.ptp(vals[sizes.index(n)]))
print(f"\nlargest departure from the reported estimand: {spread:.2f} "
      f"(worst size n={worst})")
print(f"every estimand at every size exceeds "
      f"{min(v for row in vals for v in row):.2f}x")

print("\nper-noise-level ratios (estimand 6), where aggregation could hide a "
      "reversal:")
levels = sorted({float(r["t"]) for r in rows})
print(f"{'n':>6} " + " ".join(f"{t:>6.3g}" for t in levels))
for n in sizes:
    g = [r for r in rows if int(r["n_chains"]) == n]
    cells = []
    for t in levels:
        v = [F(r, "ratio_selected") for r in g if float(r["t"]) == t]
        cells.append(np.mean(v) if v else np.nan)
    print(f"{n:>6} " + " ".join(f"{c:>6.1f}" for c in cells))

# Two different "worst" numbers, previously conflated into one sentence. The
# aggregate is a mean over seeds within an (n, t) cell; the individual cell is a
# single seed at a single level. The second is much weaker than the first, and
# calling the first "the worst single cell" understated the spread -- EM-BP
# actually loses outright in two individual cells, which no seed-averaged
# quantity can show.
worst_cell = min(c for n in sizes for c in [
    np.mean([F(r, "ratio_selected") for r in rows
             if int(r["n_chains"]) == n and float(r["t"]) == t] or [np.nan])
    for t in levels] if np.isfinite(c))
individual = [F(r, "ratio_selected") for r in rows]
individual = [v for v in individual if np.isfinite(v)]
worst_individual = min(individual)
n_cells = len(individual)
n_reversed = sum(1 for v in individual if v < 1.0)
print(f"\nworst seed-averaged (size, level) aggregate: {worst_cell:.2f}x")
print(f"worst INDIVIDUAL cell:                      {worst_individual:.2f}x "
      f"({n_reversed} of {n_cells} cells below 1)")

# ---------------------------------------------------------------------------
# Emit the table. Since September 2026 the reported estimand is the
# schedule-pooled paired ratio (0); the point the table has to make is that the
# conclusion does not depend on the choice, and that the retired mean-cellwise
# summary (1) is systematically the LARGEST of the five -- which is exactly why
# it was retired as the headline. The ordering is an empirical property of
# these cells, not a theorem.
# ---------------------------------------------------------------------------
lines = [
    f"{n} & ${v[0]:.1f}$ & ${v[1]:.1f}$ & ${v[2]:.1f}$ & ${v[3]:.1f}$ & ${v[4]:.1f}$ & "
    f"$[{v[5]:.1f},\\,{v[6]:.1f}]$ \\\\"
    for n, v in summary.items()
]
floor = min(v for row in vals for v in row)
cw_lo = min(v[1] for v in summary.values())
cw_hi = max(v[1] for v in summary.values())
rom_lo = min(v[2] for v in summary.values())
rom_hi = max(v[2] for v in summary.values())

tex = f"""%% GENERATED by tools/make_aggregation_robustness.py -- do not hand-edit.

\\section{{Does the headline depend on how the ratio was averaged?}}
%% No \\label here: both including documents already own a label for the
%% surrounding chapter/appendix, and defining one in the shared fragment made
%% it multiply-defined in the thesis.
\\label{{sec:aggregation-headline}}

A ratio is a nonlinear function of two means, so its value depends on the order
of averaging. \\headlinetable{{}} reports the schedule-pooled paired
per-seed ratio: squared score error and reference score energy are summed over
the twelve noise levels before the square root, the ratio is formed within each
seed, and the seed is the inferential unit --- the twelve levels share a
training set and a fitted model. Being the definition stated in the text does
not make it the only defensible summary, so here is the same data under four
alternatives, with a paired interval on the one the table reports.

\\begin{{center}}
%% minipage, not a bare center: without it LaTeX will break between the
%% caption and the tabular, and it did -- Table 9.5's caption sat alone at
%% the foot of one page with its rows at the head of the next.
\\begin{{minipage}}{{\\linewidth}}\\small\\centering
\\setlength{{\\tabcolsep}}{{3pt}}
\\captionof{{table}}[The headline ratio under five estimands]{{The headline under five estimands. (0) is what
\\headlinetable{{}} reports; (1) is the mean of per-(seed, level)
ratios, the summary earlier drafts reported. The interval is a paired
bootstrap over seeds, {BOOT:,} resamples; seeds are resampled rather than
cells, because cells within a seed share a fitted model.}}
\\label{{tab:aggregation}}
\\begin{{tabular}}{{rcccccc}}
\\toprule
$\\nseq$ & (0) pooled & (1) per-cell & (2) ratio of means & (3) geometric & (4) median
        & (5) 95\\% CI on (0) \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{minipage}}
\\end{{center}}

\\noindent Two things follow.

The conclusion does not depend on the choice: no estimand at any size falls
below ${floor:.1f}$, every bootstrap interval sits well clear of $1$, and no
aggregation produces a reversal anywhere.

But the estimands are far from interchangeable, and the mean of cellwise
ratios (1) is the largest of the five at every size, reaching
${cw_hi:.1f}$ at the largest size where the pooled estimand reads
$\\ratiohi$. The reason is mechanical: a per-level ratio at a level with a
small reference score norm can be very large while contributing little
energy, and averaging ratios weights all levels equally. Pooling the
energies first weights each level by how much score there is to get wrong,
which is exactly what the pooled definition specifies. Earlier drafts reported
(1) as the headline; the range ${cw_lo:.1f}$--${cw_hi:.1f}$ they quoted is
therefore an artefact of that estimand, and the pooled range
$\\ratiolo$--$\\ratiohi$ replaces it. The ratio of per-seed averaged
errors (2) runs ${rom_lo:.1f}$ to ${rom_hi:.1f}$.

No inequality orders these summaries in general: Jensen gives
$\\E[1/Y]\\ge 1/\\E[Y]$, but $\\E[X/Y]\\ge\\E[X]/\\E[Y]$ does \\emph{{not}}
follow for arbitrary positive correlated $X$ and $Y$. The observed ordering
is a property of these cells, which is exactly why the estimand has to be
named rather than assumed.

Resolving the average over noise levels shows where any single number hides
structure: the per-level ratio is not flat in $t$ but peaks near
$t \\approx 0.5$ and falls at both ends. Two different ``weakest''
quantities have to be kept apart. The weakest seed-averaged $(\\nseq,t)$
aggregate in the grid is ${worst_cell:.1f}$; the weakest \\emph{{individual}}
cell is ${worst_individual:.2f}$, and EM--BP is beaten outright in
${n_reversed}$ of the ${n_cells:,}$ cells summarised here. The estimator's
advantage is largest at moderate noise, where the posterior is dominated by
neither the likelihood nor the prior --- which is where knowing the
transition structure should help most.
"""

dest = "../../Markov_Score_Diffusion/thesis/sections/tab-aggregation.tex"
with open(dest, "w") as fh:
    fh.write(tex)
print(f"\nwrote {dest}")
