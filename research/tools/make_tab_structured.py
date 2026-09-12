#!/usr/bin/env python
"""The structured-baseline table: EM-BP against the winning structure-aware
network (Section 9.2 of the thesis / Appendix N of the paper).

WHY THIS EXISTS. The headline table (tools/make_tab_efficiency.py) compares
EM-BP against a fully connected MLP and reports 7-18x. The MLP carries no
locality prior and no weight sharing, so the single most obvious objection to
that number is that it measures the architecture's mismatch to sequence data
rather than the value of knowing the Markov structure. This table answers it.

This is the SECOND generation of this table. The first read outputs/exp_12_scaled/,
whose provenance turned out not to verify (a recorded commit that does not
define an override its own command passes -- outputs/README_exp12_scaled.md
has the detail) and whose protocol differed from the headline table's in four
ways (center-site-only scoring, a shorter chain, five noise levels instead of
twelve, a fixed step budget). exp_31_structured_baseline.py is the corrected
replacement named as pending everywhere that table's numbers were disclaimed:
same protocol throughout, all-site AND predeclared-interior scoring alongside
the centre-site diagnostic, both arms checkpointed and selected on a
validation bundle disjoint from test, deployed and gated through
hpc/deploy_clean.sh so provenance is verified rather than assumed. It runs a
screening stage on development seeds to pick the strongest of three
architectures (weight-shared window, dilated convolution, bidirectional
message passing) and a confirmatory stage, at the headline protocol, on the
full sixteen frozen seeds.

Aggregation follows the headline table: average within a seed first (the
noise levels share a training set and a fitted model, so they are not
independent), then across seeds, paired per seed since both arms are fit on
the same data. Region 'all' is the honest all-site metric this table exists
to supply; 'centre' is kept for continuity with the retired table and is
never the headline.

    python tools/make_tab_structured.py

Writes Markov_Score_Diffusion/thesis/sections/tab-structured.tex and
Markov_Score_Diffusion/thesis/sections/structured-numbers.tex. Do not hand-edit either.
"""
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance_gate import load_params, require_clean  # noqa: E402

SOURCE_DIR = "outputs/frozen/exp_31_confirm_merged"
CSV = f"{SOURCE_DIR}/confirm.csv"
HEADLINE_REGION = "all"


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    return rows


rows = load(CSV)
if not rows:
    print(f"REFUSING: no exp_31 output at {CSV}", file=sys.stderr)
    sys.exit(1)

methods = sorted({r["method"] for r in rows})
if methods != ["em_bp", "window"]:
    print(f"REFUSING: expected methods [em_bp, window], got {methods} -- "
          "the winning screened architecture may have changed; check "
          "exp_31_screen/screening.csv and update this generator's "
          "assumptions before trusting its output.", file=sys.stderr)
    sys.exit(1)

try:
    require_clean(load_params([SOURCE_DIR]))
    provenance_verified = True
except SystemExit:
    provenance_verified = False

if not provenance_verified:
    print("REFUSING: exp_31 merged output does not pass the provenance gate",
          file=sys.stderr)
    sys.exit(1)

seeds = sorted({r["seed"] for r in rows}, key=int)
sizes = sorted({int(r["n_chains"]) for r in rows})
F = lambda r, k: float(r[k])


def _risks(region, n, method):
    out = []
    for s in seeds:
        v = [F(r, "risk") for r in rows if r["seed"] == s
             and int(r["n_chains"]) == n and r["method"] == method
             and r["region"] == region]
        assert len(v) == 1, (s, n, region, method, v)
        out.append(v[0])
    return np.array(out)


def per_seed_ratio(region, n):
    """Paired per-seed ratio window/em_bp risk, mean and s.e. across seeds."""
    v = _risks(region, n, "window") / _risks(region, n, "em_bp")
    return v.mean(), v.std(ddof=1) / np.sqrt(v.size)


lines, headline_ratios = [], []
for n in sizes:
    ratio, se = per_seed_ratio(HEADLINE_REGION, n)
    ratio_c, se_c = per_seed_ratio("centre", n)
    headline_ratios.append(ratio)
    # The raw risks alongside the ratio. A ratio is a nonlinear function of two
    # means and hides the scale both arms operate at; a reader cannot tell from
    # 2.34 alone whether both arms are accurate or both are poor.
    em = _risks(HEADLINE_REGION, n, "em_bp").mean()
    wn = _risks(HEADLINE_REGION, n, "window").mean()
    lines.append(f"{n:<4} & ${em:.4f}$ & ${wn:.4f}$ & "
                 f"${ratio:.2f} \\pm {se:.2f}$ & "
                 f"${ratio_c:.2f} \\pm {se_c:.2f}$ \\\\")

n_cells = len(rows)
REGIONS = tuple(sorted({r["region"] for r in rows}))
pairs = []
for s in seeds:
    for n in sizes:
        for region in REGIONS:
            em = [F(r, "risk") for r in rows if r["seed"] == s
                  and int(r["n_chains"]) == n and r["method"] == "em_bp"
                  and r["region"] == region][0]
            wn = [F(r, "risk") for r in rows if r["seed"] == s
                  and int(r["n_chains"]) == n and r["method"] == "window"
                  and r["region"] == region][0]
            pairs.append((em, wn))
em_wins = sum(1 for em, wn in pairs if em < wn)
n_cells_scored = len(pairs)

at_cap_em = sum(1 for r in rows if r["method"] == "em_bp" and r["at_cap"] == "True")
at_cap_wn = sum(1 for r in rows if r["method"] == "window" and r["at_cap"] == "True")
n_arm_cells = sum(1 for r in rows if r["method"] == "em_bp")

lo, hi = min(headline_ratios), max(headline_ratios)

tex = f"""%% GENERATED by tools/make_tab_structured.py from {SOURCE_DIR}
%% ({len(seeds)} seeds, {len(sizes)} sizes, {n_cells} rows, provenance-clean).
%% Do not hand-edit the numbers; rerun the generator.

\\begin{{center}}
%% minipage, not a bare center: without it LaTeX will break between the
%% caption and the tabular, and it did -- Table 9.5's caption sat alone at
%% the foot of one page with its rows at the head of the next.
\\begin{{minipage}}{{\\linewidth}}\\small\\centering
\\captionof{{table}}[EM--BP against the screened structure-aware baseline]{{EM--BP against the screened structure-aware baseline, at the
headline protocol throughout. The baseline is a weight-shared window head whose
architecture and hyperparameters were fixed by the screen of
Table~\\ref{{tab:screening}} on development seeds; only the checkpoint is chosen
here, on a validation bundle disjoint from test, and EM--BP's iteration count is
chosen the same way. Risks are all-site and averaged over the {len(seeds)} seeds; the ratio is
risk(window)/risk(EM--BP) formed within each seed and then averaged, $\\pm$ one
s.e. \\emph{{All}} scores every site, and is the headline; \\emph{{centre}} is a
single-site diagnostic. A third region, the predeclared interior slice, is scored
in the stored output and used for architecture selection but not reported here.}}
\\label{{tab:structured}}
\\begin{{tabular}}{{rcccc}}
\\toprule
 & \\multicolumn{{2}}{{c}}{{risk, all sites}} & \\multicolumn{{2}}{{c}}{{ratio}} \\\\
$\\nseq$ & EM--BP & window & all sites & centre site \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{minipage}}
\\end{{center}}
"""

macros = f"""%% GENERATED by tools/make_tab_structured.py -- do not hand-edit.
%%
%% The prose cites these and never types the numbers, for the reason given in
%% tools/make_tab_efficiency.py: a typed number and a computed number drift, and
%% nothing catches it.
\\newcommand{{\\structratiolo}}{{{lo:.1f}}}
\\newcommand{{\\structratiohi}}{{{hi:.1f}}}
\\newcommand{{\\structseeds}}{{{len(seeds)}}}
\\newcommand{{\\structsizes}}{{{len(sizes)}}}
%% \\structcells counts (seed, size, region) triples across THREE regions, which
%% is why it is 192 and not the 128 a two-column table suggests. It is a
%% descriptive diagnostic: the regions and sizes share fitted models and data, so
%% the inferential unit for each displayed row is \\structseeds paired seeds.
\\newcommand{{\\structcells}}{{{n_cells_scored}}}
\\newcommand{{\\structregions}}{{{len(REGIONS)}}}
\\newcommand{{\\structregionlist}}{{{', '.join(REGIONS)}}}
\\newcommand{{\\structemwins}}{{{em_wins}}}
\\newcommand{{\\structarch}}{{weight-shared window head}}
%% Iteration-cap censoring rate for each arm, over the n_chains x seed cells
%% (not further split by region, since region is a scoring choice made after
%% fitting, not a separate fit).
\\newcommand{{\\structcapem}}{{{100*at_cap_em/n_arm_cells:.0f}}}
\\newcommand{{\\structcapwindow}}{{{100*at_cap_wn/n_arm_cells:.0f}}}
"""

for path, blob, label in (
    ("../../Markov_Score_Diffusion/thesis/sections/tab-structured.tex", tex, "table"),
    ("../../Markov_Score_Diffusion/thesis/sections/structured-numbers.tex", macros, "macros"),
):
    with open(path, "w") as fh:
        fh.write(blob)
    print(f"wrote {path}")

print(f"  source: {SOURCE_DIR} (provenance-clean, {len(seeds)} seeds)")
print(f"  window/EM-BP ratio (all-site) {lo:.2f}-{hi:.2f} across nseq={sizes}")
print(f"  EM-BP beats the window head in {em_wins}/{n_cells_scored} scored (seed,size,region) cells")
print(f"  at-cap censoring: em_bp {at_cap_em}/{n_arm_cells}, window {at_cap_wn}/{n_arm_cells}")
