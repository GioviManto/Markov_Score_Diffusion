#!/usr/bin/env python
"""The architecture screen behind Table 9.2's baseline, shown rather than asserted.

WHY THIS EXISTS. Section 9.2 reports a ratio against "the winning structure-aware
architecture", and round-three review asked the obvious question: winning under
what search, and was the tuning effort comparable across the three candidates?
Stating the outcome is not the same as showing the screen. A reader who cannot
see the candidate grid, the parameter counts and the checkpoint ladder has no way
to judge whether the window head won on merit or on being the only arm that was
tuned properly.

SELECTION MUST MATCH THE SELECTOR THAT RAN. `_winners_from_screening` in
experiments/exp_31_structured_baseline.py pools sq_err and sq_ref over the
development seeds and sizes on the BULK region BEFORE the square root -- the
schedule-pooled estimand -- and under that rule the winner is the window head
at RADIUS 2, width 64 (0.1477), with radius 4 at width 64 second (0.1494) and
the confirmatory stage demonstrably ran radius 2: its stored n_params is
4,801 = (2*2+4)*64 + 64^2 + 3*64 + 1. An earlier revision of this generator
averaged the per-(seed, size) risks instead, which promotes radius 4 (0.137)
and mislabelled the confirmatory baseline in three documents while claiming,
in this very docstring, to reproduce the selector exactly. The winner is
emitted as a macro so the prose cannot hold a stale value, and the aggregation
below now IS the selector's.

The parameter counts are the interesting disclosure. The dilated convolutional
stack carries 66k-264k parameters against the window head's 4.8k-21k, an order of
magnitude more, and finishes last. Whatever the screen failed to do, it did not
fail by starving the long-range architectures of capacity.

    python tools/make_tab_screening.py

Writes Markov_Score_Diffusion/thesis/sections/tab-screening.tex and
Markov_Score_Diffusion/thesis/sections/screening-numbers.tex. Do not hand-edit either.
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance_gate import load_params, require_clean  # noqa: E402

SOURCE_DIR = "outputs/frozen/exp_31_screen"
CSV = f"{SOURCE_DIR}/screening.csv"

# The region the selection rule scores on. Not a free choice here: it has to
# match part_screen's, or this table would document a screen that never ran.
SELECTION_REGION = "bulk"

PRETTY = {
    "window": "weight-shared window head",
    "conv": "dilated convolutional stack",
    "bimp": "bidirectional message passing",
}
# Ordered as the chapter introduces them: the local one first, then the two
# built to propagate further.
ORDER = ("window", "conv", "bimp")

with open(CSV) as fh:
    rows = list(csv.DictReader(fh))
if not rows:
    print(f"REFUSING: no screening output at {CSV}", file=sys.stderr)
    sys.exit(1)

require_clean(load_params([SOURCE_DIR]))

params = json.loads(open(f"{SOURCE_DIR}/params_screen.json").read())

archs = sorted({r["arch"] for r in rows})
if set(archs) != set(ORDER):
    print(f"REFUSING: expected architectures {sorted(ORDER)}, got {archs} -- "
          "the screen's candidate set changed; update this generator before "
          "trusting its output.", file=sys.stderr)
    sys.exit(1)

dev_seeds = sorted({int(r["seed"]) for r in rows})
sizes = sorted({int(r["n_chains"]) for r in rows})
regions = sorted({r["region"] for r in rows})

# Pooled risk per (arch, hp) over every development seed and size, on the
# selection region: sum sq_err and sq_ref, then sqrt of the quotient. This
# reproduces _winners_from_screening exactly (same accumulation, same region);
# if it ever stops doing so, the table below documents a screen that did not
# happen -- which is precisely the failure the docstring records.
scored = defaultdict(lambda: [0.0, 0.0])
for r in rows:
    if r["region"] == SELECTION_REGION:
        acc = scored[(r["arch"], r["hp"])]
        acc[0] += float(r["sq_err"])
        acc[1] += float(r["sq_ref"])
ranked = sorted((float(np.sqrt(e / d)), k) for k, (e, d) in scored.items())

best_per_arch, n_configs, params_range = {}, {}, {}
for risk, (arch, hp) in ranked:
    best_per_arch.setdefault(arch, (risk, hp))
for arch in ORDER:
    n_configs[arch] = len({r["hp"] for r in rows if r["arch"] == arch})
    p = sorted({int(r["n_params"]) for r in rows if r["arch"] == arch})
    params_range[arch] = (p[0], p[-1])

winner_arch, winner_hp = ranked[0][1]
winner_risk = ranked[0][0]
winner = json.loads(winner_hp)
runner_up = next(r for r in ranked if r[1][0] != winner_arch)

checkpoints = sorted({int(r["checkpoint"]) for r in rows})
lrs = sorted({json.loads(r["hp"])["lr"] for r in rows})
modes = sorted({json.loads(r["hp"])["parameterization"] for r in rows})


def _fmt_params(lo, hi):
    f = lambda v: f"{v/1000:.1f}k" if v < 100_000 else f"{v/1000:.0f}k"
    return f"{f(lo)}--{f(hi)}"


lines = []
for arch in ORDER:
    risk, hp = best_per_arch[arch]
    lo, hi = params_range[arch]
    mark = r"\;$\star$" if arch == winner_arch else ""
    lines.append(
        f"{PRETTY[arch]}{mark} & {n_configs[arch]} & {_fmt_params(lo, hi)} & "
        f"${risk:.3f}$ \\\\"
    )

tex = f"""%% GENERATED by tools/make_tab_screening.py from {SOURCE_DIR}
%% ({len(dev_seeds)} development seeds, {len(rows)} scored rows, provenance-clean).
%% Do not hand-edit the numbers; rerun the generator.

\\begin{{center}}
%% minipage, not a bare center: without it LaTeX will break between the
%% caption and the tabular, and it did -- Table 9.5's caption sat alone at
%% the foot of one page with its rows at the head of the next.
\\begin{{minipage}}{{\\linewidth}}\\small\\centering
\\captionof{{table}}[The architecture screen, in full]{{The architecture screen that selected Table~\\ref{{tab:structured}}'s
baseline. Each candidate architecture was run over its own hyperparameter grid on
{len(dev_seeds)} development seeds (disjoint from the confirmatory seeds),
at $\\nseq \\in \\{{{', '.join(str(s) for s in sizes)}\\}}$, with a shared learning-rate grid,
both output parameterisations, and a shared checkpoint ladder; the reported risk is
the schedule-pooled relative score error, pooled over the development seeds and
sizes on the interior (bulk) slice used for selection --- the same estimand and
accumulation as the selector inside the confirmatory job. $\\star$
marks the winner carried into the confirmatory run. The dilated stack carries an
order of magnitude more parameters than the winner and finishes last, so the
outcome is not an artefact of one arm being given more capacity than another.}}
\\label{{tab:screening}}
\\begin{{tabular}}{{lccc}}
\\toprule
architecture & configurations & parameters & selection risk \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{minipage}}
\\end{{center}}
"""

macros = f"""%% GENERATED by tools/make_tab_screening.py -- do not hand-edit.
%%
%% The prose cites these and never types the numbers. The radius below is the
%% reason why: it was typed once as 2, from the wrong aggregation, and stood in
%% three documents until this generator was written.
\\newcommand{{\\screenwinnerradius}}{{{winner['radius']}}}
\\newcommand{{\\screenwinnerwidth}}{{{winner['width']}}}
\\newcommand{{\\screenwinnerlr}}{{{winner['lr']:g}}}
\\newcommand{{\\screenwinnermode}}{{{winner['parameterization']}}}
\\newcommand{{\\screenwinnerrisk}}{{{winner_risk:.3f}}}
\\newcommand{{\\screenrunnerup}}{{{PRETTY[runner_up[1][0]]}}}
\\newcommand{{\\screenrunneruprisk}}{{{runner_up[0]:.3f}}}
\\newcommand{{\\screendevseeds}}{{{len(dev_seeds)}}}
\\newcommand{{\\screenconfigs}}{{{sum(n_configs.values())}}}
\\newcommand{{\\screenrows}}{{{len(rows)}}}
\\newcommand{{\\screenregion}}{{{SELECTION_REGION}}}
\\newcommand{{\\screenckptlo}}{{{checkpoints[0]}}}
\\newcommand{{\\screenckpthi}}{{{checkpoints[-1]}}}
\\newcommand{{\\screenlrs}}{{{', '.join(f'{v:g}' for v in lrs)}}}
%% Parameter counts, for the "was the tuning effort equal" question: the dilated
%% stack carries the most parameters of the three and finishes last.
\\newcommand{{\\screenwindowparams}}{{{_fmt_params(*params_range['window'])}}}
\\newcommand{{\\screenconvparams}}{{{_fmt_params(*params_range['conv'])}}}
\\newcommand{{\\screenbimpparams}}{{{_fmt_params(*params_range['bimp'])}}}
"""

for path, blob in (
    ("../../Markov_Score_Diffusion/thesis/sections/tab-screening.tex", tex),
    ("../../Markov_Score_Diffusion/thesis/sections/screening-numbers.tex", macros),
):
    with open(path, "w") as fh:
        fh.write(blob)
    print(f"wrote {path}")

print(f"  source: {SOURCE_DIR} (provenance-clean, {len(rows)} rows, "
      f"{len(dev_seeds)} dev seeds {dev_seeds})")
print(f"  selection region: {SELECTION_REGION}; regions present: {regions}")
print(f"  WINNER: {winner_arch} {winner_hp}  risk {winner_risk:.5f}")
for arch in ORDER:
    risk, hp = best_per_arch[arch]
    print(f"    {arch:7} best {risk:.5f}  {hp}  "
          f"({n_configs[arch]} configs, {_fmt_params(*params_range[arch])} params)")
print(f"  lr grid {lrs}, parameterisations {modes}, "
      f"checkpoints {checkpoints[0]}..{checkpoints[-1]}")
