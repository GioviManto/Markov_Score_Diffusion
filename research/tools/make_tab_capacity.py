#!/usr/bin/env python
"""The capacity-equivalence table: does mixture capacity actually saturate at C=8?

WHY THIS EXISTS. Chapter 9 reported held-out evidence at C in {2,4,8,16} paired
against C=1, six seeds, one training size, one initialisation per cell. C=8 and
C=16 differed by 5e-6 and an early draft read that as saturation. Round-two
review was right that this is a failure to resolve rather than a demonstration
of equivalence: six seeds with no predeclared equivalence region cannot establish
a null it was never positioned to test. The chapter was corrected to say exactly
that, and exp_32 was built to let the stronger claim be made honestly IF the data
supported it -- sixteen paired seeds, two sizes, three initialisations selected
on validation, three metrics, and an equivalence region fixed before the contrast
was computed.

The data do not support it, and they do not support the opposite of it either in
the shape anyone expected. At nseq=128 the C=8 -> C=16 step is RESOLVED, and in
the direction of C=16 being worse: held-out evidence falls and schedule-level
score risk rises, with both bootstrap intervals lying entirely outside the
predeclared region. At nseq=512 both intervals straddle zero. So the honest
reading is neither "saturates by eight" nor "keeps paying past eight": past the
capacity the data supports, extra components cost.

THE CAVEAT THAT HAS TO TRAVEL WITH IT. Every cell at C >= 4 stopped at the
em_cap of 1200 rather than at the tolerance -- `em_converged` is False for all
96 of them and the median `em_iters_used` is the cap. The experiment's own
docstring records why the cap sits there: a representative C=16 fit run to 2000
iterations has per-edge gain 6.9e-8 at 400, ~6e-9 by 800, and then plateaus
(6.08e-9 at 1200, 5.92e-9 at 1600, 6.09e-9 at 2000), never crossing the strict
1e-9 threshold at any practical count. The cap targets that plateau, not the
formal tolerance. That makes the residual convergence asymmetry small rather than
absent, and the section reports it rather than claiming it away.

    python tools/make_tab_capacity.py

Writes Markov_Score_Diffusion/thesis/sections/tab-capacity.tex and
Markov_Score_Diffusion/thesis/sections/capacity-numbers.tex. Do not hand-edit either.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance_gate import load_params, require_clean  # noqa: E402

SOURCE_DIR = "outputs/frozen/exp_32_capacity_merged"
SWEEP = f"{SOURCE_DIR}/capacity_equivalence.csv"

METRICS = (
    ("test_log_evidence_per_edge", r"$\Delta$ evidence/edge"),
    ("schedule_score_risk", r"$\Delta$ score risk"),
)

# The equivalence region, predeclared in the experiment's own SETTINGS before the
# contrast was computed. Restated here rather than imported so that changing it
# is a visible edit to a committed file, which is the whole point of
# predeclaring it.
EQUIV_LOGEV_ABS = 1e-4
EQUIV_RISK_REL = 0.01
BOOT_RESAMPLES = 20_000
BOOT_SEED = 20260824  # part_contrast's, so the intervals reproduce it exactly

with open(SWEEP) as fh:
    sweep = list(csv.DictReader(fh))
if not sweep:
    print(f"REFUSING: exp_32 output missing under {SOURCE_DIR}", file=sys.stderr)
    sys.exit(1)

require_clean(load_params([SOURCE_DIR]))

seeds = sorted({int(r["seed"]) for r in sweep})
sizes = sorted({int(r["n_chains"]) for r in sweep})
comps = sorted({int(r["n_components"]) for r in sweep})
expected = len(seeds) * len(sizes) * len(comps)
if len(sweep) != expected:
    print(f"REFUSING: {len(sweep)} rows, expected {expected} "
          f"({len(seeds)} seeds x {len(sizes)} sizes x {len(comps)} capacities) "
          "-- a shard is missing and the paired contrast would silently run "
          "over an unbalanced design.", file=sys.stderr)
    sys.exit(1)

B = lambda r, k: float(r[k])
I = lambda r, k: int(r[k])

# Recomputed here rather than read from a precomputed contrast file. The
# contrast is a pure re-aggregation of the frozen sweep -- no fitting, no
# randomness beyond a fixed bootstrap seed -- so consuming it as a separate
# artefact would mean certifying a file that this generator could rebuild, and
# a locally-run rebuild carries the local tree's provenance rather than the
# sweep's. Same reasoning as tools/make_aggregation_robustness.py.
_cache = {}


def contrast(n, lo, hi, metric):
    key = (n, lo, hi, metric)
    if key in _cache:
        return _cache[key]
    by = lambda c: {I(r, "seed"): B(r, metric) for r in sweep
                    if I(r, "n_chains") == n and I(r, "n_components") == c}
    a, b = by(lo), by(hi)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit(f"no paired seeds for N={n} C{lo}->C{hi}")
    diffs = np.array([b[s] - a[s] for s in shared])
    # Seeded per contrast, not from one shared stream: a single stream makes
    # every interval depend on the order the table happens to request its rows
    # in, so reordering the table would silently move the numbers.
    rng = np.random.default_rng([BOOT_SEED, n, lo, hi, len(metric)])
    boot = rng.choice(diffs, size=(BOOT_RESAMPLES, len(diffs)), replace=True)
    ci_lo, ci_hi = np.percentile(boot.mean(axis=1), [2.5, 97.5])
    region = (EQUIV_LOGEV_ABS if metric == "test_log_evidence_per_edge"
              else EQUIV_RISK_REL * abs(float(np.mean([a[s] for s in shared]))))
    _cache[key] = {
        "n_chains": str(n), "c_lo": str(lo), "c_hi": str(hi), "metric": metric,
        "n_pairs": str(len(shared)), "mean_diff": repr(float(diffs.mean())),
        "ci_lo": repr(float(ci_lo)), "ci_hi": repr(float(ci_hi)),
        "equivalence_region": repr(region),
        "ci_entirely_inside_region": str(abs(ci_lo) < region and abs(ci_hi) < region),
    }
    return _cache[key]


def verdict(n, lo, hi):
    """Three outcomes, and they are genuinely different claims."""
    rows = [contrast(n, lo, hi, m) for m, _ in METRICS]
    if all(r["ci_entirely_inside_region"] == "True" for r in rows):
        return r"equivalent"
    # Resolved means the interval excludes zero: a difference this design can see.
    resolved = [B(r, "ci_lo") * B(r, "ci_hi") > 0 for r in rows]
    if all(resolved):
        return r"resolved"
    if any(resolved):
        return r"partly resolved"
    return r"unresolved"


def sci(v, digits=2, signed=True):
    """LaTeX scientific notation. Python's %e renders as a literal 'e-02'."""
    s = f"{v:+.{digits}e}" if signed else f"{v:.{digits}e}"
    mant, exp = s.split("e")
    return f"{mant} \\times 10^{{{int(exp)}}}"


def against_baseline(n, c, metric, base=1):
    """Paired difference from the C=1 fit, and how many seeds it wins on.

    The mean is reported with the seeds' own standard error rather than a
    bootstrap interval, because this contrast is not the predeclared
    equivalence test -- it is the ordinary question of whether capacity buys
    anything at all, and it is dominated by a handful of large-magnitude seeds
    that make the sign count the more honest summary of the two.
    """
    by = lambda k: {I(r, "seed"): B(r, metric) for r in sweep
                    if I(r, "n_chains") == n and I(r, "n_components") == k}
    a, b = by(base), by(c)
    shared = sorted(set(a) & set(b))
    d = np.array([b[s] - a[s] for s in shared])
    better = d > 0 if metric == "test_log_evidence_per_edge" else d < 0
    return (float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size)),
            int(better.sum()), len(shared), float(np.median(d)))


# Grid resolution. `s_min_over_h` is the narrowest fitted component in grid
# spacings; below about 2 it is a lattice artefact rather than a density, and the
# share of such cells is not constant across C -- it climbs with capacity, which
# is a candidate mechanism for the result below and has to be reported with it.
RESOLUTION_FLOOR = 2.0
n_per_c = {c: sum(1 for r in sweep if I(r, "n_components") == c) for c in comps}
unresolved = {c: sum(1 for r in sweep if I(r, "n_components") == c
                     and B(r, "s_min_over_h") < RESOLUTION_FLOOR)
              for c in comps}

# Every mean and standard error in this table sits between 4e-3 and 3e-2, so a
# per-cell \times 10^{-n} is eight repetitions of the same exponent. Factoring it
# into the header buys back roughly 180pt -- the table ran off the page with it
# in -- and lets the reader compare cells by reading the digits.
TABLE_UNIT = -2


def in_units(v, signed=False):
    """The value in units of 10**TABLE_UNIT. Signed for the paired differences,
    so a positive one -- which would mean the mixture bought something -- is
    unmissable rather than inferred from the absence of a minus."""
    return f"{v / 10 ** TABLE_UNIT:{'+' if signed else ''}.2f}"


lines = []
for c in [k for k in comps if k != 1]:
    cells = []
    for n in sizes:
        mean, se, wins, tot, med = against_baseline(n, c, METRICS[0][0])
        cells.append(f"${in_units(mean, signed=True)}$ & ${in_units(se)}$ "
                     f"& ${wins}/{tot}$")
    cells.append(f"${unresolved[c]}/{n_per_c[c]}$")
    lines.append(f"${c}$ & " + " & ".join(cells) + " \\\\")

# The convergence disclosure. `em_converged` is the tolerance stop, not the cap.
at_cap = {c: sum(1 for r in sweep
                 if I(r, "n_components") == c and r["em_converged"] != "True")
          for c in comps}
cap_iter = max(I(r, "em_iters_used") for r in sweep)
capped_from = min(c for c in comps if at_cap[c] == n_per_c[c])

head = contrast(sizes[0], 8, 16, "test_log_evidence_per_edge")
head_risk = contrast(sizes[0], 8, 16, "schedule_score_risk")

size_head = " & ".join(
    f"\\multicolumn{{3}}{{c}}{{$\\nseq = {n}$}}" for n in sizes)
sub_head = " & ".join("mean & s.e. & seeds better" for _ in sizes)
sub_head += r" & under-resolved"

tex = f"""%% GENERATED by tools/make_tab_capacity.py from {SOURCE_DIR}
%% ({len(seeds)} paired seeds, {len(sizes)} sizes, {len(sweep)} rows, provenance-clean).
%% Do not hand-edit the numbers; rerun the generator.

\\begin{{center}}
%% minipage, not a bare center: without it LaTeX will break between the
%% caption and the tabular, and it did -- Table 9.5's caption sat alone at
%% the foot of one page with its rows at the head of the next.
\\begin{{minipage}}{{\\linewidth}}\\small\\centering
\\captionof{{table}}[Held-out log-evidence per edge, by mixture capacity]{{Held-out log-evidence per edge at each mixture capacity,
paired within seed against the single-Gaussian innovation $C=1$ on identical
training and evaluation data, {len(seeds)} seeds, each fit run to an iteration cap
placed at its measured per-edge-gain plateau (the tolerance stop never fires
there, so the cap is the effective stopping rule). Positive would mean the mixture buys something.
No capacity does, at either size, and the sign count is the more honest summary of
the two columns: the mean is dominated by a few large-magnitude seeds while the
direction is consistent across most of them. The last column counts cells, over
both sizes, whose narrowest fitted component falls below the grid's resolution
floor of {RESOLUTION_FLOOR:.0f} spacings --- a share that climbs with $C$, and a
reason not to read the table as a pure statement about capacity. Means and
standard errors are in units of $10^{{{TABLE_UNIT}}}$.}}
\\label{{tab:capacity}}
\\begin{{tabular}}{{@{{}}r{'ccc' * len(sizes)}c@{{}}}}
\\toprule
$C$ & {size_head} & \\\\
 & {sub_head} \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{minipage}}
\\end{{center}}
"""

w_lo = {n: against_baseline(n, 8, METRICS[0][0]) for n in sizes}
worst_win = max(w_lo[n][2] for n in sizes)  # seeds on which C=8 beats C=1

macros = f"""%% GENERATED by tools/make_tab_capacity.py -- do not hand-edit.
\\newcommand{{\\capseeds}}{{{len(seeds)}}}
\\newcommand{{\\capsizes}}{{{', '.join(str(s) for s in sizes)}}}
\\newcommand{{\\capcomps}}{{{', '.join(str(c) for c in comps)}}}
\\newcommand{{\\capcells}}{{{len(sweep)}}}
%% Against C=1: the direction, stated as the sign count, which is what carries
%% the result. \\capcbestwins is the LARGEST number of seeds any mixture manages
%% at C=8 across the two sizes -- i.e. the most favourable reading available.
\\newcommand{{\\capcbestwins}}{{{worst_win}}}
\\newcommand{{\\capcbestlo}}{{{min(w_lo[n][2] for n in sizes)}}}
%% Math-mode content WITHOUT delimiters: the prose supplies the $...$, so a
%% macro carrying its own would nest and fail to compile.
\\newcommand{{\\capmedlo}}{{{sci(min(w_lo[n][4] for n in sizes))}}}
\\newcommand{{\\capmedhi}}{{{sci(max(w_lo[n][4] for n in sizes))}}}
%% The predeclared C=16 vs C=8 equivalence test, at the size where it resolves.
\\newcommand{{\\capheadsize}}{{{sizes[0]}}}
\\newcommand{{\\capheadlogev}}{{{sci(B(head, 'mean_diff'), 1)}}}
\\newcommand{{\\capheadlogevci}}{{[{sci(B(head, 'ci_lo'), 1)},\\, {sci(B(head, 'ci_hi'), 1)}]}}
\\newcommand{{\\capheadrisk}}{{{sci(B(head_risk, 'mean_diff'), 1)}}}
\\newcommand{{\\capheadriskci}}{{[{sci(B(head_risk, 'ci_lo'), 1)},\\, {sci(B(head_risk, 'ci_hi'), 1)}]}}
\\newcommand{{\\capheadverdict}}{{{verdict(sizes[0], 8, 16)}}}
\\newcommand{{\\capothersize}}{{{sizes[-1]}}}
\\newcommand{{\\capotherverdict}}{{{verdict(sizes[-1], 8, 16)}}}
%% The convergence disclosure that has to travel with the result.
\\newcommand{{\\capiterlimit}}{{{cap_iter - 1}}}
\\newcommand{{\\capcappedfrom}}{{{capped_from}}}
\\newcommand{{\\capcappedcells}}{{{sum(at_cap[c] for c in comps if c >= capped_from)}}}
\\newcommand{{\\capcappedtotal}}{{{sum(n_per_c[c] for c in comps if c >= capped_from)}}}
%% Grid resolution: the narrowest fitted component, in grid spacings. Below ~2 a
%% component is a lattice artefact rather than a density (src/kernels.py). The
%% share climbs with C, which is the second caveat the result travels with.
\\newcommand{{\\capsminfloor}}{{{RESOLUTION_FLOOR:.0f}}}
\\newcommand{{\\capunresolvedone}}{{{unresolved[1]}}}
\\newcommand{{\\capunresolvedhi}}{{{unresolved[comps[-1]]}}}
\\newcommand{{\\capunresolvedtotal}}{{{n_per_c[comps[-1]]}}}
"""

for path, blob in (
    ("../../Markov_Score_Diffusion/thesis/sections/tab-capacity.tex", tex),
    ("../../Markov_Score_Diffusion/thesis/sections/capacity-numbers.tex", macros),
):
    with open(path, "w") as fh:
        fh.write(blob)
    print(f"wrote {path}")

print(f"  source: {SOURCE_DIR} (provenance-clean, {len(sweep)} rows, "
      f"{len(seeds)} paired seeds, sizes {sizes}, C in {comps})")
for n in sizes:
    print(f"  N={n}: C8->C16 {verdict(n, 8, 16)}")
    for metric, _ in METRICS:
        c = contrast(n, 8, 16, metric)
        print(f"    {metric:28} {B(c, 'mean_diff'):+.3e} "
              f"CI=[{B(c, 'ci_lo'):+.3e}, {B(c, 'ci_hi'):+.3e}] "
              f"region=+-{B(c, 'equivalence_region'):.3e}")
print(f"  stopped at the {cap_iter - 1}-iteration cap rather than tolerance: "
      + ", ".join(f"C={c} {at_cap[c]}/{n_per_c[c]}" for c in comps))
