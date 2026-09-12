"""Emit the convergence-rate macros into Markov_Score_Diffusion/thesis/sections/.

WHY THIS EXISTS. Table~\\ref{tab:pointwise} reports a ratio rising from 7.0 to
17.6 and the prose says "rising with nseq" without quantifying it. A ratio at one
size is a point comparison, and the obvious objection to a point comparison is
that it reflects tuning. A *rate* difference is a different kind of claim: it says
the gap widens with sample size, so no amount of tuning at one size explains it.
The data already on disk supports the stronger statement, and this measures it.

WHAT IS AND IS NOT CLAIMED. This fits log--log slopes over the measured range,
nseq = 32 to 4096. It is not an asymptotic statement, and the macros are named so
that the prose cannot accidentally make one: `\\rateem` is "the fitted slope over
the measured range", not "the rate". Seven doublings with both arms partly
budget-censored is not the regime in which to assert an asymptotic exponent.

THE CENSORING, WHICH IS THE ONLY SERIOUS OBJECTION. Both arms increasingly select
their largest allowed budget as nseq grows -- the network from 6% of cells at
nseq=32 to 42% at 4096, EM--BP from 12% to 61%. By the project's own rule (a
budget is a regularisation knob, so two arms with different cap-hit rates are
being compared on budgets rather than methods) that is disqualifying unless its
direction is known.

It is known, and it is conservative. The paired budget calibration at nseq=2048
-- same seeds, same bundles, both caps tripled -- lowers the network's error by
0.35% and EM--BP's by 4.5%, so EM--BP is the arm being held back. Because its
cap-hit rate *rises* with nseq, the censoring inflates its error more at large
nseq than at small, which makes its measured curve SHALLOWER than the truth.
Removing the censoring would therefore steepen EM--BP's slope and widen the gap.
The measurement is a lower bound on the effect it reports.

Aggregation is per seed, as in make_tab_efficiency.py, and for the same reason:
cells within a seed share a training set and move together.
"""
import csv
import glob
import sys
from collections import defaultdict

import numpy as np

ROOT = "outputs/frozen"
SOURCE = f"{ROOT}/exp_07_certified_seed*/sample_efficiency_val.csv"
EXTENDED = f"{ROOT}/exp_07_n4096_seed*/sample_efficiency_val.csv"
CALIB = f"{ROOT}/exp_07_budget2048_seed*/sample_efficiency_val.csv"
CALIB_N = 2048

NET = "net_score_rel_l2_selected"
EM = "em_bp_score_rel_l2"
# These macros are quoted by the PAPER, not the thesis, so they are written
# where they are used. Writing them into thesis/sections/ left a generated
# file the thesis never inputs sitting in its tree and its build digest.
OUT = "../../Markov_Score_Diffusion/paper/sections/rate-numbers.tex"

# If the paired budget calibration ever shows the NETWORK gaining more than
# EM--BP, the conservative-direction argument above is void and the rate claim
# must not ship. Refusing is the point: this is the one assumption the result
# rests on that is not visible in the fitted slopes themselves.
MIN_EM_ADVANTAGE = 1.5


def load(pattern, what):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"REFUSING: no {what} run at {pattern}", file=sys.stderr)
        sys.exit(1)
    out = []
    for f in files:
        seed = f.split("seed")[1].split("/")[0]
        with open(f) as fh:
            for r in csv.DictReader(fh):
                # Same resolution gate as the table: an under-resolved mixture
                # component is a lattice artefact, not evidence.
                if int(r.get("em_resolved", 1)):
                    r["seed"] = seed
                    out.append(r)
    return out


# Reference score energies of the shared frozen test bundle, one per level
# (tools/make_ref_energy.py): the weights that turn the stored per-level
# relative errors back into the schedule-pooled estimand of the headline.
with open(f"{ROOT}/exp_07_ref_energy/ref_energy.csv") as _fh:
    _B = {float(r["t"]): float(r["ref_sq"]) for r in csv.DictReader(_fh)}


def per_seed_by_size(rows, key):
    """{seed: {n: schedule-POOLED relative error}} -- sum_t B_t E_t^2 / sum_t
    B_t under the square root, the same estimand as the headline table."""
    num = defaultdict(lambda: defaultdict(float))
    den = defaultdict(lambda: defaultdict(float))
    for r in rows:
        b = _B[float(r["t"])]
        num[r["seed"]][int(r["n_chains"])] += b * float(r[key]) ** 2
        den[r["seed"]][int(r["n_chains"])] += b
    return {s: {n: float(np.sqrt(num[s][n] / den[s][n])) for n in num[s]}
            for s in num}


def fitted_slopes(rows, key, sizes, seeds):
    ln_n = np.log(np.array(sizes, dtype=float))
    table = per_seed_by_size(rows, key)
    return np.array([
        np.polyfit(ln_n, np.log([table[s][n] for n in sizes]), 1)[0]
        for s in seeds
    ])


def se(a):
    return float(np.std(a, ddof=1) / np.sqrt(len(a)))


def main():
    rows = load(SOURCE, "certified") + load(EXTENDED, "extended-nseq")
    calib = load(CALIB, "budget-calibration")

    seeds = sorted({r["seed"] for r in rows}, key=int)
    sizes = sorted({int(r["n_chains"]) for r in rows})
    if len(seeds) != 16:
        print(f"REFUSING: {len(seeds)} seeds, expected 16", file=sys.stderr)
        sys.exit(1)

    net = fitted_slopes(rows, NET, sizes, seeds)
    em = fitted_slopes(rows, EM, sizes, seeds)
    diff = em - net                       # paired within seed
    t_stat = float(np.mean(diff) / se(diff))
    n_steeper = int(np.sum(diff < 0))

    # Robustness: the nseq=4096 row runs a different protocol (raised budget), so
    # the fit must not depend on it.
    #
    # The exclusion below is hardcoded to 4096, and \rategapnofourk is named for
    # it. nseq=8192 now exists (array 633361, 16 seeds) and also runs a raised
    # budget, so adding it to EXTENDED without touching this line would leave the
    # macro claiming to exclude the raised-budget rows while including one of
    # them -- and the prose quotes that macro. Fail loudly instead: the fix is to
    # exclude every raised-budget size and RENAME the macro, so that any prose
    # still carrying the old name breaks the build rather than rendering a number
    # that no longer means what its sentence says.
    RAISED_BUDGET = {4096}
    present_raised = {n for n in sizes if n in RAISED_BUDGET} | {
        n for n in sizes if n > max(RAISED_BUDGET)}
    if present_raised != RAISED_BUDGET:
        sys.exit(
            f"REFUSING: sizes {sorted(present_raised)} run a raised budget but the "
            f"robustness exclusion and \\rategapnofourk are hardcoded to 4096. "
            f"Exclude all of them and rename the macro before regenerating.")
    no4096 = [n for n in sizes if n != 4096]
    d_no4096 = (fitted_slopes(rows, EM, no4096, seeds)
                - fitted_slopes(rows, NET, no4096, seeds))

    # The censoring direction, which licenses the whole claim.
    gains = {}
    for label, key in (("net", NET), ("em", EM)):
        a = per_seed_by_size([r for r in rows if int(r["n_chains"]) == CALIB_N], key)
        b = per_seed_by_size(calib, key)
        rel = np.array([
            (b[s][CALIB_N] - a[s][CALIB_N]) / a[s][CALIB_N] for s in seeds
        ])
        gains[label] = (100.0 * float(np.mean(rel)), 100.0 * se(rel))

    advantage = gains["em"][0] / gains["net"][0] if gains["net"][0] else float("inf")
    if advantage < MIN_EM_ADVANTAGE:
        print(
            f"REFUSING: tripling the budget helps EM-BP {advantage:.2f}x the "
            f"network (em {gains['em'][0]:.2f}%, net {gains['net'][0]:.2f}%). The "
            f"rate claim rests on EM-BP being the censored arm, so that the "
            f"measured gap is a lower bound. At this ratio it is not, and the "
            f"claim must be rewritten or withdrawn rather than reworded.",
            file=sys.stderr,
        )
        sys.exit(1)

    body = f"""%% GENERATED by tools/make_rate_analysis.py -- do not hand-edit.
%%
%% Fitted log--log slopes of the SCHEDULE-POOLED relative denoising error
%% against nseq, per seed and
%% then across seeds, over the MEASURED RANGE nseq = {sizes[0]} to {sizes[-1]}. Not an
%% asymptotic rate; the macro names say "fitted" so the prose cannot drift into
%% claiming one.
%%
%% The only serious objection is budget censoring, and its direction is measured
%% rather than assumed: tripling both caps at nseq={CALIB_N} lowers the network's
%% error by {abs(gains['net'][0]):.2f}% and EM--BP's by {abs(gains['em'][0]):.2f}%. EM--BP is the arm held
%% back, its cap-hit rate rises with nseq, so the censoring flattens ITS curve and
%% the gap below is a lower bound. The generator refuses to emit if that ever
%% reverses.
\\newcommand{{\\ratenet}}{{{np.mean(net):.3f}}}
\\newcommand{{\\ratenetse}}{{{se(net):.3f}}}
\\newcommand{{\\rateem}}{{{np.mean(em):.3f}}}
\\newcommand{{\\rateemse}}{{{se(em):.3f}}}
\\newcommand{{\\rategap}}{{{abs(np.mean(diff)):.3f}}}
\\newcommand{{\\rategapse}}{{{se(diff):.3f}}}
\\newcommand{{\\rategapt}}{{{abs(t_stat):.1f}}}
\\newcommand{{\\ratesteeperseeds}}{{{n_steeper}}}
\\newcommand{{\\rateseeds}}{{{len(seeds)}}}
\\newcommand{{\\rategapnofourk}}{{{abs(np.mean(d_no4096)):.3f}}}
\\newcommand{{\\rategapnofourkse}}{{{se(d_no4096):.3f}}}
\\newcommand{{\\ratecalibnet}}{{{abs(gains['net'][0]):.2f}}}
\\newcommand{{\\ratecalibem}}{{{abs(gains['em'][0]):.2f}}}
"""
    with open(OUT, "w") as fh:
        fh.write(body)

    print(f"wrote {OUT}")
    print(f"  network slope   {np.mean(net):+.4f} +/- {se(net):.4f}")
    print(f"  EM-BP   slope   {np.mean(em):+.4f} +/- {se(em):.4f}")
    print(f"  gap             {np.mean(diff):+.4f} +/- {se(diff):.4f}  "
          f"t={t_stat:.2f}, steeper in {n_steeper}/{len(seeds)} seeds")
    print(f"  gap without 4096 {np.mean(d_no4096):+.4f} +/- {se(d_no4096):.4f}")
    print(f"  budget calibration: net {gains['net'][0]:+.2f}%, "
          f"em {gains['em'][0]:+.2f}%  (EM held back {advantage:.1f}x more)")


if __name__ == "__main__":
    main()
