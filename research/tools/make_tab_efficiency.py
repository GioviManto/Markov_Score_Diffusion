"""Emit the efficiency table and its macros into Markov_Score_Diffusion/thesis/sections/.

Generated rather than hand-typed, because this table is the paper's headline and
it has already been wrong twice: once from an EM budget nobody had checked, and
once from selecting EM's budget while pinning the network's.

Aggregation is PER SEED and then across seeds. Cells within a seed share one
training set and one set of fitted models, so the twelve noise levels move
together; treating 84 cells as 84 independent observations would misstate the
interval in both directions (it understates it for the ratio and overstates it
for the error columns).

THE ESTIMAND IS THE SCHEDULE-POOLED RELATIVE L2 ERROR the thesis defines
(Eq. 10.2): squared score error and reference score energy are summed over the
twelve noise levels BEFORE the square root, per seed, and the headline ratio is
the paired per-seed ratio of the two pooled errors, averaged over seeds. The
per-level reference energies B_t needed for the pooling are recovered exactly
by tools/make_ref_energy.py: the test bundle is shared across seeds, so B_t is
one number per level, and B_t * E_t^2 reconstructs the raw pooled sums from the
stored relative errors with no approximation. Earlier revisions of this file
reported the arithmetic mean over levels of the per-level errors and the mean
of the per-cell ratios instead -- a different, systematically larger estimand
(7.0-17.6 mean-cellwise against 6.6-10.9 pooled on the same cells). That
summary survives as an alternative in the aggregation appendix; it is no
longer the headline.
"""
import csv, glob, sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance_gate import code_path_dirty, import_closure, load_params  # noqa: E402

sys.path.insert(0, "experiments")
from frozen_config import FROZEN  # noqa: E402

# Derived, never typed. theta is [rho, pi (C), mu (C), s2 (C)] = 1 + 3C numbers,
# of which pi is simplex-constrained, so 3C are free. The table said "12 free"
# -- correct at C=4, and the frozen config has used C=8 since the paired-design
# sweep, so the headline column was understating the estimator by half.
N_FREE = 3 * FROZEN.n_components

# The abstract spells small integers rather than setting them in maths. Only the
# range the prose actually quotes needs covering; a KeyError here is the right
# failure, because it means the ratio moved somewhere the wording was not written
# for and the sentence needs rereading, not a wider lookup table.
_WORDS = {6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
          12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
          16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
          20: "twenty", 21: "twenty-one", 22: "twenty-two"}

# The certified run (job 630845): same protocol and seeds as exp_07_symmetric,
# plus the resolution certificate, the density-level Hellinger columns, and the
# corrected ECM inner budget. exp_07_symmetric_seed* is its predecessor and is
# kept only so the two can be compared; it must not be the source, because it
# carries no em_resolved and the gate below would then have nothing to check.
SOURCE = "outputs/frozen/exp_07_certified_seed*/sample_efficiency_val.csv"

# Sizes beyond the certified grid, each run at a raised budget because the
# certified caps were chosen for nseq <= 2048. That budget difference is not
# assumed harmless -- it is measured, by CALIB below.
#
#   nseq=4096  jobs 631496/631497, H200, em_iters=2400, net_steps=60000
#   nseq=8192  job 633361, H200, em_iters=3200, net_steps=100000
#
# 8192 only became affordable once the EM E-step went on the device: until then
# BP_DEVICE reached bp_grid and denoiser but not src/em.py, so the arm that
# dominates the cost ran on the CPU whatever partition the job was sent to.
# nseq=8192 IS WITHDRAWN. Round-two review, blocking item 8.
#
# Its params record a tree with 207 uncommitted paths, and unlike the 4096 rows
# that is not incidental. Intersecting the dirty list with exp_07's fifteen-file
# import closure (tools/provenance_gate.import_closure) leaves three files that
# this run actually executes:
#
#     experiments/exp_07_em_vs_score_network.py
#     experiments/frozen_config.py
#     src/em.py
#
# The experiment, the configuration and the EM implementation were all
# uncommitted. Whatever produced those numbers is not recoverable, and the row
# sits at the end of the curve where it carries the most rhetorical weight.
#
# The 4096 rows stay. Their tree carried eight uncommitted paths and the same
# intersection is EMPTY -- all eight are wavelet and FID work, none of it on
# this code path, so no edit in that tree could have moved these numbers. That
# is a materially weaker defect than 8192's and is disclosed in the caption
# rather than treated as equivalent. Refusing both alike would be over-strict
# and, worse, would tell a reader nothing about which is which.
EXTENDED = (
    "outputs/frozen/exp_07_n4096_seed*/sample_efficiency_val.csv",
)
WITHDRAWN = {
    # LaTeX-safe: this string goes straight into a caption, and bare
    # underscores in text mode are a build error.
    8192: (r"source tree had \texttt{exp\_07}, \texttt{frozen\_config} and "
           r"\texttt{src/em.py} uncommitted"),
}

# nseq=2048 at the raised budget (job 631467), the same seeds and bundles as the
# certified run. This exists to answer one question and no other: the caption used
# to call the largest ratio "budget-limited" because both arms often selected
# their largest allowed budget there. Selecting the cap is not the same as being
# bounded by it, and the difference is measurable -- so it was measured, by
# rerunning one size with both caps raised and pairing on seed.
CALIB = "outputs/frozen/exp_07_budget2048_seed*/sample_efficiency_val.csv"
CALIB_N = 2048


def load(pattern, what):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"REFUSING: no {what} run found at {pattern}", file=sys.stderr)
        sys.exit(1)
    out = []
    for f in files:
        seed = f.split("seed")[1].split("/")[0]
        with open(f) as fh:
            for r in csv.DictReader(fh):
                r["seed"] = seed
                out.append(r)
    return out


rows = load(SOURCE, "certified")
ext_blocks = [load(p, f"extended-nseq ({p.split('exp_07_')[1].split('_seed')[0]})")
              for p in EXTENDED]
calib = load(CALIB, "budget-calibration")

seeds = sorted({r["seed"] for r in rows}, key=int)
F = lambda r, k: float(r[k])

EXPECTED = 16
checks = [("certified", rows), ("calibration", calib)]
checks += [(f"extended[{i}]", b) for i, b in enumerate(ext_blocks)]
for label, block in checks:
    s = sorted({r["seed"] for r in block}, key=int)
    if len(s) != EXPECTED:
        print(f"REFUSING: {label} has {len(s)} seeds, expected {EXPECTED}: "
              f"{','.join(s)}", file=sys.stderr)
        sys.exit(1)
    # The calibration is a PAIRED comparison and the extended row is aggregated
    # per seed alongside the rest; both are meaningless if the seeds differ.
    if s != seeds:
        print(f"REFUSING: {label} seeds differ from certified", file=sys.stderr)
        sys.exit(1)

# Each extended run must contribute exactly one size, and one the table does not
# already have. Two runs landing on the same nseq would silently average two
# different budgets into one row, which is the failure this whole file exists to
# prevent.
base_sizes = {int(r["n_chains"]) for r in rows}
seen = set(base_sizes)
ext_ns = []
for pattern, block in zip(EXTENDED, ext_blocks):
    s = {int(r["n_chains"]) for r in block}
    if len(s) != 1 or (s & seen):
        print(f"REFUSING: {pattern} covers {sorted(s)}, which is not a single "
              f"new size beyond {sorted(seen)}", file=sys.stderr)
        sys.exit(1)
    n = s.pop()
    seen.add(n)
    ext_ns.append(n)
    rows = rows + block

# The largest extended size is the one the refusal message names; the caption
# names them all, since a reader checking which rows share a protocol should not
# have to infer it from the largest.
EXT_N = max(ext_ns)
sizes = sorted(seen)
_ns = sorted(ext_ns)
# Named from the globs actually loaded, so the provenance header cannot go on
# crediting exp_07_n4096_seed*/ for rows that came from somewhere else.
ext_provenance = " and ".join(
    p.split("/")[2] + "/" for p in EXTENDED
)

# The verb has to agree with the number of extended rows, or the caption reads
# "The nseq = 4096 and nseq = 8192 rows uses a raised budget" the moment a
# second size is added -- which is exactly what happened when 8192 landed.
ext_phrase = (f"The $\\nseq = {_ns[0]}$ row uses"
              if len(_ns) == 1 else
              "The $\\nseq = " + "$ and $\\nseq = ".join(str(n) for n in _ns)
              + "$ rows use")

# Resolution gate.
#
# An under-resolved mixture component -- narrower than two grid cells -- can raise
# the quadrature likelihood without corresponding to anything in the innovation
# law, so a cell fitted that way is not evidence. Those cells are dropped.
#
# Dropping rather than refusing outright, because the question that matters is
# not "were any cells unresolved" but "does the answer depend on them". So the
# exclusion is measured: if removing the unresolved cells moves any ratio by more
# than MAX_SHIFT, the result rested on fits that are lattice artefacts and must
# not ship. If it does not, the result is robust and the drop is bookkeeping.
#
# Measured on the certified run: 16 of 1344 cells, 13 of them at the smallest
# sample size and 11 from a single seed, almost all having run to the iteration
# cap -- small sample, long budget, one component collapsing onto the mesh. The
# largest ratio shift from excluding them is 0.081.
UNRESOLVED_FRAC_MAX = 0.05
# Set to the tolerance the PAPER states, not to a value the realised shift
# happens to clear. The caption says excluding the unresolved cells changes no
# ratio by more than 0.08; a gate at 0.5 would let the shift grow sixfold and
# still pass, i.e. it would permit the caption to become false while reporting
# success. 0.1 is the smallest round number above the realised 0.081, so the
# gate now fails at roughly the point the sentence it protects stops being true.
MAX_SHIFT = 0.1

dropped = []
if "em_resolved" in (rows[0] if rows else {}):
    dropped = [r for r in rows if not int(r["em_resolved"])]
    frac = len(dropped) / len(rows)
    if frac > UNRESOLVED_FRAC_MAX:
        print(f"REFUSING: {len(dropped)} of {len(rows)} cells ({100*frac:.1f}%) "
              f"have a mixture component under two grid cells wide. That is too "
              f"many to treat as bookkeeping -- refit on a finer grid or with a "
              f"stronger variance floor.", file=sys.stderr)
        sys.exit(1)
else:
    print("  resolution gate: NOT CERTIFIED -- these outputs predate "
          "em_resolved. Rerun exp_07 part5 to certify.", file=sys.stderr)

# The analysis in the header, made executable.
#
# exp_07 predates hpc/deploy_clean.sh, so no archive digest exists and the plain
# gate refuses it. What makes the retained rows citable anyway is not that fact
# but a narrower one: the intersection of each run's recorded dirty list with
# this experiment's import closure is EMPTY. That was checked by hand once and
# written into the header above, where it could go stale silently the moment a
# new seed directory was added. It is checked here on every run instead.
_closure = import_closure("experiments/exp_07_em_vs_score_network.py")
_dirs = sorted({os.path.dirname(f) for pat in (SOURCE, *[e[0] for e in EXTENDED])
                for f in glob.glob(pat)})
_bad = code_path_dirty(load_params(_dirs), _closure)
if _bad:
    print("REFUSING: these runs were produced with files uncommitted on the "
          "code path they execute, so their recorded commit does not "
          "reconstruct them:", file=sys.stderr)
    for k, v in sorted(_bad.items()):
        print(f"    {k}: {', '.join(v)}", file=sys.stderr)
    sys.exit(1)
print(f"  provenance: {len(_dirs)} run directories, legacy digest, "
      f"empty dirty-list intersection with the executed code path")

all_rows = rows
rows = [r for r in rows if "em_resolved" not in r or int(r["em_resolved"])]

# Reference score energies per level, from the shared frozen test bundle.
# These are the denominators the stored relative errors were formed with, so
# B_t * E_t^2 recovers each cell's raw pooled squared error exactly.
REF_ENERGY = "outputs/frozen/exp_07_ref_energy/ref_energy.csv"
try:
    with open(REF_ENERGY) as fh:
        B = {float(r["t"]): float(r["ref_sq"]) for r in csv.DictReader(fh)}
except FileNotFoundError:
    print(f"REFUSING: no reference energies at {REF_ENERGY}; run "
          f"tools/make_ref_energy.py first.", file=sys.stderr)
    sys.exit(1)
missing = {float(r["t"]) for r in rows} - set(B)
if missing:
    print(f"REFUSING: levels {sorted(missing)} have no reference energy",
          file=sys.stderr)
    sys.exit(1)


def pooled_rel_l2(cells, key):
    """Schedule-pooled relative L2 (thesis Eq. 10.2) for one seed and size:
    sqrt( sum_t B_t E_t^2 / sum_t B_t ) over the retained levels."""
    num = sum(B[float(r["t"])] * F(r, key) ** 2 for r in cells)
    den = sum(B[float(r["t"])] for r in cells)
    if not (num >= 0.0 and den > 0.0):
        raise ValueError((num, den))
    return (num / den) ** 0.5


def per_seed_pooled(g, key):
    v = np.array([pooled_rel_l2([r for r in g if r["seed"] == s], key)
                  for s in seeds])
    return v


def per_seed(g, key):
    """Legacy summary: per-seed arithmetic mean over levels. Kept for the
    aggregation appendix and the cross-check printout; not the headline."""
    v = np.array([np.mean([F(r, key) for r in g if r["seed"] == s]) for s in seeds])
    return v.mean(), v.std(ddof=1) / np.sqrt(v.size)


def paired_ci(values, draws=20000, seed=20260912):
    """Percentile bootstrap over the PAIRED per-seed ratios.

    The seed is the resampling unit because it owns a training set and both
    fitted models; a standard error alone assumes a symmetry that a ratio of
    two positive quantities over sixteen seeds need not have, which is why the
    audit asked for an interval rather than more digits on the s.e.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(draws, values.size))
    boot = values[idx].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


lines, ratios, ratios_cellwise, ratio_cis = [], [], [], []
for n in sizes:
    g = [r for r in rows if int(r["n_chains"]) == n]
    net_p = per_seed_pooled(g, "net_score_rel_l2_selected")
    em_p = per_seed_pooled(g, "em_bp_score_rel_l2")
    R_s = net_p / em_p                     # paired within seed
    rm, rse = R_s.mean(), R_s.std(ddof=1) / np.sqrt(R_s.size)
    nm, nse = net_p.mean(), net_p.std(ddof=1) / np.sqrt(net_p.size)
    em, ese = em_p.mean(), em_p.std(ddof=1) / np.sqrt(em_p.size)
    lo_ci, hi_ci = paired_ci(R_s)
    ratios.append(rm)
    ratio_cis.append((lo_ci, hi_ci))
    ratios_cellwise.append(per_seed(g, "ratio_selected")[0])
    lines.append(f"{n:<4} & ${nm:.4f} \\pm {nse:.4f}$ & "
                 f"$\\mathbf{{{em:.4f} \\pm {ese:.4f}}}$ & ${rm:.1f} \\pm {rse:.1f}$ "
                 f"& $[{lo_ci:.1f},\\, {hi_ci:.1f}]$ \\\\")

n_cells = len(rows)
net_agree = 100 * np.mean([int(r["net_steps_agrees"]) for r in rows])
em_agree = 100 * np.mean([int(r["em_iters_agrees"]) for r in rows])

# The dedicated validation-selection audit, a separate sweep from the table
# above. Chapter 9 used to quote it as "33 of 35 cells" and a mean advantage
# moving 11.65 -> 11.66; those came from a handful of seeds and do not survive
# at sixteen. The point of the paragraph is that the correction is small, and
# it still is -- but the numbers are now read off the run rather than typed.
VALSEL = "outputs/frozen/exp_07_valsel_seed*/sample_efficiency_val.csv"
vs_rows = []
for f in sorted(glob.glob(VALSEL)):
    with open(f) as fh:
        vs_rows += list(csv.DictReader(fh))
if not vs_rows:
    print(f"REFUSING: no validation-selection audit at {VALSEL}", file=sys.stderr)
    sys.exit(1)
vs_agree = 100 * np.mean([int(r["selection_agrees"]) for r in vs_rows])
vs_sel = float(np.mean([F(r, "ratio_selected") for r in vs_rows]))
vs_orc = float(np.mean([F(r, "ratio_oracle") for r in vs_rows]))
wins = sum(1 for r in rows if F(r, "ratio_selected") < 1)
big = sizes[-1]
lo, hi = min(ratios), max(ratios)
cw_lo, cw_hi = min(ratios_cellwise), max(ratios_cellwise)

# Finite-range descriptive slopes of the pooled error curves: log-log OLS of
# the seed-mean pooled error against nseq over the tested sizes. Quoted as
# fitted exponents of an eight-point fit, never as asymptotic rates.
def _fit_rate(key):
    """(slope, s.e.) of the log--log fit, the seed being the unit.

    The point estimate fits the seed-mean curve, as before. The standard
    error comes from fitting EACH SEED's own curve and taking the spread
    across seeds: a seed owns a training set and a fitted model, so it is
    the independent replicate, and a residual-based s.e. from the eight
    averaged points would treat correlated sizes as independent and read
    far too small.
    """
    ys, per_seed = [], {}
    for n in sizes:
        g = [r for r in rows if int(r["n_chains"]) == n]
        col = per_seed_pooled(g, key)
        ys.append(col.mean())
        for i, v in enumerate(col):
            per_seed.setdefault(i, []).append(v)
    Amat = np.vstack([np.log(sizes), np.ones(len(sizes))]).T
    slope = np.linalg.lstsq(Amat, np.log(ys), rcond=None)[0][0]
    each = np.array([
        np.linalg.lstsq(Amat, np.log(np.array(v)), rcond=None)[0][0]
        for v in per_seed.values()])
    se = float(each.std(ddof=1) / np.sqrt(each.size))
    return -float(slope), se


rate_em, rate_em_se = _fit_rate("em_bp_score_rel_l2")
rate_base, rate_base_se = _fit_rate("net_score_rel_l2_selected")

# The budget calibration.
#
# The previous caption asserted that the ratio at the largest size was
# "budget-limited", inferring it from how often each arm selected its largest
# allowed budget. That inference is wrong, and the error is instructive: an arm
# selects the last checkpoint whenever validation error is still falling, however
# slowly, so cap-selection measures the SIGN of the remaining gain and says
# nothing about its SIZE.
#
# So the size was measured. exp_07_budget2048 reruns nseq=2048 with the EM cap
# tripled and the network's cap tripled, on the same seeds and the same
# validation and test bundles, which makes the comparison paired -- necessary
# here, because the seed-to-seed spread in the ratio is an order of magnitude
# larger than the effect being looked for.
def paired_delta(a_rows, b_rows, n, key="ratio_selected"):
    """Per-seed pooled paired ratio at size `n`, differenced seed by seed.

    `key="ratio_selected"` is interpreted as the pooled paired ratio (the
    headline estimand); any other key is the pooled error for that column."""
    def by_seed(block):
        keep = [r for r in block if int(r["n_chains"]) == n
                and ("em_resolved" not in r or int(r["em_resolved"]))]
        out = []
        for s in seeds:
            cells = [r for r in keep if r["seed"] == s]
            if key == "ratio_selected":
                out.append(pooled_rel_l2(cells, "net_score_rel_l2_selected")
                           / pooled_rel_l2(cells, "em_bp_score_rel_l2"))
            else:
                out.append(pooled_rel_l2(cells, key))
        return np.array(out)
    d = by_seed(b_rows) - by_seed(a_rows)
    return d.mean(), d.std(ddof=1) / np.sqrt(d.size)


cal_d, cal_se = paired_delta(rows, calib, CALIB_N)
cal_net = paired_delta(rows, calib, CALIB_N, "net_score_rel_l2_selected")[0]
cal_net_base = np.mean([F(r, "net_score_rel_l2_selected") for r in rows
                        if int(r["n_chains"]) == CALIB_N])

# If tripling both budgets moved the ratio by more than this, the caps really were
# binding and the caption must say so rather than dismiss it. The measured value
# is -0.16 +/- 0.18, so the branch below reports "does not move"; the threshold is
# here so that a future rerun which DOES move cannot quietly keep the old wording.
CALIB_MAX = 1.0
if abs(cal_d) > CALIB_MAX:
    print(f"REFUSING: tripling both budgets moves the nseq={CALIB_N} ratio by "
          f"{cal_d:+.2f} (> {CALIB_MAX}). The caps bind, so the table must not "
          f"carry an nseq={EXT_N} row run at a different budget, and the caption "
          f"must state the limitation rather than dismiss it.", file=sys.stderr)
    sys.exit(1)

# Does the answer depend on the cells that were dropped?
#
# This is the check that makes dropping them legitimate rather than convenient.
# Recompute every ratio on the FULL set and compare: if excluding the
# under-resolved fits moves a ratio materially, the published number rested on
# lattice artefacts and must not ship, however few of them there were.
if dropped:
    shifts = []
    for i, n in enumerate(sizes):
        g_all = [r for r in all_rows if int(r["n_chains"]) == n]
        v = np.array([pooled_rel_l2([r for r in g_all if r["seed"] == s],
                                    "net_score_rel_l2_selected")
                      / pooled_rel_l2([r for r in g_all if r["seed"] == s],
                                      "em_bp_score_rel_l2")
                      for s in seeds])
        shifts.append(abs(ratios[i] - v.mean()))
    worst_i = int(np.argmax(shifts))
    if shifts[worst_i] > MAX_SHIFT:
        print(f"REFUSING: excluding the {len(dropped)} under-resolved cell(s) "
              f"moves the ratio at n={sizes[worst_i]} by {shifts[worst_i]:.3f} "
              f"(> {MAX_SHIFT}). The result depends on fits whose narrowest "
              f"mixture component is thinner than the mesh; refit before "
              f"reporting.", file=sys.stderr)
        sys.exit(1)
    print(f"  resolution gate: dropped {len(dropped)} of {len(all_rows)} cells "
          f"(worst s_min/h {min(float(r['em_s_min_over_h']) for r in dropped):.2f}); "
          f"largest ratio shift {max(shifts):.3f}, at n={sizes[worst_i]}")
else:
    print(f"  resolution gate: all {len(rows)} cells resolved "
          f"(min s_min/h {min(float(r['em_s_min_over_h']) for r in rows):.2f})")

# Density-level recovery, reported alongside the score error it can outlive.
hell = np.array([F(r, "em_hellinger") for r in rows]) if "em_hellinger" in rows[0] else None

# Stated in the caption rather than left to a footnote: a reader is entitled to
# know that cells were excluded, how many, and that excluding them did not move
# the answer.
drop_note = ""
if dropped:
    drop_note = (f" {len(dropped)} of {len(all_rows)} cells are excluded because the "
                 f"narrowest fitted mixture component is under two grid cells wide; "
                 f"including them changes no ratio by more than ${max(shifts):.2f}$.")

# An excluded row must be visible in the artefact a reader actually looks at.
# nseq=8192 set the top of the quoted range, so dropping it silently would let
# a reader who saw an earlier circulated draft conclude the number drifted on
# its own -- and would hide a real limit on what this table covers.
withdrawn_note = ""
if WITHDRAWN:
    # A pointer, not the argument: the caption is already dense and the
    # appendix is where provenance belongs. What must not happen is the row
    # disappearing with no trace.
    _w = ", ".join(f"$\\nseq={n}$" for n in sorted(WITHDRAWN))
    # \provenanceappendix, not a hard \ref: this file is included by BOTH the
    # paper and the thesis, and they keep the provenance discussion in
    # differently-named appendices. A literal label resolves in one document
    # and prints "??" in the other -- which is how the thesis came to reference
    # the paper's app:protocol.
    # State the exclusion as a current fact about the table, not as a
    # revision history of it. A production caption should tell a reader what
    # this table covers and why; which drafts once carried the row is the
    # compendium's business.
    withdrawn_note = (
        f" {_w} was run but is excluded: its source is not recoverable "
        f"(\\provenanceappendix)."
    )

body = "\n".join(lines)
# `$1,344$` sets the comma as a binary-operator-ish relation with the wrong
# spacing; `$1{,}344$` is the correct way to write a thousands separator in math
# mode, and is what the hand-written table used.
th = lambda x: f"{x:,}".replace(",", "{,}")
tex = f"""%% Included by the paper and the thesis (Chapter 10). The
%% workshop does not carry it: at four pages it presents the analytical result,
%% and a comparison whose whole content is a protocol cannot be defended in the
%% space available.
%%
%% GENERATED by tools/make_tab_efficiency.py from outputs/frozen/
%% exp_07_certified_seed*/ (nseq <= {CALIB_N}) and {ext_provenance}
%% ({len(seeds)} seeds, {n_cells} cells). The budget calibration in the caption comes
%% from exp_07_budget2048_seed*/. Do not hand-edit the numbers; rerun the generator.

\\begin{{table}}[!htbp]
\\caption[Relative denoising error against training-set size]{{Schedule-pooled relative score error against the deterministic grid-BP
reference under the true kernel: squared error and reference score energy are summed over the
twelve noise levels before the square root, per seed; {len(seeds)} seeds $\\pm$ one standard error.
Both arms select their optimisation budget on a disjoint validation bundle. EM--BP is
more accurate in ${th(n_cells - wins)}$ of ${th(n_cells)}$ (seed, size, level) cells --- a
descriptive count, not $\\,{th(n_cells)}$ replications, since the twelve levels of a
cell share a training set and a fitted model and the inferential unit stays the
$16$ paired seeds. The ratio
column is the paired per-seed ratio of the two pooled errors, averaged over seeds,
with a percentile bootstrap over those ratios beside it (seeds resampled, not
cells) --- close to, but
not identical with, the quotient of the two columns beside it;
Appendix~\\ref{{app:aggregation}} reports the same cells under five alternative aggregations.
Protocol, cell exclusions and the budget calibration are in \\provenanceappendix.}}
\\label{{tab:pointwise}}
\\centering
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{rcccc}}
\\toprule
sequences $\\nseq$ & network & EM--BP (${N_FREE}$ free) & ratio & 95\\% CI \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

dest = "../../Markov_Score_Diffusion/thesis/sections/tab-efficiency.tex"

# NOTHING IS WRITTEN UNTIL BOTH FILES ARE BUILT.
#
# The table used to be written here, before the macro block below. That made the
# two outputs non-atomic, and the failure is not hypothetical: when the ratio
# first exceeded eighteen, the deliberate KeyError in _WORDS fired *after* the
# table had already been written, leaving a nine-row table quoting 20.2 beside a
# macro file still saying \ratiohi{17.6} and \biggestn{4096}. The paper would
# have built, with the prose and the table disagreeing -- which is the single
# thing this generator exists to make impossible.
#
# So both strings are constructed first and written only once neither can fail.

# The prose gets macros, not typed numbers.
#
# The abstract said "between 8 and 14" for three weeks after this generator
# started emitting 7.3-15.7, and no check caught it because one number was typed
# in main.tex and the other computed here. A gate comparing the two would need a
# tolerance, and the tolerance would be tuned until it passed -- so instead there
# is nothing to compare: the prose cites \ratiolo and cannot hold a stale value.
macros = f"""%% GENERATED by tools/make_tab_efficiency.py -- do not hand-edit.
%%
%% Every efficiency number the prose quotes is defined here and nowhere else, so
%% that the abstract, the body and the table cannot disagree. If a value looks
%% wrong, rerun the generator; do not edit the number in the text.
%%
%% \\ratiolosub, \\ratiohisub, \\ratiohisubword and \\nsizessub USED TO BE DEFINED
%% HERE and deliberately are not any more. They existed to quote the range over
%% the sizes at which neither arm selected its largest budget, because the largest
%% size was believed to be budget-limited. That belief was tested by rerunning
%% nseq={CALIB_N} with both budgets tripled and found to be false. Deleting the
%% macros rather than redefining them is the point: any prose still carrying the
%% old framing fails to build instead of silently rendering a number that no
%% longer means what the surrounding sentence says it means.
%% \\ratiolo/\\ratiohi are the SCHEDULE-POOLED paired per-seed ratios of
%% thesis Eq. (10.2) -- numerator and denominator energies summed over the
%% twelve levels before the square root, ratio formed within seed, then
%% averaged. Before September 2026 these macros carried the mean of the
%% per-(seed,size,level) ratios, a systematically larger summary
%% ({cw_lo:.1f}-{cw_hi:.1f} on the same cells); that survives as
%% \\ratiocellwiselo/\\ratiocellwisehi for the aggregation appendix.
\\newcommand{{\\ratiolo}}{{{lo:.1f}}}
\\newcommand{{\\ratiohi}}{{{hi:.1f}}}
\\newcommand{{\\ratioloword}}{{{_WORDS[round(lo)]}}}
\\newcommand{{\\ratiohiword}}{{{_WORDS[round(hi)]}}}
\\newcommand{{\\ratiocellwiselo}}{{{cw_lo:.1f}}}
\\newcommand{{\\ratiocellwisehi}}{{{cw_hi:.1f}}}
%% Endpoints of the pooled ratio across the size grid, and the fitted
%% finite-range slopes of the two pooled error curves (log-log OLS over the
%% {len(sizes)} sizes; descriptive, not asymptotic).
\\newcommand{{\\ratiofirst}}{{{ratios[0]:.1f}}}
\\newcommand{{\\ratiolast}}{{{ratios[-1]:.1f}}}
\\newcommand{{\\ratiogrowth}}{{{ratios[-1]/ratios[0]:.2f}}}
\\newcommand{{\\pooledslopeem}}{{{rate_em:.3f}}}
\\newcommand{{\\pooledslopeemse}}{{{rate_em_se:.3f}}}
\\newcommand{{\\pooledslopebase}}{{{rate_base:.3f}}}
\\newcommand{{\\pooledslopebasese}}{{{rate_base_se:.3f}}}
\\newcommand{{\\pooledgrowthpred}}{{{(sizes[-1]/sizes[0])**(rate_em-rate_base):.2f}}}
\\newcommand{{\\nfreeparams}}{{{N_FREE}}}
\\newcommand{{\\nseedsused}}{{{len(seeds)}}}
\\newcommand{{\\ncellsused}}{{{th(n_cells)}}}
%% Cells where the network wins outright. Quoted in the reproducibility
%% appendix, which used to type it as a literal 2.
\\newcommand{{\\nreversed}}{{{wins}}}
\\newcommand{{\\nsizesused}}{{{len(sizes)}}}
\\newcommand{{\\biggestn}}{{{big}}}
\\newcommand{{\\netagree}}{{{net_agree:.1f}}}
\\newcommand{{\\emagree}}{{{em_agree:.1f}}}
%% The validation-selection audit (exp_07_valsel_*), quoted in Section 9.1.
\\newcommand{{\\valselcells}}{{{th(len(vs_rows))}}}
\\newcommand{{\\valselagree}}{{{vs_agree:.1f}}}
\\newcommand{{\\valselratiosel}}{{{vs_sel:.2f}}}
\\newcommand{{\\valselratioorc}}{{{vs_orc:.2f}}}
%% The budget calibration, quoted wherever the text claims the comparison is not
%% an artefact of the optimisation caps.
\\newcommand{{\\nseqcalib}}{{{CALIB_N}}}
\\newcommand{{\\budgetdelta}}{{{cal_d:+.2f}}}
\\newcommand{{\\budgetdeltase}}{{{cal_se:.2f}}}
%% The resolution gate, exported because the caption no longer states it. It
%% used to: the caption carried the protocol, the exclusions, the validation
%% agreement rates and the budget calibration, and had grown into a methods
%% subsection set in small type under a four-row table. The facts still have to
%% be somewhere a reader can find them, so they moved to \\provenanceappendix
%% and are quoted there through these macros rather than retyped.
\\newcommand{{\\ndropped}}{{{len(dropped)}}}
\\newcommand{{\\ncellstotal}}{{{th(len(all_rows))}}}
\\newcommand{{\\dropshift}}{{{max(shifts) if shifts else 0:.2f}}}
\\newcommand{{\\extendedn}}{{{_ns[0] if len(_ns) == 1 else _ns[-1]}}}
"""
mdest = "../../Markov_Score_Diffusion/thesis/sections/efficiency-numbers.tex"

# The paper is at a hard nine-page body limit and the bootstrap column does
# not fit there; the thesis has the room and the audit asked for the interval.
# Rather than keep two hand-maintained variants -- which is exactly how two
# documents come to report the same cells differently -- render BOTH from this
# one computation, so the numbers cannot diverge and only the column set does.
compact = tex
for _lo, _hi in ratio_cis:
    compact = compact.replace(f" & $[{_lo:.1f},\\, {_hi:.1f}]$ \\\\", " \\\\")
compact = (compact
           .replace("\\begin{tabular}{rcccc}", "\\begin{tabular}{rccc}")
           .replace(" & ratio & 95\\% CI \\\\", " & ratio \\\\")
           .replace(", \nwith a percentile bootstrap over those ratios beside it (seeds resampled, not\ncells)", ""))
if "95\\% CI" in compact or "$[" in compact:
    raise SystemExit("REFUSING: the compact rendering still carries the CI column; "
                     "the row or header pattern changed and the two renderings "
                     "would disagree about what they show.")
pdest = "../../Markov_Score_Diffusion/paper/sections/tab-efficiency.tex"

# All strings exist and no construction raised. Now write.
open(dest, "w").write(tex)
open(pdest, "w").write(compact)
open(mdest, "w").write(macros)

print(f"wrote {dest}: {len(seeds)} seeds, {n_cells} cells, "
      f"{len(sizes)} sizes (to nseq={big}), ratio {lo:.1f}-{hi:.1f}")
print(f"wrote {mdest}: \\ratiolo={lo:.1f} \\ratiohi={hi:.1f} "
      f"\\biggestn={big} \\nfreeparams={N_FREE}")
print(f"  network val==test {net_agree:.1f}%, EM {em_agree:.1f}%, "
      f"network wins {wins} cells")
print(f"  budget calibration at nseq={CALIB_N}: ratio {cal_d:+.2f} +/- {cal_se:.2f}, "
      f"network error {100*cal_net/cal_net_base:+.1f}% (does not bind)")
