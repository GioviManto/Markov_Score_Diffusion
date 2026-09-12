"""Emit the non-Markov robustness table from the frozen outputs.

Written because the hand-typed version could not be reproduced. The compendium
quotes 18.3 for (Gaussian, vs CNN, beta=0); averaging `ratio_to_em` over t in
outputs/exp_21/gauss_beta0.0 gives 10.76, the median gives 11.43, and no single
t exceeds 14.21. The gamma rows are near but not equal (published 24.1, 1.19,
0.98, 0.86, 0.77 against computed 20.94, 1.24, 1.03, 0.85, 0.76), which is the
signature of a different RUN rather than a different aggregation. Whatever its
provenance, a table nobody can regenerate from the committed outputs is not
evidence, and this is the one table in the document that was still typed.

Source is the CLEAN-DEPLOYMENT rerun (arrays 645687 and 645903), which
replaces the frozen rerun (633406), which replaced the original (627165). The
original used em_iters=80 and net_steps=8000 against the frozen protocol's 400
and 20000, so its BASELINE was trained on 40% of the budget with no checkpoint
selection. Undertraining the baseline inflates every ratio in the table, and it
is the one direction of error that would not survive review. The frozen rerun
fixed that but ran from a dirty tree; the clean rerun fixes the provenance and
reproduces the frozen rerun's Gaussian numbers exactly, and adds the Laplace
arm, which the frozen tree never contained.

What the rerun changed, beyond making the numbers quotable:

  * beta=0 and gamma=0 are the SAME configuration -- no contamination, a pure
    Markov chain. At the old budget they gave 10.76 and 20.94, a factor of two
    apart. At the frozen budget they give 16.41 and 16.07, agreeing to 2%. The
    uncontaminated control now reproduces across two independent runs; before,
    it did not, and that alone should have blocked the table.
  * The old beta sweep ran 10.76 -> 31.47 -> 8.27 -> 3.44 -> 2.43. Contamination
    appeared to HELP at beta=0.1, which is not physical. The rerun gives
    16.41 -> 14.84 -> 6.68 -> 3.21 -> 2.16, monotone.

The qualitative conclusion is unchanged and is what the section should lead with:
rank-one contamination is survivable, long-range coupling is not, and the
crossover sits at gamma ~ 0.10.
"""
import csv, glob, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provenance_gate import code_path_dirty, import_closure, load_params  # noqa: E402

# Prefer the clean-deployment rerun; fall back to the tree it replaces.
#
# THE FALLBACK IS NOT A CONVENIENCE. outputs/exp_21_frozen ran with src/em.py
# uncommitted in all ten cells, so the recorded commit cannot reconstruct the
# program that produced those numbers -- the exp_12 defect, on the executed code
# path rather than elsewhere in the tree, which is why allow_legacy does not
# rescue it the way it rescues exp_07 and exp_27. This generator used to call no
# gate at all and emit a table that looked exactly as certified as the others.
# It now marks what it cannot certify, which is what Appendix C always claimed
# the generators did.
CLEAN_SOURCE = "outputs/frozen/exp_21_clean"
LEGACY_SOURCE = "outputs/exp_21_frozen"
ENTRY = "experiments/exp_21_nonmarkov.py"
SOURCE = CLEAN_SOURCE if glob.glob(f"{CLEAN_SOURCE}/*/nonmarkov_*.csv") else LEGACY_SOURCE
MECHANISMS = (
    ("beta", "global latent, rank-one, strength $\\beta$", (0.0, 0.1, 0.25, 0.5, 1.0)),
    ("gamma", "long-range precision coupling, strength $\\gamma$", (0.0, 0.05, 0.1, 0.2, 0.4)),
)
ARMS = ("cnn", "mlp")


def ratio(family, mech, strength, arm):
    """Mean of `ratio_to_em` over the noise schedule for one cell.

    Aggregated over t and stated as such, because the quantity varies by a factor
    of two across the schedule and any single-t number would be a choice the
    reader cannot see.
    """
    # Directory names carry the strength as it was formatted at submission, so
    # match numerically rather than by string to avoid 0.1 vs 0.10 misses.
    for d in glob.glob(f"{SOURCE}/{family}_{mech}*"):
        m = re.search(rf"{mech}([0-9.]+)$", d)
        if not m or abs(float(m.group(1)) - strength) > 1e-9:
            continue
        files = glob.glob(f"{d}/nonmarkov_{family}.csv")
        if not files:
            return None
        rows = [r for r in csv.DictReader(open(files[0])) if r["arm"] == arm]
        if not rows:
            return None
        return float(np.mean([float(r["ratio_to_em"]) for r in rows]))
    return None


def em_stat(family, mech, strength, column, reduce=np.mean):
    """A diagnostic of the EM--BP arm itself, rather than a ratio against a
    baseline. The mechanism claim -- a chain absorbs rank-one contamination and
    cannot absorb long-range coupling -- is a statement about the fitted chain,
    so it has to be read off the fit and not off the comparison."""
    for d in glob.glob(f"{SOURCE}/{family}_{mech}*"):
        m = re.search(rf"{mech}([0-9.]+)$", d)
        if not m or abs(float(m.group(1)) - strength) > 1e-9:
            continue
        files = glob.glob(f"{d}/nonmarkov_{family}.csv")
        if not files:
            return None
        rows = [r for r in csv.DictReader(open(files[0])) if r["arm"] == "em_bp"]
        if not rows:
            return None
        # Not every family records every diagnostic. The Laplace arm has no
        # Chow--Liu reference -- there is no closed form for the best chain
        # approximation to a contaminated Laplace law -- so the column is
        # absent rather than empty, and the row is simply not drawn for it.
        # Returning None here, rather than raising, is what keeps a missing
        # DIAGNOSTIC from being confused with a missing CELL.
        if column not in rows[0]:
            return None
        return float(reduce([float(r[column]) for r in rows]))
    return None


def worst_ratio(family, mech, strength, arm):
    """The ratio at its worst noise level, not its schedule mean. `never
    inverts` is a claim about every level, so it needs the minimum."""
    for d in glob.glob(f"{SOURCE}/{family}_{mech}*"):
        m = re.search(rf"{mech}([0-9.]+)$", d)
        if not m or abs(float(m.group(1)) - strength) > 1e-9:
            continue
        files = glob.glob(f"{d}/nonmarkov_{family}.csv")
        if not files:
            return None
        rows = [r for r in csv.DictReader(open(files[0])) if r["arm"] == arm]
        return float(min(float(r["ratio_to_em"]) for r in rows)) if rows else None
    return None


families = sorted({d.split("/")[-1].split("_")[0] for d in glob.glob(f"{SOURCE}/*")})
if not families:
    sys.exit(f"REFUSING: no non-Markov output at {SOURCE}")

# Certify, or say plainly that we cannot.
DIRTY = {}
try:
    DIRTY = code_path_dirty(load_params(sorted(glob.glob(f"{SOURCE}/*"))),
                            import_closure(ENTRY))
except SystemExit:
    DIRTY = {"(no params files)": ["provenance record absent"]}
# The Laplace and Gaussian arms were deployed separately and therefore carry
# different commits. The provenance gate's standing objection to mixed commits
# is that "two programs' results [get] averaged into one cell", so the two
# things that have to be true are checked here rather than assumed:
#
#   (1) no single CELL mixes commits -- each output directory is one deployment;
#   (2) across the commits present, every file in the experiment's import
#       closure is byte-identical, so "different commit" does not mean
#       "different program".
#
# If either fails the table is refused. A mixed-commit table that passes both
# is exactly as trustworthy as a single-commit one, and saying so beats either
# hiding the mixture or redoing work that would produce identical bytes.
def _commits_by_cell():
    by = {}
    for path, d in load_params(sorted(glob.glob(f"{SOURCE}/*"))):
        by.setdefault(path.parent.name, set()).add(str(d.get("git_commit", "")))
    return by


COMMITS = set()
_mixed_cells = []
for _cell, _cs in sorted(_commits_by_cell().items()):
    COMMITS |= _cs
    if len(_cs) > 1:
        _mixed_cells.append(f"{_cell}: {', '.join(sorted(c[:7] for c in _cs))}")
if _mixed_cells:
    sys.exit("REFUSING: these cells pool more than one commit:\n  "
             + "\n  ".join(_mixed_cells))

if len(COMMITS) > 1:
    import subprocess
    _repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    _rel = "research/nongaussian-bp"
    _blobs, _differ = {}, []
    for _f in sorted(import_closure(ENTRY)):
        _seen = set()
        for _c in sorted(COMMITS):
            _r = subprocess.run(["git", "-C", _repo, "rev-parse", f"{_c}:{_rel}/{_f}"],
                                capture_output=True, text=True)
            _seen.add(_r.stdout.strip() if _r.returncode == 0 else f"ABSENT@{_c[:7]}")
        if len(_seen) > 1:
            _differ.append(_f)
    if _differ:
        sys.exit("REFUSING: the source spans commits "
                 + ", ".join(sorted(c[:7] for c in COMMITS))
                 + " and these executed files differ between them:\n  "
                 + "\n  ".join(_differ)
                 + "\nThese are different programs; do not pool them.")
    print(f"  {len(COMMITS)} commits, identical across all "
          f"{len(import_closure(ENTRY))} files of the executed closure")

UNCERTIFIED = bool(DIRTY)
if UNCERTIFIED:
    offenders = sorted({f for v in DIRTY.values() for f in v})
    print(f"  NOT CERTIFIED: {SOURCE} ran with {', '.join(offenders)} "
          f"uncommitted on the executed code path, in {len(DIRTY)} cell(s). "
          f"The table is emitted with that stated in its caption.", file=sys.stderr)
else:
    print(f"  certified: {SOURCE}")

# The family directories are named `gauss_*` / `laplace_*`; the table names the
# innovation LAW, which is what the reader is tracking.
PRETTY = {"gauss": "Gaussian", "laplace": "Laplace"}

lines, missing = [], []
for i, (mech, header, strengths) in enumerate(MECHANISMS):
    if i:  # \toprule already supplies the rule above the first block
        lines.append("\\midrule")
    lines.append(f"& \\multicolumn{{{len(strengths)}}}{{c}}{{{header}}} \\\\")
    lines.append(f"\\cmidrule(lr){{2-{len(strengths)+1}}}")
    lines.append("& " + " & ".join(f"${s:.2f}$" for s in strengths) + " \\\\")
    lines.append("\\midrule")
    for family in families:
        for arm in ARMS:
            vals = [ratio(family, mech, s, arm) for s in strengths]
            if all(v is None for v in vals):
                continue
            missing += [(family, mech, s) for s, v in zip(strengths, vals) if v is None]
            cells = []
            for v in vals:
                if v is None:
                    cells.append("--")
                # Bold the crossover: below 1 the baseline wins, and that is the
                # scope statement, not a detail.
                elif v < 1.0:
                    cells.append(f"$\\mathbf{{{v:.2f}}}$")
                else:
                    cells.append(f"${v:.2f}$")
            lines.append(f"{PRETTY.get(family, family.capitalize())}, vs "
                         f"{arm.upper()} & " + " & ".join(cells) + " \\\\")
    # The two diagnostic rows are independent. They used to share a `continue`,
    # so a family with no Chow--Liu column lost its ERROR row as well -- which
    # would have silently dropped the one row that says whether a moving ratio
    # is the estimator degrading or the baseline improving.
    for family in families:
        tag = f"{PRETTY.get(family, family.capitalize())}: " if len(families) > 1 else ""
        dev = [em_stat(family, mech, s, "rho_minus_chow_liu") for s in strengths]
        if not all(v is None for v in dev):
            lines.append(tag + "$\\widehat{\\corr} - \\corr_{\\mathrm{CL}}$ & "
                         + " & ".join("--" if v is None else f"${v:+.4f}$" for v in dev)
                         + " \\\\")
        err = [em_stat(family, mech, s, "score_rel_l2", max) for s in strengths]
        if not all(v is None for v in err):
            lines.append(tag + "EM--BP error, worst $t$ & "
                         + " & ".join("--" if v is None else f"${v:.3f}$" for v in err)
                         + " \\\\")

if missing:
    print(f"  {len(missing)} cell(s) absent, emitted as '--': "
          + ", ".join(f"{f}/{m}={s}" for f, m, s in missing[:6]), file=sys.stderr)

body = "\n".join(lines)
caveat = ("\\textbf{Provenance: not certified.} The run behind this table was "
          "produced with " + ", ".join(f"\\texttt{{{f}}}".replace("_", "\\_")
                                       for f in sorted({f for v in DIRTY.values() for f in v}))
          + " uncommitted, so its recorded commit does not reconstruct the "
          "program that produced these numbers. A clean-deployment rerun of the "
          "same protocol is what should stand here. "
          ) if UNCERTIFIED else ""
tex = f"""%% GENERATED by tools/make_tab_nonmarkov.py from {SOURCE}/ -- do not hand-edit.
%% commit(s): {", ".join(sorted(c[:7] for c in COMMITS))} -- verified byte-identical over the executed import closure.
%%
%% Supersedes the hand-typed table, whose numbers could not be reproduced from
%% any committed output by any aggregation, and then the frozen-budget rerun
%% (array 633406), which was certifiable in protocol but not in provenance --
%% it ran with src/em.py uncommitted on the executed code path.
%%
%% Source is now the clean-deployment rerun: arrays 645687 (Laplace, five beta
%% cells) and 645903 (Gaussian, five beta and five gamma cells), both from
%% git-archive deployments with an empty dirty list. Every Gaussian number it
%% produces is identical, to the precision printed here, to the run it replaces
%% -- which is the evidence that the uncommitted file was not on this
%% experiment's path in any way that mattered. That could not be asserted
%% before; it can be now, and it is the reason the caption no longer carries a
%% provenance caveat.

\\begin{{center}}
%% minipage, not a bare center: without it LaTeX will break between the
%% caption and the tabular, and it did -- Table 9.5's caption sat alone at
%% the foot of one page with its rows at the head of the next.
\\begin{{minipage}}{{\\linewidth}}\\small\\centering
\\captionof{{table}}[Two controlled violations of the Markov assumption]{{{caveat}Two controlled violations of the Markov assumption, each
evaluated against the appropriate reference: the Gaussian rows against an exact linear-algebra reference for the contaminated model, the Laplace rank-one rows against the validated grid-BP reference. The first two rows of
each block are the baseline's relative score error divided by EM--BP's,
averaged over the noise schedule, so a value above $1$ means the structured
estimator wins and a bold value means it loses. The third row is the fitted
correlation minus the Chow--Liu correlation of the true contaminated law: the
distance between the chain that was fitted and the best chain there is. The
fourth is EM--BP's own relative score error at its worst noise level, which
says how much of a change in the ratio is the estimator degrading rather than
the baseline improving.}}
\\label{{tab:nonmarkov}}
\\begin{{tabular}}{{@{{}}lccccc@{{}}}}
\\toprule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{minipage}}
\\end{{center}}
"""

# The numbers Chapter 9 quotes in prose. They were typed, and they were typed
# from the superseded run: "never falls below 2.08" at beta=1 is 1.44 here, and
# the Chow--Liu column differed in every cell.
fam = "gauss"


def _cldev(mech):
    """Largest |fitted correlation - Chow--Liu correlation| over the mechanism's
    strengths. This is the distance between the chain that was fitted and the
    best chain available, so it separates "fitting the wrong chain" from
    "fitting the right chain badly"."""
    # MECHANISMS entries are (key, header, strengths) -- match on the KEY.
    vals = [em_stat(fam, mech, s, "rho_minus_chow_liu")
            for m, _, ss in MECHANISMS if m == mech for s in ss]
    vals = [abs(v) for v in vals if v is not None]
    if not vals:
        sys.exit(f"REFUSING: no Chow--Liu column for {fam}/{mech}")
    return max(vals)


crossover = next((s for s in MECHANISMS[1][2]
                  if (ratio(fam, "gamma", s, "cnn") or 9e9) < 1.0), None)
last_holding = max((s for s in MECHANISMS[1][2]
                    if (ratio(fam, "gamma", s, "cnn") or 0) >= 1.0), default=None)
beta_max = MECHANISMS[0][2][-1]
gamma_max = MECHANISMS[1][2][-1]
macros = "\n".join([
    "%% GENERATED by tools/make_tab_nonmarkov.py -- do not hand-edit.",
    f"\\newcommand{{\\nmbetamax}}{{{beta_max:g}}}",
    f"\\newcommand{{\\nmgammamax}}{{{gamma_max:g}}}",
    f"\\newcommand{{\\nmbetaminratio}}{{{min(worst_ratio(fam, 'beta', beta_max, a) for a in ARMS):.2f}}}",
    f"\\newcommand{{\\nmgammaminratio}}{{{min(worst_ratio(fam, 'gamma', gamma_max, a) for a in ARMS):.2f}}}",
    f"\\newcommand{{\\nmcrossover}}{{{crossover:g}}}",
    f"\\newcommand{{\\nmlastholding}}{{{last_holding:g}}}",
    f"\\newcommand{{\\nmemworsterr}}{{{100 * em_stat(fam, 'gamma', gamma_max, 'score_rel_l2', max):.0f}}}",
    f"\\newcommand{{\\nmrhoclean}}{{{em_stat(fam, 'beta', 0.0, 'fitted_rho'):.3f}}}",
    f"\\newcommand{{\\nmrhobetamax}}{{{em_stat(fam, 'beta', beta_max, 'fitted_rho'):.3f}}}",
    f"\\newcommand{{\\nmrhogammamax}}{{{em_stat(fam, 'gamma', gamma_max, 'fitted_rho'):.3f}}}",
    # The two uncontaminated controls are the same configuration reached from
    # two directions; their agreement is what makes the rest of the table
    # readable, and their disagreement is what invalidated the previous one.
    f"\\newcommand{{\\nmcontrolbeta}}{{{ratio(fam, 'beta', 0.0, 'cnn'):.2f}}}",
    f"\\newcommand{{\\nmcontrolgamma}}{{{ratio(fam, 'gamma', 0.0, 'cnn'):.2f}}}",
    # How far the fitted chain sits from the BEST chain (Chow--Liu) under each
    # mechanism, and the factor between them. Section 9.5 typed "within 0.015"
    # and "nearly an order of magnitude" by hand; both are read off the same
    # table row, so both belong here where they cannot drift from it.
    f"\\newcommand{{\\nmcldevbeta}}{{{_cldev('beta'):.3f}}}",
    f"\\newcommand{{\\nmcldevgamma}}{{{_cldev('gamma'):.3f}}}",
    f"\\newcommand{{\\nmcldevfactor}}{{{_cldev('gamma') / _cldev('beta'):.0f}}}",
    # The Laplace arm. It carries the rank-one mechanism only -- there is no
    # closed-form reference for a long-range-coupled Laplace chain, so that
    # half of the design does not exist for it and the table leaves the row
    # out rather than printing dashes. Section 9.5 reads the two innovation
    # laws together on the mechanism they share.
    f"\\newcommand{{\\nmlapcontrol}}{{{ratio('laplace', 'beta', 0.0, 'cnn'):.2f}}}",
    # Per-arm, because the two arms do NOT agree at the top of the beta sweep:
    # against the CNN the Laplace advantage is intact at every noise level,
    # against the MLP it lands on the break-even line at the largest t. Quoting
    # only the min over arms (1.00) would read as "survives, just", and quoting
    # only the CNN would hide the tie. Section 9.5 states both.
    f"\\newcommand{{\\nmlapbetaminratio}}{{{min(worst_ratio('laplace', 'beta', beta_max, a) for a in ARMS):.3f}}}",
    f"\\newcommand{{\\nmlapbetaminratiocnn}}{{{worst_ratio('laplace', 'beta', beta_max, 'cnn'):.2f}}}",
    f"\\newcommand{{\\nmbetaminratiocnn}}{{{worst_ratio(fam, 'beta', beta_max, 'cnn'):.2f}}}",
    f"\\newcommand{{\\nmlapemworsterr}}{{{100 * em_stat('laplace', 'beta', beta_max, 'score_rel_l2', max):.0f}}}",
    "",
])
mdest = "../../Markov_Score_Diffusion/thesis/sections/nonmarkov-numbers.tex"
open(mdest, "w").write(macros)

dest = "../../Markov_Score_Diffusion/thesis/sections/tab-nonmarkov.tex"
open(dest, "w").write(tex)
print(f"wrote {mdest}")
print(f"wrote {dest}: {len(families)} famil(y/ies), {len(ARMS)} baselines, "
      f"{sum(len(s) for _, _, s in MECHANISMS)} strengths per arm")
print("  ratio_to_em > 1 means EM-BP wins; bolded cells are where it loses")
