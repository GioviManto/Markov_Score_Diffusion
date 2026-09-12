"""Generate every figure in the note and the compendium from committed outputs.

No number in any figure is typed in by hand: each panel reads a CSV under
``outputs/`` and fails loudly if that file is missing, so a figure can never
silently disagree with the data it claims to show.

    python3 tools/make_figures.py            # all figures
    python3 tools/make_figures.py fig_capacity

Writes PDF (for LaTeX) and PNG (for quick viewing) side by side.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# This file sits at research/nongaussian-bp/tools/, so:
#   parents[0] tools   parents[1] nongaussian-bp   parents[2] research   parents[3] Diffusion
# It used to live at paper/figures/, which is why these were parents[2] and
# "the script's own directory" -- both wrong from here.
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = Path(__file__).resolve().parents[3] / "Markov_Score_Diffusion" / "paper" / "figures"

sys.path.insert(0, str(ROOT / "experiments"))
from frozen_config import FROZEN  # noqa: E402

# Restrained academic style: no gradients, no decoration, colourblind-safe.
#
# Sans-serif (Arial), matching the thesis's figstyle.py: since September 2026
# the thesis builds its own all-sans figures, and these four paper figures are
# the only other generated graphics in the bundle, so serif here would once
# again split the documents' figure typography in two. Mathtext renders in
# Arial as well (STIX-sans fallback), so labels never mix two families.
# Type 42 embeds the fonts as TrueType rather than subsetting to Type 3,
# which is what print wants.
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial",
        "mathtext.cal": "Arial:italic",
        "mathtext.fallback": "stixsans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        # Sized for PRINT, not for the screen. The thesis text block is
        # 453pt wide and the body is 12pt; a 7in-wide figure placed at
        # \linewidth is scaled down by about a ninth, so every point set here
        # loses that much on the page. Measured across the seventeen figures,
        # the old 8.5pt base put tick labels on the page at 5.3pt against
        # 12pt body text, and the four-panel grid diagnostic at 3.7pt.
        # Nothing below is smaller than 8.5pt, so nothing prints below ~7.5pt.
        "font.size": 9.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        "figure.constrained_layout.use": True,
    }
)

# One colour and marker per estimator, used identically in every figure.
STYLE = {
    "exact": dict(color="#222222", marker="o", ls="-", label="grid-BP reference (true kernel)"),
    "em_bp": dict(color="#0072B2", marker="s", ls="-", label="EM\u2013BP estimator"),
    "cnn": dict(color="#D55E00", marker="^", ls="-", label="local CNN"),
    "mlp": dict(color="#009E73", marker="v", ls="-", label="global MLP"),
    "closure": dict(color="#CC79A7", marker="D", ls="-", label="Gaussian second-order baseline"),
}


def read(path: Path) -> list[dict]:
    full = OUT / path
    if not full.exists():
        raise FileNotFoundError(f"missing committed output: {full}")
    with full.open() as fh:
        return list(csv.DictReader(fh))


# Figures admitted as evidence in the thesis, paper or workshop note. Anything
# this script draws that is NOT in this set goes to a sibling directory instead.
#
# The separation is the point. This generator also draws withdrawn and
# exploratory panels -- fig_pointwise_vs_generative rests on the retired
# 40-iteration fits, and the sampler/cost/families panels were diagnostics that
# never entered an argument. All of them used to be written into the same
# directory the four documents include from, so a command advertised as
# "generate every thesis figure" recreated a withdrawn artifact inside the
# thesis's own include path. LaTeX only draws what it is told to draw, so this
# was never a live defect; it was a defect waiting for someone to \includegraphics
# the wrong basename. Keeping them apart makes that mistake impossible rather
# than merely unlikely.
ADMITTED = {
    "fig_closure_vs_t",
    "fig_sample_efficiency",
    "fig_value_of_structure",
    "fig_toymodel_score",
    "fig_capacity",
    "fig_nonmarkov",
    "fig_screening",
    "fig_grid_domain",
    "fig_em_diagnostics",
    "fig_innovation_density",
}


def save(fig, name: str) -> None:
    out = FIG if name in ADMITTED else FIG / "exploratory"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    where = "" if name in ADMITTED else " (exploratory/)"
    print(f"  wrote {name}.pdf / .png{where}")


def _f(rows, key):
    return [float(r[key]) for r in rows]


# ---------------------------------------------------------------------------
# 1. Gaussian second-order baseline: how wrong is it, and where
# ---------------------------------------------------------------------------

def fig_closure() -> None:
    rows = read(Path("frozen/exp_02/laplace_summary.csv"))
    t = _f(rows, "t")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))

    ax[0].loglog(t, _f(rows, "rel_score_error_median"), **STYLE["closure"])
    ax[0].fill_between(
        t, _f(rows, "rel_score_error_q10"), _f(rows, "rel_score_error_q90"),
        color=STYLE["closure"]["color"], alpha=0.18, lw=0,
    )
    ax[0].set_xlabel("diffusion time $t$")
    ax[0].set_ylabel("relative score error")
    ax[0].set_title("score")

    ax[1].loglog(t, _f(rows, "posterior_mean_mse_median"), **STYLE["closure"])
    ax[1].set_xlabel("diffusion time $t$")
    ax[1].set_ylabel("posterior-mean MSE")
    ax[1].set_title("posterior mean")

    ax[2].semilogx(t, [1.0 - c for c in _f(rows, "cosine_median")], **STYLE["closure"])
    ax[2].set_yscale("log")
    ax[2].set_xlabel("diffusion time $t$")
    ax[2].set_ylabel(r"$1-\cos$ similarity")
    ax[2].set_title("direction")

    # Second panel: the family sweep, to show it is not a Laplace artefact.
    sweep = read(Path("frozen/exp_03/innovation_sweep.csv"))
    fam = defaultdict(list)
    for r in sweep:
        if abs(float(r["rho"]) - 0.85) < 1e-9:
            fam[r["family"]].append((float(r["t"]), float(r["score_rel_median"])))
    save(fig, "fig_closure_vs_t")

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for name, pts in sorted(fam.items()):
        pts.sort()
        ax.loglog(*zip(*pts), marker="o", label=name.replace("_", " "))
    ax.set_xlabel("diffusion time $t$")
    ax.set_ylabel("relative score error")
    ax.set_title("Gaussian baseline, autoregressive coefficient $0.85$")
    ax.legend()
    save(fig, "fig_closure_families")


# ---------------------------------------------------------------------------
# 2. Sample efficiency
# ---------------------------------------------------------------------------

def fig_sample_efficiency() -> None:
    """The same data, protocol and estimand as Table~\\ref{tab:pointwise}.

    The figure reads the certified directories directly, with the seed-first
    aggregation the table uses, so that the two are guaranteed to agree. The
    estimand is score error, matching the table.

    THE SIZE GRID MUST MATCH make_tab_efficiency.py. nseq=8192 is withdrawn
    there (its source tree had exp_07, frozen_config and src/em.py all
    uncommitted, so the run is not recoverable) and is therefore not plotted
    here either. It used to be: the table stopped at 4096 while the curve
    beside it ran to 8192, so the figure displayed, at the far right where it
    carries the most rhetorical weight, the one point the table had removed
    for being unreproducible. A caption asserting the two cannot disagree does
    not make them agree.
    """
    import glob

    # Keep in step with EXTENDED/WITHDRAWN in make_tab_efficiency.py.
    patterns = [
        "frozen/exp_07_certified_seed*/sample_efficiency_val.csv",
        "frozen/exp_07_n4096_seed*/sample_efficiency_val.csv",
    ]
    rows = []
    for pattern in patterns:
        files = sorted(glob.glob(str(OUT / pattern)))
        if not files:
            raise SystemExit(
                f"fig_sample_efficiency: no certified output at {pattern}. "
                "Refusing to fall back to outputs/replicates/, which is a "
                "six-seed pre-certification run and does not match the table."
            )
        for f in files:
            seed = f.split("seed")[1].split("/")[0]
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    r["seed"] = seed
                    rows.append(r)

    # Same resolution gate as the table, or the figure would show cells the
    # table excludes.
    rows = [r for r in rows if "em_resolved" not in r or int(r["em_resolved"])]
    seeds = sorted({r["seed"] for r in rows}, key=int)
    sizes = sorted({int(r["n_chains"]) for r in rows})

    def per_seed(n, key):
        """Average the schedule within a seed, then across seeds: the seed is
        the inferential unit, because the twelve levels share a fitted model."""
        g = [r for r in rows if int(r["n_chains"]) == n]
        v = np.array([
            np.mean([float(r[key]) for r in g if r["seed"] == s]) for s in seeds
        ])
        return v.mean(), v.std(ddof=1) / np.sqrt(v.size)

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    series = {
        # Derived from the frozen component count, not typed: theta is
        # [rho, pi(C), mu(C), s2(C)] with pi simplex-constrained, so 3C are free.
        "em_bp_score_rel_l2": (
            "em_bp", f"EM\u2013BP ({3 * FROZEN.n_components} free parameters)"),
        "net_score_rel_l2_selected": ("mlp", "network (tuned on validation)"),
    }
    for key, (style, label) in series.items():
        m, se = zip(*(per_seed(n, key) for n in sizes))
        st = dict(STYLE[style])
        st["label"] = label
        ax.errorbar(sizes, m, yerr=se, capsize=2, **st)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    # No symbol on the axis. One figure is shared by two documents that render
    # \nseq differently -- N in the thesis, M in the paper -- so hardcoding
    # either one puts the figure at odds with the table beside it in the other.
    ax.set_xlabel("training sequences")
    ax.set_ylabel("relative score error")
    ax.legend()
    save(fig, "fig_sample_efficiency")
    print(f"  fig_sample_efficiency: {len(seeds)} seeds, sizes {sizes}, "
          f"{len(rows)} cells (certified)")


def fig_value_of_structure() -> None:
    """The chapter's principal empirical figure: what supplying the chain buys.

    Two panels, because the headline number alone is misleading. The left panel
    is the gap against a network given no locality prior at all. The right is
    the gap that remains once locality and weight sharing ARE supplied, through
    the window head the screen selected -- the same comparison, against a
    baseline that already has most of what the estimator has.

    Reading them together is the point. The left gap is large and the right one
    is small, and a reader shown only the left would attribute all of it to the
    Markov model. Neither panel identifies a causal decomposition: architecture,
    parameter sharing, capacity and optimisation all change when the baseline is
    swapped. What the pair shows is how much of the headline margin survives
    supplying the two structural biases that are easiest to supply.

    Both panels read the same certified directories as the tables they sit
    beside -- exp_07 for the left, exp_31's merged confirm for the right -- with
    the same seed-first aggregation, so figure and table cannot drift apart.
    """
    import glob

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.7))

    # ---- left: against a network with no locality prior -----------------
    rows = []
    for pattern in ("frozen/exp_07_certified_seed*/sample_efficiency_val.csv",
                    "frozen/exp_07_n4096_seed*/sample_efficiency_val.csv"):
        files = sorted(glob.glob(str(OUT / pattern)))
        if not files:
            raise SystemExit(f"fig_value_of_structure: no certified output at {pattern}")
        for f in files:
            seed = f.split("seed")[1].split("/")[0]
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    r["seed"] = seed
                    rows.append(r)
    rows = [r for r in rows if "em_resolved" not in r or int(r["em_resolved"])]
    seeds = sorted({r["seed"] for r in rows}, key=int)
    sizes = sorted({int(r["n_chains"]) for r in rows})

    def per_seed(n, key):
        g = [r for r in rows if int(r["n_chains"]) == n]
        v = np.array([np.mean([float(r[key]) for r in g if r["seed"] == s])
                      for s in seeds])
        return v.mean(), v.std(ddof=1) / np.sqrt(v.size)

    for key, style, label in (
            ("em_bp_score_rel_l2", "em_bp", "EM\u2013BP"),
            ("net_score_rel_l2_selected", "mlp", "network, no locality prior")):
        m, se = zip(*(per_seed(n, key) for n in sizes))
        st = dict(STYLE[style]); st["label"] = label
        ax[0].errorbar(sizes, m, yerr=se, capsize=2, **st)
    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
    ax[0].set_xlabel("training sequences")
    ax[0].set_ylabel("relative score error")
    ax[0].set_title("against an unstructured network")
    ax[0].legend()

    # ---- right: against the screened, locality-respecting baseline -------
    conf = read(Path("frozen/exp_31_confirm_merged/confirm.csv"))
    cseeds = sorted({r["seed"] for r in conf}, key=int)
    csizes = sorted({int(r["n_chains"]) for r in conf})

    def risks(n, method, region="all"):
        out = []
        for s in cseeds:
            v = [float(r["risk"]) for r in conf if r["seed"] == s
                 and int(r["n_chains"]) == n and r["method"] == method
                 and r["region"] == region]
            assert len(v) == 1, (s, n, method, v)
            out.append(v[0])
        return np.array(out)

    m, se = [], []
    for n in csizes:                      # paired per seed: both arms share a fit
        v = risks(n, "window") / risks(n, "em_bp")
        m.append(v.mean()); se.append(v.std(ddof=1) / np.sqrt(v.size))
    st = dict(STYLE["cnn"]); st["label"] = "window head / EM\u2013BP"
    ax[1].errorbar(csizes, m, yerr=se, capsize=2, **st)
    ax[1].axhline(1.0, color="#222222", lw=1.0, ls="--", zorder=1)
    ax[1].annotate("equal error", xy=(csizes[0], 1.0), xytext=(2, 4),
                   textcoords="offset points", fontsize=8.5, color="#222222")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("training sequences")
    ax[1].set_ylabel("error ratio")
    ax[1].set_title("against a locality-respecting baseline")
    ax[1].legend()

    save(fig, "fig_value_of_structure")
    print(f"  fig_value_of_structure: left {len(seeds)} seeds sizes {sizes}; "
          f"right {len(cseeds)} seeds sizes {csizes} (certified)")


# ---------------------------------------------------------------------------
# 3. Capacity: both axes at once
# ---------------------------------------------------------------------------

def _capacity_tables():
    point, gen = {}, {}
    for c in (2, 4, 8, 12, 16):
        rows = read(Path(f"exp_16/cpoint_C{c}/pointwise.csv"))
        by = defaultdict(list)
        for r in rows:
            by[r["arm"]].append(float(r["mse_vs_bayes_denoiser"]))
        point[c] = {k: median(v) for k, v in by.items()}
        rows = read(Path(f"exp_16/components_C{c}/generation.csv"))
        by = defaultdict(list)
        for r in rows:
            by[r["arm"]].append(float(r["innov_kurtosis"]))
        gen[c] = {k: median(v) for k, v in by.items()}
    return point, gen


def _stamp_unconverged(fig) -> None:
    """Mark a figure whose fits used the withdrawn fixed 40-iteration budget.

    Every panel drawn from `outputs/exp_16/` rests on fits that stopped at 40 EM
    iterations, far short of where the shape coordinate settles: the dedicated
    convergence sweep puts its median at 229 updates, with a tail past 600. The
    claim audit withdraws the capacity attribution on exactly that basis. (The
    "2,000 iterations / 93% unsettled" figures this docstring used to quote came
    from a superseded reading of a single run and are not the thesis's numbers.)

    The thesis says so in a remark next to the figure, which is necessary and
    not sufficient: a figure that leaves the document -- into a slide, a poster,
    a supervisor's email -- leaves the caveat behind and reads as evidence for
    a claim that has been withdrawn. Being generated from committed outputs
    guarantees the number matches its CSV; it guarantees nothing about the
    estimand still being one anybody stands behind. So the caveat travels on the
    canvas.
    """
    fig.text(
        0.5, -0.06,
        "Fixed 40-iteration EM budget: shape-dependent curves are confounded "
        "with convergence rate (claim audit §corr-capacity).",
        ha="center", va="top", fontsize=8.5, style="italic", color="#a03020",
    )


# ---------------------------------------------------------------------------
# Capacity, non-Markov robustness and the architecture screen.
#
# These three read the frozen outputs behind Chapter 9's Sections 9.2, 9.3 and
# 9.5, which carried tables and no picture. Each shows the one thing its table
# makes the reader assemble: that no capacity clears zero, that one
# contamination crosses the break-even line and the other does not, and that the
# screen's winner is also its smallest arm.
# ---------------------------------------------------------------------------

CAPACITY_SRC = Path("frozen/exp_32_capacity_merged/capacity_equivalence.csv")
NONMARKOV_SRC = (Path("frozen/exp_21_clean")
                 if (OUT / "frozen" / "exp_21_clean").exists()
                 else Path("exp_21_frozen"))
SCREEN_SRC = Path("frozen/exp_31_screen/screening.csv")
RESOLUTION_FLOOR = 2.0
SELECTION_REGION = "bulk"

# One colour and marker per architecture, as STYLE does for estimators.
ARCH_STYLE = {
    "window": dict(color="#0072B2", marker="s", label="weight-shared window head"),
    "conv": dict(color="#D55E00", marker="^", label="dilated convolutional stack"),
    "bimp": dict(color="#009E73", marker="v", label="bidirectional message passing"),
}


# ---------------------------------------------------------------------------
# The K=2 toy model: what the coupling looks like, and what a per-frame view
# throws away.
#
# WHY THIS IS HERE. Figure 4.1 was the one figure in the thesis with no
# surviving plotting code -- the script that produced it lived in the old
# thesis/ tree, removed during the restructuring of 18 August 2026 and never
# committed. Appendix C disclosed that, which is better than naming a script
# that does not exist, but a figure nobody can regenerate is a standing
# reproducibility hole in a document whose whole argument is that its numbers
# are traceable. This recreates it from the closed forms its caption states, so
# the exception can be removed rather than explained.
#
# Everything is analytic. A Gaussian-mixture prior on x0 with x1 = x0 + c + eta
# makes the clean pair a Gaussian mixture; the variance-preserving channel adds
# a multiple of the identity to every component covariance and scales every
# mean, so the noised pair is a Gaussian mixture too, and its score is the
# responsibility-weighted average of the component scores.
# ---------------------------------------------------------------------------

TOY_MEANS = (-2.0, 0.0, 2.0)   # trimodal prior on x0
TOY_PRIOR_SD = 0.42
TOY_DRIFT = 1.0                # c
TOY_INNOV_SD = 0.45            # sigma
TOY_T = 0.15                   # diffusion time


def fig_toymodel_score() -> None:
    a = float(np.exp(-TOY_T))                 # e^{-t}
    d = float(1.0 - np.exp(-2.0 * TOY_T))     # 1 - e^{-2t}

    # Clean pair per component: x0 ~ N(m, s^2), x1 = x0 + c + eta.
    s2, g2 = TOY_PRIOR_SD ** 2, TOY_INNOV_SD ** 2
    comps = []
    for m in TOY_MEANS:
        mu = np.array([m, m + TOY_DRIFT])
        cov = np.array([[s2, s2], [s2, s2 + g2]])
        # Variance-preserving channel, coordinatewise: x = a*clean + sqrt(d)*z.
        comps.append((a * mu, a * a * cov + d * np.eye(2)))

    lim, n = 4.4, 240
    gx = np.linspace(-lim, lim, n)
    X0, X1 = np.meshgrid(gx, gx)
    pts = np.stack([X0.ravel(), X1.ravel()], axis=1)

    dens = np.zeros(pts.shape[0])
    grad = np.zeros_like(pts)
    for mu, cov in comps:
        P = np.linalg.inv(cov)
        z = pts - mu
        q = np.einsum("ni,ij,nj->n", z, P, z)
        w = np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
        dens += w / len(comps)
        # Each component contributes -P z, weighted by its responsibility; the
        # mixture score is the weighted average, which is what makes it
        # nonlinear.
        grad += (w / len(comps))[:, None] * (-(z @ P))
    score = grad / np.maximum(dens, 1e-300)[:, None]

    D = dens.reshape(n, n)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.2))

    ax[0].contour(X0, X1, D, levels=9, colors="#222222", linewidths=0.7)
    ax[0].set_title(r"noised joint density $p_t(x_0, x_1)$")

    ax[1].contour(X0, X1, D, levels=9, colors="#BBBBBB", linewidths=0.6)
    step = 14
    sel = (slice(None, None, step), slice(None, None, step))
    U = score[:, 0].reshape(n, n)[sel]
    V = score[:, 1].reshape(n, n)[sel]
    mask = D[sel] > D.max() * 0.02      # no arrows where there is no density
    ax[1].quiver(X0[sel][mask], X1[sel][mask], U[mask], V[mask],
                 color="#222222", width=0.004, scale=90.0)
    ax[1].set_title(r"joint score field $\nabla \log p_t$")

    for k in (0, 1):
        ax[k].set_xlabel(r"$x_0$")
        ax[k].set_ylabel(r"$x_1$")
        ax[k].set_xlim(-lim, lim)
        ax[k].set_ylim(-lim, lim)
        ax[k].set_aspect("equal")
        ax[k].grid(False)
    save(fig, "fig_toymodel_score")


def fig_capacity() -> None:
    """Held-out evidence against a single Gaussian innovation, by capacity.

    Rebuilt on exp_32. The previous version drew exp_16, whose six seeds at a
    fixed forty-iteration budget reported the opposite sign and are withdrawn;
    a figure of withdrawn numbers is worse than no figure, which is why the
    chapter carried none in between.
    """
    rows = read(CAPACITY_SRC)
    sizes = sorted({int(r["n_chains"]) for r in rows})
    comps = sorted({int(r["n_components"]) for r in rows})
    others = [c for c in comps if c != 1]

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for k, n in enumerate(sizes):
        base = {int(r["seed"]): float(r["test_log_evidence_per_edge"])
                for r in rows if int(r["n_chains"]) == n
                and int(r["n_components"]) == 1}
        mean, err = [], []
        for c in others:
            here = {int(r["seed"]): float(r["test_log_evidence_per_edge"])
                    for r in rows if int(r["n_chains"]) == n
                    and int(r["n_components"]) == c}
            d = np.array([here[s] - base[s] for s in sorted(set(base) & set(here))])
            mean.append(d.mean())
            err.append(d.std(ddof=1) / np.sqrt(d.size))
        ax[0].errorbar(others, mean, yerr=err, capsize=2.5,
                       color=("#0072B2", "#CC79A7")[k % 2],
                       marker=("s", "D")[k % 2], ls="-",
                       label=rf"$N = {n}$")
    ax[0].axhline(0.0, color="#222222", ls="--", lw=1.0)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks(others)
    ax[0].set_xticklabels([str(c) for c in others])
    ax[0].set_xlabel(r"mixture components $C$")
    ax[0].set_ylabel("held-out log-evidence per edge,\npaired difference from $C=1$")
    ax[0].set_title("held-out evidence relative to $C=1$")
    ax[0].legend()

    share = [100.0 * sum(1 for r in rows if int(r["n_components"]) == c
                         and float(r["s_min_over_h"]) < RESOLUTION_FLOOR)
             / sum(1 for r in rows if int(r["n_components"]) == c)
             for c in comps]
    ax[1].plot(comps, share, color="#D55E00", marker="o", ls="-")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xticks(comps)
    ax[1].set_xticklabels([str(c) for c in comps])
    ax[1].set_xlabel(r"mixture components $C$")
    ax[1].set_ylabel("cells below the grid's resolution floor (%)")
    ax[1].set_title("convergence diagnostics by capacity")
    save(fig, "fig_capacity")


def fig_nonmarkov() -> None:
    """Where each violation of the Markov assumption stops paying."""
    import glob as _glob
    import re as _re

    panels = (("beta", r"global latent, rank-one strength $\beta$"),
              ("gamma", r"long-range coupling strength $\gamma$"))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    # Both innovation laws, so the figure shows the same cells the table does.
    # The Laplace arm exists for the rank-one mechanism only -- there is no
    # closed-form reference for a long-range-coupled Laplace chain -- so it
    # appears in the left panel and is simply absent from the right, rather
    # than being drawn as a gap.
    for k, (mech, xlabel) in enumerate(panels):
        for family, dash in (("gauss", "-"), ("laplace", "--")):
            cells = {}
            for d in sorted(_glob.glob(str(OUT / NONMARKOV_SRC / f"{family}_{mech}*"))):
                m = _re.search(rf"{mech}([0-9.]+)$", d)
                f = Path(d) / f"nonmarkov_{family}.csv"
                if not m or not f.exists():
                    continue
                cells[float(m.group(1))] = list(csv.DictReader(f.open()))
            if not cells:
                if family == "gauss":
                    raise FileNotFoundError(
                        f"missing committed output: {OUT / NONMARKOV_SRC}")
                continue
            xs = sorted(cells)
            for arm in ("cnn", "mlp"):
                ys = [float(np.mean([float(r["ratio_to_em"]) for r in cells[x]
                                     if r["arm"] == arm])) for x in xs]
                st = dict(STYLE[arm], ls=dash)
                st["label"] = (st["label"] + ", Laplace") if family == "laplace" \
                    else (st["label"] + ", Gaussian")
                if family == "laplace":
                    st["markerfacecolor"] = "white"
                ax[k].plot(xs, ys, **st)
        ax[k].axhline(1.0, color="#222222", ls="--", lw=1.0)
        ax[k].set_yscale("log")
        ax[k].set_xlabel(xlabel)
        ax[k].set_title(("rank-one global latent",
                         "long-range precision coupling")[k])
    ax[0].set_ylabel("baseline error / EM\u2013BP error")
    ax[0].legend()
    save(fig, "fig_nonmarkov")


def fig_screening() -> None:
    """Risk against parameter count for the three screened architectures.

    Scored by POOLING sq_err and sq_ref over development seeds and sizes
    before the square root -- the accumulation the confirmatory selector
    uses. An earlier revision averaged per-cell risks, which promotes a
    different window radius than the one the confirmatory stage ran."""
    rows = [r for r in read(SCREEN_SRC) if r["region"] == SELECTION_REGION]
    fig, ax = plt.subplots(figsize=(3.9, 2.9))
    for arch, st in ARCH_STYLE.items():
        by_hp = defaultdict(lambda: [0.0, 0.0, []])
        for r in rows:
            if r["arch"] == arch:
                acc = by_hp[r["hp"]]
                acc[0] += float(r["sq_err"])
                acc[1] += float(r["sq_ref"])
                acc[2].append(float(r["n_params"]))
        if not by_hp:
            continue
        pts = sorted((np.mean(ps), np.sqrt(e / d))
                     for e, d, ps in by_hp.values())
        ax.plot([p for p, _ in pts], [q for _, q in pts], ls="none",
                marker=st["marker"], color=st["color"], label=st["label"],
                alpha=0.85)
        best = min(pts, key=lambda z: z[1])
        ax.plot(*best, marker=st["marker"], color=st["color"], ms=9,
                markeredgecolor="#222222", markeredgewidth=0.8, ls="none")
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters")
    ax.set_ylabel(f"pooled risk on the {SELECTION_REGION} region")
    ax.set_title("validation risk against parameter count")
    ax.legend(loc="upper left")
    save(fig, "fig_screening")


def fig_pointwise_vs_generative() -> None:
    """Each configuration is one point; disagreeing rankings are visible as
    points that are better on one axis and worse on the other."""
    point, gen = _capacity_tables()
    target = 1.9098
    fig, ax = plt.subplots(figsize=(3.6, 2.9))

    cs = sorted(point)
    xs = [point[c]["em_bp"] for c in cs]
    ys = [abs(gen[c]["em_bp"] - target) for c in cs]
    ax.plot(xs, ys, color=STYLE["em_bp"]["color"], marker="s", ls="-",
            label=r"EM$-$BP, $C=2\ldots16$")
    for c, x, y in zip(cs, xs, ys):
        ax.annotate(f"$C={c}$", (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=8.5)
    for arm in ("cnn", "mlp"):
        if arm not in point[cs[0]]:
            continue
        st = dict(STYLE[arm])
        st.pop("ls")
        ax.plot([point[cs[0]][arm]], [abs(gen[cs[0]][arm] - target)], ls="none", **st)
    ax.set_xscale("log")
    ax.set_xlabel("MSE against the Bayes denoiser (lower better)")
    ax.set_ylabel("generated kurtosis deficit (lower better)")
    ax.legend()
    _stamp_unconverged(fig)
    save(fig, "fig_pointwise_vs_generative")


# ---------------------------------------------------------------------------
# 4. Grid and domain convergence
# ---------------------------------------------------------------------------

def fig_grid_domain() -> None:
    heat = read(Path("frozen/exp_01/grid_heatmap.csv"))
    bnd = read(Path("frozen/exp_18/boundary.csv"))
    fig, ax = plt.subplots(2, 2, figsize=(6.4, 4.4))
    ax = ax.ravel()

    # (a) fixed A, increasing N_g -- resolution alone.
    sizes = sorted({int(r["grid_size"]) for r in heat})
    for t in sorted({r["t"] for r in heat}, key=float):
        pts = sorted(
            (int(r["grid_size"]), float(r["score_rel_error_mean"]))
            for r in heat
            if r["t"] == t and abs(float(r["half_width"]) - 8.0) < 1e-9
        )
        if pts:
            ax[0].loglog(*zip(*pts), marker="o", label=f"$t={t}$")
    ax[0].set_xticks(sizes)
    ax[0].set_xticklabels([str(s_) for s_ in sizes])
    ax[0].minorticks_off()
    ax[0].set_xlabel(r"grid points $N_{\mathrm{g}}$   (fixed $A=8$)")
    ax[0].set_ylabel("relative score error")
    ax[0].set_title("(a) resolution")
    ax[0].legend()

    # (b) fixed N_g, increasing A -- domain alone.
    for t in sorted({r["t"] for r in heat}, key=float):
        pts = sorted(
            (float(r["half_width"]), float(r["score_rel_error_mean"]))
            for r in heat
            if r["t"] == t and int(r["grid_size"]) == 401
        )
        if pts:
            ax[1].semilogy(*zip(*pts), marker="o", label=f"$t={t}$")
    ax[1].set_xlabel(r"half-width $A$   (fixed $N_{\mathrm{g}}=401$)")
    ax[1].set_ylabel("relative score error")
    ax[1].set_title("(b) domain")
    ax[1].legend()

    # (c) truncation: mass approaching the grid edge, over a sample of chains.
    #
    # This panel used to plot `worst_boundary_mass`, a single sampled trajectory's worst site,
    # and label it a bound. It was neither: not a bound, and not a sample. The diagnostic now
    # runs 256 chains per cell, so the maximum and the 90th percentile are both available and
    # both plotted -- the gap between them is the point, since the old single-draw number sat
    # near the *median* while the maximum is some two orders of magnitude higher.
    by_a_max = defaultdict(list)
    by_a_p90 = defaultdict(list)
    for r in bnd:
        by_a_max[float(r["half_width"])].append(float(r["boundary_mass_max"]))
        by_a_p90[float(r["half_width"])].append(float(r["boundary_mass_p90"]))
    a_vals = sorted(by_a_max)
    ax[2].semilogy(a_vals, [max(by_a_max[a]) for a in a_vals],
                   color="#222222", marker="o", label="max over chains")
    ax[2].semilogy(a_vals, [max(by_a_p90[a]) for a in a_vals],
                   color="#888888", marker="s", ls="--", label="90th percentile")
    ax[2].set_xlabel(r"half-width $A$")
    ax[2].set_ylabel("edge-cell mass")
    ax[2].set_title("(c) truncation")
    ax[2].legend(fontsize=8.5)

    # (d) quadrature: interior column-normalisation residual against spacing.
    by_h = defaultdict(list)
    for r in bnd:
        if abs(float(r["half_width"]) - 8.0) < 1e-9:
            by_h[float(r["spacing"])].append(float(r["kernel_norm_residual_interior"]))
    hs = sorted(by_h)
    vals = [max(by_h[h]) for h in hs]
    ax[3].loglog(hs, vals, color="#0072B2", marker="s", label=r"interior residual, $A=8$")
    ref = [vals[-1] * (h / hs[-1]) ** 2 for h in hs]
    ax[3].loglog(hs, ref, ls=":", color="#222222", label=r"$O(h^2)$")
    ax[3].set_xlabel(r"grid spacing $h$")
    ax[3].set_ylabel("column-mass residual")
    ax[3].set_title("(d) quadrature")
    ax[3].legend()
    save(fig, "fig_grid_domain")


# ---------------------------------------------------------------------------
# 5. Reverse-sampler convergence
# ---------------------------------------------------------------------------

def fig_sampler_steps() -> None:
    rows = []
    for d in sorted((OUT / "exp_16").glob("calibrate_steps*")):
        f = d / "steps.csv"
        if f.exists():
            rows += list(csv.DictReader(f.open()))
    if not rows:
        raise FileNotFoundError("no calibrate_steps*/steps.csv found")
    by = defaultdict(list)
    for r in rows:
        by[r["arm"]].append(
            (int(r["n_steps"]), float(r["innov_kurtosis"]),
             float(r["innov_kurtosis_se"]), float(r["innov_kl"]))
        )
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))
    for arm, pts in by.items():
        pts.sort()
        n, k, se, kl = zip(*pts)
        st = dict(STYLE.get(arm, {"color": "grey", "marker": "o", "ls": "-"}))
        st.setdefault("label", arm)
        ax[0].errorbar(n, k, yerr=se, capsize=2, **st)
        st2 = dict(st)
        st2.pop("label")
        ax[1].plot(n, kl, **st2)
    ax[0].axhline(1.9098, color="#222222", ls="--", lw=1.0, label="closed-form target")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("reverse integration steps")
    ax[0].set_ylabel("generated innovation excess kurtosis")
    ax[0].legend()
    ax[1].set_xscale("log", base=2)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("reverse integration steps")
    ax[1].set_ylabel(r"$D_{\mathrm{KL}}$(true $\|$ generated)")
    save(fig, "fig_sampler_steps")


# ---------------------------------------------------------------------------
# 6. Generated-law diagnostics across innovation families
# ---------------------------------------------------------------------------

def fig_families() -> None:
    fams = ["gaussian", "mixture", "uniform", "laplace", "student"]
    data = {}
    for f in fams:
        d = OUT / "exp_16" / f"family_{f}" / "generation.csv"
        if not d.exists():
            continue
        by = defaultdict(list)
        for r in csv.DictReader(d.open()):
            by[r["arm"]].append((float(r["innov_kurtosis"]), float(r["cov_worst_lag_abs"])))
        data[f] = {k: (median([a for a, _ in v]), median([b for _, b in v]))
                   for k, v in by.items()}
    present = [f for f in fams if f in data]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    width, arms = 0.26, ["em_bp", "cnn", "mlp"]
    idx = np.arange(len(present))
    for j, arm in enumerate(arms):
        gaps = [abs(data[f][arm][0] - data[f]["exact"][0]) for f in present]
        ax[0].bar(idx + (j - 1) * width, gaps, width,
                  color=STYLE[arm]["color"], label=STYLE[arm]["label"])
        ax[1].bar(idx + (j - 1) * width, [data[f][arm][1] for f in present], width,
                  color=STYLE[arm]["color"])
    ax[0].set_xticks(idx, present, rotation=20)
    ax[0].set_ylabel("kurtosis gap to reference")
    ax[0].set_title("innovation shape")
    ax[0].legend()
    ax[1].set_xticks(idx, present, rotation=20)
    ax[1].set_ylabel("worst covariance-lag error")
    ax[1].set_title("second-order structure")
    save(fig, "fig_families")


# ---------------------------------------------------------------------------
# 7. Cost
# ---------------------------------------------------------------------------

def fig_cost() -> None:
    inf = read(Path("exp_07_em_vs_score_network/inference_cost.csv"))
    tr = read(Path("exp_07_em_vs_score_network/training_cost.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5))

    b = _f(inf, "batch")
    ax[0].loglog(b, _f(inf, "bp_per_chain_ms"), **STYLE["em_bp"])
    st = dict(STYLE["mlp"])
    st["label"] = "network"
    ax[0].loglog(b, _f(inf, "net_per_chain_ms"), **st)
    ax[0].set_xlabel("batch size")
    ax[0].set_ylabel("inference, ms per sequence")
    ax[0].set_title("inference cost")
    ax[0].legend()

    n = _f(tr, "n_chains")
    ax[1].loglog(n, _f(tr, "em_seconds"), **STYLE["em_bp"])
    st = dict(STYLE["mlp"])
    st["label"] = "network (fixed step budget)"
    ax[1].loglog(n, [a + b_ for a, b_ in zip(_f(tr, "net_eps_seconds"),
                                             _f(tr, "net_x0_seconds"))], **st)
    # No symbol on this axis. The figure is shared, and the two documents give
    # the training-set size different letters -- the paper writes M, the thesis
    # writes N, because the thesis needs M for grid points (Remark "Symbols
    # that are easy to swap"). A baked-in "$M$" is therefore correct in one
    # document and a direct contradiction in the other. The companion axis in
    # fig_sample_efficiency already spells the word out, so this also makes the
    # two training-size axes agree.
    ax[1].set_xlabel("training sequences")
    ax[1].set_ylabel("training wall-clock, s")
    ax[1].set_title("training cost")
    ax[1].legend()
    save(fig, "fig_cost")


# ---------------------------------------------------------------------------
# 8 & 9. Optimisation diagnostics and the learned innovation law
# ---------------------------------------------------------------------------

def fig_em_diagnostics() -> None:
    trace = read(Path("exp_18/em_trace.csv"))
    # The frozen rho=0.85 rate sweep, not outputs/exp_06_em_parameter_recovery/,
    # which is a pre-migration run at rho=0.8. That directory was still being
    # read here after the project moved rho into frozen_config.py, which put a
    # 0.8 panel under a caption whose centre panel draws its true-value line at
    # 0.85. Same columns, same sizes, sixteen seeds, and it passes the gate.
    rate = read(Path("frozen/exp_06_rate16/em_rate.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))

    by = defaultdict(list)
    for r in trace:
        by[(int(r["n_components"]), int(r["init_id"]), float(r["rho_init"]))].append(
            (int(r["iteration"]), float(r["log_evidence"]), float(r["rho_hat"]))
        )
    for (c, _, rho0), pts in sorted(by.items()):
        if c != 4:
            continue
        pts.sort()
        it, ll, rh = zip(*pts)
        ax[0].plot(it, ll, lw=1.0, label=rf"init.\\ ${rho0}$")
        ax[1].plot(it, rh, lw=1.0)
    ax[0].set_xlabel("EM iteration")
    ax[0].set_ylabel("marginal log-likelihood")
    ax[0].set_title(r"objective, $C=4$")
    ax[0].legend()
    ax[1].axhline(0.85, color="#222222", ls="--", lw=1.0, label="true value")
    ax[1].set_xlabel("EM iteration")
    ax[1].set_ylabel("estimate")
    ax[1].set_title("autoregressive coefficient")
    ax[1].legend()

    n = _f(rate, "n_chains")
    ax[2].loglog(n, _f(rate, "mixture_rho_rmse"), marker="s", color="#0072B2",
                 label="autoregressive coefficient")
    ax[2].loglog(n, _f(rate, "mixture_var_rmse"), marker="o", color="#D55E00",
                 label="innovation variance")
    ref = [_f(rate, "mixture_rho_rmse")[0] * (n[0] / v) ** 0.5 for v in n]
    ax[2].loglog(n, ref, ls=":", color="#222222", label="slope $-1/2$")
    ax[2].set_xlabel("training sequences")
    ax[2].set_ylabel("RMSE")
    ax[2].set_title("parameter recovery")
    ax[2].legend()
    save(fig, "fig_em_diagnostics")


def fig_innovation_density() -> None:
    rows = read(Path("exp_18/innovation_density.csv"))
    true = [(float(r["e"]), float(r["density"])) for r in rows if r["arm"] == "true"]
    true.sort()
    budgets = sorted({int(r["budget"]) for r in rows if r["arm"] == "fitted"})
    fig, ax = plt.subplots(1, len(budgets), figsize=(2.4 * len(budgets), 2.4),
                           sharey=True)
    ax = np.atleast_1d(ax)
    for j, bud in enumerate(budgets):
        ax[j].plot(*zip(*true), color="#222222", lw=1.4, label="true Laplace")
        for comp, colour in ((4, "#0072B2"), (8, "#D55E00")):
            pts = sorted(
                (float(r["e"]), float(r["density"]))
                for r in rows
                if r["arm"] == "fitted" and int(r["budget"]) == bud
                and int(r["n_components"]) == comp
            )
            if pts:
                ax[j].plot(*zip(*pts), color=colour, lw=1.1, ls="--",
                           label=rf"fitted, $C={comp}$")
        ax[j].set_yscale("log")
        ax[j].set_ylim(1e-4, 3)
        ax[j].set_xlabel(r"innovation $\varepsilon$")
        ax[j].set_title(rf"{bud} EM updates")
    ax[0].set_ylabel("density")
    ax[0].legend()
    save(fig, "fig_innovation_density")


# ---------------------------------------------------------------------------
# Rung 4: recovering a rotation that no single-frame marginal can see
# ---------------------------------------------------------------------------

def fig_ring() -> None:
    """Three panels: recovered p(psi) on the joint, the same on the blind
    marginal arm, and recovery error against the noise level.

    The point of the middle panel is that it is empty of structure by theorem,
    not by underperformance -- so it is drawn on the same y-axis as the left
    panel, which is the only way the comparison is honest.
    """
    dens = read(Path("frozen/exp_28_ring_em/ring_density.csv"))

    # Two panels, not three. The recovery-error-against-noise panel that used to
    # sit on the right is now a table in the text, and a figure that repeats a
    # table costs a page without adding anything.
    fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.3))

    # Pick the lowest noise level present, for the two density panels.
    t_show = min(float(r["t"]) for r in dens)
    # The y-limit must cover the truth spikes (0.5 each) as well as the
    # estimate, or the truth is drawn clipped and looks like a plotting error.
    ymax = max(
        max(float(r["p_hat"]) for r in dens if float(r["t"]) == t_show),
        max(float(r["p_true"]) for r in dens if float(r["t"]) == t_show),
    )

    for k, (arm, title) in enumerate(
        [("joint", "joint likelihood"), ("marginal", "per-frame marginals")]
    ):
        rows = [r for r in dens if r["arm"] == arm and float(r["t"]) == t_show]
        rows.sort(key=lambda r: float(r["psi"]))
        psi = np.degrees(_f(rows, "psi"))
        ax[k].plot(psi, _f(rows, "p_hat"), color="#0072B2", lw=1.3,
                   label=r"recovered $\hat p(\psi)$")
        # The truth is two point masses; draw them as stems so they cannot be
        # confused with a density.
        for p_t, val in zip(psi, _f(rows, "p_true")):
            if val > 0:
                ax[k].vlines(p_t, 0, val, color="#222222", lw=1.6, zorder=3)
        ax[k].plot([], [], color="#222222", lw=1.6, label=r"truth $p^*(\psi)$")
        ax[k].axhline(1.0 / len(rows), color="#999999", ls=":", lw=1.0,
                      label="uniform (null)")
        ax[k].set_title(title)
        ax[k].set_xlabel(r"rotation angle $\psi$ (degrees)")
        ax[k].set_xlim(-180, 180)
        ax[k].set_xticks([-180, -90, 0, 90, 180])
        ax[k].set_ylim(0, ymax * 1.15)
    ax[0].set_ylabel(r"$p(\psi)$")
    # One legend for both density panels, placed outside so it never covers
    # the spikes it is describing.
    ax[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=8.5)

    save(fig, "fig_ring")


FIGURES = {
    "fig_closure": fig_closure,
    "fig_ring": fig_ring,
    "fig_sample_efficiency": fig_sample_efficiency,
    "fig_value_of_structure": fig_value_of_structure,
    "fig_toymodel_score": fig_toymodel_score,
    "fig_capacity": fig_capacity,
    "fig_nonmarkov": fig_nonmarkov,
    "fig_screening": fig_screening,
    "fig_pointwise_vs_generative": fig_pointwise_vs_generative,
    "fig_grid_domain": fig_grid_domain,
    "fig_sampler_steps": fig_sampler_steps,
    "fig_families": fig_families,
    "fig_cost": fig_cost,
    "fig_em_diagnostics": fig_em_diagnostics,
    "fig_innovation_density": fig_innovation_density,
}


def main() -> None:
    wanted = sys.argv[1:] or list(FIGURES)
    failed = []
    for name in wanted:
        print(f"{name}:")
        try:
            FIGURES[name]()
        except FileNotFoundError as exc:
            print(f"  SKIPPED -- {exc}")
            failed.append(name)
    if failed:
        print(f"\n{len(failed)} figure(s) skipped for missing data: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
