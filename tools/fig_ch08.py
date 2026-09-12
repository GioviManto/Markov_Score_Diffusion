#!/usr/bin/env python3
"""Chapter 8 figures: EM on the chain.

    fig_em_grid.pdf        -- the two numerical error sources, separated:
                              truncation (half-width A) and quadrature
                              (spacing h), plus the resolution gate that
                              decides the smallest usable diffusion time.
                              Replaces fig_grid_domain.pdf, which was drawn
                              in a serif font by the research code.
    fig_em_innovation.pdf  -- what EM actually recovers: the fitted innovation
                              density against the truth at three data budgets,
                              and the asymmetry between recovering the
                              correlation, the variance and the shape.
    fig_em_diagnostics.pdf -- monotone ascent, recovery of alpha from distant
                              initialisations, and the n^{-1/2} rate.

All three read the frozen research outputs; nothing is recomputed here.

    /usr/local/bin/python3.12 tools/fig_ch08.py [name ...]
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (BLUE, VERM, GREEN, PURPLE, GREY, FULL,  # noqa: E402
                      ramp, new_figure, save)

THESIS = Path(__file__).resolve().parents[1]
FIG = THESIS / "figures"

def _find_frozen() -> Path:
    """Locate the frozen experiment outputs in either repository layout.

    The working tree keeps the documents beside the research package; the
    published repository keeps the thesis at the root with the research
    package inside it. A hardcoded absolute path worked in exactly one of
    those and named a home directory, so a clone could not run this script at
    all -- which the code README promises it can.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "research" / "outputs" / "frozen",
        here.parents[3] / "research" / "nongaussian-bp" / "outputs" / "frozen",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise SystemExit(
        "frozen outputs not found; looked in:\n  "
        + "\n  ".join(str(c) for c in candidates))

FROZEN = _find_frozen()


def read(rel: str) -> list[dict]:
    with open(FROZEN / rel) as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------- 1. the two numerical budgets --
def fig_em_grid() -> None:
    """Truncation and quadrature are different errors with different cures.

    The panel that matters is the middle one: halving h divides the interior
    column-mass residual by four, exactly O(h^2), and leaves the edge residual
    untouched. Resolution does not buy domain, which is the whole reason the
    two diagnostics are reported separately rather than as one number.
    """
    b = read("exp_18/boundary.csv")

    fig, (axA, axB, axC) = new_figure(1, 3, width=FULL, height=2.9)

    # -- (a) truncation: boundary mass against half-width -------------------
    # worst case over the 12 noise levels, at the working resolution
    agg: dict[float, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for r in b:
        if int(r["n_grid"]) != 401:
            continue
        v = agg[float(r["half_width"])]
        v[0] = max(v[0], float(r["boundary_mass_max"]))
        v[1] = max(v[1], float(r["boundary_mass_p90"]))
        v[2] = max(v[2], float(r["boundary_mass_median"]))
    A = np.array(sorted(agg))
    mx = np.array([agg[a][0] for a in A])
    p90 = np.array([agg[a][1] for a in A])
    med = np.array([agg[a][2] for a in A])

    axA.fill_between(A, med, mx, color=VERM, alpha=0.13, lw=0)
    axA.semilogy(A, mx, "o-", color=VERM, lw=1.8, ms=4, label="max")
    axA.semilogy(A, p90, "s--", color=GREY, lw=1.2, ms=3.4, label="p90")
    axA.semilogy(A, med, "^-", color=BLUE, lw=1.4, ms=3.8, label="median")
    axA.axvline(8.0, color=GREY, ls=":", lw=1.0)
    axA.set_xlabel(r"half-width $A$")
    axA.set_ylabel("forward boundary mass")
    axA.set_title("(a) truncation")
    axA.set_xticks([4, 6, 8, 10])
    axA.set_ylim(1e-11, 1e-1)
    axA.grid(True, which="both", alpha=0.16)
    axA.legend(loc="lower left", fontsize=7.4, frameon=False,
               handlelength=1.7, labelspacing=0.28, borderaxespad=0.2,
               title="over 256 chains", title_fontsize=7.0)
    axA.annotate("working\n$A=8$", xy=(7.8, 2.0e-2), color=GREY, fontsize=7.2,
                 ha="right", va="top")

    # -- (b) quadrature: column-mass residual against spacing ---------------
    seen, hs, inte, edge = set(), [], [], []
    for r in b:
        if float(r["half_width"]) != 8.0:
            continue
        n = int(r["n_grid"])
        if n in seen:
            continue
        seen.add(n)
        hs.append(float(r["spacing"]))
        inte.append(float(r["kernel_norm_residual_interior"]))
        edge.append(float(r["kernel_norm_residual_max"]))
    o = np.argsort(hs)
    hs, inte, edge = (np.array(v)[o] for v in (hs, inte, edge))

    axB.loglog(hs, inte, "o-", color=BLUE, lw=1.8, ms=4)
    axB.loglog(hs, edge, "s-", color=VERM, lw=1.8, ms=4)
    axB.loglog(hs, inte[-1] * (hs / hs[-1]) ** 2, ":", color=GREY, lw=1.1)
    axB.set_xlabel(r"grid spacing $h$")
    axB.set_ylabel("column-mass residual")
    axB.set_title("(b) quadrature")
    axB.set_ylim(1e-4, 1e-1)
    axB.set_xticks(list(hs))
    axB.set_xticklabels([f"{v:.3g}" for v in hs])
    axB.minorticks_off()
    axB.grid(True, which="both", alpha=0.16)
    axB.annotate("edge column\n(truncation-limited)", xy=(0.021, 2.6e-2),
                 color=VERM, fontsize=7.6, ha="left", va="bottom")
    axB.annotate("interior column", xy=(0.021, 1.7e-4), color=BLUE,
                 fontsize=7.6, ha="left", va="bottom")
    axB.annotate(r"$O(h^2)$", xy=(0.056, 3.0e-4), color=GREY, fontsize=8,
                 ha="left", va="top")

    # -- (c) the resolution gate --------------------------------------------
    m = read("exp_01/grid_sweep_M.csv")
    by: dict[int, dict[float, float]] = defaultdict(dict)
    for r in m:
        if abs(float(r["rho"]) - 0.8) > 1e-9:
            continue
        by[int(r["grid_size"])][float(r["t"])] = float(r["score_rel_error_mean"])
    sizes = [101, 201, 401]
    cols = ramp(len(sizes))
    for M, c in zip(sizes, cols):
        ts = np.array(sorted(by[M]))
        er = np.array([max(by[M][t], 1e-16) for t in ts])
        axC.loglog(ts, er, "o-", color=c, lw=1.8, ms=3.4,
                   label=rf"$N_{{\mathrm{{g}}}}={M}$")
    axC.set_xlabel(r"diffusion time $t$")
    axC.set_ylabel("relative score error")
    axC.set_title("(c) the resolution gate")
    axC.set_ylim(1e-17, 1e-1)
    axC.grid(True, which="both", alpha=0.16)
    axC.legend(loc="upper right", fontsize=7.4, frameon=False,
               handlelength=1.7, labelspacing=0.28, borderaxespad=0.3)
    axC.annotate("machine floor", xy=(0.09, 3.5e-17), color=GREY, fontsize=7.2,
                 ha="center", va="bottom")

    save(fig, FIG / "fig_em_grid")


# ------------------------------------------------ 2. what EM actually learns --
def fig_em_innovation() -> None:
    """The fitted innovation law, and which of its moments EM gets.

    The right panel is the honest one: the autoregressive coefficient is
    recovered to the third decimal at every budget, the variance to a few
    percent, and the fourth cumulant not reliably at all.
    """
    dens = read("exp_18/innovation_density.csv")
    summ = read("exp_18/innovation_summary.csv")

    fig, (axL, axR) = new_figure(1, 2, width=FULL, height=3.0)

    # -- left: fitted innovation density against the truth, C = 8 -----------
    def curve(arm: str, budget: int, C: int):
        pts = [(float(r["e"]), float(r["density"])) for r in dens
               if r["arm"] == arm and int(r["budget"]) == budget
               and int(r["n_components"]) == C]
        pts.sort()
        return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])

    e0, d0 = curve("true", 0, 0)
    axL.semilogy(e0, d0, color="black", lw=2.0, ls=(0, (4, 2.5)), zorder=5)

    budgets = [128, 512, 2048]
    cols = ramp(len(budgets))
    for nb, c in zip(budgets, cols):
        e, d = curve("fitted", nb, 8)
        axL.semilogy(e, d, color=c, lw=1.7, zorder=3)

    axL.set_xlim(-2.6, 2.6)
    axL.set_ylim(1e-4, 6.0)
    axL.set_xlabel(r"innovation $\varepsilon$")
    axL.set_ylabel("density (log scale)")
    axL.set_title(r"recovered innovation law, $C=8$")
    axL.grid(True, which="both", alpha=0.16)
    hL = [axL.plot([], [], color="black", lw=2.0, ls=(0, (4, 2.5)))[0]]
    hL += [axL.plot([], [], color=c, lw=1.7)[0] for c in cols]
    axL.legend(hL, ["truth (Laplace)"] + [rf"$N={n}$" for n in budgets],
               loc="lower center", fontsize=7.0, frameon=False,
               handlelength=1.9, labelspacing=0.3, borderaxespad=0.2, ncol=2)

    # -- right: which moments survive ---------------------------------------
    rows = [r for r in summ if int(r["n_components"]) == 8]
    rows.sort(key=lambda r: int(r["n_chains"]))
    n = np.array([int(r["n_chains"]) for r in rows], float)
    e_rho = np.array([abs(float(r["rho_hat"]) - float(r["rho_true"]))
                      / float(r["rho_true"]) for r in rows])
    e_var = np.array([abs(float(r["innovation_var"])
                          - float(r["innovation_var_true"]))
                      / float(r["innovation_var_true"]) for r in rows])
    e_kur = np.array([abs(float(r["innovation_excess_kurtosis"])
                          - float(r["innovation_excess_kurtosis_true"]))
                      / float(r["innovation_excess_kurtosis_true"])
                      for r in rows])

    axR.loglog(n, e_kur, "o-", color=VERM, lw=1.9, ms=4.5)
    axR.loglog(n, e_var, "s-", color=BLUE, lw=1.9, ms=4.0)
    axR.loglog(n, e_rho, "^-", color=GREEN, lw=1.9, ms=4.5)
    axR.set_xlim(100, 2600)
    axR.set_ylim(1e-4, 1.0)
    axR.set_xticks(n)
    axR.set_xticklabels([f"{int(v)}" for v in n])
    axR.set_xlabel(r"training sequences $N$")
    axR.set_ylabel("relative error of the estimate")
    axR.set_title("which moments are recovered")
    axR.grid(True, which="both", alpha=0.16)
    # curves are well separated in y at the right edge, so label them there
    for lab, arr, c in ((r"kurtosis $\kappa_4$", e_kur, VERM),
                        (r"variance $\sigma_\eta^2$", e_var, BLUE),
                        (r"coefficient $\alpha$", e_rho, GREEN)):
        axR.annotate(lab, xy=(n[-1], arr[-1]), xytext=(-4, -11),
                     textcoords="offset points", color=c, fontsize=7.8,
                     ha="right", va="top")

    save(fig, FIG / "fig_em_innovation")


# ------------------------------------------------ 3. optimisation diagnostics --
def fig_em_diagnostics() -> None:
    trace = read("../exp_18/em_trace.csv")
    rate = read("exp_06_rate16/em_rate.csv")

    fig, ax = new_figure(1, 3, width=FULL, height=2.7)

    by = defaultdict(list)
    for r in trace:
        by[(int(r["n_components"]), int(r["init_id"]),
            float(r["rho_init"]))].append(
            (int(r["iteration"]), float(r["log_evidence"]), float(r["rho_hat"])))
    series = [(r0, p) for (C, _, r0), p in sorted(by.items()) if C == 4]
    cols = ramp(len(series))
    for (r0, pts), c in zip(series, cols):
        pts.sort()
        it, ll, rh = zip(*pts)
        ax[0].plot(it, ll, lw=1.5, color=c, label=rf"$\alpha_0={r0:g}$")
        ax[1].plot(it, rh, lw=1.5, color=c)

    ax[0].set_xlabel("EM iteration")
    ax[0].set_ylabel("marginal log-likelihood")
    ax[0].set_title(r"(a) monotone ascent, $C=4$")

    ax[1].axhline(0.85, color=GREY, ls=(0, (4, 3)), lw=1.1)
    ax[1].annotate(r"truth $0.85$", xy=(0.97, 0.85),
                   xycoords=("axes fraction", "data"), ha="right",
                   va="bottom", fontsize=7.8, color=GREY)
    ax[1].set_xlabel("EM iteration")
    ax[1].set_ylabel(r"$\widehat\alpha$")
    ax[1].set_ylim(-0.55, 1.02)
    ax[1].set_title("(b) recovery from distant starts")

    n = np.array([float(r["n_chains"]) for r in rate])
    rr = np.array([float(r["mixture_rho_rmse"]) for r in rate])
    vr = np.array([float(r["mixture_var_rmse"]) for r in rate])
    ax[2].loglog(n, rr, "s-", color=BLUE, lw=1.7, ms=4)
    ax[2].loglog(n, vr, "o-", color=VERM, lw=1.7, ms=4)
    ax[2].loglog(n, rr[0] * (n[0] / n) ** 0.5, ":", color=GREY, lw=1.2)
    ax[2].set_xlabel(r"training sequences $N$")
    ax[2].set_ylabel("RMSE")
    ax[2].set_title(r"(c) the $N^{-1/2}$ rate")
    ax[2].grid(True, which="both", alpha=0.16)
    # Labels go in fresh margin to the RIGHT of the last point: placed inside
    # the axes they sat between the two curves, and the rising blue line ran
    # straight through the alpha label.
    ax[2].set_xlim(n[0] * 0.82, n[-1] * 2.05)
    for lab, y, c in ((r"$\alpha$", rr[-1], BLUE),
                      (r"$\sigma_\eta^2$", vr[-1], VERM),
                      (r"$N^{-1/2}$", rr[0] * (n[0] / n[-1]) ** 0.5, GREY)):
        ax[2].annotate(lab, xy=(n[-1], y), xytext=(5, 0),
                       textcoords="offset points", color=c, fontsize=8,
                       ha="left", va="center", annotation_clip=False)

    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=len(series),
               frameon=False, fontsize=7.6)
    save(fig, FIG / "fig_em_diagnostics")


ALL = {
    "em_grid": fig_em_grid,
    "em_innovation": fig_em_innovation,
    "em_diagnostics": fig_em_diagnostics,
}

if __name__ == "__main__":
    for nm in (sys.argv[1:] or list(ALL)):
        print(f"building {nm} ...")
        ALL[nm]()
