#!/usr/bin/env python3
"""Chapter 9 figure: the two convergence rates, and what the channel costs.

    fig_em_two_rates.pdf -- (a) the geometric decay of each coordinate's
                            distance to its own end-of-run value, whose slopes
                            are the EM contraction factors; (b) the settling
                            times those slopes produce, against the fixed
                            budget several experiments used; (c) the channel's
                            accuracy cost, which is a different axis from rate
                            and does not shrink with data at the same speed.

Reads the frozen research outputs; nothing is refitted here.

    /usr/local/bin/python3.12 tools/fig_ch09.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import BLUE, VERM, GREEN, GREY, FULL, new_figure, save  # noqa: E402

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

BUDGET = 40            # the fixed iteration count several experiments defaulted to
FIT_RHO = (20, 150)    # windows over which the linear-convergence rate is read
FIT_KUR = (60, 400)


def read(rel: str) -> list[dict]:
    with open(FROZEN / rel) as fh:
        return list(csv.DictReader(fh))


def _traces() -> dict[int, dict[int, tuple[float, float]]]:
    by: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    for x in read("exp_27_seed16/shape_trace.csv"):
        by[int(x["seed"])][int(x["update"])] = (
            float(x["rho"]), float(x["innovation_excess_kurtosis"]))
    return by


def _decay(by, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Relative distance to each seed's own end-of-run value; median and band."""
    curves = []
    for s in sorted(by):
        d = by[s]
        u = np.array(sorted(d))
        v = np.array([d[k][idx] for k in u])
        curves.append(np.abs(v - v[-1]) / abs(v[-1]))
    C = np.vstack(curves)
    return (u, np.median(C, axis=0),
            np.quantile(C, 0.1, axis=0), np.quantile(C, 0.9, axis=0))


def rate(by, idx: int, lo: int, hi: int) -> float:
    """Median over seeds of the per-update contraction factor lambda."""
    out = []
    for s in sorted(by):
        d = by[s]
        u = np.array(sorted(d))
        v = np.array([d[k][idx] for k in u])
        e = np.abs(v - v[-1])
        m = (u >= lo) & (u <= hi) & (e > 1e-12)
        if m.sum() >= 5:
            out.append(np.exp(np.polyfit(u[m], np.log(e[m]), 1)[0]))
    return float(np.median(out))


def fig_em_two_rates() -> None:
    by = _traces()
    fig, (axA, axB, axC) = new_figure(1, 3, width=FULL, height=2.95)

    # -- (a) the two contraction rates --------------------------------------
    lam_r = rate(by, 0, *FIT_RHO)
    lam_k = rate(by, 1, *FIT_KUR)
    for idx, col, (lo, hi), lam in ((0, BLUE, FIT_RHO, lam_r),
                                    (1, VERM, FIT_KUR, lam_k)):
        u, med, q10, q90 = _decay(by, idx)
        keep = med > 1e-6
        axA.fill_between(u[keep], q10[keep], q90[keep], color=col, alpha=0.13, lw=0)
        axA.semilogy(u[keep], med[keep], color=col, lw=1.9, zorder=3)
        w = np.linspace(lo, hi, 2)
        i0 = int(np.argmin(np.abs(u - lo)))
        axA.semilogy(w, med[i0] * lam ** (w - lo), ":", color=GREY, lw=1.2,
                     zorder=4)

    axA.axvline(BUDGET, color=GREY, ls=(0, (4, 3)), lw=1.0)
    axA.set_xlim(0, 500)
    axA.set_ylim(1e-6, 3.0)
    axA.set_xlabel("EM update")
    axA.set_ylabel("relative distance to end of run")
    axA.set_title("(a) two contraction rates")
    axA.grid(True, which="both", alpha=0.16)
    # the two curves separate by four decades on the right, so the labels go
    # at the right-hand end where nothing else is
    axA.annotate(rf"$\kappa_4$:  $\lambda={lam_k:.3f}$", xy=(235, 0.22),
                 color=VERM, fontsize=8, ha="left", va="bottom")
    axA.annotate(rf"$\alpha$:  $\lambda={lam_r:.3f}$", xy=(210, 4e-6),
                 color=BLUE, fontsize=8, ha="left", va="bottom")
    axA.annotate(f"budget {BUDGET}", xy=(BUDGET + 12, 1.6), color=GREY,
                 fontsize=7.4, ha="left", va="center")

    # -- (b) the settling times they produce --------------------------------
    st = read("exp_27_seed16/shape_settle.csv")
    keys = [("settle_rho", r"$\alpha$", BLUE),
            ("settle_innovation_var", r"$\sigma_\eta^2$", GREEN),
            ("settle_innovation_excess_kurtosis", r"$\kappa_4$", VERM)]
    rng = np.random.default_rng(0)
    for i, (k, lab, col) in enumerate(keys):
        v = np.array([float(r[k]) for r in st])
        axB.scatter(i + rng.uniform(-0.16, 0.16, v.size), v, s=13, color=col,
                    alpha=0.75, lw=0, zorder=3)
        axB.plot([i - 0.32, i + 0.32], [np.median(v)] * 2, color=col, lw=2.2,
                 zorder=4)
        axB.annotate(f"{np.median(v):.0f}", xy=(i + 0.36, np.median(v)),
                     color=col, fontsize=7.8, ha="left", va="center")

    axB.axhline(BUDGET, color=GREY, ls=(0, (4, 3)), lw=1.0)
    axB.set_yscale("log")
    axB.set_xlim(-0.55, 2.75)
    axB.set_ylim(20, 900)
    axB.set_xticks(range(len(keys)))
    axB.set_xticklabels([lab for _, lab, _ in keys])
    axB.set_ylabel("updates to settle")
    axB.set_title("(b) settling time, 16 seeds")
    axB.grid(True, which="both", axis="y", alpha=0.16)
    axB.annotate(f"fixed budget {BUDGET}", xy=(-0.45, BUDGET * 1.13),
                 color=GREY, fontsize=7.4, ha="left", va="bottom")

    # -- (c) the channel's accuracy cost ------------------------------------
    cv = read("exp_06/clean_vs_noised_shape.csv")
    agg: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for x in cv:
        agg[int(x["n_chains"])][x["arm"]].append(float(x["score_rel_l2_t0.2"]))
    n = np.array(sorted(agg), float)
    clean = np.array([np.mean(agg[int(v)]["clean"]) for v in n])
    noised = np.array([np.mean(agg[int(v)]["noised"]) for v in n])

    axC.loglog(n, noised, "o-", color=VERM, lw=1.9, ms=4.5)
    axC.loglog(n, clean, "s-", color=BLUE, lw=1.9, ms=4.0)
    axC.set_xlim(n[0] * 0.8, n[-1] * 2.4)
    axC.set_ylim(3e-3, 1e-1)
    axC.set_xticks(n)
    axC.set_xticklabels([f"{int(v)}" for v in n])
    axC.minorticks_off()
    axC.set_xlabel(r"training sequences $N$")
    axC.set_ylabel(r"score error at $t=0.2$")
    axC.set_title("(c) what the channel costs")
    axC.grid(True, which="both", alpha=0.16)
    axC.annotate("through the\nchannel", xy=(n[-1], noised[-1]), xytext=(5, 0),
                 textcoords="offset points", color=VERM, fontsize=7.6,
                 ha="left", va="center", annotation_clip=False)
    axC.annotate("clean\npairs", xy=(n[-1], clean[-1]), xytext=(5, 0),
                 textcoords="offset points", color=BLUE, fontsize=7.6,
                 ha="left", va="center", annotation_clip=False)
    axC.annotate(rf"gap ${noised[0]/clean[0]:.1f}\times \to "
                 rf"{noised[-1]/clean[-1]:.1f}\times$",
                 xy=(n[0] * 0.95, 4.2e-3), color=GREY, fontsize=7.6,
                 ha="left", va="center")

    save(fig, FIG / "fig_em_two_rates")

    print(f"  lambda_rho = {lam_r:.4f}   lambda_kurt = {lam_k:.4f}")
    print(f"  log-rate ratio = {np.log(lam_r)/np.log(lam_k):.2f}")
    for k, lab, _ in keys:
        v = np.array([float(r[k]) for r in st])
        print(f"  {lab:>16}: median {np.median(v):.0f}  "
              f"[{v.min():.0f}, {v.max():.0f}]")
    for i, v in enumerate(n):
        print(f"  N={int(v):<5} clean {clean[i]:.4f}  noised {noised[i]:.4f}  "
              f"ratio {noised[i]/clean[i]:.2f}")


if __name__ == "__main__":
    fig_em_two_rates()
