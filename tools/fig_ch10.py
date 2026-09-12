#!/usr/bin/env python3
"""Chapter 10 figures: what structure buys, and where it stops paying.

    fig_value_of_structure.pdf  the two comparisons and the two ratios
    fig_screening.pdf           the architecture screen, and the radius question
                                against the closed-form truncation floor
    fig_capacity.pdf            mixture capacity against a single Gaussian
    fig_nonmarkov.pdf           the two Markov violations, and the one scalar
                                that collapses them onto a single curve
    fig_information.pdf         Fisher information against the cumulant law

Reads the frozen research outputs; the analytic overlays are recomputed here
from closed forms and nothing is refitted.

    /usr/local/bin/python3.12 tools/fig_ch10.py
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import BLUE, VERM, GREEN, PURPLE, GREY, FULL, new_figure, save  # noqa: E402

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

RHO = 0.85
NSITES = 32
SCHEDULE = [0.05, 0.0725, 0.1053, 0.1527, 0.2216, 0.3215,
            0.4665, 0.6769, 0.9821, 1.4250, 2.0676, 3.0]


def read(rel: str) -> list[dict]:
    with open(FROZEN / rel) as fh:
        return list(csv.DictReader(fh))


# Per-level reference score energies of the shared exp_07 test bundle
# (tools/make_ref_energy.py). These recover the schedule-pooled estimand of
# Eq. (10.2) exactly from the stored per-level relative errors.
REF_ENERGY = {float(r["t"]): float(r["ref_sq"])
              for r in read("exp_07_ref_energy/ref_energy.csv")}


def pooled_rel_l2(cells) -> tuple[float, float]:
    """(EM, net) schedule-pooled relative L2 for one seed's cells."""
    out = []
    for key in ("em_bp_score_rel_l2", "net_score_rel_l2_selected"):
        num = sum(REF_ENERGY[float(r["t"])] * float(r[key]) ** 2 for r in cells)
        den = sum(REF_ENERGY[float(r["t"])] for r in cells)
        out.append((num / den) ** 0.5)
    return tuple(out)


# --------------------------------------------------------------- closed forms
def ar1_cov(r: float, n: int = NSITES) -> np.ndarray:
    i = np.arange(n)
    return r ** np.abs(i[:, None] - i[None, :])


def ar1_prec(r: float, n: int = NSITES) -> np.ndarray:
    q = 1.0 - r * r
    Q = np.zeros((n, n))
    d = np.arange(n)
    Q[d, d] = (1.0 + r * r) / q
    Q[0, 0] = Q[n - 1, n - 1] = 1.0 / q
    Q[d[:-1], d[1:]] = -r / q
    Q[d[1:], d[:-1]] = -r / q
    return Q


def bulk_q(t: float, r: float = RHO) -> float:
    """The geometric influence-decay rate of Chapter 6's bulk theorem."""
    ivar = 1.0 - r * r
    delta = 1.0 - np.exp(-2.0 * t)
    Jd = np.exp(-2.0 * t) / delta + (1.0 + r * r) / ivar
    b = r / ivar
    return float((Jd - np.sqrt(Jd * Jd - 4.0 * b * b)) / (2.0 * b))


def schedule_weights() -> np.ndarray:
    """How the pooled risk weights the noise schedule: w_t ~ E||S*||^2."""
    S0 = ar1_cov(RHO)
    w = []
    for t in SCHEDULE:
        St = np.exp(-2.0 * t) * S0 + (1.0 - np.exp(-2.0 * t)) * np.eye(NSITES)
        w.append(np.trace(np.linalg.inv(St)))
    w = np.array(w)
    return w / w.sum()


def truncation_floor(radii: np.ndarray) -> np.ndarray:
    """Coefficient-square tail of a radius-r window, pooled over the schedule.

    A DIAGNOSTIC, NOT A RISK: it is the fraction of squared influence-
    coefficient mass beyond radius r for the ideal bulk kernel,
    2 q^{2(r+1)} / (1 + q^2). Prediction risk depends on the covariance of the
    omitted observations and on reoptimising the retained coefficients, which
    this quantity ignores; windowed_oracle() below is the valid risk anchor.
    """
    w = schedule_weights()
    q = np.array([bulk_q(t) for t in SCHEDULE])
    pref = 2.0 / (1.0 + q ** 2)
    return np.array([float((w * pref * q ** (2 * (r + 1))).sum()) for r in radii])


def windowed_oracle(radii) -> np.ndarray:
    """Exact windowed-inference relative score error of the covariance-matched
    Gaussian design model, pooled over the schedule.

    For each site the radius-r window posterior mean E[a_k | x_{k-r..k+r}] is
    the Bayes-optimal radius-r estimator of the Gaussian chain (Chapter 6's
    windowed inference), so this is the exact local-oracle risk of the design
    calculation -- a closed form, no fit and no sampling. It is a Gaussian
    design anchor for the Laplace experiment, not that experiment's own local
    oracle.
    """
    S0 = ar1_cov(RHO)
    out = []
    for r in radii:
        num = den = 0.0
        for t in SCHEDULE:
            al = np.exp(-t)
            de = 1.0 - np.exp(-2.0 * t)
            St = al * al * S0 + de * np.eye(NSITES)
            Qt = np.linalg.inv(St)
            M = np.zeros((NSITES, NSITES))
            for k in range(NSITES):
                lo, hi = max(0, k - r), min(NSITES, k + r + 1)
                Jw = np.linalg.inv(S0[lo:hi, lo:hi]) + (al * al / de) * np.eye(hi - lo)
                M[k, lo:hi] = (al / de) * np.linalg.solve(Jw, np.eye(hi - lo)[:, k - lo])
            D = (al * M - np.eye(NSITES)) / de + Qt
            num += float(np.trace(D @ St @ D.T))
            den += float(np.trace(Qt))
        out.append(np.sqrt(num / den))
    return np.array(out)


def global_cov(beta: float) -> np.ndarray:
    return (ar1_cov(RHO) + beta ** 2) / (1.0 + beta ** 2)


def longrange_cov(gamma: float, ell: float = 8.0) -> np.ndarray:
    n = NSITES
    Q = ar1_prec(RHO, n)
    i = np.arange(n)
    dist = np.abs(i[:, None] - i[None, :])
    off = -np.exp(-dist / ell)
    off[dist < 2] = 0.0
    Ql = off.copy()
    Ql[i, i] = -off.sum(axis=1) * 1.05
    S = np.linalg.inv(Q + gamma * Ql)
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


def kl_per_site(Sp: np.ndarray, rho_cl: float) -> float:
    """Excess code length per site of the best first-order chain for Sp."""
    Sq = ar1_cov(rho_cl)
    _, ldp = np.linalg.slogdet(Sp)
    _, ldq = np.linalg.slogdet(Sq)
    tr = np.trace(np.linalg.solve(Sq, Sp))
    return float(0.5 * (tr - NSITES + ldq - ldp) / NSITES)


# ------------------------------------------------------- 1. value of structure
def _efficiency():
    rows = []
    for d in sorted(glob.glob(str(FROZEN / "exp_07_certified_seed*"))) + \
            sorted(glob.glob(str(FROZEN / "exp_07_n4096_seed*"))):
        seed = int(d.split("seed")[-1])
        with open(Path(d) / "sample_efficiency_val.csv") as fh:
            for x in csv.DictReader(fh):
                if x["em_resolved"] in ("1", "True", "true"):
                    x["seed"] = seed
                    rows.append(x)
    by = defaultdict(lambda: defaultdict(list))
    for x in rows:
        by[int(x["n_chains"])][x["seed"]].append(x)
    N, em, net, rat = [], [], [], []
    for n in sorted(by):
        e, m, r = [], [], []
        for s in sorted(by[n]):
            e_em, e_net = pooled_rel_l2(by[n][s])
            e.append(e_em)
            m.append(e_net)
            r.append(e_net / e_em)
        N.append(n)
        em.append(np.mean(e))
        net.append(np.mean(m))
        rat.append(np.mean(r))
    return (np.array(N, float), np.array(em), np.array(net), np.array(rat))


def _confirm():
    r = [x for x in read("exp_31_confirm_merged/confirm.csv") if x["region"] == "all"]
    by = defaultdict(dict)
    for x in r:
        by[int(x["n_chains"])].setdefault(x["method"], {})[int(x["seed"])] = float(x["risk"])
    N = np.array(sorted(by), float)
    E = np.array([np.mean(list(by[int(n)]["em_bp"].values())) for n in N])
    W = np.array([np.mean(list(by[int(n)]["window"].values())) for n in N])
    pr, se = [], []
    for n in N:
        seeds = sorted(by[int(n)]["em_bp"])
        v = np.array([by[int(n)]["window"][s] / by[int(n)]["em_bp"][s] for s in seeds])
        pr.append(v.mean())
        se.append(v.std(ddof=1) / np.sqrt(v.size))
    return N, E, W, np.array(pr), np.array(se)


def fig_value_of_structure() -> None:
    N7, em7, net7, rat7 = _efficiency()
    N3, E3, W3, pr3, se3 = _confirm()
    fig, (a, b, c) = new_figure(1, 3, width=FULL, height=3.05)

    # -- (a) against a denoiser with no structure at all --------------------
    s_net = np.polyfit(np.log(N7), np.log(net7), 1)[0]
    s_em = np.polyfit(np.log(N7), np.log(em7), 1)[0]
    a.loglog(N7, net7, "o-", color=VERM, lw=1.8, ms=4.2,
             label=f"global MLP,  $N^{{{s_net:.2f}}}$")
    a.loglog(N7, em7, "s-", color=BLUE, lw=1.8, ms=3.8,
             label=f"EM–BP,  $N^{{{s_em:.2f}}}$")
    a.set_xlim(N7[0] * 0.75, N7[-1] * 1.5)
    a.set_ylim(6e-3, 0.9)
    a.set_xticks(N7[::2])
    a.set_xticklabels([f"{int(v)}" for v in N7[::2]])
    a.minorticks_off()
    a.set_xlabel("training sequences $N$")
    a.set_ylabel("relative score error")
    a.set_title("(a) no structure supplied")
    # Lower left: anchored upper-right the first legend row lay across the
    # descending MLP curve.
    a.legend(loc="lower left", frameon=False, fontsize=7.8, borderaxespad=0.3)
    a.grid(True, which="both", alpha=0.16)

    # -- (b) against a denoiser that is given locality ----------------------
    # Two-parameter floor fit E(N)^2 = E_inf^2 + c/N on the window head's
    # squared risks (a floor plus a 1/N estimation term), plotted on the
    # risk axis itself. An earlier revision fitted the un-squared risks and
    # then plotted sqrt(risk), which mislabelled the axis by a square root.
    A = np.stack([np.ones_like(N3), 1.0 / N3], 1)
    sol = np.linalg.lstsq(A, W3 ** 2, rcond=None)[0]
    floor = float(np.sqrt(max(sol[0], 0.0)))
    # Does a FLOORLESS curve fit these four points about as well? Same
    # parameter count, so the residual sums are directly comparable, and if
    # they are close the data cannot distinguish "converged to a floor" from
    # "still decaying" -- which is a claim the caption would otherwise make.
    rss_floor = float(np.sum((W3 - np.sqrt(A @ sol)) ** 2))
    pw = np.polyfit(np.log(N3), np.log(W3), 1)
    rss_power = float(np.sum((W3 - np.exp(np.polyval(pw, np.log(N3)))) ** 2))
    rss_ratio = rss_power / rss_floor
    b.loglog(N3, W3, "o-", color=VERM, lw=1.8, ms=4.2)
    b.loglog(N3, E3, "s-", color=BLUE, lw=1.8, ms=3.8)
    ng = np.geomspace(N3[0], N3[-1], 60)
    b.loglog(ng, np.sqrt(sol[0] + sol[1] / ng), ":", color=GREY, lw=1.3)
    b.set_xlim(N3[0] * 0.75, N3[-1] * 1.5)
    b.set_ylim(0.017, 0.30)
    b.set_xticks(N3)
    b.set_xticklabels([f"{int(v)}" for v in N3])
    b.set_yticks([0.02, 0.04, 0.08, 0.15, 0.25])
    b.set_yticklabels(["0.02", "0.04", "0.08", "0.15", "0.25"])
    b.minorticks_off()
    b.set_xlabel("training sequences $N$")
    b.set_ylabel("relative score error, all sites")
    b.set_title("(b) locality supplied")
    # Curves labelled in place, no legend: anchored upper-right it overflowed
    # onto the 0.25 tick, and lower-left its first row met the EM-BP curve's
    # descent. The top-left and bottom-left corners are the two regions no
    # curve enters.
    b.annotate(f"window head, floor ${floor:.2f}$", xy=(0.04, 0.97),
               xycoords="axes fraction", ha="left", va="top",
               color=VERM, fontsize=7.8)
    b.annotate("EM–BP, no floor", xy=(0.04, 0.03),
               xycoords="axes fraction", ha="left", va="bottom",
               color=BLUE, fontsize=7.8)
    b.grid(True, which="both", alpha=0.16)

    # -- (c) the two ratios -------------------------------------------------
    c.semilogx(N7, rat7, "o-", color=PURPLE, lw=1.8, ms=4.2,
               label="vs global MLP")
    c.errorbar(N3, pr3, yerr=se3, fmt="s-", color=GREEN, lw=1.8, ms=3.8,
               capsize=2.5, elinewidth=0.9, label="vs window head")
    c.axhline(1.0, color=GREY, ls=(0, (4, 3)), lw=1.0)
    c.set_xlim(N7[0] * 0.75, N7[-1] * 1.5)
    c.set_ylim(0, 20)
    c.set_xticks(N7[::2])
    c.set_xticklabels([f"{int(v)}" for v in N7[::2]])
    c.minorticks_off()
    c.set_xlabel("training sequences $N$")
    c.set_ylabel("error ratio, baseline / EM–BP")
    c.set_title("(c) what each comparison says")
    c.legend(loc="upper left", frameon=False, fontsize=8)
    c.annotate("break-even", xy=(N7[-1] * 1.4, 1.0), xytext=(0, -3),
               textcoords="offset points", color=GREY, fontsize=7.4,
               ha="right", va="top")
    c.grid(True, alpha=0.16)

    save(fig, FIG / "fig_value_of_structure")
    print(f"  (a) slopes: net {s_net:.3f}  em {s_em:.3f}")
    print(f"  (a) ratios: {rat7[0]:.1f} -> {rat7[-1]:.1f}")
    floor_pct = 100.0 * floor / W3[-1]
    print(f"  (b) window floor E_inf={floor:.4f}  c={sol[1]:.4f} "
          f"(risk at N={int(N3[-1])} is {W3[-1]:.4f}; floor is "
          f"{floor_pct:.0f}% of it)")
    print(f"  (c) paired ratios: {pr3.round(2)}")
    # Numbers the Chapter 10 captions quote; generated, not typed.
    shared = THESIS / "sections" / "thesis-fig-numbers.tex"
    shared.write_text(
        "%% GENERATED by thesis/tools/fig_ch10.py -- do not hand-edit.\n"
        f"\\newcommand{{\\figwinfloor}}{{{floor:.2f}}}\n"
        f"\\newcommand{{\\figwinfloorpct}}{{{floor_pct:.0f}}}\n"
        f"\\newcommand{{\\figwinpowerslope}}{{{-pw[0]:.2f}}}\n"
        f"\\newcommand{{\\figwinrssratio}}{{{rss_ratio:.1f}}}\n"
        f"\\newcommand{{\\figslopeem}}{{{-s_em:.2f}}}\n"
        f"\\newcommand{{\\figslopenet}}{{{-s_net:.2f}}}\n")
    print(f"  wrote {shared}")


# ------------------------------------------------------------ 2. the screen
def fig_screening() -> None:
    rows = [x for x in read("exp_31_screen/screening.csv") if x["region"] == "bulk"]
    # Pooled over development seeds and sizes before the square root: the same
    # accumulation as the selector that chose the confirmatory baseline
    # (_winners_from_screening), not a mean of per-cell risks.
    sc, npar = defaultdict(lambda: [0.0, 0.0]), {}
    for x in rows:
        k = (x["arch"], x["hp"])
        sc[k][0] += float(x["sq_err"])
        sc[k][1] += float(x["sq_ref"])
        npar[k] = int(x["n_params"])
    scored = {k: float(np.sqrt(e / d)) for k, (e, d) in sc.items()}

    fig, (a, b) = new_figure(1, 2, width=FULL, height=2.95)

    style = {"window": (BLUE, "o", "weight-shared window"),
             "conv": (VERM, "^", "dilated convolution"),
             "bimp": (GREEN, "s", "bidirectional passing")}
    for arch, (col, mk, lab) in style.items():
        ks = [k for k in scored if k[0] == arch]
        x = np.array([npar[k] for k in ks], float)
        y = np.array([scored[k] for k in ks])
        a.loglog(x, y, mk, color=col, ms=4.4, alpha=0.45, mew=0, label=lab)
        j = int(np.argmin(y))
        a.loglog([x[j]], [y[j]], mk, mfc="none", mec=col, ms=9.5, mew=1.6)
    a.set_xlim(3.2e3, 4.2e5)
    a.set_ylim(0.080, 0.62)
    a.set_yticks([0.1, 0.15, 0.2, 0.3, 0.45])
    a.set_yticklabels(["0.10", "0.15", "0.20", "0.30", "0.45"])
    a.minorticks_off()
    a.set_xticks([1e4, 1e5])
    a.set_xticklabels(["10k", "100k"])
    a.set_xlabel("free parameters")
    a.set_ylabel("selection risk")
    a.set_title("(a) every configuration screened")
    a.legend(loc="lower right", frameon=False, fontsize=7.4)
    a.grid(True, which="both", alpha=0.16)

    # -- (b) radius, measured against the closed-form truncation floor ------
    radii = sorted({json.loads(k[1])["radius"] for k in scored if k[0] == "window"})
    meas = {}
    for width, col, mk in ((64, BLUE, "o"), (128, GREEN, "s")):
        y = []
        for r in radii:
            v = [scored[k] for k in scored if k[0] == "window"
                 and json.loads(k[1])["radius"] == r
                 and json.loads(k[1])["width"] == width
                 and json.loads(k[1])["parameterization"] == "eps"]
            y.append(min(v))
        meas[width] = y
        b.plot(radii, y, mk + "-", color=col, lw=1.8, ms=4.2)
    rr = np.arange(0, 17)
    b.plot(rr, windowed_oracle(rr), ":", color=GREY, lw=1.6)
    b.set_xlim(0.5, 17.4)
    b.set_ylim(0.0, 0.272)
    b.set_xticks(radii)
    b.set_xlabel("window radius $r$")
    b.set_ylabel("risk")
    b.set_title("(b) receptive field: cost and benefit")
    # Curves labelled in place, no legend: every corner box met the oracle's
    # descent (left) or its tail (lower right). The measured curves are
    # labelled at their right ends, the oracle in the empty band between the
    # measured risks and its own tail.
    b.annotate("width 128", xy=(radii[-1], meas[128][-1]),
               xytext=(0, 6), textcoords="offset points", ha="right",
               va="bottom", color=GREEN, fontsize=7.4)
    b.annotate("width 64", xy=(radii[-1], meas[64][-1]),
               xytext=(0, -11), textcoords="offset points", ha="right",
               va="top", color=BLUE, fontsize=7.4)
    b.annotate("Gaussian windowed oracle (exact)", xy=(10.0, 0.052),
               ha="center", va="bottom", color=GREY, fontsize=7.4)
    best_r = min(
        (v, json.loads(k[1])["radius"]) for k, v in scored.items()
        if k[0] == "window" and json.loads(k[1])["width"] == 64
        and json.loads(k[1])["parameterization"] == "eps")
    b.annotate("best", xy=(best_r[1], best_r[0]), xytext=(0, -14),
               textcoords="offset points", color=BLUE, fontsize=7.6,
               ha="center", va="top",
               bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.4))
    b.grid(True, alpha=0.16)

    save(fig, FIG / "fig_screening")
    best = {}
    for k, v in sorted(scored.items(), key=lambda kv: kv[1]):
        best.setdefault(k[0], (v, k[1]))
    for arch in ("window", "conv", "bimp"):
        print(f"  {arch:7} best {best[arch][0]:.4f}  {best[arch][1]}")
    tf = truncation_floor(np.array(radii))
    wo = windowed_oracle(np.array(radii))
    for r, f, w in zip(radii, tf, wo):
        print(f"  r={r:2d}: coeff-tail {f:.3e}   windowed oracle {w:.4f}")


# ---------------------------------------------------------------- 3. capacity
def fig_capacity() -> None:
    rows = read("exp_32_capacity_merged/capacity_equivalence.csv")
    by = {}
    for x in rows:
        by[(int(x["seed"]), int(x["n_chains"]), int(x["n_components"]))] = x
    Cs = [2, 4, 8, 16]
    fig, (a, b) = new_figure(1, 2, width=FULL, height=2.85)

    for n, col, mk, off in ((128, BLUE, "o", -0.10), (512, VERM, "s", +0.10)):
        m, e = [], []
        for C in Cs:
            d = np.array([float(by[(s, n, C)]["test_log_evidence_per_edge"])
                          - float(by[(s, n, 1)]["test_log_evidence_per_edge"])
                          for s in range(16) if (s, n, C) in by and (s, n, 1) in by])
            m.append(d.mean())
            e.append(d.std(ddof=1) / np.sqrt(d.size))
        x = np.arange(len(Cs), dtype=float) + off
        a.errorbar(x, m, yerr=e, fmt=mk + "-", color=col, lw=1.7, ms=4.2,
                   capsize=2.8, elinewidth=0.9, label=f"$N = {n}$")
    a.axhline(0.0, color=GREY, ls=(0, (4, 3)), lw=1.0)
    a.set_xticks(range(len(Cs)))
    a.set_xticklabels([str(C) for C in Cs])
    a.set_xlim(-0.55, len(Cs) - 0.45)
    a.set_ylim(-0.055, 0.016)
    a.set_xlabel("mixture components $C$")
    a.set_ylabel("held-out log-evidence per edge,\npaired against $C = 1$")
    a.set_title("(a) what capacity buys")
    a.legend(loc="lower left", frameon=False, fontsize=8)
    a.annotate("nothing above this line", xy=(len(Cs) - 0.55, 0.0),
               xytext=(0, 4), textcoords="offset points", color=GREY,
               fontsize=7.4, ha="right", va="bottom")
    a.grid(True, alpha=0.16)

    allC = [1] + Cs
    unres, capped = [], []
    for C in allC:
        v = [by[(s, n, C)] for n in (128, 512) for s in range(16) if (s, n, C) in by]
        unres.append(100.0 * sum(1 for x in v if float(x["s_min_over_h"]) < 2) / len(v))
        capped.append(100.0 * sum(1 for x in v
                                  if x["em_converged"] in ("False", "false")) / len(v))
    x = np.arange(len(allC), dtype=float)
    b.bar(x - 0.19, capped, width=0.36, color=VERM, alpha=0.85, lw=0,
          label="stopped at the iteration cap")
    b.bar(x + 0.19, unres, width=0.36, color=BLUE, alpha=0.85, lw=0,
          label="narrowest component under the grid floor")
    b.set_xticks(x)
    b.set_xticklabels([str(C) for C in allC])
    b.set_xlim(-0.6, len(allC) - 0.4)
    b.set_ylim(0, 128)
    b.set_yticks([0, 25, 50, 75, 100])
    b.set_xlabel("mixture components $C$")
    b.set_ylabel("share of fits (\\%)")
    b.set_title("(b) why (a) is not a clean capacity test")
    b.legend(loc="upper left", frameon=False, fontsize=7.4)
    b.grid(True, axis="y", alpha=0.16)

    save(fig, FIG / "fig_capacity")
    print(f"  capped %: {[round(v) for v in capped]}")
    print(f"  under-resolved %: {[round(v) for v in unres]}")


# --------------------------------------------------------------- 4. non-Markov
CHOW_LIU = {("global", 0.0): 0.8500, ("global", 0.1): 0.8515,
            ("global", 0.25): 0.8588, ("global", 0.5): 0.8800,
            ("global", 1.0): 0.9250,
            ("longrange", 0.0): 0.8500, ("longrange", 0.05): 0.7860,
            ("longrange", 0.1): 0.7501, ("longrange", 0.2): 0.7020,
            ("longrange", 0.4): 0.6415}


def _nonmarkov():
    rows = []
    for f in sorted(glob.glob(str(FROZEN / "exp_21_clean/*/nonmarkov_*.csv"))):
        fam = "laplace" if "laplace" in f else "gauss"
        with open(f) as fh:
            for x in csv.DictReader(fh):
                if fam == "laplace":
                    x["mechanism"], x["strength"] = "global", x["beta"]
                x["fam"] = fam
                rows.append(x)
    by = defaultdict(dict)
    for x in rows:
        by[(x["fam"], x["mechanism"], float(x["strength"]), float(x["t"]))][x["arm"]] = x
    out = {}
    for fam, mech, s, _ in list(by):
        if (fam, mech, s) in out:
            continue
        ts = sorted({k[3] for k in by if k[:3] == (fam, mech, s)})
        out[(fam, mech, s)] = {
            arm: float(np.mean([float(by[(fam, mech, s, t)][arm]["ratio_to_em"])
                                for t in ts]))
            for arm in ("cnn", "mlp")}
    return out


def fig_nonmarkov() -> None:
    adv = _nonmarkov()
    betas = [0.0, 0.1, 0.25, 0.5, 1.0]
    gammas = [0.0, 0.05, 0.1, 0.2, 0.4]
    kl_b = [kl_per_site(global_cov(x), CHOW_LIU[("global", x)]) for x in betas]
    kl_g = [kl_per_site(longrange_cov(x), CHOW_LIU[("longrange", x)]) for x in gammas]

    fig, (a, b, c) = new_figure(1, 3, width=FULL, height=3.05)

    # -- (a) the analytic distance to the chain family ----------------------
    a.semilogy(betas[1:], kl_b[1:], "o-", color=BLUE, lw=1.9, ms=4.2,
               label="global latent, $\\beta$")
    a.semilogy(gammas[1:], kl_g[1:], "^-", color=VERM, lw=1.9, ms=4.6,
               label="long-range, $\\gamma$")
    a.axhspan(1.1e-2, 2.3e-2, color=GREY, alpha=0.13, lw=0)
    a.set_xlim(0.0, 1.08)
    a.set_ylim(1e-6, 0.4)
    a.set_xlabel("contamination strength")
    a.set_ylabel(r"covariance divergence $\delta$ (nats/site)")
    # Two-line titles on the two long panels: at a third of the text block a
    # one-line title ran across the neighbouring panel's own title.
    a.set_title("(a) covariance distance\nto the chain family")
    a.legend(loc="lower right", frameon=False, fontsize=7.6)
    a.annotate("break-even band", xy=(1.05, 1.6e-2), color=GREY, fontsize=7.2,
               ha="right", va="center")
    a.grid(True, which="both", alpha=0.16)

    # -- (b) the measured advantage against strength ------------------------
    for key, xs, col, mk, lab in (
            (("gauss", "global"), betas, BLUE, "o", "Gaussian, $\\beta$"),
            (("laplace", "global"), betas, GREEN, "s", "Laplace, $\\beta$"),
            (("gauss", "longrange"), gammas, VERM, "^", "Gaussian, $\\gamma$")):
        y = [adv[(key[0], key[1], s)]["cnn"] for s in xs]
        b.semilogy(xs, y, mk + "-", color=col, lw=1.8, ms=4.3, label=lab)
    b.axhline(1.0, color=GREY, ls=(0, (4, 3)), lw=1.0)
    b.set_xlim(-0.05, 1.06)
    b.set_ylim(0.4, 40)
    b.set_yticks([0.5, 1, 2, 5, 10, 20])
    b.set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
    b.minorticks_off()
    b.set_xlabel("contamination strength")
    b.set_ylabel("error ratio, baseline / EM–BP")
    b.set_title("(b) advantage against strength")
    b.legend(loc="upper right", frameon=False, fontsize=7.6)
    b.annotate("break-even", xy=(1.04, 1.0), xytext=(0, -3),
               textcoords="offset points", color=GREY, fontsize=7.2,
               ha="right", va="top")
    b.grid(True, which="both", alpha=0.16)

    # -- (c) the collapse ---------------------------------------------------
    floor = 3e-7
    for key, xs, kl, col, mk, lab in (
            (("gauss", "global"), betas, kl_b, BLUE, "o", "Gaussian, $\\beta$"),
            (("laplace", "global"), betas, kl_b, GREEN, "s", "Laplace, $\\beta$"),
            (("gauss", "longrange"), gammas, kl_g, VERM, "^", "Gaussian, $\\gamma$")):
        y = [adv[(key[0], key[1], s)]["cnn"] for s in xs]
        c.loglog(np.maximum(kl, floor), y, mk, color=col, ms=5.0, mew=0,
                 alpha=0.9, label=lab)
    order = np.argsort(kl_b[1:] + kl_g[1:])
    xx = np.array(kl_b[1:] + kl_g[1:])[order]
    yy = np.array([adv[("gauss", "global", s)]["cnn"] for s in betas[1:]]
                  + [adv[("gauss", "longrange", s)]["cnn"] for s in gammas[1:]])[order]
    c.loglog(xx, yy, "-", color=GREY, lw=1.0, alpha=0.6, zorder=1)
    c.axhline(1.0, color=GREY, ls=(0, (4, 3)), lw=1.0)
    c.set_xlim(2e-6, 0.25)
    c.set_ylim(0.4, 40)
    c.set_yticks([0.5, 1, 2, 5, 10, 20])
    c.set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
    # Short form: the full "covariance divergence" wording ran past the
    # figure's right edge under this panel and was clipped in the PDF.
    c.set_xlabel(r"$\delta$ (nats/site)")
    c.set_ylabel("error ratio, baseline / EM–BP")
    c.set_title("(c) one empirical curve,\nboth mechanisms")
    c.legend(loc="upper right", frameon=False, fontsize=7.6)
    c.annotate("break-even", xy=(3e-6, 1.0), xytext=(0, -3),
               textcoords="offset points", color=GREY, fontsize=7.2,
               ha="left", va="top")
    c.grid(True, which="both", alpha=0.16)

    save(fig, FIG / "fig_nonmarkov")
    print("  beta   KL/site   ratio(cnn)")
    for s, k in zip(betas, kl_b):
        print(f"   {s:4.2f}  {k:9.3e}  {adv[('gauss','global',s)]['cnn']:6.2f}  "
              f"(laplace {adv[('laplace','global',s)]['cnn']:6.2f})")
    print("  gamma  KL/site   ratio(cnn)")
    for s, k in zip(gammas, kl_g):
        print(f"   {s:4.2f}  {k:9.3e}  {adv[('gauss','longrange',s)]['cnn']:6.2f}")


# ------------------------------------------------------------- 5. information
def fig_information() -> None:
    rows = read("exp_22/shape_information.csv")
    ps = sorted({float(x["beta_true"]) for x in rows})
    fig, (a, b) = new_figure(1, 2, width=FULL, height=2.95)

    sel = 1.0  # the Laplace case, which is the chapter's running example
    rr = sorted([x for x in rows if float(x["beta_true"]) == sel],
                key=lambda z: float(z["t"]))
    t = np.array([float(x["t"]) for x in rr])
    for key, col, mk, lab in ((("eff_rho"), BLUE, "s", r"$\alpha$"),
                              (("eff_q"), GREEN, "D", r"$\sigma_\eta^2$"),
                              (("eff_beta"), VERM, "o", "shape $p$")):
        y = np.array([float(x[key]) for x in rr])
        a.semilogy(t, y, mk + "-", color=col, lw=1.8, ms=4.0, label=lab)
    g4 = np.linspace(0.05, 1.74, 60)
    g8 = np.linspace(0.05, 1.16, 60)
    a.semilogy(g4, 128 * np.exp(-4.0 * g4), ":", color=GREY, lw=1.4)
    a.semilogy(g8, 2.4 * np.exp(-8.0 * g8), ":", color=GREY, lw=1.4)
    a.set_xlim(0.0, 2.05)
    a.set_ylim(5e-5, 900)
    a.set_xlabel("diffusion time $t$")
    a.set_ylabel("efficient Fisher information")
    a.set_title("(a) Laplace innovations, $p = 1$")
    a.legend(loc="lower left", frameon=False, fontsize=8)
    a.annotate(r"$e^{-4t}$", xy=(1.74, 128 * np.exp(-4 * 1.74)), xytext=(4, 0),
               textcoords="offset points", color=GREY, fontsize=8.0,
               ha="left", va="center")
    a.annotate(r"$e^{-8t}$", xy=(1.16, 2.4 * np.exp(-8 * 1.16)), xytext=(4, -5),
               textcoords="offset points", color=GREY, fontsize=8.0,
               ha="left", va="top")
    a.grid(True, which="both", alpha=0.16)

    cols = [BLUE, GREEN, PURPLE, VERM, GREY]
    slopes = []
    for p, col in zip(ps, cols):
        rr = sorted([x for x in rows if float(x["beta_true"]) == p],
                    key=lambda z: float(z["t"]))
        t = np.array([float(x["t"]) for x in rr])
        r = np.array([float(x["eff_beta"]) / float(x["eff_rho"]) for x in rr])
        b.semilogy(t, r / r[0], "o-", color=col, lw=1.6, ms=3.6, label=f"$p = {p:g}$")
        slopes.append(np.polyfit(t, np.log(r), 1)[0])
    tg = np.linspace(0.05, 1.6, 30)
    b.semilogy(tg, np.exp(-4.0 * (tg - 0.05)), "--", color="black", lw=1.5,
               zorder=5)
    b.set_xlim(0.0, 2.35)
    b.set_ylim(2e-4, 2.0)
    b.set_xlabel("diffusion time $t$")
    b.set_ylabel("shape / correlation information,\nrelative to $t = 0.05$")
    b.set_title("(b) the predicted contrast, five shapes")
    b.legend(loc="lower left", frameon=False, fontsize=7.4, ncol=2)
    b.annotate("predicted\n" + r"$e^{-4t}$", xy=(1.63, np.exp(-4 * 1.55)),
               xytext=(6, 0), textcoords="offset points", color="black",
               fontsize=7.8, ha="left", va="center", annotation_clip=False)
    b.grid(True, which="both", alpha=0.16)

    save(fig, FIG / "fig_information")

    # Conditioning of the outer-product information matrix. The reported
    # quantity is 1/(J^{-1})_pp, so how invertible J is decides how much of
    # the far tail means anything; exp_22 records it per cell and the thesis
    # should not have to take that on trust.
    cond = np.array([float(x["condition_number"]) for x in rows])
    best = min(rows, key=lambda x: float(x["condition_number"]))
    worst = max(rows, key=lambda x: float(x["condition_number"]))
    condfile = THESIS / "sections" / "info-conditioning-numbers.tex"
    condfile.write_text(
        "%% GENERATED by thesis/tools/fig_ch10.py -- do not hand-edit.\n"
        "%% Condition numbers of the outer-product information matrix J,\n"
        "%% from exp_22's own per-cell record.\n"
        f"\\newcommand{{\\infcondbest}}{{{float(best['condition_number']):.0f}}}\n"
        f"\\newcommand{{\\infcondbestt}}{{{float(best['t']):.2f}}}\n"
        f"\\newcommand{{\\infcondmedian}}{{{np.median(cond):.0f}}}\n"
        f"\\newcommand{{\\infcondworst}}{{{float(worst['condition_number']):.1e}}}\n"
        f"\\newcommand{{\\infcondworstt}}{{{float(worst['t']):.1f}}}\n"
        f"\\newcommand{{\\infcondworstp}}{{{float(worst['beta_true']):g}}}\n"
        f"\\newcommand{{\\infcondworstinfo}}{{{float(worst['eff_beta']):.0e}}}\n")
    print(f"  wrote {condfile}")
    print(f"  condition numbers: best {float(best['condition_number']):.0f} "
          f"(t={float(best['t'])}), median {np.median(cond):.0f}, "
          f"worst {float(worst['condition_number']):.2e} "
          f"(t={float(worst['t'])}, p={float(worst['beta_true'])})")
    print(f"  measured log-slope of I_p/I_alpha: {np.round(slopes, 2)} "
          f"(mean {np.mean(slopes):.2f}, predicted -4)")
    ex = []
    for p in ps:
        rr = sorted([x for x in rows if float(x["beta_true"]) == p],
                    key=lambda z: float(z["t"]))
        al = [float(x["eff_rho"]) for x in rr]
        sh = [float(x["eff_beta"]) for x in rr]
        ex.append((sh[0] / sh[-1]) / (al[0] / al[-1]))
        print(f"  p={p:4.2f}  alpha loses {al[0]/al[-1]:7.1f}x   "
              f"shape loses {sh[0]/sh[-1]:10.1f}x   extra {ex[-1]:7.1f}x")
    print(f"  geometric mean extra factor {np.exp(np.mean(np.log(ex))):.1f}, "
          f"predicted {np.exp(4*1.55):.1f}")


if __name__ == "__main__":
    fig_value_of_structure()
    fig_screening()
    fig_capacity()
    fig_nonmarkov()
    fig_information()
