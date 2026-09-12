#!/usr/bin/env python3
"""Chapter 7 figures: Laplace innovations and the Gaussian-message closure.

    fig_laplace_nonlinear.pdf -- what changes when the innovation law goes
                                 Laplace: the same second moment, excess
                                 kurtosis +3, and a joint score that stops
                                 being linear in x.
    fig_laplace_closure.pdf   -- the measured cost of the Gaussian closure
                                 against validated grid BP (frozen exp_02).

The right panel of the first figure runs the exact functional recursion
numerically (grid BP, the solver of Sec. 7.4) on a short Laplace chain, and
overlays the exactly linear score of the covariance-matched Gaussian chain --
which, by Thm 7.x, is precisely what the Gaussian closure returns. The gap
between the two curves IS the object the chapter measures.

House style (sans-serif, text-block width, Okabe-Ito) from figstyle.py.

    /usr/local/bin/python3.12 tools/fig_ch07.py [name ...]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import BLUE, VERM, GREEN, GREY, FULL, ramp, new_figure, save  # noqa: E402

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

EXP02 = _find_frozen() / "exp_02" / "laplace_summary.csv"


# ----------------------------------------------------------------- algebra --
def channel(t: float) -> tuple[float, float]:
    return float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))


def laplace_scale(alpha: float) -> float:
    """b with Var = 2b^2 = 1 - alpha^2: the variance-matched innovation."""
    return float(np.sqrt((1.0 - alpha ** 2) / 2.0))


def ar1_cov(n: int, alpha: float) -> np.ndarray:
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return alpha ** d


# ------------------------------------------------------- grid BP (Sec. 7.4) --
def grid_bp_score(x: np.ndarray, alpha: float, t: float, k: int,
                  A: float = 8.0, N: int = 401) -> float:
    """Exact-in-the-limit joint score component S_k(x,t) for the Laplace chain.

    The discretised forward-backward recursion of Sec. 7.4: messages live on N
    equispaced points of [-A, A], transition integrals are quadratures against
    the transition matrix, and every message is renormalised after each step.
    """
    L = len(x)
    b = laplace_scale(alpha)
    mu, D = channel(t)
    g = np.linspace(-A, A, N)
    h = g[1] - g[0]

    # transition kernel T[i, j] = M(a_i | a_j), and the unary OU likelihoods
    T = np.exp(-np.abs(g[:, None] - alpha * g[None, :]) / b) / (2.0 * b)
    ell = np.exp(-((x[:, None] - mu * g[None, :]) ** 2) / (2.0 * D))
    prior0 = np.exp(-g ** 2 / 2.0)

    fwd = np.empty((L, N))
    m = prior0 * ell[0]
    fwd[0] = m / (m.sum() * h)
    for i in range(1, L):
        m = (T @ fwd[i - 1]) * h * ell[i]
        fwd[i] = m / (m.sum() * h)

    bwd = np.ones((L, N))
    for i in range(L - 2, -1, -1):
        m = (T.T @ (ell[i + 1] * bwd[i + 1])) * h
        bwd[i] = m / (m.max() + 1e-300)

    belief = fwd[k] * bwd[k]
    mean = float((g * belief).sum() / belief.sum())
    return (mu * mean - x[k]) / D


def gaussian_score(x: np.ndarray, alpha: float, t: float, k: int) -> float:
    """The covariance-matched Gaussian chain's score: exactly linear in x."""
    mu, D = channel(t)
    S = mu * mu * ar1_cov(len(x), alpha) + D * np.eye(len(x))
    return float(-(np.linalg.inv(S) @ x)[k])


# --------------------------------------------------- 1. what non-Gaussianity is --
def fig_laplace_nonlinear(alpha: float = 0.85, L: int = 9) -> None:
    """Three panels, one colour convention: Laplace is vermilion and solid,
    Gaussian is blue and dashed, in every panel.

    An earlier two-panel version labelled the curves inline and put four
    entries in one legend; both collided with the data (the kurtosis labels sat
    on the density curves, the legend text ran across the zero axis). Splitting
    the two diffusion times into their own panels lets colour carry the model
    rather than the time, which needs one two-entry legend for the whole
    figure and leaves every label in empty space.
    """
    b = laplace_scale(alpha)
    sig = np.sqrt(1.0 - alpha ** 2)
    k = L // 2

    fig, (ax0, ax1, ax2) = new_figure(1, 3, width=FULL, height=2.8,
                                      sharey=False)

    # -- 1: the innovation laws, matched to the second moment ---------------
    e = np.linspace(-3.2, 3.2, 800)
    lap = np.exp(-np.abs(e) / b) / (2.0 * b)
    gau = np.exp(-e ** 2 / (2 * sig ** 2)) / np.sqrt(2 * np.pi * sig ** 2)

    ax0.semilogy(e, lap, color=VERM, lw=1.9,
                 label="Laplace, kurtosis $+3$")
    ax0.semilogy(e, gau, color=BLUE, lw=1.7, ls=(0, (5, 2.5)),
                 label="Gaussian, kurtosis $0$")
    # headroom above the peak (1.3) so the legend never meets a curve
    ax0.set_ylim(1e-4, 20.0)
    ax0.set_xlim(e[0], e[-1])
    ax0.set_xlabel(r"innovation $\varepsilon_k$")
    ax0.set_ylabel("density (log scale)")
    ax0.set_title("the innovation law")
    ax0.legend(loc="upper right", fontsize=7.0, frameon=False,
               handlelength=1.8, labelspacing=0.3, borderaxespad=0.1)

    # -- 2 and 3: the joint score at the middle frame, two diffusion times ---
    xk = np.linspace(-4.0, 4.0, 121)
    for ax, t in ((ax1, 0.05), (ax2, 1.0)):
        sl, sg = [], []
        for v in xk:
            x = np.zeros(L)
            x[k] = v
            sl.append(grid_bp_score(x, alpha, t, k))
            sg.append(gaussian_score(x, alpha, t, k))
        ax.plot(xk, sl, color=VERM, lw=2.0, zorder=3, label="Laplace chain")
        # dashed on top: at t = 1 the two coincide, and drawing the dashes over
        # the solid is what makes that visible rather than merely absent
        ax.plot(xk, sg, color=BLUE, lw=1.3, ls=(0, (4, 2.5)), zorder=4,
                label="matched Gaussian")
        ax.set_xlim(xk[0], xk[-1])
        ax.set_ylim(-16, 16)
        ax.set_xlabel(r"observation $x_k$")
        ax.set_title(rf"score at $t={t:g}$")

    ax1.set_ylabel(r"joint score $S_k(x,t)$")
    ax2.set_yticklabels([])
    # the curves occupy the upper-left and lower-right; the lower-left corner
    # of the t = 0.05 panel is the one region both of them avoid
    ax1.legend(loc="lower left", fontsize=7.0, frameon=False,
               handlelength=1.8, labelspacing=0.3, borderaxespad=0.15)

    save(fig, FIG / "fig_laplace_nonlinear")


# ------------------------------------------- the metric sweep (Sec. 7.5.3) --
# Reproduces the frozen exp_02 configuration (alpha = 0.85, L = 32, 60 trials)
# and recomputes it under four metrics rather than one. The relative score
# error agrees with the frozen CSV to within the seed spread (0.236 vs 0.241
# at t = 0.05; cosine 0.9761 vs 0.9756), which is what licenses reporting the
# other three from this run.
SWEEP_ALPHA, SWEEP_L, SWEEP_TRIALS = 0.85, 32, 60
SWEEP_CSV = Path(__file__).resolve().parent / "data" / "ch07_metrics.csv"
SWEEP_TS = [0.02, 0.032, 0.05, 0.0725, 0.1053, 0.1527, 0.2216, 0.3215,
            0.4665, 0.6769, 0.9821, 1.425, 2.0676, 3.0]


def grid_bp_marginals(x: np.ndarray, alpha: float, t: float,
                      A: float = 8.0, N: int = 401
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and variance at every site, by the recursion of Sec. 7.4."""
    n = len(x)
    b = laplace_scale(alpha)
    mu, D = channel(t)
    g = np.linspace(-A, A, N)
    h = g[1] - g[0]

    K = np.exp(-np.abs(g[:, None] - alpha * g[None, :]) / b) / (2.0 * b)
    ll = -((x[:, None] - mu * g[None, :]) ** 2) / (2.0 * D)
    ell = np.exp(ll - ll.max(axis=1, keepdims=True))   # per-site rescale, exact

    fwd = np.empty((n, N))
    m = np.exp(-g ** 2 / 2.0) * ell[0]
    fwd[0] = m / (m.sum() * h)
    for i in range(1, n):
        m = (K @ fwd[i - 1]) * h * ell[i]
        fwd[i] = m / (m.sum() * h)

    bwd = np.ones((n, N))
    for i in range(n - 2, -1, -1):
        m = (K.T @ (ell[i + 1] * bwd[i + 1])) * h
        bwd[i] = m / m.max()

    mean = np.empty(n)
    var = np.empty(n)
    for i in range(n):
        be = fwd[i] * bwd[i]
        be = be / be.sum()
        mean[i] = float((g * be).sum())
        var[i] = float(((g - mean[i]) ** 2 * be).sum())
    return mean, var


def lmmse_mean(x: np.ndarray, alpha: float, t: float) -> np.ndarray:
    """The Gaussian closure's posterior mean, in closed form (Thm 7.2)."""
    mu, D = channel(t)
    S0 = ar1_cov(len(x), alpha)
    return mu * S0 @ np.linalg.solve(mu * mu * S0 + D * np.eye(len(x)), x)


def sample_laplace_chain(rng, n: int, alpha: float) -> np.ndarray:
    b = laplace_scale(alpha)
    a = np.empty(n)
    a[0] = rng.standard_normal()
    e = rng.laplace(0.0, b, size=n - 1)
    for i in range(1, n):
        a[i] = alpha * a[i - 1] + e[i - 1]
    return a


def run_sweep() -> None:
    """Write tools/data/ch07_metrics.csv. Takes a couple of minutes."""
    alpha, L, n_tr = SWEEP_ALPHA, SWEEP_L, SWEEP_TRIALS
    SWEEP_CSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["t", "eps_pm_med", "eps_pm_q10", "eps_pm_q90",
            "score_med", "score_q10", "score_q90",
            "excess_frac_med", "mmse_floor_med", "cos_med",
            "p_gt_0p1", "grid_selfconv_med"]
    rows = []
    for t in SWEEP_TS:
        mu, D = channel(t)
        pm, sc, ex, fl, cs, gc = [], [], [], [], [], []
        for tr in range(n_tr):
            rng = np.random.default_rng([0xC7, int(round(t * 1e4)), tr])
            a = sample_laplace_chain(rng, L, alpha)
            x = mu * a + np.sqrt(D) * rng.standard_normal(L)
            m_ref, v_ref = grid_bp_marginals(x, alpha, t, N=801)
            m_cnv, _ = grid_bp_marginals(x, alpha, t, N=401)
            m_g = lmmse_mean(x, alpha, t)
            d = m_g - m_ref
            pm.append(np.mean(d ** 2))                       # eps_PM^2
            fl.append(np.mean(v_ref))                        # MMSE floor / site
            ex.append(np.sum(d ** 2) / np.sum(v_ref))        # excess / floor
            s_ref = (mu * m_ref - x) / D
            s_g = (mu * m_g - x) / D
            s_cnv = (mu * m_cnv - x) / D
            nref = np.linalg.norm(s_ref)
            sc.append(np.linalg.norm(s_g - s_ref) / nref)
            gc.append(np.linalg.norm(s_cnv - s_ref) / nref)
            cs.append(float(np.dot(s_g, s_ref)
                            / (np.linalg.norm(s_g) * nref)))
        pm, sc = np.sqrt(np.array(pm)), np.array(sc)
        rows.append(dict(zip(cols, [
            t, np.median(pm), np.quantile(pm, .1), np.quantile(pm, .9),
            np.median(sc), np.quantile(sc, .1), np.quantile(sc, .9),
            np.median(ex), np.median(fl), np.median(cs),
            float(np.mean(sc > 0.1)), np.median(gc)])))
        print(f"  t={t:<7.4f} eps_PM={rows[-1]['eps_pm_med']:.4f} "
              f"score={rows[-1]['score_med']:.4f} "
              f"excess%={100*rows[-1]['excess_frac_med']:.3f} "
              f"cos={rows[-1]['cos_med']:.4f} "
              f"p>.1={rows[-1]['p_gt_0p1']:.2f} "
              f"grid={rows[-1]['grid_selfconv_med']:.2e}")

    with open(SWEEP_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {SWEEP_CSV.name}")


# ---------------------------------------------- 3. the four-metric comparison --
def fig_laplace_metrics() -> None:
    if not SWEEP_CSV.exists():
        run_sweep()
    with open(SWEEP_CSV) as fh:
        rows = list(csv.DictReader(fh))
    d = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    t = d["t"]

    fig, (axA, axB, axC) = new_figure(1, 3, width=FULL, height=2.9)

    # -- A: the primary metric, posterior-mean NRMSE ------------------------
    axA.fill_between(t, d["eps_pm_q10"], d["eps_pm_q90"], color=GREEN,
                     alpha=0.16, lw=0)
    axA.loglog(t, d["eps_pm_med"], color=GREEN, lw=2.0, zorder=3)
    axA.set_xlim(t[0], t[-1])
    axA.set_ylim(1e-5, 1.0)
    axA.set_xlabel(r"diffusion time $t$")
    axA.set_ylabel(r"$\varepsilon_{\mathrm{PM}}$")
    axA.set_title("what inference loses")
    axA.grid(True, which="both", alpha=0.16)
    axA.annotate("median,\n10-90 pct", xy=(0.023, 1.5e-4), color=GREEN,
                 fontsize=7.6, ha="left", va="bottom")

    # -- B: the same error seen through the score ---------------------------
    axB.loglog(t, d["score_med"], color=VERM, lw=2.0, zorder=3)
    axB.loglog(t, d["eps_pm_med"], color=GREEN, lw=2.0, zorder=3)
    axB.set_xlim(t[0], t[-1])
    axB.set_ylim(1e-5, 1.0)
    axB.set_xlabel(r"diffusion time $t$")
    axB.set_ylabel("relative error")
    axB.set_title("the same error, two metrics")
    axB.grid(True, which="both", alpha=0.16)
    axB.set_yticklabels([])
    axB.annotate("score", xy=(0.023, 0.46), color=VERM, fontsize=8.0,
                 ha="left", va="bottom")
    axB.annotate(r"$\varepsilon_{\mathrm{PM}}$", xy=(0.023, 2.0e-2),
                 color=GREEN, fontsize=8.0, ha="left", va="top")

    # -- C: the risk statement ----------------------------------------------
    axC.loglog(t, 100.0 * d["excess_frac_med"], color=BLUE, lw=2.0, zorder=3)
    axC.set_xlim(t[0], t[-1])
    axC.set_xlabel(r"diffusion time $t$")
    axC.set_ylabel("excess MSE, % of floor")
    axC.set_title("what denoising costs")
    axC.grid(True, which="both", alpha=0.16)

    save(fig, FIG / "fig_laplace_metrics")


# ------------------------------------------------- 2. the measured closure error --
def _read_summary() -> dict[str, np.ndarray]:
    with open(EXP02) as fh:
        rows = list(csv.DictReader(fh))
    keys = rows[0].keys()
    return {k: np.array([float(r[k]) for r in rows]) for k in keys}


def fig_laplace_closure() -> None:
    # same sweep as fig_laplace_metrics, so every number in Sec. 7.5 comes
    # from one run; the frozen exp_02 CSV is the independent replication
    if not SWEEP_CSV.exists():
        run_sweep()
    with open(SWEEP_CSV) as fh:
        rows = list(csv.DictReader(fh))
    d = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    d = {"t": d["t"],
         "rel_score_error_median": d["score_med"],
         "rel_score_error_q10": d["score_q10"],
         "rel_score_error_q90": d["score_q90"],
         "grid_self_conv_median": d["grid_selfconv_med"],
         "cosine_median": d["cos_med"]}
    t = d["t"]

    fig, (axL, axR) = new_figure(1, 2, width=FULL, height=3.0)

    # -- left: the closure error against the reference budget ---------------
    axL.fill_between(t, d["rel_score_error_q10"], d["rel_score_error_q90"],
                     color=VERM, alpha=0.16, lw=0)
    axL.loglog(t, d["rel_score_error_median"], color=VERM, lw=2.0, zorder=3)
    axL.loglog(t, d["grid_self_conv_median"], color=BLUE, lw=1.6,
               ls=(0, (5, 2.5)), zorder=3)

    axL.set_xlim(t[0], t[-1])
    axL.set_ylim(1e-8, 3.0)
    axL.set_xlabel(r"diffusion time $t$")
    axL.set_ylabel("relative score error")
    axL.set_title("closure error against the reference budget")
    axL.grid(True, which="both", alpha=0.16)
    axL.annotate("Gaussian closure\n(median, 10-90 pct)", xy=(0.023, 3.0e-3),
                 color=VERM, fontsize=8.5, ha="left", va="top")
    # the reference curve runs across the lower half, so its label goes in the
    # empty bottom-left corner rather than on top of it
    axL.annotate("grid BP self-convergence", xy=(0.055, 5.5e-8),
                 color=BLUE, fontsize=8.5, ha="left", va="bottom")

    # -- right: magnitude error against direction error ---------------------
    axR.loglog(t, d["rel_score_error_median"], color=VERM, lw=2.0, zorder=3)
    axR.loglog(t, 1.0 - d["cosine_median"], color=GREEN, lw=2.0, zorder=3)

    axR.set_xlim(t[0], t[-1])
    axR.set_ylim(1e-7, 3.0)
    axR.set_xlabel(r"diffusion time $t$")
    axR.set_ylabel("median error")
    axR.set_title("magnitude fails before direction does")
    axR.grid(True, which="both", alpha=0.16)
    axR.annotate("relative magnitude", xy=(0.055, 0.62), color=VERM,
                 fontsize=8.5, ha="left", va="bottom")
    axR.annotate(r"$1-\cos$ angle", xy=(0.055, 1.5e-4), color=GREEN,
                 fontsize=8.5, ha="left", va="top")

    # The failure-probability threshold used to be marked here with a rule and
    # a label; both collided with the panel title. It is a table fact, and the
    # table states it.
    save(fig, FIG / "fig_laplace_closure")


ALL = {
    "laplace_nonlinear": fig_laplace_nonlinear,
    "laplace_closure": fig_laplace_closure,
    "laplace_metrics": fig_laplace_metrics,
    "sweep": run_sweep,
}

if __name__ == "__main__":
    for n in (sys.argv[1:] or list(ALL)):
        print(f"building {n} ...")
        ALL[n]()
