#!/usr/bin/env python3
"""Chapter 6 figure: the price of locality.

    fig_truncation.pdf  -- relative RMS score error of the two ways of going
                           local, against diffusion time and against the
                           locality budget.

Two truncations of the exact Gaussian score S = -Q_t x (the experiments
proposed in the 31:27 / 31:48 exchange with Jerome, implemented in
research/bp-from-scratch/):

  * truncate the ANSWER    -- band Q_t at |i-j| <= b;
  * truncate the ALGORITHM -- let BP messages travel at most r hops, which
    (a contiguous window of a stationary AR(1) chain being another one) is
    exact inference on the window x_{k-r..k+r}.

Both estimators are linear in x, so the RMS error over x ~ N(0, Sigma_t) is
closed form -- no sampling anywhere in this figure.

House style (sans-serif, text-block width, Okabe-Ito) from figstyle.py.

    /usr/local/bin/python3.12 tools/fig_ch06.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import BLUE, VERM, GREY, FULL, ramp, new_figure, save  # noqa: E402

THESIS = Path(__file__).resolve().parents[1]
FIG = THESIS / "figures"

ALPHA, L = 0.8, 40


# ----------------------------------------------------------------- algebra --
def ar1_cov(n: int, alpha: float) -> np.ndarray:
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return alpha ** d


def channel(t: float) -> tuple[float, float]:
    return float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))


def clean_precision(n: int, alpha: float) -> np.ndarray:
    return np.linalg.inv(ar1_cov(n, alpha))


def noisy_cov(n: int, alpha: float, t: float) -> np.ndarray:
    mu, D = channel(t)
    return mu * mu * ar1_cov(n, alpha) + D * np.eye(n)


def banded_score_matrix(n: int, alpha: float, t: float, b: int) -> np.ndarray:
    """Q_t with every entry at index distance > b zeroed."""
    Q = np.linalg.inv(noisy_cov(n, alpha, t))
    keep = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) <= b
    return Q * keep


def local_score_matrix(n: int, alpha: float, t: float, r: int) -> np.ndarray:
    """M_r with S_local = -M_r x: messages cut at r hops, i.e. exact inference
    on each window x_{k-r..k+r}."""
    mu, D = channel(t)
    C = np.zeros((n, n))
    for k in range(n):
        lo, hi = max(0, k - r), min(n - 1, k + r)
        w = hi - lo + 1
        J = (mu * mu / D) * np.eye(w) + clean_precision(w, alpha)
        C[k, lo:hi + 1] = ((mu / D) * np.linalg.inv(J))[k - lo]
    return (np.eye(n) - mu * C) / D


def rel_err(M: np.ndarray, n: int, alpha: float, t: float) -> float:
    """sqrt( E||(M - Q_t)x||^2 / E||Q_t x||^2 ) for x ~ N(0, Sigma_t)."""
    St = noisy_cov(n, alpha, t)
    Qt = np.linalg.inv(St)
    Dl = M - Qt
    return float(np.sqrt(max(np.trace(Dl @ St @ Dl.T), 0.0) / np.trace(Qt)))


# ------------------------------------------------------------------ figure --
def fig_truncation() -> None:
    fig, (axA, axB) = new_figure(1, 2, width=FULL, height=3.2)

    # ---- A: error against diffusion time -------------------------------
    ts = np.logspace(-2, 0.78, 46)
    radii = [0, 1, 2]
    cols = ramp(len(radii))
    # label each solid curve in the clear space just below it, at t = 0.03
    ylab = {0: 0.50, 1: 0.055, 2: 0.0042}

    for r, c in zip(radii, cols):
        e = [rel_err(local_score_matrix(L, ALPHA, t, r), L, ALPHA, t) for t in ts]
        axA.loglog(ts, e, "-", lw=1.9, color=c, zorder=3)
        axA.annotate(rf"$r={r}$", xy=(0.030, ylab[r]), color=c, fontsize=8.5,
                     ha="left", va="center")
    for b, c in zip(radii[1:], cols[1:]):
        e = [rel_err(banded_score_matrix(L, ALPHA, t, b), L, ALPHA, t) for t in ts]
        axA.loglog(ts, e, "--", lw=1.2, color=c, alpha=0.9, zorder=2)

    axA.set_xlabel(r"diffusion time $t$")
    axA.set_ylabel("relative RMS score error")
    axA.set_title("cost of locality across the diffusion")
    axA.grid(True, which="both", alpha=0.16)
    axA.set_xlim(ts[0], ts[-1])
    axA.set_ylim(1e-3, 1.6)

    hA = [axA.plot([], [], "-", lw=1.9, color=GREY)[0],
          axA.plot([], [], "--", lw=1.2, color=GREY)[0]]
    axA.legend(hA, ["truncate the algorithm ($r$ hops)",
                    "truncate the answer (band $b$)"],
               loc="lower left", fontsize=7.6, frameon=False,
               handlelength=1.9, borderaxespad=0.2)

    # ---- B: error against the locality budget --------------------------
    budgets = list(range(0, 7))
    tsB = [0.05, 0.3, 1.0]
    colsB = ramp(len(tsB))

    for t, c in zip(tsB, colsB):
        er = [rel_err(local_score_matrix(L, ALPHA, t, r), L, ALPHA, t)
              for r in budgets]
        eb = [rel_err(banded_score_matrix(L, ALPHA, t, b), L, ALPHA, t)
              for b in budgets]
        axB.semilogy(budgets, er, "o-", ms=3.4, lw=1.7, color=c, zorder=3)
        axB.semilogy(budgets, eb, "s--", ms=3.2, lw=1.1, color=c, alpha=0.85,
                     zorder=2)
        axB.annotate(rf"$t={t:g}$", xy=(budgets[-1], er[-1]), xytext=(4, 0),
                     textcoords="offset points", color=c, fontsize=8.5,
                     va="center", annotation_clip=False)

    axB.set_xlabel(r"locality budget  ($r$ hops  /  band $b$)")
    axB.set_ylabel("relative RMS score error")
    axB.set_title("each extra hop buys a constant factor")
    axB.grid(True, which="both", alpha=0.16)
    axB.set_xticks(budgets)

    hB = [axB.plot([], [], "o-", ms=3.4, lw=1.7, color=GREY)[0],
          axB.plot([], [], "s--", ms=3.2, lw=1.1, color=GREY)[0]]
    axB.legend(hB, ["$r$ hops", "band $b$"], loc="lower left", fontsize=7.6,
               frameon=False, handlelength=1.9, borderaxespad=0.2)

    save(fig, FIG / "fig_truncation")


# ------------------------------------------------------- numbers for the text --
def numbers() -> None:
    print(f"\nrelative RMS score error, L={L}, alpha={ALPHA}\n")
    print(f"{'t':>6} | {'r=0':>7} {'r=1':>7} {'r=2':>7} | "
          f"{'b=1':>7} {'b=2':>7} | {'1-r1/r0':>8} {'b1/r1':>6}")
    for t in (0.05, 0.3, 1.0, 3.0):
        r0 = rel_err(local_score_matrix(L, ALPHA, t, 0), L, ALPHA, t)
        r1 = rel_err(local_score_matrix(L, ALPHA, t, 1), L, ALPHA, t)
        r2 = rel_err(local_score_matrix(L, ALPHA, t, 2), L, ALPHA, t)
        b1 = rel_err(banded_score_matrix(L, ALPHA, t, 1), L, ALPHA, t)
        b2 = rel_err(banded_score_matrix(L, ALPHA, t, 2), L, ALPHA, t)
        print(f"{t:>6} | {r0:>7.4f} {r1:>7.4f} {r2:>7.4f} | "
              f"{b1:>7.4f} {b2:>7.4f} | {1 - r1 / r0:>8.3f} {b1 / r1:>6.2f}")

    print("\nper-hop attenuation err(r+1)/err(r), measured:")
    for t in (0.05, 0.3, 1.0, 2.0):
        e = [rel_err(local_score_matrix(L, ALPHA, t, r), L, ALPHA, t)
             for r in range(1, 7)]
        ratios = [e[i + 1] / e[i] for i in range(len(e) - 1)]
        print(f"  t={t:<5} " + " ".join(f"{x:.3f}" for x in ratios)
              + f"   mean {np.mean(ratios):.3f}")


if __name__ == "__main__":
    fig_truncation()
    numbers()
