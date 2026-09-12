#!/usr/bin/env python3
"""Chapter 5 figures: the sparsity lifecycle of the Gaussian chain.

    fig_sparsity_lifecycle.pdf  -- |Sigma_t| (dense) and |Q_t| (tridiagonal at
                                   t=0) as magnitude heatmaps across diffusion
                                   time; the precision band fills, then fades.
    fig_markov_survival.pdf     -- Frobenius lifecycle of the precision's
                                   inter-frame coupling: the nearest-neighbour
                                   (Markov) mass that survives, the
                                   off-tridiagonal mass that fills in, the total
                                   coupling, all vs t.

House style (sans-serif, text-block width, Okabe-Ito) comes from figstyle.py.

    /usr/local/bin/python3.12 tools/fig_ch05.py [name ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import BLUE, VERM, GREEN, GREY, FULL, ramp, new_figure, save  # noqa: E402

THESIS = Path(__file__).resolve().parents[1]
FIG = THESIS / "figures"


# ----------------------------------------------------------------- algebra --
def ar1_cov(L: int, alpha: float) -> np.ndarray:
    d = np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
    return alpha ** d


def dt_noise(t: float) -> float:
    return 1.0 - np.exp(-2.0 * t)


def sigma_t(S0: np.ndarray, t: float) -> np.ndarray:
    return np.exp(-2.0 * t) * S0 + dt_noise(t) * np.eye(S0.shape[0])


def precision_t(S0: np.ndarray, t: float) -> np.ndarray:
    return np.linalg.inv(sigma_t(S0, t))


def dist_mass(M: np.ndarray, dlo: int, dhi: int) -> float:
    """Frobenius norm of the entries at index distance dlo..dhi."""
    idx = np.abs(np.subtract.outer(np.arange(M.shape[0]), np.arange(M.shape[1])))
    sel = (idx >= dlo) & (idx <= dhi)
    return float(np.sqrt(np.sum(M[sel] ** 2)))


# ---------------------------------------------------- 1. sparsity lifecycle --
def fig_sparsity_lifecycle(L: int = 24, alpha: float = 0.85) -> None:
    ts = [0.0, 0.1, 0.3, 0.7, 1.8]
    S0 = ar1_cov(L, alpha)

    fig, axes = new_figure(2, len(ts), width=FULL, height=3.3)
    im_s = im_q = None
    for j, t in enumerate(ts):
        S = sigma_t(S0, t)
        Q = precision_t(S0, t)
        gamma = np.inf if t == 0 else np.exp(-2 * t) / dt_noise(t)

        im_s = axes[0, j].imshow(np.abs(S), norm=LogNorm(vmin=1e-2, vmax=1.0),
                                 cmap="Greys", interpolation="nearest")
        im_q = axes[1, j].imshow(np.abs(Q), norm=LogNorm(vmin=1e-3, vmax=5.0),
                                 cmap="Greys", interpolation="nearest")

        gtxt = r"$\gamma_t=\infty$" if t == 0 else rf"$\gamma_t={gamma:.2f}$"
        axes[0, j].set_title(rf"$t={t:.2f}$" + "\n" + gtxt, fontsize=8.5)
        for ax in (axes[0, j], axes[1, j]):
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    axes[0, 0].set_ylabel(r"$|\Sigma_t|$", fontsize=10.5)
    axes[1, 0].set_ylabel(r"$|Q_t| = |\Sigma_t^{-1}|$", fontsize=10.5)
    fig.colorbar(im_s, ax=axes[0, :].tolist(), location="right",
                 shrink=0.82, pad=0.015)
    fig.colorbar(im_q, ax=axes[1, :].tolist(), location="right",
                 shrink=0.82, pad=0.015)
    save(fig, FIG / "fig_sparsity_lifecycle")


# ------------------------------------------------------ 2. markov survival --
def fig_markov_survival(L: int = 48, alpha: float = 0.9) -> None:
    ts = np.logspace(-3.0, 0.62, 340)
    S0 = ar1_cov(L, alpha)

    ref = dist_mass(precision_t(S0, 0.0), 1, 1)      # Q_0 nearest-neighbour mass
    nn, off = [], []
    for t in ts:
        Q = precision_t(S0, t)
        nn.append(dist_mass(Q, 1, 1) / ref)
        off.append(dist_mass(Q, 2, L) / ref)
    nn, off = map(np.array, (nn, off))

    k = int(np.argmax(off))
    t_star = ts[k]
    i0 = int(np.argmin(np.abs(ts - 1.0)))
    asy = nn[i0] * np.exp(-2.0 * (ts - ts[i0]))      # e^{-2t} guide on the tail

    fig, ax = new_figure(1, 1, width=FULL, height=3.0)

    ax.axvspan(ts[0], t_star, color=GREY, alpha=0.07, lw=0)
    ax.axvline(t_star, color=GREY, ls=":", lw=1.0)
    ax.plot(ts, nn, color=GREEN, lw=1.9)
    ax.fill_between(ts, 0, off, color=BLUE, alpha=0.15, lw=0)
    ax.plot(ts, off, color=BLUE, lw=1.9)
    ax.plot([t_star], [off[k]], "o", ms=4.5, color=BLUE, mec="white", mew=0.5)

    guide = ts >= 0.9
    ax.plot(ts[guide], asy[guide], color=GREY, lw=0.9, ls=":")

    ax.set_xscale("log")
    ax.set_xlabel(r"diffusion time $t$")
    ax.set_ylabel(r"coupling mass  (rel. to $Q_0$ n.n. coupling)")
    ax.set_ylim(0, 1.26)
    ax.set_xlim(ts[0], ts[-1])
    ax.grid(True, which="both", alpha=0.16)

    # labels placed in the empty regions: top-right for the green curve, the
    # wedge between the curves for the blue one, above the axes for t*.
    ax.annotate("nearest-neighbour (Markov)\ncoupling that survives",
                xy=(0.13, 0.72), ha="left", va="center",
                color=GREEN, fontsize=8.5)
    ax.annotate(rf"off-tridiagonal mass" "\n" rf"(peak $\approx {off[k]:.2f}$)",
                xy=(6e-3, 0.32), ha="left", va="center",
                color=BLUE, fontsize=8.5)
    ax.annotate(r"$t^\star$: densest precision", xy=(t_star * 1.18, 1.17),
                ha="left", va="center", fontsize=8.5, color=GREY,
                annotation_clip=False)
    ax.annotate(r"$\propto e^{-2t}$", xy=(ts[-1], asy[-1]), xytext=(-2, 10),
                textcoords="offset points", ha="right", fontsize=8.5,
                color=GREY)
    save(fig, FIG / "fig_markov_survival")


# ------------------------------------------------- 3. eigen-axis relaxation --
def fig_eigen_relax(L: int = 24, alpha: float = 0.7) -> None:
    """The spectral picture of Sec. 5.6.3: in the eigenbasis of Sigma_0 the noisy
    covariance is an ellipsoid with fixed axis directions whose semi-axes
    sqrt(lambda_i(t)) all relax to 1 -- anisotropic at t=0, the unit sphere at
    t=infty. Left: two eigen-modes drawn as a 2-D covariance ellipse at several
    t. Right: the full eigenvalue flow lambda_i(t) = e^{-2t} omega_i + Delta_t."""
    S0 = ar1_cov(L, alpha)
    w = np.sort(np.linalg.eigvalsh(S0))[::-1]          # omega_i, descending
    w_hi, w_lo = w[0], w[-1]

    fig, (axL, axR) = new_figure(1, 2, width=FULL, height=3.1)

    # -- left: the covariance ellipse in the fixed eigenbasis ---------------
    phi = np.deg2rad(27.0)                              # arbitrary fixed rotation U
    e1 = np.array([np.cos(phi), np.sin(phi)])
    e2 = np.array([-np.sin(phi), np.cos(phi)])
    s = np.linspace(0, 2 * np.pi, 400)

    ts = [0.0, 0.15, 0.4, 0.9]
    cols = ramp(len(ts))
    R = 1.05 * np.sqrt(w_hi)
    for uvec, name in ((e1, r"$u_1$"), (e2, r"$u_2$")):
        axL.plot([-R * uvec[0], R * uvec[0]], [-R * uvec[1], R * uvec[1]],
                 color=GREY, lw=0.7, ls="--", zorder=1)
        axL.annotate(name, xy=(1.12 * R * uvec[0], 1.12 * R * uvec[1]),
                     color=GREY, fontsize=9, ha="center", va="center",
                     annotation_clip=False)

    for t, c in zip(ts, cols):
        a1 = np.sqrt(np.exp(-2 * t) * w_hi + dt_noise(t))
        a2 = np.sqrt(np.exp(-2 * t) * w_lo + dt_noise(t))
        xy = a1 * np.cos(s)[:, None] * e1 + a2 * np.sin(s)[:, None] * e2
        lw = 2.1 if t == 0.0 else 1.5
        axL.plot(xy[:, 0], xy[:, 1], color=c, lw=lw, zorder=3,
                 label=rf"$t={t:.2f}$")
    circ = np.cos(s)[:, None] * e1 + np.sin(s)[:, None] * e2
    axL.plot(circ[:, 0], circ[:, 1], color="black", lw=1.6, ls=(0, (4, 3)),
             zorder=4, label=r"$t\to\infty$")

    axL.legend(loc="lower left", fontsize=7.6, frameon=False,
               handlelength=1.4, labelspacing=0.35, borderaxespad=0.1)
    axL.set_aspect("equal")
    axL.set_xlim(-R * 1.18, R * 1.18); axL.set_ylim(-R * 1.18, R * 1.18)
    axL.set_xticks([]); axL.set_yticks([]); axL.grid(False)
    for sp in axL.spines.values():
        sp.set_visible(False)
    axL.set_title(r"covariance ellipse, eigenbasis of $\Sigma_0$")

    # -- right: the eigenvalue flow ---------------------------------------
    tt = np.linspace(0, 2.0, 200)
    for wi in w:
        axR.plot(tt, np.exp(-2 * tt) * wi + (1 - np.exp(-2 * tt)),
                 color=GREY, lw=0.8, alpha=0.35)
    # Each label sits INSIDE the axes next to its curve's start, on a white
    # backing; the y-limits leave margin above omega_max and below omega_min
    # so neither label escapes the plot area or reaches the title.
    for wi, c, lab, dy, va in (
            (w_hi, VERM, r"$\omega_{\max}$", 4, "bottom"),
            (w[np.argmin(np.abs(w - 1))], GREEN, r"$\omega_i\approx1$", 9, "bottom"),
            (w_lo, BLUE, r"$\omega_{\min}$", -4, "top")):
        y = np.exp(-2 * tt) * wi + (1 - np.exp(-2 * tt))
        axR.plot(tt, y, color=c, lw=1.9)
        axR.annotate(lab, xy=(0.02, wi), xytext=(3, dy),
                     textcoords="offset points",
                     color=c, fontsize=8.5, va=va, ha="left",
                     bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.4))
    axR.axhline(1.0, color="black", lw=1.0, ls=(0, (4, 3)))
    axR.set_yscale("log")
    axR.set_ylim(w_lo * 0.72, w_hi * 1.55)
    axR.set_xlim(0, 2.0)
    axR.set_xlabel(r"diffusion time $t$")
    axR.set_ylabel(r"eigenvalue $\lambda_i(t) = e^{-2t}\omega_i + \Delta_t$")
    axR.set_title(r"every eigenvalue relaxes to $1$")
    axR.grid(True, which="both", alpha=0.16)
    axR.annotate(r"$\omega_i>1$: descend to $1$", xy=(0.85, 3.1), color=GREY,
                 fontsize=8,
                 bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.4))
    axR.annotate(r"$\omega_i<1$: climb to $1$", xy=(0.85, 0.42), color=GREY,
                 fontsize=8,
                 bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.4))

    save(fig, FIG / "fig_eigen_relax")


ALL = {
    "sparsity_lifecycle": fig_sparsity_lifecycle,
    "markov_survival": fig_markov_survival,
    "eigen_relax": fig_eigen_relax,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    for n in names:
        print(f"building {n} ...")
        ALL[n]()
