#!/usr/bin/env python3
"""Rebuild the thesis figures into ``thesis/figures``.

Writes only inside the thesis folder. any shared figures folder is left alone
so the paper, the workshop note and the compendium keep the figures they were
built against.

    python3 tools/make_thesis_figures.py [name ...]

With no arguments it rebuilds everything it knows how to build.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import (BLUE, FULL, GREEN, GREY, ORANGE, VERM,  # noqa: E402
                      label_lines, legend_above, new_figure, ramp, save)

THESIS = Path(__file__).resolve().parents[1]
FIG = THESIS / "figures"
OUT = THESIS.parents[1] / "research" / "nongaussian-bp" / "outputs"


def read(rel: str) -> list[dict]:
    p = OUT / rel
    if not p.exists():
        raise FileNotFoundError(f"missing committed output: {p}")
    with p.open() as fh:
        return list(csv.DictReader(fh))


def col(rows, name, cast=float):
    return [cast(r[name]) for r in rows]


# ===========================================================================
# Chapter 2 — the three scores, memorisation, and the speciation cascade
# ===========================================================================

def fig_forward_corruption() -> None:
    """One clean AR(1) trajectory, and the same trajectory at four noise levels.

    The reader should see the problem before reading a description of it.
    """
    rng = np.random.default_rng(0)
    alpha, L = 0.85, 60
    a = np.empty(L)
    a[0] = rng.normal(0, 1)
    for k in range(1, L):
        a[k] = alpha * a[k - 1] + np.sqrt(1 - alpha**2) * rng.normal()

    times = [0.0, 0.25, 0.75, 2.0]
    fig, axes = new_figure(1, 4, width=FULL, height=1.85, sharey=True)
    cols = ramp(len(times))
    for ax, t, c in zip(axes, times, cols):
        mu, dt = np.exp(-t), 1 - np.exp(-2 * t)
        x = mu * a + np.sqrt(dt) * rng.normal(size=L)
        ax.plot(a, color=GREY, lw=0.8, alpha=0.45)
        ax.plot(x, color=c, lw=1.3)
        ax.set_title(rf"$t={t:g}$")
        ax.set_xlabel("site $k$")
        ax.set_xticks([0, 30, 60])
    axes[0].set_ylabel("$x_k$")
    axes[0].annotate("clean chain", xy=(0.04, 0.06), xycoords="axes fraction",
                     fontsize=7.5, color=GREY)
    save(fig, FIG / "fig_forward_corruption")


# ===========================================================================
# Rebuilds of the two figures whose legends collided
# ===========================================================================

def fig_marginal_blindness() -> None:
    """Same one-frame marginals, different joint score: the L=2 Gaussian pair.

    Left: alpha=0, no coupling, purely radial score. Right: alpha=0.85, the
    joint score tilts because of the cross term r_t/(1-r_t^2). Both panels
    have identical N(0,1) marginals at every t -- the point of the figure.
    """
    t = 0.15
    alphas = [0.0, 0.85]
    titles = [r"no coupling: $\alpha=0$", r"coupling: $\alpha=0.85$"]

    xs = np.linspace(-3, 3, 300)
    X0, X1 = np.meshgrid(xs, xs)
    qx = np.linspace(-2.4, 2.4, 11)
    QX0, QX1 = np.meshgrid(qx, qx)

    # Row 0 is the two plots; row 1 is a caption strip with no axes at all,
    # so the score formula sits in its own reserved space, never on the data.
    fig, axes = new_figure(2, 2, width=FULL, height=3.6,
                           height_ratios=[5, 1])
    plots, capaxes = axes[0], axes[1]
    formulas = [r"$s_{t,0}=-x_0$  (no $x_1$)",
                r"$s_{t,0}=-\dfrac{x_0-r_tx_1}{1-r_t^2}$  (uses $x_1$)"]
    colors = [GREY, VERM]
    for ax, cap, alpha, title, formula, colr in zip(
            plots, capaxes, alphas, titles, formulas, colors):
        r = alpha * np.exp(-2 * t)
        dens = (np.exp(-(X0**2 - 2*r*X0*X1 + X1**2) / (2*(1 - r**2)))
                / (2*np.pi*np.sqrt(1 - r**2)))
        ax.contour(X0, X1, dens, levels=6, colors=GREY, linewidths=0.7,
                   alpha=0.75)
        S0 = -(QX0 - r*QX1) / (1 - r**2)
        S1 = -(QX1 - r*QX0) / (1 - r**2)
        norm = np.hypot(S0, S1)
        norm[norm == 0] = 1
        ax.quiver(QX0, QX1, S0/norm, S1/norm, color=BLUE, scale=18,
                  width=0.006, alpha=0.9)
        ax.set_title(rf"{title}, $r_t={r:.2f}$")
        ax.set_xlabel("$x_0$")
        ax.set_aspect("equal")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        cap.axis("off")
        cap.text(0.5, 0.5, formula, ha="center", va="center",
                 fontsize=10, color=colr, transform=cap.transAxes)
    plots[0].set_ylabel("$x_1$")
    save(fig, FIG / "fig_marginal_blindness")


# ===========================================================================
# Chapter 4 -- the rotating ring
# ===========================================================================

def _simulate_ring(rng, L=20, kappa=2.0, Dr=0.015, Dtheta=0.005, omega=1.0,
                   h=2 * np.pi / 20, substeps=50, r0=1.0, theta0=0.0):
    """One clean trajectory of the polar model, Euler--Maruyama with fine
    substeps, subsampled at the L recording times u_k = kh."""
    dt = h / substeps
    r, theta = r0, theta0
    rs, thetas = [r], [theta]
    for _ in range(L - 1):
        for _ in range(substeps):
            r = r - kappa * (r - 1.0) * dt + np.sqrt(2 * Dr * dt) * rng.normal()
            theta = theta + omega * dt + np.sqrt(2 * Dtheta * dt) * rng.normal()
        rs.append(r)
        thetas.append(theta)
    rs, thetas = np.array(rs), np.array(thetas)
    return np.stack([rs * np.cos(thetas), rs * np.sin(thetas)], axis=1)  # (L,2)


def fig_ring_model_dynamics() -> None:
    """The confining potential/force, and one clean trajectory of the polar
    model. Two panels, no figure text that isn't a plain axis label -- the
    frame-index colour ramp and the start/end markers carry the story."""
    kappa = 2.0
    rng = np.random.default_rng(0)
    traj = _simulate_ring(rng, L=20, kappa=kappa)
    L = len(traj)
    cols = ramp(L, "viridis")

    fig, (ax0, ax1) = new_figure(1, 2, width=FULL, height=3.0)

    r = np.linspace(0.4, 1.6, 200)
    V = 0.5 * kappa * (r - 1) ** 2
    F = -kappa * (r - 1)
    ax0.axvline(1.0, color=GREY, lw=0.7, ls=":", alpha=0.6)
    ax0.plot(r, V, color=BLUE, lw=1.7)
    ax0.plot(r, F, color=VERM, lw=1.5, ls="--")
    ax0.axhline(0, color=GREY, lw=0.5, alpha=0.4)
    ax0.set_xlabel("$r$")
    ax0.set_title("radial confinement")
    label_lines(ax0, [(r"$V(r)$", V[-1], BLUE), (r"$-V'(r)$", F[-1], VERM)])

    ax1.plot(traj[:, 0], traj[:, 1], color=GREY, lw=0.8, alpha=0.6, zorder=1)
    ax1.scatter(traj[:, 0], traj[:, 1], c=cols, s=22, zorder=2,
               edgecolor="none")
    ax1.plot(*traj[0], marker="o", ms=8, mfc="none", mec="black", mew=1.3,
             zorder=3)
    ax1.plot(*traj[-1], marker="s", ms=7, mfc="none", mec="black", mew=1.3,
             zorder=3)
    # Rotation-sense arrow: offset radially outward from the ring, at a point
    # with open space around it, thick and dark enough to read at a glance.
    p = 3
    centre = traj.mean(axis=0)
    outward = (traj[p] - centre) / np.linalg.norm(traj[p] - centre)
    tangent = traj[p + 1] - traj[p - 1]
    tangent = tangent / np.linalg.norm(tangent)
    base = traj[p] + 0.16 * outward
    ax1.annotate("", xy=base + 0.3 * tangent, xytext=base - 0.1 * tangent,
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2,
                                mutation_scale=18), annotation_clip=False)
    ax1.set_aspect("equal")
    ax1.margins(0.12)
    ax1.set_xlabel("$a_{u,1}$")
    ax1.set_ylabel("$a_{u,2}$")
    ax1.set_title("one clean trajectory")
    save(fig, FIG / "fig_ring_model_dynamics")


def fig_ring_model_channel() -> None:
    """The same clean datum, then after the OU channel at t=0.5. Frame
    colour is shared between the two panels so a reader can track one frame
    across the corruption."""
    rng = np.random.default_rng(0)
    traj = _simulate_ring(rng, L=20)
    L = len(traj)
    cols = ramp(L, "viridis")
    t = 0.5
    dt_noise = 1 - np.exp(-2 * t)
    noisy = np.exp(-t) * traj + np.sqrt(dt_noise) * rng.normal(size=traj.shape)

    fig, (ax0, ax1) = new_figure(1, 2, width=FULL, height=3.0)
    for ax, data, title in [(ax0, traj, "clean datum, $t=0$"),
                             (ax1, noisy, "after the channel, $t=0.5$")]:
        ax.plot(data[:, 0], data[:, 1], color=GREY, lw=0.8, alpha=0.5,
               zorder=1)
        ax.scatter(data[:, 0], data[:, 1], c=cols, s=22, zorder=2,
                  edgecolor="none")
        ax.set_aspect("equal")
        ax.set_title(title)
    ax0.set_xlabel("$a_{u,1}$"); ax0.set_ylabel("$a_{u,2}$")
    ax1.set_xlabel("$x_{u,1}$"); ax1.set_ylabel("$x_{u,2}$")
    save(fig, FIG / "fig_ring_model_channel")


def fig_ring_gauge() -> None:
    """The rotation as a linear map (panel a), and three surrogate
    trajectories before/after the gauge U_psi (panels b, c). Panel a carries
    only the two vectors, the angle arc and their labels -- the matrix
    R_psi and the orthogonality identity belong in the caption, not drawn
    into the axes."""
    psi = 2 * np.pi / 14
    L, sigma = 14, 0.16
    rng = np.random.default_rng(1)

    fig, (ax0, ax1, ax2) = new_figure(1, 3, width=FULL, height=3.0)

    # --- (a) the rotation, as two vectors and the angle between them -----
    ang = np.linspace(0, 2 * np.pi, 200)
    ax0.plot(np.cos(ang), np.sin(ang), color=GREY, lw=0.6, ls=":")
    theta0 = 0.55
    z = np.array([np.cos(theta0), np.sin(theta0)])
    Rpsi_z = np.array([np.cos(theta0 + psi), np.sin(theta0 + psi)])
    for v, colr, lab in [(z, BLUE, "$z$"), (Rpsi_z, VERM, r"$R_\psi z$")]:
        ax0.annotate("", xy=v, xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=colr, lw=1.8))
        ax0.annotate(lab, xy=v, xytext=v * 1.18, color=colr, ha="center",
                     va="center", fontsize=9.5)
    arc = np.linspace(theta0, theta0 + psi, 30)
    ax0.plot(0.32 * np.cos(arc), 0.32 * np.sin(arc), color=GREY, lw=1.0)
    ax0.annotate(r"$\psi$", xy=(0.44 * np.cos(theta0 + psi / 2),
                              0.44 * np.sin(theta0 + psi / 2)),
                 color=GREY, ha="center", va="center", fontsize=9)
    ax0.set_xlim(-0.2, 1.3); ax0.set_ylim(-0.2, 1.3)
    ax0.set_aspect("equal")
    ax0.set_title("one rotation")

    # --- (b), (c) three surrogate trajectories, lab frame and co-rotating -
    colors = [BLUE, VERM, ORANGE]
    Rpsi = np.array([[np.cos(psi), -np.sin(psi)],
                     [np.sin(psi), np.cos(psi)]])
    for ax, title in [(ax1, "laboratory frame $z_k$"),
                      (ax2, "co-rotating frame $y_k$")]:
        ax.plot(np.cos(ang), np.sin(ang), color=GREY, lw=0.6, ls=":")
        ax.set_aspect("equal")
        ax.set_title(title)
    for colr in colors:
        r0 = 1.0 + 0.05 * rng.normal()
        a0 = rng.uniform(0, 2 * np.pi)
        z0 = r0 * np.array([np.cos(a0), np.sin(a0)])
        zs = [z0]
        for _ in range(L - 1):
            zs.append(Rpsi @ zs[-1] + sigma * rng.normal(size=2))
        zs = np.array(zs)
        ys = np.array([np.linalg.matrix_power(Rpsi.T, k) @ zs[k]
                       for k in range(L)])
        for ax, data in [(ax1, zs), (ax2, ys)]:
            ax.plot(data[:, 0], data[:, 1], color=colr, lw=1.1, alpha=0.85,
                   marker="o", ms=3.5)
            ax.plot(*data[0], marker="o", ms=8, mfc="none", mec=colr, mew=1.4)
    save(fig, FIG / "fig_ring_gauge")


BUILDERS = {
    "fig_forward_corruption": fig_forward_corruption,
    "fig_ring_model_dynamics": fig_ring_model_dynamics,
    "fig_ring_model_channel": fig_ring_model_channel,
    "fig_ring_gauge": fig_ring_gauge,
    "fig_marginal_blindness": fig_marginal_blindness,
    # fig_em_diagnostics is built by tools/fig_ch08.py, and
    # fig_nonmarkov and fig_screening by tools/fig_ch10.py; the
    # earlier builders that lived here drew superseded versions of both, and
    # a bare run of this script must not overwrite the current ones.
}


def main(argv: list[str]) -> int:
    names = argv[1:] or list(BUILDERS)
    bad = [n for n in names if n not in BUILDERS]
    if bad:
        print(f"unknown figure(s): {', '.join(bad)}", file=sys.stderr)
        print(f"known: {', '.join(BUILDERS)}", file=sys.stderr)
        return 2
    for n in names:
        print(n)
        BUILDERS[n]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
