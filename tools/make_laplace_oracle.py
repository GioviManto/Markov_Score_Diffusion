#!/usr/bin/env python3
"""The Laplace chain's own radius-r local oracle, measured.

WHY THIS EXISTS. Section 10.4.1 anchors the receptive-field question on
W(r), the exact windowed-inference error of the covariance-matched GAUSSIAN
chain (closed form, eq:em-oracle). The experiment, however, runs on a Laplace
chain, and the text asserted that the Laplace chain's own local oracle is "a
different (and larger) quantity" -- an ordering of two Bayes risks that no
theorem in the thesis supplies. The direction matters: the chapter subtracts
W(2) from the measured window-head risk and reads the difference as what
remains winnable inside the radius-2 information set, which is an upper bound
only if the Laplace oracle is at least W(2).

So measure it. For each site k the radius-r Laplace oracle is the exact
posterior mean E[a_k | x_{k-r..k+r}] under the Laplace chain restricted to
that window, which grid BP computes to the same accuracy as the chapter's
reference score. The window's prior is the TRUE marginal law of its first
state -- obtained by propagating p_0 = N(0,1) forward through the Laplace
transition on the same grid -- and not a refreshed N(0,1): the chain is
covariance- but not strictly stationary (Sec. 7.2), so that distinction is
real, and getting it wrong would bias exactly the boundary windows.

The estimand is the pooled relative score error of eq:em-oracle, with the
closed-form traces replaced by Monte Carlo over the chain's own law:

    W_L(r) = sqrt( sum_t E|| S_win(x,t) - S_true(x,t) ||^2
                 / sum_t E|| S_true(x,t) ||^2 ),

S_true being the exact full-chain Laplace score. The Gaussian W(r) is
recomputed here from the same schedule so the two columns are comparable, and
the seed is fixed so the emitted macros are reproducible.

    python3 tools/make_laplace_oracle.py [n_draws]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THESIS = Path(__file__).resolve().parents[1]
OUT = THESIS / "sections" / "laplace-oracle-numbers.tex"

RHO = 0.85
NSITES = 32
SCHEDULE = [0.05, 0.0725, 0.1053, 0.1527, 0.2216, 0.3215,
            0.4665, 0.6769, 0.9821, 1.4250, 2.0676, 3.0]
RADII = [0, 1, 2, 4, 8]

# The chapter's validated reference resolution (Sec. 7.4.3).
A_HALF, NGRID = 8.0, 401


def channel(t: float) -> tuple[float, float]:
    return float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))


def laplace_scale(alpha: float) -> float:
    """2 b^2 = 1 - alpha^2, so every state has unit variance."""
    return float(np.sqrt((1.0 - alpha ** 2) / 2.0))


def ar1_cov(n: int, alpha: float) -> np.ndarray:
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return alpha ** d


# ------------------------------------------------------------------ grid --
def grid() -> tuple[np.ndarray, float]:
    g = np.linspace(-A_HALF, A_HALF, NGRID)
    return g, float(g[1] - g[0])


def transition(g: np.ndarray, alpha: float) -> np.ndarray:
    """T[i, j] = M(a_i | a_j) for the Laplace innovation."""
    b = laplace_scale(alpha)
    return np.exp(-np.abs(g[:, None] - alpha * g[None, :]) / b) / (2.0 * b)


def marginals(g: np.ndarray, h: float, T: np.ndarray, n: int) -> np.ndarray:
    """p_j(a) for j = 0..n-1: the chain's true state marginals on the grid."""
    p = np.empty((n, g.size))
    m = np.exp(-g ** 2 / 2.0)
    p[0] = m / (m.sum() * h)
    for j in range(1, n):
        m = (T @ p[j - 1]) * h
        p[j] = m / (m.sum() * h)
    return p


def posterior_means(xb: np.ndarray, g: np.ndarray, h: float, T: np.ndarray,
                    p_init: np.ndarray, t: float) -> np.ndarray:
    """E[a_i | x] at every site, for a batch of observation windows.

    xb is (B, n). p_init is the prior density of the FIRST state of the
    window. Returns (B, n). One forward-backward per batch, vectorised over
    the batch dimension -- the per-site recomputation of the chapter's
    plotting helper would multiply the cost by n for no gain.
    """
    B, n = xb.shape
    mu, D = channel(t)
    # ell[b, i, :] = N(x_bi ; mu g, D), scaled per row for stability.
    z = xb[:, :, None] - mu * g[None, None, :]
    ell = np.exp(-(z ** 2) / (2.0 * D) + (z ** 2).min(axis=2, keepdims=True) / (2.0 * D))

    fwd = np.empty((B, n, g.size))
    m = p_init[None, :] * ell[:, 0, :]
    fwd[:, 0, :] = m / (m.sum(axis=1, keepdims=True) * h)
    for i in range(1, n):
        m = (fwd[:, i - 1, :] @ T.T) * h * ell[:, i, :]
        fwd[:, i, :] = m / (m.sum(axis=1, keepdims=True) * h)

    bwd = np.ones((B, n, g.size))
    for i in range(n - 2, -1, -1):
        m = ((ell[:, i + 1, :] * bwd[:, i + 1, :]) @ T) * h
        bwd[:, i, :] = m / (m.max(axis=1, keepdims=True) + 1e-300)

    bel = fwd * bwd
    return (bel * g[None, None, :]).sum(axis=2) / bel.sum(axis=2)


def score_from_means(means: np.ndarray, x: np.ndarray, t: float) -> np.ndarray:
    """Tweedie: S = (mu E[a|.] - x) / Delta_t."""
    mu, D = channel(t)
    return (mu * means - x) / D


def sample_chain(rng, n_draws: int, alpha: float, n: int) -> np.ndarray:
    b = laplace_scale(alpha)
    a = np.empty((n_draws, n))
    a[:, 0] = rng.normal(size=n_draws)
    for k in range(1, n):
        a[:, k] = alpha * a[:, k - 1] + rng.laplace(0.0, b, size=n_draws)
    return a


# --------------------------------------------------- the Gaussian anchor --
def gaussian_oracle(radii) -> np.ndarray:
    """W(r) of eq:em-oracle, recomputed here so both columns share a schedule."""
    S0 = ar1_cov(NSITES, RHO)
    out = []
    for r in radii:
        num = den = 0.0
        for t in SCHEDULE:
            al, de = channel(t)
            St = al * al * S0 + de * np.eye(NSITES)
            Qt = np.linalg.inv(St)
            M = np.zeros((NSITES, NSITES))
            for k in range(NSITES):
                lo, hi = max(0, k - r), min(NSITES, k + r + 1)
                Jw = np.linalg.inv(S0[lo:hi, lo:hi]) + (al * al / de) * np.eye(hi - lo)
                M[k, lo:hi] = (al / de) * np.linalg.solve(Jw, np.eye(hi - lo)[:, k - lo])
            Dl = (al * M - np.eye(NSITES)) / de + Qt
            num += float(np.trace(Dl @ St @ Dl.T))
            den += float(np.trace(Qt))
        out.append(float(np.sqrt(num / den)))
    return np.array(out)


# ------------------------------------------------------ the Laplace one --
def laplace_oracle(n_draws: int, seed: int = 20260912):
    rng = np.random.default_rng(seed)
    g, h = grid()
    T = transition(g, RHO)
    pmarg = marginals(g, h, T, NSITES)

    a = sample_chain(rng, n_draws, RHO, NSITES)
    # Per-DRAW accumulators: the draw (a whole chain) is the independent unit,
    # so the bootstrap below resamples draws, not sites -- sites within a chain
    # are dependent and resampling them would understate the interval.
    num = {r: np.zeros(n_draws) for r in RADII}
    den = np.zeros(n_draws)

    for t in SCHEDULE:
        mu, D = channel(t)
        x = mu * a + np.sqrt(D) * rng.normal(size=a.shape)

        s_true = score_from_means(
            posterior_means(x, g, h, T, pmarg[0], t), x, t)
        den += (s_true ** 2).sum(axis=1)

        for r in RADII:
            s_win = np.empty_like(s_true)
            for k in range(NSITES):
                lo, hi = max(0, k - r), min(NSITES, k + r + 1)
                mw = posterior_means(x[:, lo:hi], g, h, T, pmarg[lo], t)
                s_win[:, k] = score_from_means(mw[:, k - lo], x[:, k], t)
            num[r] += ((s_win - s_true) ** 2).sum(axis=1)

    point = {r: float(np.sqrt(num[r].sum() / den.sum())) for r in RADII}

    # Bootstrap over draws, on the ratio itself (it is a ratio of sums, so
    # resampling numerator and denominator together is the paired thing to do).
    boot = rng.integers(0, n_draws, size=(2000, n_draws))
    se = {}
    for r in RADII:
        vals = np.sqrt(num[r][boot].sum(axis=1) / den[boot].sum(axis=1))
        se[r] = float(vals.std(ddof=1))
    return point, se


def main() -> None:
    n_draws = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    wl, se = laplace_oracle(n_draws)
    wg = gaussian_oracle(RADII)

    print(f"pooled relative score error of the radius-r local oracle "
          f"(alpha={RHO}, L={NSITES}, {len(SCHEDULE)} levels, {n_draws} draws)\n")
    print(f"{'r':>3} {'Gaussian W(r)':>15} {'Laplace W_L(r)':>16} "
          f"{'boot SE':>9} {'ratio L/G':>11} {'sigmas':>8}")
    for r in RADII:
        gi = wg[RADII.index(r)]
        print(f"{r:>3} {gi:>15.4f} {wl[r]:>16.4f} {se[r]:>9.4f} "
              f"{wl[r] / gi:>11.3f} {(wl[r] - gi) / se[r]:>8.1f}")

    body = [
        "%% GENERATED by thesis/tools/make_laplace_oracle.py -- do not hand-edit.",
        "%%",
        "%% The Laplace chain's OWN radius-r local oracle, by window-restricted",
        "%% grid BP over its true state marginals, pooled exactly as",
        "%% eq:em-oracle pools the Gaussian one. Monte Carlo over the chain's",
        f"%% law, {n_draws} draws, fixed seed.",
    ]
    names = {0: "zero", 1: "one", 2: "two", 4: "four", 8: "eight"}
    for r in RADII:
        body.append(f"\\newcommand{{\\laporacle{names[r]}}}{{{wl[r]:.3f}}}")
        # Four places: at r >= 2 the bootstrap standard error is below 1e-3,
        # and a macro reading 0.000 would be quoted as an exact zero.
        body.append(f"\\newcommand{{\\laporacle{names[r]}se}}{{{se[r]:.4f}}}")
    body.append(f"\\newcommand{{\\laporacledraws}}{{{n_draws}}}")
    # Whether the r=2 ordering the chapter relies on is resolved by the
    # Monte Carlo, stated as a macro so the prose cannot outrun the data.
    gi = wg[RADII.index(2)]
    body.append(f"\\newcommand{{\\laporacletwosigmas}}{{{(wl[2] - gi) / se[2]:.1f}}}")
    OUT.write_text("\n".join(body) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
