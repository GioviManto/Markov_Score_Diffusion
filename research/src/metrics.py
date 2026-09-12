"""Shared error metrics. All take (estimate, reference) in that order."""

from __future__ import annotations

import numpy as np


def rel_l2(est: np.ndarray, ref: np.ndarray) -> float:
    """||est - ref||_2 / ||ref||_2."""
    denom = float(np.linalg.norm(ref))
    return float(np.linalg.norm(est - ref)) / denom if denom > 0 else np.nan


def mse(est: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean((est - ref) ** 2))


def max_abs(est: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(est - ref)))


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.dot(u, v)) / denom if denom > 0 else np.nan


def transition_hellinger(
    log_k_est: np.ndarray,
    log_k_ref: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    interior_frac: float = 0.5,
    parent_law: np.ndarray | None = None,
) -> dict:
    """Hellinger distance between two transition kernels, at the density level.

    Everything else in this package measures the fitted kernel through what it
    does -- the score it induces, the posterior mean it produces. Those are the
    quantities the diffusion model actually uses, but they are also forgiving:
    two visibly different innovation densities can induce scores that agree to
    the third decimal at moderate noise, because the channel has already blurred
    the difference away. A claim that the transition itself was recovered has to
    be stated on the transition.

    For each parent state u_j the columns K(.|u_j) are densities on the grid, so

        H^2(p, q) = 1 - sum_k w_k sqrt(p_k q_k)

    with w the quadrature weights. H is in [0, 1]; 0 is identical, 1 is disjoint
    support. Reported as a summary over parent states rather than a single
    number, because the tails are where a fitted mixture goes wrong and averaging
    over columns hides it.

    Returns median, mean, p90 and max over parent states, plus the same over
    interior parents only. The interior restriction matters for the same reason
    it does in the quadrature diagnostic: at |u| near the half-width the true
    transition density has a tail outside the grid, so its column is not
    normalised there and the distance picks up truncation rather than fit error.
    `interior_frac` is the fraction of the half-width kept, matching
    `bp_grid`'s convention.

    THERE IS NO RESOLUTION FLOOR, and the claim that there was one was wrong.
    This function used to compute H = sqrt(1 - BC) directly, which subtracts two
    numbers near 1: identical kernels came back at 3.7e-8 rather than 0, and that
    3.7e-8 was documented here as "inherent to the metric, not to the
    implementation", on the argument that nothing in double precision separates
    BC = 1 from BC = 1 - 1e-17. The argument is sound and the conclusion does not
    follow, because it assumes BC has to be formed. Expanding the square,

        H^2 = (1/2) sum_k w_k (sqrt(p_k) - sqrt(q_k))^2

    is the same number, never subtracts from 1, and returns exactly 0 at
    identity. The floor was an artefact of the expression, and it is gone. The
    fitted distances this is used on are ~0.05, so no reported result moves;
    what changes is that a small Hellinger can now be believed.

    THE SUMMARY OVER PARENT STATES WAS ALSO SCOPED, NOT SOLVED (round-two
    review, "Hellinger", closing paragraph). `hellinger_median` and friends are
    an UNWEIGHTED median/mean/max over grid columns -- one vote per parent grid
    POINT, regardless of how much posterior mass ever lands near that parent.
    Since `grid` is typically wide and evenly spaced while the parent marginal
    concentrates near zero, this over-weights parents in the tails, which is
    exactly where a fitted mixture is most likely to disagree with the truth for
    reasons that have nothing to do with the fit (a component with little
    support there is barely constrained by the data). A single number meant to
    say "how well was the transition recovered, in the regime that is actually
    exercised" should be weighted by the parent LAW, not by the parent grid's
    point density.

    `parent_law` supplies that weighting: an array of shape `grid.shape` giving
    the parent marginal density (unnormalised is fine; it is renormalised
    against `weights` below). Defaults to the standard normal density, which is
    the right default under this project's convention that `Var(a_i) = 1` at
    every site (`FrozenConfig.innovation_variance`), so every family here shares
    the same parent law by construction and a caller does not usually need to
    pass one. Adds `hellinger_weighted_mean` and `hellinger_weighted_median`
    without removing the unweighted fields, so existing callers and figures are
    unaffected; new uses should prefer the weighted ones.
    """
    # Column-normalise under the SAME quadrature both kernels are represented
    # in. Comparing an analytically normalised density against a grid-normalised
    # one measures the normalisation convention, not the fit -- a real hazard
    # here, where the analytic kernels normalise on the continuum and the MDN
    # kernel normalises on the grid.
    #
    # This is also why there is no "are these normalised?" guard: after this step
    # they are, by construction, so such a check could never fire.
    def cols(log_k):
        k = np.exp(log_k - log_k.max(axis=0, keepdims=True))
        mass = (k * weights[:, None]).sum(axis=0)
        return k / np.maximum(mass, 1e-300)

    p, q = cols(log_k_est), cols(log_k_ref)
    # H^2 = 1 - BC is the definition, but computing it that way subtracts two
    # numbers near 1 and loses everything below ~1e-15 to cancellation, which
    # the sqrt then amplifies to ~1e-8. The algebraically identical
    #
    #     H^2 = (1/2) sum_k w_k (sqrt(p_k) - sqrt(q_k))^2
    #
    # -- identical because sum w p = sum w q = 1 after column-normalisation --
    # never forms BC at all, so identical columns give exactly 0 and small
    # differences keep their relative accuracy. No clamp is needed either: the
    # summand is a square, so it cannot go negative and reach the sqrt.
    h = np.sqrt(
        0.5 * (weights[:, None] * (np.sqrt(p) - np.sqrt(q)) ** 2).sum(axis=0)
    )

    interior = np.abs(grid) <= interior_frac * float(np.max(np.abs(grid)))

    if parent_law is None:
        # Standard normal: the shared parent law under Var(a_i) = 1.
        parent_law = np.exp(-0.5 * grid**2) / np.sqrt(2.0 * np.pi)
    pw = weights * parent_law
    pw = pw / pw.sum()  # a probability mass over parent grid points

    # Weighted median: the smallest h such that the weighted CDF reaches 1/2.
    # Reduces to the ordinary median when pw is uniform, which is the check in
    # tests/test_hellinger.py that pins this against the unweighted function.
    order = np.argsort(h)
    cdf = np.cumsum(pw[order])
    med_idx = int(np.searchsorted(cdf, 0.5))
    weighted_median = float(h[order[min(med_idx, len(h) - 1)]])

    return {
        "hellinger_median": float(np.median(h)),
        "hellinger_mean": float(h.mean()),
        "hellinger_p90": float(np.quantile(h, 0.90)),
        "hellinger_max": float(h.max()),
        "hellinger_median_interior": float(np.median(h[interior])),
        "hellinger_max_interior": float(h[interior].max()),
        "hellinger_weighted_mean": float((pw * h).sum()),
        "hellinger_weighted_median": weighted_median,
        # E_{parent law}[H^2], not (E[H])^2 -- the two differ because squaring
        # does not commute with averaging (Jensen again). The capacity-
        # equivalence experiment (round-two review §10.6) names this quantity
        # specifically, so it is provided directly rather than left for a
        # caller to reconstruct incorrectly by squaring hellinger_weighted_mean.
        "hellinger_weighted_mean_sq": float((pw * h**2).sum()),
    }


def score_mean_identity_residual(
    s_est: np.ndarray,
    s_ref: np.ndarray,
    m_est: np.ndarray,
    m_ref: np.ndarray,
    alpha: float,
    delta: float,
) -> float:
    """Relative residual of the exact identity  s_est - s_ref = (a/d)(m_est - m_ref).

    Any score produced through `score_from_posterior_mean` satisfies this to
    machine precision; a large residual indicates an implementation bug.
    """
    direct = s_est - s_ref
    predicted = (alpha / delta) * (m_est - m_ref)
    return float(
        np.linalg.norm(direct - predicted) / (np.linalg.norm(direct) + 1e-300)
    )
