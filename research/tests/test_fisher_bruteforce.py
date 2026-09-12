"""An INDEPENDENT control on Fisher's identity, by enumerating the posterior.

WHY THIS EXISTS. The gradient is validated in the paper against a finite
difference of the grid evidence, agreeing to ~1e-9. That check is real but it is
not independent: the evidence it differences comes from the forward recursion,
and the posterior statistics in the gradient come from the same recursion and
the same discretised transition matrix. An error in a shared component -- a
transposed index, a misplaced likelihood factor, a normalisation applied at the
wrong point -- can satisfy both sides of that comparison and leave the agreement
looking perfect. The external review made exactly this point.

What follows shares nothing with `e_step` except the kernel and the grid. On a
tiny grid the whole posterior is enumerable: with M states and n sites there are
M^(n-1) latent paths per chain given the first site, so for M=5, n=4 the sum has
125 terms and can be written directly from the model definition.

Three things are then checked against that enumeration:

  1. the log-evidence,
  2. Xi, INCLUDING its index orientation -- the one thing a self-consistent
     recursion cannot catch, because a transposed Xi paired with a transposed
     kernel still differences correctly,
  3. the analytic gradient, against a central finite difference of the
     BRUTE-FORCE evidence rather than of the recursion's.

Run for a smooth kernel (Gaussian) and a non-smooth one (Laplace, whose residual
has a kink at zero), because the two exercise different parts of the quadrature.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.bp_grid import make_grid
from src.em import e_step, q_gradient
from src.kernels import GaussianAR1Kernel, LaplaceAR1Kernel
from src.noising import alpha_delta
from src.utils import rng_for

# Small enough to enumerate, large enough that a transpose is not a symmetry:
# an odd grid on an asymmetric domain means K is not symmetric, so K and K^T
# genuinely differ and orientation errors have somewhere to show up.
M, N_SITES, N_CHAINS = 5, 4, 3
HALF_WIDTH = 2.5
T = 0.35


def _setup(kernel):
    grid, weights = make_grid(HALF_WIDTH, M)
    rng = rng_for("fisher-bruteforce")
    alpha, delta = alpha_delta(T)
    # Observations drawn near the grid so no chain is pure tail.
    X = alpha * rng.uniform(-1.0, 1.0, (N_CHAINS, N_SITES)) + np.sqrt(delta) * rng.standard_normal(
        (N_CHAINS, N_SITES)
    )
    log_mu = np.full(M, -np.log(M))          # uniform initial law over states
    return grid, weights, X, alpha, delta, log_mu


def _brute_force(kernel, grid, weights, X, alpha, delta, log_mu):
    """Enumerate every latent path. Returns (log_evidence, Xi).

    Written from the model definition, deliberately without any recursion:

        p(a, x) = mu(a_1) prod_i K(a_{i+1} | a_i) prod_i ell(x_i | a_i)

    with the quadrature weight w_k attached to each latent site, which is what
    makes the finite sum a discretisation of the continuous integral.
    """
    log_K = kernel.log_transition_matrix(grid)   # log_K[out, in]
    total_log_ev = 0.0
    xi = np.zeros((M, M))
    for c in range(N_CHAINS):
        # The FULLY NORMALISED Gaussian likelihood, written out here rather than
        # taken from `noising.log_likelihood_matrix`. That helper returns a
        # row-shifted table -- each row's maximum subtracted, and the
        # -0.5 log(2 pi Delta) constant dropped -- because BP normalises
        # messages and neither matters to it; `e_step` adds both back when it
        # assembles the evidence. Reusing it here would have meant comparing the
        # recursion against a quantity that shares its normalisation
        # bookkeeping, which is one of the couplings this test exists to avoid.
        # (It also fails loudly if you get it wrong: the first version of this
        # file used the helper and the evidence came out 7.43 nats low, exactly
        # the dropped constants.)
        z = X[c][:, None] - alpha * grid[None, :]
        log_ell = -0.5 * z**2 / delta - 0.5 * np.log(2.0 * np.pi * delta)
        terms, paths = [], []
        for path in itertools.product(range(M), repeat=N_SITES):
            lp = log_mu[path[0]] + np.log(weights[path[0]])
            for i in range(N_SITES - 1):
                # log_K is indexed [child, parent]; path[i] is the parent.
                lp += log_K[path[i + 1], path[i]] + np.log(weights[path[i + 1]])
            for i, k in enumerate(path):
                lp += log_ell[i, k]
            terms.append(lp)
            paths.append(path)
        terms = np.array(terms)
        mx = terms.max()
        ev = np.exp(terms - mx).sum()
        total_log_ev += float(mx + np.log(ev))
        post = np.exp(terms - mx) / ev
        for p, w in zip(paths, post):
            for i in range(N_SITES - 1):
                xi[p[i + 1], p[i]] += w          # [child, parent]
    return total_log_ev, xi


KERNELS = [
    pytest.param(GaussianAR1Kernel(rho=0.7, q=1 - 0.7**2), id="gaussian"),
    pytest.param(LaplaceAR1Kernel(rho=0.7, b=np.sqrt((1 - 0.7**2) / 2)), id="laplace"),
]


@pytest.mark.parametrize("kernel", KERNELS)
def test_evidence_matches_independent_enumeration(kernel):
    grid, weights, X, alpha, delta, log_mu = _setup(kernel)
    ref_ev, _ = _brute_force(kernel, grid, weights, X, alpha, delta, log_mu)
    stats = e_step(grid, weights, kernel.log_transition_matrix(grid), X, alpha,
                   delta, log_mu)
    assert stats.log_evidence == pytest.approx(ref_ev, rel=1e-10), (
        f"recursion {stats.log_evidence:.12f} vs enumeration {ref_ev:.12f}"
    )


@pytest.mark.parametrize("kernel", KERNELS)
def test_xi_matches_enumeration_including_orientation(kernel):
    """Xi must match, and must NOT match its own transpose.

    The second half is the part the finite-difference check cannot do. If Xi
    were built as [parent, child] while `log_transition_matrix` returns
    [child, parent], the inner product <Xi, grad log K> would contract the wrong
    pairs -- and against a self-consistent recursion it could still difference
    correctly, because the same transposition would sit on both sides.
    """
    grid, weights, X, alpha, delta, log_mu = _setup(kernel)
    _, ref_xi = _brute_force(kernel, grid, weights, X, alpha, delta, log_mu)
    stats = e_step(grid, weights, kernel.log_transition_matrix(grid), X, alpha,
                   delta, log_mu)

    np.testing.assert_allclose(stats.xi, ref_xi, rtol=1e-9, atol=1e-12)
    assert stats.xi.sum() == pytest.approx(N_CHAINS * (N_SITES - 1))
    # The orientation guard only means something if the matrix is asymmetric.
    asym = np.abs(ref_xi - ref_xi.T).max()
    assert asym > 1e-3, "Xi is near-symmetric here, so this test proves nothing"


@pytest.mark.parametrize("kernel", KERNELS)
def test_gradient_matches_finite_difference_of_the_enumerated_evidence(kernel):
    """The gradient, differenced against evidence the recursion never touched.

    Laplace is included on purpose. Its rho-derivative reads a kernel with a
    kink at zero residual, so the quadrature is less accurate there and the
    tolerance is correspondingly looser -- that looseness is a property of
    differentiating a non-smooth density on a mesh, not a defect in the
    identity, and pinning it here records which is which.
    """
    grid, weights, X, alpha, delta, log_mu = _setup(kernel)
    stats = e_step(grid, weights, kernel.log_transition_matrix(grid), X, alpha,
                   delta, log_mu)
    analytic = q_gradient(stats, kernel.grad_log_transition_matrix(grid))

    eps = 1e-5
    theta = np.asarray(kernel.theta, dtype=float)
    numeric = np.empty_like(theta)
    for p in range(theta.size):
        hi, lo = theta.copy(), theta.copy()
        hi[p] += eps
        lo[p] -= eps
        k_hi = type(kernel)(*hi)
        k_lo = type(kernel)(*lo)
        ev_hi, _ = _brute_force(k_hi, grid, weights, X, alpha, delta, log_mu)
        ev_lo, _ = _brute_force(k_lo, grid, weights, X, alpha, delta, log_mu)
        numeric[p] = (ev_hi - ev_lo) / (2 * eps)

    tol = 2e-4 if kernel.name == "gaussian_ar1" else 5e-3
    rel = np.abs(analytic - numeric) / np.maximum(np.abs(numeric), 1.0)
    assert rel.max() < tol, (
        f"{kernel.name}: analytic {analytic} vs brute-force finite difference "
        f"{numeric} (relative {rel})"
    )
