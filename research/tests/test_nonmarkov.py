"""Correctness tests for the non-Markov reference.

`src/nonmarkov.py` computes the exact posterior mean of a chain-plus-global-latent prior by
conditioning on the latent, running grid BP on a change of variables, and marginalising with
Gauss-Hermite. Every step of that is a place to be subtly wrong in a way that still produces
plausible numbers, so it is pinned from two directions that share no code with it:

* at ``beta = 0`` the construction must collapse *exactly* onto plain grid BP, since the
  latent then does nothing -- this catches errors in the change of variables, the Jacobian,
  and the quadrature weights all at once;

* with Gaussian innovations the answer is available in closed form from
  `exact_scores.exact_gaussian_posterior_mean` under `GaussianAR1PlusGlobal.covariance`,
  which is linear algebra on an n-by-n matrix and shares nothing with the grid recursion.

The second is the real check. A reference that agreed with itself would prove nothing; this
one has to agree with a completely separate derivation of the same quantity.

The batched evidence that `grid_bp_batch` now returns is tested here too, against the
single-chain `grid_bp` it duplicates, because the latent posterior is built from it.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bp_grid import grid_bp, grid_bp_batch, make_grid
from src.exact_scores import exact_gaussian_posterior_mean
from src.noising import alpha_delta, log_likelihood_matrix
from src.nonmarkov import (
    ChainPlusGlobal,
    gauss_hermite_normal,
    gaussian_plus_global,
    global_latent_posterior_mean,
    laplace_plus_global,
)
from src.priors import GaussianAR1, GaussianAR1PlusGlobal, LaplaceAR1

RHO = 0.85
GRID_A = 8.0
GRID_M = 401
N_SITES = 12


@pytest.fixture(scope="module")
def grid_pair():
    return make_grid(GRID_A, GRID_M)


def test_gauss_hermite_weights_integrate_the_normal(grid_pair):
    """The substitution from exp(-x^2) to the standard normal, checked on moments it must
    reproduce exactly: a 41-point rule is exact for polynomials up to degree 81."""
    nodes, w = gauss_hermite_normal(41)
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-12)
    assert float(w @ nodes) == pytest.approx(0.0, abs=1e-12)
    assert float(w @ nodes**2) == pytest.approx(1.0, abs=1e-12)
    assert float(w @ nodes**4) == pytest.approx(3.0, abs=1e-10)


def test_batched_evidence_matches_single_chain(grid_pair):
    """`grid_bp_batch(return_evidence=True)` against the `grid_bp` it duplicates.

    The two accumulate the same three pieces -- forward rescaling constants, the tail
    integral, and the per-(chain, site) likelihood shifts -- but through different array
    layouts, and the batched one had been discarding all of it. The latent posterior weights
    chains by this number, so an error here would tilt the marginalisation rather than fail.
    """
    grid, weights = grid_pair
    prior = LaplaceAR1(RHO)
    log_k = prior.log_transition_matrix(grid)
    t = 0.3
    alpha, delta = alpha_delta(t)

    rng = np.random.default_rng(3)
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(5)])
    X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)

    _, _, log_ev_batch = grid_bp_batch(
        grid, weights, log_k, X, alpha, delta, return_evidence=True
    )
    for b, x in enumerate(X):
        single = grid_bp(
            grid, weights, log_k, log_likelihood_matrix(grid, x, alpha, delta),
            x, alpha, delta,
        )
        assert log_ev_batch[b] == pytest.approx(single.log_evidence, rel=1e-12)


@pytest.mark.parametrize("base", [LaplaceAR1(RHO), GaussianAR1(RHO)])
def test_zero_beta_reduces_to_plain_bp(base, grid_pair):
    """With no latent, the whole construction must be plain grid BP -- exactly.

    Every node of the quadrature then sees the same observations and the same kernel, so the
    posterior over g collapses back to its prior and the weighted average is a no-op. Any
    residual difference would be a bug in the change of variables or the weights, since
    there is no approximation left to blame.
    """
    grid, weights = grid_pair
    prior = ChainPlusGlobal(base=base, beta=0.0)
    t = 0.25
    alpha, delta = alpha_delta(t)

    rng = np.random.default_rng(11)
    A = np.stack([base.sample(rng, N_SITES) for _ in range(8)])
    X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)

    means, log_ev = global_latent_posterior_mean(
        prior, grid, weights, X, alpha, delta, n_nodes=21
    )
    m_plain, _, ev_plain = grid_bp_batch(
        grid, weights, base.log_transition_matrix(grid), X, alpha, delta,
        return_evidence=True,
    )

    assert np.max(np.abs(means - m_plain)) < 1e-12
    assert np.max(np.abs(log_ev - ev_plain)) < 1e-10


@pytest.mark.parametrize("beta", [0.25, 0.5, 1.0])
def test_gaussian_case_matches_exact_linear_algebra(beta, grid_pair):
    """The independent check: same law, two unrelated computations.

    With Gaussian innovations the prior is jointly Gaussian, so `GaussianAR1PlusGlobal`
    supplies the covariance and the posterior mean is one linear solve. The reference under
    test instead runs the chain recursion on a grid at each of 41 latent nodes and averages.
    Nothing is shared beyond the model definition, so agreement pins the change of variables,
    the inflated noise variance, the Jacobian and the latent posterior simultaneously.

    The tolerance is the grid's, not the method's: at M = 401 over +-8 the recursion carries
    the same O(h^2) every other reference in this project carries.
    """
    grid, weights = grid_pair
    prior = gaussian_plus_global(RHO, beta)
    reference = GaussianAR1PlusGlobal(RHO, beta)
    t = 0.3
    alpha, delta = alpha_delta(t)

    rng = np.random.default_rng(int(100 * beta))
    A = np.stack([reference.sample(rng, N_SITES) for _ in range(16)])
    X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)

    means, _ = global_latent_posterior_mean(
        prior, grid, weights, X, alpha, delta, n_nodes=41
    )
    sigma0 = reference.covariance(N_SITES)
    m_exact = np.stack([exact_gaussian_posterior_mean(x, sigma0, alpha, delta) for x in X])

    assert np.max(np.abs(means - m_exact)) < 1e-6


def test_covariance_matches_the_gaussian_construction():
    """The Laplace and Gaussian versions must share second moments exactly, or the whole
    comparison stops being controlled -- any difference between them would then be
    attributable to correlation rather than to shape."""
    for beta in (0.0, 0.3, 1.0):
        lap = laplace_plus_global(RHO, beta).covariance(N_SITES)
        gauss = GaussianAR1PlusGlobal(RHO, beta).covariance(N_SITES)
        assert np.max(np.abs(lap - gauss)) < 1e-14
        assert np.max(np.abs(np.diag(lap) - 1.0)) < 1e-14


def test_sampled_covariance_matches_the_analytic_one():
    """The sampler and the covariance formula are written independently; this is the only
    thing tying them together."""
    prior = laplace_plus_global(RHO, 0.5)
    rng = np.random.default_rng(20260807)
    A = prior.sample_batch(rng, 60000, 8)
    emp = np.cov(A, rowvar=False)
    assert np.max(np.abs(emp - prior.covariance(8))) < 0.02


def test_latent_quadrature_is_converged_at_the_default(grid_pair):
    """41 nodes is the default; this shows it is not a number that matters.

    If the answer still moved between 21 and 81 nodes, the reference would be quoting its
    quadrature rather than the model, and the non-Markov results would inherit that.
    """
    grid, weights = grid_pair
    prior = laplace_plus_global(RHO, 0.75)
    t = 0.2
    alpha, delta = alpha_delta(t)

    rng = np.random.default_rng(99)
    A = prior.sample_batch(rng, 8, N_SITES)
    X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)

    coarse, _ = global_latent_posterior_mean(prior, grid, weights, X, alpha, delta, 21)
    default, _ = global_latent_posterior_mean(prior, grid, weights, X, alpha, delta, 41)
    fine, _ = global_latent_posterior_mean(prior, grid, weights, X, alpha, delta, 81)

    assert np.max(np.abs(default - fine)) < 1e-9
    assert np.max(np.abs(default - fine)) < np.max(np.abs(coarse - fine)) + 1e-12
