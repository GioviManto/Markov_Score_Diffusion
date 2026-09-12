"""The two forward--backward implementations must not drift apart.

WHY THIS EXISTS (review item H2). `bp_grid.grid_bp` and `em._e_step_chunk` both
run the same scaled linear-domain forward--backward recursion over the same
finite-state model. The duplication has a reason -- EM needs pairwise edge
statistics and the evidence, inference needs node beliefs, and fusing them would
make the inference path pay for statistics it never uses -- but the comment
claiming they are "the same recursion" was an assertion, not a guarantee.
Nothing stopped one of them acquiring a different normalisation point, a
likelihood factor placed on the other side of a transition, or a backend change
the other did not get.

The clean fix is a shared internal routine returning scaled messages, their
normalisers and the likelihood shifts, with each caller forming the beliefs it
wants. That is the right design and it is a refactor of the hottest code path in
the package; doing it three weeks before a submission deadline, to code that
currently passes, trades a hypothetical divergence for a real chance of
introducing one.

So this pins the property the refactor would protect. If the two recursions ever
disagree about the evidence or the site-1 posterior, on any of the families and
noise levels below, this fails -- which is the actual content of "they are the
same recursion". The refactor can then happen behind a test that would catch it
going wrong, which is the order these things should be done in.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bp_grid import grid_bp, make_grid
from src.em import e_step
from src.noising import alpha_delta, log_likelihood_matrix
from src.priors import GaussianAR1, GaussianMixtureAR1, LaplaceAR1, StudentTAR1
from src.utils import rng_for

RHO, N_SITES, M, HALF = 0.85, 12, 201, 8.0

PRIORS = [
    pytest.param(GaussianAR1(RHO), id="gaussian"),
    pytest.param(LaplaceAR1(RHO), id="laplace"),
    pytest.param(StudentTAR1(RHO), id="student"),
    pytest.param(GaussianMixtureAR1(RHO, kappa=0.9), id="bimodal"),
]
# Both ends of the schedule plus the middle: the low end is where the likelihood
# is narrowest against the mesh, the high end is where the posterior is widest
# against the domain, and a normalisation error would surface differently at each.
TIMES = [0.05, 0.322, 3.0]


@pytest.mark.parametrize("prior", PRIORS)
@pytest.mark.parametrize("t", TIMES)
def test_evidence_agrees_between_bp_grid_and_em(prior, t):
    grid, weights = make_grid(HALF, M)
    rng = rng_for("recursion-agreement", prior.name, f"{t:.4f}")
    a = prior.sample(rng, N_SITES)
    alpha, delta = alpha_delta(t)
    x = alpha * a + np.sqrt(delta) * rng.standard_normal(a.shape)

    log_k = prior.log_transition_matrix(grid)
    # Both are given the SAME uniform initial law, or they are entitled to
    # differ and the comparison means nothing.
    log_mu = np.full(M, -np.log(M)) + np.log(weights)

    res = grid_bp(grid, weights, log_k,
                  log_likelihood_matrix(grid, x, alpha, delta), x, alpha, delta,
                  log_mu=log_mu)
    stats = e_step(grid, weights, log_k, x[None, :], alpha, delta, log_mu)

    assert stats.log_evidence == pytest.approx(res.log_evidence, rel=1e-12), (
        f"{prior.name} at t={t}: bp_grid {res.log_evidence:.12f} vs "
        f"em {stats.log_evidence:.12f} -- the two recursions have diverged"
    )


@pytest.mark.parametrize("prior", PRIORS)
def test_site1_posterior_agrees_between_bp_grid_and_em(prior):
    """`site1` from EM must be the site-1 belief `bp_grid` reports.

    This is the check that catches a likelihood factor attached on the wrong
    side of a transition: such an error shifts where mass sits without
    necessarily changing the total, so the evidence can still agree.
    """
    grid, weights = make_grid(HALF, M)
    rng = rng_for("recursion-agreement-site1", prior.name)
    a = prior.sample(rng, N_SITES)
    alpha, delta = alpha_delta(0.322)
    x = alpha * a + np.sqrt(delta) * rng.standard_normal(a.shape)

    log_k = prior.log_transition_matrix(grid)
    log_mu = np.full(M, -np.log(M)) + np.log(weights)

    res = grid_bp(grid, weights, log_k,
                  log_likelihood_matrix(grid, x, alpha, delta), x, alpha, delta,
                  log_mu=log_mu)
    stats = e_step(grid, weights, log_k, x[None, :], alpha, delta, log_mu)

    # bp_grid's beliefs are densities on the grid; site1 is a probability mass
    # summing to the number of chains, i.e. the belief times the weights.
    expected = res.beliefs[0] * weights
    np.testing.assert_allclose(stats.site1, expected, rtol=1e-10, atol=1e-14)
