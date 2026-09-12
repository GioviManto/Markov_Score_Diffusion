"""Equivalence proposition and Markov-approximation identities.

1. Single-Gaussian moment-matched (ADF) BP on a *linear* non-Gaussian chain
   equals analytic Gaussian BP on the covariance-matched Gaussian model.
   The legacy grid projection realizes the same map up to grid numerics, so at a
   moderate t (no boundary collapse) the two must agree closely.
2. The Chow-Liu chain approximation of an AR(1) covariance is the identity.
3. The Woodbury rank-one correction reproduces the exact residual score of the
   AR(1)+global model against its naive chain part.
"""

import numpy as np

from src.bp_gaussian import gaussian_chain_bp, grid_projected_gaussian_bp
from src.bp_grid import make_grid
from src.exact_scores import precision_matrix_score, sigma_t
from src.markov_approx import chow_liu_chain_covariance, rank_one_global_correction
from src.noising import log_likelihood_matrix, ou_noise_sample
from src.priors import GaussianAR1, LaplaceAR1
from src.utils import rng_for

N, RHO, T = 30, 0.85, 0.4


def test_adf_equals_gaussian_model_bp_on_laplace_chain():
    prior = LaplaceAR1(RHO)
    rng = rng_for("test-equiv", 5)
    a = prior.sample(rng, N)
    x, alpha, delta = ou_noise_sample(rng, a, T)

    grid, w = make_grid(8.0, 801)
    m_proj, _ = grid_projected_gaussian_bp(
        grid, w, prior.log_transition_matrix(grid),
        log_likelihood_matrix(grid, x, alpha, delta),
    )
    res = gaussian_chain_bp(x, RHO, prior.q, alpha, delta)
    # Same projection map, one computed on the grid, one analytically:
    # they agree up to grid quadrature/truncation error.
    assert np.max(np.abs(m_proj - res.means)) < 5e-3


def test_chow_liu_is_identity_on_markov_covariance():
    sigma = GaussianAR1(0.7).covariance(20)
    sigma_cl = chow_liu_chain_covariance(sigma)
    assert np.allclose(sigma_cl, sigma, atol=1e-12)


def test_woodbury_rank_one_residual():
    n, beta = 25, 0.5
    rng = rng_for("test-woodbury")
    x = rng.standard_normal(n)
    alpha, delta = np.exp(-0.5), 1 - np.exp(-1.0)
    sigma_chain = GaussianAR1(RHO).covariance(n)
    c = beta**2
    sigma_true = sigma_chain + c * np.ones((n, n))

    s_true = precision_matrix_score(x, sigma_true, alpha, delta)
    s_chain = precision_matrix_score(x, sigma_chain, alpha, delta)
    correction = rank_one_global_correction(
        x, sigma_t(sigma_chain, alpha, delta), alpha**2 * c
    )
    assert np.allclose(s_true, s_chain + correction, atol=1e-10)


# ----------------------------------------------------------------------------
# The defect that motivated replacing grid-mediated projection (audit F1)
# ----------------------------------------------------------------------------

def test_information_form_is_exact_where_grid_projection_collapses():
    """Pins both halves of audit finding F1, so neither can silently regress.

    The original package formed Gaussian messages by evaluating them on the
    grid, pushing them through the exact grid update, and moment-matching the
    result. At weakly informative t the outgoing message is a near-flat ramp
    whose maximizer lies outside [-A, A]; moment-matching that *truncated*
    function drags the message toward the grid boundary, and the next update
    pushes it further -- positive feedback.

    The replacement updates precision and information analytically. A flat
    message is lambda = 0 exactly, nothing is ever truncated, and on a Gaussian
    AR(1) prior the result is the exact posterior mean.

    Asserted here: the analytic form is exact to machine precision at every t,
    and the legacy form is orders of magnitude worse once the likelihood stops
    being informative. The second assertion is deliberately a *lower* bound on
    the legacy error -- if someone "fixes" the legacy routine this test should
    fail and be deleted along with the routine.
    """
    import numpy as np

    from src.bp_gaussian import gaussian_chain_bp, grid_projected_gaussian_bp
    from src.bp_grid import make_grid
    from src.exact_scores import exact_gaussian_posterior_mean
    from src.noising import alpha_delta, log_likelihood_matrix
    from src.priors import GaussianAR1
    from src.utils import rng_for

    rho, n = 0.85, 20
    prior = GaussianAR1(rho)
    sigma0 = prior.covariance(n)
    grid, weights = make_grid(8.0, 401)
    log_k = prior.log_transition_matrix(grid)
    rng = rng_for("test-f1-collapse")
    a = prior.sample(rng, n)

    worst_legacy = 0.0
    for t in (0.2, 0.6, 1.0, 1.3, 1.8):
        alpha, delta = alpha_delta(t)
        x = alpha * a + np.sqrt(delta) * rng.standard_normal(n)
        m_exact = exact_gaussian_posterior_mean(x, sigma0, alpha, delta)

        m_info = gaussian_chain_bp(x, rho, prior.q, alpha, delta).means
        assert np.abs(m_info - m_exact).max() < 1e-12, (
            f"information-form BP is no longer exact on a Gaussian prior at t={t}"
        )

        m_legacy, _ = grid_projected_gaussian_bp(
            grid, weights, log_k, log_likelihood_matrix(grid, x, alpha, delta)
        )
        worst_legacy = max(worst_legacy, float(np.abs(m_legacy - m_exact).max()))

    assert worst_legacy > 1e-2, (
        "the legacy grid-projected routine no longer exhibits the F1 instability; "
        "if it was repaired, delete it and this assertion together"
    )
