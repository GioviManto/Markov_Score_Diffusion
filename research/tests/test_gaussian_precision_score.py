"""Grid BP and analytic Gaussian BP against the closed-form Gaussian AR(1) truth."""

import numpy as np
import pytest

from src.bp_gaussian import gaussian_chain_bp
from src.bp_grid import grid_bp, make_grid, score_from_posterior_mean
from src.exact_scores import (
    exact_gaussian_posterior_mean,
    gaussian_log_evidence,
    precision_matrix_score,
)
from src.noising import alpha_delta, log_likelihood_matrix, ou_noise_sample
from src.priors import GaussianAR1
from src.utils import rng_for

N, RHO, T = 40, 0.85, 0.4


@pytest.fixture(scope="module")
def setup():
    prior = GaussianAR1(RHO)
    rng = rng_for("test-gauss", 0)
    a = prior.sample(rng, N)
    x, alpha, delta = ou_noise_sample(rng, a, T)
    sigma0 = prior.covariance(N)
    return prior, x, alpha, delta, sigma0


def test_grid_bp_score_matches_precision_score(setup):
    prior, x, alpha, delta, sigma0 = setup
    grid, w = make_grid(8.0, 401)
    res = grid_bp(
        grid, w, prior.log_transition_matrix(grid),
        log_likelihood_matrix(grid, x, alpha, delta), x, alpha, delta,
    )
    s_grid = score_from_posterior_mean(x, res.means, alpha, delta)
    s_true = precision_matrix_score(x, sigma0, alpha, delta)
    assert np.linalg.norm(s_grid - s_true) / np.linalg.norm(s_true) < 1e-5


def test_analytic_gaussian_bp_is_exact(setup):
    prior, x, alpha, delta, sigma0 = setup
    res = gaussian_chain_bp(x, RHO, prior.q, alpha, delta)
    m_true = exact_gaussian_posterior_mean(x, sigma0, alpha, delta)
    # Information-form chain BP must agree with the matrix formula to
    # machine precision: both are exact for the Gaussian chain.
    assert np.max(np.abs(res.means - m_true)) < 1e-10


def test_grid_bp_evidence_matches_gaussian_logpdf(setup):
    prior, x, alpha, delta, sigma0 = setup
    grid, w = make_grid(8.0, 401)
    res = grid_bp(
        grid, w, prior.log_transition_matrix(grid),
        log_likelihood_matrix(grid, x, alpha, delta), x, alpha, delta,
    )
    log_p = gaussian_log_evidence(x, sigma0, alpha, delta)
    assert abs(res.log_evidence - log_p) < 1e-4 * abs(log_p) + 1e-6


def test_posterior_mean_and_score_formulas_consistent(setup):
    _, x, alpha, delta, sigma0 = setup
    s = precision_matrix_score(x, sigma0, alpha, delta)
    m = exact_gaussian_posterior_mean(x, sigma0, alpha, delta)
    # s = -(x - alpha m)/delta must hold between the two closed forms.
    assert np.allclose(s, -(x - alpha * m) / delta, atol=1e-10)
