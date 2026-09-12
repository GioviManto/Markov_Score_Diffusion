"""Grid self-convergence and truncation sanity checks."""

import numpy as np

from src.bp_grid import grid_bp, make_grid, score_from_posterior_mean
from src.noising import log_likelihood_matrix, ou_noise_sample
from src.priors import LaplaceAR1
from src.utils import rng_for

N, RHO, T = 30, 0.85, 0.3


def _score(prior, x, alpha, delta, half_width, size):
    grid, w = make_grid(half_width, size)
    res = grid_bp(
        grid, w, prior.log_transition_matrix(grid),
        log_likelihood_matrix(grid, x, alpha, delta), x, alpha, delta,
    )
    return score_from_posterior_mean(x, res.means, alpha, delta)


def test_doubling_grid_changes_little():
    prior = LaplaceAR1(RHO)
    rng = rng_for("test-conv", 3)
    a = prior.sample(rng, N)
    x, alpha, delta = ou_noise_sample(rng, a, T)
    s_coarse = _score(prior, x, alpha, delta, 8.0, 201)
    s_fine = _score(prior, x, alpha, delta, 8.0, 401)
    assert np.linalg.norm(s_coarse - s_fine) / np.linalg.norm(s_fine) < 1e-3


def test_truncation_hurts_at_small_half_width():
    prior = LaplaceAR1(RHO)
    rng = rng_for("test-trunc", 4)
    a = prior.sample(rng, N)
    x, alpha, delta = ou_noise_sample(rng, a, T)
    s_ref = _score(prior, x, alpha, delta, 8.0, 801)
    err_wide = np.linalg.norm(_score(prior, x, alpha, delta, 6.0, 601) - s_ref)
    err_narrow = np.linalg.norm(_score(prior, x, alpha, delta, 3.0, 301) - s_ref)
    # Same grid step (dx = 0.02); the narrow window must be strictly worse.
    assert err_narrow > err_wide
