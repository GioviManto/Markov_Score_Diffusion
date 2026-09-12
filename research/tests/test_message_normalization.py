"""Normalization invariances of grid BP."""

import numpy as np

from src.bp_grid import grid_bp, make_grid
from src.noising import likelihood_resolution_ok, log_likelihood_matrix, ou_noise_sample
from src.priors import LaplaceAR1
from src.utils import rng_for

N, RHO, T = 25, 0.8, 0.3


def _run(log_ell_shift=None):
    prior = LaplaceAR1(RHO)
    rng = rng_for("test-norm", 1)
    a = prior.sample(rng, N)
    x, alpha, delta = ou_noise_sample(rng, a, T)
    grid, w = make_grid(8.0, 201)
    log_ell = log_likelihood_matrix(grid, x, alpha, delta)
    if log_ell_shift is not None:
        log_ell = log_ell + log_ell_shift[:, None]
    res = grid_bp(
        grid, w, prior.log_transition_matrix(grid), log_ell, x, alpha, delta
    )
    return grid, w, res


def test_beliefs_integrate_to_one():
    grid, w, res = _run()
    masses = res.beliefs @ w
    assert np.allclose(masses, 1.0, atol=1e-10)


def test_means_invariant_to_per_site_likelihood_constants():
    # ell_i is only defined up to a per-site constant for the posterior; means
    # must be identical when each row of log ell is shifted arbitrarily.
    rng = rng_for("test-norm-shift")
    shifts = rng.uniform(-3.0, 3.0, size=N)
    _, _, res_a = _run()
    _, _, res_b = _run(log_ell_shift=shifts)
    assert np.allclose(res_a.means, res_b.means, atol=1e-10)


def test_resolution_check_flags_narrow_likelihoods():
    grid, _ = make_grid(8.0, 101)  # dx = 0.16
    # t = 0.4: width sqrt(delta)/alpha ~ 1.1 -> fine.
    assert likelihood_resolution_ok(grid, np.exp(-0.4), 1 - np.exp(-0.8))
    # t = 0.005: width ~ 0.1 < 3 dx -> flagged.
    assert not likelihood_resolution_ok(grid, np.exp(-0.005), 1 - np.exp(-0.01))
