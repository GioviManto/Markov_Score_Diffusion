"""Coordinatewise Ornstein-Uhlenbeck (variance-preserving) forward process.

Convention (stated once, used everywhere)
-----------------------------------------
Forward SDE in diffusion time t >= 0:

    dX_t = -X_t dt + sqrt(2) dW_t,          X_0 = a  ~  p_0.

Its transition kernel is

    X_t | a  ~  N(alpha_t a, Delta_t I),
    alpha_t = exp(-t),      Delta_t = 1 - exp(-2t).

For unit-variance clean data the noisy marginal variance is
alpha_t^2 + Delta_t = 1 at every t (variance preserving).

The per-site likelihood of the clean value a_i given the observation x_i is

    ell_i(a_i; x_i) = N(x_i; alpha_t a_i, Delta_t),

an *exact* Gaussian factor. All BP algorithms consume it through
`log_likelihood_matrix`, which is shifted per row so it can never underflow to an
all-zero row (audit finding F4); per-site constant shifts are absorbed by message
normalization and therefore change nothing.
"""

from __future__ import annotations

import numpy as np


def alpha_delta(t: float) -> tuple[float, float]:
    """Return (alpha_t, Delta_t) for the OU convention above."""
    alpha = float(np.exp(-t))
    delta = float(1.0 - np.exp(-2.0 * t))
    return alpha, delta


def ou_noise_sample(
    rng: np.random.Generator, a: np.ndarray, t: float
) -> tuple[np.ndarray, float, float]:
    """Sample x = alpha_t a + sqrt(Delta_t) z with z ~ N(0, I)."""
    alpha, delta = alpha_delta(t)
    x = alpha * a + np.sqrt(delta) * rng.standard_normal(a.shape)
    return x, alpha, delta


def log_likelihood_matrix(
    grid: np.ndarray, x: np.ndarray, alpha: float, delta: float
) -> np.ndarray:
    """Row-shifted log-likelihood table  log ell[i, k] = log N(x_i; alpha u_k, Delta) + c_i.

    The per-row constant c_i (the negative row max) is irrelevant for BP because
    messages and beliefs are normalized; shifting guarantees each row has max 0,
    so `exp` of the result never underflows everywhere at once.
    """
    z = x[:, None] - alpha * grid[None, :]
    log_ell = -0.5 * z**2 / delta
    return log_ell - log_ell.max(axis=1, keepdims=True)


def likelihood_matrix(
    grid: np.ndarray, x: np.ndarray, alpha: float, delta: float
) -> np.ndarray:
    """Likelihood table ell[i, k] proportional (per row) to N(x_i; alpha u_k, Delta)."""
    return np.exp(log_likelihood_matrix(grid, x, alpha, delta))


def likelihood_resolution_ok(
    grid: np.ndarray, alpha: float, delta: float, points_per_std: float = 3.0
) -> bool:
    """Check that the grid resolves the likelihood width (audit finding F5).

    The likelihood, as a function of a, has standard deviation sqrt(Delta)/alpha.
    We require at least `points_per_std` grid points per standard deviation;
    below that, quadrature of the near-delta likelihood is unreliable.
    """
    dx = float(grid[1] - grid[0])
    width = float(np.sqrt(delta) / alpha)
    return width / dx >= points_per_std
