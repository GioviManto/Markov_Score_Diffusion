"""Closed-form scores and posterior means for Gaussian priors.

For any Gaussian clean prior a ~ N(0, Sigma_0) under the OU kernel,

    x_t ~ N(0, Sigma_t),        Sigma_t = alpha_t^2 Sigma_0 + Delta_t I,
    s_true(x, t) = -Sigma_t^{-1} x                     (exact score),
    m_true(x, t) = alpha_t Sigma_0 Sigma_t^{-1} x      (exact posterior mean),
    log p_t(x)   = logpdf of N(0, Sigma_t) at x        (exact evidence).

These are the calibration ground truths for Layers 2 and 4. Everything uses
`solve` (never explicit inverses) for numerical hygiene.
"""

from __future__ import annotations

import numpy as np


def sigma_t(sigma0: np.ndarray, alpha: float, delta: float) -> np.ndarray:
    n = sigma0.shape[0]
    return alpha**2 * sigma0 + delta * np.eye(n)


def precision_matrix_score(
    x: np.ndarray, sigma0: np.ndarray, alpha: float, delta: float
) -> np.ndarray:
    """Exact Gaussian score s = -Sigma_t^{-1} x."""
    return -np.linalg.solve(sigma_t(sigma0, alpha, delta), x)


def exact_gaussian_posterior_mean(
    x: np.ndarray, sigma0: np.ndarray, alpha: float, delta: float
) -> np.ndarray:
    """Exact Gaussian posterior mean E[a | x] = alpha Sigma_0 Sigma_t^{-1} x."""
    return alpha * sigma0 @ np.linalg.solve(sigma_t(sigma0, alpha, delta), x)


def gaussian_log_evidence(
    x: np.ndarray, sigma0: np.ndarray, alpha: float, delta: float
) -> float:
    """log p_t(x) for the Gaussian prior (multivariate normal logpdf)."""
    st = sigma_t(sigma0, alpha, delta)
    n = len(x)
    sign, logdet = np.linalg.slogdet(st)
    if sign <= 0:
        raise np.linalg.LinAlgError("Sigma_t is not positive definite.")
    quad = float(x @ np.linalg.solve(st, x))
    return -0.5 * (n * np.log(2.0 * np.pi) + logdet + quad)


def score_batch(
    X: np.ndarray, sigma0: np.ndarray, alpha: float, delta: float
) -> np.ndarray:
    """Exact scores for a batch of observations, X of shape (B, n)."""
    return -np.linalg.solve(sigma_t(sigma0, alpha, delta), X.T).T
