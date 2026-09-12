"""Gaussian message passing on the chain posterior.

Two implementations with very different numerical status:

1. `gaussian_chain_bp` (recommended): analytic information-form forward-backward.
   Messages are parametrized by (precision lambda >= 0, information h); a flat
   message is exactly lambda = 0. All updates are closed-form, no grid, no
   truncation, no boundary pathologies.

   * On a Gaussian AR(1) prior this is EXACT (Gaussian closure).
   * On any *linear-transition* chain a_i = rho a_{i-1} + eps_i with
     Var(eps) = q and Gaussian OU likelihoods, this coincides with
     moment-matched assumed-density (single-Gaussian projected) BP:
     the tilted product (Gaussian message x Gaussian likelihood) is exactly
     Gaussian, and the transition step maps N(m, v) to a density whose first two
     moments are (rho m, rho^2 v + q) regardless of the innovation shape.
     Consequently single-Gaussian ADF-BP == exact Gaussian BP on the
     covariance-matched Gaussian model (equivalence proposition; verified in
     tests). Its error on a non-Gaussian chain is therefore both the "Gaussian
     message error" and the "Gaussian model error" -- they are the same object
     at the single-Gaussian level.

2. `grid_projected_gaussian_bp` (legacy, for the audit): the original package's
   grid-mediated moment-matching projection. Kept verbatim in spirit to
   reproduce and document its boundary-collapse instability at weakly
   informative t (audit finding F1). Not used as a baseline in new experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianBPResult:
    means: np.ndarray
    variances: np.ndarray
    # Message parameters (prior-side, excluding the local likelihood):
    L_prec: np.ndarray
    L_info: np.ndarray
    R_prec: np.ndarray
    R_info: np.ndarray


def gaussian_chain_bp(
    x: np.ndarray,
    rho: float,
    q: float,
    alpha: float,
    delta: float,
    prior_var0: float = 1.0,
) -> GaussianBPResult:
    """Analytic information-form BP for the linear chain posterior.

    Model: a_1 ~ N(0, prior_var0), a_i = rho a_{i-1} + innovation(var q),
    x_i | a_i ~ N(alpha a_i, delta).

    The likelihood contributes precision alpha^2/delta and information
    alpha x_i / delta at each site. Forward step: absorb the local likelihood,
    convert to moments, push through the transition (m, v) -> (rho m,
    rho^2 v + q). Backward step: absorb, then invert the transition
    (m, v) -> (m / rho, (v + q) / rho^2).
    """
    n = len(x)
    lam_lik = alpha**2 / delta
    h_lik = alpha * x / delta

    L_prec = np.empty(n)
    L_info = np.empty(n)
    R_prec = np.zeros(n)
    R_info = np.zeros(n)

    L_prec[0] = 1.0 / prior_var0
    L_info[0] = 0.0

    for i in range(n - 1):
        lam_t = L_prec[i] + lam_lik
        m_t = (L_info[i] + h_lik[i]) / lam_t
        v_t = 1.0 / lam_t
        v_next = rho**2 * v_t + q
        L_prec[i + 1] = 1.0 / v_next
        L_info[i + 1] = rho * m_t / v_next

    # Terminal right message is flat: precision 0.
    for i in range(n - 1, 0, -1):
        lam_t = R_prec[i] + lam_lik
        m_t = (R_info[i] + h_lik[i]) / lam_t
        v_t = 1.0 / lam_t
        v_prev = (v_t + q) / rho**2
        R_prec[i - 1] = 1.0 / v_prev
        R_info[i - 1] = (m_t / rho) / v_prev

    lam_b = L_prec + R_prec + lam_lik
    h_b = L_info + R_info + h_lik
    means = h_b / lam_b
    variances = 1.0 / lam_b
    return GaussianBPResult(
        means=means,
        variances=variances,
        L_prec=L_prec,
        L_info=L_info,
        R_prec=R_prec,
        R_info=R_info,
    )


# ----------------------------------------------------------------------------
# Legacy grid-projected variant (audit finding F1) -- kept for diagnosis only.
# ----------------------------------------------------------------------------

def _density_moments(
    values: np.ndarray, grid: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    values = np.maximum(values, 0.0)
    mass = float(np.sum(values * weights))
    if not np.isfinite(mass) or mass <= 0.0:
        raise FloatingPointError("Cannot normalize projected message.")
    p = values / mass
    mean = float(np.sum(grid * p * weights))
    var = float(np.sum((grid - mean) ** 2 * p * weights))
    return mean, max(var, 1e-12)


def _gaussian_values(grid: np.ndarray, mean: float, var: float) -> np.ndarray:
    if np.isinf(var):
        return np.ones_like(grid)
    return np.exp(-0.5 * (grid - mean) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def grid_projected_gaussian_bp(
    grid: np.ndarray,
    weights: np.ndarray,
    log_K: np.ndarray,
    log_ell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Original grid-mediated Gaussian projection (means, variances).

    Each message is evaluated on the grid, pushed through the exact grid update,
    then moment-matched. This is mathematically the same projection as
    `gaussian_chain_bp` for linear transitions, but its *numerics* differ: when
    the outgoing message function is a near-flat ramp with maximizer outside
    [-A, A] (weakly informative t), moment-matching the truncated function
    collapses the message onto the grid boundary. Retained to demonstrate the
    artifact; do not use as a scientific baseline.
    """
    n, m = log_ell.shape
    K = np.exp(log_K)
    ell = np.exp(log_ell)

    L_mean = np.empty(n)
    L_var = np.empty(n)
    R_mean = np.empty(n)
    R_var = np.empty(n)
    L_mean[0], L_var[0] = 0.0, 1.0
    R_mean[-1], R_var[-1] = 0.0, np.inf

    for i in range(n - 1):
        incoming = _gaussian_values(grid, L_mean[i], L_var[i]) * ell[i] * weights
        L_mean[i + 1], L_var[i + 1] = _density_moments(K @ incoming, grid, weights)

    for i in range(n - 1, 0, -1):
        incoming = _gaussian_values(grid, R_mean[i], R_var[i]) * ell[i] * weights
        R_mean[i - 1], R_var[i - 1] = _density_moments(K.T @ incoming, grid, weights)

    means = np.empty(n)
    variances = np.empty(n)
    for i in range(n):
        b = (
            _gaussian_values(grid, L_mean[i], L_var[i])
            * ell[i]
            * _gaussian_values(grid, R_mean[i], R_var[i])
        )
        means[i], variances[i] = _density_moments(b, grid, weights)
    return means, variances
