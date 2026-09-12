"""Best Markov (chain) approximations of Gaussian priors and residual analysis.

For a Gaussian prior p = N(0, Sigma) that is NOT Markov, the best chain-structured
approximation q (minimizing KL(p || q) over Gaussians that are Markov with respect
to the path 1-2-...-n) matches the pairwise marginals of adjacent sites
(Chow-Liu on a fixed tree). Its covariance is the "Markov telescoping" of the
adjacent correlations:

    Corr_q(a_i, a_j) = prod_{k=i}^{j-1} r_k,     r_k = Corr_p(a_k, a_{k+1}),

with the exact marginal variances of p on the diagonal.

The residual score operator at time t is

    s_true - s_markov = (Sigma_markov,t^{-1} - Sigma_t^{-1}) x  =:  D_t x,

a LINEAR map for Gaussian models. Its singular-value spectrum measures how
"simple" the non-Markov correction is: for the AR(1)+global-latent model the
correction is exactly rank one at every t when the chain part is matched
(Woodbury identity; see report), and approximately rank one for the Chow-Liu
chain. A steep spectrum is the quantitative version of "the residual is simpler
than the full score".
"""

from __future__ import annotations

import numpy as np

from .exact_scores import sigma_t


def chow_liu_chain_covariance(sigma: np.ndarray) -> np.ndarray:
    """Covariance of the best chain-Gaussian approximation (adjacent pairs matched)."""
    n = sigma.shape[0]
    std = np.sqrt(np.diag(sigma))
    corr = sigma / np.outer(std, std)
    r = np.array([corr[k, k + 1] for k in range(n - 1)])
    # Corr_q(i, j) = prod of adjacent correlations between i and j.
    log_r = np.log(np.abs(r))
    sign_r = np.sign(r)
    cum_log = np.concatenate([[0.0], np.cumsum(log_r)])
    cum_sign = np.concatenate([[1.0], np.cumprod(sign_r)])
    idx = np.arange(n)
    i, j = np.minimum.outer(idx, idx), np.maximum.outer(idx, idx)
    corr_q = (cum_sign[j] / cum_sign[i]) * np.exp(cum_log[j] - cum_log[i])
    return corr_q * np.outer(std, std)


def residual_operator(
    sigma_true: np.ndarray, sigma_markov: np.ndarray, alpha: float, delta: float
) -> np.ndarray:
    """D_t with  s_true(x) - s_markov(x) = D_t x."""
    st_true = sigma_t(sigma_true, alpha, delta)
    st_markov = sigma_t(sigma_markov, alpha, delta)
    n = sigma_true.shape[0]
    eye = np.eye(n)
    return np.linalg.solve(st_markov, eye) - np.linalg.solve(st_true, eye)


def operator_spectrum(D: np.ndarray) -> np.ndarray:
    """Singular values of the residual operator, descending."""
    return np.linalg.svd(D, compute_uv=False)


def effective_rank(singular_values: np.ndarray, tol: float = 1e-10) -> float:
    """Entropy-based effective rank (Roy & Vetterli): exp(H(p)), p = s / sum(s)."""
    s = singular_values[singular_values > tol * singular_values.max()]
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def rank_one_global_correction(
    x: np.ndarray,
    sigma_chain_t: np.ndarray,
    coupling: float,
) -> np.ndarray:
    """Woodbury rank-one score correction for Sigma_t = Sigma_chain,t + c 11^T.

    If the true noisy covariance differs from a chain one by c * 11^T (the
    AR(1)+global model with the *unnormalized* chain part), then

        Sigma_t^{-1} = S^{-1} - (c / (1 + c 1^T S^{-1} 1)) S^{-1} 1 1^T S^{-1},

    so  s_true(x) - s_chain(x) = (c / (1 + c 1^T S^{-1} 1)) (1^T S^{-1} x) S^{-1} 1:
    a rank-one, x-linear correction along the fixed smooth direction S^{-1} 1.
    """
    ones = np.ones(len(x))
    s_inv_one = np.linalg.solve(sigma_chain_t, ones)
    s_inv_x = np.linalg.solve(sigma_chain_t, x)
    gain = coupling / (1.0 + coupling * float(ones @ s_inv_one))
    return gain * float(ones @ s_inv_x) * s_inv_one
