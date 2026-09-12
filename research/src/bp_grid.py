"""Reference grid BP on a chain posterior.

Probability objects
-------------------
The posterior on a chain is

    p_t(a | x)  propto  mu(a_1) prod_i K(a_i | a_{i-1}) prod_i ell_i(a_i; x_i).

BP on this tree is *exact at the level of continuous message functions*; the only
approximation made here is the representation of those functions by their values
on an equally spaced grid u_1..u_M on [-A, A] with composite trapezoidal
quadrature. Two error sources follow:
  (i)  tail truncation: posterior mass outside [-A, A] is discarded;
  (ii) quadrature: O(dx^2) trapezoid error, which becomes serious only when an
       integrand (in practice the likelihood at small t) is narrower than a few
       grid steps.
Layer-2 experiments quantify both against the exactly solvable Gaussian case.

Messages are kept in the linear domain but renormalized to unit quadrature mass
after every step; the discarded constants are accumulated so that the model
evidence log p_t(x) can also be returned (testable against the closed-form
Gaussian marginal likelihood).

What that returned number is, precisely: a **quadrature estimate of the marginal
evidence under the truncated grid representation**. It used to be documented as
the "exact model evidence", which cannot be right two paragraphs after listing
truncation and O(dx^2) quadrature as error sources. What *is* exact is the
bookkeeping -- no rescaling constant is lost, so the returned value is the
evidence of the finite-state model the grid defines, to floating point. Reserve
"exact" for three things: the analytic Gaussian likelihood, the continuous
functional recursion before discretisation, and finite-state BP once the grid
model has been declared to be the model of interest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.backend import to_device


def make_grid(half_width: float, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Equally spaced grid on [-A, A] with composite trapezoidal weights."""
    grid = np.linspace(-half_width, half_width, size)
    dx = float(grid[1] - grid[0])
    weights = np.full(size, dx)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return grid, weights


@dataclass(frozen=True)
class GridBPResult:
    """Posterior summaries from grid BP.

    means, variances : posterior marginal mean / variance per site (exact up to
                       grid error).
    beliefs          : normalized posterior marginals on the grid, shape (n, M).
    log_evidence     : log p_t(x) under the truncated grid model -- a quadrature
                       estimate of the continuous evidence, including all
                       constants (likelihood normalizers and message rescalings).
                       Exact for the finite-state model the grid defines; see the
                       module docstring on what "exact" does and does not cover.
    """

    means: np.ndarray
    variances: np.ndarray
    beliefs: np.ndarray
    log_evidence: float


def grid_bp(
    grid: np.ndarray,
    weights: np.ndarray,
    log_K: np.ndarray,
    log_ell: np.ndarray,
    x: np.ndarray,
    alpha: float,
    delta: float,
    log_mu: np.ndarray | None = None,
) -> GridBPResult:
    """Sum-product BP with grid-represented messages on the chain posterior.

    Parameters
    ----------
    log_K   : log K[out, in] = log p(a_out | a_in) on the grid, shape (M, M).
    log_ell : row-shifted log-likelihood table from `noising.log_likelihood_matrix`,
              shape (n, M). Row shifts are undone in the evidence computation.
    x       : observations (needed only to restore the exact evidence constants).
    log_mu  : log initial density on the grid; defaults to standard normal.

    Left message L_i excludes the local likelihood ell_i; right message R_i
    likewise. Recursions (trapezoidal quadrature of the exact continuous ones):

        L_{i+1} = K @ (L_i * ell_i * w),      R_{i-1} = K^T @ (R_i * ell_i * w).
    """
    n, m = log_ell.shape
    if log_mu is None:
        log_mu = -0.5 * grid**2 - 0.5 * np.log(2.0 * np.pi)

    K = np.exp(log_K)
    ell = np.exp(log_ell)

    # Evidence bookkeeping: true log ell_i(a) = log_ell[i] + shift_i + gauss_norm,
    # where shift_i is the row max removed in noising.log_likelihood_matrix.
    z = x[:, None] - alpha * grid[None, :]
    row_shift = (-0.5 * z**2 / delta).max(axis=1)
    gauss_norm = -0.5 * np.log(2.0 * np.pi * delta)
    log_const = float(np.sum(row_shift) + n * gauss_norm)

    L = np.empty((n, m))
    R = np.empty((n, m))
    L[0] = np.exp(log_mu)
    log_z_fwd = 0.0  # accumulated log of message rescaling constants

    for i in range(n - 1):
        incoming = L[i] * ell[i] * weights
        out = K @ incoming
        c = float(np.sum(out * weights))
        if not np.isfinite(c) or c <= 0.0:
            raise FloatingPointError(f"Forward message {i + 1} lost all mass.")
        L[i + 1] = out / c
        log_z_fwd += np.log(c)

    R[-1] = np.ones(m)
    for i in range(n - 1, 0, -1):
        incoming = R[i] * ell[i] * weights
        out = K.T @ incoming
        c = float(np.sum(out * weights))
        if not np.isfinite(c) or c <= 0.0:
            raise FloatingPointError(f"Backward message {i - 1} lost all mass.")
        R[i - 1] = out / c

    beliefs = np.empty((n, m))
    means = np.empty(n)
    variances = np.empty(n)
    for i in range(n):
        raw = L[i] * ell[i] * R[i]
        mass = float(np.sum(raw * weights))
        if not np.isfinite(mass) or mass <= 0.0:
            raise FloatingPointError(f"Belief {i} lost all mass.")
        b = raw / mass
        beliefs[i] = b
        means[i] = float(np.sum(grid * b * weights))
        variances[i] = float(np.sum((grid - means[i]) ** 2 * b * weights))

    # Evidence: p(x) = (prod of forward rescale constants) * int L_n ell_n da,
    # where L_n is the *normalized* forward message at the last site; restore the
    # removed likelihood constants at the end.
    tail = float(np.sum(L[-1] * ell[-1] * weights))
    log_evidence = log_z_fwd + np.log(tail) + log_const

    return GridBPResult(
        means=means, variances=variances, beliefs=beliefs, log_evidence=log_evidence
    )


def score_from_posterior_mean(
    x: np.ndarray, means: np.ndarray, alpha: float, delta: float
) -> np.ndarray:
    """Exact OU identity  s_i(x, t) = -(x_i - alpha_t E[a_i | x]) / Delta_t.

    Feeding *approximate* posterior means through this identity produces an
    approximate score whose error is exactly (alpha/delta) times the mean error.
    """
    return -(x - alpha * means) / delta


def grid_bp_batch(
    grid: np.ndarray,
    weights: np.ndarray,
    log_K: np.ndarray,
    X: np.ndarray,
    alpha: float,
    delta: float,
    log_mu: np.ndarray | None = None,
    xp=None,
    *,
    return_evidence: bool = False,
    boundary_frac: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized grid BP over a batch of observations X of shape (B, n).

    Same algorithm as `grid_bp` (independent chains, one per batch row), with
    the site loop shared and each message update a (M, M) @ (M, B) matmul.
    Used by the reverse-dynamics experiments, where BP must run at every
    integration step for every trajectory. Returns (means, variances), each of
    shape (B, n), or (means, variances, log_evidence) with `return_evidence=True`,
    the third being the per-chain log p_t(x) under the grid model, shape (B,).

    The evidence is the same quantity `grid_bp` returns for a single chain, computed
    the same way -- forward rescaling constants, tail integral, and the restored
    per-(chain, site) likelihood shifts -- and `tests/test_bp_evidence_batch.py`
    asserts the two agree. It is off by default because almost every caller wants
    only the means, and because a per-chain evidence is what the non-Markov
    reference needs to weight a latent variable it is marginalising over.

    ``xp`` selects the array module (numpy, or cupy for GPU execution). This is the *same*
    recursion either way -- there is deliberately no second implementation, because a
    separately written GPU version would put the exactness guarantees at risk in the least
    visible way possible. ``tests/test_backend_parity.py`` asserts the two devices agree to
    machine precision on identical inputs. Inputs are moved to the device here, and the
    result is returned in ``xp``'s own array type; callers that need numpy should use
    ``backend.to_host``.
    """
    if xp is None:
        xp = np
    grid = to_device(grid, xp)
    weights = to_device(weights, xp)
    log_K = to_device(log_K, xp)
    X = to_device(X, xp)

    b_size, n = X.shape
    m = len(grid)
    if log_mu is None:
        log_mu = -0.5 * grid**2 - 0.5 * xp.log(2.0 * xp.pi)
    else:
        log_mu = to_device(log_mu, xp)
    K = xp.exp(log_K)

    # ell[i] has shape (B, M): row-shifted per (batch, site). The shifts are kept rather
    # than discarded so the evidence can be restored exactly, as `grid_bp` does.
    z = X[:, :, None] - alpha * grid[None, None, :]
    log_ell = -0.5 * z**2 / delta
    row_shift = log_ell.max(axis=2)  # (B, n)
    log_ell = log_ell - row_shift[:, :, None]
    ell = xp.exp(log_ell)  # (B, n, M)

    L = xp.empty((n, b_size, m))
    R = xp.empty((n, b_size, m))
    L[0] = xp.exp(log_mu)[None, :]
    log_z_fwd = xp.zeros(b_size)

    for i in range(n - 1):
        incoming = L[i] * ell[:, i, :] * weights[None, :]  # (B, M)
        out = (K @ incoming.T).T
        mass = out @ weights
        if not bool(xp.all(xp.isfinite(mass))) or bool(xp.any(mass <= 0.0)):
            raise FloatingPointError(f"Batched forward message {i + 1} lost mass.")
        L[i + 1] = out / mass[:, None]
        log_z_fwd = log_z_fwd + xp.log(mass)

    R[-1] = 1.0
    for i in range(n - 1, 0, -1):
        incoming = R[i] * ell[:, i, :] * weights[None, :]
        out = (K.T @ incoming.T).T
        mass = out @ weights
        if not bool(xp.all(xp.isfinite(mass))) or bool(xp.any(mass <= 0.0)):
            raise FloatingPointError(f"Batched backward message {i - 1} lost mass.")
        R[i - 1] = out / mass[:, None]

    means = xp.empty((b_size, n))
    variances = xp.empty((b_size, n))
    bel_edge = xp.zeros((b_size, n)) if boundary_frac else None
    for i in range(n):
        raw = L[i] * ell[:, i, :] * R[i]
        mass = raw @ weights
        if not bool(xp.all(xp.isfinite(mass))) or bool(xp.any(mass <= 0.0)):
            raise FloatingPointError(f"Batched belief {i} lost mass.")
        bel = raw / mass[:, None]
        means[:, i] = bel @ (grid * weights)
        variances[:, i] = bel @ (grid**2 * weights) - means[:, i] ** 2
        if boundary_frac:
            e = max(1, int(boundary_frac * m))
            bel_edge[:, i] = (bel[:, :e] @ weights[:e]) + (bel[:, -e:] @ weights[-e:])

    if boundary_frac:
        # How much of the mass sits in the outermost `boundary_frac` of the
        # domain -- for the objects the recursion actually propagates.
        #
        # WHY THIS LIVES HERE (round-two review, item 9.1). The convergence tool
        # used to compute its own "boundary mass" as
        #     ell * exp(log_K.max(axis=0))
        # normalised over the grid axis. That is the unary likelihood times the
        # column-wise maximum of the transition matrix. It involves no forward
        # message, no backward message, no recursion, no quadrature weights and
        # no initial law, so whatever it measured, it was not the boundary mass
        # of anything BP propagates -- and the appendix quoted it as such.
        #
        # Computing it inside the recursion is the only way to be sure the
        # diagnostic and the algorithm refer to the same object.
        e = max(1, int(boundary_frac * m))
        fwd_edge = (L[:, :, :e] @ weights[:e]) + (L[:, :, -e:] @ weights[-e:])
        diag = {
            # L is normalised to unit mass under the quadrature, so these are
            # already fractions.
            "forward_edge_max": float(xp.max(fwd_edge)),
            "forward_edge_mean": float(xp.mean(fwd_edge)),
            "belief_edge_max": float(xp.max(bel_edge)),
            "belief_edge_mean": float(xp.mean(bel_edge)),
            "edge_cells": int(e),
        }
        if return_evidence:
            tail = (L[-1] * ell[:, -1, :]) @ weights
            log_const = row_shift.sum(axis=1) + n * (-0.5 * xp.log(2.0 * xp.pi * delta))
            return means, variances, log_z_fwd + xp.log(tail) + log_const, diag
        return means, variances, diag

    if return_evidence:
        tail = (L[-1] * ell[:, -1, :]) @ weights
        log_const = row_shift.sum(axis=1) + n * (-0.5 * xp.log(2.0 * xp.pi * delta))
        return means, variances, log_z_fwd + xp.log(tail) + log_const
    return means, variances
