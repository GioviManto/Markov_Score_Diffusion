"""Tests for Gaussian-mixture message passing.

The claims guarded here are the ones exp_09 rests on: the message algebra is
exact, collapsing is the *only* approximation, C = 1 reproduces the existing
single-Gaussian baseline exactly (so a component sweep is anchored to it), and
more components really do help on a kernel the family can represent.
"""

from __future__ import annotations

import numpy as np

from src.bp_gaussian import gaussian_chain_bp
from src.bp_grid import grid_bp_batch, make_grid
from src.bp_mixture import (
    MixtureMessage,
    collapse,
    mixture_chain_bp,
    multiply_by_gaussian,
    push_backward,
    push_forward,
)
from src.exact_scores import exact_gaussian_posterior_mean
from src.noising import alpha_delta
from src.priors import GaussianAR1, GaussianMixtureAR1
from src.utils import rng_for

RHO = 0.85


def _on_grid(msg: MixtureMessage, grid: np.ndarray) -> np.ndarray:
    """Evaluate a mixture density on a grid, for numerical cross-checks."""
    w = msg.weights()
    vals = np.zeros_like(grid)
    for wi, mi, vi in zip(w, msg.mean, msg.var):
        vals += wi * np.exp(-0.5 * (grid - mi) ** 2 / vi) / np.sqrt(2 * np.pi * vi)
    return vals


def test_c1_reproduces_the_analytic_gaussian_baseline():
    """C = 1 must BE the existing Gaussian baseline, or the sweep has no anchor.

    On a Gaussian prior the analytic information-form BP is exact, so this also
    checks mixture BP against the exact posterior mean.
    """
    n = 20
    prior = GaussianAR1(RHO)
    rng = rng_for("test-mix-c1")
    a = prior.sample(rng, n)
    for t in (0.2, 0.8, 1.8):
        alpha, delta = alpha_delta(t)
        x = alpha * a + np.sqrt(delta) * rng.standard_normal(n)
        m_gauss = gaussian_chain_bp(x, RHO, prior.q, alpha, delta).means
        m_exact = exact_gaussian_posterior_mean(x, prior.covariance(n), alpha, delta)
        m_mix, _ = mixture_chain_bp(
            x, RHO, np.array([1.0]), np.array([0.0]), np.array([prior.q]),
            alpha, delta, max_components=1,
        )
        assert np.abs(m_mix - m_gauss).max() < 1e-12
        assert np.abs(m_mix - m_exact).max() < 1e-12


def test_message_algebra_matches_numerical_integration():
    """push_forward / push_backward / multiply are exact, checked on a grid.

    These three operations carry the whole recursion, and an error in any of
    them would look like a plausible-but-wrong closure error rather than a
    crash. Cross-checked against direct quadrature of the defining integrals.
    """
    grid, weights = make_grid(14.0, 4001)
    pi = np.array([0.35, 0.65])
    mu = np.array([0.4, -0.25])
    s2 = np.array([0.09, 0.16])
    msg = MixtureMessage(np.log([0.3, 0.7]), np.array([-0.5, 0.8]), np.array([0.2, 0.05]))

    # Multiplication by a Gaussian.
    y, r = 0.3, 0.12
    got = _on_grid(multiply_by_gaussian(msg, y, r), grid)
    want = _on_grid(msg, grid) * np.exp(-0.5 * (grid - y) ** 2 / r) / np.sqrt(
        2 * np.pi * r
    )
    want = want / (want @ weights)
    got = got / (got @ weights)
    assert np.abs(got - want).max() < 1e-9

    # Forward push: L'(a') = int K(a'|a) L(a) da, K(a'|a) = sum pi_k N(a'-rho a; mu_k, s2_k)
    dens = _on_grid(msg, grid)
    kern = np.zeros((len(grid), len(grid)))
    e = grid[:, None] - RHO * grid[None, :]
    for p_, m_, v_ in zip(pi, mu, s2):
        kern += p_ * np.exp(-0.5 * (e - m_) ** 2 / v_) / np.sqrt(2 * np.pi * v_)
    want_f = kern @ (dens * weights)
    want_f /= want_f @ weights
    got_f = _on_grid(push_forward(msg, RHO, pi, mu, s2), grid)
    got_f /= got_f @ weights
    assert np.abs(got_f - want_f).max() < 1e-8

    # Backward push: R'(a) = int K(a'|a) R(a') da'
    want_b = kern.T @ (dens * weights)
    want_b /= want_b @ weights
    got_b = _on_grid(push_backward(msg, RHO, pi, mu, s2), grid)
    got_b /= got_b @ weights
    assert np.abs(got_b - want_b).max() < 1e-8


def test_collapse_preserves_mass_mean_and_variance():
    """Runnalls merging is moment preserving, which bounds how wrong it can be.

    Merging a pair by moment matching preserves that pair's mean and variance,
    hence the whole mixture's. So a collapsed message can misrepresent shape but
    never first or second moments -- worth pinning, because the posterior mean
    is exactly what the project measures.
    """
    rng = rng_for("test-mix-collapse")
    for _ in range(20):
        k = int(rng.integers(6, 24))
        msg = MixtureMessage(
            np.log(rng.dirichlet(np.ones(k))),
            rng.normal(0, 1.5, k),
            rng.uniform(0.02, 0.6, k),
        )
        m0, v0 = msg.moments()
        for target in (1, 2, 4):
            small = collapse(msg, target)
            assert small.size <= target
            m1, v1 = small.moments()
            assert abs(m1 - m0) < 1e-9
            assert abs(v1 - v0) < 1e-9


def test_more_components_reduce_error_on_a_representable_kernel():
    """The claim exp_09 Part 1 makes: with no model error, C controls the error.

    The kernel here is exactly a two-component Gaussian mixture, so mixture BP
    with enough components can in principle be exact and every deviation from
    grid BP is representation error.
    """
    n, kappa = 20, 0.9
    prior = GaussianMixtureAR1(RHO, kappa)
    pi = np.array([0.5, 0.5])
    mu = np.array([+prior.m, -prior.m])
    s2 = np.array([prior.s**2, prior.s**2])

    grid, weights = make_grid(10.0, 801)
    log_k = prior.log_transition_matrix(grid)
    rng = rng_for("test-mix-sweep")
    A = np.stack([prior.sample(rng, n) for _ in range(12)])

    alpha, delta = alpha_delta(0.05)
    X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
    m_ref, _ = grid_bp_batch(grid, weights, log_k, X, alpha, delta)

    errs = []
    for c in (1, 2, 4, 8):
        m_mix = np.stack(
            [mixture_chain_bp(x, RHO, pi, mu, s2, alpha, delta, c)[0] for x in X]
        )
        errs.append(float(np.linalg.norm(m_mix - m_ref) / np.linalg.norm(m_ref)))

    assert all(b < a for a, b in zip(errs, errs[1:])), f"not monotone in C: {errs}"
    assert errs[0] / errs[-1] > 20.0, f"C=8 barely beats C=1: {errs}"
