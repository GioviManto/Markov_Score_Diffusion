"""Tests for discrete-alphabet chains.

This is the one setting in the package where BP has *no* representation error,
so the tests can demand exactness rather than a tolerance chosen from a
convergence study. Where the continuous tests accept 1e-12 because a grid is
involved, these accept roundoff.
"""

from __future__ import annotations

import itertools

import numpy as np

from src.discrete import (
    baum_welch_m_step,
    discrete_bp,
    fit_em_discrete,
    make_random_chain,
    monotone_violation,
)
from src.noising import alpha_delta
from src.utils import rng_for


def _brute_force(chain, X, alpha, delta):
    """Enumerate all S^n configurations; no messages involved."""
    s, n = chain.n_states, X.shape[1]
    xi = np.zeros((s, s))
    means = np.zeros(X.shape)
    log_z = 0.0
    for b, x in enumerate(X):
        ell = np.exp(
            -0.5 * (x[:, None] - alpha * chain.levels[None, :]) ** 2 / delta
        ) / np.sqrt(2.0 * np.pi * delta)
        joint = {}
        for cfg in itertools.product(range(s), repeat=n):
            p = chain.mu[cfg[0]] * ell[0, cfg[0]]
            for i in range(1, n):
                p *= chain.K[cfg[i], cfg[i - 1]] * ell[i, cfg[i]]
            joint[cfg] = p
        z = sum(joint.values())
        log_z += np.log(z)
        for cfg, v in joint.items():
            for i in range(n - 1):
                xi[cfg[i + 1], cfg[i]] += v / z
            for i in range(n):
                means[b, i] += (v / z) * chain.levels[cfg[i]]
    return xi, means, log_z


def test_discrete_bp_is_exact():
    """Against explicit enumeration: Xi, posterior means and evidence."""
    rng = rng_for("test-disc-exact")
    chain = make_random_chain(4, rng)
    n, t = 5, 0.4
    alpha, delta = alpha_delta(t)
    A = np.stack([chain.sample(rng, n) for _ in range(3)])
    X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)

    res = discrete_bp(chain.levels, chain.K, X, alpha, delta, chain.mu)
    xi_bf, means_bf, log_z_bf = _brute_force(chain, X, alpha, delta)

    assert np.abs(res.xi - xi_bf).max() < 1e-12
    assert np.abs(res.means - means_bf).max() < 1e-12
    assert abs(res.log_evidence - log_z_bf) < 1e-10
    assert np.isclose(res.xi.sum(), res.n_edges, rtol=1e-12)


def test_baum_welch_m_step_maximizes_q():
    """Normalized counts maximize <Xi, log K> over column-stochastic K."""
    rng = rng_for("test-disc-mstep")
    chain = make_random_chain(4, rng)
    alpha, delta = alpha_delta(0.3)
    A = np.stack([chain.sample(rng, 24) for _ in range(40)])
    X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
    xi = discrete_bp(chain.levels, chain.K, X, alpha, delta, chain.mu).xi

    best = baum_welch_m_step(xi)
    q_best = float(np.sum(xi * np.log(best)))

    perturb = rng_for("test-disc-mstep-perturb")
    for _ in range(40):
        cand = np.abs(best + perturb.normal(0, 0.03, best.shape))
        cand /= cand.sum(axis=0, keepdims=True)
        assert float(np.sum(xi * np.log(cand))) <= q_best + 1e-9


def test_discrete_em_is_exactly_monotone_and_recovers_k():
    """No grid, so the ascent guarantee holds with no numerical slack at all."""
    rng = rng_for("test-disc-em")
    chain = make_random_chain(4, rng, concentration=0.6)
    A = np.stack([chain.sample(rng, 40) for _ in range(800)])
    groups = []
    parts = np.array_split(rng.permutation(len(A)), 4)
    for t, idx in zip((0.1, 0.2, 0.4, 0.8), parts):
        alpha, delta = alpha_delta(t)
        sub = A[idx]
        groups.append(
            (alpha * sub + np.sqrt(delta) * rng.standard_normal(sub.shape), alpha, delta)
        )

    k0 = rng.dirichlet(np.ones(4), size=4).T
    k_fit, trace = fit_em_discrete(chain.levels, k0, groups, n_iters=300)

    assert monotone_violation(trace["log_evidence"]) == 0.0
    assert trace["log_evidence"][-1] > trace["log_evidence"][0]
    assert np.abs(k_fit - chain.K).max() < np.abs(k0 - chain.K).max() / 5
    assert np.allclose(k_fit.sum(axis=0), 1.0)


def test_small_noise_posterior_concentrates_on_the_observed_level():
    """The t -> 0 limit, which here is exact rather than grid-limited."""
    rng = rng_for("test-disc-smallt")
    chain = make_random_chain(5, rng)
    n = 12
    a = chain.sample(rng, n)
    alpha, delta = alpha_delta(0.01)
    x = alpha * a + np.sqrt(delta) * rng.standard_normal(n)

    res = discrete_bp(chain.levels, chain.K, x[None, :], alpha, delta, chain.mu)
    assert np.abs(res.means[0] - a).max() < 0.05
    # And the belief is nearly a point mass on the true level.
    assert res.beliefs[0].max(axis=1).min() > 0.9
