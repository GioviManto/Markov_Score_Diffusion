"""Correctness tests for the EM layer.

The claims being guarded are the ones Section 2 of `report/em_bp_learning.tex`
rests on: the E-step is exact, Xi is a sufficient statistic that conserves mass,
Fisher's identity gives the marginal-likelihood gradient without differentiating
through BP, the M-steps really maximize Q, and EM ascends the exact evidence.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.bp_grid import make_grid
from src.em import (
    clean_statistics,
    e_step,
    e_step_multi,
    fit_clean,
    fit_em,
    q_gradient,
    q_value,
)
from src.exact_scores import gaussian_log_evidence
from src.kernels import (
    GaussianAR1Kernel,
    LaplaceAR1Kernel,
    MDNKernel,
    MixtureInnovationKernel,
)
from src.noising import alpha_delta
from src.priors import GaussianAR1, LaplaceAR1
from src.utils import rng_for

RHO = 0.8
N_SITES = 10


def _sample(prior, n_chains, t, rng, n_sites=N_SITES):
    A = np.stack([prior.sample(rng, n_sites) for _ in range(n_chains)])
    alpha, delta = alpha_delta(t)
    X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
    return A, X, alpha, delta


def test_evidence_matches_closed_form_gaussian():
    """The E-step's log-evidence is the exact chain marginal likelihood."""
    grid, weights = make_grid(8.0, 301)
    prior = GaussianAR1(RHO)
    rng = rng_for("test-em-evidence")
    _, X, alpha, delta = _sample(prior, 8, 0.4, rng)

    stats = e_step(grid, weights, prior.log_transition_matrix(grid), X, alpha, delta)
    sigma0 = prior.covariance(N_SITES)
    closed = sum(gaussian_log_evidence(x, sigma0, alpha, delta) for x in X)
    assert abs(stats.log_evidence - closed) / abs(closed) < 1e-12


def test_xi_conserves_mass():
    """Xi sums to the edge count and site1 to the chain count, exactly."""
    grid, weights = make_grid(8.0, 201)
    prior = LaplaceAR1(RHO)
    rng = rng_for("test-em-mass")
    _, X, alpha, delta = _sample(prior, 5, 0.3, rng)

    stats = e_step(grid, weights, prior.log_transition_matrix(grid), X, alpha, delta)
    assert stats.n_edges == 5 * (N_SITES - 1)
    assert np.isclose(stats.xi.sum(), stats.n_edges, rtol=1e-10)
    assert np.isclose(stats.site1.sum(), stats.n_chains, rtol=1e-10)
    assert np.all(stats.xi >= 0.0)


def test_chunking_does_not_change_statistics():
    """Batching is a memory knob only."""
    grid, weights = make_grid(8.0, 161)
    prior = GaussianAR1(RHO)
    rng = rng_for("test-em-chunk")
    _, X, alpha, delta = _sample(prior, 7, 0.5, rng)
    log_k = prior.log_transition_matrix(grid)

    big = e_step(grid, weights, log_k, X, alpha, delta, chunk=64)
    small = e_step(grid, weights, log_k, X, alpha, delta, chunk=2)
    assert np.allclose(big.xi, small.xi, rtol=1e-12, atol=1e-14)
    assert np.isclose(big.log_evidence, small.log_evidence, rtol=1e-12)


def test_multi_level_statistics_are_additive():
    """Observations at different noise levels contribute additively to Xi."""
    grid, weights = make_grid(8.0, 161)
    prior = GaussianAR1(RHO)
    rng = rng_for("test-em-multi")
    _, X1, a1, d1 = _sample(prior, 4, 0.2, rng)
    _, X2, a2, d2 = _sample(prior, 4, 0.9, rng)
    log_k = prior.log_transition_matrix(grid)

    joint = e_step_multi(grid, weights, log_k, [(X1, a1, d1), (X2, a2, d2)])
    s1 = e_step(grid, weights, log_k, X1, a1, d1)
    s2 = e_step(grid, weights, log_k, X2, a2, d2)
    assert np.allclose(joint.xi, s1.xi + s2.xi, rtol=1e-12)
    assert np.isclose(joint.log_evidence, s1.log_evidence + s2.log_evidence, rtol=1e-12)


def test_fisher_identity_matches_finite_differences():
    """grad of the exact evidence == <Xi, grad log K>, for a smooth kernel.

    Uses the Gaussian kernel: the Laplace kernel's rho-derivative carries a sign
    discontinuity whose quadrature error is documented in kernels.py and tested
    separately below.
    """
    grid, weights = make_grid(8.0, 401)
    prior = GaussianAR1(RHO)
    rng = rng_for("test-em-fisher")
    _, X, alpha, delta = _sample(prior, 6, 0.4, rng)

    kernel = GaussianAR1Kernel(rho=0.72, q=0.5)
    stats = e_step(grid, weights, kernel.log_transition_matrix(grid), X, alpha, delta)
    analytic = q_gradient(stats, kernel.grad_log_transition_matrix(grid))

    eps = 1e-5
    builders = (
        lambda v: GaussianAR1Kernel(v, kernel.q),
        lambda v: GaussianAR1Kernel(kernel.rho, v),
    )
    for p, (base, build) in enumerate(zip((kernel.rho, kernel.q), builders)):
        hi = e_step(
            grid, weights, build(base + eps).log_transition_matrix(grid),
            X, alpha, delta,
        ).log_evidence
        lo = e_step(
            grid, weights, build(base - eps).log_transition_matrix(grid),
            X, alpha, delta,
        ).log_evidence
        fd = (hi - lo) / (2 * eps)
        assert abs(analytic[p] - fd) / abs(fd) < 1e-6


def test_laplace_scale_gradient_is_exact_but_rho_gradient_is_not():
    """Documents the quadrature caveat of Remark 'A discretization caveat'.

    d/db is smooth and matches a finite difference to high accuracy; d/drho
    carries sign(e) and does not. The test pins both behaviours so that a future
    change to the quadrature is noticed.
    """
    grid, weights = make_grid(8.0, 401)
    prior = LaplaceAR1(RHO)
    rng = rng_for("test-em-laplace-grad")
    _, X, alpha, delta = _sample(prior, 6, 0.4, rng)

    kernel = LaplaceAR1Kernel(rho=0.75, b=0.4)
    stats = e_step(grid, weights, kernel.log_transition_matrix(grid), X, alpha, delta)
    analytic = q_gradient(stats, kernel.grad_log_transition_matrix(grid))

    eps = 1e-5

    def fd(build, base):
        hi = e_step(grid, weights, build(base + eps).log_transition_matrix(grid),
                    X, alpha, delta).log_evidence
        lo = e_step(grid, weights, build(base - eps).log_transition_matrix(grid),
                    X, alpha, delta).log_evidence
        return (hi - lo) / (2 * eps)

    fd_b = fd(lambda v: LaplaceAR1Kernel(kernel.rho, v), kernel.b)
    assert abs(analytic[1] - fd_b) / abs(fd_b) < 1e-6

    fd_rho = fd(lambda v: LaplaceAR1Kernel(v, kernel.b), kernel.rho)
    rel = abs(analytic[0] - fd_rho) / abs(fd_rho)
    assert rel > 1e-3, "sign discontinuity unexpectedly integrated exactly"
    assert rel < 0.5, "rho gradient is far worse than the documented quadrature error"


def test_gaussian_m_step_maximizes_q():
    """The closed-form Gaussian M-step beats nearby parameters on Q."""
    grid, weights = make_grid(8.0, 301)
    prior = GaussianAR1(RHO)
    rng = rng_for("test-em-mstep")
    _, X, alpha, delta = _sample(prior, 12, 0.3, rng)

    start = GaussianAR1Kernel(0.4, 0.7)
    stats = e_step(grid, weights, start.log_transition_matrix(grid), X, alpha, delta)
    best = start.m_step(stats, grid)
    q_best = q_value(stats, best.log_transition_matrix(grid))

    perturb = rng_for("test-em-mstep-perturb")
    for _ in range(20):
        cand = GaussianAR1Kernel(
            best.rho + float(perturb.normal(0, 0.05)),
            max(best.q + float(perturb.normal(0, 0.05)), 1e-3),
        )
        assert q_value(stats, cand.log_transition_matrix(grid)) <= q_best + 1e-9


def test_laplace_m_step_maximizes_q():
    """Same for the weighted-median / mean-absolute-residual M-step."""
    grid, weights = make_grid(8.0, 301)
    prior = LaplaceAR1(RHO)
    rng = rng_for("test-em-mstep-lap")
    _, X, alpha, delta = _sample(prior, 12, 0.3, rng)

    start = LaplaceAR1Kernel(0.4, 0.7)
    stats = e_step(grid, weights, start.log_transition_matrix(grid), X, alpha, delta)
    best = start.m_step(stats, grid)
    q_best = q_value(stats, best.log_transition_matrix(grid))

    perturb = rng_for("test-em-mstep-lap-perturb")
    for _ in range(20):
        cand = LaplaceAR1Kernel(
            best.rho + float(perturb.normal(0, 0.05)),
            max(best.b + float(perturb.normal(0, 0.05)), 1e-3),
        )
        assert q_value(stats, cand.log_transition_matrix(grid)) <= q_best + 1e-9


@pytest.mark.parametrize("kernel_name", ["gaussian", "laplace", "mixture"])
def test_em_is_monotone(kernel_name):
    """The exact marginal log-likelihood never decreases across EM iterations."""
    grid, weights = make_grid(8.0, 201)
    prior = LaplaceAR1(RHO)
    rng = rng_for("test-em-monotone", kernel_name)
    groups = []
    for t in (0.2, 0.6):
        _, X, alpha, delta = _sample(prior, 24, t, rng)
        groups.append((X, alpha, delta))

    start = {
        "gaussian": GaussianAR1Kernel(0.2, 0.9),
        "laplace": LaplaceAR1Kernel(0.2, 0.9),
        "mixture": MixtureInnovationKernel.init(
            3, rho=0.2, var=0.9, rng=rng_for("test-em-monotone-init")
        ),
    }[kernel_name]

    _, trace = fit_em(start, grid, weights, groups, n_iters=25)
    assert trace.monotone_violation < 1e-8
    assert trace.log_evidence[-1] > trace.log_evidence[0]


@pytest.mark.parametrize("kernel_name", ["gaussian", "laplace", "mixture"])
@pytest.mark.parametrize("n_iters,tol", [(4, 0.0), (200, 1e-9)])
def test_returned_kernel_is_the_one_the_trace_ends_on(kernel_name, n_iters, tol):
    """The returned kernel's evidence must be `trace.log_evidence[-1]`.

    Both exit paths are covered on purpose, because they used to fail differently. With a
    small `n_iters` and `tol=0` the loop runs out of budget; with a large `n_iters` it
    converges and breaks. The old implementation logged the evidence before each M-step and
    returned the parameters that M-step produced, so on *both* paths the returned kernel was
    one update past anything evaluated -- its likelihood was never computed, the monotonicity
    check never covered the final update, and comparing the reported evidence against a
    held-out number silently compared two different models.

    Recomputing the evidence here rather than trusting the trace is the point: it is an
    independent evaluation of the returned object, so it catches the misalignment rather than
    restating whatever the loop happened to record.
    """
    grid, weights = make_grid(8.0, 201)
    prior = LaplaceAR1(RHO)
    rng = rng_for("test-em-final-trace", kernel_name, n_iters)
    groups = []
    for t in (0.2, 0.6):
        _, X, alpha, delta = _sample(prior, 24, t, rng)
        groups.append((X, alpha, delta))

    start = {
        "gaussian": GaussianAR1Kernel(0.2, 0.9),
        "laplace": LaplaceAR1Kernel(0.2, 0.9),
        "mixture": MixtureInnovationKernel.init(
            3, rho=0.2, var=0.9, rng=rng_for("test-em-final-init", n_iters)
        ),
    }[kernel_name]

    fitted, trace = fit_em(start, grid, weights, groups, n_iters=n_iters, tol=tol)

    assert len(trace.log_evidence) == len(trace.theta) == len(trace.seconds)
    np.testing.assert_allclose(
        np.asarray(fitted.theta, dtype=float), trace.theta[-1], rtol=0, atol=0
    )

    recomputed = e_step_multi(
        grid, weights, fitted.log_transition_matrix(grid), groups
    ).log_evidence
    assert recomputed == pytest.approx(trace.log_evidence[-1], rel=1e-12)

    # Monotonicity now covers every update, including the last one.
    assert trace.monotone_violation < 1e-8


def test_stopping_tolerance_is_scale_free_in_both_units():
    """Both the old and the new tolerance are invariant in the sample size.

    Written to record a NEGATIVE result, because an external audit asserted that
    the dataset-relative threshold `|dL| <= tol*|L|` had to be normalised by the
    number of transitions "because total log-likelihood scales with N(L-1)". It
    does -- but so does dL, and the ratio is therefore already scale-free. The
    per-transition form is a change of units for interpretability, not a fix.

    An earlier version of this test asserted that the two rules stop at different
    iterations, and its own vacuity guard caught it: they agree, because both are
    invariant. So the property actually worth pinning is the invariance itself,
    which is what is checked here -- if a future change breaks it, that is a real
    regression in either unit.
    """
    grid, weights = make_grid(8.0, 201)
    prior = LaplaceAR1(RHO)
    rng = rng_for("test-em-tol-scale")
    _, X, alpha, delta = _sample(prior, 32, 0.3, rng)
    X4 = np.concatenate([X, X, X, X])          # same law, four times the edges

    traces = []
    for data in (X, X4):
        start = MixtureInnovationKernel.init(
            3, rho=0.2, var=0.9, rng=rng_for("test-em-tol-init")
        )
        _, tr = fit_em(start, grid, weights, [(data, alpha, delta)],
                       n_iters=30, tol=0.0)
        traces.append(tr)
    t1, t4 = traces

    assert t4.n_edges == 4 * t1.n_edges
    L1 = np.asarray(t1.log_evidence)
    L4 = np.asarray(t4.log_evidence)
    d1, d4 = np.diff(L1), np.diff(L4)

    # Relative-to-total: invariant, which is the audit's claim refuted.
    np.testing.assert_allclose(np.abs(d4) / np.abs(L4[:-1]),
                               np.abs(d1) / np.abs(L1[:-1]), rtol=1e-9)
    # Per transition: invariant too, so the change of units is safe.
    np.testing.assert_allclose(d4 / t4.n_edges, d1 / t1.n_edges, rtol=1e-9)


@pytest.mark.parametrize("kernel_name", ["gaussian", "mixture"])
def test_a_decrease_is_not_reported_as_convergence(kernel_name):
    """A run that ends by exhausting its budget must say so.

    The old test accepted `|dL| <= tol*|L|`, so a small DECREASE ended the loop
    and the run was returned looking exactly like a converged one. Two things are
    checked: an impossible tolerance must produce a censored trace rather than a
    converged one, and a generous tolerance must produce a converged one -- so
    the status tracks reality instead of being constant.
    """
    grid, weights = make_grid(8.0, 201)
    prior = LaplaceAR1(RHO)
    rng = rng_for("test-em-censored", kernel_name)
    _, X, alpha, delta = _sample(prior, 24, 0.3, rng)
    groups = [(X, alpha, delta)]
    start = {
        "gaussian": GaussianAR1Kernel(0.2, 0.9),
        "mixture": MixtureInnovationKernel.init(
            3, rho=0.2, var=0.9, rng=rng_for("test-em-censored-init")
        ),
    }[kernel_name]

    _, censored = fit_em(start, grid, weights, groups, n_iters=6, tol=0.0)
    assert censored.stop_reason == "censored" and not censored.converged
    # n_iters + 1: the budget-exhausted path scores the final proposal too, so
    # the returned kernel is one the trace has an evidence for. That alignment is
    # tested by test_returned_kernel_is_the_one_the_trace_ends_on.
    assert len(censored.log_evidence) == 7

    _, done = fit_em(start, grid, weights, groups, n_iters=500, tol=1e-3)
    assert done.stop_reason == "converged" and done.converged
    assert len(done.log_evidence) < 500

    # And the increment that stopped it was an increase, not a decrease.
    assert done.monotone_violation < 1e-8
    gain = (done.log_evidence[-1] - done.log_evidence[-2]) / done.n_edges
    assert gain >= 0.0, f"stopped on a decrease of {gain:.3e} nats/edge"


def test_em_recovers_gaussian_parameters():
    """A well-specified fit lands on the truth from a bad initialization."""
    grid, weights = make_grid(8.0, 301)
    prior = GaussianAR1(RHO)
    rng = rng_for("test-em-recover")
    groups = []
    for t in (0.1, 0.3, 0.6):
        _, X, alpha, delta = _sample(prior, 200, t, rng, n_sites=24)
        groups.append((X, alpha, delta))

    fitted, _ = fit_em(GaussianAR1Kernel(0.1, 1.2), grid, weights, groups, n_iters=100)
    assert abs(fitted.rho - RHO) < 0.02
    assert abs(fitted.q - (1.0 - RHO**2)) < 0.02


def test_clean_statistics_recover_the_least_squares_estimate():
    """The t -> 0 limit reduces to ordinary AR(1) regression."""
    grid, weights = make_grid(8.0, 601)
    prior = GaussianAR1(RHO)
    rng = rng_for("test-em-clean")
    A = np.stack([prior.sample(rng, 40) for _ in range(400)])

    fitted, _ = fit_clean(GaussianAR1Kernel(0.1, 1.0), grid, A)
    src, dst = A[:, :-1].ravel(), A[:, 1:].ravel()
    rho_ols = float(src @ dst / (src @ src))
    q_ols = float(np.mean((dst - rho_ols * src) ** 2))

    assert abs(fitted.rho - rho_ols) < 5e-3
    assert abs(fitted.q - q_ols) < 5e-3


def test_small_noise_em_approaches_clean_mle():
    """EM at a small noise level agrees with the clean-data MLE.

    The continuity claimed in Section 3 ("the two limits"): as t -> 0 the
    likelihood factors sharpen, Xi degenerates to a transition histogram, and
    the estimator tends to the complete-data one.
    """
    grid, weights = make_grid(8.0, 801)
    prior = GaussianAR1(RHO)
    rng = rng_for("test-em-smallt")
    A = np.stack([prior.sample(rng, 24) for _ in range(300)])
    clean, _ = fit_clean(GaussianAR1Kernel(0.1, 1.0), grid, A)

    alpha, delta = alpha_delta(0.05)
    X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
    noisy, _ = fit_em(
        GaussianAR1Kernel(0.1, 1.0), grid, weights, [(X, alpha, delta)], n_iters=80
    )
    assert abs(noisy.rho - clean.rho) < 0.03
    assert abs(noisy.q - clean.q) < 0.03


def test_mixture_kernel_recovers_a_heavy_tail():
    """A mixture that never heard of Laplace reproduces its excess kurtosis."""
    grid, weights = make_grid(8.0, 301)
    prior = LaplaceAR1(RHO)
    rng = rng_for("test-em-mixture-tail")
    A = np.stack([prior.sample(rng, 48) for _ in range(600)])

    fitted, _ = fit_clean(
        MixtureInnovationKernel.init(
            6, rho=0.3, var=0.8, rng=rng_for("test-em-mixture-init")
        ),
        grid, A, n_iters=60,
    )
    mom = fitted.innovation_moments
    assert abs(fitted.rho - RHO) < 0.03
    assert abs(mom["innovation_var"] - (1.0 - RHO**2)) < 0.03
    assert mom["innovation_excess_kurtosis"] > 2.0  # true value 3.0


def test_mdn_gradient_matches_finite_differences():
    """The fused Fisher-identity gradient through the network is correct."""
    grid, weights = make_grid(6.0, 101)
    prior = LaplaceAR1(RHO)
    rng = rng_for("test-em-mdn")
    _, X, alpha, delta = _sample(prior, 4, 0.4, rng)

    from src.kernels import _logsumexp_axis0, _quad_weights

    mdn = MDNKernel.init(2, 12, rng_for("test-em-mdn-init"))
    stats = e_step(grid, weights, mdn.log_transition_matrix(grid), X, alpha, delta)

    comp = mdn._component_logs(grid)
    resp = np.exp(comp - _logsumexp_axis0(comp)[None, :, :])
    _, m, log_s2, cache = mdn._heads(grid)
    log_k = _logsumexp_axis0(comp)
    log_w = np.log(_quad_weights(grid))
    p_col = np.exp(
        log_k + log_w[:, None] - _logsumexp_axis0(log_k + log_w[:, None])[None, :]
    )
    xi = stats.xi - stats.xi.sum(axis=0)[None, :] * p_col
    d = grid[None, :, None] - m[:, None, :]
    inv_s2 = np.exp(-log_s2)[:, None, :]
    grad_out = np.concatenate([
        np.einsum("ckj,kj->cj", resp, xi).T,
        np.einsum("ckj,ckj,kj->cj", resp, d * inv_s2, xi).T,
        0.5 * np.einsum("ckj,ckj,kj->cj", resp, d**2 * inv_s2 - 1.0, xi).T,
    ], axis=1)
    grads = mdn.net.backward(cache, grad_out)

    eps = 1e-6
    probe = rng_for("test-em-mdn-probe")
    for layer in (0, 2):
        P = mdn.net.params[layer]
        for _ in range(3):
            i = int(probe.integers(0, P.shape[0]))
            j = int(probe.integers(0, P.shape[1]))
            old = P[i, j]
            P[i, j] = old + eps
            hi = e_step(grid, weights, mdn.log_transition_matrix(grid),
                        X, alpha, delta).log_evidence
            P[i, j] = old - eps
            lo = e_step(grid, weights, mdn.log_transition_matrix(grid),
                        X, alpha, delta).log_evidence
            P[i, j] = old
            fd = (hi - lo) / (2 * eps)
            if abs(fd) > 1e-6:
                assert abs(grads[layer][i, j] - fd) / abs(fd) < 1e-4


# ----------------------------------------------------------------------------
# Independent cross-checks: routes that share no code with the E-step
# ----------------------------------------------------------------------------

def _brute_force_xi(grid, weights, log_K, X, alpha, delta, log_mu):
    """Pairwise marginals by enumerating every discretized configuration.

    Deliberately shares nothing with the forward-backward recursion: it builds
    the joint mass over the M^n configuration space and marginalizes by
    summation. Only feasible for a tiny grid and chain, which is the point --
    it pins the E-step against a route with no messages in it at all.
    """
    import itertools

    m, n = len(grid), X.shape[1]
    K, mu = np.exp(log_K), np.exp(log_mu)
    xi = np.zeros((m, m))
    total_log_z = 0.0

    for x in X:
        ell = np.exp(
            -0.5 * (x[:, None] - alpha * grid[None, :]) ** 2 / delta
        ) / np.sqrt(2.0 * np.pi * delta)
        joint = {}
        for cfg in itertools.product(range(m), repeat=n):
            val = np.prod([weights[j] for j in cfg]) * mu[cfg[0]] * ell[0, cfg[0]]
            for i in range(1, n):
                val *= K[cfg[i], cfg[i - 1]] * ell[i, cfg[i]]
            joint[cfg] = val
        z = sum(joint.values())
        total_log_z += np.log(z)
        for cfg, val in joint.items():
            for i in range(n - 1):
                xi[cfg[i + 1], cfg[i]] += val / z
    return xi, total_log_z


def test_xi_matches_brute_force_enumeration():
    """The whole E-step, against explicit enumeration of the joint posterior."""
    grid, weights = make_grid(3.0, 11)
    log_mu = -0.5 * grid**2 - 0.5 * np.log(2.0 * np.pi)
    prior = GaussianAR1(0.7)
    rng = rng_for("test-em-bruteforce")
    alpha, delta = alpha_delta(0.5)
    A = np.stack([prior.sample(rng, 4) for _ in range(3)])
    X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
    log_k = prior.log_transition_matrix(grid)

    stats = e_step(grid, weights, log_k, X, alpha, delta, log_mu)
    xi_bf, log_z_bf = _brute_force_xi(grid, weights, log_k, X, alpha, delta, log_mu)

    assert np.abs(stats.xi - xi_bf).max() / xi_bf.max() < 1e-12
    assert abs(stats.log_evidence - log_z_bf) / abs(log_z_bf) < 1e-12


def test_evidence_matches_monte_carlo_on_a_nongaussian_chain():
    """Evidence on the Laplace chain, where no closed form exists.

    The Gaussian test above pins the evidence against an analytic formula; this
    one pins it on a prior that has none, by importance sampling
    p_t(x) = E_{a ~ p_0}[prod_i N(x_i; alpha a_i, Delta)] directly from the
    clean chain. Tolerance is set from the Monte Carlo standard error, not
    guessed.
    """
    grid, weights = make_grid(10.0, 1201)
    prior = LaplaceAR1(0.8)
    rng = rng_for("test-em-mc")
    n = 4
    alpha, delta = alpha_delta(0.6)
    a_true = prior.sample(rng, n)
    x = alpha * a_true + np.sqrt(delta) * rng.standard_normal(n)

    stats = e_step(
        grid, weights, prior.log_transition_matrix(grid), x[None, :], alpha, delta
    )

    rng_mc = np.random.default_rng(12345)
    chunk, n_chunks = 200_000, 5
    acc = acc_sq = 0.0
    for _ in range(n_chunks):
        a = np.empty((chunk, n))
        a[:, 0] = rng_mc.standard_normal(chunk)
        for i in range(1, n):
            a[:, i] = 0.8 * a[:, i - 1] + rng_mc.laplace(0.0, prior.b, size=chunk)
        log_w = (
            -0.5 * (x[None, :] - alpha * a) ** 2 / delta
            - 0.5 * np.log(2.0 * np.pi * delta)
        ).sum(axis=1)
        w = np.exp(log_w)
        acc += w.sum()
        acc_sq += (w**2).sum()

    total = chunk * n_chunks
    mean = acc / total
    log_se = np.sqrt(max(acc_sq / total - mean**2, 0.0) / total) / mean
    assert abs(stats.log_evidence - np.log(mean)) < 5.0 * log_se


def test_em_fixed_point_is_stationary_for_the_marginal_likelihood():
    """EM converges to a stationary point of L, not merely of the M-step map.

    Checked through Fisher's identity, which computes grad L without reference
    to how the M-step chose its update -- so a M-step that converged to the
    wrong place would be caught here.
    """
    grid, weights = make_grid(8.0, 301)
    prior = GaussianAR1(0.8)
    rng = rng_for("test-em-stationary")
    groups = []
    for t in (0.2, 0.6):
        alpha, delta = alpha_delta(t)
        A = np.stack([prior.sample(rng, 20) for _ in range(120)])
        groups.append(
            (alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape), alpha, delta)
        )

    start = GaussianAR1Kernel(0.2, 0.9)
    grad_start = q_gradient(
        e_step_multi(grid, weights, start.log_transition_matrix(grid), groups),
        start.grad_log_transition_matrix(grid),
    )
    fitted, trace = fit_em(start, grid, weights, groups, n_iters=200, tol=0.0)
    grad_end = q_gradient(
        e_step_multi(grid, weights, fitted.log_transition_matrix(grid), groups),
        fitted.grad_log_transition_matrix(grid),
    )

    assert trace.monotone_violation < 1e-8
    assert np.linalg.norm(grad_end) < 1e-5 * np.linalg.norm(grad_start)


# ----------------------------------------------------------------------------
# Validity of the exp_07 comparison itself
# ----------------------------------------------------------------------------

def test_bp_reference_denoiser_matches_monte_carlo():
    """The yardstick exp_07 scores everything against.

    Every number in the EM-BP vs network comparison is a deviation from
    `bp_posterior_mean` under the true prior. If that reference were wrong,
    every number would be. Cross-checked here by importance sampling
    E[a | x] straight from the clean chain -- no BP involved.
    """
    from src.denoiser import bp_posterior_mean

    grid, weights = make_grid(10.0, 1201)
    prior = LaplaceAR1(0.8)
    rng = rng_for("test-ref-denoiser")
    n, t = 5, 0.5
    alpha, delta = alpha_delta(t)
    a_true = prior.sample(rng, n)
    x = alpha * a_true + np.sqrt(delta) * rng.standard_normal(n)

    m_bp = bp_posterior_mean(prior, grid, weights, x[None, :], t)[0]

    rng_mc = np.random.default_rng(777)
    chunk, n_chunks = 250_000, 8
    num, den = np.zeros(n), 0.0
    for _ in range(n_chunks):
        a = np.empty((chunk, n))
        a[:, 0] = rng_mc.standard_normal(chunk)
        for i in range(1, n):
            a[:, i] = 0.8 * a[:, i - 1] + rng_mc.laplace(0.0, prior.b, size=chunk)
        log_w = (-0.5 * (x[None, :] - alpha * a) ** 2 / delta).sum(axis=1)
        w = np.exp(log_w - log_w.max())
        num += (w[:, None] * a).sum(axis=0)
        den += w.sum()
    m_mc = num / den

    assert np.linalg.norm(m_bp - m_mc) / np.linalg.norm(m_mc) < 1e-2


def test_dsm_baseline_is_not_broken():
    """The score network must be able to learn, or the comparison is worthless.

    Trained on a *Gaussian* chain, where the exact denoiser is available in
    closed form and is linear in x, with generous data. If the baseline were
    subtly broken the exp_07 result would be an artifact rather than a
    sample-efficiency effect, so this is a load-bearing test.

    It also pins the complementarity of the two parameterizations that the
    write-up claims: eps recovers the mean as (x - sqrt(Delta) z)/alpha and so
    amplifies network error by sqrt(Delta)/alpha, which is small at low noise
    and large at high noise; x0 has the opposite profile.
    """
    from src.denoiser import dsm_posterior_mean, train_dsm_denoiser
    from src.exact_scores import exact_gaussian_posterior_mean

    n_sites = 8
    prior = GaussianAR1(0.8)
    sigma0 = prior.covariance(n_sites)
    t_train = (0.1, 0.4, 1.6)

    rng = rng_for("test-dsm-sound")
    A = np.stack([prior.sample(rng, n_sites) for _ in range(3000)])
    rng_test = rng_for("test-dsm-sound-test")
    A_test = np.stack([prior.sample(rng_test, n_sites) for _ in range(256)])

    errors = {}
    for mode in ("eps", "x0"):
        dsm = train_dsm_denoiser(
            A, t_train, rng_for("test-dsm-sound-train", mode),
            hidden=(128, 128), n_steps=12000, parameterization=mode,
        )
        per_t = {}
        for t in t_train:
            alpha, delta = alpha_delta(t)
            X = alpha * A_test + np.sqrt(delta) * rng_test.standard_normal(A_test.shape)
            m_exact = np.stack(
                [exact_gaussian_posterior_mean(x, sigma0, alpha, delta) for x in X]
            )
            m_net = dsm_posterior_mean(dsm, X, t)
            per_t[t] = float(
                np.linalg.norm(m_net - m_exact) / np.linalg.norm(m_exact)
            )
        errors[mode] = per_t

    best_mean = min(float(np.mean(list(v.values()))) for v in errors.values())
    assert best_mean < 0.25, f"baseline network cannot learn a linear denoiser: {errors}"

    # Complementarity: eps wins at low noise, x0 wins at high noise.
    assert errors["eps"][0.1] < errors["x0"][0.1]
    assert errors["x0"][1.6] < errors["eps"][1.6]


def test_mixture_m_step_reaches_a_local_maximum_of_q():
    """The ECM used by the headline kernel really maximizes Q.

    The Gaussian and Laplace M-steps are single closed-form solves and are
    checked above. The mixture M-step is an inner EM over the component label
    alternated with a weighted least-squares solve for rho, so "it looks like a
    maximizer" is a weaker claim there. Iterated to convergence on a fixed Xi,
    no nearby parameter may beat it.
    """
    grid, weights = make_grid(8.0, 201)
    prior = LaplaceAR1(RHO)
    rng = rng_for("test-mix-mstep")
    _, X, alpha, delta = _sample(prior, 40, 0.3, rng)

    start = MixtureInnovationKernel.init(
        3, rho=0.4, var=0.7, rng=rng_for("test-mix-mstep-init")
    )
    stats = e_step(grid, weights, start.log_transition_matrix(grid), X, alpha, delta)

    current = start
    q_prev = -np.inf
    for _ in range(60):
        current = current.m_step(stats, grid)
        q_now = q_value(stats, current.log_transition_matrix(grid))
        assert q_now >= q_prev - 1e-9, "inner ECM decreased Q"
        q_prev = q_now

    q_best = q_value(stats, current.log_transition_matrix(grid))
    perturb = rng_for("test-mix-mstep-perturb")
    for _ in range(30):
        cand = MixtureInnovationKernel(
            rho=current.rho + float(perturb.normal(0, 0.02)),
            pi=np.abs(current.pi + perturb.normal(0, 0.01, current.pi.shape)),
            mu=current.mu + perturb.normal(0, 0.02, current.mu.shape),
            s2=np.maximum(current.s2 + perturb.normal(0, 0.01, current.s2.shape), 1e-4),
        )
        cand = MixtureInnovationKernel(cand.rho, cand.pi / cand.pi.sum(),
                                       cand.mu, cand.s2)
        assert q_value(stats, cand.log_transition_matrix(grid)) <= q_best + 1e-7


def test_innovation_moments_match_sampling_from_the_fitted_mixture():
    """The moment formulas behind the reported kurtosis recovery.

    Section 4.8 of the compendium reports a recovered excess kurtosis; that
    number comes from a closed-form expression over the mixture components, so
    an error in it would silently corrupt the headline claim about recovering a
    heavy tail. Checked against draws from the very same mixture.
    """
    rng = rng_for("test-innovation-moments")
    kernel = MixtureInnovationKernel(
        rho=0.8,
        pi=np.array([0.5, 0.3, 0.2]),
        mu=np.array([0.15, -0.2, 0.05]),
        s2=np.array([0.05, 0.25, 0.9]),
    )
    mom = kernel.innovation_moments

    n = 4_000_000
    comp = rng.choice(len(kernel.pi), size=n, p=kernel.pi)
    eps = kernel.mu[comp] + np.sqrt(kernel.s2[comp]) * rng.standard_normal(n)

    m1 = eps.mean()
    m2 = eps.var()
    excess = ((eps - m1) ** 4).mean() / m2**2 - 3.0

    assert abs(mom["innovation_mean"] - m1) < 5e-3
    assert abs(mom["innovation_var"] - m2) / m2 < 5e-3
    assert abs(mom["innovation_excess_kurtosis"] - excess) < 2e-2


def test_mixture_rho_gradient_matches_finite_difference():
    """Guards the sign of d log K / d rho for the mixture kernel.

    e = u_k - rho u_j, so de/drho = -u_j and the two minus signs in
    d/drho log N(e; mu_c, s2_c) cancel: the derivative is *plus*
    u_j sum_c r_c (e - mu_c)/s2_c. The branch previously returned the negative
    of this, which a finite difference catches immediately -- the ratio came out
    at -1.0000000000239 and analytic + finite_difference vanished to 2e-10.

    Nothing in the ECM M-step calls this method (its rho block solves the
    weighted least-squares normal equations directly and had the right sign), so
    the bug was invisible to every recovery experiment. It corrupts only the
    routes that consume the analytic derivative: Fisher-gradient ascent, score
    tests, and the observed-information estimates of exp_22.
    """
    grid = np.array([-1.2, -0.4, 0.3, 1.1])
    kernel = MixtureInnovationKernel(
        rho=0.37,
        pi=np.array([0.35, 0.65]),
        mu=np.array([-0.25, 0.4]),
        s2=np.array([0.55, 1.3]),
    )

    h = 1e-6
    hi = replace(kernel, rho=kernel.rho + h).log_transition_matrix(grid)
    lo = replace(kernel, rho=kernel.rho - h).log_transition_matrix(grid)
    fd = (hi - lo) / (2 * h)
    analytic = kernel.grad_log_transition_matrix(grid)[0]

    np.testing.assert_allclose(analytic, fd, rtol=1e-5, atol=1e-7)


def test_mixture_rho_gradient_reduces_to_gaussian_case():
    """A one-component, zero-mean mixture *is* a Gaussian AR(1) kernel.

    Pins the two branches against each other so they can never again disagree
    by a sign: this is the invariant the sign bug violated.
    """
    grid = np.linspace(-3.0, 3.0, 25)
    rho, q = 0.63, 0.8
    mixture = MixtureInnovationKernel(
        rho=rho,
        pi=np.array([1.0]),
        mu=np.array([0.0]),
        s2=np.array([q]),
    )
    gaussian = GaussianAR1Kernel(rho=rho, q=q)

    np.testing.assert_allclose(
        mixture.grad_log_transition_matrix(grid)[0],
        gaussian.grad_log_transition_matrix(grid)[0],
        rtol=1e-12,
        atol=1e-12,
    )
