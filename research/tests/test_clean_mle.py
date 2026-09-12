"""Correctness of the grid-free clean-data estimators in src/clean_mle.py.

These exist to be the reference that `em.fit_clean` is checked against, so they
have to be right independently of the grid code -- a shared bug would make the
comparison vacuous. Each estimator is therefore pinned against something outside
the package: OLS against its own normal equations and against a direct likelihood
maximisation, and the mixture ECM against monotone ascent of the exact
clean-data log-likelihood.

Run:  python -m pytest tests/test_clean_mle.py -q
"""

from __future__ import annotations

import numpy as np

from src.bp_grid import make_grid
from src.clean_mle import (
    gaussian_log_lik,
    gaussian_ols,
    mixture_ecm_raw,
    mixture_log_lik,
)
from src.em import fit_clean
from src.kernels import GaussianAR1Kernel
from src.priors import GaussianAR1, LaplaceAR1
from src.utils import rng_for

RHO, N_SITES = 0.8, 32


def test_gaussian_ols_solves_its_own_normal_equations():
    """The residual must be orthogonal to the regressor -- the defining property."""
    rng = rng_for("test-ols-normal")
    A = np.stack([GaussianAR1(RHO).sample(rng, N_SITES) for _ in range(200)])
    rho, _ = gaussian_ols(A)

    x, y = A[:, :-1].ravel(), A[:, 1:].ravel()
    assert abs(float((y - rho * x) @ x)) / float(x @ x) < 1e-12


def test_gaussian_ols_maximizes_the_exact_log_likelihood():
    """No nearby (rho, q) beats it on the exact clean-data likelihood."""
    rng = rng_for("test-ols-max")
    A = np.stack([GaussianAR1(RHO).sample(rng, N_SITES) for _ in range(200)])
    rho, q = gaussian_ols(A)
    best = gaussian_log_lik(A, rho, q)

    perturb = rng_for("test-ols-max-perturb")
    for _ in range(30):
        cand_rho = rho + float(perturb.normal(0, 0.02))
        cand_q = max(q + float(perturb.normal(0, 0.02)), 1e-4)
        assert gaussian_log_lik(A, cand_rho, cand_q) <= best + 1e-9


def test_grid_clean_fit_converges_to_ols_under_refinement():
    """The grid-binned MLE approaches exact OLS as the grid is refined.

    This is what licenses `fit_clean` to be described as a clean-data MLE at all,
    and it bounds how much of the clean arm's error the grid can account for.
    """
    rng = rng_for("test-ols-refine")
    A = np.stack([GaussianAR1(RHO).sample(rng, N_SITES) for _ in range(256)])
    rho_ols, q_ols = gaussian_ols(A)

    errs = {}
    for M in (201, 401, 801):
        grid, _ = make_grid(8.0, M)
        k, _ = fit_clean(GaussianAR1Kernel(0.2, 0.8), grid, A)
        errs[M] = (abs(k.rho - rho_ols), abs(k.q - q_ols))

    assert errs[801][0] < errs[201][0], f"rho not converging to OLS: {errs}"
    assert errs[801][1] < errs[201][1], f"q not converging to OLS: {errs}"
    # At the grid the experiments actually use, the gap must be small enough that
    # it cannot explain an error of order 1e-3. See exp_06 --only clean_raw_mle.
    assert errs[401][1] < 1e-3


def test_mixture_ecm_raw_ascends_the_exact_log_likelihood():
    """Every sweep increases the exact clean-data likelihood, over several seeds."""
    for seed in range(4):
        rng = rng_for("test-ecm-ascent-data", seed)
        A = np.stack([LaplaceAR1(RHO).sample(rng, N_SITES) for _ in range(150)])
        res = mixture_ecm_raw(
            A, 4, rng_for("test-ecm-ascent-init", seed), n_iters=40
        )
        assert res["monotone_violation"] < 1e-8, (
            f"seed {seed}: ECM decreased the likelihood by "
            f"{res['monotone_violation']:.3e}"
        )


def test_mixture_ecm_raw_beats_its_initialisation_and_a_gaussian_fit():
    """A flexible mixture must explain Laplace innovations better than one Gaussian."""
    rng = rng_for("test-ecm-vs-gauss")
    A = np.stack([LaplaceAR1(RHO).sample(rng, N_SITES) for _ in range(400)])

    res = mixture_ecm_raw(A, 4, rng_for("test-ecm-vs-gauss-init"), n_iters=120)
    rho_g, q_g = gaussian_ols(A)

    assert res["log_lik"] > gaussian_log_lik(A, rho_g, q_g)
    assert res["trace"][-1] >= res["trace"][0]


def test_mixture_ecm_raw_recovers_a_heavy_tail():
    """The fitted innovation law must show the Laplace excess kurtosis of 3.

    Loose tolerance on purpose: at this budget a single draw carries real spread
    (the audit in commit 60a3080 measured 2.972 +- 0.074 over eight draws), so a
    tight assertion here would be a flake, not a stronger test.
    """
    rng = rng_for("test-ecm-kurtosis")
    A = np.stack([LaplaceAR1(RHO).sample(rng, N_SITES) for _ in range(2000)])
    res = mixture_ecm_raw(A, 8, rng_for("test-ecm-kurtosis-init"), n_iters=200)

    assert abs(res["innovation_excess_kurtosis"] - 3.0) < 1.2
    assert abs(res["innovation_var"] - (1.0 - RHO**2)) < 0.05


def test_mixture_log_lik_matches_a_direct_sum():
    """Guards the vectorised logsumexp against a transparent loop."""
    rng = rng_for("test-mixture-loglik")
    A = np.stack([LaplaceAR1(RHO).sample(rng, 8) for _ in range(10)])
    pi = np.array([0.3, 0.45, 0.25])
    mu = np.array([-0.2, 0.05, 0.3])
    s2 = np.array([0.2, 0.5, 0.9])
    rho = 0.77

    x, y = A[:, :-1].ravel(), A[:, 1:].ravel()
    direct = 0.0
    for xi, yi in zip(x, y):
        e = yi - rho * xi
        dens = np.sum(pi / np.sqrt(2 * np.pi * s2) * np.exp(-0.5 * (e - mu) ** 2 / s2))
        direct += np.log(dens)

    assert abs(mixture_log_lik(A, rho, pi, mu, s2) - direct) < 1e-9
