"""Clean-data maximum likelihood on the *raw* transitions, with no grid at all.

This is the reference the rest of the package was missing. `em.fit_clean` answers
Marc's "no noising is needed" remark, but it answers it through
`em.clean_statistics`, which bins the observed transitions onto the same grid the
BP E-step uses. That is deliberate -- it puts the clean and noised arms on
identical M-step code, so their difference is the channel and not the estimator --
but it means the clean arm is the MLE of a *binned* objective, and any residual
error it shows is open to being blamed on the grid.

The estimators here remove that ambiguity by never discretising:

    gaussian_ols          exact closed-form MLE for a Gaussian AR(1)
    mixture_ecm_raw       ECM on the raw pairs for a Gaussian-mixture innovation

Measured against `fit_clean` on identical chains (N = 64...1024, 12 replicates,
M = 401), the two agree to RMS 4.4e-4 in the innovation variance and 2.1e-4 in
rho, while the estimation errors themselves run 2.7e-3 to 1.1e-2. The grid is
therefore an order of magnitude too small to explain the clean arm's error, which
is what `experiments/exp_06 --only clean_raw_mle` reports.
"""

from __future__ import annotations

import numpy as np

_VAR_FLOOR = 1e-6


def _pairs(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten clean chains into (x, y) = (a_{i-1}, a_i) transition pairs."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    return A[:, :-1].ravel(), A[:, 1:].ravel()


def gaussian_ols(A: np.ndarray) -> tuple[float, float]:
    """Exact clean-data MLE for K(a'|a) = N(a'; rho a, q). Returns (rho, q).

    Regression through the origin, so

        rho = sum a_{i-1} a_i / sum a_{i-1}^2,
        q   = sum (a_i - rho a_{i-1})^2 / [N (n-1)].

    `q` uses the MLE normalisation (divide by the number of edges, not by
    edges - 1), because that is what the grid M-step in
    `GaussianAR1Kernel.m_step` maximises and the point of this function is to be
    comparable to it.
    """
    x, y = _pairs(A)
    rho = float(x @ y / (x @ x))
    q = float(np.sum((y - rho * x) ** 2) / x.size)
    return rho, max(q, _VAR_FLOOR)


def gaussian_log_lik(A: np.ndarray, rho: float, q: float) -> float:
    """Exact clean-data transition log-likelihood, summed over edges.

    Excludes the initial-law term, which is held fixed and shared by every
    estimator being compared, so differences are meaningful.
    """
    x, y = _pairs(A)
    e = y - rho * x
    return float(-0.5 * np.sum(e**2) / q - 0.5 * x.size * (np.log(2 * np.pi) + np.log(q)))


def mixture_log_lik(
    A: np.ndarray, rho: float, pi: np.ndarray, mu: np.ndarray, s2: np.ndarray
) -> float:
    """Exact clean-data transition log-likelihood for a mixture innovation."""
    from scipy.special import logsumexp

    x, y = _pairs(A)
    e = y - rho * x
    comp = (
        np.log(pi)[:, None]
        - 0.5 * (np.log(2 * np.pi) + np.log(s2))[:, None]
        - 0.5 * (e[None, :] - mu[:, None]) ** 2 / s2[:, None]
    )
    return float(np.sum(logsumexp(comp, axis=0)))


def mixture_ecm_raw(
    A: np.ndarray,
    n_components: int,
    rng: np.random.Generator,
    n_iters: int = 200,
    tol: float = 0.0,
    rho_init: float = 0.3,
    var_init: float = 0.8,
) -> dict:
    """ECM for K(a'|a) = sum_c pi_c N(a' - rho a; mu_c, s2_c) on raw pairs.

    Alternates, at each sweep,

        r_nc  ∝ pi_c N(y_n - rho x_n; mu_c, s2_c)
        pi_c  = mean_n r_nc
        mu_c  = sum_n r_nc (y_n - rho x_n) / sum_n r_nc
        s2_c  = sum_n r_nc (y_n - rho x_n - mu_c)^2 / sum_n r_nc
        rho   = [sum_nc r_nc x_n (y_n - mu_c)/s2_c] / [sum_nc r_nc x_n^2/s2_c]

    Each block is a conditional maximiser at the others' current values, so the
    exact clean-data log-likelihood ascends monotonically. Component means are
    left unconstrained, matching `MixtureInnovationKernel` -- recentring is not a
    conditional maximiser and breaks the ascent.

    Returns the fitted parameters, the exact log-likelihood trace, the moments of
    the fitted innovation law, and the largest likelihood decrease observed.
    """
    from scipy.special import logsumexp

    x, y = _pairs(A)
    n = x.size

    rho = float(rho_init)
    mu = np.linspace(-np.sqrt(var_init), np.sqrt(var_init), n_components)
    mu = mu + 0.05 * np.sqrt(var_init) * rng.standard_normal(n_components)
    s2 = np.full(n_components, float(var_init))
    pi = np.full(n_components, 1.0 / n_components)

    trace = []
    for _ in range(n_iters):
        e = y - rho * x
        comp = (
            np.log(pi)[:, None]
            - 0.5 * (np.log(2 * np.pi) + np.log(s2))[:, None]
            - 0.5 * (e[None, :] - mu[:, None]) ** 2 / s2[:, None]
        )
        denom = logsumexp(comp, axis=0)
        trace.append(float(np.sum(denom)))
        r = np.exp(comp - denom[None, :])          # (C, n)

        mass = np.maximum(r.sum(axis=1), 1e-300)
        pi = mass / n
        mu = (r * e[None, :]).sum(axis=1) / mass
        dev = e[None, :] - mu[:, None]
        s2 = np.maximum((r * dev**2).sum(axis=1) / mass, _VAR_FLOOR)

        # rho block, at the just-updated mixture: weighted least squares with
        # component-specific offsets.
        w = r / s2[:, None]
        num = float(np.sum(w * x[None, :] * (y[None, :] - mu[:, None])))
        den = float(np.sum(w * x[None, :] ** 2))
        rho = num / den

        if tol > 0.0 and len(trace) > 1:
            prev, cur = trace[-2], trace[-1]
            if abs(cur - prev) <= tol * abs(prev):
                break

    final = mixture_log_lik(A, rho, pi, mu, s2)
    trace.append(final)

    m1 = float(pi @ mu)
    m2 = float(pi @ (s2 + mu**2) - m1**2)
    cen = mu - m1
    m4 = float(pi @ (3 * s2**2 + 6 * s2 * cen**2 + cen**4))
    d = np.diff(np.asarray(trace))
    return {
        "rho": rho,
        "pi": pi,
        "mu": mu,
        "s2": s2,
        "log_lik": final,
        "trace": trace,
        "innovation_mean": m1,
        "innovation_var": m2,
        "innovation_excess_kurtosis": m4 / m2**2 - 3.0,
        "monotone_violation": float(max(0.0, -d.min())) if d.size else 0.0,
    }
