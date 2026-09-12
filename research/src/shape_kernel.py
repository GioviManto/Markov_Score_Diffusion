"""A transition kernel with a genuine *shape* parameter at fixed variance.

Why this module exists
----------------------
The project claims that the OU channel destroys higher-order information faster than
second-order information, on the strength of `exp_06`: Fisher information for the innovation
variance ``q`` falls by 142x between t = 0.05 and t = 1.6 while information for the
correlation ``rho`` falls by only 26x.

An external review pointed out that this does not support the claim. The experiment runs in a
**Gaussian** AR(1) model with parameters ``(rho, q)``, and ``q`` is a second-order scale
parameter -- not a higher-order shape parameter. What was measured is that variance
information decays faster than correlation information, both of which are second order. The
sentence about higher-order structure needs a shape parameter, and there was none.

The generalised Gaussian supplies one. With

    p(e) = beta / (2 s Gamma(1/beta)) * exp( -(|e|/s)^beta ),

``beta`` controls shape alone once ``s`` is chosen to hold the variance at ``q``:

    Var(e) = s^2 Gamma(3/beta) / Gamma(1/beta)   =>   s(q, beta) = sqrt( q Gamma(1/beta) / Gamma(3/beta) ),

so ``beta = 1`` is Laplace, ``beta = 2`` is Gaussian, and every member of the family has
exactly the same variance. Excess kurtosis is
``Gamma(5/beta) Gamma(1/beta) / Gamma(3/beta)^2 - 3`` -- 3 at beta = 1, 0 at beta = 2 -- so
``beta`` moves the fourth moment while leaving the second fixed. That is precisely the
separation the claim needs and the ``(rho, q)`` parameterisation cannot provide.

What the experiment then reports
--------------------------------
Comparing raw diagonal entries ``J[beta, beta]`` against ``J[rho, rho]`` would repeat a
second mistake the review identified: diagonal comparisons ignore nuisance coupling. The
quantity that answers "how much information about shape survives" is the **efficient**
information, obtained by projecting out the nuisance directions ``eta = (rho, q)``:

    I_{beta beta . eta} = J_{beta beta} - J_{beta eta} J_{eta eta}^{-1} J_{eta beta}.

That is the inverse of the ``(beta, beta)`` entry of ``J^{-1}``, and it is what bounds the
variance of any estimator of ``beta`` that must also estimate ``rho`` and ``q``.

This module provides only what the Fisher study needs: the kernel, and its exact gradient
with respect to ``(rho, q, beta)``. There is no M-step, because nothing here is fitted by EM
-- the information is evaluated at the truth. `tests/test_shape_kernel.py` checks every
gradient component against central finite differences, which is the only way hand-derived
derivatives of this shape should be trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import digamma, gammaln


def _log_scale(q: float, beta: float) -> float:
    """log s(q, beta) with s chosen so the innovation variance is exactly q."""
    return 0.5 * (np.log(q) + gammaln(1.0 / beta) - gammaln(3.0 / beta))


def generalized_gaussian_scale(q: float, beta: float) -> float:
    return float(np.exp(_log_scale(q, beta)))


def generalized_gaussian_excess_kurtosis(beta: float) -> float:
    """Gamma(5/b)Gamma(1/b)/Gamma(3/b)^2 - 3: 3 at b=1 (Laplace), 0 at b=2 (Gaussian)."""
    return float(
        np.exp(gammaln(5.0 / beta) + gammaln(1.0 / beta) - 2.0 * gammaln(3.0 / beta)) - 3.0
    )


@dataclass(frozen=True)
class GeneralizedGaussianKernel:
    """K(a' | a) = p_beta(a' - rho a) with Var(innovation) = q for every beta.

    Parameter vector is ``(rho, q, beta)``. ``beta`` is the shape coordinate: it changes the
    fourth moment at fixed second moment, which is exactly what the higher-order-information
    claim requires and what ``(rho, q)`` alone cannot express.
    """

    rho: float
    q: float
    beta: float

    @property
    def name(self) -> str:
        return f"gengauss_beta{self.beta:g}"

    @property
    def theta(self) -> np.ndarray:
        return np.array([self.rho, self.q, self.beta], dtype=float)

    @property
    def scale(self) -> float:
        return generalized_gaussian_scale(self.q, self.beta)

    @property
    def innovation_excess_kurtosis(self) -> float:
        return generalized_gaussian_excess_kurtosis(self.beta)

    # -- density ------------------------------------------------------------

    def _residual(self, grid: np.ndarray) -> np.ndarray:
        return grid[:, None] - self.rho * grid[None, :]

    def log_transition_matrix(self, grid: np.ndarray) -> np.ndarray:
        e = self._residual(grid)
        log_s = _log_scale(self.q, self.beta)
        u = np.abs(e) * np.exp(-log_s)
        log_norm = np.log(self.beta) - np.log(2.0) - log_s - gammaln(1.0 / self.beta)
        return log_norm - u**self.beta

    # -- gradient -----------------------------------------------------------

    def grad_log_transition_matrix(self, grid: np.ndarray) -> np.ndarray:
        """(3, M, M) gradient of log K with respect to (rho, q, beta).

        Derived from ``log K = log b - log 2 - log s - lgamma(1/b) - u^b`` with
        ``u = |e|/s`` and ``e = a' - rho a``:

            d/drho : e enters as ``-a``, and ``d(-u^b)/de = -b u^(b-1) sign(e)/s``,
                     so the term is ``+ b u^(b-1) sign(e) a / s``.

            d/dq   : ``s`` depends on q only through ``sqrt(q)``, so ``ds/dq = s/(2q)`` and
                     ``dlogK/dq = (b u^b - 1) / (2q)``.

            d/dbeta: three contributions -- the explicit ``log b`` and ``-lgamma(1/b)``, the
                     dependence of ``s`` on ``b`` at fixed variance, and the exponent itself.
                     Writing ``log s = 0.5 log q + log c(b)`` with
                     ``log c = 0.5 [lgamma(1/b) - lgamma(3/b)]``,

                         dlogc/db = ( -psi(1/b) + 3 psi(3/b) ) / (2 b^2),

                     and the exponent contributes ``-u^b [log u - b dlogc/db]``.

        The ``|e|`` kink means the rho-derivative is undefined exactly at ``e = 0``, which on
        a symmetric grid is the diagonal. It is assigned zero there, which is the correct
        subgradient by symmetry and is what the finite-difference test confirms.

        **Validity range.** That reasoning holds for ``beta >= 1``. Below 1 the one-sided
        derivatives at ``e = 0`` genuinely diverge rather than approaching a finite subgradient
        the way ``|e|`` does at ``beta = 1``, so the zero assigned here is a convention rather
        than a subgradient. `exp_22` sweeps ``beta`` in [1, 3] and never enters that region;
        anything that does should revisit this line rather than trust it.
        """
        b, q = self.beta, self.q
        e = self._residual(grid)
        log_s = _log_scale(q, b)
        s = np.exp(log_s)
        abs_e = np.abs(e)
        u = abs_e / s

        # u**b and u**(b-1), guarding u = 0 where the power is singular for b < 1.
        with np.errstate(divide="ignore", invalid="ignore"):
            ub = np.where(u > 0, u**b, 0.0)
            ubm1 = np.where(u > 0, u ** (b - 1.0), 0.0)
            log_u = np.where(u > 0, np.log(np.where(u > 0, u, 1.0)), 0.0)

        out = np.empty((3, len(grid), len(grid)))

        # d/drho
        out[0] = b * ubm1 * np.sign(e) * grid[None, :] / s

        # d/dq
        out[1] = (b * ub - 1.0) / (2.0 * q)

        # d/dbeta
        dlogc_db = (-digamma(1.0 / b) + 3.0 * digamma(3.0 / b)) / (2.0 * b**2)
        explicit = 1.0 / b + digamma(1.0 / b) / b**2 - dlogc_db
        out[2] = explicit - ub * (log_u - b * dlogc_db)

        return out
