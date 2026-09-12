"""Clean-data priors.

Two families:

1. Chain priors (exactly Markov): linear AR(1)-type recursions
       a_1 ~ N(0, 1),   a_i = rho a_{i-1} + eps_i,   Var(eps_i) = 1 - rho^2,
   with Gaussian, Laplace, Student-t, or symmetric two-component Gaussian-mixture
   innovations. All share the same second-order structure
       Cov(a_i, a_j) = rho^{|i-j|}
   (unit marginal variance), so they differ *only* beyond second moments. The
   innovation's excess kurtosis is the scalar "non-Gaussianity knob":
     - Gaussian: 0
     - Laplace: +3 (heavy tails)
     - Student-t(nu): 6/(nu-4) for nu > 4 (heavy tails)
     - symmetric mixture with separation kappa: -2 kappa^2 (bimodality).

2. Gaussian but non-Markov priors (for Layer 4): AR(1) plus a global latent
   (low-rank violation) and AR(1) plus weak long-range precision coupling. These
   are Gaussian so the exact score is available by linear algebra at every t.

All chain priors expose `log_transition_matrix(grid)` returning
log K[out, in] = log p(a_out | a_in) for grid BP, and `sample(rng, n)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.special import gammaln


# ----------------------------------------------------------------------------
# Chain (exactly Markov) priors
# ----------------------------------------------------------------------------

class ChainPrior(Protocol):
    """A stationary linear-AR(1)-type Markov chain with unit marginal variance."""

    rho: float

    @property
    def name(self) -> str: ...

    @property
    def innovation_excess_kurtosis(self) -> float: ...

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray: ...

    def log_transition_matrix(self, grid: np.ndarray) -> np.ndarray: ...


def _outer_residual(grid: np.ndarray, rho: float) -> np.ndarray:
    """Matrix of residuals  e[out, in] = a_out - rho a_in  on the grid."""
    return grid[:, None] - rho * grid[None, :]


@dataclass(frozen=True)
class GaussianAR1:
    """a_i = rho a_{i-1} + N(0, q),  q = 1 - rho^2. Exactly Gaussian AR(1)."""

    rho: float

    @property
    def name(self) -> str:
        return "gaussian"

    @property
    def q(self) -> float:
        return 1.0 - self.rho**2

    @property
    def innovation_excess_kurtosis(self) -> float:
        return 0.0

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        a = np.empty(n)
        a[0] = rng.standard_normal()
        eps = np.sqrt(self.q) * rng.standard_normal(n - 1)
        for i in range(1, n):
            a[i] = self.rho * a[i - 1] + eps[i - 1]
        return a

    def log_transition_matrix(self, grid: np.ndarray) -> np.ndarray:
        e = _outer_residual(grid, self.rho)
        return -0.5 * e**2 / self.q - 0.5 * np.log(2.0 * np.pi * self.q)

    def covariance(self, n: int) -> np.ndarray:
        idx = np.arange(n)
        return self.rho ** np.abs(idx[:, None] - idx[None, :])


@dataclass(frozen=True)
class LaplaceAR1:
    """a_i = rho a_{i-1} + Laplace(0, b),  2 b^2 = 1 - rho^2 (variance matched)."""

    rho: float

    @property
    def name(self) -> str:
        return "laplace"

    @property
    def q(self) -> float:
        return 1.0 - self.rho**2

    @property
    def b(self) -> float:
        return float(np.sqrt(self.q / 2.0))

    @property
    def innovation_excess_kurtosis(self) -> float:
        return 3.0

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        a = np.empty(n)
        a[0] = rng.standard_normal()
        eps = rng.laplace(0.0, self.b, size=n - 1)
        for i in range(1, n):
            a[i] = self.rho * a[i - 1] + eps[i - 1]
        return a

    def log_transition_matrix(self, grid: np.ndarray) -> np.ndarray:
        e = _outer_residual(grid, self.rho)
        return -np.abs(e) / self.b - np.log(2.0 * self.b)


@dataclass(frozen=True)
class StudentTAR1:
    """a_i = rho a_{i-1} + s T_nu,  s^2 nu/(nu-2) = 1 - rho^2 (variance matched).

    Requires nu > 2 for a finite variance. Excess kurtosis is 6/(nu-4) for nu > 4
    and infinite for 2 < nu <= 4.
    """

    rho: float
    nu: float = 5.0

    @property
    def name(self) -> str:
        return f"student_t_nu{self.nu:g}"

    @property
    def q(self) -> float:
        return 1.0 - self.rho**2

    @property
    def scale(self) -> float:
        if self.nu <= 2.0:
            raise ValueError("Student-t innovations need nu > 2 for finite variance.")
        return float(np.sqrt(self.q * (self.nu - 2.0) / self.nu))

    @property
    def innovation_excess_kurtosis(self) -> float:
        return 6.0 / (self.nu - 4.0) if self.nu > 4.0 else float("inf")

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        a = np.empty(n)
        a[0] = rng.standard_normal()
        eps = self.scale * rng.standard_t(self.nu, size=n - 1)
        for i in range(1, n):
            a[i] = self.rho * a[i - 1] + eps[i - 1]
        return a

    def log_transition_matrix(self, grid: np.ndarray) -> np.ndarray:
        e = _outer_residual(grid, self.rho) / self.scale
        nu = self.nu
        log_norm = (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi)
            - np.log(self.scale)
        )
        return log_norm - 0.5 * (nu + 1.0) * np.log1p(e**2 / nu)


@dataclass(frozen=True)
class UniformAR1:
    """a_i = rho a_{i-1} + Uniform(-h, h),  h^2/3 = 1 - rho^2 (variance matched).

    The bounded-support case (question R3). It is the interesting one because
    its failure mode is different in kind from the others in this module: heavy
    tails (Laplace, Student-t) and bimodality (Gaussian mixture) both fail a
    Gaussian closure through *shape*, whereas a compactly supported innovation
    fails through *support* -- the true posterior is exactly zero outside a
    finite interval and no Gaussian message can represent that edge.

    Excess kurtosis is -1.2, which places it between the kappa = 0.6 and
    kappa = 0.9 mixtures. If |excess kurtosis| really is the one-number summary
    of closure difficulty, this prior should land between them; if the support
    mismatch matters independently, it should not.
    """

    rho: float

    @property
    def name(self) -> str:
        return "uniform"

    @property
    def q(self) -> float:
        return 1.0 - self.rho**2

    @property
    def half_width(self) -> float:
        return float(np.sqrt(3.0 * self.q))

    @property
    def innovation_excess_kurtosis(self) -> float:
        return -1.2

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        a = np.empty(n)
        a[0] = rng.standard_normal()
        h = self.half_width
        eps = rng.uniform(-h, h, size=n - 1)
        for i in range(1, n):
            a[i] = self.rho * a[i - 1] + eps[i - 1]
        return a

    def log_transition_matrix(self, grid: np.ndarray) -> np.ndarray:
        e = _outer_residual(grid, self.rho)
        h = self.half_width
        with np.errstate(divide="ignore"):
            out = np.where(np.abs(e) <= h, -np.log(2.0 * h), -np.inf)
        return out


@dataclass(frozen=True)
class GaussianMixtureAR1:
    """a_i = rho a_{i-1} + eps,  eps ~ 1/2 N(+m, s^2) + 1/2 N(-m, s^2).

    Parametrized by the separation fraction kappa in [0, 1):
        m^2 = kappa q,    s^2 = (1 - kappa) q,    q = 1 - rho^2,
    so Var(eps) = m^2 + s^2 = q for every kappa. kappa = 0 recovers the Gaussian
    chain; kappa -> 1 concentrates on two points. Excess kurtosis = -2 kappa^2
    (bimodal / platykurtic), the opposite sign of Laplace / Student-t.
    """

    rho: float
    kappa: float = 0.6

    @property
    def name(self) -> str:
        return f"gauss_mix_kappa{self.kappa:g}"

    @property
    def q(self) -> float:
        return 1.0 - self.rho**2

    @property
    def m(self) -> float:
        return float(np.sqrt(self.kappa * self.q))

    @property
    def s(self) -> float:
        return float(np.sqrt((1.0 - self.kappa) * self.q))

    @property
    def innovation_excess_kurtosis(self) -> float:
        return -2.0 * self.kappa**2

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        a = np.empty(n)
        a[0] = rng.standard_normal()
        signs = rng.choice([-1.0, 1.0], size=n - 1)
        eps = signs * self.m + self.s * rng.standard_normal(n - 1)
        for i in range(1, n):
            a[i] = self.rho * a[i - 1] + eps[i - 1]
        return a

    def log_transition_matrix(self, grid: np.ndarray) -> np.ndarray:
        e = _outer_residual(grid, self.rho)
        s2 = self.s**2
        log_norm = -0.5 * np.log(2.0 * np.pi * s2) + np.log(0.5)
        lp = -0.5 * (e - self.m) ** 2 / s2
        lm = -0.5 * (e + self.m) ** 2 / s2
        return log_norm + np.logaddexp(lp, lm)


# ----------------------------------------------------------------------------
# Gaussian, non-Markov priors (Layer 4)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class GaussianAR1PlusGlobal:
    """a = (y + beta g) / sqrt(1 + beta^2),  y ~ Gaussian AR(1)(rho),  g ~ N(0,1).

    Exactly Gaussian, unit marginal variance, but NOT Markov for beta != 0:
        Cov(a_i, a_j) = (rho^{|i-j|} + beta^2) / (1 + beta^2),  i != j.
    The normalization keeps the variance-preserving convention p_T ~ N(0, I).
    """

    rho: float
    beta: float

    @property
    def name(self) -> str:
        return f"ar1_global_beta{self.beta:g}"

    def covariance(self, n: int) -> np.ndarray:
        idx = np.arange(n)
        sigma_y = self.rho ** np.abs(idx[:, None] - idx[None, :])
        return (sigma_y + self.beta**2) / (1.0 + self.beta**2)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        y = GaussianAR1(self.rho).sample(rng, n)
        g = rng.standard_normal()
        return (y + self.beta * g) / np.sqrt(1.0 + self.beta**2)


@dataclass(frozen=True)
class GaussianLongRange:
    """Gaussian prior with precision Q = Q_AR1(rho) + gamma * Q_long.

    Q_AR1 is the tridiagonal AR(1) precision. Q_long adds weak couplings
    -exp(-|i-j| / ell_long) for |i-j| >= 2, with a diagonal compensation that
    keeps Q strictly diagonally dominant (hence positive definite). The
    covariance is inverted numerically and rescaled to unit diagonal, so the
    prior stays variance preserving. Not Markov for gamma != 0.
    """

    rho: float
    gamma: float
    ell_long: float = 8.0

    @property
    def name(self) -> str:
        return f"ar1_longrange_gamma{self.gamma:g}"

    def precision(self, n: int) -> np.ndarray:
        q = 1.0 - self.rho**2
        Q = np.zeros((n, n))
        # Tridiagonal AR(1) precision.
        Q[np.arange(n), np.arange(n)] = (1.0 + self.rho**2) / q
        Q[0, 0] = Q[n - 1, n - 1] = 1.0 / q
        off = -self.rho / q
        Q[np.arange(n - 1), np.arange(1, n)] = off
        Q[np.arange(1, n), np.arange(n - 1)] = off
        # Long-range coupling: negative off-diagonal entries at |i-j| >= 2.
        idx = np.arange(n)
        dist = np.abs(idx[:, None] - idx[None, :])
        long_off = -np.exp(-dist / self.ell_long)
        long_off[dist < 2] = 0.0
        Q_long = long_off.copy()
        # Diagonal compensation: strictly dominate the added couplings.
        Q_long[np.arange(n), np.arange(n)] = -long_off.sum(axis=1) * 1.05
        return Q + self.gamma * Q_long

    def covariance(self, n: int) -> np.ndarray:
        sigma = np.linalg.inv(self.precision(n))
        d = np.sqrt(np.diag(sigma))
        return sigma / np.outer(d, d)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        cov = self.covariance(n)
        chol = np.linalg.cholesky(cov)
        return chol @ rng.standard_normal(n)
