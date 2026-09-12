"""A non-Gaussian, non-Markov prior with an exact posterior mean.

The estimator in this project assumes the clean data is a Markov chain. That assumption has
never been violated in any experiment, which is the largest gap in the story: `exp_04`
analyses the score *residual* of known non-Markov Gaussian priors, but nothing has ever
fitted a kernel to data the chain family cannot represent.

Testing that needs a prior which is neither Gaussian nor Markov and still has a computable
reference. Gaussian non-Markov priors (`GaussianAR1PlusGlobal`, `GaussianLongRange` in
`priors`) give the first half for free -- the exact score is linear algebra -- but they leave
the interesting corner untouched, because a Gaussian chain is where a Gaussian baseline is
already exact. Bolting a global latent onto a *Laplace* chain covers the corner:

    a = (y + beta g) / sqrt(1 + beta^2),   y a Laplace AR(1),   g ~ N(0, 1),

which is non-Markov for beta != 0 (every site shares g) and non-Gaussian for every beta, and
whose covariance is the same (rho^|i-j| + beta^2)/(1 + beta^2) as the Gaussian version, so
the two are directly comparable at matched second moments.

Why the reference is exact, and cheap
-------------------------------------
Conditional on ``g`` the prior is Markov again, so grid BP applies. The naive way to use
that is to build a kernel with a g-dependent mean shift, which needs new per-family
innovation code. A change of variables avoids it entirely. With ``c = sqrt(1 + beta^2)``,
``x = alpha a + sqrt(Delta) z`` gives

    c x - alpha beta g  =  alpha y + c sqrt(Delta) z,

so defining ``x' = c x - alpha beta g``, the pair ``(y, x')`` is *exactly* the standard
setup: a plain Laplace AR(1) chain observed through an OU channel with the same forward
factor ``alpha`` and inflated noise variance ``c^2 Delta``. No shifted kernel, no new
innovation density, no second implementation of anything -- `grid_bp_batch` runs unmodified
on ``x'``, with the prior's own ``log_transition_matrix`` and its own ``N(0,1)`` initial law,
which is the correct law for ``y_1``.

Then

    E[a | x] = int p(g | x) E[a | x, g] dg,     E[a | x, g] = (E[y | x', g] + beta g) / c,
    p(g | x) propto p(x | g) phi(g) propto p_{x'}(c x - alpha beta g | g) phi(g),

the Jacobian ``c^n`` of ``x -> x'`` being independent of ``g`` and cancelling. The
conditional evidence ``p(x' | g)`` is what `grid_bp_batch(..., return_evidence=True)`
returns, so both factors come out of one call per quadrature node.

A Gauss-Hermite rule in ``g`` costs one batched BP pass per node, so the whole reference is
tens of passes -- minutes, not hours. Its two error sources are the grid, shared with every
other number in the project, and the ``g`` quadrature, which is refinable and measured.

Two independent checks pin it, both in `tests/test_nonmarkov.py`: at ``beta = 0`` it must
reduce exactly to plain grid BP, and with Gaussian innovations it must match
`exact_scores.exact_gaussian_posterior_mean` under `GaussianAR1PlusGlobal.covariance` -- a
completely separate computation of the same quantity by linear algebra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.bp_grid import grid_bp_batch
from src.priors import GaussianAR1, LaplaceAR1


@dataclass(frozen=True)
class ChainPlusGlobal:
    """`base` chain prior plus a shared global latent, variance preserving.

    a = (y + beta g)/sqrt(1 + beta^2) with y ~ base and g ~ N(0, 1) drawn once per chain.
    Markov exactly when beta = 0. Unit marginal variance for every beta, and

        Cov(a_i, a_j) = (rho^|i-j| + beta^2) / (1 + beta^2),   i != j,

    which is the same covariance as `priors.GaussianAR1PlusGlobal` at matched beta -- so a
    Laplace and a Gaussian version of this prior differ only beyond second moments, exactly
    the controlled comparison the rest of the project uses.
    """

    base: object
    beta: float

    @property
    def name(self) -> str:
        return f"{self.base.name}_global_beta{self.beta:g}"

    @property
    def rho(self) -> float:
        return self.base.rho

    @property
    def scale(self) -> float:
        """c = sqrt(1 + beta^2), the variance-preserving normaliser."""
        return float(np.sqrt(1.0 + self.beta**2))

    @property
    def innovation_excess_kurtosis(self) -> float:
        return self.base.innovation_excess_kurtosis

    def covariance(self, n: int) -> np.ndarray:
        idx = np.arange(n)
        sigma_y = self.rho ** np.abs(idx[:, None] - idx[None, :])
        return (sigma_y + self.beta**2) / (1.0 + self.beta**2)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        y = self.base.sample(rng, n)
        g = float(rng.standard_normal())
        return (y + self.beta * g) / self.scale

    def sample_batch(self, rng: np.random.Generator, n_chains: int, n: int) -> np.ndarray:
        return np.stack([self.sample(rng, n) for _ in range(n_chains)])


def gauss_hermite_normal(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for ``int f(g) phi(g) dg`` with phi the standard normal density.

    `numpy.polynomial.hermite.hermgauss` integrates against ``exp(-x^2)``; substituting
    ``g = sqrt(2) x`` and dividing by ``sqrt(pi)`` converts it. The weights sum to 1, which
    is the check worth making rather than trusting the substitution.
    """
    x, w = np.polynomial.hermite.hermgauss(n_nodes)
    return np.sqrt(2.0) * x, w / np.sqrt(np.pi)


def global_latent_posterior_mean(
    prior: ChainPlusGlobal,
    grid: np.ndarray,
    weights: np.ndarray,
    X: np.ndarray,
    alpha: float,
    delta: float,
    n_nodes: int = 41,
    log_mu: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact ``E[a | x]`` for a chain-plus-global prior, and the per-chain ``log p(x)``.

    Exact up to two refinable approximations and no others: the grid, shared with every
    reference in this project, and the `n_nodes`-point Gauss-Hermite rule in ``g``.

    Returns ``(means, log_evidence)`` with shapes ``(B, n)`` and ``(B,)``.

    The log-sum-exp over nodes is not decoration. The conditional evidence is a product over
    ``n`` sites, so its log runs to tens of nats and the raw likelihoods underflow long
    before the posterior over ``g`` becomes concentrated enough for it to matter.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    c = prior.scale
    log_k = prior.base.log_transition_matrix(grid)
    nodes, node_w = gauss_hermite_normal(n_nodes)

    # Conditional on g the chain is Markov in y, observed through an OU channel with the
    # same alpha and an inflated noise variance -- see the module docstring.
    delta_y = (c**2) * delta

    log_terms = np.empty((n_nodes, X.shape[0]))
    means_y = np.empty((n_nodes,) + X.shape)
    for k, g in enumerate(nodes):
        x_prime = c * X - alpha * prior.beta * g
        m_y, _, log_ev = grid_bp_batch(
            grid, weights, log_k, x_prime, alpha, delta_y, log_mu, return_evidence=True
        )
        means_y[k] = m_y
        log_terms[k] = log_ev + np.log(node_w[k])

    shift = log_terms.max(axis=0)                       # (B,)
    post = np.exp(log_terms - shift[None, :])           # (n_nodes, B)
    norm = post.sum(axis=0)                             # (B,)
    post = post / norm[None, :]

    # E[a | x, g] = (E[y | x', g] + beta g) / c, averaged under p(g | x).
    means_a = (means_y + prior.beta * nodes[:, None, None]) / c
    means = np.einsum("kb,kbn->bn", post, means_a)

    # p(x) = c^n int p(x' | g) phi(g) dg. The Jacobian is independent of g, which is what
    # let it cancel out of the posterior above, but it does belong in the evidence itself.
    log_evidence = shift + np.log(norm) + X.shape[1] * np.log(c)
    return means, log_evidence


def laplace_plus_global(rho: float, beta: float) -> ChainPlusGlobal:
    """The non-Gaussian, non-Markov prior: Laplace innovations plus a shared latent."""
    return ChainPlusGlobal(base=LaplaceAR1(rho), beta=beta)


def gaussian_plus_global(rho: float, beta: float) -> ChainPlusGlobal:
    """The Gaussian counterpart, used only to check the reference against linear algebra.

    `priors.GaussianAR1PlusGlobal` is the same law and is what the exact-score machinery
    consumes; this exists so both routes can be driven from the identical construction.
    """
    return ChainPlusGlobal(base=GaussianAR1(rho), beta=beta)
