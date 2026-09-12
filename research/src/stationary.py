"""Strictly stationary initialisation of the chain priors.

Every prior in `src/priors.py` starts its recursion at ``a[0] ~ N(0, 1)`` and then applies
non-Gaussian innovations. That makes the chain **covariance**-stationary -- ``Var(a_i) = 1``
and ``Cov(a_i, a_j) = rho^|i-j|`` exactly, for every variance-matched innovation law -- but
not strictly stationary. The invariant law of

    a_i = rho a_{i-1} + eps_i

is the law of ``sum_{k>=0} rho^k eps_{-k}``, which is Gaussian only when ``eps`` is. So the
marginal of ``a_i`` drifts away from ``N(0, 1)`` towards that law over a burn-in of order
``1 / log(1/rho)``, about six sites at ``rho = 0.85``.

Two places this matters, and the second is not a wording question.

1. The note described the construction as stationary. It is not, and that correction has
   already been made in the text.

2. `exp_11` measures the receptive field by running grid BP on a window ``[C-r, C+r]``.
   That window estimator equals the exact conditional expectation ``E[a_C | x_window]`` only
   if BP is given the true marginal law at the window's **left endpoint** -- the marginal of
   a contiguous window of a Markov chain is a chain with the same kernel and that endpoint
   law. `grid_bp_batch` defaults it to ``N(0, 1)``, which is correct for the Gaussian chain
   and wrong for every other family, by an amount that grows as the window's left edge moves
   further from site 1. So the measurement is exact for precisely the family the others are
   compared *against*, and biased for the families whose less-local behaviour is the result.

   Under strict stationarity the marginal at every site is the invariant law, one ``log_mu``
   serves every window regardless of where it sits, and the window estimator becomes exact
   again -- a T3 quantity rather than a proxy.

This module supplies the invariant law on both sides: as a density on the BP grid, for
``log_mu``, and as a sampler, for the data. The two are computed by independent routes --
power iteration of the transition operator, and burn-in of the recursion -- so their
agreement is a real check rather than a restatement of one computation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InvariantDensity:
    """The invariant law of a chain prior's kernel, on the BP grid.

    log_density : (M,) log pi(u_j), in exactly the form `grid_bp`, `grid_bp_batch` and
                  `e_step` accept as `log_mu`. Entries may be -inf where the invariant law
                  has no support (the uniform family has compact support), which those
                  callers handle: they exponentiate it.
    n_iter      : power iterations actually taken.
    residual    : final quadrature-L1 change between successive iterates.
    """

    log_density: np.ndarray
    n_iter: int
    residual: float


def invariant_log_density(
    prior,
    grid: np.ndarray,
    weights: np.ndarray,
    tol: float = 1e-14,
    max_iter: int = 4000,
) -> InvariantDensity:
    """Invariant density of `prior`'s transition kernel, by power iteration.

    Iterates ``pi <- K pi`` under the *same* trapezoidal quadrature the BP recursion uses,
    so the fixed point is the invariant law of the discretised operator. That is deliberate
    and is the right object: it is what the grid recursion would actually propagate, so
    seeding BP with it makes the forward message stationary to machine precision instead of
    to quadrature precision. The continuous invariant law is what this approximates, at the
    same O(h^2) the rest of the grid machinery carries.

    Convergence is geometric at rate ``rho`` -- the Markov operator contracts by exactly the
    autoregressive coefficient -- so at ``rho = 0.85`` a few hundred iterations reach the
    floor. Not converging is raised, never silently returned: an unconverged density fed to
    BP as `log_mu` would look like a plausible prior rather than like a failure.

    Starting from ``N(0, 1)`` is not a special choice. Power iteration on a contraction
    forgets its seed at the same rate it converges; the standard normal is used only because
    it is the law the priors currently start from, which makes ``n_iter`` directly readable
    as "how long the existing construction takes to forget its initialisation".
    """
    K = np.exp(prior.log_transition_matrix(grid))

    pi = np.exp(-0.5 * grid**2 - 0.5 * np.log(2.0 * np.pi))
    pi = pi / float(pi @ weights)

    residual = float("inf")
    for n_iter in range(1, max_iter + 1):
        nxt = K @ (pi * weights)
        mass = float(nxt @ weights)
        if not np.isfinite(mass) or mass <= 0.0:
            raise FloatingPointError(
                f"Invariant-density iteration lost all mass at step {n_iter} "
                f"(mass={mass!r}). The grid is probably too narrow for this kernel."
            )
        nxt = nxt / mass
        residual = float(np.abs(nxt - pi) @ weights)
        pi = nxt
        if residual <= tol:
            break
    else:
        raise RuntimeError(
            f"Invariant density did not converge in {max_iter} iterations "
            f"(residual {residual:.3e} > tol {tol:.3e}) for {prior.name!r}."
        )

    # -inf where the invariant law has no support is the honest value, not a defect: the
    # uniform family's invariant law lives on [-h/(1-rho), h/(1-rho)]. Callers exponentiate.
    with np.errstate(divide="ignore"):
        log_density = np.log(pi)

    return InvariantDensity(log_density=log_density, n_iter=n_iter, residual=residual)


def drifted_log_density(
    prior, grid: np.ndarray, weights: np.ndarray, n_steps: int
) -> np.ndarray:
    """log of the marginal law at site ``n_steps`` of the *existing* construction.

    The priors start at ``a_1 ~ N(0, 1)``, so the marginal at site ``1 + k`` is the standard
    normal pushed ``k`` times through the kernel. That law is neither ``N(0, 1)`` (except at
    ``k = 0``, or for the Gaussian chain at any ``k``) nor the invariant law (except in the
    limit): it is partway between, and it is what a window whose left edge sits at that site
    actually needs as its initial law.

    This is what makes the bias in the committed locality numbers *measurable* rather than
    merely arguable. Window BP given this density is exact on the existing data; window BP
    given ``N(0, 1)`` is what `exp_11` ran. The difference between the two is the error, and
    it needs no new data and no new assumption to compute.

    Applies the operator directly rather than reusing `invariant_log_density`, because the
    point here is the transient, not the fixed point.
    """
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps!r}.")
    K = np.exp(prior.log_transition_matrix(grid))
    pi = np.exp(-0.5 * grid**2 - 0.5 * np.log(2.0 * np.pi))
    pi = pi / float(pi @ weights)
    for _ in range(n_steps):
        pi = K @ (pi * weights)
        mass = float(pi @ weights)
        if not np.isfinite(mass) or mass <= 0.0:
            raise FloatingPointError("Drifted marginal lost all mass; widen the grid.")
        pi = pi / mass
    with np.errstate(divide="ignore"):
        return np.log(pi)


def density_moments(log_density: np.ndarray, grid: np.ndarray, weights: np.ndarray) -> dict:
    """Mass and first four central moments of a grid density.

    The diagnostics that say whether a computed invariant law is the right object: mass must
    be 1, mean 0, and variance 1 for every variance-matched family in `priors`, since that
    normalisation is what makes the families comparable in the first place. Excess kurtosis
    is the one that should *differ* between families -- it is the non-Gaussianity knob.
    """
    pi = np.exp(log_density)
    mass = float(pi @ weights)
    if mass <= 0.0:
        raise FloatingPointError("Density has non-positive mass.")
    p = pi / mass
    m1 = float((grid * p) @ weights)
    cen = grid - m1
    m2 = float((cen**2 * p) @ weights)
    m4 = float((cen**4 * p) @ weights)
    return {
        "mass": mass,
        "mean": m1,
        "var": m2,
        "excess_kurtosis": m4 / m2**2 - 3.0 if m2 > 0 else float("nan"),
    }


def stationary_burn_in(rho: float, target: float = 1e-12) -> int:
    """Recursion steps after which the initial law is forgotten to `target`.

    After ``B`` steps the initial value enters ``a_B`` only through ``rho^B a_0``, so
    ``B = log(target) / log|rho|`` bounds its contribution. At ``rho = 0.85`` and the default
    target this is 170 steps, which costs nothing next to the grid recursion and removes the
    initialisation from the problem rather than making it small enough to argue about.
    """
    r = abs(float(rho))
    if r == 0.0:
        return 1
    if r >= 1.0:
        raise ValueError(f"A stationary law needs |rho| < 1, got {rho!r}.")
    return int(np.ceil(np.log(target) / np.log(r)))


def sample_stationary(
    prior, rng: np.random.Generator, n: int, burn_in: int | None = None
) -> np.ndarray:
    """One chain of length `n` whose every site is distributed as the invariant law.

    Implemented as burn-in of `prior.sample` rather than by inverting an invariant CDF, and
    that is the point: the recursion *is* the definition of the invariant law, so this needs
    no per-family code and stays correct for any innovation law a prior chooses to implement.
    Inverting a CDF would mean five separate derivations, five chances to be wrong, and no
    way to check the result except against this.
    """
    if burn_in is None:
        burn_in = stationary_burn_in(prior.rho)
    return prior.sample(rng, n + burn_in)[burn_in:]


def sample_stationary_batch(
    prior,
    rng: np.random.Generator,
    n_chains: int,
    n_sites: int,
    burn_in: int | None = None,
) -> np.ndarray:
    """(n_chains, n_sites) batch of strictly stationary chains."""
    return np.stack(
        [sample_stationary(prior, rng, n_sites, burn_in) for _ in range(n_chains)]
    )
