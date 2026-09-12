"""Speciation and collapse time scales for the OU forward process.

Biroli, Bonnaire, de Bortoli and Mezard (Nat. Commun. 15, 9957, 2024) show that
the backward generative dynamics of a diffusion model passes through two
cross-overs: a *speciation* time, fixed by the spectrum of the data covariance,
at which the coarse structure of the sample is decided; and a *collapse* time,
fixed by an excess entropy of the data, past which trajectories driven by the
empirical score are captured by a single training point. This module supplies
both time scales in the conventions of `src/noising.py`, so they can be compared
against measured dynamics rather than quoted.

Speciation
----------
Along an eigendirection of the clean covariance with eigenvalue `Lambda`, the
noisy marginal splits into signal and noise variance

    Var(<v, x_t>) = alpha_t^2 Lambda + Delta_t,      alpha_t^2 = e^{-2t},
                                                     Delta_t = 1 - e^{-2t}.

The mode is "already decided" while signal dominates and undecided once noise
does, so the cross-over sits at alpha_t^2 Lambda = Delta_t, i.e.

    t_S(Lambda) = 1/2 log(1 + Lambda).                              (*)

Equivalently, the correlation between the projection at time t and at time 0 is
`sqrt(Lambda) alpha_t / sqrt(alpha_t^2 Lambda + Delta_t)`, which passes through
1/sqrt(2) exactly at t_S. For Lambda >> 1 this is the familiar t_S ~ 1/2 log
Lambda; (*) is its finite-Lambda form under the variance-preserving convention
used throughout this package, and `commitment` below is the statement that can
actually be measured on trajectories.

The point of carrying the whole spectrum rather than only its top eigenvalue is
that a hierarchical prior has a *ladder* of well-separated eigenvalues, one per
level (see `src/hierarchy.py`), and therefore a ladder of speciation times: the
reverse dynamics resolves the hierarchy coarse-to-fine, one transition per
level. A chain, by contrast, has a continuum of eigenvalues bounded by
`(1 + rho) / (1 - rho)` no matter how long it is -- so a stationary Markov chain
has no diverging speciation time, and no coarse-to-fine cascade to resolve.

Collapse
--------
The collapse time is where the empirical score, built from `N` training points,
stops describing the population and starts describing the sample. Following the
entropic criterion, memorization is avoided only while

    N  >~  exp(n * s),      s = per-site excess entropy of the clean data,

with `s` measured against the terminal noise measure. For a stationary Gaussian
AR(1) chain the differential entropy is exactly extensive,

    H(a_1..a_n) = 1/2 log(2 pi e) + (n - 1)/2 log(2 pi e q),   q = 1 - rho^2,

so relative to `N(0, I_n)` the per-site excess entropy tends to

    s = -1/2 log q = 1/2 log(1 / (1 - rho^2)),

exactly and with no fitting. That closed form is what makes the curse of
dimensionality checkable here rather than merely invoked: it predicts the
dataset size at which a memorizing score must fail, as a function of chain
length, before any experiment is run.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Speciation
# ----------------------------------------------------------------------------

def speciation_time(eigenvalue: float | np.ndarray) -> float | np.ndarray:
    """t_S = 1/2 log(1 + Lambda): where noise overtakes signal in that mode."""
    return 0.5 * np.log1p(np.asarray(eigenvalue, dtype=float))


def commitment(t: float | np.ndarray, eigenvalue: float) -> float | np.ndarray:
    """Corr(<v, x_t>, <v, x_0>) for a mode of variance `eigenvalue`.

    Under the variance-preserving OU process the joint law of (x_0, x_t) is
    Gaussian in each eigendirection with covariance alpha_t * Lambda, so this is
    `alpha sqrt(Lambda) / sqrt(alpha^2 Lambda + Delta)`. It equals 1/sqrt(2) at
    `speciation_time(eigenvalue)`, which is how the cross-over is located in the
    measured trajectories.
    """
    t = np.asarray(t, dtype=float)
    alpha2 = np.exp(-2.0 * t)
    delta = 1.0 - alpha2
    lam = float(eigenvalue)
    return alpha2**0.5 * np.sqrt(lam) / np.sqrt(alpha2 * lam + delta)


def chain_covariance(n: int, rho: float) -> np.ndarray:
    """Stationary AR(1) covariance `rho^{|i-j|}` (unit marginal variance)."""
    idx = np.arange(n)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def chain_spectrum(n: int, rho: float) -> np.ndarray:
    """Eigenvalues of `chain_covariance`, ascending."""
    return np.linalg.eigvalsh(chain_covariance(n, rho))


def chain_top_eigenvalue_limit(rho: float) -> float:
    """`(1 + rho) / (1 - rho)`: the n -> infinity bound on the AR(1) spectrum.

    It is the spectral density of the AR(1) process at zero frequency. Because
    it is *finite*, the top eigenvalue of a chain does not grow with its length,
    unlike the `Lambda ~ d` behaviour that makes the speciation time of natural
    image data diverge logarithmically with dimension.
    """
    return (1.0 + rho) / (1.0 - rho)


# ----------------------------------------------------------------------------
# Collapse
# ----------------------------------------------------------------------------

def gaussian_chain_excess_entropy_rate(rho: float) -> float:
    """Per-site excess entropy `-1/2 log(1 - rho^2)` of a Gaussian AR(1) chain.

    Measured against the terminal measure `N(0, 1)` per site, which is the
    reference the forward process relaxes to. Exact, not asymptotic in `rho`;
    the only approximation is dropping the O(1/n) boundary term.
    """
    return -0.5 * np.log(1.0 - rho**2)


def gaussian_chain_excess_entropy(n: int, rho: float) -> float:
    """Total excess entropy of `n` sites, boundary term included."""
    q = 1.0 - rho**2
    h_chain = 0.5 * np.log(2 * np.pi * np.e) + 0.5 * (n - 1) * np.log(
        2 * np.pi * np.e * q
    )
    h_noise = 0.5 * n * np.log(2 * np.pi * np.e)
    return float(h_noise - h_chain)


def collapse_dataset_size(n: int, rho: float) -> float:
    """`exp(excess entropy)`: dataset size below which memorization is forced.

    A score built from `N` samples can only avoid collapsing onto them if `N`
    exceeds this. It is exponential in the chain length, which is the curse of
    dimensionality in its sharpest form: for rho = 0.85 the per-site cost is
    0.64 nats, so 33 sites already demand ~1e9 chains.
    """
    return float(np.exp(gaussian_chain_excess_entropy(n, rho)))
