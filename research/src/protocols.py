"""How a clean chain becomes observed data, made explicit and named.

Three different questions have been run through the same code path in this project, and the
write-up then described whichever one the reader assumed. An external review caught it. The
distinction is not bookkeeping: it changes what likelihood is being maximised and what the
effective sample size is.

The three protocols
-------------------
``one_view``
    Each latent chain is observed **once**, at one noise level. Chains are split disjointly
    across the levels. The objective is then the exact marginal log-likelihood of the data
    that was actually generated,

        L(theta) = sum_mu log int p_theta(a) p(x^mu | a, t_mu) da,

    and the sample budget is unambiguous: one clean chain, one noisy observation. This is
    what `exp_07` and `exp_12` already did; `exp_16` did not.

``multi_view``
    Each latent chain has **several paired observations** at known, different noise levels.
    The joint likelihood is

        L(theta) = sum_mu log int p_theta(a) prod_k p(x^{mu k} | a, t_k) da,

    which is *not* what you get by treating the views as separate chains. Several
    observations of the same site contribute only unary factors, so the correct handling is
    to add their log-likelihoods into a single site potential

        psi_i(a_i) = prod_k ell_{t_k}(x^{mu k}_i | a_i)

    before one BP pass. The posterior is still a chain, so nothing in the E-step machinery
    changes -- only how the likelihood tables are built.

``clean``
    No noising at all: the transition kernel is fitted directly from clean chains via
    `em.fit_clean`. The structured baseline, and the only one of the three that is not a
    latent-variable problem.

What the old path actually computed
-----------------------------------
`exp_16` drew one batch ``A`` and noised **all of it** at each of five levels, then passed the
five groups to `fit_em` as if they were independent latent sequences. That maximises

        Ltilde(theta) = sum_{mu,k} log int p_theta(a) p(x^{mu k} | a, t_k) da,

a *composite* likelihood: each term is correctly specified, so the estimator is not wrong,
but the sum is not the log-likelihood of the generated dataset. The effective sample size is
the number of clean chains, not five times it, and any per-observation likelihood computed
from it must be labelled composite. `composite` is retained below so that objective can still
be produced deliberately -- it is a legitimate estimator, and reproducing the committed
numbers requires it.
"""

from __future__ import annotations

import numpy as np

PROTOCOLS = ("one_view", "multi_view", "clean", "composite")


def _noise(A: np.ndarray, t: float, rng: np.random.Generator):
    alpha = float(np.exp(-t))
    delta = float(1.0 - np.exp(-2.0 * t))
    return alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape), alpha, delta


def one_view_groups(A: np.ndarray, t_values, rng: np.random.Generator):
    """Disjoint chains across noise levels: one observation per latent chain.

    Returns the ``[(X, alpha, delta)]`` list `em.e_step_multi` consumes. Every chain appears
    in exactly one group, so summing the groups' evidences gives the exact marginal
    log-likelihood of the dataset.
    """
    parts = np.array_split(rng.permutation(len(A)), len(t_values))
    groups = []
    for t, idx in zip(t_values, parts):
        X, alpha, delta = _noise(A[idx], t, rng)
        groups.append((X, alpha, delta))
    return groups


def composite_groups(A: np.ndarray, t_values, rng: np.random.Generator):
    """Every chain noised at every level, treated as independent. **Composite objective.**

    Kept because it is what the committed `exp_16` outputs were produced under, and a
    protocol comparison needs the old arm. Anything reported from it must say composite.
    """
    return [_noise(A, t, rng) for t in t_values]


def multi_view_likelihood(
    grid: np.ndarray, X_views, alphas, deltas
) -> tuple[np.ndarray, float]:
    """Combined log-likelihood table for several paired views of ONE latent chain set.

    ``X_views`` is a sequence of ``(B, n)`` arrays, all rows referring to the same latent
    chains, observed at ``alphas[k]``/``deltas[k]``. Returns ``(log_psi, log_const)`` where
    ``log_psi`` has shape ``(B, n, M)`` and is row-shifted per (chain, site), and
    ``log_const`` collects the removed shifts and Gaussian normalisers so the evidence can
    be restored exactly.

    The whole content of the joint protocol is here: repeated observations of ``a_i`` are
    conditionally independent given ``a_i``, so their log-likelihoods **add** into one site
    potential. Handing BP five separate chains instead multiplies the prior in five times,
    which is what makes the composite objective a different estimator rather than a
    rearrangement of the same one.
    """
    if not len(X_views):
        raise ValueError("multi_view needs at least one view.")
    shapes = {x.shape for x in X_views}
    if len(shapes) != 1:
        raise ValueError(f"All views must share a shape; got {shapes}.")

    total = None
    log_const = 0.0
    for X, alpha, delta in zip(X_views, alphas, deltas):
        z = X[:, :, None] - alpha * grid[None, None, :]
        log_ell = -0.5 * z**2 / delta
        # Every (chain, site) pair carries one Gaussian normaliser per view, so the batch
        # dimension belongs here too. Omitting it understates the constant by a factor of
        # the batch size -- invisible in any single-chain check, which is why the test
        # compares a whole batch against a brute-force sum.
        log_const += X.shape[0] * X.shape[1] * (-0.5 * np.log(2.0 * np.pi * delta))
        total = log_ell if total is None else total + log_ell

    shift = total.max(axis=2)
    return total - shift[:, :, None], float(log_const) + float(shift.sum())


def sample_budget(protocol: str, n_chains: int, n_sites: int, n_levels: int) -> dict:
    """The columns every output file should carry, so the budget is legible.

    The review's complaint was not that any one number was wrong but that a reader could not
    tell how many latent chains and how many observations produced a curve. These are those
    numbers.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}; choose from {PROTOCOLS}.")
    if protocol == "one_view":
        views, observed = 1, n_chains
    elif protocol in ("multi_view", "composite"):
        views, observed = n_levels, n_chains * n_levels
    else:  # clean
        views, observed = 0, 0
    return {
        "protocol": protocol,
        "unique_latent_chains": n_chains,
        "noisy_observations": observed,
        "views_per_latent_chain": views,
        "views_paired": protocol == "multi_view",
        "clean_values_consumed": n_chains * n_sites if protocol == "clean" else 0,
        "noisy_values_consumed": observed * n_sites,
    }
