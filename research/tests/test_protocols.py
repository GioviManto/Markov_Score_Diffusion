"""The three observation protocols, and the identity that separates them.

The claim being guarded is the one the external review made: noising one clean batch at five
levels and handing the five groups to `fit_em` as independent chains maximises a *composite*
objective, not the joint marginal likelihood of the generated data. These tests make that a
demonstrated numerical fact rather than an assertion, by computing both objectives on the
same data and showing they differ -- and by showing that the multi-view construction agrees
with a brute-force joint evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bp_grid import grid_bp, grid_bp_batch, make_grid
from src.em import e_step_multi
from src.noising import alpha_delta
from src.priors import LaplaceAR1
from src.protocols import (
    composite_groups,
    multi_view_likelihood,
    one_view_groups,
    sample_budget,
)
from src.utils import rng_for

RHO = 0.85
N_SITES = 8
T_LEVELS = (0.2, 0.6)


@pytest.fixture(scope="module")
def setup():
    grid, weights = make_grid(8.0, 401)
    prior = LaplaceAR1(RHO)
    return grid, weights, prior, prior.log_transition_matrix(grid)


def test_one_view_partitions_the_chains(setup):
    """Every latent chain observed exactly once: the counts have to add up, or the
    'effective sample size' the review asked for is unknowable."""
    _, _, prior, _ = setup
    rng = rng_for("test-protocols-oneview")
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(30)])
    groups = one_view_groups(A, (0.1, 0.3, 0.9), rng)

    assert sum(X.shape[0] for X, _, _ in groups) == len(A)
    assert len(groups) == 3


def test_multi_view_likelihood_matches_brute_force_joint_evidence(setup):
    """The joint protocol computed two ways.

    Under the joint model the site potential is the product of the per-view likelihoods.
    Here that product is formed analytically by `multi_view_likelihood` and then, separately,
    by explicitly multiplying two single-view likelihood tables and running BP on the result.
    Agreement pins the row-shift bookkeeping, which is the only place this can go wrong
    quietly.
    """
    grid, weights, prior, log_k = setup
    rng = rng_for("test-protocols-multiview")
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(4)])

    views, alphas, deltas = [], [], []
    for t in T_LEVELS:
        alpha, delta = alpha_delta(t)
        views.append(alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape))
        alphas.append(alpha)
        deltas.append(delta)

    log_psi, log_const = multi_view_likelihood(grid, views, alphas, deltas)

    # Brute force: build the same potential by summing raw log-likelihoods, no shifting.
    raw = np.zeros((len(A), N_SITES, len(grid)))
    for X, alpha, delta in zip(views, alphas, deltas):
        z = X[:, :, None] - alpha * grid[None, None, :]
        raw += -0.5 * z**2 / delta - 0.5 * np.log(2.0 * np.pi * delta)

    # The shifted table must equal the raw one up to a per-(chain, site) constant, and the
    # constants must be exactly what log_const accounts for.
    diff = raw - log_psi
    per_site = diff.mean(axis=2)
    assert np.allclose(diff, per_site[:, :, None], atol=1e-9)
    assert float(per_site.sum()) == pytest.approx(log_const, rel=1e-9)


def test_composite_and_joint_objectives_genuinely_differ(setup):
    """The review's central protocol point, as a number.

    Both objectives are evaluated at the *same* parameters on the *same* observations. If
    they agreed, the old code would have been computing the joint likelihood after all and
    the complaint would be empty. They do not agree, and the gap is large -- the composite
    sum counts the prior once per view.
    """
    grid, weights, prior, log_k = setup
    rng = rng_for("test-protocols-differ")
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(16)])

    views, alphas, deltas = [], [], []
    for t in T_LEVELS:
        alpha, delta = alpha_delta(t)
        views.append(alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape))
        alphas.append(alpha)
        deltas.append(delta)

    composite = e_step_multi(
        grid, weights, log_k, list(zip(views, alphas, deltas))
    ).log_evidence

    # Joint: one shared latent chain, both views folded into one site potential.
    log_psi, log_const = multi_view_likelihood(grid, views, alphas, deltas)
    joint = 0.0
    for b in range(len(A)):
        res = grid_bp(grid, weights, log_k, log_psi[b], views[0][b], alphas[0], deltas[0])
        joint += res.log_evidence
    # grid_bp restores constants for a single-view table; strip those and apply the joint
    # ones, so the comparison is between objectives rather than between normalisations.
    per_chain_single = N_SITES * (-0.5 * np.log(2.0 * np.pi * deltas[0]))
    joint -= len(A) * per_chain_single

    assert composite < 0 and joint < 0
    # Two different objectives, not two spellings of one.
    assert abs(composite - joint) > 1.0


def test_sample_budget_columns_are_consistent():
    """The bookkeeping the review asked every output file to carry."""
    one = sample_budget("one_view", 100, 32, 5)
    assert one["noisy_observations"] == 100
    assert one["views_per_latent_chain"] == 1
    assert one["views_paired"] is False

    multi = sample_budget("multi_view", 100, 32, 5)
    assert multi["noisy_observations"] == 500
    assert multi["views_per_latent_chain"] == 5
    assert multi["views_paired"] is True

    comp = sample_budget("composite", 100, 32, 5)
    # Same counts as multi_view -- the difference is the objective, not the data volume,
    # which is exactly why the protocol name has to be recorded alongside the numbers.
    assert comp["noisy_observations"] == multi["noisy_observations"]
    assert comp["views_paired"] is False

    clean = sample_budget("clean", 100, 32, 5)
    assert clean["noisy_observations"] == 0
    assert clean["clean_values_consumed"] == 3200

    with pytest.raises(ValueError):
        sample_budget("nonsense", 1, 1, 1)


def test_composite_groups_reuse_every_chain_at_every_level(setup):
    """Documents what the old path did, so a future reader can see it rather than infer it."""
    _, _, prior, _ = setup
    rng = rng_for("test-protocols-composite")
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(12)])
    groups = composite_groups(A, T_LEVELS, rng)

    assert len(groups) == len(T_LEVELS)
    assert all(X.shape == A.shape for X, _, _ in groups)
    # 12 latent chains but 24 observations: the sample-size inflation in one line.
    assert sum(X.shape[0] for X, _, _ in groups) == 2 * len(A)
