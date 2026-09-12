"""Rectangular transition matrices: a parent grid and a child grid that differ.

This is what per-depth grids need. A subband whose likelihood is narrow (a coarse
one, with a large scale) needs a fine mesh; its child may not; the edge between
them is then an (M_out x M_in) matrix rather than a square one.

The checks are chosen so that a wrong index order cannot pass. Population Xi is
built analytically from a *known* kernel on mismatched grids, and the M-step has
to return the generating parameters -- transposing the two axes, or reading the
parent value off the child grid, changes the answer and fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bp_grid import make_grid
from src.em import ExpectedStatistics
from src.kernels import GaussianAR1Kernel, LaplaceAR1Kernel, MixtureInnovationKernel
from src.scale_kernel import ScaleMixtureKernel
from src.utils import rng_for

_LOG_2PI = float(np.log(2.0 * np.pi))


def _population_xi(kernel, grid_in, w_in, grid_out, w_out, n_edges=1.0):
    """Xi[k, j] = n_edges * P(parent = u_j) * K(u_k | u_j), on the two meshes.

    The exact expectation a perfect E-step would return for a standard-normal
    parent, so the M-step's job is pure parameter recovery with no sampling
    noise in the way.
    """
    prior = np.exp(-0.5 * grid_in**2) * w_in
    prior = prior / prior.sum()
    k = np.exp(kernel.log_transition_matrix(grid_in, grid_out)) * w_out[:, None]
    k = k / np.maximum(k.sum(axis=0, keepdims=True), 1e-300)
    return n_edges * k * prior[None, :]


def _stats(xi):
    return ExpectedStatistics(
        xi=xi, site1=np.zeros(xi.shape[1]), log_evidence=0.0,
        n_edges=int(max(1, round(float(xi.sum())))), n_chains=1,
    )


# ----------------------------------------------------------------------------
# Shape and value
# ----------------------------------------------------------------------------

def test_rectangular_matrix_has_the_right_shape_and_entries():
    gi, _ = make_grid(8.0, 121)
    go, _ = make_grid(8.0, 401)
    k = GaussianAR1Kernel(rho=0.6, q=0.5)
    got = k.log_transition_matrix(gi, go)
    assert got.shape == (401, 121)
    want = (-0.5 * (go[:, None] - 0.6 * gi[None, :]) ** 2 / 0.5
            - 0.5 * (_LOG_2PI + np.log(0.5)))
    assert np.max(np.abs(got - want)) < 1e-13


@pytest.mark.parametrize("kernel", [
    GaussianAR1Kernel(rho=0.6, q=0.5),
    LaplaceAR1Kernel(rho=0.5, b=0.4),
])
def test_square_call_is_unchanged(kernel):
    """Backward compatibility: omitting the child grid must be exactly the old
    behaviour, since every existing caller relies on it."""
    g, _ = make_grid(8.0, 201)
    assert np.array_equal(
        kernel.log_transition_matrix(g), kernel.log_transition_matrix(g, g)
    )


def test_rectangular_columns_integrate_to_one_on_the_child_mesh():
    """Up to truncation, which is a property of the child domain and not of the
    rectangular indexing: at parent u = 8 and rho = 0.6 the child sits at 4.8
    with sigma 0.71, so a +-8 child grid loses ~1e-6 off its tail. Measured in
    the interior, where no such loss occurs, and separately confirmed to shrink
    when the child domain grows."""
    gi, _ = make_grid(8.0, 121)
    k = GaussianAR1Kernel(rho=0.6, q=0.5)

    def deficit(half_width, size, interior):
        go, wo = make_grid(half_width, size)
        mass = (np.exp(k.log_transition_matrix(gi, go)) * wo[:, None]).sum(axis=0)
        sel = np.abs(gi) <= interior
        return float(np.max(np.abs(mass[sel] - 1.0)))

    assert deficit(8.0, 601, 4.0) < 1e-10
    assert deficit(8.0, 601, 8.0) < 1e-5
    assert deficit(12.0, 901, 8.0) < 1e-10


# ----------------------------------------------------------------------------
# The M-step, which is where an index error would actually bite
# ----------------------------------------------------------------------------

def test_gaussian_m_step_recovers_parameters_on_mismatched_grids():
    gi, wi = make_grid(8.0, 241)
    go, wo = make_grid(8.0, 601)
    truth = GaussianAR1Kernel(rho=0.6, q=0.5)
    xi = _population_xi(truth, gi, wi, go, wo, n_edges=1000.0)

    got = GaussianAR1Kernel(rho=0.1, q=1.0).m_step(_stats(xi), gi, go)
    assert abs(got.rho - 0.6) < 1e-6
    assert abs(got.q - 0.5) < 1e-6


def test_transposing_the_grids_changes_the_answer():
    """Guards the check above against being vacuous.

    If the M-step ignored which grid was which, feeding the parent grid as the
    child grid would recover the same parameters. It must not.
    """
    gi, wi = make_grid(8.0, 241)
    go, wo = make_grid(4.0, 241)          # different half-width, same size
    truth = GaussianAR1Kernel(rho=0.6, q=0.5)
    xi = _population_xi(truth, gi, wi, go, wo, n_edges=1000.0)

    right = GaussianAR1Kernel(rho=0.1, q=1.0).m_step(_stats(xi), gi, go)
    wrong = GaussianAR1Kernel(rho=0.1, q=1.0).m_step(_stats(xi), go, gi)
    # The narrow child domain truncates a little, hence 1e-3 rather than 1e-6.
    assert abs(right.rho - 0.6) < 1e-3
    # Swapping the roles must be wrong by orders of magnitude more than that.
    assert abs(wrong.rho - 0.6) > 100 * abs(right.rho - 0.6)


def test_laplace_m_step_recovers_parameters_on_mismatched_grids():
    gi, wi = make_grid(9.0, 401)
    go, wo = make_grid(9.0, 801)
    truth = LaplaceAR1Kernel(rho=0.5, b=0.4)
    xi = _population_xi(truth, gi, wi, go, wo, n_edges=2000.0)

    got = LaplaceAR1Kernel(rho=0.1, b=1.0).m_step(_stats(xi), gi, go)
    assert abs(got.rho - 0.5) < 0.02
    assert abs(got.b - 0.4) < 0.02


def test_mixture_m_step_agrees_across_grid_pairings():
    """A finer child mesh may not change the fitted parameters materially."""
    gi, wi = make_grid(8.0, 241)
    truth = MixtureInnovationKernel(
        rho=0.55, pi=np.array([0.7, 0.3]), mu=np.array([0.0, 0.0]),
        s2=np.array([0.15, 0.9]),
    )
    rng = rng_for("two-grid-mixture")
    fits = []
    for size in (241, 601):
        go, wo = make_grid(8.0, size)
        xi = _population_xi(truth, gi, wi, go, wo, n_edges=5000.0)
        k = MixtureInnovationKernel.init(2, rho=0.2, var=0.6, rng=rng)
        for _ in range(30):
            k = k.m_step(_stats(xi), gi, go)
        fits.append(k)
    assert abs(fits[0].rho - fits[1].rho) < 0.01
    assert abs(fits[0].rho - 0.55) < 0.05
    a = fits[0].innovation_moments["innovation_var"]
    b = fits[1].innovation_moments["innovation_var"]
    assert abs(a - b) < 0.02


def test_scale_mixture_m_step_runs_rectangular_and_keeps_the_gate_on_the_parent():
    """The gate is a function of the parent alone, so its length must follow the
    *parent* grid however fine the child grid is."""
    gi, wi = make_grid(8.0, 201)
    rng = rng_for("two-grid-scale")
    truth = ScaleMixtureKernel.init(3, rho=0.4, var=0.7, rng=rng)

    for size in (201, 501):
        go, wo = make_grid(8.0, size)
        xi = _population_xi(truth, gi, wi, go, wo, n_edges=4000.0)
        k = ScaleMixtureKernel.init(3, rho=0.2, var=1.0, rng=rng)
        before = float((xi * k.log_transition_matrix(gi, go)).sum())
        for _ in range(6):
            k = k.m_step(_stats(xi), gi, go)
        after = float((xi * k.log_transition_matrix(gi, go)).sum())
        assert after >= before - 1e-6 * abs(before)
        assert k.gate(gi).shape == (201, 3)
