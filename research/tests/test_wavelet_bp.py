"""Exact BP on a fully observed quadtree, against two independent references.

The checks that matter here are the ones that do not reuse the implementation:

* a dense Gaussian solve of the same posterior, built from the linear map that
  defines the generative recursion rather than from a covariance formula, so a
  sign or index error in the tree bookkeeping cannot cancel;
* the closed-form Gaussian log-evidence;
* reduction to the already-validated `hierarchy.tree_bp_grid` when the internal
  observations are switched off, which is what ties this module to the code the
  package has been trusting all along.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bp_grid import make_grid
from src.hierarchy import TreeIndex, tree_bp_grid
from src.noising import alpha_delta
from src.utils import rng_for
from src.wavelet_bp import (
    as_grid_list, node_delta, stats_by_level, wavelet_tree_bp,
)

DEPTH = 3
BRANCHING = 4
RHOS = [0.8, 0.65, 0.5]          # one per edge level
_LOG_2PI = float(np.log(2.0 * np.pi))


def _tree_linear_map(ti: TreeIndex, rhos) -> np.ndarray:
    """`M` with `a = M eps`, eps independent standard normal, one per node.

    Built by forward substitution over the generative recursion itself:
    a_root = eps_root, a_child = rho_d a_parent + sqrt(1 - rho_d^2) eps_child.
    Every node then has unit marginal variance, matching `GaussianTree`.
    """
    n = ti.n_nodes
    mat = np.zeros((n, n))
    mat[0, 0] = 1.0
    for d in range(ti.depth):
        rho = rhos[d]
        for node in ti.nodes_at(d):
            for child in ti.children(int(node), d):
                mat[child] = rho * mat[node]
                mat[child, child] += np.sqrt(1.0 - rho**2)
    return mat


def _dense_posterior(sigma, x, alpha, deltas):
    """Exact Gaussian posterior mean and log evidence for x = alpha a + noise."""
    d_mat = np.diag(deltas)
    obs_cov = alpha**2 * sigma + d_mat
    mean = alpha * sigma @ np.linalg.solve(obs_cov, x.T)
    sign, logdet = np.linalg.slogdet(obs_cov)
    assert sign > 0
    quad = np.einsum("bi,ib->b", x, np.linalg.solve(obs_cov, x.T))
    log_ev = float(np.sum(-0.5 * (quad + logdet + len(deltas) * _LOG_2PI)))
    return mean.T, log_ev


def _gaussian_kernels(grid, rhos):
    out = []
    for rho in rhos:
        var = 1.0 - rho**2
        resid = grid[:, None] - rho * grid[None, :]
        out.append(-0.5 * (_LOG_2PI + np.log(var)) - 0.5 * resid**2 / var)
    return out


def _setup(deltas_by_depth, n_images=6, size=1201, half_width=9.0):
    ti = TreeIndex(DEPTH, BRANCHING)
    grid, weights = make_grid(half_width, size)
    log_k = _gaussian_kernels(grid, RHOS)
    log_root = -0.5 * grid**2 - 0.5 * _LOG_2PI

    mat = _tree_linear_map(ti, RHOS)
    sigma = mat @ mat.T
    rng = rng_for("wavelet-bp-test")
    a = rng.standard_normal((n_images, ti.n_nodes)) @ mat.T
    alpha, _ = alpha_delta(0.6)
    nd = node_delta(ti, deltas_by_depth)
    x = alpha * a + rng.standard_normal(a.shape) * np.sqrt(nd)
    return ti, grid, weights, log_k, log_root, sigma, x, alpha, nd


def test_posterior_mean_matches_dense_gaussian_solve():
    deltas = [0.35, 0.5, 0.8, 1.1]
    ti, grid, weights, log_k, log_root, sigma, x, alpha, nd = _setup(deltas)

    got = wavelet_tree_bp(
        grid, weights, log_k, log_root, x, alpha, deltas, BRANCHING, DEPTH
    )
    want, _ = _dense_posterior(sigma, x, alpha, nd)
    assert np.max(np.abs(got.posterior_mean - want)) < 1e-9


def test_log_evidence_matches_closed_form():
    deltas = [0.35, 0.5, 0.8, 1.1]
    ti, grid, weights, log_k, log_root, sigma, x, alpha, nd = _setup(deltas)

    got = wavelet_tree_bp(
        grid, weights, log_k, log_root, x, alpha, deltas, BRANCHING, DEPTH
    )
    _, want = _dense_posterior(sigma, x, alpha, nd)
    assert abs(got.log_evidence - want) / abs(want) < 1e-8


def test_per_scale_deltas_spanning_orders_of_magnitude():
    """The regime the image model actually runs in: subband scales differ a lot,
    so the per-depth Delta = Delta / s_d^2 spans decades."""
    deltas = [0.02, 0.2, 2.0, 20.0]
    ti, grid, weights, log_k, log_root, sigma, x, alpha, nd = _setup(deltas)

    got = wavelet_tree_bp(
        grid, weights, log_k, log_root, x, alpha, deltas, BRANCHING, DEPTH
    )
    want, want_ev = _dense_posterior(sigma, x, alpha, nd)
    assert np.max(np.abs(got.posterior_mean - want)) < 1e-7
    assert abs(got.log_evidence - want_ev) / abs(want_ev) < 1e-7


def test_chunking_does_not_change_the_answer():
    deltas = [0.35, 0.5, 0.8, 1.1]
    ti, grid, weights, log_k, log_root, sigma, x, alpha, nd = _setup(deltas, n_images=7)
    a = wavelet_tree_bp(grid, weights, log_k, log_root, x, alpha, deltas,
                        BRANCHING, DEPTH, chunk=2)
    b = wavelet_tree_bp(grid, weights, log_k, log_root, x, alpha, deltas,
                        BRANCHING, DEPTH, chunk=64)
    assert np.max(np.abs(a.posterior_mean - b.posterior_mean)) < 1e-12
    assert abs(a.log_evidence - b.log_evidence) < 1e-8 * abs(b.log_evidence)


def test_reduces_to_hierarchy_tree_bp_when_internal_evidence_is_removed():
    """With the internal nodes made uninformative, this must reproduce the
    already-validated leaves-only implementation.

    "Uninformative" needs both a large Delta *and* an observation at zero. A
    large Delta alone is not enough: the internal x drawn at Delta = 1e12 is of
    order 1e6, and the likelihood's linear term 2 alpha x u / Delta is then ~1e-6
    -- a weak exponential tilt across the grid, not a flat factor. That tilt
    moves the posterior mean by ~4e-7 and would look exactly like a bug in one
    of the two implementations. It is neither; it is a badly posed limit.
    """
    huge = 1e12
    deltas = [huge, huge, huge, 0.7]
    ti, grid, weights, log_k, log_root, _, x, alpha, _ = _setup(
        deltas, n_images=4, size=801, half_width=8.0
    )
    leaves = ti.leaf_nodes()
    internal = np.setdiff1d(np.arange(ti.n_nodes), leaves)
    x = x.copy()
    x[:, internal] = 0.0

    # hierarchy's implementation takes one shared kernel, so compare on a tree
    # whose levels share rho.
    shared = _gaussian_kernels(grid, [RHOS[0]] * DEPTH)
    got = wavelet_tree_bp(
        grid, weights, shared, log_root, x, alpha, deltas, BRANCHING, DEPTH
    )
    want = tree_bp_grid(
        shared[0], grid, log_root, x[:, leaves], alpha, 0.7, BRANCHING, DEPTH
    )
    assert np.max(np.abs(got.posterior_mean[:, leaves] - want)) < 1e-11
    assert got.posterior_mean.shape == x.shape


def test_xi_by_level_has_the_right_mass():
    """Each level's Xi must carry exactly one unit of mass per edge at that level."""
    deltas = [0.35, 0.5, 0.8, 1.1]
    ti, grid, weights, log_k, log_root, _, x, alpha, _ = _setup(deltas, n_images=5)
    res = wavelet_tree_bp(
        grid, weights, log_k, log_root, x, alpha, deltas,
        BRANCHING, DEPTH, want_stats=True,
    )
    assert res.xi_by_level is not None
    stats = stats_by_level(res.xi_by_level, res.log_evidence, 5, BRANCHING)
    for d, st in enumerate(stats):
        assert st.n_edges == 5 * BRANCHING ** (d + 1)
        assert abs(float(st.xi.sum()) - st.n_edges) < 1e-6 * st.n_edges


def _per_depth_setup(deltas, sizes, n_images=6, half_width=9.0):
    """Same model as `_setup`, but a different grid at every depth."""
    ti = TreeIndex(DEPTH, BRANCHING)
    grids, wts = [], []
    for size in sizes:
        g, w = make_grid(half_width, size)
        grids.append(g)
        wts.append(w)
    log_k = []
    for d, rho in enumerate(RHOS):
        var = 1.0 - rho**2
        resid = grids[d + 1][:, None] - rho * grids[d][None, :]
        log_k.append(-0.5 * (_LOG_2PI + np.log(var)) - 0.5 * resid**2 / var)
    log_root = -0.5 * grids[0] ** 2 - 0.5 * _LOG_2PI

    mat = _tree_linear_map(ti, RHOS)
    sigma = mat @ mat.T
    rng = rng_for("wavelet-bp-perdepth")
    a = rng.standard_normal((n_images, ti.n_nodes)) @ mat.T
    alpha, _ = alpha_delta(0.6)
    nd = node_delta(ti, deltas)
    x = alpha * a + rng.standard_normal(a.shape) * np.sqrt(nd)
    return ti, grids, wts, log_k, log_root, sigma, x, alpha, nd


def test_per_depth_grids_match_dense_gaussian_solve():
    """The point of the whole per-depth machinery: a different mesh at every
    level must not change the answer, only where the resolution is spent."""
    deltas = [0.35, 0.5, 0.8, 1.1]
    sizes = [1501, 701, 351, 181]
    ti, grids, wts, log_k, log_root, sigma, x, alpha, nd = _per_depth_setup(
        deltas, sizes
    )
    got = wavelet_tree_bp(
        grids, wts, log_k, log_root, x, alpha, deltas, BRANCHING, DEPTH
    )
    want, want_ev = _dense_posterior(sigma, x, alpha, nd)
    assert np.max(np.abs(got.posterior_mean - want)) < 1e-8
    assert abs(got.log_evidence - want_ev) / abs(want_ev) < 1e-7


def test_per_depth_grids_agree_with_a_uniform_fine_grid():
    """Cheap non-uniform mesh vs expensive uniform one, same model.

    This is the claim that makes per-depth grids worth having: the coarse levels
    carry almost no nodes, so refining them is nearly free, and the answer is the
    one a uniformly fine grid would have given.
    """
    deltas = [0.35, 0.5, 0.8, 1.1]
    ti, grids, wts, log_k, log_root, _, x, alpha, _ = _per_depth_setup(
        deltas, [1501, 701, 351, 181]
    )
    cheap = wavelet_tree_bp(
        grids, wts, log_k, log_root, x, alpha, deltas, BRANCHING, DEPTH
    ).posterior_mean

    _, g_u, w_u, lk_u, lr_u, _, x_u, a_u, _ = _setup(deltas, size=1501, half_width=9.0)
    dear = wavelet_tree_bp(
        g_u, w_u, lk_u, lr_u, x, alpha, deltas, BRANCHING, DEPTH
    ).posterior_mean
    assert np.max(np.abs(cheap - dear)) < 1e-8


def test_per_depth_statistics_are_rectangular_with_the_right_mass():
    deltas = [0.35, 0.5, 0.8, 1.1]
    sizes = [1501, 701, 351, 181]
    ti, grids, wts, log_k, log_root, _, x, alpha, _ = _per_depth_setup(
        deltas, sizes, n_images=5
    )
    res = wavelet_tree_bp(
        grids, wts, log_k, log_root, x, alpha, deltas,
        BRANCHING, DEPTH, want_stats=True,
    )
    for d, xi in enumerate(res.xi_by_level):
        assert xi.shape == (sizes[d + 1], sizes[d]), f"level {d}: {xi.shape}"
        want = 5 * BRANCHING ** (d + 1)
        assert abs(float(xi.sum()) - want) < 1e-6 * want


def test_rejects_a_kernel_that_does_not_match_its_two_grids():
    """A square kernel where a rectangular one is needed is the easiest mistake
    to make here, and it must fail loudly rather than broadcast."""
    deltas = [0.35, 0.5, 0.8, 1.1]
    ti, grids, wts, log_k, log_root, _, x, alpha, _ = _per_depth_setup(
        deltas, [301, 201, 151, 101]
    )
    bad = list(log_k)
    bad[1] = bad[1][: grids[2].size, : grids[2].size]
    with pytest.raises(ValueError, match="child x parent"):
        wavelet_tree_bp(
            grids, wts, bad, log_root, x, alpha, deltas, BRANCHING, DEPTH
        )


def test_rejects_mismatched_kernel_and_delta_counts():
    deltas = [0.35, 0.5, 0.8, 1.1]
    ti, grid, weights, log_k, log_root, _, x, alpha, _ = _setup(deltas, size=401)
    with pytest.raises(ValueError, match="kernels"):
        wavelet_tree_bp(grid, weights, log_k[:-1], log_root, x, alpha, deltas,
                        BRANCHING, DEPTH)
    with pytest.raises(ValueError, match="deltas"):
        wavelet_tree_bp(grid, weights, log_k, log_root, x, alpha, deltas[:-1],
                        BRANCHING, DEPTH)
