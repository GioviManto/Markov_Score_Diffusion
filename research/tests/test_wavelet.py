"""Orthonormal Haar transform and the quadtree it induces.

The load-bearing checks are the three that the rest of the image work rests on:
perfect reconstruction, orthonormality (which is what makes the diffusion
commute with the change of basis), and agreement between the spatial
parent-child map and `hierarchy.TreeIndex`'s breadth-first child rule -- the
latter because every BP routine in the package addresses nodes by that rule and
a silent permutation would produce plausible-looking nonsense.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.hierarchy import TreeIndex
from src.noising import alpha_delta
from src.utils import rng_for
from src.wavelet import (
    ORIENTATIONS,
    WaveletQuadtree,
    haar_analysis,
    haar_synthesis,
    images_to_tree,
    morton_positions,
    tree_to_images,
)


# ----------------------------------------------------------------------------
# The transform
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("levels", [1, 2, 3, 5])
def test_perfect_reconstruction_to_machine_precision(levels):
    rng = rng_for("wavelet-recon")
    img = rng.standard_normal((7, 32, 32))
    back = haar_synthesis(haar_analysis(img, levels), levels)
    assert np.max(np.abs(back - img)) < 1e-13


def test_reconstruction_of_a_structured_image():
    """A smooth ramp plus an edge -- the case where a buggy inverse still looks
    plausible on random input but visibly rings on structure."""
    x = np.linspace(0, 1, 32)
    img = x[None, :] + 0.5 * x[:, None]
    img[16:, :] += 2.0
    back = haar_synthesis(haar_analysis(img, 5), 5)
    assert np.max(np.abs(back - img)) < 1e-13


@pytest.mark.parametrize("levels", [1, 3, 5])
def test_transform_is_orthonormal(levels):
    """Parseval, and the explicit matrix check that W W^T = I.

    Orthonormality is not cosmetic: it is exactly the hypothesis under which
    coordinatewise OU noising in pixel space is coordinatewise OU noising in
    wavelet space, with the same alpha and Delta.
    """
    rng = rng_for("wavelet-ortho")
    img = rng.standard_normal((4, 32, 32))
    c = haar_analysis(img, levels)
    assert np.allclose((c**2).sum(axis=(1, 2)), (img**2).sum(axis=(1, 2)), atol=1e-11)

    basis = haar_analysis(np.eye(32 * 32).reshape(-1, 32, 32), levels)
    w = basis.reshape(32 * 32, 32 * 32)
    assert np.max(np.abs(w @ w.T - np.eye(32 * 32))) < 1e-12


def test_diffusion_commutes_with_the_transform():
    """The claim the whole image extension rests on.

    Noising in pixel space then transforming must have the same law as
    transforming then noising with the *same* (alpha, Delta). Checked in
    distribution via the sample covariance of the transformed noise, which must
    be the identity.
    """
    rng = rng_for("wavelet-commute")
    _, delta = alpha_delta(0.7)
    z = rng.standard_normal((20000, 32, 32))
    wz = haar_analysis(z, 5).reshape(20000, -1)
    cov = wz.T @ wz / wz.shape[0]
    off = cov - np.diag(np.diag(cov))
    # 20k samples: diagonal within ~3/sqrt(N) of 1, off-diagonal within ~4/sqrt(N).
    assert np.max(np.abs(np.diag(cov) - 1.0)) < 0.05
    assert np.max(np.abs(off)) < 0.05
    assert delta > 0


# ----------------------------------------------------------------------------
# The quadtree
# ----------------------------------------------------------------------------

def test_quadtree_shape_accounts_for_every_coefficient():
    qt = WaveletQuadtree(side=32, levels=5)
    assert qt.depth == 4
    assert qt.branching == 4
    assert qt.n_nodes == 341
    assert 3 * qt.n_nodes + 1 == 32 * 32


def test_partial_decomposition_is_refused():
    with pytest.raises(ValueError, match="forest"):
        WaveletQuadtree(side=32, levels=3)


def test_node_packing_round_trips():
    rng = rng_for("wavelet-pack")
    qt = WaveletQuadtree(side=32, levels=5)
    coeffs = rng.standard_normal((5, 32, 32))
    nodes, scaling = qt.to_nodes(coeffs)
    assert nodes.shape == (5, 3, 341)
    assert scaling.shape == (5, 1)
    assert np.max(np.abs(qt.from_nodes(nodes, scaling) - coeffs)) < 1e-14


def test_image_to_tree_and_back():
    rng = rng_for("wavelet-endtoend")
    img = rng.standard_normal((6, 32, 32))
    qt, nodes, scaling = images_to_tree(img, levels=5)
    assert np.max(np.abs(tree_to_images(qt, nodes, scaling) - img)) < 1e-13


def test_morton_order_matches_treeindex_child_rule():
    """The spatial child map and the breadth-first index must agree.

    For every node, the four children that `TreeIndex` assigns it must be the
    four coefficients that sit at (2m, 2n) .. (2m+1, 2n+1) in the finer subband.
    """
    qt = WaveletQuadtree(side=32, levels=5)
    ti = TreeIndex(depth=qt.depth, branching=4)
    for d in range(qt.depth):
        rows_p, cols_p = morton_positions(d)
        rows_c, cols_c = morton_positions(d + 1)
        for node in ti.nodes_at(d):
            pos = int(node) - ti.nodes_at(d)[0]
            m, n = rows_p[pos], cols_p[pos]
            kids = ti.children(int(node), d) - ti.nodes_at(d + 1)[0]
            got = {(int(rows_c[k]), int(cols_c[k])) for k in kids}
            want = {(2 * m, 2 * n), (2 * m, 2 * n + 1),
                    (2 * m + 1, 2 * n), (2 * m + 1, 2 * n + 1)}
            assert got == want


def test_nodes_recover_the_named_subbands():
    """A coefficient planted in one subband must appear at the expected depth of
    the expected orientation tree and nowhere else."""
    qt = WaveletQuadtree(side=32, levels=5)
    for oi, orient in enumerate(ORIENTATIONS):
        coeffs = np.zeros((32, 32))
        rs, cs = qt._subband_slice(3, orient)
        coeffs[rs, cs] = 1.0
        nodes, scaling = qt.to_nodes(coeffs)
        depth = qt.node_depth
        assert np.all(nodes[oi, depth == 3] == 1.0)
        assert np.all(nodes[oi, depth != 3] == 0.0)
        assert np.all(nodes[np.arange(3) != oi] == 0.0)
        assert np.all(scaling == 0.0)
