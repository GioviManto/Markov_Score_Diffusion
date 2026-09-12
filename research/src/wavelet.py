"""Orthonormal 2-D Haar transform and the wavelet quadtree it induces.

Why this module exists
----------------------
Everything else in this package needs a *tree*. A pixel lattice is not one: it
has loops, so BP on it is loopy and approximate, and the package's central claim
-- that sum-product returns the *exact* score -- would be lost. A multiscale
wavelet decomposition is a tree, which is the whole reason for this file.

The enabling fact, and it is exact
----------------------------------
The forward process of `src/noising.py` is

    x_t = alpha_t a + sqrt(Delta_t) z,      z ~ N(0, I),

applied coordinatewise in *pixel* space. Let `W` be the orthonormal Haar
analysis operator. Then

    W x_t = alpha_t (W a) + sqrt(Delta_t) (W z),     W z ~ N(0, I),

because an orthonormal map sends an isotropic Gaussian to an isotropic Gaussian.
So the noising process in wavelet space is *the same* coordinatewise OU process,
with the same alpha_t and Delta_t. Nothing is approximated by changing basis:
the per-site Gaussian likelihood factor that makes the posterior factor graph a
tree survives verbatim. And because `W` is orthonormal, the posterior mean and
the score map back by `W^T`:

    E[a | x_t] = W^T E[Wa | W x_t],        score_pixel = W^T score_wavelet.

This is why the transform must be *orthonormal* rather than merely invertible,
and it is checked in `tests/test_wavelet.py` rather than asserted.

The tree, concretely
--------------------
A J-level decomposition of a 2^J x 2^J image yields one scaling coefficient
(LL, a 1x1 band) and, at each scale, three detail subbands with orientations
HL / LH / HH. Following Crouse, Nowak and Baraniuk (1998), the tree is built
*within* an orientation: a coefficient at position (m, n) of a subband of side S
has four children at (2m, 2n), (2m, 2n+1), (2m+1, 2n), (2m+1, 2n+1) in the
subband of side 2S, same orientation. There are no edges between orientations
and none to LL, so the coefficient set decomposes into three disjoint quadtrees
plus one isolated scalar -- a forest of trees, still loop-free, still exact.

For a 32x32 image with J = 5 that is three quadtrees of branching 4 and depth 4
(subbands of side 1, 2, 4, 8, 16), i.e. 3 * 341 = 1023 coefficients, plus the
single LL coefficient: 1024 = 32 * 32, every coefficient accounted for exactly
once.

Note on the tree depth: a depth-5 quadtree would have 4^5 = 1024 leaves, i.e.
one leaf per *pixel*. That is a quadtree of the image, not of its wavelet
coefficients, and it is not what the wavelet HMT is. The correct object here has
depth J - 1 per orientation.

Node ordering
-------------
Nodes are numbered breadth-first to match `src/hierarchy.TreeIndex`: node p at
depth d occupies position p - offset(d) within its level, and its children are
offset(d+1) + 4 (p - offset(d)) + k. Composing that rule with the spatial
child map above makes the within-level ordering the Morton (Z-order) curve, so
`subband_positions(d)` returns the Morton permutation of the (2^d, 2^d) grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_SQRT2 = np.sqrt(2.0)

# Orientation names, in the order used by every packed array in this module.
ORIENTATIONS = ("HL", "LH", "HH")


# ----------------------------------------------------------------------------
# The transform
# ----------------------------------------------------------------------------

def _haar_step_1d(a: np.ndarray) -> np.ndarray:
    """One orthonormal Haar step along the last axis (even length)."""
    s = (a[..., 0::2] + a[..., 1::2]) / _SQRT2
    d = (a[..., 0::2] - a[..., 1::2]) / _SQRT2
    return np.concatenate([s, d], axis=-1)


def _ihaar_step_1d(c: np.ndarray) -> np.ndarray:
    """Inverse of `_haar_step_1d`; exact transpose, hence exact inverse."""
    half = c.shape[-1] // 2
    s, d = c[..., :half], c[..., half:]
    out = np.empty(c.shape, dtype=c.dtype)
    out[..., 0::2] = (s + d) / _SQRT2
    out[..., 1::2] = (s - d) / _SQRT2
    return out


def haar_analysis(images: np.ndarray, levels: int) -> np.ndarray:
    """Multi-level orthonormal 2-D Haar analysis, Mallat pyramid layout.

    `images` is (..., N, N) with N divisible by 2**levels. After k steps the
    leading (N/2^k, N/2^k) corner holds LL_k; at scale j (j = 1 finest) with
    S = N / 2^j the detail subbands sit at

        HL_j = [0:S,   S:2S]     LH_j = [S:2S, 0:S]     HH_j = [S:2S, S:2S].
    """
    a = np.asarray(images, dtype=float)
    n = a.shape[-1]
    if a.shape[-2] != n:
        raise ValueError(f"images must be square, got {a.shape[-2]}x{n}")
    if n % (2**levels) != 0:
        raise ValueError(f"side {n} is not divisible by 2**{levels}")

    out = a.copy()
    size = n
    for _ in range(levels):
        block = out[..., :size, :size]
        block = _haar_step_1d(block)                       # along rows
        block = _haar_step_1d(block.swapaxes(-1, -2)).swapaxes(-1, -2)
        out[..., :size, :size] = block
        size //= 2
    return out


def haar_synthesis(coeffs: np.ndarray, levels: int) -> np.ndarray:
    """Inverse of `haar_analysis`, same layout convention."""
    c = np.asarray(coeffs, dtype=float)
    n = c.shape[-1]
    if c.shape[-2] != n:
        raise ValueError(f"coefficients must be square, got {c.shape[-2]}x{n}")

    out = c.copy()
    size = n // (2**levels)
    for _ in range(levels):
        size *= 2
        block = out[..., :size, :size]
        block = _ihaar_step_1d(block.swapaxes(-1, -2)).swapaxes(-1, -2)
        block = _ihaar_step_1d(block)
        out[..., :size, :size] = block
    return out


# ----------------------------------------------------------------------------
# The quadtree induced on the detail coefficients
# ----------------------------------------------------------------------------

def morton_positions(d: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows and columns of the (2^d, 2^d) grid in Morton (Z-) order.

    Built by the same recursion that defines the tree, so it agrees with the
    breadth-first child rule by construction rather than by a bit-twiddling
    identity that would need its own proof.
    """
    rows = np.zeros(1, dtype=np.intp)
    cols = np.zeros(1, dtype=np.intp)
    for _ in range(d):
        rows = np.stack([2 * rows, 2 * rows, 2 * rows + 1, 2 * rows + 1], axis=1).ravel()
        cols = np.stack([2 * cols, 2 * cols + 1, 2 * cols, 2 * cols + 1], axis=1).ravel()
    return rows, cols


@dataclass(frozen=True)
class WaveletQuadtree:
    """Index bookkeeping between the pyramid layout and breadth-first tree order.

    `side` is the image side, `levels` the number of decomposition steps. The
    per-orientation tree has `depth = levels - 1` and branching 4.
    """

    side: int
    levels: int

    def __post_init__(self) -> None:
        if self.levels < 2:
            raise ValueError("need at least two levels for a non-trivial tree")
        if self.side != 2**self.levels:
            # A partial decomposition leaves an m x m coarsest detail band with
            # m > 1, i.e. m^2 roots per orientation: a forest, not a single tree.
            # BP is still exact on a forest, but the indexing is no longer
            # TreeIndex's, so the case is refused rather than half-supported.
            raise ValueError(
                f"side {self.side} must equal 2**levels = {2**self.levels}; "
                "a partial decomposition gives a forest of roots, unsupported here"
            )

    @property
    def depth(self) -> int:
        return self.levels - 1

    @property
    def branching(self) -> int:
        return 4

    @property
    def n_nodes(self) -> int:
        return (4 ** (self.depth + 1) - 1) // 3

    @property
    def node_depth(self) -> np.ndarray:
        """(n_nodes,) depth of each node; depth 0 is the coarsest detail band."""
        return np.concatenate([np.full(4**d, d) for d in range(self.depth + 1)])

    def subband_side(self, d: int) -> int:
        """Side of the detail subband holding tree depth `d`."""
        return 2**d

    def _subband_slice(self, d: int, orientation: str) -> tuple[slice, slice]:
        """Slices of the pyramid array for the subband at tree depth `d`.

        The subband has side s = 2^d and occupies one quadrant of the (2s, 2s)
        block, the remaining quadrant being everything coarser.
        """
        s = 2**d
        if orientation == "HL":
            return slice(0, s), slice(s, 2 * s)
        if orientation == "LH":
            return slice(s, 2 * s), slice(0, s)
        if orientation == "HH":
            return slice(s, 2 * s), slice(s, 2 * s)
        raise ValueError(f"unknown orientation {orientation!r}")

    def to_nodes(self, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pyramid array -> (nodes, scaling).

        `coeffs` is (..., side, side). Returns `nodes` of shape
        (..., 3, n_nodes) in breadth-first order per orientation, and `scaling`
        of shape (..., n_ll) holding the flattened LL band.
        """
        c = np.asarray(coeffs, dtype=float)
        lead = c.shape[:-2]
        nodes = np.empty(lead + (3, self.n_nodes), dtype=float)
        for oi, orient in enumerate(ORIENTATIONS):
            start = 0
            for d in range(self.depth + 1):
                rs, cs = self._subband_slice(d, orient)
                band = c[..., rs, cs]
                rows, cols = morton_positions(d)
                n_here = 4**d
                nodes[..., oi, start : start + n_here] = band[..., rows, cols]
                start += n_here
        ll = self.side // (2**self.levels)
        scaling = c[..., :ll, :ll].reshape(lead + (ll * ll,))
        return nodes, scaling

    def from_nodes(self, nodes: np.ndarray, scaling: np.ndarray) -> np.ndarray:
        """Inverse of `to_nodes`: rebuild the pyramid array."""
        nodes = np.asarray(nodes, dtype=float)
        lead = nodes.shape[:-2]
        out = np.zeros(lead + (self.side, self.side), dtype=float)
        for oi, orient in enumerate(ORIENTATIONS):
            start = 0
            for d in range(self.depth + 1):
                rs, cs = self._subband_slice(d, orient)
                rows, cols = morton_positions(d)
                n_here = 4**d
                band = np.zeros(lead + (2**d, 2**d), dtype=float)
                band[..., rows, cols] = nodes[..., oi, start : start + n_here]
                out[..., rs, cs] = band
                start += n_here
        ll = self.side // (2**self.levels)
        out[..., :ll, :ll] = np.asarray(scaling, dtype=float).reshape(lead + (ll, ll))
        return out

    def subband_label(self, node: int, orientation_index: int) -> str:
        """Human-readable name, e.g. `HL_d2` for depth 2 of the HL tree."""
        d = int(self.node_depth[node])
        return f"{ORIENTATIONS[orientation_index]}_d{d}"


# ----------------------------------------------------------------------------
# Convenience: the whole pipeline in one call
# ----------------------------------------------------------------------------

def images_to_tree(
    images: np.ndarray, levels: int
) -> tuple[WaveletQuadtree, np.ndarray, np.ndarray]:
    """(B, N, N) images -> (tree, nodes (B, 3, n_nodes), scaling (B, n_ll))."""
    images = np.asarray(images, dtype=float)
    qt = WaveletQuadtree(side=images.shape[-1], levels=levels)
    coeffs = haar_analysis(images, levels)
    nodes, scaling = qt.to_nodes(coeffs)
    return qt, nodes, scaling


def tree_to_images(
    qt: WaveletQuadtree, nodes: np.ndarray, scaling: np.ndarray
) -> np.ndarray:
    """Inverse of `images_to_tree`."""
    return haar_synthesis(qt.from_nodes(nodes, scaling), qt.levels)
