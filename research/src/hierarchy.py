"""Hierarchical (tree) priors and exact BP on them.

Why this module exists
----------------------
Everything else in this package lives on a *chain*. A chain is a tree, so BP is
exact there, but a chain has only one length scale: correlations decay as
rho^{|i-j|} with a single correlation length. Garnier-Brun, Mezard, Moscato and
Saglietti (arXiv:2408.15138) study data generated on a *balanced tree*, where
correlations live on a discrete ladder of length scales, one per level of the
hierarchy, and can be switched off level by level ("hierarchical filtering").
That is the natural setting in which "which correlations has the model learned"
becomes a question with a graded answer rather than a yes/no.

This module provides the continuous analogue of that data model:

    z_root ~ N(0, 1)
    z_child = rho z_parent + sqrt(1 - rho^2) eps,     eps ~ p_innov (unit var)
    leaves are observed; internal nodes are latent.

Two leaves whose lowest common ancestor sits at depth d are separated by
2(L - d) edges, so for Gaussian innovations

    Cov(leaf_i, leaf_j) = rho^{2(L - d(i,j))},

an ultrametric covariance with exactly L + 1 distinct eigenvalues. Those
eigenvalues are what `src/spectral.py` turns into a ladder of speciation times.

Two BP implementations, mirroring Layers 1-2 of the package:

- `tree_bp_gaussian` -- information form (h, lambda), exact for Gaussian
  innovations, O(n) and free of any discretization. This is the reference.
- `tree_bp_grid` -- grid messages, exact up to quadrature for *any* innovation
  law. Same two primitives as the chain (`K.T @ f` upward, `K @ g` downward),
  so the conventions of `src/bp_grid.py` carry over unchanged: K[out, in].

`tree_bp_grid` agrees with `tree_bp_gaussian` to ~1e-10 on Gaussian innovations
(tests/test_hierarchy.py), which is the same cross-check that validated the
chain code.

Node indexing
-------------
Nodes are numbered breadth-first, root = 0. A node at depth d has index in
[offset(d), offset(d) + b^d). Children of node i at depth d are
offset(d+1) + b * (i - offset(d)) + k for k in 0..b-1. Leaves are the last
b^L entries, in left-to-right order, so leaf j of the sequence is node
offset(L) + j.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------------
# Tree bookkeeping
# ----------------------------------------------------------------------------

def level_offset(depth: int, branching: int) -> int:
    """Index of the first node at `depth` in the breadth-first numbering."""
    if branching == 1:
        return depth
    return (branching**depth - 1) // (branching - 1)


@dataclass(frozen=True)
class TreeIndex:
    """Breadth-first index of a balanced `branching`-ary tree of depth `depth`."""

    depth: int
    branching: int

    @property
    def n_nodes(self) -> int:
        return level_offset(self.depth + 1, self.branching)

    @property
    def n_leaves(self) -> int:
        return self.branching**self.depth

    def nodes_at(self, d: int) -> np.ndarray:
        start = level_offset(d, self.branching)
        return np.arange(start, start + self.branching**d)

    def children(self, node: int, d: int) -> np.ndarray:
        pos = node - level_offset(d, self.branching)
        start = level_offset(d + 1, self.branching) + self.branching * pos
        return np.arange(start, start + self.branching)

    def leaf_nodes(self) -> np.ndarray:
        return self.nodes_at(self.depth)

    def lca_depth_matrix(self) -> np.ndarray:
        """`D[i, j]` = depth of the lowest common ancestor of leaves i, j.

        Computed from the base-`b` digit expansion of the leaf positions: the
        LCA depth is the length of the common prefix.
        """
        b, L = self.branching, self.depth
        n = self.n_leaves
        idx = np.arange(n)
        digits = np.stack([(idx // b**(L - 1 - k)) % b for k in range(L)], axis=1)
        same = digits[:, None, :] == digits[None, :, :]
        # Common prefix length = number of leading True values.
        prefix = np.cumprod(same, axis=2).sum(axis=2)
        return prefix.astype(int)


# ----------------------------------------------------------------------------
# The prior
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class GaussianTree:
    """Balanced-tree Gaussian prior on `branching**depth` observed leaves.

    `rho` is the per-edge correlation; innovations have variance 1 - rho^2 so
    every node, including the leaves, has unit marginal variance -- the same
    normalization the chain priors use, which is what makes the two families
    directly comparable at fixed noise schedule.
    """

    depth: int
    branching: int = 2
    rho: float = 0.9
    root_separation: float = 0.0
    filter_level: int = 0

    @property
    def name(self) -> str:
        tag = f",mu={self.root_separation}" if self.root_separation else ""
        tag += f",k={self.filter_level}" if self.filter_level else ""
        return f"tree(L={self.depth},b={self.branching},rho={self.rho}{tag})"

    def __post_init__(self) -> None:
        if not 0 <= self.filter_level <= self.depth:
            raise ValueError("filter_level must satisfy 0 <= k <= depth.")

    # -- hierarchical filtering (Garnier-Brun et al., arXiv:2408.15138 §2.2) --
    #
    # At filter level k the b^k nodes at depth k are drawn **conditionally
    # independently given the root**, each with the marginal it would have in
    # the unfiltered model -- for the Gaussian tree, correlation rho^k with the
    # root and unit variance. Levels 1..k-1 are skipped entirely. Below depth k
    # the ordinary recursion resumes, so correlations survive inside blocks of
    # b^{L-k} leaves.
    #
    # The consequence for the covariance is exact and worth stating, because it
    # is what makes the filtered model a clean experimental knob: two leaves in
    # the *same* depth-k block keep the covariance they had, while every
    # cross-block pair collapses to rho^{2L} -- the value the unfiltered model
    # assigns to the most distant pair. Filtering flattens the top of the
    # hierarchy without touching the marginals or the bottom of it.
    #
    # Note this is NOT "b^k independent subtrees": the blocks stay correlated
    # through the root. At k = L the leaves become conditionally i.i.d. given
    # the root, which is the regime where the paper notes a Naive Bayes
    # classifier is optimal and attention is superfluous.

    @property
    def block_size(self) -> int:
        """Leaves per depth-`k` block; the whole sequence when unfiltered."""
        return self.branching ** (self.depth - self.filter_level)

    @property
    def n_blocks(self) -> int:
        return self.branching**self.filter_level

    @property
    def subtree(self) -> "GaussianTree":
        """The unfiltered depth-(L-k) tree that lives inside one block."""
        return GaussianTree(
            depth=self.depth - self.filter_level,
            branching=self.branching,
            rho=self.rho,
        )

    @property
    def cross_block_covariance(self) -> float:
        """`rho^{2L}`: what every cross-block leaf pair collapses to."""
        return float(self.rho ** (2 * self.depth))

    @property
    def top_kernel(self) -> tuple[float, float]:
        """`(rho^k, 1 - rho^{2k})` -- the root-to-block-root edge after filtering."""
        r = float(self.rho**self.filter_level)
        return r, float(1.0 - r**2)

    @property
    def index(self) -> TreeIndex:
        return TreeIndex(self.depth, self.branching)

    @property
    def n_leaves(self) -> int:
        return self.index.n_leaves

    @property
    def q(self) -> float:
        return 1.0 - self.rho**2

    def log_root_density(self, grid: np.ndarray) -> np.ndarray:
        """Log prior of the root on `grid`.

        With `root_separation = mu > 0` the root is the symmetric two-component
        mixture `1/2 N(-mu, 1-mu^2) + 1/2 N(+mu, 1-mu^2)`, which keeps unit
        variance and therefore leaves the leaf covariance -- and hence the whole
        eigenvalue ladder and every speciation time -- **exactly unchanged**.
        Only the modality changes. That is the point: it turns the coarsest
        transition from an information cross-over into a genuine choice between
        two classes, at the same predicted time, with nothing else moved.
        """
        mu = float(self.root_separation)
        if mu <= 0.0:
            return -0.5 * grid**2 - 0.5 * np.log(2.0 * np.pi)
        var = 1.0 - mu**2
        if var <= 0.0:
            raise ValueError("root_separation must be < 1 to keep unit variance.")
        a = -0.5 * (grid - mu) ** 2 / var
        b = -0.5 * (grid + mu) ** 2 / var
        hi = np.maximum(a, b)
        return (
            hi + np.log(np.exp(a - hi) + np.exp(b - hi))
            - 0.5 * np.log(2.0 * np.pi * var) - np.log(2.0)
        )

    def sample_nodes(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Sample `n` full trees; returns `(n, n_nodes)` including latents."""
        ti = self.index
        z = np.empty((n, ti.n_nodes))
        mu = float(self.root_separation)
        if mu > 0.0:
            sign = rng.integers(0, 2, size=n) * 2.0 - 1.0
            z[:, 0] = sign * mu + np.sqrt(1.0 - mu**2) * rng.standard_normal(n)
        else:
            z[:, 0] = rng.standard_normal(n)
        scale = np.sqrt(self.q)
        k = self.filter_level
        for d in range(self.depth):
            parents = ti.nodes_at(d)
            kids = ti.nodes_at(d + 1)
            if k > 0 and d < k:
                # Filtered levels. Every node at depth d+1 <= k is drawn
                # directly from the *root*, conditionally independently, with
                # the marginal the unfiltered model would give it: correlation
                # rho^{d+1} with the root and unit variance. Levels 1..k-1 are
                # therefore placeholders carrying the right marginal but no
                # sibling structure -- exactly the paper's construction.
                r = self.rho ** (d + 1)
                z[:, kids] = r * np.repeat(
                    z[:, :1], kids.size, axis=1
                ) + np.sqrt(1.0 - r**2) * rng.standard_normal((n, kids.size))
            else:
                # Children of parents[p] are kids[b*p : b*p+b].
                z[:, kids] = (
                    self.rho * np.repeat(z[:, parents], self.branching, axis=1)
                    + scale * rng.standard_normal((n, kids.size))
                )
        return z

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Sample `n` leaf sequences; returns `(n, n_leaves)`."""
        return self.sample_nodes(rng, n)[:, self.index.leaf_nodes()]

    def leaf_covariance(self) -> np.ndarray:
        """Dense `(n_leaves, n_leaves)` covariance -- reference, O(n^2) memory.

        Unfiltered: `rho^{2(L - d_LCA)}`, the ultrametric form. Filtered at
        level k: unchanged for pairs inside a depth-k block (`d_LCA >= k`), and
        flattened to `rho^{2L}` for every cross-block pair, since those are
        correlated only through the root.
        """
        d = self.index.lca_depth_matrix()
        cov = self.rho ** (2.0 * (self.depth - d))
        if self.filter_level:
            cov = np.where(d >= self.filter_level, cov, self.cross_block_covariance)
        return cov

    # -- spectrum -----------------------------------------------------------

    def subtree_row_sum(self, d: int) -> float:
        """Sum of Cov(i, j) over leaves j inside a depth-`d` subtree, i fixed in it.

        With m = L - d levels below, the subtree holds b^m leaves; of these,
        (b-1) b^{k-1} sit at LCA distance k for k = 1..m, contributing
        rho^{2k} each, plus the leaf itself.
        """
        b, m = self.branching, self.depth - d
        k = np.arange(1, m + 1)
        return float(1.0 + (b - 1) * np.sum(b ** (k - 1.0) * self.rho ** (2.0 * k)))

    def level_eigenvalues(self) -> list[tuple[int, float, int]]:
        """Exact spectrum as `[(level, eigenvalue, multiplicity), ...]`.

        The ultrametric covariance is diagonalized by the uniform vector plus,
        at each depth d = 0..L-1, the contrasts between the b children of a
        depth-d node (constant on each child subtree, summing to zero). For such
        a contrast, cross-subtree terms collapse because the weights sum to zero:

            Lambda_d = S_{d+1} - b^{L-d-1} rho^{2(L-d)},

        with multiplicity (b-1) b^d; the uniform vector has Lambda = S_0 with
        multiplicity 1. Multiplicities sum to b^L and the trace to b^L, both
        asserted in the tests.

        Levels are labelled by the depth `d` of the node whose children the
        contrast separates, with `-1` reserved for the uniform mode. So the
        label increases as the mode gets *finer*: level -1 is the mean of the
        whole tree, level 0 splits it in half, level L-1 separates sibling
        leaves. Eigenvalues decrease monotonically in that order, which is what
        makes the speciation ladder well ordered.

        Filtering collapses the top of the ladder. At filter level k the
        covariance is block-diagonal-plus-constant: the depth-(L-k) subtree
        covariance inside each of the b^k blocks, and `rho^{2L}` between them.
        Its eigenvectors are (i) within-block contrasts, which are the
        subtree's own non-uniform modes, each with multiplicity multiplied by
        b^k; (ii) between-block contrasts, constant on each block and summing
        to zero across blocks, with the single eigenvalue

            S_0^{(L-k)} - b^{L-k} rho^{2L},   multiplicity b^k - 1;

        and (iii) the uniform mode. So the k top rungs of the ladder merge into
        **one**, and the number of distinct speciation times falls from L + 1
        to L - k + 2 (and to 2 at k = L, where the leaves are conditionally
        i.i.d. given the root and the covariance is equicorrelated). Filtering
        is therefore a knob that removes rungs from the speciation ladder --
        which is the point at which the two papers meet.

        Filtered levels are labelled `-2` for the merged between-block mode;
        within-block levels keep the depth label they have in the *full* tree,
        i.e. `k + d'` for the subtree's level `d'`.
        """
        b, L, k = self.branching, self.depth, self.filter_level
        if k == 0:
            out: list[tuple[int, float, int]] = [(-1, self.subtree_row_sum(0), 1)]
            for d in range(L):
                lam = self.subtree_row_sum(d + 1) - b ** (L - d - 1) * self.rho ** (
                    2.0 * (L - d)
                )
                out.append((d, float(lam), (b - 1) * b**d))
            return out

        sub = self.subtree
        n_blocks, m = self.n_blocks, self.block_size
        c0 = self.cross_block_covariance
        s_sub = sub.subtree_row_sum(0)

        out = [(-1, float(s_sub + (n_blocks - 1) * m * c0), 1)]
        if n_blocks > 1:
            out.append((-2, float(s_sub - m * c0), n_blocks - 1))
        for level, lam, mult in sub.level_eigenvalues():
            if level < 0:
                continue                      # the subtree's uniform mode is (ii)
            out.append((k + level, float(lam), mult * n_blocks))
        return out

    def level_projector_basis(self) -> tuple[np.ndarray, np.ndarray]:
        """Orthonormal eigenbasis `(V, levels)` with `V[:, k]` an eigenvector.

        Built explicitly (rather than by `eigh`) so that each column carries the
        hierarchy level it belongs to; `eigh` would return an arbitrary rotation
        inside each degenerate eigenspace, which destroys exactly the labelling
        this module is for.
        """
        b, L = self.branching, self.depth
        n = self.n_leaves
        cols: list[np.ndarray] = []
        levels: list[int] = []

        uniform = np.full(n, 1.0 / np.sqrt(n))
        cols.append(uniform)
        levels.append(-1)

        if self.filter_level:
            # Between-block contrasts (label -2), then the subtree's own basis
            # replicated per block with its levels shifted by k.
            k, m, n_blocks = self.filter_level, self.block_size, self.n_blocks
            for r in range(1, n_blocks):
                v = np.zeros(n)
                v[: r * m] = 1.0
                v[r * m: (r + 1) * m] = -float(r)
                cols.append(v / np.linalg.norm(v))
                levels.append(-2)
            v_sub, lev_sub = self.subtree.level_projector_basis()
            for block in range(n_blocks):
                for col in range(v_sub.shape[1]):
                    if lev_sub[col] < 0:
                        continue              # subtree uniform mode is spanned above
                    v = np.zeros(n)
                    v[block * m: (block + 1) * m] = v_sub[:, col]
                    cols.append(v)
                    levels.append(k + int(lev_sub[col]))
            return np.stack(cols, axis=1), np.asarray(levels)

        for d in range(L):
            block = b ** (L - d)          # leaves under a depth-d node
            child = b ** (L - d - 1)      # leaves under each of its children
            for node in range(b**d):
                base = node * block
                # (b-1) orthonormal contrasts among the b child blocks: use the
                # Helmert basis, which is orthonormal and sums to zero.
                for r in range(1, b):
                    v = np.zeros(n)
                    for k in range(r):
                        v[base + k * child: base + (k + 1) * child] = 1.0
                    v[base + r * child: base + (r + 1) * child] = -float(r)
                    v /= np.linalg.norm(v)
                    cols.append(v)
                    levels.append(d)
        return np.stack(cols, axis=1), np.asarray(levels)


# ----------------------------------------------------------------------------
# Exact BP on the tree -- information form (Gaussian)
# ----------------------------------------------------------------------------

def tree_bp_gaussian(
    tree: GaussianTree, x: np.ndarray, alpha: float, delta: float
) -> np.ndarray:
    """Exact posterior mean of the leaves given noisy leaves, by BP.

    `x` is `(n_chains, n_leaves)` of observations y_i = alpha a_i + sqrt(delta) z.
    Returns `(n_chains, n_leaves)` posterior means E[a | y].

    Messages are carried in information form (h, lam) = (Sigma^{-1} mu,
    Sigma^{-1}), the parameterization the audit identified as the correct one:
    the update touches *only* these two numbers per message, with no projection
    or moment-matching step anywhere. Upward and downward passes use

        up:    lam' = rho^2 lam / (1 + q lam),      h' = rho h / (1 + q lam)
        down:  lam' = 1 / (rho^2 / lam + q),        h' = lam' rho h / lam

    which are the marginalize-then-transport and transport-then-condition forms
    of the same linear-Gaussian edge.
    """
    ti = tree.index
    b, L, q, rho = tree.branching, tree.depth, tree.q, tree.rho
    if tree.root_separation:
        raise ValueError(
            "The information form assumes a Gaussian root; a mixture root has "
            "no (h, lambda) representation. Use tree_bp_grid with "
            "tree.log_root_density(grid)."
        )
    x = np.atleast_2d(x)
    n_chains = x.shape[0]
    if x.shape[1] != tree.n_leaves:
        raise ValueError(f"x has {x.shape[1]} leaves, tree has {tree.n_leaves}")

    n_nodes = ti.n_nodes
    # Evidence: only leaves are observed.
    lam_obs = np.zeros((n_chains, n_nodes))
    h_obs = np.zeros((n_chains, n_nodes))
    leaves = ti.leaf_nodes()
    lam_obs[:, leaves] = alpha**2 / delta
    h_obs[:, leaves] = alpha * x / delta

    # Upward pass: message from each node to its parent.
    lam_up = np.zeros((n_chains, n_nodes))
    h_up = np.zeros((n_chains, n_nodes))
    lam_tot = lam_obs.copy()   # evidence + all upward messages received
    h_tot = h_obs.copy()
    for d in range(L, 0, -1):
        nodes = ti.nodes_at(d)
        lam, h = lam_tot[:, nodes], h_tot[:, nodes]
        denom = 1.0 + q * lam
        lam_up[:, nodes] = rho**2 * lam / denom
        h_up[:, nodes] = rho * h / denom
        # Accumulate into parents: children are contiguous in blocks of b.
        parents = ti.nodes_at(d - 1)
        lam_tot[:, parents] += lam_up[:, nodes].reshape(n_chains, -1, b).sum(axis=2)
        h_tot[:, parents] += h_up[:, nodes].reshape(n_chains, -1, b).sum(axis=2)

    # Root prior N(0, 1): precision 1, information 0.
    lam_tot[:, 0] += 1.0

    # Downward pass.
    lam_down = np.zeros((n_chains, n_nodes))
    h_down = np.zeros((n_chains, n_nodes))
    for d in range(L):
        parents = ti.nodes_at(d)
        kids = ti.nodes_at(d + 1)
        lam_p = lam_tot[:, parents] + lam_down[:, parents]
        h_p = h_tot[:, parents] + h_down[:, parents]
        # Exclude each child's own upward message (repeat parent along children).
        lam_excl = np.repeat(lam_p, b, axis=1) - lam_up[:, kids]
        h_excl = np.repeat(h_p, b, axis=1) - h_up[:, kids]
        lam_new = 1.0 / (rho**2 / lam_excl + q)
        lam_down[:, kids] = lam_new
        h_down[:, kids] = lam_new * rho * h_excl / lam_excl

    lam_leaf = lam_tot[:, leaves] + lam_down[:, leaves]
    h_leaf = h_tot[:, leaves] + h_down[:, leaves]
    return h_leaf / lam_leaf


def filtered_tree_bp_gaussian(
    tree: GaussianTree, x: np.ndarray, alpha: float, delta: float
) -> np.ndarray:
    """Exact posterior mean of the leaves under a *filtered* tree prior.

    This is `BP_k` in the language of Garnier-Brun et al.: the exact inference
    algorithm for the hierarchy truncated at level `k`. Running it on data drawn
    from a *different* filter level is exactly the mismatched-oracle comparison
    their Figs. 1(c–d) and 2 are built on, and it is what makes "which
    correlation range has this model learned" a question with a graded answer.

    The filtered graph is still a tree, so BP is still exact: a root, `b^k`
    block roots attached to it by the composed edge `(rho^k, 1 - rho^{2k})`, and
    an ordinary depth-`(L-k)` tree under each. Only the top edge and the root's
    degree differ from the unfiltered case, so the same information-form updates
    apply throughout.
    """
    k = tree.filter_level
    if k == 0:
        return tree_bp_gaussian(tree, x, alpha, delta)
    if tree.root_separation:
        raise ValueError("Mixture root has no information form; use grid BP.")

    x = np.atleast_2d(x)
    n_chains = x.shape[0]
    sub, m, n_blocks = tree.subtree, tree.block_size, tree.n_blocks
    rho_top, q_top = tree.top_kernel

    if k == tree.depth:
        # Degenerate: each "block" is a single leaf, no subtree pass at all.
        lam_up = np.full((n_chains, n_blocks), alpha**2 / delta)
        h_up = alpha * x / delta
        bu_lam, bu_h = lam_up, h_up
    else:
        blocks = x.reshape(n_chains * n_blocks, m)
        bu_lam, bu_h = _subtree_upward(sub, blocks, alpha, delta)
        bu_lam = bu_lam.reshape(n_chains, n_blocks)
        bu_h = bu_h.reshape(n_chains, n_blocks)

    # Block root -> root, through the composed top edge.
    denom = 1.0 + q_top * bu_lam
    lam_msg = rho_top**2 * bu_lam / denom
    h_msg = rho_top * bu_h / denom

    lam_root = 1.0 + lam_msg.sum(axis=1, keepdims=True)
    h_root = h_msg.sum(axis=1, keepdims=True)

    # Root -> block root, excluding each block's own upward message.
    lam_excl = lam_root - lam_msg
    h_excl = h_root - h_msg
    lam_down = 1.0 / (rho_top**2 / lam_excl + q_top)
    h_down = lam_down * rho_top * h_excl / lam_excl

    if k == tree.depth:
        lam_leaf = lam_down + alpha**2 / delta
        h_leaf = h_down + alpha * x / delta
        return h_leaf / lam_leaf

    return _subtree_downward(
        sub,
        x.reshape(n_chains * n_blocks, m),
        lam_down.reshape(-1),
        h_down.reshape(-1),
        alpha,
        delta,
    ).reshape(n_chains, tree.n_leaves)


def _subtree_upward(tree: GaussianTree, x: np.ndarray, alpha: float, delta: float):
    """Upward pass only; returns the `(lam, h)` the subtree root sends upward."""
    ti = tree.index
    b, L, q, rho = tree.branching, tree.depth, tree.q, tree.rho
    n_chains = x.shape[0]
    lam_tot = np.zeros((n_chains, ti.n_nodes))
    h_tot = np.zeros((n_chains, ti.n_nodes))
    leaves = ti.leaf_nodes()
    lam_tot[:, leaves] = alpha**2 / delta
    h_tot[:, leaves] = alpha * x / delta
    for d in range(L, 0, -1):
        nodes = ti.nodes_at(d)
        denom = 1.0 + q * lam_tot[:, nodes]
        lam_up = rho**2 * lam_tot[:, nodes] / denom
        h_up = rho * h_tot[:, nodes] / denom
        parents = ti.nodes_at(d - 1)
        lam_tot[:, parents] += lam_up.reshape(n_chains, -1, b).sum(axis=2)
        h_tot[:, parents] += h_up.reshape(n_chains, -1, b).sum(axis=2)
    return lam_tot[:, 0], h_tot[:, 0]


def _subtree_downward(
    tree: GaussianTree, x: np.ndarray, lam_in: np.ndarray, h_in: np.ndarray,
    alpha: float, delta: float,
) -> np.ndarray:
    """Full BP on a subtree whose root carries an external prior `(lam_in, h_in)`.

    Reruns the upward pass rather than caching it: the subtrees here are tiny
    and a second pass is far cheaper than the bookkeeping needed to thread
    cached messages through a differently-rooted call.
    """
    ti = tree.index
    b, L, q, rho = tree.branching, tree.depth, tree.q, tree.rho
    n_chains = x.shape[0]
    lam_up = np.zeros((n_chains, ti.n_nodes))
    h_up = np.zeros((n_chains, ti.n_nodes))
    lam_tot = np.zeros((n_chains, ti.n_nodes))
    h_tot = np.zeros((n_chains, ti.n_nodes))
    leaves = ti.leaf_nodes()
    lam_tot[:, leaves] = alpha**2 / delta
    h_tot[:, leaves] = alpha * x / delta

    for d in range(L, 0, -1):
        nodes = ti.nodes_at(d)
        denom = 1.0 + q * lam_tot[:, nodes]
        lam_up[:, nodes] = rho**2 * lam_tot[:, nodes] / denom
        h_up[:, nodes] = rho * h_tot[:, nodes] / denom
        parents = ti.nodes_at(d - 1)
        lam_tot[:, parents] += lam_up[:, nodes].reshape(n_chains, -1, b).sum(axis=2)
        h_tot[:, parents] += h_up[:, nodes].reshape(n_chains, -1, b).sum(axis=2)

    # The subtree root's prior comes from above instead of N(0, 1).
    lam_down = np.zeros((n_chains, ti.n_nodes))
    h_down = np.zeros((n_chains, ti.n_nodes))
    lam_down[:, 0] = lam_in
    h_down[:, 0] = h_in

    for d in range(L):
        parents = ti.nodes_at(d)
        kids = ti.nodes_at(d + 1)
        lam_p = lam_tot[:, parents] + lam_down[:, parents]
        h_p = h_tot[:, parents] + h_down[:, parents]
        lam_excl = np.repeat(lam_p, b, axis=1) - lam_up[:, kids]
        h_excl = np.repeat(h_p, b, axis=1) - h_up[:, kids]
        lam_new = 1.0 / (rho**2 / lam_excl + q)
        lam_down[:, kids] = lam_new
        h_down[:, kids] = lam_new * rho * h_excl / lam_excl

    lam_leaf = lam_tot[:, leaves] + lam_down[:, leaves]
    h_leaf = h_tot[:, leaves] + h_down[:, leaves]
    return h_leaf / lam_leaf


def tree_posterior_mean_dense(
    tree: GaussianTree, x: np.ndarray, alpha: float, delta: float
) -> np.ndarray:
    """Same quantity by dense linear algebra -- O(n^3), used only to check BP."""
    c = tree.leaf_covariance()
    n = c.shape[0]
    # Posterior of a given y = alpha a + sqrt(delta) z:  mean = C A^T (A C A^T + D)^-1 y
    gram = alpha**2 * c + delta * np.eye(n)
    return np.atleast_2d(x) @ np.linalg.solve(gram, alpha * c).T


def tree_score_gaussian(
    tree: GaussianTree, x: np.ndarray, t: float
) -> np.ndarray:
    """Exact score via the identity s = -(x - alpha m) / Delta."""
    alpha = float(np.exp(-t))
    delta = float(1.0 - np.exp(-2.0 * t))
    m = tree_bp_gaussian(tree, x, alpha, delta)
    return -(np.atleast_2d(x) - alpha * m) / delta


# ----------------------------------------------------------------------------
# Exact BP on the tree -- grid messages (any innovation law)
# ----------------------------------------------------------------------------

def tree_bp_grid(
    log_k: np.ndarray,
    grid: np.ndarray,
    log_root: np.ndarray,
    x: np.ndarray,
    alpha: float,
    delta: float,
    branching: int,
    depth: int,
) -> np.ndarray:
    """Posterior mean of the leaves by grid BP on the tree.

    `log_k[k, j] = log p(u_k | u_j)` -- the same `K[out, in]` convention as
    `src/bp_grid.py`, so the upward message is `K.T @ f` (marginalize the child)
    and the downward message is `K @ g` (transport the parent). `log_root` is
    the root's log prior on the grid.

    Works for any innovation law, which is the whole point: for a non-Gaussian
    tree there is no information-form shortcut, and the dense reference does not
    exist either.
    """
    from .noising import likelihood_matrix

    ti = TreeIndex(depth, branching)
    x = np.atleast_2d(x)
    n_chains, n_leaves = x.shape
    m = grid.size
    if n_leaves != ti.n_leaves:
        raise ValueError(f"x has {n_leaves} leaves, tree has {ti.n_leaves}")

    k_mat = np.exp(log_k - log_k.max())
    dx = float(grid[1] - grid[0])
    root_prior = np.exp(log_root - log_root.max())

    def norm(v: np.ndarray) -> np.ndarray:
        return v / np.maximum(v.sum(axis=-1, keepdims=True), 1e-300)

    n_nodes = ti.n_nodes
    up = np.ones((n_chains, n_nodes, m))         # message node -> parent
    belief_up = np.ones((n_chains, n_nodes, m))  # evidence * all children

    leaves = ti.leaf_nodes()
    for c in range(n_chains):
        belief_up[c, leaves] = likelihood_matrix(grid, x[c], alpha, delta)

    for d in range(depth, 0, -1):
        nodes = ti.nodes_at(d)
        f = norm(belief_up[:, nodes])
        msg = norm(np.einsum("cnk,kj->cnj", f, k_mat) * dx)
        up[:, nodes] = msg
        parents = ti.nodes_at(d - 1)
        prod = msg.reshape(n_chains, -1, branching, m).prod(axis=2)
        belief_up[:, parents] = norm(belief_up[:, parents] * prod)

    down = np.ones((n_chains, n_nodes, m))       # message parent -> node
    for d in range(depth):
        parents = ti.nodes_at(d)
        kids = ti.nodes_at(d + 1)
        # Leave-one-out product over siblings, by prefix/suffix scans. Doing it
        # as (product / own message) instead would divide by message entries
        # that are legitimately ~1e-16 in the tails, which is where a tree BP
        # implementation usually starts quietly losing digits.
        msgs = up[:, kids].reshape(n_chains, -1, branching, m)
        loo = _leave_one_out_product(msgs)
        extra = down[:, parents]
        if d == 0:
            extra = extra * root_prior
        excl = norm(loo * extra[:, :, None, :]).reshape(n_chains, -1, m)
        down[:, kids] = norm(np.einsum("cnj,kj->cnk", excl, k_mat) * dx)

    belief = norm(belief_up[:, leaves] * down[:, leaves])
    return belief @ grid


def tree_root_belief(
    log_k: np.ndarray,
    grid: np.ndarray,
    log_root: np.ndarray,
    x: np.ndarray,
    alpha: float,
    delta: float,
    branching: int,
    depth: int,
) -> np.ndarray:
    """Exact posterior of the *root* given the noisy leaves, `(n_chains, M)`.

    Only the upward pass is needed: the root receives no downward message, so
    its belief is its prior times the product of its children's messages. This
    is the quantity that makes symmetry breaking measurable rather than
    inferred -- with a two-component root prior, `P(root > 0 | x_t)` *is* the
    class posterior, exactly, at every noise level.
    """
    from .noising import likelihood_matrix

    ti = TreeIndex(depth, branching)
    x = np.atleast_2d(x)
    n_chains, n_leaves = x.shape
    if n_leaves != ti.n_leaves:
        raise ValueError(f"x has {n_leaves} leaves, tree has {ti.n_leaves}")
    m = grid.size
    k_mat = np.exp(log_k - log_k.max())
    dx = float(grid[1] - grid[0])

    def norm(v):
        return v / np.maximum(v.sum(axis=-1, keepdims=True), 1e-300)

    bu = np.ones((n_chains, ti.n_nodes, m))
    leaves = ti.leaf_nodes()
    for c in range(n_chains):
        bu[c, leaves] = likelihood_matrix(grid, x[c], alpha, delta)

    for d in range(depth, 0, -1):
        nodes = ti.nodes_at(d)
        msg = norm(np.einsum("cnk,kj->cnj", norm(bu[:, nodes]), k_mat) * dx)
        parents = ti.nodes_at(d - 1)
        prod = msg.reshape(n_chains, -1, branching, m).prod(axis=2)
        bu[:, parents] = norm(bu[:, parents] * prod)

    return norm(bu[:, 0] * np.exp(log_root - log_root.max()))


def _leave_one_out_product(msgs: np.ndarray, xp=None) -> np.ndarray:
    """`out[..., i, :] = prod_{j != i} msgs[..., j, :]`, without dividing.

    ``xp`` is the array module `msgs` belongs to; it defaults to numpy. cupy
    implements `__array_function__`, so `np.ones_like` on a device array would
    dispatch correctly anyway -- but relying on that makes the device path work
    by accident rather than by construction, and it is the kind of thing that
    breaks silently on a cupy upgrade. Pass it explicitly.
    """
    if xp is None:
        xp = np
    b = msgs.shape[-2]
    prefix = xp.ones_like(msgs)
    suffix = xp.ones_like(msgs)
    for i in range(1, b):
        prefix[..., i, :] = prefix[..., i - 1, :] * msgs[..., i - 1, :]
    for i in range(b - 2, -1, -1):
        suffix[..., i, :] = suffix[..., i + 1, :] * msgs[..., i + 1, :]
    return prefix * suffix


# ----------------------------------------------------------------------------
# EM on a tree: the same Xi, one level up in graph complexity
# ----------------------------------------------------------------------------

def tree_e_step(
    grid: np.ndarray,
    weights: np.ndarray,
    log_k: np.ndarray,
    log_root: np.ndarray,
    x: np.ndarray,
    alpha: float,
    delta: float,
    branching: int,
    depth: int,
):
    """Exact BP E-step on the tree, returning `em.ExpectedStatistics`.

    The argument of `src/em.py` transfers verbatim: the posterior of a tree
    under sitewise likelihood factors is a tree, so BP is exact, and every
    expectation the M-step needs is a sum of pairwise edge beliefs. The E-step
    therefore still compresses to one `M x M` matrix

        Xi[k, j] = sum over edges of the posterior mass on parent u_j -> child u_k,

    which sums to the number of *tree* edges rather than chain edges. Because
    `Xi` is where the graph topology stops being visible, every kernel in
    `src/kernels.py` consumes this without modification -- a chain-trained
    M-step and a tree-trained M-step are the same code.

    The one genuinely new piece of bookkeeping is the evidence. Messages are
    renormalized at every node for numerical range, so `log p(x)` is recovered
    by accumulating the discarded log-scales; that is what makes the monotone
    check available on trees too, and it is verified against the closed-form
    Gaussian evidence in the tests.
    """
    from .em import ExpectedStatistics
    from .noising import log_likelihood_matrix

    ti = TreeIndex(depth, branching)
    x = np.atleast_2d(x)
    n_chains, n_leaves = x.shape
    if n_leaves != ti.n_leaves:
        raise ValueError(f"x has {n_leaves} leaves, tree has {ti.n_leaves}")
    m = grid.size
    k_mat = np.exp(log_k)
    root = np.exp(log_root)
    leaves = ti.leaf_nodes()
    n_nodes = ti.n_nodes

    def norm(v):
        s = v.sum(axis=-1, keepdims=True)
        return v / np.maximum(s, 1e-300), np.log(np.maximum(s, 1e-300))

    # -- upward pass -------------------------------------------------------
    bu = np.ones((n_chains, n_nodes, m))    # belief-up: evidence * children msgs
    up = np.ones((n_chains, n_nodes, m))    # message node -> parent
    log_scale = np.zeros(n_chains)

    for c in range(n_chains):
        bu[c, leaves] = np.exp(log_likelihood_matrix(grid, x[c], alpha, delta))
    # log_likelihood_matrix subtracts each row's max and drops the Gaussian
    # normalizer; both are constants in `a` and are added back here.
    z = x[:, :, None] - alpha * grid[None, None, :]
    row_max = (-0.5 * z**2 / delta).max(axis=2)
    log_scale += row_max.sum(axis=1) - 0.5 * n_leaves * np.log(2 * np.pi * delta)

    for d in range(depth, 0, -1):
        nodes = ti.nodes_at(d)
        msg = np.einsum("cnk,kj->cnj", weights * bu[:, nodes], k_mat)
        msg, ls = norm(msg)
        up[:, nodes] = msg
        log_scale += ls.sum(axis=(1, 2))
        parents = ti.nodes_at(d - 1)
        prod = msg.reshape(n_chains, -1, branching, m).prod(axis=2)
        bu[:, parents], ls = norm(bu[:, parents] * prod)
        log_scale += ls.sum(axis=(1, 2))

    log_evidence = float(
        np.sum(log_scale + np.log(np.maximum((weights * bu[:, 0] * root).sum(1), 1e-300)))
    )

    # -- downward pass, accumulating Xi ------------------------------------
    down = np.ones((n_chains, n_nodes, m))
    c_mat = np.zeros((m, m))
    for d in range(depth):
        parents = ti.nodes_at(d)
        kids = ti.nodes_at(d + 1)
        msgs = up[:, kids].reshape(n_chains, -1, branching, m)
        loo = _leave_one_out_product(msgs)
        extra = down[:, parents]
        if d == 0:
            extra = extra * root
        excl, _ = norm(loo * extra[:, :, None, :])
        excl = excl.reshape(n_chains, -1, m)

        # Pairwise edge belief b(u_j, u_k) ~ excl_parent(j) K[k, j] bu_child(k),
        # in the weighted representation that makes sums quadratures.
        f_all = (weights * excl).reshape(-1, m)               # parent side
        g_all = (weights * bu[:, kids]).reshape(-1, m)        # child side
        partition = np.einsum("ek,ek->e", g_all, f_all @ k_mat.T)
        c_mat += (g_all / np.maximum(partition, 1e-300)[:, None]).T @ f_all

        down[:, kids], _ = norm(np.einsum("cnj,kj->cnk", weights * excl, k_mat))

    xi = c_mat * k_mat
    site1 = np.zeros(m)   # root density is held fixed; nothing to accumulate
    n_edges = n_chains * (ti.n_nodes - 1)
    return ExpectedStatistics(
        xi=xi, site1=site1, log_evidence=log_evidence,
        n_edges=n_edges, n_chains=n_chains,
    )


def fit_em_tree(
    kernel,
    grid: np.ndarray,
    weights: np.ndarray,
    groups,
    branching: int,
    depth: int,
    log_root: np.ndarray | None = None,
    n_iters: int = 50,
    tol: float = 1e-9,
):
    """EM for a tree prior. `groups` is a list of `(x, alpha, delta)` batches.

    Deliberately a thin wrapper: the E-step differs from the chain case, the
    M-step does not, because both hand the kernel the same `Xi`.
    """
    import time

    from .em import EMTrace, ExpectedStatistics

    if log_root is None:
        log_root = -0.5 * grid**2 - 0.5 * np.log(2.0 * np.pi)

    trace = EMTrace(log_evidence=[], theta=[], seconds=[])
    current = kernel
    prev = -np.inf
    for _ in range(n_iters):
        t0 = time.perf_counter()
        log_k = current.log_transition_matrix(grid)
        total: ExpectedStatistics | None = None
        for x, alpha, delta in groups:
            part = tree_e_step(
                grid, weights, log_k, log_root, x, alpha, delta, branching, depth
            )
            total = part if total is None else total + part
        assert total is not None
        trace.log_evidence.append(total.log_evidence)
        trace.theta.append(np.asarray(current.theta, dtype=float).copy())
        current = current.m_step(total, grid)
        trace.seconds.append(time.perf_counter() - t0)
        if np.isfinite(prev) and abs(total.log_evidence - prev) <= tol * abs(prev):
            break
        prev = total.log_evidence
    return current, trace
