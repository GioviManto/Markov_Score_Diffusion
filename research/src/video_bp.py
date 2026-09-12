"""Exact BP on video: a temporal chain of spatial trees.

The structural question, stated precisely
-----------------------------------------
Video has two dependence structures, and the project already computes exactly on
each of them separately: a **spatial tree** per frame (`src/wavelet_bp.py`) and a
**temporal chain** across frames (everything else in the package). The tempting
claim is that the combination is therefore also exact. It is not, and the reason
is worth being exact about.

Take the *fully coupled* model: coefficient v in frame f depends on its scale
parent `p(v)` in frame f, and on itself in frame f-1. Then

    v_f -- p(v)_f -- p(v)_{f-1} -- v_{f-1} -- v_f

is a 4-cycle. The graph has loops, BP is loopy, and the exactness claim is gone.
Coupling every coefficient in both directions is exactly what one would want for
video, and it is exactly what cannot be done here.

What *is* loop-free is a spanning structure. The natural one couples the frames
only through the **root** of each frame's tree:

    root_1 --- root_2 --- root_3 --- ...        (temporal chain)
      |          |          |
    quadtree   quadtree   quadtree              (spatial, per frame)

Each root has one pendant subtree and at most two chain neighbours; the subtrees
are disjoint; there is no cycle anywhere. This is a "caterpillar": a chain
backbone with trees hanging off it. BP on it is exact, and the model it
corresponds to is a real one -- frames are temporally coupled at the coarsest
scale, and fine detail is temporally independent *given* the coarse trajectory.

That restriction is the price, and it should be stated rather than glossed:
this model cannot express a moving edge whose fine-scale detail persists
independently of the coarse content. It is the maximal tree-structured
spatio-temporal model of this shape, not the model one would write down if
exactness were not required.

How it is computed
------------------
Not with new inference code. BP on a caterpillar factorises into the two passes
the package already has:

1. run each frame's quadtree **upward**; the root's belief-up is then a single
   message summarising that entire frame;
2. run an exact **chain** BP over the frames with those messages as unary
   potentials;
3. feed each root's chain context back in as its incoming message and run each
   frame's quadtree **downward**.

Step 2 needs a chain BP over *arbitrary* unary potentials rather than Gaussian
likelihoods built from observations, which is the one genuinely new routine here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .wavelet_bp import as_grid_list, wavelet_tree_bp


def spatiotemporal_has_cycle(
    n_per_frame: int,
    n_frames: int,
    spatial_edges,
    coupled_nodes,
) -> bool:
    """Is the spatio-temporal graph loopy, given which nodes carry temporal edges?

    `spatial_edges` are within-frame edges, repeated identically in every frame;
    `coupled_nodes` are the per-frame indices joined to their counterparts in the
    next frame. Union-find, so this is a decision procedure rather than an
    argument -- which matters, because the ceiling on video coherence rests on
    it.

    The result it establishes: **at most one temporal edge per connected
    component**. If u and v lie in the same component and both are coupled, the
    within-frame path u -> v exists in both frames, and together with the two
    temporal edges that closes u_f -> v_f -> v_{f-1} -> u_{f-1} -> u_f. So
    coupling a second node in a component is never loop-free, whatever the
    choice.
    """
    total = n_per_frame * n_frames
    parent = list(range(total))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = []
    for f in range(n_frames):
        edges += [(a + f * n_per_frame, b + f * n_per_frame) for a, b in spatial_edges]
    for f in range(n_frames - 1):
        for v in coupled_nodes:
            edges.append((v + f * n_per_frame, v + (f + 1) * n_per_frame))

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            return True
        parent[ra] = rb
    return False


@dataclass(frozen=True)
class ChainBPResult:
    """Forward-backward on a chain with supplied unary potentials.

    context : (B, F, M) product of the messages from the left and right
              neighbours, *excluding* the site's own potential -- which is
              exactly what a site must pass on to a subtree hanging below it.
    log_z   : (B,) log normalisation of the chain, given the potentials as
              supplied (so the caller's own normalisation must be added back).
    xi      : (M, M) expected transition mass on the temporal edges, summed over
              every edge of every sequence, or None if not requested. Same
              `Xi[k, j]` convention as everywhere else, so the temporal kernel's
              M-step is the existing M-step with no changes.
    """

    context: np.ndarray
    log_z: np.ndarray
    xi: np.ndarray | None = None


def chain_bp_potentials(
    pot: np.ndarray,
    k_time: np.ndarray,
    weights: np.ndarray,
    mu: np.ndarray,
    want_stats: bool = False,
) -> ChainBPResult:
    """Exact forward-backward on a chain of arbitrary unary potentials.

    `pot` is (B, F, M), `k_time[k, j] = K(a_{f+1} = u_k | a_f = u_j)` -- the same
    `K[out, in]` convention as everywhere else -- and `mu` is the (M,) initial
    law. Messages are renormalised at each step and the discarded scales are
    accumulated, so `log_z` is usable at any chain length.
    """
    pot = np.asarray(pot, dtype=float)
    b, f_len, m = pot.shape
    if k_time.shape != (m, m):
        raise ValueError(f"k_time must be ({m}, {m}), got {k_time.shape}")

    def norm(v):
        s = v.sum(axis=-1, keepdims=True)
        return v / np.maximum(s, 1e-300), np.log(np.maximum(s, 1e-300))[..., 0]

    fwd = np.empty((b, f_len, m))
    fwd[:, 0] = mu[None, :]
    for i in range(1, f_len):
        src = weights[None, :] * pot[:, i - 1] * fwd[:, i - 1]
        fwd[:, i], _ = norm(src @ k_time.T)

    # Only the *backward* scales enter the evidence. The normalisation is
    # closed by contracting at site 0, where the forward message is `mu` itself
    # and carries no accumulated scale; adding the forward scales as well would
    # count the same normalisation twice, which is a mistake that leaves the
    # posterior means untouched and so survives every check but this one.
    log_z = np.zeros(b)
    bwd = np.empty((b, f_len, m))
    bwd[:, -1] = 1.0
    for i in range(f_len - 2, -1, -1):
        src = weights[None, :] * pot[:, i + 1] * bwd[:, i + 1]
        msg, ls = norm(src @ k_time)
        bwd[:, i] = msg
        log_z += ls

    context = fwd * bwd
    total = (weights[None, :] * pot[:, 0] * context[:, 0]).sum(axis=1)
    log_z = log_z + np.log(np.maximum(total, 1e-300))

    xi = None
    if want_stats:
        # Pairwise belief on each temporal edge:
        #   b(a_f = u_j, a_{f+1} = u_k)  ~  K[k, j] left_f[j] right_{f+1}[k],
        # with `left` carrying everything known at f from the left and its own
        # potential, and `right` the same from the right. Normalised per edge, so
        # Xi sums to the number of temporal edges -- the property every M-step in
        # `src/kernels.py` relies on.
        xi = np.zeros((m, m))
        for i in range(f_len - 1):
            left = weights[None, :] * pot[:, i] * fwd[:, i]
            right = weights[None, :] * pot[:, i + 1] * bwd[:, i + 1]
            partition = np.einsum("bk,bk->b", right, left @ k_time.T)
            c_mat = (right / np.maximum(partition, 1e-300)[:, None]).T @ left
            xi += c_mat * k_time

    return ChainBPResult(context, log_z, xi)


@dataclass(frozen=True)
class VideoBPResult:
    posterior_mean: np.ndarray                  # (B, F, n_nodes)
    log_evidence: float
    xi_space: list[np.ndarray] | None = None    # per spatial level
    xi_time: np.ndarray | None = None           # temporal edges


def caterpillar_bp(
    grid: np.ndarray,
    weights: np.ndarray,
    log_k_space: list[np.ndarray],
    k_time: np.ndarray,
    log_mu: np.ndarray,
    x: np.ndarray,
    alpha: float,
    delta_by_depth,
    branching: int,
    depth: int,
    chunk: int = 32,
    want_stats: bool = False,
    xp=None,
) -> VideoBPResult:
    """Exact BP on a temporal chain of spatial quadtrees.

    `x` is (B, F, n_nodes) standardised coefficients: B videos, F frames, one
    quadtree per frame in breadth-first order.

    Each frame's tree is traversed twice -- once to obtain its root message, once
    to push the chain context back down. That is a factor of two against a
    single-pass implementation and is deliberate: it reuses `wavelet_tree_bp`
    unchanged rather than duplicating the upward and downward passes, and a
    duplicated BP implementation that silently drifts from the tested one is a
    worse problem than a factor of two.

    `xp` is forwarded to the two spatial passes, which are where the cost is:
    they run B x F trees against the chain's B x F root-grid messages. The
    temporal chain itself stays on the host -- it is one M-wide recursion over F
    frames, and moving it would cost more in transfers than it saves.
    """
    x = np.asarray(x, dtype=float)
    b, f_len, n_nodes = x.shape
    grids = as_grid_list(grid, depth)
    wts = as_grid_list(weights, depth)
    # Everything temporal happens at the root, so the chain's grid is grids[0]
    # whatever the deeper levels use.
    m = grids[0].size
    flat = x.reshape(b * f_len, n_nodes)

    # -- 1. upward pass per frame -----------------------------------------
    # The root prior is *not* applied here: the chain supplies what the root
    # knows from outside, and applying a prior as well would double-count it.
    uniform = np.zeros(m)
    up = wavelet_tree_bp(
        grids, wts, log_k_space, uniform, flat, alpha, delta_by_depth,
        branching, depth, chunk=chunk, root_message=np.ones((b * f_len, m)),
        xp=xp,
    )
    pot = up.root_belief_up.reshape(b, f_len, m)
    tree_scale = up.log_scale.reshape(b, f_len).sum(axis=1)

    # -- 2. exact chain over the frames -----------------------------------
    chain = chain_bp_potentials(
        pot, k_time, wts[0], np.exp(log_mu), want_stats=want_stats
    )

    # -- 3. downward pass per frame, with the chain context as the root message
    # Once the root holds its correct incoming message, every belief the downward
    # pass computes is the full-model posterior, not the per-frame one. So the
    # spatial Xi taken from *this* call is the caterpillar's, which is the whole
    # reason the statistics are collected here and not in step 1.
    context = chain.context.reshape(b * f_len, m)
    down = wavelet_tree_bp(
        grids, wts, log_k_space, uniform, flat, alpha, delta_by_depth,
        branching, depth, chunk=chunk, root_message=context,
        want_stats=want_stats, xp=xp,
    )

    log_evidence = float(np.sum(tree_scale + chain.log_z))
    return VideoBPResult(
        down.posterior_mean.reshape(b, f_len, n_nodes), log_evidence,
        down.xi_by_level, chain.xi,
    )


# ----------------------------------------------------------------------------
# Trading spatial edges for temporal ones
# ----------------------------------------------------------------------------

def _subtree_indices(branching: int, depth: int, cut: int) -> np.ndarray:
    """(n_subtrees, n_nodes_per_subtree) global node indices, subtree-BF order.

    Cutting below depth `cut` severs the edges from depth cut-1 into depth cut,
    leaving the top piece (depths 0..cut-1) and one subtree under each depth-cut
    node. In breadth-first numbering the descendants of a node are *contiguous*
    at every level, which is what makes the extraction a slice rather than a
    search: the level-e nodes of subtree j sit at
    `offset(cut + e) + j * b^e + [0, b^e)`.
    """
    from .hierarchy import level_offset

    if not 0 <= cut <= depth:
        raise ValueError(f"cut must be in [0, {depth}], got {cut}")
    n_sub_trees = branching**cut
    sub_depth = depth - cut
    cols = []
    for e in range(sub_depth + 1):
        width = branching**e
        base = level_offset(cut + e, branching)
        cols.append(
            base + np.arange(n_sub_trees)[:, None] * width + np.arange(width)[None, :]
        )
    return np.concatenate(cols, axis=1)


def cut_caterpillar_bp(
    grids,
    weights,
    log_k_space: list[np.ndarray],
    k_time_top: np.ndarray,
    k_time_sub: np.ndarray,
    log_mu_top: np.ndarray,
    log_mu_sub: np.ndarray,
    x: np.ndarray,
    alpha: float,
    delta_by_depth,
    branching: int,
    depth: int,
    cut: int,
    chunk: int = 32,
    want_stats: bool = False,
):
    """Exact BP after cutting the spatial tree below depth `cut`.

    Why this exists. §9.2 of the write-up establishes that at most **one**
    temporal edge per connected component is loop-free, so the caterpillar --
    with one component per orientation -- can couple only the root. The way to
    couple more is therefore not to add temporal edges to the same tree, which
    always closes a cycle, but to *cut* the tree into more components and give
    each its own. A spanning forest on F frames of n coefficients with c
    components has `Fn - c` edges, `F(n-c)` spatial and `c(F-1)` temporal, so
    the two are traded one for one against a fixed budget.

    Cutting below depth `cut` leaves `1 + b^cut` components per orientation: the
    top piece, plus one subtree under every depth-`cut` node. Each is itself a
    caterpillar, so this is composition rather than new inference -- and every
    point on the trade-off remains **exact**, which is the whole reason the
    curve is worth measuring instead of argued about.

    `cut = 0` is the plain caterpillar. `cut = depth` couples every leaf and
    keeps no spatial edge below the top piece.

    Returns `(posterior_mean, log_evidence, xi_space, xi_time_top, xi_time_sub)`
    with `xi_space[cut - 1] = None`, that level's edges having been severed.
    """
    x = np.asarray(x, dtype=float)
    b, f_len, n_nodes = x.shape
    grids = as_grid_list(grids, depth)
    wts = as_grid_list(weights, depth)
    deltas = np.asarray(delta_by_depth, dtype=float)

    if cut == 0:
        res = caterpillar_bp(
            grids, wts, log_k_space, k_time_top, log_mu_top, x, alpha, deltas,
            branching, depth, chunk=chunk, want_stats=want_stats,
        )
        return res.posterior_mean, res.log_evidence, res.xi_space, res.xi_time, None
    if not 1 <= cut <= depth:
        raise ValueError(f"cut must be in [0, {depth}], got {cut}")

    from .hierarchy import level_offset

    n_top = level_offset(cut, branching)
    means = np.empty_like(x)
    log_ev = 0.0

    # -- the top piece: depths 0 .. cut-1, chained at its own root ----------
    top = caterpillar_bp(
        grids[:cut], wts[:cut], log_k_space[: cut - 1], k_time_top, log_mu_top,
        x[:, :, :n_top], alpha, deltas[:cut], branching, cut - 1,
        chunk=chunk, want_stats=want_stats,
    )
    means[:, :, :n_top] = top.posterior_mean
    log_ev += top.log_evidence

    # -- the subtrees: structurally identical, so batch them as extra videos
    idx = _subtree_indices(branching, depth, cut)          # (n_sub, n_per_sub)
    n_sub, n_per = idx.shape
    x_sub = x[:, :, idx]                                   # (B, F, n_sub, n_per)
    x_sub = x_sub.transpose(0, 2, 1, 3).reshape(b * n_sub, f_len, n_per)
    sub = caterpillar_bp(
        grids[cut:], wts[cut:], log_k_space[cut:], k_time_sub, log_mu_sub,
        x_sub, alpha, deltas[cut:], branching, depth - cut,
        chunk=chunk, want_stats=want_stats,
    )
    post = sub.posterior_mean.reshape(b, n_sub, f_len, n_per).transpose(0, 2, 1, 3)
    means[:, :, idx] = post
    log_ev += sub.log_evidence

    xi_space = None
    if want_stats:
        xi_space = [None] * depth
        for d in range(cut - 1):
            xi_space[d] = top.xi_space[d]
        for e in range(depth - cut):
            xi_space[cut + e] = sub.xi_space[e]

    return means, log_ev, xi_space, top.xi_time, sub.xi_time
