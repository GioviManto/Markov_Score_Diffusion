"""Exact BP on a wavelet quadtree, where *every* node is observed.

Four things separate this from `hierarchy.tree_bp_grid`, and each is forced by
the data rather than chosen:

1. **Evidence at every node, not only at the leaves.** In `hierarchy` the
   internal nodes are latent and only the leaves are observed. A wavelet
   decomposition is invertible: every coefficient, at every scale, is part of
   the data, and the forward process noises all of them. So every node carries a
   likelihood factor. This changes nothing structurally -- one unary factor per
   site is exactly the situation under which the posterior factor graph equals
   the prior graph -- so BP is still exact. It only changes where `ell` is
   applied, and it means the downward message must exclude the child's upward
   message while *including* the parent's own evidence.

2. **A per-scale noise level.** Subband coefficient variances in natural images
   span orders of magnitude across scales, and one grid cannot resolve the
   finest while covering the coarsest. Each subband is therefore standardised by
   its own training scale s_d. That is a diagonal reparametrisation, so the
   likelihood stays an exact per-site Gaussian: from

       x_v = alpha a_v + sqrt(Delta) z_v,   xt_v = x_v / s_d,   at_v = a_v / s_d,

   one gets xt_v = alpha at_v + sqrt(Delta / s_d^2) z_v, i.e. the same alpha and a
   per-depth Delta_d = Delta / s_d^2. Nothing is approximated; the noise level is
   simply not the same number at every scale once the coordinates are rescaled.

3. **A per-depth *grid*.** This is what (2) forces, and getting it wrong is what
   confined an earlier version of this package to t >= 0.9. Because
   Delta_d = Delta_t / s_d^2, a subband with a *large* scale gets a *narrow*
   likelihood -- and in a natural image the largest scales are the coarsest
   subbands (LH depth 0 has scale 10.7 against 0.14 at HH depth 4, a factor of
   76). On a shared mesh those are the ones that go under-resolved, at small t,
   which is exactly where a reverse sampler spends its last steps.

   The fix is to spend grid points where the likelihood is narrow. It is close
   to free, because the coarse subbands are the ones with almost no nodes: level
   d holds 4^d nodes and wants M_d proportional to s_d, and since s_d roughly
   halves per level the product 4^d * M_d * M_{d+1} is nearly constant. Resolving
   down to t = 0.05 costs *less* than the old uniform M = 241 that only reached
   t = 0.9.

   `grids` and `weights` may therefore be lists, one entry per depth. Passing a
   single array keeps the old uniform behaviour, which is what the chain-derived
   tests still exercise.

4. **One kernel per scale.** Wavelet coefficients are not scale-stationary, so a
   single shared transition kernel is misspecified by construction. `log_k[d]`
   governs the edge from depth d to depth d+1 and is **(M_{d+1}, M_d)** -- child
   values down the rows, parent values across the columns. Every kernel in
   `src/kernels.py` accepts the two grids separately, so the per-level M-step is
   the existing M-step called once per level.

Convention, inherited unchanged from `src/bp_grid.py`: `log_k[d][k, j]` is
log K_d(u_k | u_j) with k the *child* value and j the *parent* value, so the
upward message is `K.T @ f` and the downward message is `K @ g`. Note which grid
each message lives on: a node's message *to its parent* is a function of the
parent's value, so it lives on the parent's grid.

Device
------
The recursion is parameterised by its array module, exactly as `src/bp_grid.py`
is, so the quadtree runs on a GPU from the same code path that runs on CPU --
see `src/backend.py` for why there is no second implementation. Until 19 Aug
2026 `backend` was wired into `bp_grid` and `denoiser` only, which meant the 1-D
chain work could use a device and *every image and video experiment could not*:
`BP_DEVICE=gpu` was silently a no-op for exp_23..exp_26.

Unlike `grid_bp_batch`, this returns **numpy** arrays whatever device it ran on.
The M-step, the model wrapper and every experiment downstream are host code, and
the results are small next to the work: posterior means are O(B x n_nodes) and
the per-level Xi are O(M_child x M_parent), against an O(B x n_nodes x M^2)
recursion. Converting at this boundary keeps every caller unchanged rather than
pushing device arrays into code that would then need its own port.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backend import to_device, to_host
from .em import ExpectedStatistics
from .hierarchy import TreeIndex, _leave_one_out_product


@dataclass(frozen=True)
class WaveletBPResult:
    """Posterior summaries on the tree.

    posterior_mean : (B, n_nodes) E[a_v | x], exact up to quadrature.
    log_evidence   : summed over the batch, log p_t(x), all constants included.
    log_evidence_per_item : (B,) the same quantity before summation. Kept because
                     a summed likelihood cannot carry a standard error, so a
                     comparison between two models has no scale to be judged on.
    xi_by_level    : per-level expected transition mass, or None if not asked for.
                     `xi_by_level[d]` is (M_{d+1}, M_d) for the edge d -> d+1.
    root_belief_up : (B, M_0) the root's evidence times all its children's upward
                     messages, *excluding* the root prior. This is the message
                     the whole spatial tree sends to whatever sits above it, and
                     it is what lets a tree be attached to a temporal chain
                     without reimplementing either (see `src/video_bp.py`).
    log_scale      : (B,) the accumulated log-normalisation, excluding the final
                     root contraction. Returned for the same reason: a caller
                     that supplies its own root message has to finish the
                     evidence bookkeeping itself.
    """

    posterior_mean: np.ndarray
    log_evidence: float
    xi_by_level: list[np.ndarray] | None
    root_belief_up: np.ndarray | None = None
    log_scale: np.ndarray | None = None
    log_evidence_per_item: np.ndarray | None = None


def node_delta(ti: TreeIndex, delta_by_depth) -> np.ndarray:
    """(n_nodes,) noise variance for each node, from its depth."""
    d_arr = np.asarray(delta_by_depth, dtype=float)
    return np.concatenate([
        np.full(ti.branching**d, d_arr[d]) for d in range(ti.depth + 1)
    ])


def as_grid_list(grid, depth: int) -> list[np.ndarray]:
    """Accept either one grid for every depth, or one grid per depth."""
    if isinstance(grid, (list, tuple)):
        out = [np.asarray(g, dtype=float) for g in grid]
        if len(out) != depth + 1:
            raise ValueError(f"need {depth + 1} grids, got {len(out)}")
        return out
    g = np.asarray(grid, dtype=float)
    return [g] * (depth + 1)


def wavelet_tree_bp(
    grid,
    weights,
    log_k: list[np.ndarray],
    log_root: np.ndarray,
    x: np.ndarray,
    alpha: float,
    delta_by_depth,
    branching: int,
    depth: int,
    want_stats: bool = False,
    chunk: int = 64,
    root_message: np.ndarray | None = None,
    xp=None,
) -> WaveletBPResult:
    """Exact sum-product on the quadtree with observations at every node.

    `x` is (B, n_nodes) *standardised* coefficients in breadth-first order.
    `grid` and `weights` are either one array each (uniform mesh) or a list of
    `depth + 1` arrays (per-depth mesh). `log_k[d]` is the (M_{d+1}, M_d) kernel
    for the edge from depth d to depth d+1. `delta_by_depth` has length
    `depth + 1`.

    `root_message` is what the root receives from *outside* the tree, shape
    (B, M_0) or (M_0,). It defaults to the root prior `exp(log_root)`, which is
    the standalone case. Supplying something else is how a tree gets attached to
    a larger loop-free graph -- a temporal chain of roots, in `src/video_bp.py`.

    `xp` selects the array module; it defaults to numpy, *not* to `get_xp()`.
    A default of `get_xp()` would let an unrelated `BP_DEVICE` in the environment
    silently change which device a library call runs on, which is the opposite of
    what `backend.py` is for -- the caller chooses, once, and says so. Returned
    arrays are numpy regardless.
    """
    if xp is None:
        xp = np
    ti = TreeIndex(depth, branching)
    x_host = np.atleast_2d(np.asarray(x, dtype=float))
    if x_host.shape[1] != ti.n_nodes:
        raise ValueError(f"x has {x_host.shape[1]} nodes, tree has {ti.n_nodes}")
    if len(log_k) != depth:
        raise ValueError(f"need {depth} kernels, got {len(log_k)}")
    if len(np.asarray(delta_by_depth)) != depth + 1:
        raise ValueError(f"need {depth + 1} deltas, got {len(np.asarray(delta_by_depth))}")

    grids_host = as_grid_list(grid, depth)
    wts_host = as_grid_list(weights, depth)
    for d, lk in enumerate(log_k):
        want = (grids_host[d + 1].size, grids_host[d].size)
        if np.shape(lk) != want:
            raise ValueError(
                f"log_k[{d}] must be {want} (child x parent), got {np.shape(lk)}"
            )

    grids = [to_device(g, xp) for g in grids_host]
    wts = [to_device(w, xp) for w in wts_host]
    k_mats = [xp.exp(to_device(np.asarray(lk, dtype=float), xp)) for lk in log_k]
    root = xp.exp(to_device(np.asarray(log_root, dtype=float), xp))
    deltas = np.asarray(delta_by_depth, dtype=float)
    x_dev = to_device(x_host, xp)

    n_batch = x_host.shape[0]
    m_root = grids_host[0].size
    means = np.empty_like(x_host)
    root_up = np.empty((n_batch, m_root))
    log_scale = np.empty(n_batch)
    log_ev_items = np.empty(n_batch)
    log_evidence = 0.0
    xi_total = (
        [
            np.zeros((grids_host[d + 1].size, grids_host[d].size))
            for d in range(depth)
        ]
        if want_stats else None
    )

    for start in range(0, n_batch, chunk):
        sl = slice(start, start + chunk)
        rm = root_message
        if rm is not None:
            rm = np.asarray(rm, dtype=float)
            rm = to_device(rm[sl] if rm.ndim == 2 else rm, xp)
        part = _bp_chunk(
            ti, grids, wts, k_mats, root, x_dev[sl], alpha, deltas,
            want_stats, rm, xp,
        )
        means[sl] = to_host(part.posterior_mean)
        root_up[sl] = to_host(part.root_belief_up)
        log_scale[sl] = to_host(part.log_scale)
        log_ev_items[sl] = to_host(part.log_evidence_per_item)
        log_evidence += part.log_evidence
        if xi_total is not None and part.xi_by_level is not None:
            for d in range(depth):
                xi_total[d] += to_host(part.xi_by_level[d])

    return WaveletBPResult(
        means, log_evidence, xi_total, root_up, log_scale, log_ev_items,
    )


def _bp_chunk(
    ti: TreeIndex,
    grids: list[np.ndarray],
    wts: list[np.ndarray],
    k_mats: list[np.ndarray],
    root: np.ndarray,
    x: np.ndarray,
    alpha: float,
    deltas: np.ndarray,
    want_stats: bool,
    root_message: np.ndarray | None = None,
    xp=None,
) -> WaveletBPResult:
    if xp is None:
        xp = np
    b = x.shape[0]
    depth, branching = ti.depth, ti.branching

    def norm(v):
        s = v.sum(axis=-1, keepdims=True)
        return v / xp.maximum(s, 1e-300), xp.log(xp.maximum(s, 1e-300))

    # -- evidence at every node, level by level ---------------------------
    # Row-shifted so exp() cannot underflow a whole row; the discarded shift and
    # the Gaussian normaliser are both restored in `log_scale`.
    # A level is a contiguous block in breadth-first order, so address it with a
    # slice rather than the `nodes_at` index array: fancy-indexing a device array
    # with a *host* index array is not portable across array modules, and a slice
    # is a view rather than a gather in either.
    def level_slice(d: int) -> slice:
        nodes = ti.nodes_at(d)
        return slice(int(nodes[0]), int(nodes[-1]) + 1)

    ev: list[np.ndarray] = []
    log_scale = xp.zeros(b)
    for d in range(depth + 1):
        sl_d = level_slice(d)
        n_d = branching**d
        z = x[:, sl_d, None] - alpha * grids[d][None, None, :]
        log_ell = -0.5 * z**2 / deltas[d]
        row_max = log_ell.max(axis=2)
        ev.append(xp.exp(log_ell - row_max[:, :, None]))       # (B, n_d, M_d)
        log_scale += row_max.sum(axis=1)
        log_scale -= 0.5 * n_d * float(np.log(2.0 * np.pi * deltas[d]))

    # -- upward pass -------------------------------------------------------
    # bu[d][v] = evidence(v) * prod over children of their upward messages.
    # up[d] is the message from a depth-d node to its parent, and is therefore a
    # function of the *parent's* value: it lives on grids[d - 1].
    bu = [e.copy() for e in ev]
    up: list[np.ndarray | None] = [None] * (depth + 1)
    for d in range(depth, 0, -1):
        msg, ls = norm(xp.einsum("cnk,kj->cnj", wts[d] * bu[d], k_mats[d - 1]))
        up[d] = msg                                            # (B, n_d, M_{d-1})
        log_scale += ls.sum(axis=(1, 2))
        n_parents = branching ** (d - 1)
        prod = msg.reshape(b, n_parents, branching, grids[d - 1].size).prod(axis=2)
        bu[d - 1], ls = norm(bu[d - 1] * prod)
        log_scale += ls.sum(axis=(1, 2))

    root_belief_up = bu[0][:, 0].copy()
    incoming = root if root_message is None else to_device(root_message, xp)
    # Kept per item as well as summed. The summand is already computed here, and
    # discarding it costs every downstream likelihood comparison its standard
    # error: a total says model A beats model B, a per-item vector says by how
    # much against how much spread, and only the second can be tested. The test
    # images are shared across models, so the paired difference is the estimator
    # with the small variance -- which is the whole reason to keep the vector
    # rather than recompute a total per model.
    log_ev_items = (
        log_scale
        + xp.log(xp.maximum((wts[0] * bu[0][:, 0] * incoming).sum(-1), 1e-300))
    )
    log_evidence = float(xp.sum(log_ev_items))

    # -- downward pass -----------------------------------------------------
    down: list[np.ndarray | None] = [None] * (depth + 1)
    incoming_2d = incoming if incoming.ndim == 2 else incoming[None, :]
    down[0] = xp.broadcast_to(
        incoming_2d[:, None, :], (b, 1, grids[0].size)
    ).copy()
    xi_by_level = (
        [xp.zeros((grids[d + 1].size, grids[d].size)) for d in range(depth)]
        if want_stats else None
    )

    for d in range(depth):
        m_par, m_kid = grids[d].size, grids[d + 1].size
        n_par = branching**d
        # Leave-one-out over siblings by prefix/suffix scan rather than division:
        # sibling messages are legitimately ~1e-16 in the tails, and dividing by
        # them is where a tree BP implementation quietly loses digits. The
        # sibling messages live on the *parent's* grid, which is why this is
        # m_par wide.
        loo = _leave_one_out_product(
            up[d + 1].reshape(b, n_par, branching, m_par), xp=xp
        )
        # What the parent knows *apart from* this child: its own evidence, the
        # message from above, and its other children.
        extra = ev[d] * down[d]
        excl, _ = norm(loo * extra[:, :, None, :])
        excl = excl.reshape(b, n_par * branching, m_par)

        if want_stats:
            f_all = (wts[d] * excl).reshape(-1, m_par)          # parent side
            g_all = (wts[d + 1] * bu[d + 1]).reshape(-1, m_kid)  # child side
            partition = xp.einsum("ek,ek->e", g_all, f_all @ k_mats[d].T)
            c_mat = (g_all / xp.maximum(partition, 1e-300)[:, None]).T @ f_all
            xi_by_level[d] += c_mat * k_mats[d]

        down[d + 1], _ = norm(
            xp.einsum("cnj,kj->cnk", wts[d] * excl, k_mats[d])
        )

    # -- beliefs, reassembled in breadth-first order -----------------------
    means = xp.empty_like(x)
    for d in range(depth + 1):
        belief, _ = norm(bu[d] * down[d])
        means[:, level_slice(d)] = belief @ grids[d]

    return WaveletBPResult(
        means, log_evidence, xi_by_level, root_belief_up, log_scale,
        log_ev_items,
    )


def stats_by_level(
    xi_by_level: list[np.ndarray],
    log_evidence: float,
    n_images: int,
    branching: int,
) -> list[ExpectedStatistics]:
    """Wrap per-level Xi as `ExpectedStatistics`, one per edge level.

    The evidence is attached to level 0 only, so that summing the list does not
    count it `depth` times; the M-step ignores it in any case. `site1` follows
    the *parent* grid, since that is the axis an initial-density estimate would
    live on.
    """
    return [
        ExpectedStatistics(
            xi=xi,
            site1=np.zeros(xi.shape[1]),
            log_evidence=log_evidence if d == 0 else 0.0,
            n_edges=n_images * branching ** (d + 1),
            n_chains=n_images,
        )
        for d, xi in enumerate(xi_by_level)
    ]
