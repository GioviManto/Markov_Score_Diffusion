"""A fitted video model: spatial trees per frame, a temporal chain over roots.

Composition, not a new model class. The spatial half is `WaveletTreeModel`
verbatim -- same subband scales, same per-level kernels, same per-depth Delta --
and this adds exactly two things: a temporal kernel on the roots, and a temporal
treatment of the LL coefficient.

**The LL band is handled in closed form, on purpose.** Every other coefficient
goes through the grid, but LL has a subband scale of about 17 against 0.14 at the
finest detail level, so `Delta_d = Delta_t / s_d^2` would put it far below the
resolution the shared grid can support (see `WaveletTreeModel.resolution_report`
-- this is the same limit, at its most extreme). It is a scalar Gaussian AR(1)
across frames, so a dense F x F solve is exact, needs no grid at all, and
sidesteps the problem entirely rather than papering over it.

**The control.** `rho_time = 0` gives frames that are independent *given* the LL
trajectory, with every other part of the model identical. That isolates the one
thing being tested -- temporal coupling of the spatial trees -- instead of
confounding it with the LL treatment, which is shared.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .bp_grid import make_grid
from .em import ExpectedStatistics
from .noising import alpha_delta
from .video_bp import cut_caterpillar_bp
from .wavelet import images_to_tree, tree_to_images
from .wavelet_model import SubbandScales, per_depth_grid_sizes

_LOG_2PI = float(np.log(2.0 * np.pi))


# ----------------------------------------------------------------------------
# The LL band: exact scalar Gaussian AR(1) across frames
# ----------------------------------------------------------------------------

def _ll_covariance(n_frames: int, var: float, rho: float) -> np.ndarray:
    lag = np.abs(np.subtract.outer(np.arange(n_frames), np.arange(n_frames)))
    return var * rho**lag


def ll_posterior_mean(scaling, mean, var, rho, alpha, delta):
    """E[LL | noisy LL] for the whole sequence, by dense solve. `scaling` is (B, F)."""
    f_len = scaling.shape[1]
    sigma = _ll_covariance(f_len, var, rho)
    obs = alpha**2 * sigma + delta * np.eye(f_len)
    centred = scaling - alpha * mean
    return mean + alpha * (sigma @ np.linalg.solve(obs, centred.T)).T


def ll_log_likelihood(scaling, mean, var, rho, alpha, delta) -> float:
    f_len = scaling.shape[1]
    sigma = _ll_covariance(f_len, var, rho)
    obs = alpha**2 * sigma + delta * np.eye(f_len)
    centred = scaling - alpha * mean
    sign, logdet = np.linalg.slogdet(obs)
    if sign <= 0:
        return float("-inf")
    quad = np.einsum("bi,ib->b", centred, np.linalg.solve(obs, centred.T))
    return float(np.sum(-0.5 * (quad + logdet + f_len * _LOG_2PI)))


def fit_ll_ar1(scaling: np.ndarray) -> tuple[float, float, float]:
    """(mean, var, rho) of the LL band, by moments across frames."""
    mean = float(scaling.mean())
    var = float(scaling.var())
    a = scaling[:, :-1].ravel() - mean
    b = scaling[:, 1:].ravel() - mean
    rho = float(np.clip((a * b).mean() / max(var, 1e-12), -0.999, 0.999))
    return mean, var, rho


# ----------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class VideoTreeModel:
    qt: object
    scales: SubbandScales
    kernels: list                 # [orientation][level] spatial kernels
    k_time: object                # temporal kernel on the top piece's root
    log_root: np.ndarray          # (3, M_0) prior on the first frame's root
    grids: list                   # one grid per spatial depth, coarsest first
    weights: list
    ll_mean: float
    ll_var: float
    ll_rho: float
    # Defaults last, as a dataclass requires. cut_depth = 0 is the plain
    # caterpillar, in which case k_time_sub is unused.
    k_time_sub: object = None     # temporal kernel on the severed subtree roots
    cut_depth: int = 0            # spatial level severed to buy temporal edges

    @property
    def depth(self) -> int:
        return self.qt.depth

    def _log_k_space(self, oi):
        return [
            k.log_transition_matrix(self.grids[d], self.grids[d + 1])
            for d, k in enumerate(self.kernels[oi])
        ]

    @property
    def grid(self) -> np.ndarray:
        """The root grid -- the one the temporal chain lives on."""
        return self.grids[0]

    def _bp(self, nodes_std, alpha, delta, want_stats=False, chunk=16):
        """Run the (possibly cut) caterpillar per orientation. `nodes_std` is
        (B, F, 3, n). Returns per-level spatial Xi plus the two temporal Xi --
        the severed level's spatial entry is None, its edges having been cut."""
        c = self.cut_depth
        out = np.empty_like(nodes_std)
        log_ev = 0.0
        xi_space = None
        xi_top = None
        xi_sub = None
        k_top = np.exp(self.k_time.log_transition_matrix(self.grids[0]))
        k_sub = (
            np.exp(self.k_time_sub.log_transition_matrix(self.grids[c]))
            if c > 0 else k_top
        )
        # The severed subtrees are rooted at depth c, so their chain's initial
        # law lives on grids[c] -- not on the root grid, which is a different
        # length entirely once the meshes differ by depth.
        log_mu_sub = -0.5 * self.grids[c] ** 2 - 0.5 * _LOG_2PI

        for oi in range(3):
            means, ev, xs, xt, xb = cut_caterpillar_bp(
                self.grids, self.weights, self._log_k_space(oi), k_top, k_sub,
                self.log_root[oi], log_mu_sub,
                nodes_std[:, :, oi, :], alpha,
                self.scales.delta_by_depth(oi, delta), self.qt.branching,
                self.depth, c, chunk=chunk, want_stats=want_stats,
            )
            out[:, :, oi, :] = means
            log_ev += ev
            if want_stats:
                if xi_space is None:
                    xi_space = [None if x is None else x.copy() for x in xs]
                    xi_top = xt.copy()
                    xi_sub = None if xb is None else xb.copy()
                else:
                    for d, x in enumerate(xs):
                        if x is not None:
                            xi_space[d] += x
                    xi_top += xt
                    if xb is not None:
                        xi_sub += xb
        return out, log_ev, xi_space, (xi_top, xi_sub)

    def _to_std_nodes(self, videos):
        b, f_len = videos.shape[:2]
        flat = videos.reshape(b * f_len, *videos.shape[2:])
        _, nodes, scaling = images_to_tree(flat, self.qt.levels)
        std = self.scales.standardise(self.qt, nodes)
        n = std.shape[-1]
        return (std.reshape(b, f_len, 3, n), scaling.reshape(b, f_len), b, f_len)

    def denoise_videos(self, noisy: np.ndarray, t: float, chunk: int = 16):
        alpha, delta = alpha_delta(t)
        std, scaling, b, f_len = self._to_std_nodes(noisy)
        post, _, _, _ = self._bp(std, alpha, delta, chunk=chunk)
        post = self.scales.restore(
            self.qt, post.reshape(b * f_len, 3, post.shape[-1])
        )
        ll = ll_posterior_mean(
            scaling, self.ll_mean, self.ll_var, self.ll_rho, alpha, delta
        )
        imgs = tree_to_images(self.qt, post, ll.reshape(b * f_len, 1))
        return imgs.reshape(noisy.shape)

    def resolution_report(self, t: float, points_per_std: float = 3.0) -> dict:
        """Same grid-resolution limit as the image model, and it applies here too.

        The LL band escapes it -- that one is solved in closed form, no grid --
        but every tree coefficient is subject to it, and the coarsest detail
        subbands of a natural frame have the largest scales, hence the narrowest
        likelihoods. A video likelihood quoted below `min_resolved_t` is being
        integrated against a mesh that cannot see it. See
        `WaveletTreeModel.resolution_report` for the derivation.
        """
        alpha, delta = alpha_delta(t)
        dx = np.array([float(g[1] - g[0]) for g in self.grids])
        pps = np.sqrt(delta) / (alpha * self.scales.scales) / dx[None, :]
        need = float(np.max(points_per_std * dx[None, :] * self.scales.scales))
        return {
            "t": float(t),
            "min_points_per_std": float(pps.min()),
            "resolved": bool(pps.min() >= points_per_std),
            "min_resolved_t": float(0.5 * np.log(need**2 + 1.0)),
        }

    def log_likelihood_videos(self, videos: np.ndarray, t: float, chunk: int = 16) -> float:
        """Exact log p_t of whole sequences, in pixel coordinates.

        Same two exact corrections as the image model -- the diagonal
        standardisation contributes -sum log s_v per frame, the orthonormal
        transform contributes nothing -- plus the LL sequence likelihood.
        """
        alpha, delta = alpha_delta(t)
        std, scaling, b, f_len = self._to_std_nodes(videos)
        _, log_ev, _, _ = self._bp(std, alpha, delta, chunk=chunk)
        log_jac = -float(np.sum(np.log(self.scales.per_node(self.qt)))) * b * f_len
        ll = ll_log_likelihood(
            scaling, self.ll_mean, self.ll_var, self.ll_rho, alpha, delta
        )
        return log_ev + log_jac + ll

    # -- generation --------------------------------------------------------

    def sample_ancestral(self, n: int, n_frames: int, rng, jitter: bool = True):
        """Sample whole sequences: temporal chain of roots, then each frame's tree."""
        from .wavelet_model import _sample_columns
        from .hierarchy import TreeIndex

        ti = TreeIndex(self.depth, self.qt.branching)
        nodes = np.empty((n, n_frames, 3, ti.n_nodes))
        dx = [float(g[1] - g[0]) for g in self.grids]
        grid, w = self.grids[0], self.weights[0]
        k_time = np.exp(self.k_time.log_transition_matrix(grid)) * w[:, None]
        cdf_t = np.cumsum(k_time, axis=0)
        cdf_t /= cdf_t[-1][None, :]

        for oi in range(3):
            root_p = np.exp(self.log_root[oi] - self.log_root[oi].max()) * w
            root_p /= root_p.sum()
            state = np.empty((n, n_frames, ti.n_nodes), dtype=np.intp)
            # Temporal backbone first: the top piece's root is a Markov chain
            # in time.
            state[:, 0, 0] = rng.choice(len(grid), size=n, p=root_p)
            for f in range(1, n_frames):
                state[:, f, 0] = _sample_columns(cdf_t, state[:, f - 1, 0], rng)
            cdfs = []
            for d in range(self.depth):
                k = np.exp(
                    self.kernels[oi][d].log_transition_matrix(
                        self.grids[d], self.grids[d + 1]
                    )
                ) * self.weights[d + 1][:, None]
                c = np.cumsum(k, axis=0)
                cdfs.append(c / c[-1][None, :])

            c_cut = self.cut_depth
            flat = state.reshape(n * n_frames, ti.n_nodes)

            def descend(lo, hi):
                """Fill depths lo+1 .. hi from their parents, in order."""
                for d in range(lo, hi):
                    parents = ti.nodes_at(d)
                    kids = ti.nodes_at(d + 1)
                    ps = np.repeat(flat[:, parents], ti.branching, axis=1)
                    flat[:, kids] = _sample_columns(cdfs[d], ps, rng)

            # Above the cut only. Descending past it here would read the
            # depth-`c_cut` states before their temporal chains have drawn them,
            # i.e. use uninitialised indices as parents.
            descend(0, self.depth if c_cut == 0 else c_cut - 1)
            state3 = flat.reshape(n, n_frames, ti.n_nodes)

            if c_cut > 0:
                # Each depth-cut node roots an independent temporal chain.
                g_sub, w_sub = self.grids[c_cut], self.weights[c_cut]
                k_sub = np.exp(
                    self.k_time_sub.log_transition_matrix(g_sub)
                ) * w_sub[:, None]
                cdf_sub = np.cumsum(k_sub, axis=0)
                cdf_sub /= cdf_sub[-1][None, :]
                sub_roots = ti.nodes_at(c_cut)
                p_sub = np.exp(-0.5 * g_sub**2) * w_sub
                p_sub /= p_sub.sum()
                st = np.empty((n, n_frames, len(sub_roots)), dtype=np.intp)
                st[:, 0] = rng.choice(len(g_sub), size=(n, len(sub_roots)), p=p_sub)
                for f in range(1, n_frames):
                    st[:, f] = _sample_columns(cdf_sub, st[:, f - 1], rng)
                state3[:, :, sub_roots] = st
                # Now push each subtree down from its freshly drawn root.
                flat = state3.reshape(n * n_frames, ti.n_nodes)
                descend(c_cut, self.depth)
                state3 = flat.reshape(n, n_frames, ti.n_nodes)
            vals = np.empty((n, n_frames, ti.n_nodes))
            for d in range(self.depth + 1):
                idx = ti.nodes_at(d)
                v = self.grids[d][state3[:, :, idx]]
                if jitter:
                    v = v + (rng.random(v.shape) - 0.5) * dx[d]
                vals[:, :, idx] = v
            nodes[:, :, oi, :] = vals

        flat_nodes = self.scales.restore(
            self.qt, nodes.reshape(n * n_frames, 3, ti.n_nodes)
        )
        ll = self._sample_ll(n, n_frames, rng)
        imgs = tree_to_images(self.qt, flat_nodes, ll.reshape(n * n_frames, 1))
        side = self.qt.side
        return imgs.reshape(n, n_frames, side, side)

    def _sample_ll(self, n: int, n_frames: int, rng) -> np.ndarray:
        sigma = _ll_covariance(n_frames, self.ll_var, self.ll_rho)
        chol = np.linalg.cholesky(sigma + 1e-12 * np.eye(n_frames))
        return self.ll_mean + rng.standard_normal((n, n_frames)) @ chol.T


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------

@dataclass
class VideoEMTrace:
    log_evidence: list
    seconds: list

    @property
    def monotone_violation(self) -> float:
        if len(self.log_evidence) < 2:
            return 0.0
        return float(max(0.0, -np.diff(self.log_evidence).min()))


def fit_video_tree(
    videos: np.ndarray,
    levels: int,
    t_train,
    kernel_factory,
    time_kernel_factory,
    n_iters: int = 12,
    half_width: float = 8.0,
    grid_size: int | None = None,
    t_resolve: float | None = None,
    freeze_time: bool = False,
    cut_depth: int = 0,
    chunk: int = 16,
    verbose: bool = False,
):
    """Generalised EM for the caterpillar: spatial kernels and temporal kernels.

    `freeze_time=True` keeps the temporal kernels at their initial values, which
    is how the no-temporal-coupling control is built: initialise at rho = 0 and
    they stay there, with every other part of the fit identical.

    `cut_depth = c > 0` severs the spatial edges into depth c, giving each of the
    resulting `1 + 4^c` components per orientation its own temporal edge -- the
    space-for-time trade described in `video_bp.cut_caterpillar_bp`. Level c-1
    then has no spatial statistics, and its kernel is left at its initial value
    because the model no longer contains those edges.
    """
    from .utils import rng_for

    rng = rng_for("video-em", levels, tuple(t_train))
    b, f_len = videos.shape[:2]
    flat = videos.reshape(b * f_len, *videos.shape[2:])
    qt, clean_nodes, clean_scaling = images_to_tree(flat, levels)
    scales = SubbandScales.fit(qt, clean_nodes)
    std_clean = scales.standardise(qt, clean_nodes)
    n_nodes = std_clean.shape[-1]
    std_clean = std_clean.reshape(b, f_len, 3, n_nodes)

    if grid_size is not None and t_resolve is not None:
        raise ValueError("give grid_size or t_resolve, not both")
    if grid_size is not None:
        sizes = [grid_size] * (qt.depth + 1)
    else:
        target = t_resolve if t_resolve is not None else min(t_train)
        sizes = per_depth_grid_sizes(scales.scales, target, half_width)
    grids, weights = [], []
    for size in sizes:
        g, w = make_grid(half_width, size)
        grids.append(g)
        weights.append(w)
    m_root = sizes[0]
    log_root = np.tile(-0.5 * grids[0] ** 2 - 0.5 * _LOG_2PI, (3, 1))
    if verbose:
        print(f"  grid sizes by depth: {sizes}")
    ll_mean, ll_var, ll_rho = fit_ll_ar1(clean_scaling.reshape(b, f_len))

    kern = [[kernel_factory(d, rng) for d in range(qt.depth)] for _ in range(3)]
    k_time = time_kernel_factory(rng)
    k_time_sub = time_kernel_factory(rng) if cut_depth > 0 else None
    m_sub = sizes[cut_depth]

    per_node = scales.per_node(qt)[None, None, :, :]
    obs = []
    for t in t_train:
        alpha, delta = alpha_delta(t)
        noise = rng.standard_normal(std_clean.shape)
        obs.append((alpha * std_clean + np.sqrt(delta) * noise / per_node, alpha, delta))

    trace = VideoEMTrace([], [])
    for it in range(n_iters):
        t0 = time.perf_counter()
        model = VideoTreeModel(
            qt=qt, scales=scales, kernels=kern, k_time=k_time, log_root=log_root,
            grids=grids, weights=weights,
            ll_mean=ll_mean, ll_var=ll_var, ll_rho=ll_rho,
            k_time_sub=k_time_sub, cut_depth=cut_depth,
        )
        tot_space = [None] * qt.depth
        # Each temporal edge joins two nodes at the same depth, so its Xi is
        # square -- on the root grid for the top piece, on grid[cut] for the
        # severed subtrees.
        tot_time = np.zeros((m_root, m_root))
        tot_sub = np.zeros((m_sub, m_sub)) if cut_depth > 0 else None
        log_ev = 0.0
        for x_std, alpha, delta in obs:
            _, ev, xi_s, (xi_t, xi_b) = model._bp(
                x_std, alpha, delta, want_stats=True, chunk=chunk
            )
            log_ev += ev
            tot_time += xi_t
            if tot_sub is not None:
                tot_sub += xi_b
            for d, x in enumerate(xi_s):
                if x is None:
                    continue
                tot_space[d] = x if tot_space[d] is None else tot_space[d] + x

        trace.log_evidence.append(log_ev)
        n_edge_time = b * (f_len - 1) * 3 * len(obs)
        for d in range(qt.depth):
            if tot_space[d] is None:
                continue          # the severed level: no edges, nothing to fit
            stats = ExpectedStatistics(
                xi=tot_space[d], site1=np.zeros(sizes[d]), log_evidence=0.0,
                n_edges=int(tot_space[d].sum()), n_chains=b,
            )
            new = kern[0][d].m_step(stats, grids[d], grids[d + 1])
            for oi in range(3):
                kern[oi][d] = new
        if not freeze_time:
            k_time = k_time.m_step(
                ExpectedStatistics(
                    xi=tot_time, site1=np.zeros(m_root), log_evidence=0.0,
                    n_edges=n_edge_time, n_chains=b,
                ),
                grids[0],
            )
            if tot_sub is not None:
                k_time_sub = k_time_sub.m_step(
                    ExpectedStatistics(
                        xi=tot_sub, site1=np.zeros(m_sub), log_evidence=0.0,
                        n_edges=int(tot_sub.sum()), n_chains=b,
                    ),
                    grids[cut_depth],
                )
        trace.seconds.append(time.perf_counter() - t0)
        if verbose:
            print(f"  video EM {it + 1:2d}/{n_iters}  log-ev {log_ev:.6e}  "
                  f"rho_time {getattr(k_time, 'rho', float('nan')):.4f}  "
                  f"{trace.seconds[-1]:.1f}s")

    model = VideoTreeModel(
        qt=qt, scales=scales, kernels=kern, k_time=k_time, log_root=log_root,
        grids=grids, weights=weights,
        ll_mean=ll_mean, ll_var=ll_var, ll_rho=ll_rho,
        k_time_sub=k_time_sub, cut_depth=cut_depth,
    )
    return model, trace
