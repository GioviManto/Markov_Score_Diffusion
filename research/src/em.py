"""Tree-structured grid EM for chain-prior parameters, with belief propagation as
the E-step and exact or generalised kernel updates.

The name is deliberate and this module used to be called "Exact EM", which
overstated it in two directions at once. The E-step is exact *structurally* -- the
posterior really is a tree and sum-product really does return its marginals -- but
what runs is a truncated grid quadrature, so the numbers carry a discretisation
error. And the M-step is an exact maximiser only for the Gaussian and Laplace
kernels; the mixture and MDN kernels take conditional-maximisation sweeps, which
ascend Q without maximising it. "Exact" belongs to the continuous recursion, not
to this implementation of it.

Problem
-------
The clean data is a Markov chain whose transition kernel is *unknown*:

    p_theta(a) = mu(a_1) prod_{i=2}^n K_theta(a_i | a_{i-1}),

observed only through the OU channel  x = alpha_t a + sqrt(Delta_t) z. We are
given a sample {(x^(d), t_d)} and want theta. This is a latent-variable problem
(the clean chain a is never seen), so the natural estimator is maximum marginal
likelihood, computed by EM.

Why the E-step is structurally exact here
-----------------------------------------
The posterior p_theta(a | x) is again a chain: the likelihood factorizes
sitewise, so it only reweights the node potentials and leaves the edge
structure untouched. A chain is a tree, hence sum-product BP returns the
*exact* single-site and pairwise posterior marginals. There is no loopy-BP
approximation anywhere.

The remaining error is the grid representation of the continuous messages, and it
is measured rather than assumed. At M = 401, A = 8 the recursion agrees with the
Gaussian closed form to 9.2e-15 relative and with brute-force enumeration of the
discretised posterior to 1.6e-14 -- at that configuration, on those tests. That is
not a 1e-15 error bound for arbitrary kernels, tails or noise levels: trapezoidal
quadrature is O(h^2) and degrades when a likelihood is narrower than a few grid
cells, and the truncation diagnostic is a statistic over sampled chains whose
worst case at A = 8 is 1.2e-6, not its 90th percentile of 1.5e-8. See
frozen_config.half_width and experiments/exp_18_revision_diagnostics.py.

The one matrix that is the whole E-step
---------------------------------------
Write the grid as u_1..u_M with quadrature weights w, and let

    f_i = w * L_i * ell_i        ("forward message including its local factor")
    g_i = w * R_i * ell_i        ("backward message including its local factor")

for one chain. The pairwise posterior on edge (i, i+1) is

    b_{i,i+1}(u_j, u_k)  =  f_i(j) K[k, j] g_{i+1}(k) / Z_i,
    Z_i = g_{i+1}^T K f_i.

Every quantity EM needs is an expectation under these pairwise beliefs summed
over edges and over the dataset, so the entire E-step compresses into a single
M x M matrix

    C = sum_{d, i} g^{(d)}_{i+1} (f^{(d)}_i)^T / Z^{(d)}_i,      Xi = C * K,

with `*` elementwise. Xi[k, j] is the expected posterior mass placed on the
transition u_j -> u_k, summed over the whole dataset: exactly the continuous
analogue of the expected transition-count matrix of Baum-Welch.

Xi is a *complete summary for the current Q-function and M-step*: it does not
depend on how K_theta is parameterized, so the maximisation never revisits the
data, and sum(Xi) equals the number of edges in the dataset. It is NOT a
sufficient statistic of the observations, which is what calling it "a sufficient
statistic" unqualified suggests -- it is computed from the posterior at the
current theta, and so is conditional on that theta, on the model family, on the
grid, on the fixed initial law, and on the transition being homogeneous. Change
any of those and Xi changes.

Consequences used throughout this module:

1. The expected complete-data log-likelihood is a finite bilinear form,

       Q(theta | theta') = <Xi(theta'), log K_theta>  +  site-1 term  +  const,

   so the M-step never touches the data again -- only the M x M matrix Xi.

2. Fisher's identity gives the gradient of the *exact* marginal log-likelihood
   with no differentiation through the forward-backward recursion:

       grad_theta L(theta) = <Xi(theta), grad_theta log K_theta>.

   This is the key computational point: BP is not something we backpropagate
   through, it is the thing that supplies the expectation. No autodiff needed.

3. Cost per EM iteration is O(N n M^2), independent of the number of
   parameters, and the M-step is O(P M^2) with P = dim(theta).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.backend import get_xp, to_device, to_host


@dataclass(frozen=True)
class ExpectedStatistics:
    """Sufficient statistics of one E-step over a whole dataset.

    xi           : (M, M) expected pairwise transition mass, xi[k, j] is the mass
                   on u_j -> u_k summed over all edges of all chains. Sums to
                   the total number of edges.
    site1        : (M,) expected posterior mass at the first site, summed over
                   chains (sums to the number of chains). Only needed if the
                   initial density mu is also learned.
    log_evidence : sum_d log p_{t_d, theta}(x^(d)), the exact marginal
                   log-likelihood at the theta used in the E-step. Monotone
                   increase of this number across EM iterations is the
                   correctness check for the whole algorithm.
    n_edges      : total number of edges (chains x (n - 1)).
    n_chains     : total number of chains.
    """

    xi: np.ndarray
    site1: np.ndarray
    log_evidence: float
    n_edges: int
    n_chains: int

    def __add__(self, other: "ExpectedStatistics") -> "ExpectedStatistics":
        return ExpectedStatistics(
            xi=self.xi + other.xi,
            site1=self.site1 + other.site1,
            log_evidence=self.log_evidence + other.log_evidence,
            n_edges=self.n_edges + other.n_edges,
            n_chains=self.n_chains + other.n_chains,
        )


def e_step(
    grid: np.ndarray,
    weights: np.ndarray,
    log_K: np.ndarray,
    X: np.ndarray,
    alpha: float,
    delta: float,
    log_mu: np.ndarray | None = None,
    chunk: int = 256,
) -> ExpectedStatistics:
    """Exact BP E-step over a batch of noisy chains observed at one noise level.

    Parameters
    ----------
    log_K : (M, M) log K[out, in] = log K_theta(u_out | u_in) at the current theta.
    X     : (B, n) noisy observations, all at the same (alpha, delta).
    log_mu: (M,) log initial density; defaults to the standard normal used by
            `priors` when it draws a_1.
    chunk : batch rows processed at once (memory knob; results are exact and
            chunk-independent).

    Returns the accumulated `ExpectedStatistics`. The forward-backward pass is
    the same recursion as `bp_grid.grid_bp_batch`; the extra work is the rank-one
    accumulation of the pairwise statistics and the evidence bookkeeping.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    m = len(grid)
    if log_mu is None:
        log_mu = -0.5 * grid**2 - 0.5 * np.log(2.0 * np.pi)
    K = np.exp(log_K)

    total = None
    for start in range(0, X.shape[0], chunk):
        part = _e_step_chunk(
            grid, weights, K, X[start : start + chunk], alpha, delta, log_mu, m
        )
        total = part if total is None else total + part
    if total is None:
        raise ValueError("Empty observation batch.")
    return total


def _e_step_chunk(
    grid: np.ndarray,
    weights: np.ndarray,
    K: np.ndarray,
    X: np.ndarray,
    alpha: float,
    delta: float,
    log_mu: np.ndarray,
    m: int,
) -> ExpectedStatistics:
    b_size, n = X.shape
    if n < 2:
        raise ValueError("EM on a chain needs at least two sites.")

    # The E-step is where this project's compute actually goes: the forward and
    # backward sweeps are 2(n-1) matmuls of (B, M) against (M, M), and the
    # pairwise accumulation is two more at (E, M) x (M, M) with E = B(n-1). At
    # M = 401 and nseq in the thousands that is the whole cost of a fit, so it
    # is the one part worth putting on a device.
    #
    # Everything below runs through `xp`, which is numpy unless BP_DEVICE asks
    # for a GPU. The device boundary is drawn HERE rather than at the caller:
    # results are returned to the host before `ExpectedStatistics` is built, so
    # the M-step, the kernels and every consumer stay numpy-only and unaware.
    # That keeps the port to one function, and it keeps the CPU path bit-for-bit
    # what it was: with xp = numpy, `to_device`/`to_host` are `np.asarray` and
    # every call below is the one that was there before. The single place where
    # the two devices would have diverged is flagged where it happens.
    xp = get_xp()
    grid = to_device(np.asarray(grid, dtype=float), xp)
    weights = to_device(np.asarray(weights, dtype=float), xp)
    K = to_device(np.asarray(K, dtype=float), xp)
    X = to_device(np.asarray(X, dtype=float), xp)
    log_mu = to_device(np.asarray(log_mu, dtype=float), xp)

    # Row-shifted likelihood tables, one per (chain, site); the removed shifts
    # are restored in the evidence at the end.
    z = X[:, :, None] - alpha * grid[None, None, :]
    log_ell = -0.5 * z**2 / delta
    row_shift = log_ell.max(axis=2)  # (B, n)
    log_ell = log_ell - row_shift[:, :, None]
    ell = xp.exp(log_ell)  # (B, n, M)

    L = xp.empty((n, b_size, m))
    R = xp.empty((n, b_size, m))
    L[0] = xp.exp(log_mu)[None, :]
    log_z_fwd = xp.zeros(b_size)

    for i in range(n - 1):
        incoming = L[i] * ell[:, i, :] * weights[None, :]  # (B, M)
        out = incoming @ K.T  # (B, M);  out[b, k] = sum_j K[k, j] incoming[b, j]
        mass = out @ weights
        _guard(xp, mass, f"Forward message {i + 1} lost all mass.")
        L[i + 1] = out / mass[:, None]
        log_z_fwd += xp.log(mass)

    R[-1] = 1.0
    for i in range(n - 1, 0, -1):
        incoming = R[i] * ell[:, i, :] * weights[None, :]
        out = incoming @ K
        mass = out @ weights
        _guard(xp, mass, f"Backward message {i - 1} lost all mass.")
        R[i - 1] = out / mass[:, None]

    # Pairwise accumulation. F[i] and G[i] carry the local factors so that the
    # unnormalized pairwise belief on edge (i, i+1) is F[i](j) K[k, j] G[i+1](k).
    F = L * ell.transpose(1, 0, 2) * weights[None, None, :]  # (n, B, M)
    G = R * ell.transpose(1, 0, 2) * weights[None, None, :]  # (n, B, M)

    f_all = F[:-1].reshape(-1, m)  # (B(n-1), M)
    g_all = G[1:].reshape(-1, m)  # (B(n-1), M)
    # Z_e = g_e^T K f_e for every edge e, as two BLAS calls rather than a
    # three-operand einsum (which would contract at O(E M^2) outside BLAS).
    #
    # ONE contraction for both backends. This used to branch -- numpy on
    # `einsum`, cupy on multiply-and-sum -- because the two accumulate in
    # different orders and disagree at ~1e-15 relative, and the grid-validation
    # figures quoted in the paper had been measured on the einsum path. That is
    # not a reason: 1e-15 is four orders below the tightest number the paper
    # reports (the 1e-11 Gaussian agreement) and eleven below anything
    # scientific. Preserving an incidental floating-point summation order so a
    # published digit does not move is superstition, and it costs a real thing
    # -- the two backends were running different arithmetic, which is precisely
    # what the parity test exists to rule out.
    partition = (g_all * (f_all @ K.T)).sum(axis=1)
    _guard(xp, partition, "Pairwise belief lost all mass.")
    c_mat = (g_all / partition[:, None]).T @ f_all  # (M, M), C[k, j]
    xi = c_mat * K

    # Site-1 posterior mass (for a learnable initial density).
    raw1 = L[0] * ell[:, 0, :] * R[0]
    mass1 = raw1 @ weights
    site1 = ((raw1 / mass1[:, None]) * weights[None, :]).sum(axis=0)

    # Exact evidence: forward normalizers x tail integral x restored constants.
    tail = (L[-1] * ell[:, -1, :]) @ weights
    log_const = row_shift.sum(axis=1) + n * (-0.5 * np.log(2.0 * np.pi * delta))
    log_evidence = float(to_host(xp.sum(log_z_fwd + xp.log(tail) + log_const)))

    return ExpectedStatistics(
        xi=to_host(xi),
        site1=to_host(site1),
        log_evidence=log_evidence,
        n_edges=b_size * (n - 1),
        n_chains=b_size,
    )


def _guard(xp, v, message: str) -> None:
    """Raise if a normaliser has gone non-finite or non-positive.

    Kept as a per-message check rather than deferred to the end of the sweep,
    even though on a GPU each call forces a synchronisation: the site index in
    the message is how these failures have actually been diagnosed, and a fit
    that has lost mass is not worth the microseconds saved by finding out later.
    """
    if not bool(xp.all(xp.isfinite(v))) or bool(xp.any(v <= 0.0)):
        raise FloatingPointError(message)


def e_step_multi(
    grid: np.ndarray,
    weights: np.ndarray,
    log_K: np.ndarray,
    groups,
    log_mu: np.ndarray | None = None,
    chunk: int = 256,
) -> ExpectedStatistics:
    """E-step over several noise levels at once.

    `groups` is an iterable of (X, alpha, delta). Because the noise level enters
    only through the local likelihood factors, the statistics of different levels
    are simply added: one Xi matrix summarizes a dataset spread over the whole
    diffusion schedule.
    """
    total = None
    for X, alpha, delta in groups:
        part = e_step(grid, weights, log_K, X, alpha, delta, log_mu, chunk)
        total = part if total is None else total + part
    if total is None:
        raise ValueError("No observation groups supplied.")
    return total


def clean_statistics(
    grid: np.ndarray, A: np.ndarray
) -> ExpectedStatistics:
    """Statistics for *clean* chains, i.e. the t -> 0 limit of the E-step.

    With no noise the posterior collapses onto the observation, the pairwise
    belief becomes a product of deltas, and Xi degenerates to a histogram of the
    observed transitions. Binning those deltas onto the same grid (linear
    interpolation, which preserves the first moment exactly) puts the clean-data
    MLE and the noisy-data EM on a common footing: identical M-step code, only
    the way Xi is built differs. This is the "no noising required" baseline.

    No quadrature weights appear here, and that is not an oversight: the
    clean-data Xi is a *counting* measure over observed transitions, whereas the
    BP Xi is belief mass that already absorbed the weights. Both sum to the edge
    count, which is exactly the property the M-steps rely on, so the two are
    interchangeable inputs despite being built from different objects.

    A : (B, n) clean chains.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m = len(grid)
    src = _interp_weights(grid, A[:, :-1].ravel())
    dst = _interp_weights(grid, A[:, 1:].ravel())
    xi = _bilinear_accumulate(m, src, dst)
    site1 = np.zeros(m)
    s0 = _interp_weights(grid, A[:, 0])
    np.add.at(site1, s0[0], 1.0 - s0[2])
    np.add.at(site1, s0[1], s0[2])
    return ExpectedStatistics(
        xi=xi,
        site1=site1,
        log_evidence=float("nan"),  # complete-data case: use `complete_log_lik`
        n_edges=A.shape[0] * (A.shape[1] - 1),
        n_chains=A.shape[0],
    )


def _interp_weights(grid: np.ndarray, values: np.ndarray):
    """Linear-interpolation indices/weights placing `values` on `grid`."""
    lo = grid[0]
    dx = float(grid[1] - grid[0])
    pos = np.clip((values - lo) / dx, 0.0, len(grid) - 1 - 1e-12)
    i0 = np.floor(pos).astype(int)
    frac = pos - i0
    i1 = np.minimum(i0 + 1, len(grid) - 1)
    return i0, i1, frac


def _bilinear_accumulate(m: int, src, dst) -> np.ndarray:
    """Scatter unit mass at each (source, destination) pair bilinearly."""
    j0, j1, fj = src
    k0, k1, fk = dst
    xi = np.zeros((m, m))
    for kk, wk in ((k0, 1.0 - fk), (k1, fk)):
        for jj, wj in ((j0, 1.0 - fj), (j1, fj)):
            np.add.at(xi, (kk, jj), wk * wj)
    return xi


@dataclass
class EMTrace:
    """Per-iteration record of an EM run.

    log_evidence is the exact marginal log-likelihood *at the parameters used
    for that E-step*, so entry k is L(theta^(k)). A non-monotone sequence means
    a bug in the M-step (or a learning rate too large in the generalized case),
    which is why it is recorded rather than discarded.
    """

    log_evidence: list[float]
    theta: list[np.ndarray]
    seconds: list[float]

    # New fields carry defaults so the ~30 existing call sites are untouched.
    n_edges: int = 0
    """Transitions in the dataset, chains x (n_sites - 1). Needed to read any of
    the log-evidence numbers per transition rather than as a dataset total."""

    converged: bool = False
    stop_reason: str = ""
    """Why the loop ended: "converged", "censored" (hit the iteration cap), or
    "" for a trace built before this field existed. A censored run used to be
    returned indistinguishably from a converged one, so a caller comparing
    estimators could not tell whether it was comparing converged estimators or
    just two runs that had been given the same number of iterations."""

    @property
    def monotone_violation(self) -> float:
        """Largest decrease of the log-likelihood across iterations (0 if none)."""
        d = np.diff(np.asarray(self.log_evidence, dtype=float))
        return float(max(0.0, -d.min())) if d.size else 0.0

    @property
    def per_edge_log_evidence(self) -> list[float]:
        """The evidence trace in nats per transition.

        The comparable quantity across sample sizes: the total sums over edges,
        so it grows with the dataset and two runs at different n cannot have
        their increments compared directly.
        """
        if not self.n_edges:
            raise ValueError("n_edges was not recorded on this trace")
        return [v / self.n_edges for v in self.log_evidence]


def fit_em(
    kernel,
    grid: np.ndarray,
    weights: np.ndarray,
    groups,
    n_iters: int = 50,
    log_mu: np.ndarray | None = None,
    tol: float = 1e-9,
    chunk: int = 256,
    checkpoints: "set[int] | None" = None,
):
    """Run EM on noisy observations. Returns (fitted_kernel, EMTrace).

    `groups` is a list of (X, alpha, delta): one entry per noise level, each
    with its own batch of noisy chains. Every iteration is one exact BP E-step
    over the whole dataset followed by the kernel's own M-step.

    Stops early when the marginal log-likelihood increases by less than `tol`
    NATS PER TRANSITION, and only if it increased.

    The load-bearing change is the sign. The old test was

        abs(L_k - L_{k-1}) <= tol * abs(L_{k-1})

    which accepts a small DECREASE as though it were a small increase. Exact EM
    cannot decrease the evidence, so a decrease is a defect -- a broken M-step, or
    a generalised M-step overshooting -- and stopping on it reported the defect as
    success. The loop now requires a non-negative increment, records any decrease
    in `trace.monotone_violation`, and labels a run that never achieves a small
    non-negative increment as censored rather than converged.

    Normalising by `stats.n_edges` is a change of units, NOT a bug fix, and the
    distinction is worth stating because an external audit read it the other way.
    The old threshold was relative to the dataset total, and since dL and L are
    both sums over the same edges, dL/|L| is invariant under changing the sample
    size -- measured on the same law at 1x and 4x the data it agrees to every
    digit. So the old rule was already scale-free; it was not the n-dependent
    tolerance that had to be fixed in the ring experiment's plateau test. What
    per-transition units buy is interpretability: a constant threshold in nats per
    transition, rather than one that tightens over the run as |L| falls and that
    means something different at every noise level. At the budgets used here the
    default 1e-9 is far below the increments EM actually takes (~1e-4 per edge at
    iteration 30), so this guard rarely fires and the cap or the shape rule
    governs -- which is why the change is close to behaviourally inert.

    The returned kernel is always the one whose evidence is `trace.log_evidence[-1]`, and
    that alignment is load-bearing rather than cosmetic. An earlier version logged the
    evidence *before* each M-step and returned the parameters produced by the last one, so
    the kernel handed back was one update beyond anything that had been evaluated: its
    likelihood was never computed, the monotonicity check never covered the final step, and
    a caller comparing the reported evidence against a held-out number was comparing two
    different models. The loop below therefore evaluates each proposal after the M-step that
    produced it, and stops on the post-update value.
    """
    import time

    trace = EMTrace(log_evidence=[], theta=[], seconds=[])
    current = kernel
    prev = -np.inf
    saved: dict[int, object] = {}
    for it in range(n_iters):
        t0 = time.perf_counter()
        # `current` here is the iterate about to be scored, so trace.log_evidence[it]
        # is its evidence. Snapshotting at this point keeps the same alignment the
        # return value has: a checkpoint is a kernel whose likelihood was computed.
        if checkpoints is not None and it in checkpoints:
            saved[it] = current
        log_k = current.log_transition_matrix(grid)
        stats = e_step_multi(grid, weights, log_k, groups, log_mu, chunk)
        trace.log_evidence.append(stats.log_evidence)
        trace.theta.append(np.asarray(current.theta, dtype=float).copy())
        proposal = current.m_step(stats, grid)
        trace.seconds.append(time.perf_counter() - t0)
        trace.n_edges = stats.n_edges
        # Per transition, and signed: a decrease is not convergence.
        gain = (stats.log_evidence - prev) / stats.n_edges
        converged = np.isfinite(prev) and 0.0 <= gain <= tol
        prev = stats.log_evidence
        if converged:
            # `current` is the parameter whose evidence is the last logged entry. The
            # proposal is discarded rather than returned unevaluated.
            trace.converged = True
            trace.stop_reason = "converged"
            break
        current = proposal
    else:
        # Iteration budget exhausted: `current` is the last proposal, which has not been
        # scored yet. Score it, so the returned kernel and the final trace entry agree here
        # too -- this is the branch the old code silently got wrong.
        log_k = current.log_transition_matrix(grid)
        t0 = time.perf_counter()
        stats = e_step_multi(grid, weights, log_k, groups, log_mu, chunk)
        trace.log_evidence.append(stats.log_evidence)
        trace.theta.append(np.asarray(current.theta, dtype=float).copy())
        trace.seconds.append(time.perf_counter() - t0)
        trace.n_edges = stats.n_edges
        # Reached the cap without a small non-negative increment. Censored, not
        # converged: the returned kernel is the best seen, but nothing here says
        # the optimiser had finished, and a comparison of "converged estimators"
        # may not use it as one.
        trace.converged = False
        trace.stop_reason = "censored"
        if checkpoints is not None and n_iters in checkpoints:
            saved[n_iters] = current

    if checkpoints is not None:
        # The final iterate is always available under its own index, so a caller asking
        # for a checkpoint past an early convergence break still gets the best kernel
        # rather than a KeyError.
        saved.setdefault(len(trace.log_evidence) - 1, current)
        return current, trace, saved
    return current, trace


def fit_clean(kernel, grid: np.ndarray, A: np.ndarray, n_iters: int = 1):
    """Maximum likelihood from *clean* chains -- the t = 0 degenerate case.

    There are no latent variables here, so this is plain MLE and one M-step
    already maximizes the likelihood for the closed-form kernels. Kernels with
    their own inner latent variable (a mixture label, a network) still need
    iterations, but none of them re-touches the data: Xi is built once.

    This is the baseline Marc's remark points at -- if the clean chain is
    available, no noising is needed to identify the prior, and the denoiser at
    every noise level follows from BP for free.
    """
    stats = clean_statistics(grid, A)
    current = kernel
    for _ in range(n_iters):
        current = current.m_step(stats, grid)
    return current, stats


def q_value(stats: ExpectedStatistics, log_K: np.ndarray) -> float:
    """Q(theta | theta') = <Xi(theta'), log K_theta>, the edge part of the ELBO.

    Constants that do not depend on theta (the likelihood terms and the
    posterior entropy) are dropped, so only *differences* of `q_value` across
    theta at fixed Xi are meaningful -- which is all the M-step needs.
    """
    return float(np.sum(stats.xi * log_K))


def q_gradient(stats: ExpectedStatistics, grad_log_K: np.ndarray) -> np.ndarray:
    """grad_theta Q = <Xi, grad_theta log K_theta>, given grad_log_K of shape (P, M, M).

    By Fisher's identity this is also the exact gradient of the marginal
    log-likelihood L(theta) whenever Xi was computed at the same theta.
    """
    return np.einsum("pkj,kj->p", grad_log_K, stats.xi)
