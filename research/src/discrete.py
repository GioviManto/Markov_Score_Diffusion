"""Discrete-alphabet chains: the setting with no discretization error at all.

Question F2 of the result ledger. Everything else in this package represents
continuous messages on a grid and pays for it: trapezoidal quadrature error,
tail truncation, the resolution condition at small t, and -- worst of the lot --
the ratio-lattice quantization that makes the Laplace M-step a biased estimator
of rho no matter how fine the grid gets.

Let the clean chain take values in a finite set of levels v_1..v_S instead. The
noising is still the continuous OU channel, so x = alpha a + sqrt(Delta) z is
still real-valued, but the *latent* variable now lives in a finite set. Messages
are therefore S-vectors, BP is exact with no representation whatsoever, and
every one of those error sources disappears simultaneously.

Two things this buys beyond cleanliness:

1. **The Baum-Welch analogy becomes an identity.** In the continuous case Xi is
   "the continuum analogue of the expected transition-count matrix". Here it is
   the expected transition-count matrix, and the M-step is exactly Baum-Welch's:
   normalize the counts. Nothing is analogous; it is the same object.

2. **A confound-free retest of the headline.** The EM-versus-network comparison
   currently lives in a setting with grid approximation. Reproducing it where
   BP is exact removes any suspicion that the margin is a numerical artifact.

The prior is genuinely non-Gaussian by construction -- a distribution on a
finite set of levels is about as far from Gaussian as a scalar prior gets --
so this is not a retreat to an easy case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiscreteChain:
    """Markov chain on levels v_1..v_S with transition K[out, in].

    Columns of K sum to one: K[s2, s1] = P(a_i = v_{s2} | a_{i-1} = v_{s1}),
    matching the K[out, in] convention used everywhere else in the package.
    """

    levels: np.ndarray
    K: np.ndarray
    mu: np.ndarray

    def __post_init__(self) -> None:
        if not np.allclose(self.K.sum(axis=0), 1.0, atol=1e-10):
            raise ValueError("Columns of K must sum to one (K[out, in]).")
        if not np.isclose(self.mu.sum(), 1.0, atol=1e-10):
            raise ValueError("mu must sum to one.")

    @property
    def n_states(self) -> int:
        return len(self.levels)

    @property
    def name(self) -> str:
        return f"discrete_S{self.n_states}"

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        idx = np.empty(n, dtype=int)
        idx[0] = rng.choice(self.n_states, p=self.mu)
        for i in range(1, n):
            idx[i] = rng.choice(self.n_states, p=self.K[:, idx[i - 1]])
        return self.levels[idx]

    def stationary(self) -> np.ndarray:
        """Leading eigenvector of K, normalized to a probability vector."""
        vals, vecs = np.linalg.eig(self.K)
        v = np.real(vecs[:, int(np.argmin(np.abs(vals - 1.0)))])
        v = np.abs(v)
        return v / v.sum()

    def moments(self) -> tuple[float, float]:
        p = self.stationary()
        m = float(p @ self.levels)
        return m, float(p @ self.levels**2 - m**2)


def make_random_chain(
    n_states: int, rng: np.random.Generator, concentration: float = 0.7,
    spread: float = 1.0,
) -> DiscreteChain:
    """A random chain with unit-variance levels and a Dirichlet transition.

    `concentration` < 1 gives sparse, peaked columns -- a chain that actually
    has structure to learn rather than one close to uniform.
    """
    levels = np.linspace(-spread, spread, n_states)
    levels = (levels - levels.mean()) / levels.std()
    K = rng.dirichlet(np.full(n_states, concentration), size=n_states).T
    chain = DiscreteChain(levels=levels, K=K, mu=np.full(n_states, 1.0 / n_states))
    return chain


@dataclass(frozen=True)
class DiscreteBPResult:
    beliefs: np.ndarray      # (B, n, S) posterior marginals
    means: np.ndarray        # (B, n) posterior means E[a_i | x]
    xi: np.ndarray           # (S, S) expected transition counts, summed over data
    log_evidence: float
    n_edges: int
    n_chains: int


def discrete_bp(
    levels: np.ndarray,
    K: np.ndarray,
    X: np.ndarray,
    alpha: float,
    delta: float,
    mu: np.ndarray | None = None,
) -> DiscreteBPResult:
    """Exact forward-backward on a discrete chain under the OU channel.

    No grid, no quadrature, no truncation: the latent alphabet is finite, so the
    messages are exact and the only floating-point error is roundoff.

    Returns beliefs, posterior means, the expected transition-count matrix Xi
    (the Baum-Welch statistic), and the exact log evidence.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    b_size, n = X.shape
    s = len(levels)
    if mu is None:
        mu = np.full(s, 1.0 / s)

    # Sitewise likelihood, max-shifted per (chain, site) for stability. The
    # shifts are restored in the evidence.
    z = X[:, :, None] - alpha * levels[None, None, :]
    log_ell = -0.5 * z**2 / delta - 0.5 * np.log(2.0 * np.pi * delta)
    shift = log_ell.max(axis=2)
    ell = np.exp(log_ell - shift[:, :, None])  # (B, n, S)

    L = np.empty((n, b_size, s))
    R = np.empty((n, b_size, s))
    L[0] = mu[None, :]
    log_norm = np.zeros(b_size)

    for i in range(n - 1):
        incoming = L[i] * ell[:, i, :]
        out = incoming @ K.T
        c = out.sum(axis=1)
        if np.any(c <= 0.0) or not np.all(np.isfinite(c)):
            raise FloatingPointError(f"Forward message {i + 1} lost all mass.")
        L[i + 1] = out / c[:, None]
        log_norm += np.log(c)

    R[-1] = 1.0
    for i in range(n - 1, 0, -1):
        incoming = R[i] * ell[:, i, :]
        out = incoming @ K
        c = out.sum(axis=1)
        if np.any(c <= 0.0) or not np.all(np.isfinite(c)):
            raise FloatingPointError(f"Backward message {i - 1} lost all mass.")
        R[i - 1] = out / c[:, None]

    beliefs = L.transpose(1, 0, 2) * ell * R.transpose(1, 0, 2)
    beliefs /= beliefs.sum(axis=2, keepdims=True)
    means = beliefs @ levels

    # Expected transition counts. Same construction as the grid Xi, minus the
    # quadrature weights: here the base measure is counting measure.
    f = (L * ell.transpose(1, 0, 2))[:-1].reshape(-1, s)   # (B(n-1), S)
    g = (R * ell.transpose(1, 0, 2))[1:].reshape(-1, s)
    partition = np.einsum("ek,ek->e", g, f @ K.T)
    if np.any(partition <= 0.0) or not np.all(np.isfinite(partition)):
        raise FloatingPointError("Pairwise belief lost all mass.")
    xi = ((g / partition[:, None]).T @ f) * K

    tail = (L[-1] * ell[:, -1, :]).sum(axis=1)
    log_evidence = float(np.sum(log_norm + np.log(tail) + shift.sum(axis=1)))

    return DiscreteBPResult(
        beliefs=beliefs,
        means=means,
        xi=xi,
        log_evidence=log_evidence,
        n_edges=b_size * (n - 1),
        n_chains=b_size,
    )


def baum_welch_m_step(xi: np.ndarray) -> np.ndarray:
    """The M-step: normalize expected transition counts. That is the whole thing.

    In the continuous case the M-step had to be derived separately for every
    kernel family and produced closed forms of varying difficulty (weighted
    Yule-Walker, a weighted median, an inner ECM). Here the transition matrix is
    its own parameterization and the maximizer of <Xi, log K> subject to columns
    summing to one is exactly the normalized counts.
    """
    col = xi.sum(axis=0)
    if np.any(col <= 0.0):
        raise FloatingPointError("A state received no expected transitions.")
    return xi / col[None, :]


def fit_em_discrete(
    levels: np.ndarray,
    K_init: np.ndarray,
    groups,
    n_iters: int = 200,
    mu: np.ndarray | None = None,
    tol: float = 1e-12,
):
    """EM (= Baum-Welch) for the transition matrix from noisy observations.

    `groups` is a list of (X, alpha, delta): observations at possibly different
    noise levels, whose statistics add into one Xi exactly as in the continuous
    case. Returns (K_fitted, trace) with the trace carrying the exact marginal
    log-likelihood per iteration for the monotonicity check.
    """
    K = np.array(K_init, dtype=float)
    trace = {"log_evidence": [], "K": []}
    prev = -np.inf
    for _ in range(n_iters):
        xi_total = np.zeros_like(K)
        ev = 0.0
        for X, alpha, delta in groups:
            res = discrete_bp(levels, K, X, alpha, delta, mu)
            xi_total += res.xi
            ev += res.log_evidence
        trace["log_evidence"].append(ev)
        trace["K"].append(K.copy())
        K = baum_welch_m_step(xi_total)
        if np.isfinite(prev) and abs(ev - prev) <= tol * abs(prev):
            break
        prev = ev
    return K, trace


def monotone_violation(log_evidence) -> float:
    d = np.diff(np.asarray(log_evidence, dtype=float))
    return float(max(0.0, -d.min())) if d.size else 0.0
