"""Gaussian-mixture message passing: the closure that makes "message error" real.

Motivation (audit finding F2, and question R4/F1 of the report)
--------------------------------------------------------------
For a linear-transition chain with Gaussian OU likelihoods, moment-matched
*single*-Gaussian BP is mathematically identical to exact Gaussian BP on the
covariance-matched Gaussian model. So at the single-Gaussian level "message
approximation error" and "model approximation error" are the same number, and
the distinction the project cares about does not yet exist.

It becomes real exactly here. A Gaussian *mixture* message family can represent
the true message arbitrarily well, so the error it makes is purely one of
representation -- nothing about the model has been changed. The experiment this
module enables is therefore the clean one:

    take a prior whose transition kernel is *exactly* a Gaussian mixture, so the
    model is exactly representable and no model error exists at all, then sweep
    the number of message components and watch the pure representation error.

The report asked "how many mixture components are needed to beat single-Gaussian
closure". With an exactly-representable model the answer is not confounded by
anything else.

Why closure is needed
---------------------
The forward update sends a C-component message through a C_K-component kernel
and yields C x C_K components; iterating over n sites gives C_K^n. Exact
continuous BP on a mixture chain is therefore closed in *form* but not in
*size*. Keeping the representation finite requires collapsing back to C
components after every step, and the collapse is the approximation.

Collapsing uses Runnalls' KL-based pairwise merge: repeatedly merge the pair
whose merge costs least in an upper bound on the KL divergence to the original
mixture. It is the standard choice and is deterministic, which matters here
because the experiments must be reproducible.

Conventions match the rest of the package: chain a_i = rho a_{i-1} + eps_i with
eps ~ sum_k pi_k N(mu_k, s2_k), likelihood x_i | a_i ~ N(alpha a_i, Delta).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass
class MixtureMessage:
    """A Gaussian mixture held in log-weight form.

    log_w need not be normalized: messages are defined up to a positive
    constant, and every consumer either renormalizes or takes moments.
    """

    log_w: np.ndarray
    mean: np.ndarray
    var: np.ndarray

    def __post_init__(self) -> None:
        self.log_w = np.atleast_1d(np.asarray(self.log_w, dtype=float))
        self.mean = np.atleast_1d(np.asarray(self.mean, dtype=float))
        self.var = np.atleast_1d(np.asarray(self.var, dtype=float))

    @property
    def size(self) -> int:
        return len(self.log_w)

    def weights(self) -> np.ndarray:
        """Normalized mixture weights."""
        m = self.log_w.max()
        w = np.exp(self.log_w - m)
        return w / w.sum()

    def moments(self) -> tuple[float, float]:
        """Mean and variance of the mixture."""
        w = self.weights()
        mean = float(w @ self.mean)
        var = float(w @ (self.var + self.mean**2) - mean**2)
        return mean, var

    def renormalized(self) -> "MixtureMessage":
        return MixtureMessage(self.log_w - self.log_w.max(), self.mean, self.var)


def _gauss_logpdf(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    return -0.5 * (_LOG_2PI + np.log(var) + (x - mean) ** 2 / var)


def multiply_by_gaussian(
    msg: MixtureMessage, y: float, r: float
) -> MixtureMessage:
    """Multiply a mixture by a single Gaussian N(a; y, r), exactly.

    Each component picks up the evidence N(y; m_j, v_j + r) and its Gaussian is
    updated by the usual precision-weighted combination. Exact, no approximation.
    """
    log_w = msg.log_w + _gauss_logpdf(np.asarray(y), msg.mean, msg.var + r)
    prec = 1.0 / msg.var + 1.0 / r
    var = 1.0 / prec
    mean = (msg.mean / msg.var + y / r) * var
    return MixtureMessage(log_w, mean, var)


def push_forward(
    msg: MixtureMessage, rho: float, pi: np.ndarray, mu: np.ndarray, s2: np.ndarray
) -> MixtureMessage:
    """Send a mixture through the transition: a' = rho a + eps.

    int N(a'; rho a + mu_k, s2_k) N(a; m_j, v_j) da = N(a'; rho m_j + mu_k,
    rho^2 v_j + s2_k). Component count multiplies.
    """
    log_w = (msg.log_w[:, None] + np.log(pi)[None, :]).ravel()
    mean = (rho * msg.mean[:, None] + mu[None, :]).ravel()
    var = (rho**2 * msg.var[:, None] + s2[None, :]).ravel()
    return MixtureMessage(log_w, mean, var)


def push_backward(
    msg: MixtureMessage, rho: float, pi: np.ndarray, mu: np.ndarray, s2: np.ndarray
) -> MixtureMessage:
    """Send a mixture backwards through the transition.

    int N(a'; rho a + mu_k, s2_k) N(a'; m_j, v_j) da' = N(rho a; m_j - mu_k,
    v_j + s2_k), which as a function of a is proportional to
    N(a; (m_j - mu_k)/rho, (v_j + s2_k)/rho^2). The 1/|rho| Jacobian is common
    to every component and so is absorbed by renormalization.
    """
    log_w = (msg.log_w[:, None] + np.log(pi)[None, :]).ravel()
    mean = ((msg.mean[:, None] - mu[None, :]) / rho).ravel()
    var = ((msg.var[:, None] + s2[None, :]) / rho**2).ravel()
    return MixtureMessage(log_w, mean, var)


def collapse(msg: MixtureMessage, max_components: int) -> MixtureMessage:
    """Runnalls' KL-based pairwise merge down to `max_components`.

    Repeatedly merges the pair minimizing

        B(i,j) = 0.5 [ (w_i + w_j) log v_ij - w_i log v_i - w_j log v_j ],

    an upper bound on the KL divergence between the mixture and its merged
    version. Deterministic and order-independent given the input, which is what
    the experiments need.

    This is the *only* approximation in mixture BP; everything else above is
    exact. Set max_components large enough and the recursion becomes exact.
    """
    if msg.size <= max_components:
        return msg

    w = msg.weights()
    m = msg.mean.copy()
    v = msg.var.copy()
    alive = np.ones(len(w), dtype=bool)

    while int(alive.sum()) > max_components:
        idx = np.flatnonzero(alive)
        wi, mi, vi = w[idx], m[idx], v[idx]
        # Pairwise merge cost, upper triangle only.
        w_sum = wi[:, None] + wi[None, :]
        safe = np.where(w_sum > 0, w_sum, 1.0)
        m_merged = (wi[:, None] * mi[:, None] + wi[None, :] * mi[None, :]) / safe
        v_merged = (
            wi[:, None] * (vi[:, None] + (mi[:, None] - m_merged) ** 2)
            + wi[None, :] * (vi[None, :] + (mi[None, :] - m_merged) ** 2)
        ) / safe
        cost = 0.5 * (
            w_sum * np.log(np.maximum(v_merged, 1e-300))
            - wi[:, None] * np.log(np.maximum(vi, 1e-300))[:, None]
            - wi[None, :] * np.log(np.maximum(vi, 1e-300))[None, :]
        )
        np.fill_diagonal(cost, np.inf)
        cost = np.triu(cost, 1)
        cost[cost == 0.0] = np.inf

        flat = int(np.argmin(cost))
        a, b = divmod(flat, cost.shape[1])
        ia, ib = idx[a], idx[b]

        w_new = w[ia] + w[ib]
        m_new = (w[ia] * m[ia] + w[ib] * m[ib]) / w_new
        v_new = (
            w[ia] * (v[ia] + (m[ia] - m_new) ** 2)
            + w[ib] * (v[ib] + (m[ib] - m_new) ** 2)
        ) / w_new
        w[ia], m[ia], v[ia] = w_new, m_new, v_new
        alive[ib] = False

    keep = np.flatnonzero(alive)
    return MixtureMessage(np.log(np.maximum(w[keep], 1e-300)), m[keep], v[keep])


def mixture_chain_bp(
    x: np.ndarray,
    rho: float,
    pi: np.ndarray,
    mu: np.ndarray,
    s2: np.ndarray,
    alpha: float,
    delta: float,
    max_components: int,
    prior_var0: float = 1.0,
):
    """Forward-backward BP with Gaussian-mixture messages.

    Parameters
    ----------
    pi, mu, s2 : the innovation mixture, i.e. the transition kernel is
                 K(a' | a) = sum_k pi_k N(a' - rho a; mu_k, s2_k).
    max_components : message budget C. C = 1 recovers single-Gaussian ADF-BP,
                 which for a linear chain equals exact Gaussian BP on the
                 covariance-matched model (audit F2) -- so the C = 1 column of
                 any sweep is the existing Gaussian baseline, and every C > 1
                 column is pure representation improvement.

    Returns (means, variances) per site.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    pi = np.asarray(pi, dtype=float)
    mu = np.asarray(mu, dtype=float)
    s2 = np.asarray(s2, dtype=float)

    # The likelihood as a Gaussian in a: N(a; x_i / alpha, Delta / alpha^2).
    y = x / alpha
    r = delta / alpha**2

    left: list[MixtureMessage] = [None] * n  # type: ignore[list-item]
    right: list[MixtureMessage] = [None] * n  # type: ignore[list-item]

    left[0] = MixtureMessage(np.zeros(1), np.zeros(1), np.array([prior_var0]))
    for i in range(n - 1):
        tilted = multiply_by_gaussian(left[i], y[i], r)
        pushed = push_forward(tilted, rho, pi, mu, s2)
        left[i + 1] = collapse(pushed.renormalized(), max_components)

    # Terminal right message is flat, so R_{n-1} * ell_{n-1} is just the
    # likelihood -- handled directly rather than by faking a huge variance.
    right[n - 1] = None  # type: ignore[assignment]
    tilted_last = MixtureMessage(np.zeros(1), np.array([y[n - 1]]), np.array([r]))
    if n >= 2:
        right[n - 2] = collapse(
            push_backward(tilted_last, rho, pi, mu, s2).renormalized(),
            max_components,
        )
    for i in range(n - 2, 0, -1):
        tilted = multiply_by_gaussian(right[i], y[i], r)
        pushed = push_backward(tilted, rho, pi, mu, s2)
        right[i - 1] = collapse(pushed.renormalized(), max_components)

    means = np.empty(n)
    variances = np.empty(n)
    for i in range(n):
        belief = multiply_by_gaussian(left[i], y[i], r)
        if right[i] is not None:
            belief = _multiply_mixtures(belief, right[i])
        means[i], variances[i] = belief.moments()
    return means, variances


def _multiply_mixtures(a: MixtureMessage, b: MixtureMessage) -> MixtureMessage:
    """Exact product of two Gaussian mixtures (component count multiplies)."""
    log_w = (
        a.log_w[:, None]
        + b.log_w[None, :]
        + _gauss_logpdf(a.mean[:, None], b.mean[None, :], a.var[:, None] + b.var[None, :])
    ).ravel()
    prec = 1.0 / a.var[:, None] + 1.0 / b.var[None, :]
    var = (1.0 / prec).ravel()
    mean = (
        (a.mean[:, None] / a.var[:, None] + b.mean[None, :] / b.var[None, :])
        / prec
    ).ravel()
    return MixtureMessage(log_w, mean, var)
