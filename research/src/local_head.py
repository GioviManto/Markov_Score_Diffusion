"""Local (convolutional) score heads: the receptive-field question, made concrete.

Ledger item B15 reads the Gaussian locality law as an architecture prescription:
a score head that sees only x_{i-r..i+r} should be near-optimal once
r >~ xi log(1/eps). That reading has always been classified HEURISTIC, because
nothing in the project ever trained such a head -- it was inferred from the
locality law (G12) and the truncated-inference result (B14).

This module builds the head so the prescription can be tested rather than
argued. A weight-shared window predictor

    m_i  =  f_phi( x_{i-r..i+r},  time features )

applied at every site is exactly a one-dimensional CNN with kernel width 2r+1
followed by pointwise layers, and r is its receptive-field radius. Sweeping r
turns "how big a receptive field does a score head need" into a measurement.

It also supplies the baseline the EM-versus-network comparison has been missing.
The vanilla MLP used in exp_07 is fully connected across sites and carries no
locality prior at all; a reviewer will reasonably ask whether an architecture
that *does* encode sequence structure closes the gap. This is that architecture,
and r = n recovers a global head for reference.

Boundary handling: windows are zero-padded, and every experiment also reports
the error restricted to interior sites (at least r from either end), because the
locality prediction is a statement about the interior. Reporting only the padded
number would confound receptive-field error with edge error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.nnet import MLP, sample_training_times, time_features
from src.noising import alpha_delta


def make_windows(X: np.ndarray, radius: int) -> np.ndarray:
    """(B, n) -> (B, n, 2r+1) sliding windows, zero-padded at the ends."""
    b, n = X.shape
    if radius == 0:
        return X[:, :, None]
    padded = np.zeros((b, n + 2 * radius))
    padded[:, radius : radius + n] = X
    idx = np.arange(n)[:, None] + np.arange(2 * radius + 1)[None, :]
    return padded[:, idx]


@dataclass
class LocalHeadResult:
    net: MLP
    radius: int
    parameterization: str
    loss_history: list[float]
    seconds: float
    n_params: int
    n_grad_steps: int


def train_local_head(
    A_train: np.ndarray,
    t_values,
    radius: int,
    rng: np.random.Generator,
    hidden: tuple[int, ...] = (64, 64),
    n_steps: int = 6000,
    batch_size: int = 64,
    lr: float = 2e-3,
    parameterization: str = "eps",
    log_every: int = 500,
    t_range: tuple[float, float] | None = None,
) -> LocalHeadResult:
    """Denoising score matching for a weight-shared window predictor.

    Each site contributes one training example, so a minibatch of B chains
    supplies B*n examples -- the weight sharing is what makes a small receptive
    field cheap as well as structured.

    `t_range=(t_min, t_max)` trains continuously over the integration interval instead of on
    the discrete levels in `t_values`, and is required for any head that will be called
    inside a reverse SDE. See `nnet.sample_training_times`.
    """
    import time

    if parameterization not in ("eps", "x0"):
        raise ValueError(f"Unknown parameterization {parameterization!r}.")

    n_data, n_sites = A_train.shape
    sizes = (2 * radius + 1 + 3,) + tuple(hidden) + (1,)
    net = MLP.init(sizes, rng)
    t_arr = np.asarray(t_values, dtype=float)

    m_state = [np.zeros_like(p) for p in net.params]
    v_state = [np.zeros_like(p) for p in net.params]
    b1, b2, eps_adam = 0.9, 0.999, 1e-8
    history: list[float] = []

    t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        idx = rng.integers(0, n_data, size=min(batch_size, n_data))
        a = A_train[idx]
        t = sample_training_times(
            rng, len(idx),
            t_values=None if t_range is not None else t_arr,
            t_range=t_range,
        )
        alpha = np.exp(-t)[:, None]
        delta = (1.0 - np.exp(-2.0 * t))[:, None]
        z = rng.standard_normal(a.shape)
        x = alpha * a + np.sqrt(delta) * z

        win = make_windows(x, radius).reshape(-1, 2 * radius + 1)
        tf = np.repeat(time_features(t), n_sites, axis=0)
        feats = np.concatenate([win, tf], axis=1)

        target = (z if parameterization == "eps" else a).reshape(-1, 1)
        out, cache = net.forward(feats)
        diff = out - target
        grads = net.backward(cache, 2.0 * diff / len(diff))
        for j, g in enumerate(grads):
            m_state[j] = b1 * m_state[j] + (1 - b1) * g
            v_state[j] = b2 * v_state[j] + (1 - b2) * g**2
            mh = m_state[j] / (1 - b1**step)
            vh = v_state[j] / (1 - b2**step)
            net.params[j] = net.params[j] - lr * mh / (np.sqrt(vh) + eps_adam)
        if step % log_every == 0 or step == 1:
            history.append(float(np.mean(diff**2)))
    seconds = time.perf_counter() - t0

    return LocalHeadResult(
        net=net,
        radius=radius,
        parameterization=parameterization,
        loss_history=history,
        seconds=seconds,
        n_params=net.n_params,
        n_grad_steps=n_steps,
    )


def local_posterior_mean(result: LocalHeadResult, X: np.ndarray, t: float) -> np.ndarray:
    """Estimate E[a | x] at every site from its own window."""
    alpha, delta = alpha_delta(t)
    b, n = X.shape
    win = make_windows(X, result.radius).reshape(-1, 2 * result.radius + 1)
    tf = np.repeat(time_features(np.full(b, float(t))), n, axis=0)
    out, _ = result.net.forward(np.concatenate([win, tf], axis=1))
    out = out.reshape(b, n)
    if result.parameterization == "x0":
        return out
    return (X - np.sqrt(delta) * out) / alpha


def interior_slice(n_sites: int, radius: int) -> slice:
    """Sites whose window does not touch the zero padding."""
    r = min(radius, (n_sites - 1) // 2)
    return slice(r, n_sites - r)
