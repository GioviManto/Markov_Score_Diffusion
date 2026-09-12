"""Minimal pure-numpy MLP + Adam for supervised score / residual regression.

Deliberately dependency-free and deterministic: initialization and minibatch
order are driven by an explicit `numpy.random.Generator`. The network maps
(x, time features) -> R^n and is trained on exact-score targets, so the
comparison "direct score net vs BP baseline vs BP + residual net" is a clean
supervised regression problem with a known ground truth (no denoising-score-
matching variance confound).

Time features: (t, alpha_t, sqrt(Delta_t)) -- a smooth 3-dim embedding that
lets one network serve all diffusion times.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def time_features(t: np.ndarray | float) -> np.ndarray:
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    alpha = np.exp(-t_arr)
    delta = 1.0 - np.exp(-2.0 * t_arr)
    return np.stack([t_arr, alpha, np.sqrt(delta)], axis=-1)


def sample_training_times(
    rng: np.random.Generator,
    size: int,
    t_values=None,
    t_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Draw the noise levels a denoiser trains on.

    Two regimes, and the difference is not cosmetic.

    ``t_values`` draws uniformly from a fixed finite set. That is what this project did
    originally, with five levels, and it leaves the network's score *undefined between them*:
    it is fitted at five points and then queried everywhere. A reverse integration calls the
    score at every point of ``[t_min, t_max]`` -- 400 of them on a geometric grid -- so a
    generated-sample comparison built on discrete-level training measures interpolation and
    out-of-range extrapolation on top of the estimator error it means to isolate. With
    ``t_train`` spanning 0.1 to 1.6 and integration over 0.02 to 3.0, both extrapolation
    regions are entered on every trajectory.

    ``t_range=(t_min, t_max)`` draws log-uniformly over the continuous interval. Log-uniform
    rather than uniform because `reverse.time_grid` is ``np.geomspace``: sampling in
    ``log t`` puts training density where the integrator actually spends its steps, which is
    the small-``t`` end where the drift stiffens like ``1/(2t)``. Uniform sampling would
    under-train exactly the region that governs the generated law.

    Exactly one of the two must be supplied, so that a caller cannot silently keep the old
    behaviour by omission.
    """
    if (t_values is None) == (t_range is None):
        raise ValueError("Supply exactly one of `t_values` or `t_range`.")
    if t_range is not None:
        lo, hi = float(t_range[0]), float(t_range[1])
        if not 0.0 < lo < hi:
            raise ValueError(f"Need 0 < t_min < t_max, got {t_range!r}.")
        return np.exp(rng.uniform(np.log(lo), np.log(hi), size=size))
    t_arr = np.asarray(t_values, dtype=float)
    return t_arr[rng.integers(0, len(t_arr), size=size)]


@dataclass
class MLP:
    """Fully connected tanh MLP; parameters stored as a flat list of arrays."""

    sizes: tuple[int, ...]
    params: list[np.ndarray] = field(default_factory=list)

    @classmethod
    def init(cls, sizes: tuple[int, ...], rng: np.random.Generator) -> "MLP":
        params: list[np.ndarray] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            w = rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
            b = np.zeros(fan_out)
            params.extend([w, b])
        return cls(sizes=sizes, params=params)

    @property
    def n_params(self) -> int:
        return int(sum(p.size for p in self.params))

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Return output and cached activations for backprop."""
        cache = [X]
        h = X
        n_layers = len(self.params) // 2
        for k in range(n_layers):
            w, b = self.params[2 * k], self.params[2 * k + 1]
            z = h @ w + b
            h = np.tanh(z) if k < n_layers - 1 else z
            cache.append(h)
        return h, cache

    def backward(
        self, cache: list[np.ndarray], grad_out: np.ndarray
    ) -> list[np.ndarray]:
        """Gradients of a scalar loss with upstream gradient `grad_out`."""
        grads: list[np.ndarray] = [np.empty(0)] * len(self.params)
        n_layers = len(self.params) // 2
        g = grad_out
        for k in reversed(range(n_layers)):
            h_in, h_out = cache[k], cache[k + 1]
            if k < n_layers - 1:  # undo tanh
                g = g * (1.0 - h_out**2)
            grads[2 * k] = h_in.T @ g
            grads[2 * k + 1] = g.sum(axis=0)
            g = g @ self.params[2 * k].T
        return grads


def train_score_net(
    net: MLP,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    rng: np.random.Generator,
    n_steps: int = 3000,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> MLP:
    """Adam training of net((x, timefeat)) -> Y with MSE loss. Returns the net.

    X: (N, n) observations; T: (N,) times; Y: (N, n) targets (scores or residuals).
    """
    feats = np.concatenate([X, time_features(T)], axis=1)
    n_data = X.shape[0]
    m_state = [np.zeros_like(p) for p in net.params]
    v_state = [np.zeros_like(p) for p in net.params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for step in range(1, n_steps + 1):
        idx = rng.integers(0, n_data, size=min(batch_size, n_data))
        xb, yb = feats[idx], Y[idx]
        out, cache = net.forward(xb)
        grad_out = 2.0 * (out - yb) / xb.shape[0]
        grads = net.backward(cache, grad_out)
        for j, g in enumerate(grads):
            m_state[j] = beta1 * m_state[j] + (1 - beta1) * g
            v_state[j] = beta2 * v_state[j] + (1 - beta2) * g**2
            m_hat = m_state[j] / (1 - beta1**step)
            v_hat = v_state[j] / (1 - beta2**step)
            net.params[j] = net.params[j] - lr * m_hat / (np.sqrt(v_hat) + eps)
    return net


def predict_score(net: MLP, X: np.ndarray, T: np.ndarray) -> np.ndarray:
    feats = np.concatenate([X, time_features(T)], axis=1)
    out, _ = net.forward(feats)
    return out
