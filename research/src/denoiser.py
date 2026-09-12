"""The two denoisers being compared, behind one interface.

Both estimate the same object -- the posterior mean m_i(x, t) = E[a_i | x],
which determines the score through the exact OU identity
s = -(x - alpha_t m) / Delta_t -- but they get there in opposite ways.

BP denoiser (structure first)
    Learn the transition kernel K_theta of the clean chain, then *compute* the
    posterior mean by exact belief propagation. The learned object lives on
    R x R and does not depend on t at all; the noise level enters only through
    the likelihood factors inside BP. One fit therefore serves every noise
    level, and the estimated quantity is a low-dimensional parameter, so the
    statistical rate is parametric.

DSM network (function first)
    Learn the map (x, t) -> m directly by denoising score matching, the vanilla
    diffusion-model recipe. The learned object is a function on R^n x R_+, it
    must be fitted across the whole noise schedule at once, and nothing in the
    architecture knows the chain is Markov.

The comparison is deliberately generous to the network: it trains on *paired*
(clean, noisy) data with a fresh noise draw at every gradient step, i.e. it may
consume unlimited noise realizations of the same clean chains, while EM sees a
single noisy realization per chain and never sees the clean chain at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from src.backend import get_xp, to_host
from src.bp_grid import grid_bp_batch
from src.nnet import MLP, sample_training_times, time_features
from src.noising import alpha_delta


# ----------------------------------------------------------------------------
# BP denoiser from a (learned or true) kernel
# ----------------------------------------------------------------------------

def bp_posterior_mean(
    kernel,
    grid: np.ndarray,
    weights: np.ndarray,
    X: np.ndarray,
    t: float,
    log_mu: np.ndarray | None = None,
    xp=None,
) -> np.ndarray:
    """E[a | x] under the chain prior defined by `kernel`, by exact grid BP.

    `kernel` is anything with `log_transition_matrix(grid)`, which covers both
    the ground-truth priors in `priors` and the learned ones in `kernels`.

    ``xp`` selects the device; ``None`` reads ``BP_DEVICE`` from the environment, so a batch
    script chooses once and no call site changes. **The result is always returned on the
    host**, which keeps every downstream caller device-agnostic. That transfer is cheap by
    construction: the returned array is ``(B, n)`` while the intermediate messages it is
    computed from are ``(B, n, M)``, larger by the grid size -- so the expensive object never
    leaves the device.
    """
    if xp is None:
        xp = get_xp()
    alpha, delta = alpha_delta(t)
    log_k = kernel.log_transition_matrix(grid)
    means, _ = grid_bp_batch(grid, weights, log_k, X, alpha, delta, log_mu, xp=xp)
    return to_host(means)


def score_from_mean(
    X: np.ndarray, means: np.ndarray, t: float
) -> np.ndarray:
    """Exact OU identity, applied to a batch."""
    alpha, delta = alpha_delta(t)
    return -(X - alpha * means) / delta


# ----------------------------------------------------------------------------
# Denoising-score-matching baseline
# ----------------------------------------------------------------------------

@dataclass
class DSMResult:
    net: MLP
    parameterization: str
    loss_history: list[float]
    seconds: float
    n_params: int
    n_grad_steps: int
    # {step: net} when `checkpoints` was asked for, otherwise empty. A new field
    # with a default rather than a wider return type, so the ~30 existing call
    # sites are untouched.
    checkpoints: dict[int, MLP] = field(default_factory=dict)


def train_dsm_denoiser(
    A_train: np.ndarray,
    t_values,
    rng: np.random.Generator,
    hidden: tuple[int, ...] = (128, 128),
    n_steps: int = 6000,
    batch_size: int = 128,
    lr: float = 2e-3,
    log_every: int = 200,
    parameterization: str = "eps",
    t_range: tuple[float, float] | None = None,
    checkpoints: "set[int] | None" = None,
) -> DSMResult:
    """Vanilla diffusion training, in either standard parameterization.

    With x = alpha_t a + sqrt(Delta_t) z, the network either predicts the clean
    chain ("x0") or the noise ("eps"):

        x0 :  minimize E || a_phi(x,t) - a ||^2,   m_hat = a_phi
        eps:  minimize E || z_phi(x,t) - z ||^2,   m_hat = (x - sqrt(Delta) z_phi) / alpha

    Both have the same minimizer, E[a | x, t] -- the object BP computes -- so
    neither is handicapped by a surrogate loss. They are *not* equivalent in
    finite samples, because z = (x - alpha a)/sqrt(Delta) makes the two losses
    differ by the factor alpha_t^2 / Delta_t: eps-prediction upweights small t
    sharply, and its posterior mean inherits a sqrt(Delta_t)/alpha_t prefactor
    that suppresses network error exactly where the x0 parameterization is
    weakest. Since eps-prediction is the standard diffusion recipe and the
    stronger baseline at low noise, it is the default here; `x0` is retained so
    the comparison can be reported both ways rather than resting on a choice
    that happens to flatter the structured method.

    A_train : (N, n) clean chains -- the data budget.
    t_values: discrete noise levels sampled uniformly at each step. Ignored when
              `t_range` is given.
    t_range : (t_min, t_max) for continuous log-uniform training over the whole
              integration interval. **Use this for anything feeding a reverse SDE.**
              Training on a handful of discrete levels and then integrating over a
              continuum makes the generated-sample comparison measure interpolation
              and extrapolation as much as score quality; see
              `nnet.sample_training_times` for the full argument.
    checkpoints: training steps at which to keep a copy of the network, returned
              in `DSMResult.checkpoints`. This exists so the network's training
              length can be chosen on a validation bundle, the way EM-BP's
              iteration count is in `exp_07`. Holding one arm's budget fixed
              while selecting the other's is not neutral: training length is a
              bias/variance knob for both, so a fixed choice silently favours
              whichever arm it happens to suit at that sample size.
    """
    import time

    if parameterization not in ("eps", "x0"):
        raise ValueError(f"Unknown parameterization {parameterization!r}.")

    n_data, n_sites = A_train.shape
    sizes = (n_sites + 3,) + tuple(hidden) + (n_sites,)
    net = MLP.init(sizes, rng)
    t_arr = np.asarray(t_values, dtype=float)

    m_state = [np.zeros_like(p) for p in net.params]
    v_state = [np.zeros_like(p) for p in net.params]
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    history: list[float] = []
    saved: dict[int, MLP] = {}

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

        feats = np.concatenate([x, time_features(t)], axis=1)
        out, cache = net.forward(feats)
        target = z if parameterization == "eps" else a
        diff = out - target
        grads = net.backward(cache, 2.0 * diff / len(idx))
        for j, g in enumerate(grads):
            m_state[j] = beta1 * m_state[j] + (1 - beta1) * g
            v_state[j] = beta2 * v_state[j] + (1 - beta2) * g**2
            m_hat = m_state[j] / (1 - beta1**step)
            v_hat = v_state[j] / (1 - beta2**step)
            net.params[j] = net.params[j] - lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        if step % log_every == 0 or step == 1:
            history.append(float(np.mean(diff**2)))
        # After the update, so `saved[k]` is the network as it stands having
        # taken k steps -- the same convention as n_grad_steps. `net.params` is
        # rebound in place above, so the arrays must be copied or every
        # checkpoint would alias the final weights.
        if checkpoints is not None and step in checkpoints:
            saved[step] = replace(net, params=[p.copy() for p in net.params])
    seconds = time.perf_counter() - t0

    return DSMResult(
        net=net,
        parameterization=parameterization,
        loss_history=history,
        seconds=seconds,
        n_params=net.n_params,
        n_grad_steps=n_steps,
        checkpoints=saved,
    )


def dsm_posterior_mean(
    result: DSMResult | MLP, X: np.ndarray, t: float, parameterization: str | None = None
) -> np.ndarray:
    """Network estimate of E[a | x] at a single noise level.

    Accepts a `DSMResult` (which carries its own parameterization) or a bare
    `MLP` plus an explicit parameterization.
    """
    if isinstance(result, DSMResult):
        net, mode = result.net, result.parameterization
    else:
        net, mode = result, parameterization
    if mode not in ("eps", "x0"):
        raise ValueError("Parameterization must be given for a bare MLP.")

    alpha, delta = alpha_delta(t)
    t_arr = np.full(X.shape[0], float(t))
    feats = np.concatenate([X, time_features(t_arr)], axis=1)
    out, _ = net.forward(feats)
    if mode == "x0":
        return out
    return (X - np.sqrt(delta) * out) / alpha


# ----------------------------------------------------------------------------
# Shared evaluation
# ----------------------------------------------------------------------------

def evaluate_denoiser(
    means_hat: np.ndarray, means_ref: np.ndarray, X: np.ndarray, t: float
) -> dict:
    """Relative L2 errors on the posterior mean and on the induced score.

    The two are linked by the exact identity  s_hat - s_ref =
    (alpha/Delta)(m_hat - m_ref), so `identity_residual` must sit at machine
    precision; it is the same guard used by every other experiment here.
    """
    alpha, delta = alpha_delta(t)
    s_hat = score_from_mean(X, means_hat, t)
    s_ref = score_from_mean(X, means_ref, t)

    dm = means_hat - means_ref
    ds = s_hat - s_ref
    mean_rel = float(np.linalg.norm(dm) / np.linalg.norm(means_ref))
    score_rel = float(np.linalg.norm(ds) / np.linalg.norm(s_ref))
    resid = float(
        np.linalg.norm(ds - (alpha / delta) * dm) / (np.linalg.norm(ds) + 1e-300)
    )
    return {
        "mean_rel_l2": mean_rel,
        "score_rel_l2": score_rel,
        "mean_mse": float(np.mean(dm**2)),
        "identity_residual": resid,
    }
