"""Reverse-time samplers for the OU forward process.

Convention (matches src/noising.py)
-----------------------------------
Forward SDE:   dX_t = -X_t dt + sqrt(2) dW_t,  t in [0, T],  X_0 ~ p_0.
Marginals:     X_t ~ Law(alpha_t a + sqrt(Delta_t) z).

Reverse-time SDE (Anderson 1982), integrated from t = T down to t = t_min:

    dX = [ -X - 2 s(X, t) ] dt + sqrt(2) dW_bar        (dt < 0),

implemented in Euler-Maruyama form with step h > 0:

    x <- x + h [ x + 2 s(x, t) ] + sqrt(2 h) xi,   xi ~ N(0, I).

Probability-flow ODE (same marginals, deterministic):

    dX/dt = -X - s(X, t)    =>    x <- x + h [ x + s(x, t) ]   (backwards),

with an optional Heun (trapezoidal) corrector.

Because unit-variance data keep Var(X_t) = 1, we initialize at t = T with
X_T ~ N(0, I) (the T -> infinity limit; at T = 3 the bias is alpha^2 ~ 2.5e-3),
or from a forward-noised known sample for reconstruction experiments.

The final output can be read out either as x(t_min) or through the posterior
mean ("denoising readout")  m = (x + Delta_t s(x, t)) / alpha_t, which is exact
for BP-based scores since they are built from posterior means.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

ScoreFn = Callable[[np.ndarray, float], np.ndarray]
Callback = Callable[[float, np.ndarray, np.ndarray], None]


def time_grid(t_max: float, t_min: float, n_steps: int) -> np.ndarray:
    """Geometric time grid from t_max down to t_min (denser at small t,
    where the score stiffens as alpha/Delta ~ 1/(2t))."""
    return np.geomspace(t_max, t_min, n_steps + 1)


def reverse_sde(
    x_init: np.ndarray,
    score_fn: ScoreFn,
    times: np.ndarray,
    rng: np.random.Generator,
    callback: Callback | None = None,
) -> np.ndarray:
    """Euler-Maruyama reverse SDE from times[0] down to times[-1].

    All stochasticity comes from `rng`; passing generators with identical state
    to different score functions yields common-noise paired trajectories.
    """
    x = x_init.copy()
    for t_now, t_next in zip(times[:-1], times[1:]):
        h = float(t_now - t_next)
        s = score_fn(x, float(t_now))
        if callback is not None:
            callback(float(t_now), x, s)
        x = x + h * (x + 2.0 * s) + np.sqrt(2.0 * h) * rng.standard_normal(x.shape)
    if callback is not None:
        s = score_fn(x, float(times[-1]))
        callback(float(times[-1]), x, s)
    return x


def nested_brownian_path(
    times: np.ndarray, shape: tuple[int, ...], rng: np.random.Generator, refine: int
) -> np.ndarray:
    """Brownian increments on a fine grid that *sum* to the coarse grid's increments.

    Returns increments of shape ``(refine * (len(times) - 1),) + shape``, one per fine
    sub-step of the geometric grid obtained by splitting each coarse interval into `refine`
    equal parts in ``t``.

    Why this exists. `run_steps` currently integrates at 100, 200, 400 and 800 steps with
    *independent* noise at each resolution and compares the resulting marginal statistics.
    Those numbers came back non-monotone and all within one standard error of the target,
    which supports step-size **robustness** over the tested range -- but it is not a
    convergence study, and the write-up called it one. Independent noise means the difference
    between two resolutions is dominated by Monte Carlo scatter, not by discretisation.

    Nesting removes that. If the coarse increment over an interval is exactly the sum of the
    fine increments inside it, the two trajectories are driven by the *same* Brownian path
    and their difference is discretisation error alone, so a strong (pathwise) error can be
    measured and its order estimated. That is what distinguishes "the answer stopped moving"
    from "the noise is bigger than the movement".

    Use `coarsen_increments` to obtain the matching coarse increments from the same path.
    """
    if refine < 1:
        raise ValueError(f"refine must be >= 1, got {refine}.")
    steps = []
    for t_now, t_next in zip(times[:-1], times[1:]):
        h_fine = float(t_now - t_next) / refine
        steps.extend([h_fine] * refine)
    h = np.asarray(steps, dtype=float)
    z = rng.standard_normal((len(h),) + shape)
    return np.sqrt(h).reshape((-1,) + (1,) * len(shape)) * z


def coarsen_increments(dW: np.ndarray, refine: int) -> np.ndarray:
    """Sum `refine` consecutive fine increments into one coarse increment.

    The defining property of the nesting: ``coarsen_increments(dW, r)`` is exactly the
    Brownian increment the coarse integrator would have used had it been driven by the same
    path. Summing is correct precisely because Brownian increments are additive over
    adjacent intervals -- no rescaling is involved, and any rescaling here would break the
    pairing it exists to create.
    """
    n_fine = dW.shape[0]
    if n_fine % refine:
        raise ValueError(f"{n_fine} fine increments is not divisible by refine={refine}.")
    return dW.reshape((n_fine // refine, refine) + dW.shape[1:]).sum(axis=1)


def reverse_sde_with_increments(
    x_init: np.ndarray, score_fn: ScoreFn, times: np.ndarray, dW: np.ndarray
) -> np.ndarray:
    """Euler-Maruyama driven by *supplied* Brownian increments.

    Same recursion as `reverse_sde`, with the noise passed in rather than drawn, so that two
    resolutions can be run against one Brownian path. ``dW[k]`` must already carry the
    ``sqrt(h_k)`` scaling, which is what `nested_brownian_path` returns.
    """
    if len(dW) != len(times) - 1:
        raise ValueError(f"need {len(times) - 1} increments, got {len(dW)}.")
    x = x_init.copy()
    for k, (t_now, t_next) in enumerate(zip(times[:-1], times[1:])):
        h = float(t_now - t_next)
        s = score_fn(x, float(t_now))
        x = x + h * (x + 2.0 * s) + np.sqrt(2.0) * dW[k]
    return x


def probability_flow_ode(
    x_init: np.ndarray,
    score_fn: ScoreFn,
    times: np.ndarray,
    callback: Callback | None = None,
    heun: bool = True,
) -> np.ndarray:
    """Deterministic probability-flow ODE from times[0] down to times[-1]."""
    x = x_init.copy()
    for t_now, t_next in zip(times[:-1], times[1:]):
        h = float(t_now - t_next)
        s = score_fn(x, float(t_now))
        if callback is not None:
            callback(float(t_now), x, s)
        drift = x + s
        x_euler = x + h * drift
        if heun:
            s2 = score_fn(x_euler, float(t_next))
            drift2 = x_euler + s2
            x = x + 0.5 * h * (drift + drift2)
        else:
            x = x_euler
    if callback is not None:
        s = score_fn(x, float(times[-1]))
        callback(float(times[-1]), x, s)
    return x


def denoising_readout(x: np.ndarray, s: np.ndarray, t: float) -> np.ndarray:
    """Posterior-mean readout  E[a | x] = (x + Delta_t s) / alpha_t."""
    alpha = float(np.exp(-t))
    delta = float(1.0 - np.exp(-2.0 * t))
    return (x + delta * s) / alpha
