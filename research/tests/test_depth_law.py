"""Validation of exp_17_depth_law.

Two things here are load-bearing and both fail silently if wrong:

1. **The hand-written backprop.** A subtly wrong gradient still trains, just to a worse
   optimum -- which in this experiment is indistinguishable from the headline finding
   ("depth does not buy what the law predicts"). Checked against central differences.

2. **The information horizon.** The entire law rests on a depth-``d`` network being unable to
   see beyond radius ``d``. If any layer or the padding leaks information further, the
   network can beat its own horizon and the experiment measures nothing. Checked directly by
   perturbing a distant site and asserting the output does not move.

Run:  python -m pytest tests/test_depth_law.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from exp_17_depth_law import (  # noqa: E402
    ChainLocalNet,
    fit_log_slope,
    predicted_q,
    radius_r_error,
)
from src.spectral import chain_covariance  # noqa: E402


# ---------------------------------------------------------------------------
# The information horizon -- the assumption the whole law rests on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("depth", [1, 2, 3, 5])
def test_depth_d_network_cannot_see_beyond_radius_d(depth):
    """Perturbing site j must not move the output at site i when |i - j| > depth.

    This is the architectural claim stated as an executable assertion. A dense layer, a
    global pooling, or padding that wraps would all break it, and all three are easy to
    introduce by accident.
    """
    n, width = 32, 8
    net = ChainLocalNet(depth, width, n_time=4, seed=0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, n))
    tf = rng.normal(size=(1, 4))

    base, _ = net.forward(x, tf)

    centre = n // 2
    far = centre + depth + 1          # strictly outside the horizon
    x2 = x.copy()
    x2[0, far] += 10.0                # a large perturbation, to leave no doubt
    pert, _ = net.forward(x2, tf)

    assert abs(pert[0, centre] - base[0, centre]) < 1e-12, (
        f"depth-{depth} net moved at site {centre} when site {far} changed: "
        "the information horizon is larger than the depth"
    )


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_the_network_does_use_its_full_horizon(depth):
    """The complementary check: a site exactly at radius ``depth`` MUST matter.

    Without this, a network that ignored its inputs entirely would pass the test above.
    """
    n, width = 32, 8
    net = ChainLocalNet(depth, width, n_time=4, seed=1)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, n))
    tf = rng.normal(size=(1, 4))
    base, _ = net.forward(x, tf)

    centre = n // 2
    x2 = x.copy()
    x2[0, centre + depth] += 1.0
    pert, _ = net.forward(x2, tf)
    assert abs(pert[0, centre] - base[0, centre]) > 1e-9


# ---------------------------------------------------------------------------
# The exact reference
# ---------------------------------------------------------------------------

def test_radius_zero_is_worse_than_radius_large():
    sigma0 = chain_covariance(32, 0.85)
    e0 = radius_r_error(sigma0, 0, 0.4)
    e8 = radius_r_error(sigma0, 8, 0.4)
    assert e0 > e8 > 0.0


def test_radius_r_error_decays_monotonically():
    """More context cannot hurt an *optimal* window estimator."""
    sigma0 = chain_covariance(32, 0.85)
    for t in (0.1, 0.4, 1.6):
        errs = [radius_r_error(sigma0, r, t) for r in range(0, 10)]
        d = np.diff(errs)
        assert np.all(d <= 1e-12), f"non-monotone at t={t}: {errs}"


def test_reference_decay_matches_the_closed_form_q():
    """The fitted slope of log(error) vs radius must match log(q) from ledger G12.

    This is the test that licenses calling the reference 'exact'. If it fails, either the
    window estimator or the closed form is wrong, and the depth comparison downstream has no
    yardstick.
    """
    sigma0 = chain_covariance(48, 0.85)
    for t in (0.1, 0.2, 0.4, 0.8):
        radii = np.arange(1, 12)
        errs = np.array([radius_r_error(sigma0, int(r), t) for r in radii])
        slope, se = fit_log_slope(radii.astype(float), errs)
        target = np.log(predicted_q(0.85, t))
        # Generous but meaningful: the closed form is a bulk/asymptotic statement and the
        # chain is finite, so agreement to ~10% of the slope is the right expectation.
        assert abs(slope - target) < 0.10 * abs(target) + 0.05, (
            f"t={t}: fitted {slope:.4f} vs predicted {target:.4f} (se {se:.4f})"
        )


def test_predicted_q_is_between_zero_and_one():
    for t in (0.05, 0.1, 0.4, 1.6, 3.0):
        q = predicted_q(0.85, t)
        assert 0.0 < q < 1.0, f"q={q} at t={t} is not a contraction factor"


def test_q_increases_with_noise():
    """More noise means slower spatial decay, i.e. a longer effective correlation."""
    qs = [predicted_q(0.85, t) for t in (0.1, 0.4, 1.6)]
    assert qs[0] < qs[1] < qs[2]


# ---------------------------------------------------------------------------
# Backprop
# ---------------------------------------------------------------------------

def _loss_and_grads(net, x, tf, target, depth):
    """Interior-masked squared loss, and the analytic gradients from the same code path."""
    pred, cache = net.forward(x, tf)
    n = x.shape[1]
    m = np.zeros_like(pred)
    m[:, depth:n - depth] = 1.0
    denom = max(float(m.sum()), 1.0)
    diff = pred - target
    loss = float(np.sum((diff ** 2) * m) / denom)
    g_out = 2.0 * diff * m / denom

    h_last, a_last = cache[-1]
    gW_out = a_last.reshape(-1, net.width).T @ g_out.reshape(-1, 1)
    delta_h = g_out[:, :, None] @ net.W_out.T
    gW = [None] * net.depth
    for d in range(net.depth - 1, -1, -1):
        h_in, a_d = cache[d]
        dz = delta_h * (1.0 - a_d ** 2)
        gW[d] = h_in.reshape(-1, h_in.shape[2]).T @ dz.reshape(-1, net.width)
        if d > 0:
            back = dz @ net.W[d].T
            w = net.width
            acc = np.zeros((back.shape[0], back.shape[1], w))
            acc += back[:, :, w:2 * w]
            acc[:, :-1] += back[:, 1:, :w]
            acc[:, 1:] += back[:, :-1, 2 * w:]
            delta_h = acc
    return loss, gW, gW_out


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_backprop_matches_finite_differences(depth):
    """Every layer's weight gradient against central differences.

    Done per depth because the neighbourhood scatter in the backward pass is the part most
    likely to be wrong, and it only engages for depth > 1.
    """
    n, width = 16, 6
    net = ChainLocalNet(depth, width, n_time=4, seed=3)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5, n))
    tf = rng.normal(size=(5, 4))
    target = rng.normal(size=(5, n))

    _, gW, gW_out = _loss_and_grads(net, x, tf, target, depth)

    h = 1e-6
    worst = 0.0
    for d in range(depth):
        for _ in range(8):
            i = rng.integers(net.W[d].shape[0])
            j = rng.integers(net.W[d].shape[1])
            orig = net.W[d][i, j]
            net.W[d][i, j] = orig + h
            lp, _, _ = _loss_and_grads(net, x, tf, target, depth)
            net.W[d][i, j] = orig - h
            lm, _, _ = _loss_and_grads(net, x, tf, target, depth)
            net.W[d][i, j] = orig
            num = (lp - lm) / (2 * h)
            den = max(abs(num), abs(gW[d][i, j]), 1e-8)
            worst = max(worst, abs(num - gW[d][i, j]) / den)

    assert worst < 1e-5, f"worst relative gradient error {worst:.2e}"


def test_output_layer_gradient_matches_finite_differences():
    n, width, depth = 16, 6, 2
    net = ChainLocalNet(depth, width, n_time=4, seed=4)
    rng = np.random.default_rng(4)
    x = rng.normal(size=(5, n))
    tf = rng.normal(size=(5, 4))
    target = rng.normal(size=(5, n))
    _, _, gW_out = _loss_and_grads(net, x, tf, target, depth)

    h = 1e-6
    worst = 0.0
    for i in range(width):
        orig = net.W_out[i, 0]
        net.W_out[i, 0] = orig + h
        lp, _, _ = _loss_and_grads(net, x, tf, target, depth)
        net.W_out[i, 0] = orig - h
        lm, _, _ = _loss_and_grads(net, x, tf, target, depth)
        net.W_out[i, 0] = orig
        num = (lp - lm) / (2 * h)
        den = max(abs(num), abs(gW_out[i, 0]), 1e-8)
        worst = max(worst, abs(num - gW_out[i, 0]) / den)
    assert worst < 1e-5


def test_parameter_count_grows_with_depth_and_width():
    a = ChainLocalNet(2, 16, 4, 0).n_params
    b = ChainLocalNet(4, 16, 4, 0).n_params
    c = ChainLocalNet(2, 32, 4, 0).n_params
    assert b > a and c > a
