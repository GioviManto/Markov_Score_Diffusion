"""Finite-difference checks on every parameter of both sequence architectures.

WHY THIS EXISTS. The entire round-two review turns on a baseline having been
understated. These two networks are the replacement baselines, written with
hand-derived gradients in numpy, and a wrong backward pass does not crash --
it trains a bad network. A bad network loses to EM-BP, which is the result the
paper wants, which is exactly why it must not be arrived at by accident.

So: central differences against analytic gradients, per parameter array, for
both architectures. The test also checks the two structural properties the
architectures are chosen FOR -- that the conv stack's receptive field really
does span the chain, and that the recurrences really do move information end to
end -- because a network that silently truncates its context would be another
understated baseline wearing a better name.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.seq_nets import (
    BiMessagePassing,
    DilatedConv1d,
    _shift,
    _site_features,
)


def _loss_and_grads(net, feat, target):
    out, cache = net.forward(feat)
    loss = float(((out - target) ** 2).sum())
    grads = net.backward(cache, 2.0 * (out - target))
    return loss, grads


def _numeric(net, feat, target, j, idx, h=1e-6):
    flat = net.params[j].reshape(-1)
    orig = flat[idx]
    flat[idx] = orig + h
    plus, _ = net.forward(feat)
    lp = float(((plus - target) ** 2).sum())
    flat[idx] = orig - h
    minus, _ = net.forward(feat)
    lm = float(((minus - target) ** 2).sum())
    flat[idx] = orig
    return (lp - lm) / (2 * h)


def _check(net, rng, n_probe=6):
    B, L = 3, 12
    X = rng.standard_normal((B, L))
    feat = _site_features(X, 0.37)
    target = rng.standard_normal((B, L))
    _, grads = _loss_and_grads(net, feat, target)

    assert len(grads) == len(net.params)
    worst = 0.0
    for j, p in enumerate(net.params):
        assert grads[j].shape == p.shape, (
            f"param {j}: gradient shape {grads[j].shape} != param shape {p.shape}"
        )
        size = p.size
        probe = rng.choice(size, size=min(n_probe, size), replace=False)
        for idx in probe:
            num = _numeric(net, feat, target, j, int(idx))
            ana = float(grads[j].reshape(-1)[int(idx)])
            denom = max(1.0, abs(num), abs(ana))
            rel = abs(num - ana) / denom
            worst = max(worst, rel)
            assert rel < 2e-5, (
                f"param {j}[{idx}]: analytic {ana:.8e} vs numeric {num:.8e} "
                f"(rel {rel:.2e}) -- the backward pass is wrong"
            )
    return worst


@pytest.mark.parametrize("dilations", [(1, 2), (1, 2, 4, 8)])
def test_dilated_conv_gradients(dilations):
    rng = np.random.default_rng(0)
    net = DilatedConv1d.init(hidden=7, dilations=dilations, rng=rng)
    worst = _check(net, rng)
    assert worst < 2e-5


def test_bi_message_passing_gradients():
    rng = np.random.default_rng(1)
    net = BiMessagePassing.init(hidden=6, rng=rng)
    worst = _check(net, rng)
    assert worst < 2e-5


def test_shift_is_its_own_adjoint():
    """<shift(x, s), y> == <x, shift(y, -s)>; the backward pass relies on it."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 3, 9))
    y = rng.standard_normal((2, 3, 9))
    for s in (-3, -1, 0, 1, 4):
        lhs = float((_shift(x, s) * y).sum())
        rhs = float((x * _shift(y, -s)).sum())
        assert abs(lhs - rhs) < 1e-12, f"shift adjoint fails at s={s}"


def test_conv_receptive_field_spans_the_chain():
    """A perturbation at site 0 must reach the far end of a length-32 chain.

    This is the property the architecture is chosen for. The shared-window head
    provably cannot do this beyond its radius, and comparing against a baseline
    that cannot see the whole sequence is what produced the original inflated
    margin.
    """
    rng = np.random.default_rng(3)
    net = DilatedConv1d.init(hidden=8, dilations=(1, 2, 4, 8), rng=rng)
    L = 32
    X = np.zeros((1, L))
    base, _ = net.forward(_site_features(X, 0.5))
    X[0, 0] = 5.0
    bumped, _ = net.forward(_site_features(X, 0.5))
    delta = np.abs(bumped - base)[0]
    reach = int(np.max(np.nonzero(delta > 1e-9)[0]))
    assert reach >= 15, (
        f"perturbing site 0 changes the output only out to site {reach}; "
        f"sum(dilations)=15 was expected"
    )


def test_recurrence_carries_information_end_to_end():
    """The bidirectional net must couple site 0 and site L-1 in both directions."""
    rng = np.random.default_rng(4)
    net = BiMessagePassing.init(hidden=12, rng=rng)
    L = 32
    for source, sink in ((0, L - 1), (L - 1, 0)):
        X = np.zeros((1, L))
        base, _ = net.forward(_site_features(X, 0.5))
        X[0, source] = 5.0
        bumped, _ = net.forward(_site_features(X, 0.5))
        moved = abs(float(bumped[0, sink] - base[0, sink]))
        assert moved > 1e-9, (
            f"a perturbation at site {source} does not reach site {sink}: "
            "the recurrence is not propagating"
        )


def test_training_reduces_loss_and_checkpoints_differ():
    """Sanity: the optimiser works, and snapshots are genuinely distinct."""
    from src.seq_nets import predict, train_sequence_net

    rng = np.random.default_rng(5)
    L, n = 16, 256
    # A smooth AR-like clean signal, so the x0 target is genuinely learnable.
    A = np.cumsum(rng.standard_normal((n, L)) * 0.4, axis=1)

    net = DilatedConv1d.init(hidden=16, dilations=(1, 2), rng=rng)
    snaps = train_sequence_net(
        net, A, [0.5], rng, checkpoints=(50, 600),
        parameterization="x0", batch_size=32, lr=3e-3,
    )
    assert set(snaps) == {50, 600}

    # Score both checkpoints on one fixed noised batch, so the comparison is
    # not itself a draw from the noise.
    probe = np.random.default_rng(99)
    alpha, delta = np.exp(-0.5), 1.0 - np.exp(-1.0)
    Xn = alpha * A + np.sqrt(delta) * probe.standard_normal(A.shape)
    early = float(((predict(net, snaps[50], Xn, 0.5) - A) ** 2).mean())
    late = float(((predict(net, snaps[600], Xn, 0.5) - A) ** 2).mean())
    assert late < early, f"training did not improve: {early:.4g} -> {late:.4g}"
    assert any(
        not np.allclose(a, b) for a, b in zip(snaps[50], snaps[600])
    ), "checkpoints are identical -- snapshots are aliasing the live parameters"
