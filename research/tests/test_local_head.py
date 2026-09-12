"""Tests for the local (convolutional) score head.

The head is the structured baseline that materially changed the headline
comparison, so it needs the same scrutiny as the estimator it is compared
against. The strongest test here is the r = 0 case, which has an exact
closed-form answer the head must converge to.
"""

from __future__ import annotations

import numpy as np

from src.local_head import (
    interior_slice,
    local_posterior_mean,
    make_windows,
    train_local_head,
)
from src.noising import alpha_delta
from src.priors import GaussianAR1
from src.utils import rng_for


def test_make_windows_shape_content_and_padding():
    """Windows must be the sliding neighbourhoods, zero-padded at the ends."""
    X = np.arange(12, dtype=float).reshape(2, 6)

    w0 = make_windows(X, 0)
    assert w0.shape == (2, 6, 1)
    assert np.array_equal(w0[..., 0], X)

    w2 = make_windows(X, 2)
    assert w2.shape == (2, 6, 5)
    # Interior site 2 of row 0 sees x_0..x_4 with no padding.
    assert np.array_equal(w2[0, 2], np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    # Site 0 sees two zeros then x_0, x_1, x_2.
    assert np.array_equal(w2[0, 0], np.array([0.0, 0.0, 0.0, 1.0, 2.0]))
    # Last site sees x_3, x_4, x_5 then two zeros.
    assert np.array_equal(w2[0, -1], np.array([3.0, 4.0, 5.0, 0.0, 0.0]))
    # The centre column is always the site itself.
    assert np.array_equal(w2[:, :, 2], X)


def test_interior_slice_excludes_padded_sites():
    assert interior_slice(33, 0) == slice(0, 33)
    assert interior_slice(33, 4) == slice(4, 29)
    # A radius wider than the chain must not produce an empty slice.
    sl = interior_slice(11, 40)
    assert sl.stop > sl.start


def test_radius_zero_head_learns_the_exact_single_site_denoiser():
    """r = 0 has a closed-form target, so this validates the whole pipeline.

    With unit-variance clean data alpha_t^2 + Delta_t = 1, so for a Gaussian
    chain the single-site posterior mean is exactly

        E[a_i | x_i] = alpha_t x_i / (alpha_t^2 + Delta_t) = alpha_t x_i,

    and an eps-head must therefore output z_hat = sqrt(Delta_t) x_i.

    The assertions are made in the *network's own output space*, not in
    posterior-mean space, and that distinction is the point. Writing the naive
    test -- "the recovered mean is within 6% of alpha x" -- fails at t = 0.8
    with a 17% error, which looks like a broken head and is not. Inverting an
    eps-prediction multiplies its error by sqrt(Delta_t)/alpha_t^2, which is
    0.86 at t = 0.2 and 4.42 at t = 0.8. Measured in its own space the head is
    equally accurate at both (~1.6%); it is the inversion that differs.

    So this test does two jobs: it checks the windowing, time features,
    parameterization inversion and training loop against a known answer, and it
    pins the amplification law that the write-up uses to explain why
    eps-prediction loses at high noise and x0-prediction loses at low noise.
    """
    prior = GaussianAR1(0.85)
    rng = rng_for("test-local-r0")
    A = np.stack([prior.sample(rng, 24) for _ in range(2000)])

    rng_test = rng_for("test-local-r0-test")
    A_test = np.stack([prior.sample(rng_test, 24) for _ in range(256)])
    t_values = (0.2, 0.8)

    eps_net_space = {}
    for mode in ("eps", "x0"):
        head = train_local_head(
            A, t_values, 0, rng_for("test-local-r0-train", mode),
            hidden=(32, 32), n_steps=20000, parameterization=mode,
        )
        for t in t_values:
            alpha, delta = alpha_delta(t)
            X = alpha * A_test + np.sqrt(delta) * rng_test.standard_normal(A_test.shape)
            m_hat = local_posterior_mean(head, X, t)
            rel_m = float(
                np.linalg.norm(m_hat - alpha * X) / np.linalg.norm(alpha * X)
            )
            if mode == "x0":
                # The head outputs the mean directly: no inversion, no amplification.
                assert rel_m < 0.08, f"x0 at t={t}: {rel_m:.4f}"
            else:
                amplification = np.sqrt(delta) / alpha**2
                eps_net_space[t] = rel_m / amplification
                assert eps_net_space[t] < 0.03, (
                    f"eps at t={t}: net-space error {eps_net_space[t]:.4f}"
                )

    # The head is equally accurate at both noise levels in its own space, so the
    # m-space blow-up is entirely the inversion factor and not a learning failure.
    ratio = eps_net_space[0.8] / eps_net_space[0.2]
    assert 0.4 < ratio < 2.5, f"net-space accuracy is t-dependent: {eps_net_space}"


def test_wider_window_beats_radius_zero():
    """A head with context must beat the single-site head on a correlated chain.

    Trivial to state and easy to get wrong: if the windows were misaligned, or
    the centre column were not the site being predicted, extra context would not
    help and this would fail while everything else still looked plausible.
    """
    prior = GaussianAR1(0.85)
    rng = rng_for("test-local-wider")
    A = np.stack([prior.sample(rng, 24) for _ in range(2000)])
    rng_test = rng_for("test-local-wider-test")
    A_test = np.stack([prior.sample(rng_test, 24) for _ in range(256)])

    t = 0.4
    alpha, delta = alpha_delta(t)
    X = alpha * A_test + np.sqrt(delta) * rng_test.standard_normal(A_test.shape)

    errs = {}
    for radius in (0, 3):
        head = train_local_head(
            A, (t,), radius, rng_for("test-local-wider-train", radius),
            hidden=(32, 32), n_steps=4000, parameterization="eps",
        )
        sl = interior_slice(24, 3)
        m_hat = local_posterior_mean(head, X, t)
        errs[radius] = float(
            np.linalg.norm(m_hat[:, sl] - alpha * X[:, sl])
        )

    # r=3 must differ substantially from the single-site answer; r=0 must not.
    assert errs[3] > 2.0 * errs[0], (
        f"context did not change the prediction: r0={errs[0]:.4f}, r3={errs[3]:.4f}"
    )
