"""Nested Brownian increments: the difference between robustness and convergence.

`exp_16`'s step-size study integrates at 100/200/400/800 steps with *independent* noise and
compares marginal statistics. The results came back non-monotone and all within one standard
error of the target, which the write-up read as convergence. It is not: with independent
noise the gap between two resolutions is dominated by Monte Carlo scatter, so the study can
only show that the answer is insensitive over the tested range.

Driving both resolutions with one Brownian path makes the difference discretisation error
alone, which is what a convergence study measures. These tests pin the nesting property that
makes that valid, and then confirm the resulting strong error actually decreases with step
size -- on a linear SDE with an exactly known score, so nothing about the estimator enters.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.reverse import (
    coarsen_increments,
    nested_brownian_path,
    reverse_sde,
    reverse_sde_with_increments,
    time_grid,
)


def test_coarse_increments_are_sums_of_fine_ones():
    """The defining property. Without it the two resolutions are not paired at all."""
    times = time_grid(3.0, 0.02, 8)
    rng = np.random.default_rng(0)
    dW = nested_brownian_path(times, (5, 4), rng, refine=4)

    assert dW.shape == (32, 5, 4)
    coarse = coarsen_increments(dW, 4)
    assert coarse.shape == (8, 5, 4)
    np.testing.assert_allclose(coarse, dW.reshape(8, 4, 5, 4).sum(axis=1), rtol=0, atol=0)

    with pytest.raises(ValueError):
        coarsen_increments(dW, 5)


def test_increments_have_the_right_variance_per_step():
    """Each increment must carry sqrt(h) for its own sub-interval, not a global average --
    the grid is geometric, so the sub-steps are not equal and a single scale would be
    wrong everywhere except one interval."""
    times = time_grid(3.0, 0.02, 6)
    rng = np.random.default_rng(1)
    dW = nested_brownian_path(times, (200000,), rng, refine=2)

    k = 0
    for t_now, t_next in zip(times[:-1], times[1:]):
        h_fine = float(t_now - t_next) / 2
        for _ in range(2):
            assert float(dW[k].var()) == pytest.approx(h_fine, rel=0.02)
            k += 1


def test_supplied_increment_integrator_matches_the_drawing_one():
    """`reverse_sde_with_increments` must be the same recursion as `reverse_sde`.

    Driven by the increments `reverse_sde` would itself have drawn, the two must agree to
    floating point -- otherwise the convergence study would be measuring a second integrator.

    Not bit-exact, and the reason is worth recording rather than hiding behind a tolerance:
    `reverse_sde` forms ``sqrt(2h) * z`` in one product while this one forms
    ``sqrt(2) * (sqrt(h) * z)``, since the increment already carries its own ``sqrt(h)``.
    Those are the same number in exact arithmetic and differ in the last bit under
    reassociation -- measured 1.1e-16 absolute, 1.7e-14 relative. A tolerance at 1e-12 admits
    that and nothing larger; a genuine recursion difference would be orders of magnitude
    above it.
    """
    times = time_grid(3.0, 0.02, 12)
    x0 = np.random.default_rng(2).standard_normal((6, 4))

    def score(x, t):
        delta = 1.0 - np.exp(-2.0 * t)
        return -x / delta

    seed = 1234
    ref = reverse_sde(x0, score, times, np.random.default_rng(seed))

    rng = np.random.default_rng(seed)
    dW = np.stack([
        np.sqrt(float(a - b)) * rng.standard_normal(x0.shape)
        for a, b in zip(times[:-1], times[1:])
    ])
    got = reverse_sde_with_increments(x0, score, times, dW)

    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-14)


def test_strong_error_decreases_under_refinement():
    """The point of the whole construction.

    On the Ornstein-Uhlenbeck reverse dynamics with the exact Gaussian score, refine the step
    and measure the *pathwise* distance between the coarse and fine trajectories driven by
    one Brownian path. It must fall as the step shrinks. With independent noise this quantity
    would be roughly constant -- dominated by the scatter between two unrelated paths -- which
    is exactly the failure mode the current step study has.
    """
    def score(x, t):
        delta = 1.0 - np.exp(-2.0 * t)
        return -x / delta

    x0 = np.random.default_rng(3).standard_normal((64, 8))
    errors = []
    for n_coarse in (25, 50, 100):
        times = time_grid(3.0, 0.05, n_coarse)
        rng = np.random.default_rng(7)
        dW_fine = nested_brownian_path(times, x0.shape, rng, refine=4)

        fine_times = np.concatenate([
            np.linspace(a, b, 5)[:-1] for a, b in zip(times[:-1], times[1:])
        ] + [[times[-1]]])

        x_coarse = reverse_sde_with_increments(
            x0, score, times, coarsen_increments(dW_fine, 4)
        )
        x_fine = reverse_sde_with_increments(x0, score, fine_times, dW_fine)
        errors.append(float(np.sqrt(np.mean((x_coarse - x_fine) ** 2))))

    assert errors[0] > errors[1] > errors[2], f"strong error not decreasing: {errors}"
    assert errors[2] < errors[0] / 2.0
