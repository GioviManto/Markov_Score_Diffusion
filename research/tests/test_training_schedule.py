"""The training-time schedule, and why it has to be continuous.

An external review found the blocking experimental defect in this project: the neural
denoisers were trained on five discrete noise levels while the reverse integrator called
them at every point of [t_min, t_max]. The generated-sample comparison -- the paper's
headline result -- therefore mixed estimator error with interpolation between the training
levels and extrapolation outside them.

These tests pin the fix. The interesting ones are not that the sampler returns numbers in a
range, but that the discrete schedule genuinely fails to cover the interval the integrator
uses, and that the continuous one does.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.nnet import sample_training_times
from src.reverse import time_grid

T_TRAIN = (0.1, 0.2, 0.4, 0.8, 1.6)
T_MIN, T_MAX = 0.02, 3.0


def test_exactly_one_schedule_must_be_given():
    """Neither omission nor both, so a caller cannot keep the old behaviour silently."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_training_times(rng, 8)
    with pytest.raises(ValueError):
        sample_training_times(rng, 8, t_values=T_TRAIN, t_range=(T_MIN, T_MAX))
    with pytest.raises(ValueError):
        sample_training_times(rng, 8, t_range=(0.0, 1.0))
    with pytest.raises(ValueError):
        sample_training_times(rng, 8, t_range=(2.0, 1.0))


def test_discrete_schedule_leaves_most_of_the_integration_interval_unvisited():
    """The defect itself, measured rather than described.

    The integrator visits 400 geometrically spaced points of [0.02, 3.0]. Training on
    `T_TRAIN` supplies observations at five of them, so the network is fitted on a set of
    measure zero and queried on the interval -- and crucially, both ends of the interval lie
    outside the convex hull of the training levels, so those calls are extrapolation, not
    interpolation.
    """
    rng = np.random.default_rng(1)
    drawn = np.unique(sample_training_times(rng, 5000, t_values=T_TRAIN))
    assert len(drawn) == len(T_TRAIN)

    times = time_grid(T_MAX, T_MIN, 400)
    outside = float(np.mean((times < min(T_TRAIN)) | (times > max(T_TRAIN))))
    assert outside > 0.25, "expected a substantial fraction of integrator times to be extrapolation"


def test_continuous_schedule_covers_the_integration_interval():
    rng = np.random.default_rng(2)
    t = sample_training_times(rng, 20000, t_range=(T_MIN, T_MAX))

    assert t.min() >= T_MIN and t.max() <= T_MAX
    # Both former extrapolation regions now receive training mass.
    assert float(np.mean(t < min(T_TRAIN))) > 0.05
    assert float(np.mean(t > max(T_TRAIN))) > 0.05


def test_continuous_schedule_is_log_uniform_matching_the_integration_grid():
    """Log-uniform, not uniform, because `time_grid` is geometric.

    Sampling uniformly in t would put almost no training mass below t = 0.1, which is
    exactly where the reverse drift stiffens like 1/(2t) and where the integrator spends
    most of its steps. The test compares the sampler's quantiles against the integration
    grid's own: if the two schedules agree, training density follows the density of points
    that will actually be queried.
    """
    rng = np.random.default_rng(3)
    t = sample_training_times(rng, 200000, t_range=(T_MIN, T_MAX))

    # log t should be uniform: its quantiles are linear in the probability level.
    for p in (0.1, 0.25, 0.5, 0.75, 0.9):
        expected = np.log(T_MIN) + p * (np.log(T_MAX) - np.log(T_MIN))
        assert np.quantile(np.log(t), p) == pytest.approx(expected, abs=0.02)

    # And it matches the geometric integration grid, which is the point of choosing it.
    times = np.sort(time_grid(T_MAX, T_MIN, 400))
    for p in (0.25, 0.5, 0.75):
        assert np.quantile(t, p) == pytest.approx(np.quantile(times, p), rel=0.05)

    uniform_draw = rng.uniform(T_MIN, T_MAX, size=200000)
    assert float(np.mean(t < 0.1)) > 6.0 * float(np.mean(uniform_draw < 0.1))


def test_discrete_schedule_is_unchanged_from_the_original_expression():
    """The legacy path must still reproduce `t_arr[rng.integers(0, len(t_arr), size)]`
    exactly, so a run pinned to the old protocol is bit-comparable with committed outputs."""
    t_arr = np.asarray(T_TRAIN, dtype=float)
    a = sample_training_times(np.random.default_rng(7), 64, t_values=T_TRAIN)
    b = t_arr[np.random.default_rng(7).integers(0, len(t_arr), size=64)]
    np.testing.assert_array_equal(a, b)
